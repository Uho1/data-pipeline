"""Tab-level ticker analysis service.

Phase2 goals:
- Provide tab-scoped API payloads that contain all keys requested by frontend GraphRegistry
- Fill 가능한 데이터는 DuckDB(prices, financials_quarterly) + PIT(as-of, AvailableDate) 규칙으로 계산
- 불가능한 키는 missing reason 명시
"""
from __future__ import annotations

import copy
from datetime import datetime
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_data.db_router import is_kr_ticker
from market_data.fiscal_periods import (
    attach_fiscal_metadata as _shared_attach_fiscal_metadata,
    infer_fiscal_period_meta as _shared_infer_fiscal_period_meta,
    is_annual_form as _shared_is_annual_form,
    period_year_series as _shared_period_year_series,
)
from market_data.valuation_ttm import (
    VALUATION_TTM_QUANTILES,
    VALUATION_TTM_REFERENCE_LEVELS,
    build_valuation_frame_from_precomputed,
)

# JSON-based data loading (sole data source)
from web.backend.services import json_data_service as _json_svc

# DuckDB is no longer used for serving.
_HAS_DUCKDB = False

from web.backend.schemas.ticker_analysis import (
    SeriesPoint,
    SnapshotInfo,
    TabMeta,
    TickerTabResponse,
)

_GRAPH_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "frontend-next" / "src" / "components" / "charts" / "GraphRegistry.ts"
_REPO_ROOT = Path(__file__).resolve().parents[3]
def _get_cached_kr_fin_df(ticker: str) -> pd.DataFrame | None:
    """Load KR financials — JSON first, DuckDB fallback."""
    import time
    key = str(ticker).strip().upper()
    cached = _KR_FIN_DF_CACHE.get(key)
    now = time.monotonic()
    if cached is not None:
        ts, df = cached
        if now - ts < _KR_FIN_DF_CACHE_TTL:
            return df.copy() if df is not None else None

    # Try JSON first
    df = None
    try:
        df = _json_svc.get_financials_dataframe(key, market="kr")
    except Exception:
        pass

    # Fallback to DuckDB
    if df is None and _HAS_DUCKDB:
        try:
            df = db_reader_kr.load_financials_from_db(key, market="kr")
        except Exception:
            df = None

    _KR_FIN_DF_CACHE[key] = (now, df)
    if len(_KR_FIN_DF_CACHE) > 200:
        cutoff = now - _KR_FIN_DF_CACHE_TTL
        stale = [k for k, (t, _) in _KR_FIN_DF_CACHE.items() if t < cutoff]
        for k in stale:
            _KR_FIN_DF_CACHE.pop(k, None)
    return df.copy() if df is not None else None
_COMPANY_NAME_STOPWORDS = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "ltd",
    "limited",
    "llc",
    "plc",
    "group",
    "holdings",
    "holding",
    "sa",
    "ag",
    "nv",
    "spa",
    "se",
    "the",
}


def _to_iso_date(ts: pd.Timestamp | str | None) -> str | None:
    if ts is None:
        return None
    dt = pd.to_datetime(ts, errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt).date().isoformat()


def _to_naive_normalized_ts(value: pd.Timestamp | str | None) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    out = pd.Timestamp(ts)
    if out.tzinfo is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    return out.normalize()


def _to_naive_series(values: pd.Series | Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype="datetime64[ns]")
    s = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(s, pd.Series):
        if s.dt.tz is not None:
            return s.dt.tz_convert(None)
        return s
    # scalar/ndarray path fallback
    if isinstance(s, pd.DatetimeIndex):
        if s.tz is not None:
            s = s.tz_convert(None)
        return pd.Series(s)
    result = pd.Series(pd.to_datetime(values, errors="coerce", utc=True))
    if hasattr(result, "dt") and result.dt.tz is not None:
        return result.dt.tz_convert(None)
    return result


def _parse_asof(asof: str | None) -> pd.Timestamp:
    if asof:
        ts = _to_naive_normalized_ts(asof)
        if pd.notna(ts):
            return ts
    return _to_naive_normalized_ts(pd.Timestamp.now(tz="UTC"))


_ANNUAL_FORM_MARKERS = ("10-K", "20-F", "40-F")


def _is_annual_form(form_type: Any) -> bool:
    return _shared_is_annual_form(form_type)


def _infer_fiscal_period_meta(
    period_end: pd.Series | pd.Index | list[Any],
    form_type: pd.Series | list[Any] | None = None,
    period_start: pd.Series | pd.Index | list[Any] | None = None,
) -> pd.DataFrame:
    return _shared_infer_fiscal_period_meta(period_end, form_type, period_start)


def _attach_fiscal_metadata(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    return _shared_attach_fiscal_metadata(frame, source, period_column="PeriodEnd")


def _period_year_series(frame: pd.DataFrame) -> pd.Series:
    return _shared_period_year_series(frame)


def _safe_num(v: Any) -> float | None:
    try:
        f = float(v)
        if not np.isfinite(f):
            return None
        return f
    except Exception:
        return None


def _fill_revenue_from_components(
    revenue: pd.Series,
    cogs: pd.Series,
    gross_profit: pd.Series,
) -> pd.Series:
    rev = pd.to_numeric(revenue, errors="coerce")
    cogs_num = pd.to_numeric(cogs, errors="coerce")
    gp_num = pd.to_numeric(gross_profit, errors="coerce")
    rebuilt = cogs_num + gp_num
    return rev.where(rev.notna(), rebuilt)


def _quarterize_kr_december_annual_flow_series(
    series: pd.Series,
    *,
    allow_negative: bool,
) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").copy()
    if not isinstance(out, pd.Series) or out.empty:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(out.index)
    for year in sorted(set(idx.year)):
        if pd.isna(year):
            continue
        dec_mask = (idx.year == year) & (idx.month == 12)
        if not dec_mask.any():
            continue
        prev_mask = (idx.year == year) & (idx.month < 12)
        if not prev_mask.any():
            continue
        dec_pos = np.flatnonzero(dec_mask)[-1]
        dec_val = out.iloc[dec_pos]
        prev_val = _safe_num(pd.to_numeric(out.loc[prev_mask], errors="coerce").sum(min_count=1))
        if pd.isna(dec_val) or pd.isna(prev_val):
            continue
        if abs(dec_val) <= abs(prev_val) * 1.2:
            continue
        diff = dec_val - prev_val
        if not allow_negative and diff < 0:
            continue
        out.iloc[dec_pos] = diff
    return out


def _pick_kr_fy_baseline(
    current_revenue: Any,
    prev_raw_revenue: Any,
    prev_converted_revenue: Any,
) -> str | None:
    cur = _safe_num(current_revenue)
    if cur is None:
        return None
    candidates: list[tuple[float, str]] = []
    raw_prev = _safe_num(prev_raw_revenue)
    if raw_prev is not None:
        raw_diff = cur - raw_prev
        if np.isfinite(raw_diff) and raw_diff >= 0:
            candidates.append((raw_diff, "raw"))
    conv_prev = _safe_num(prev_converted_revenue)
    if conv_prev is not None:
        conv_diff = cur - conv_prev
        if np.isfinite(conv_diff) and conv_diff >= 0:
            candidates.append((conv_diff, "converted"))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _to_numeric_series(series: pd.Series) -> pd.Series:
    s = pd.Series(series, index=series.index)
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().any():
        return out
    # Fallback for formatted numeric strings: "1,234", "(123)", "$123.4"
    as_str = s.astype("string").str.strip()
    as_str = as_str.mask(as_str.isin(["", "nan", "None", "null"]), pd.NA)
    as_str = as_str.str.replace(",", "", regex=False)
    as_str = as_str.str.replace("$", "", regex=False)
    as_str = as_str.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(as_str, errors="coerce")


def _series_points(index: pd.Index, values: pd.Series) -> list[SeriesPoint]:
    points: list[SeriesPoint] = []
    for k, v in zip(index, values, strict=False):
        date_str = _to_iso_date(k) if isinstance(k, (pd.Timestamp, datetime, str)) else str(k)
        points.append(SeriesPoint(date=date_str or str(k), value=_safe_num(v)))
    return points


def _load_graph_keys(tab: str, subtab: str | None = None) -> set[str]:
    """Parse frontend GraphRegistry.ts and collect yKeys for the requested tab."""
    try:
        text = _GRAPH_REGISTRY_PATH.read_text(encoding="utf-8")
    except Exception:
        return set()

    keys: set[str] = set()
    blocks = text.split("g({")
    for block in blocks[1:]:
        head = block.split("}),", 1)[0]
        m_tab = re.search(r'tab:\s*"([^"]+)"', head)
        if not m_tab or m_tab.group(1) != tab:
            continue
        if subtab is not None:
            m_sub = re.search(r'subtab:\s*"([^"]+)"', head)
            if m_sub is not None:
                if m_sub.group(1) != subtab:
                    continue
            else:
                # Graphs declared in helper functions may use `subtab,` (variable)
                # instead of a literal string. Treat them as applicable to all
                # business subtabs and rely on backend payload shaping.
                if "subtab" in head and "subtab," not in head and "subtab :" not in head:
                    continue
        m_y = re.search(r"yKeys:\s*\[([^\]]*)\]", head, flags=re.S)
        if not m_y:
            continue
        keys.update(re.findall(r'"([^"]+)"', m_y.group(1)))
    return keys


def _resolve_market_and_price(ticker: str, market: str) -> tuple[str, pd.DataFrame]:
    resolved = market if market != "auto" else ("kr" if is_kr_ticker(ticker) else "us")

    # JSON-first: build DataFrame from JSON price data
    try:
        data = _json_svc.load_ticker_data(ticker, resolved)
        if data and "prices" in data:
            prices = data["prices"]
            dates = prices.get("dates", [])
            if dates:
                df_dict: dict[str, list] = {}
                for col in ("open", "high", "low", "close", "volume", "market_cap", "shares_outstanding"):
                    if col in prices:
                        # Map to capitalized column names for compatibility
                        mapped = {"open": "Open", "high": "High", "low": "Low", "close": "Close",
                                  "volume": "Volume", "market_cap": "MarketCap",
                                  "shares_outstanding": "SharesOutstanding"}.get(col, col)
                        df_dict[mapped] = prices[col]
                px = pd.DataFrame(df_dict, index=pd.to_datetime(dates))
                px.index.name = "Date"
                if "Close" in px.columns:
                    px["Adj Close"] = px["Close"]
                return resolved, px
    except Exception:
        pass

    # Fallback to DuckDB
    if _HAS_DUCKDB:
        try:
            px, src = load_price_dataframe(ticker=ticker, market=market)
            resolved = src.parent.name if src is not None else resolved
            if not isinstance(px.index, pd.DatetimeIndex):
                px.index = pd.to_datetime(px.index, errors="coerce")
            px = px.loc[~px.index.isna()].sort_index()
            return resolved, px
        except Exception:
            pass

    return resolved, pd.DataFrame()


class _AnalysisPreloadedData:
    def __init__(
        self,
        *,
        ticker: str,
        requested_market: str,
        resolved_market: str,
        asof_ts: pd.Timestamp,
        px: pd.DataFrame,
        q_pit: pd.DataFrame,
    ) -> None:
        self.ticker = ticker
        self.requested_market = requested_market
        self.resolved_market = resolved_market
        self.asof_ts = asof_ts
        self.px = px
        self.q_pit = q_pit
        self.derived_frames: dict[str, pd.DataFrame] = {}
        self.quarter_frames: dict[str, pd.DataFrame] = {}
        self.financial_feature_frame: pd.DataFrame | None = None


def _build_preloaded_analysis_data(
    ticker: str,
    market: str = "auto",
    asof: str | None = None,
) -> _AnalysisPreloadedData:
    tkr = str(ticker).strip().upper()
    requested_market = str(market or "auto").strip().lower() or "auto"
    asof_ts = _parse_asof(asof)
    resolved_market, px = _resolve_market_and_price(tkr, requested_market)
    q_pit = _build_pit_quarterly_frame(tkr, resolved_market, asof_ts)
    return _AnalysisPreloadedData(
        ticker=tkr,
        requested_market=requested_market,
        resolved_market=resolved_market,
        asof_ts=asof_ts,
        px=px,
        q_pit=q_pit,
    )


def _preloaded_derived_frame(
    preloaded: _AnalysisPreloadedData,
    basis: str | None = None,
) -> pd.DataFrame:
    basis_key = str(basis or "__none__").strip().lower() or "__none__"
    if basis_key not in preloaded.derived_frames:
        load_basis = None if basis_key == "__none__" else basis_key
        preloaded.derived_frames[basis_key] = _load_pit_derived_frame(
            preloaded.ticker,
            preloaded.resolved_market,
            preloaded.asof_ts,
            basis=load_basis,
        )
    return preloaded.derived_frames[basis_key].copy()


def _preloaded_base_quarter_frame(
    preloaded: _AnalysisPreloadedData,
    frame_type: str,
) -> pd.DataFrame:
    if frame_type not in preloaded.quarter_frames:
        if frame_type == "income":
            frame = _build_income_base_quarter_frame(preloaded.q_pit, preloaded.px, preloaded.ticker, preloaded.resolved_market)
        elif frame_type == "balance":
            frame = _build_balance_base_quarter_frame(preloaded.q_pit, preloaded.px, preloaded.ticker, preloaded.resolved_market)
        elif frame_type == "cashflow":
            frame = _build_cashflow_base_quarter_frame(preloaded.q_pit, preloaded.px, preloaded.ticker, preloaded.resolved_market)
        elif frame_type == "fundamentals":
            frame = _build_fundamentals_base_quarter_frame(preloaded.q_pit, preloaded.px, preloaded.ticker, preloaded.resolved_market)
        else:
            raise ValueError(f"unsupported preloaded frame_type={frame_type}")
        preloaded.quarter_frames[frame_type] = frame
    return preloaded.quarter_frames[frame_type].copy()


def _preloaded_financial_feature_frame(preloaded: _AnalysisPreloadedData) -> pd.DataFrame:
    if preloaded.financial_feature_frame is None:
        feat = _compute_financial_feature_frame(preloaded.q_pit, preloaded.px, preloaded.ticker, preloaded.resolved_market)
        derived_quarter = _preloaded_derived_frame(preloaded, basis="quarter")
        preloaded.financial_feature_frame = _overlay_derived_columns(feat, derived_quarter)
    return preloaded.financial_feature_frame.copy()


def _build_pit_quarterly_frame(ticker: str, market: str, asof: pd.Timestamp) -> pd.DataFrame:
    # JSON-first: load financials from JSON file
    frame = None
    try:
        frame = _json_svc.get_financials_dataframe(ticker, market)
    except Exception:
        pass

    # Fallback to DuckDB PIT loader
    if (frame is None or frame.empty) and _HAS_DUCKDB:
        frame = load_financials_quarterly(
            market=market,
            tickers=[ticker],
            statement_type="merged",
            use_next_trading_day_availability=False,
            availability_fallback=True,
        )

    if frame is None or frame.empty:
        return pd.DataFrame()

    f = frame.copy()
    for col in ("PeriodEnd", "AvailableDate", "FilingDate", "AcceptedAt"):
        if col in f.columns:
            f[col] = _to_naive_series(f[col])
    asof_naive = _to_naive_normalized_ts(asof)

    # PIT filter: only apply if AvailableDate column exists (not in JSON-sourced data)
    if "AvailableDate" in f.columns:
        avail = _to_naive_series(f["AvailableDate"])
        # 공시일(AvailableDate) 누락(NaT) 시 분기말+45일(분기보고서 법정 제출기한)로 보수 추정.
        # filing 매칭 실패로 AvailableDate가 빈 분기(예: 최신 분기)가 통째로 PIT에서
        # 누락되는 것을 방지하면서, 과거 as-of 백테스트의 시점 정합성도 유지한다.
        if "PeriodEnd" in f.columns:
            avail = avail.fillna(_to_naive_series(f["PeriodEnd"]) + pd.Timedelta(days=45))
        f = f.loc[avail <= asof_naive].copy()
    if f.empty:
        return f

    fiscal_source = f.copy()
    sort_candidates = [c for c in ["PeriodEnd", "AvailableDate", "FilingDate", "AcceptedAt"] if c in fiscal_source.columns]
    if sort_candidates:
        fiscal_source = fiscal_source.sort_values(sort_candidates)
    fiscal_source = fiscal_source.groupby("PeriodEnd", as_index=False).head(1)
    sort_cols = [c for c in ["PeriodEnd", "FilingDate", "AcceptedAt", "AvailableDate", "CollectedAt"] if c in f.columns]
    if sort_cols:
        f = f.sort_values(sort_cols)

    # Collapse same-period filings into one PIT row:
    # keep the latest available filing as the base row, then fill any missing
    # metric columns from earlier filings for the same PeriodEnd.
    merged_rows: list[pd.Series] = []
    for _, chunk in f.groupby("PeriodEnd", sort=True, dropna=False):
        ordered = chunk.sort_values(sort_cols, kind="stable")
        row = ordered.iloc[-1].copy()
        for col in ordered.columns:
            if col == "PeriodEnd":
                continue
            if pd.notna(row.get(col)):
                continue
            vals = ordered[col].dropna()
            if not vals.empty:
                row[col] = vals.iloc[-1]
        merged_rows.append(row)
    f = pd.DataFrame(merged_rows).sort_values("PeriodEnd").reset_index(drop=True)
    if {"fiscal_year", "fiscal_quarter", "fiscal_label"}.issubset(fiscal_source.columns):
        fiscal_meta = fiscal_source.copy()
        fiscal_meta["period_end"] = pd.to_datetime(fiscal_meta.get("PeriodEnd"), errors="coerce").dt.normalize()
        fiscal_meta = fiscal_meta.loc[~fiscal_meta["period_end"].isna(), ["period_end", "fiscal_year", "fiscal_quarter", "fiscal_label"]]
    else:
        fiscal_meta = _infer_fiscal_period_meta(
            fiscal_source.get("PeriodEnd"),
            fiscal_source.get("FormType"),
            fiscal_source.get("PeriodStart"),
        )
    if not fiscal_meta.empty:
        fiscal_meta = fiscal_meta.set_index("period_end")
        period_index = pd.to_datetime(f.get("PeriodEnd"), errors="coerce").dt.normalize()
        for col in ("fiscal_year", "fiscal_quarter", "fiscal_label"):
            f[col] = fiscal_meta[col].reindex(period_index).to_numpy()
    return f


def _load_pit_derived_frame(
    ticker: str,
    market: str,
    asof: pd.Timestamp,
    *,
    basis: str | None = None,
) -> pd.DataFrame:
    raw = None
    if _HAS_DUCKDB:
        try:
            raw = load_derived_factors_from_db(ticker=ticker, market=market, basis=basis)
        except Exception:
            raw = None
    if raw is None or raw.empty:
        return pd.DataFrame()

    d = raw.copy()
    d["period_end"] = _to_naive_series(d.get("period_end"))
    d["available_date"] = _to_naive_series(d.get("available_date"))
    if "collected_at" in d.columns:
        d["collected_at"] = _to_naive_series(d.get("collected_at"))
    asof_naive = _to_naive_normalized_ts(asof)
    d = d.loc[d["available_date"] <= asof_naive].copy()
    if d.empty:
        return d
    if "basis" in d.columns:
        d["basis"] = d["basis"].astype(str).str.strip().str.lower()
    if basis:
        d = d.loc[d["basis"] == str(basis).strip().lower()].copy()
        if d.empty:
            return d
    sort_cols = [c for c in ["period_end", "available_date", "collected_at"] if c in d.columns]
    d = d.sort_values(sort_cols)
    group_cols = ["period_end"]
    if "basis" in d.columns:
        group_cols.append("basis")
    d = d.groupby(group_cols, as_index=False).tail(1)
    d = d.sort_values("period_end").set_index("period_end")
    return d


def _overlay_derived_columns(
    base: pd.DataFrame,
    derived: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if base is None or base.empty or derived is None or derived.empty:
        return base
    out = base.copy()
    aligned = derived.reindex(out.index)
    skip_cols = {"ticker", "market", "basis", "source", "collected_at", "available_date"}
    if columns is None:
        target_cols = [c for c in aligned.columns if c not in skip_cols]
    else:
        target_cols = columns
    for col in target_cols:
        if col not in aligned.columns:
            continue
        if col in out.columns:
            continue
        out[col] = pd.to_numeric(aligned[col], errors="coerce")
    return out


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    def _with_period_index(values: pd.Series) -> pd.Series:
        if frame is None or frame.empty:
            return values
        idx = frame.index
        if "PeriodEnd" in frame.columns:
            pe = pd.to_datetime(frame["PeriodEnd"], errors="coerce")
            if len(pe) == len(values):
                idx = pe
        return pd.Series(values.to_numpy(), index=idx)

    if frame is None or frame.empty:
        return pd.Series(np.nan, index=pd.RangeIndex(0), dtype=float)

    if name in frame.columns:
        return _with_period_index(_to_numeric_series(frame[name]))

    # Backward/forward compatible column resolution:
    # - quoted title-case DB columns ("Operating Income")
    # - snake_case aliases ("operating_income")
    # - case variations ("revenue", "REVENUE")
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", str(s).lower())
    want_norm = norm(name)
    cols = list(frame.columns)

    alias_candidates = {
        "Revenue": ["revenue"],
        "COGS": ["cogs", "cost_of_revenue"],
        "Gross Profit": ["gross_profit"],
        "SG&A": ["sga", "selling_general_admin", "sellinggeneralandadministrative"],
        "R&D": ["r_and_d", "research_and_development", "researchdevelopment", "researchanddevelopment"],
        "Operating Income": ["operating_income", "operatingincome"],
        "Net Income": ["net_income", "netincome"],
        "Operating Cash Flow": ["operating_cash_flow", "operatingcashflow", "cfo"],
        "Investing Cash Flow": ["investing_cash_flow", "investingcashflow", "cfi"],
        "Financing Cash Flow": ["financing_cash_flow", "financingcashflow", "cff"],
        "Capital Expenditure": ["capital_expenditure", "capex"],
        "Total Assets": ["total_assets", "assets"],
        "Total Liabilities": ["total_liabilities", "liabilities"],
        "Shareholders Equity": ["shareholders_equity", "equity"],
        "Current Assets": ["current_assets"],
        "Current Liabilities": ["current_liabilities"],
        "Debt Short": ["debt_short", "short_debt"],
        "Debt Long": ["debt_long", "long_debt"],
        "Deferred Revenue": ["deferred_revenue"],
        "Goodwill": ["goodwill"],
        "Intangibles": ["intangibles", "intangible_assets"],
        "Dividends Paid": ["dividends_paid", "dividends"],
        "Repurchases": ["repurchases", "buybacks"],
        "D&A": ["da", "depreciation_and_amortization"],
        "SBC": ["sbc", "stock_based_compensation"],
        "Pretax Income": ["pretax_income", "pre_tax_income"],
        "Tax": ["tax", "tax_expense"],
        "AR": ["ar", "accounts_receivable", "receivables"],
        "AP": ["ap", "accounts_payable"],
        "Cash": ["cash", "cash_and_equivalents"],
        "Shares": ["shares", "diluted_shares", "basic_shares"],
        "EPS": ["eps", "diluted_eps"],
    }
    for candidate in alias_candidates.get(name, []):
        if candidate in frame.columns:
            return _with_period_index(_to_numeric_series(frame[candidate]))

    # Generic normalized match fallback
    for c in cols:
        if norm(c) == want_norm:
            return _with_period_index(_to_numeric_series(frame[c]))
    for c in cols:
        if want_norm and want_norm in norm(c):
            return _with_period_index(_to_numeric_series(frame[c]))

    return _with_period_index(pd.Series(np.nan, index=frame.index, dtype=float))


def _rolling_sum(series: pd.Series, window: int = 4) -> pd.Series:
    return series.rolling(window=window, min_periods=window).sum()


def _pct_change(series: pd.Series, periods: int = 4) -> pd.Series:
    out = series.pct_change(periods=periods, fill_method=None) * 100.0
    return out.replace([np.inf, -np.inf], np.nan)


def _pick_close_col(px: pd.DataFrame) -> str | None:
    for c in ("Adj Close", "adj_close", "Close", "close"):
        if c in px.columns:
            return c
    return None


def _resolve_market_cap(
    ticker: str,
    market: str,
    index: "pd.Index",
) -> "pd.Series":
    """Return quarterly market cap from JSON financials.market_cap array.

    The export step writes split-adjusted market_cap (sourced from
    valuation_ttm when available) into financials.market_cap.
    """
    try:
        data = _json_svc.load_ticker_data(ticker, market)
        if data and "financials" in data:
            fin = data["financials"]
            periods = fin.get("periods", [])
            mc_array = fin.get("market_cap", [])
            if periods and mc_array and len(periods) == len(mc_array):
                json_series = pd.Series(
                    [float(v) if v is not None else np.nan for v in mc_array],
                    index=pd.to_datetime(periods, errors="coerce"),
                    dtype=float,
                ).sort_index()
                reindexed = json_series.reindex(index, method="ffill")
                if reindexed.notna().any():
                    return reindexed
    except Exception:
        pass
    return pd.Series(np.nan, index=index, dtype=float)


def _compute_financial_feature_frame(q: pd.DataFrame, px: pd.DataFrame, ticker: str = "", market: str = "kr") -> pd.DataFrame:
    if q.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=pd.to_datetime(q["PeriodEnd"], errors="coerce"))
    out.index.name = "date"

    revenue = _col(q, "Revenue")
    op_income = _col(q, "Operating Income")
    net_income = _col(q, "Net Income")
    gross_profit = _col(q, "Gross Profit")
    cogs = _col(q, "COGS")
    revenue = _fill_revenue_from_components(revenue, cogs, gross_profit)
    sga = _col(q, "SG&A")
    r_and_d = _col(q, "R&D")
    cfo = _col(q, "Operating Cash Flow")
    cfi = _col(q, "Investing Cash Flow")
    cff = _col(q, "Financing Cash Flow")
    capex = _col(q, "Capital Expenditure")
    da = _col(q, "D&A")
    sbc = _col(q, "SBC")
    interest = _col(q, "Interest")
    pretax = _col(q, "Pretax Income")
    tax = _col(q, "Tax")
    dividends = _col(q, "Dividends Paid")
    repurchases = _col(q, "Repurchases")
    assets = _col(q, "Total Assets")
    liabilities = _col(q, "Total Liabilities")
    equity = _col(q, "Shareholders Equity")
    current_assets = _col(q, "Current Assets")
    current_liabilities = _col(q, "Current Liabilities")
    ar = _col(q, "AR")
    ap = _col(q, "AP")
    inventory = _col(q, "Inventory")
    cash = _col(q, "Cash")
    debt_short = _col(q, "Debt Short")
    debt_long = _col(q, "Debt Long")
    deferred_revenue = _col(q, "Deferred Revenue")
    goodwill = _col(q, "Goodwill")
    intangibles = _col(q, "Intangibles")
    shares = _col(q, "Shares").replace(0, np.nan)
    # KR fallback: derive shares from market_cap / close when Shares tag is missing
    if shares.isna().all() and px is not None and not px.empty:
        _close_col = _pick_close_col(px)
        _mcap_col = "market_cap" if "market_cap" in px.columns else None
        if _close_col and _mcap_col:
            _close = pd.to_numeric(px[_close_col], errors="coerce").sort_index()
            _mcap = pd.to_numeric(px[_mcap_col], errors="coerce").sort_index()
            _derived_shares = (_mcap / _close.replace(0, np.nan)).dropna()
            if not _derived_shares.empty:
                _derived_at_q = _derived_shares.reindex(q.index, method="ffill")
                shares = _derived_at_q.replace(0, np.nan)
    eps = _col(q, "EPS")

    rev_ttm = _rolling_sum(revenue)
    op_ttm = _rolling_sum(op_income)
    ni_ttm = _rolling_sum(net_income)
    gp_ttm = _rolling_sum(gross_profit)
    cfo_ttm = _rolling_sum(cfo)
    capex_ttm = _rolling_sum(capex)
    da_ttm = _rolling_sum(da)
    interest_ttm = _rolling_sum(interest)
    pretax_ttm = _rolling_sum(pretax)
    tax_ttm = _rolling_sum(tax)
    dividends_ttm = _rolling_sum(dividends).abs()
    repurchases_ttm = _rolling_sum(repurchases).abs()

    avg_assets = (assets + assets.shift(1)) / 2
    avg_equity = (equity + equity.shift(1)) / 2

    out["revenue"] = revenue
    out["operating_income"] = op_income
    out["net_income"] = net_income
    out["gross_profit"] = gross_profit
    out["cogs"] = cogs
    out["sga"] = sga
    out["r_and_d"] = r_and_d

    out["r_and_d_display"], out["sga_ex_r_and_d"] = _derive_sga_split(
        gross_profit=gross_profit,
        sga=sga,
        r_and_d=r_and_d,
        operating_income=op_income,
    )
    out["r_and_d_ratio"] = out["r_and_d_display"] / revenue.replace(0, np.nan) * 100.0
    out["sga_ex_r_and_d_ratio"] = out["sga_ex_r_and_d"] / revenue.replace(0, np.nan) * 100.0
    out["r_and_d_growth"] = out["r_and_d_display"].pct_change(4) * 100
    out["sga_ex_r_and_d_growth"] = out["sga_ex_r_and_d"].pct_change(4) * 100

    out["cfo"] = cfo
    out["cfi"] = cfi
    out["cff"] = cff
    out["capex"] = capex
    out["da"] = da
    out["sbc"] = sbc
    out["interest"] = interest
    out["pre_tax_income"] = pretax
    out["tax_expense"] = tax
    out["dividends_paid"] = dividends
    out["repurchases"] = repurchases
    out["assets"] = assets
    out["liabilities"] = liabilities
    out["equity"] = equity
    out["current_assets"] = current_assets
    out["current_liabilities"] = current_liabilities
    out["receivables"] = ar
    out["accounts_payable"] = ap
    out["inventory"] = inventory
    out["cash"] = cash
    out["debt_short"] = debt_short
    out["debt_long"] = debt_long
    out["deferred_revenue"] = deferred_revenue
    out["goodwill"] = goodwill
    out["intangible_assets"] = intangibles
    out["shares"] = shares
    out["eps"] = eps

    out["revenue_ttm"] = rev_ttm
    out["operating_income_ttm"] = op_ttm
    out["net_income_ttm"] = ni_ttm
    out["gross_profit_ttm"] = gp_ttm
    out["cfo_ttm"] = cfo_ttm
    out["fcf_ttm"] = cfo_ttm - capex_ttm
    out["da_ttm"] = da_ttm
    out["interest_expense_ttm"] = interest_ttm
    out["pretax_income_ttm"] = pretax_ttm
    out["tax_ttm"] = tax_ttm
    out["dividends_ttm"] = dividends_ttm
    out["repurchases_ttm"] = repurchases_ttm
    out["debt_total"] = debt_short.fillna(0.0) + debt_long.fillna(0.0)
    out["cash_and_equivalents"] = cash
    out["ebitda_ttm"] = op_ttm + da_ttm

    out["gross_margin"] = gp_ttm / rev_ttm * 100.0
    out["op_margin"] = op_ttm / rev_ttm * 100.0
    out["net_margin"] = ni_ttm / rev_ttm * 100.0
    out["cogs_ratio"] = _rolling_sum(cogs) / rev_ttm * 100.0
    out["sga_ratio"] = _rolling_sum(sga) / rev_ttm * 100.0

    out["revenue_growth"] = _pct_change(rev_ttm, 4)
    out["operating_income_growth"] = _pct_change(op_ttm, 4)
    out["net_income_growth"] = _pct_change(ni_ttm, 4)
    out["gross_profit_growth"] = _pct_change(gp_ttm, 4)
    out["eps_growth"] = _pct_change(eps, 4)
    out["earnings_growth"] = _pct_change(ni_ttm, 4)
    out["cost_growth"] = _pct_change(_rolling_sum(cogs + sga), 4)
    out["capital_growth"] = _pct_change(equity, 4)

    out["debt_ratio"] = liabilities / equity * 100.0
    out["current_ratio"] = current_assets / current_liabilities.replace(0, np.nan) * 100.0

    out["roa"] = ni_ttm / avg_assets * 100.0
    out["roe"] = ni_ttm / avg_equity * 100.0
    out["roic"] = op_ttm / (avg_assets - liabilities.shift(1).fillna(liabilities)) * 100.0
    out["gpa"] = gp_ttm / avg_assets * 100.0
    out["controlling_roe"] = out["roe"]

    out["asset_turnover"] = rev_ttm / avg_assets
    out["inventory_turnover"] = out["asset_turnover"]
    out["receivable_turnover"] = out["asset_turnover"]
    out["operating_cycle_days"] = 365 / out["asset_turnover"].replace(0, np.nan)
    out["cash_cycle_days"] = out["operating_cycle_days"]
    out["cash_conversion_ratio"] = cfo_ttm / ni_ttm.replace(0, np.nan) * 100.0

    out["bps"] = equity / shares
    out["sps"] = rev_ttm / shares
    out["opps"] = op_ttm / shares
    out["nips"] = ni_ttm / shares
    out["cfops"] = cfo_ttm / shares
    out["fcfps"] = out["fcf_ttm"] / shares

    close_col = _pick_close_col(px)
    if close_col is not None and not px.empty:
        close = pd.to_numeric(px[close_col], errors="coerce")
        close = close.sort_index()
        price_at = close.reindex(out.index, method="ffill")
    else:
        price_at = pd.Series(np.nan, index=out.index, dtype=float)

    out["price"] = price_at
    out["market_cap"] = _resolve_market_cap(ticker, market, out.index)

    out["per"] = out["price"] / out["eps"].replace(0, np.nan)
    out["pbr"] = out["price"] / out["bps"].replace(0, np.nan)
    out["psr"] = out["price"] / out["sps"].replace(0, np.nan)
    out["por"] = out["price"] / out["opps"].replace(0, np.nan)
    out["pfcfr"] = out["price"] / out["fcfps"].replace(0, np.nan)
    enterprise_value = out["market_cap"] + out["debt_total"].fillna(0.0) - out["cash_and_equivalents"].fillna(0.0)
    out["ev_ebitda"] = enterprise_value / out["ebitda_ttm"].replace(0, np.nan)
    out["peg"] = out["per"] / (out["eps_growth"].abs().replace(0, np.nan))

    out["price_return"] = _pct_change(out["price"], 4)

    # valuation band helpers (mid/low/high with dashed lines in frontend)
    band_cols: dict[str, pd.Series] = {}
    for base in ("pbr", "psr", "per", "por", "pfcfr"):
        mid = out[base].rolling(window=8, min_periods=2).median()
        band_cols[f"{base}_mid"] = mid
        band_cols[f"{base}_low"] = mid * 0.8
        band_cols[f"{base}_high"] = mid * 1.2
    if band_cols:
        out = pd.concat([out, pd.DataFrame(band_cols, index=out.index)], axis=1)

    # aliases used by registry keys
    alias_map = {
        "operating_income": "operating_income",
        "net_income": "net_income",
        "gross_margin": "gross_margin",
        "op_margin": "op_margin",
        "net_margin": "net_margin",
        "cogs_ratio": "cogs_ratio",
        "sga_ratio": "sga_ratio",
        "cogs_growth": "cost_growth",
        "sga_growth": "cost_growth",
        "op_growth": "operating_income_growth",
        "net_growth": "net_income_growth",
        "gross_profit_growth": "gross_profit_growth",
        "pre_tax_income": "pre_tax_income",
        "tax_expense": "tax_expense",
        "non_op_total": "pre_tax_income",
        "other_gain": "pre_tax_income",
        "financial_gain": "interest",
        "equity_method_gain": "pre_tax_income",
        "non_current_assets": "assets",
        "ppe": "assets",
        "subsidiary_investment": "goodwill",
        "rev_total": "revenue",
        "rev_a": "revenue",
        "rev_b": "revenue",
        "rev_c": "revenue",
        "rev_d": "revenue",
        "rev_e": "revenue",
        "rev_f": "revenue",
        "mix_a": "revenue_growth",
        "mix_b": "op_margin",
        "mix_c": "net_margin",
        "mix_d": "revenue_growth",
        "mix_e": "op_margin",
        "mix_f": "net_margin",
        "op_a": "operating_income",
        "op_b": "operating_income",
        "op_c": "operating_income",
        "op_d": "operating_income",
        "op_e": "operating_income",
        "op_f": "operating_income",
        "op_mix_a": "op_margin",
        "op_mix_b": "net_margin",
        "op_mix_c": "gross_margin",
        "op_mix_d": "op_margin",
        "op_mix_e": "net_margin",
        "op_mix_f": "gross_margin",
        "rev_domestic": "revenue",
        "rev_export": "revenue",
        "rev_other": "revenue",
        "share_a": "gross_margin",
        "share_b": "op_margin",
        "share_c": "net_margin",
        "biz_a": "revenue_growth",
        "biz_b": "op_margin",
        "biz_c": "net_margin",
        "usa": "revenue",
        "europe": "revenue",
        "asia": "revenue",
        "other": "revenue",
        "debt_total": "debt_total",
        "cash_and_equivalents": "cash_and_equivalents",
        "ebitda_ttm": "ebitda_ttm",
        "interest_expense_ttm": "interest_expense_ttm",
        "dividends_ttm": "dividends_ttm",
        "repurchases_ttm": "repurchases_ttm",
        "deferred_revenue": "deferred_revenue",
        "goodwill": "goodwill",
    }

    alias_cols = {target: out[src] for target, src in alias_map.items() if target not in out.columns and src in out.columns}
    if alias_cols:
        out = pd.concat([out, pd.DataFrame(alias_cols, index=out.index)], axis=1)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.sort_index()
    return out


def _unit_for_key(key: str) -> str:
    low = key.lower()
    if any(tok in low for tok in ["margin", "ratio", "growth", "roe", "roa", "roic", "gpa", "return"]):
        return "%"
    if any(tok in low for tok in ["per", "pbr", "psr", "por", "peg", "ev_ebitda", "pfcfr"]):
        return "x"
    if any(tok in low for tok in ["days", "cycle"]):
        return "days"
    if any(tok in low for tok in ["price", "market_cap", "revenue", "income", "assets", "equity", "cash", "capex", "bps", "sps", "eps", "cfps", "fcfps", "opps", "nips", "cfops"]):
        return "KRW"
    return ""


def _build_series_payload(
    tab: str,
    ticker: str,
    asof: pd.Timestamp,
    feat: pd.DataFrame,
    q_pit: pd.DataFrame,
    notes: dict[str, str],
    *,
    subtab: str | None = None,
    keys_override: set[str] | None = None,
    preset_series: dict[str, list[SeriesPoint]] | None = None,
    preset_missing: dict[str, str] | None = None,
    available_snapshot: dict[str, Any] | None = None,
    business_block: dict[str, Any] | None = None,
    product_block: dict[str, Any] | None = None,
    geography_block: dict[str, Any] | None = None,
    business_missing_map: dict[str, str] | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> TickerTabResponse:
    keys = keys_override if keys_override is not None else _load_graph_keys(tab, subtab=subtab)

    series: dict[str, list[SeriesPoint]] = dict(preset_series or {})
    series_missing: dict[str, str] = dict(preset_missing or {})
    units: dict[str, str] = {}

    if feat.empty:
        for key in sorted(keys):
            units[key] = _unit_for_key(key)
            if key in series:
                continue
            series[key] = []
            series_missing.setdefault(key, "PIT 재무 스냅샷 없음")
    else:
        for key in sorted(keys):
            units[key] = _unit_for_key(key)
            if key in series:
                continue
            if key in feat.columns:
                series[key] = _series_points(feat.index, feat[key])
                if pd.to_numeric(feat[key], errors="coerce").dropna().empty:
                    series_missing.setdefault(key, "값 계산 불가(원천 컬럼 부족)")
            else:
                series[key] = []
                series_missing.setdefault(key, "GraphRegistry key는 있으나 백엔드 계산식 미정의")

    period_end = None
    available_date = None
    if not q_pit.empty:
        _pit_sort = [c for c in ["PeriodEnd", "AvailableDate"] if c in q_pit.columns]
        latest = q_pit.sort_values(_pit_sort).iloc[-1] if _pit_sort else q_pit.iloc[-1]
        period_end = _to_iso_date(latest.get("PeriodEnd"))
        available_date = _to_iso_date(latest.get("AvailableDate")) if "AvailableDate" in q_pit.columns else None

    extra_data = {
        "tab": tab,
        "subtab": subtab,
        "series_keys": len(series),
        "missing_keys": len([k for k, v in series_missing.items() if v]),
    }
    if extra_payload:
        extra_data.update(extra_payload)

    return TickerTabResponse(
        ticker=ticker,
        asof=asof.date().isoformat(),
        snapshot=SnapshotInfo(period_end=period_end, available_date=available_date),
        series=series,
        meta=TabMeta(units=units, notes=notes, missing=series_missing),
        available_snapshot=available_snapshot,
        business=business_block,
        product=product_block,
        geography=geography_block,
        missing=business_missing_map,
        extra=extra_data,
    )


def get_ticker_tab_payload(
    ticker: str,
    market: str = "auto",
    tab: str = "summary",
    asof: str | None = None,
    subtab: str | None = None,
) -> TickerTabResponse:
    preloaded = _build_preloaded_analysis_data(ticker=ticker, market=market, asof=asof)
    tkr = preloaded.ticker
    asof_ts = preloaded.asof_ts
    resolved_market = preloaded.resolved_market
    q_pit = preloaded.q_pit
    feat = _preloaded_financial_feature_frame(preloaded)

    notes: dict[str, str] = {
        "pit_rule": "AvailableDate <= asof 기준으로 기간별 최신 filing 사용",
        "market": resolved_market,
    }
    if str(market or "").strip().lower() == "auto":
        notes["market_resolution"] = f"auto -> {resolved_market}"

    if q_pit.empty:
        notes["warning"] = "asof 시점에 사용 가능한 재무 PIT 데이터가 없습니다."
    notes["pit_rows"] = str(int(q_pit.shape[0]))
    notes["pit_cols"] = str(int(q_pit.shape[1])) if not q_pit.empty else "0"

    # Summary/valuation/fundamentals/financials: feature frame 기반 series
    if tab == "summary":
        pass
    return _build_series_payload(tab=tab, ticker=tkr, asof=asof_ts, feat=feat, q_pit=q_pit, notes=notes)


def _build_chart_payload(
    *,
    title: str,
    unit_left: str,
    unit_right: str,
    notes: str,
    categories: list[str],
    series: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "meta": {
            "title": title,
            "unit_left": unit_left,
            "unit_right": unit_right,
            "notes": notes,
        },
        "series": series,
        "categories": categories,
    }


def _rebuild_pie_map_from_feature_frame(
    feat: pd.DataFrame,
    pie_map: dict[str, list[SeriesPoint]] | None,
) -> dict[str, list[SeriesPoint]]:
    if feat is None or feat.empty:
        return pie_map or {}
    base = dict(pie_map or {})
    legend_names = [str(point.date) for point in base.get("revenue_mix_value", []) if getattr(point, "date", None)]
    default_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "기타"]
    latest = feat.iloc[-1]
    revenue_points: list[SeriesPoint] = []
    for idx, suffix in enumerate((*_BUSINESS_BUCKET_KEYS, _BUSINESS_OTHER_KEY)):
        mix_key = f"mix_{suffix}"
        if mix_key not in feat.columns:
            continue
        mix_val = _safe_num(latest.get(mix_key))
        if mix_val is None:
            continue
        name = legend_names[idx] if idx < len(legend_names) else default_names[idx]
        revenue_points.append(SeriesPoint(date=name, value=mix_val))
    if revenue_points:
        base["revenue_mix_value"] = revenue_points
    return base


def _chart_series(
    *,
    key: str = "",
    name: str,
    chart_type: str,
    y_axis: str,
    x_values: list[str],
    y_values: pd.Series,
    dashed: bool = False,
) -> dict[str, Any]:
    y_ser = pd.to_numeric(y_values, errors="coerce")
    points = [
        {"x": x, "y": (_safe_num(y) if pd.notna(y) else None)}
        for x, y in zip(x_values, y_ser, strict=False)
    ]
    return {
        "key": key,
        "name": name,
        "type": chart_type,
        "yAxis": y_axis,
        "dashed": bool(dashed),
        "data": points,
    }


def _topn_with_others(series: pd.Series, top_n: int = 8) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return s
    # Sort by absolute magnitude to keep largest positive and negative drivers
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    if len(s) <= top_n:
        return s
    top = s.iloc[:top_n].copy()
    others = float(s.iloc[top_n:].sum())
    if np.isfinite(others) and others != 0.0:
        top.loc["기타"] = others
    return top


def _summary_window_mode(quarters: int) -> str:
    if quarters <= 20:
        return "5y"
    if quarters <= 40:
        return "10y"
    return "all"


def _trim_summary_chart(chart: dict[str, Any] | None, max_points: int) -> dict[str, Any] | None:
    if chart is None:
        return None

    out = {
        "meta": dict(chart.get("meta") or {}),
        "series": [],
        "categories": list(chart.get("categories") or []),
    }
    if "missing_reason" in chart:
        out["missing_reason"] = chart.get("missing_reason")

    categories = out["categories"]
    if not categories:
        for series in chart.get("series", []):
            data = series.get("data") or []
            if data:
                categories = [str(point.get("x")) for point in data]
                break
        out["categories"] = categories

    limit = max(int(max_points), 1)
    if categories and len(categories) > limit:
        out["categories"] = categories[-limit:]

    for raw_series in chart.get("series", []):
        series = dict(raw_series)
        data = list(raw_series.get("data") or [])
        if str(series.get("type")) != "pie" and len(data) > limit:
            series["data"] = data[-limit:]
        else:
            series["data"] = data
        out["series"].append(series)
    return out


def get_summary_dashboard_payload(
    ticker: str,
    market: str = "auto",
    asof: str | None = None,
    quarters: int = 20,
    basis: str = "ttm",
) -> dict[str, Any]:
    preloaded = _build_preloaded_analysis_data(ticker=ticker, market=market, asof=asof)
    tkr = preloaded.ticker
    asof_ts = preloaded.asof_ts
    q_window = max(4, int(quarters))
    basis_norm = str(basis or "ttm").strip().lower()
    if basis_norm not in {"ttm", "quarter", "annual"}:
        basis_norm = "ttm"

    resolved_market = preloaded.resolved_market
    q_pit = preloaded.q_pit

    latest_period_end = None
    latest_available_date = None
    if not q_pit.empty:
        _pit_sort2 = [c for c in ["PeriodEnd", "AvailableDate"] if c in q_pit.columns]
        latest_row = q_pit.sort_values(_pit_sort2).iloc[-1] if _pit_sort2 else q_pit.iloc[-1]
        latest_period_end = _to_iso_date(latest_row.get("PeriodEnd"))
        latest_available_date = _to_iso_date(latest_row.get("AvailableDate")) if "AvailableDate" in q_pit.columns else None
    window_mode = _summary_window_mode(q_window)
    max_points = q_window if basis_norm != "annual" else max(1, q_window // 4)

    income_payload = get_financials_income_payload(
        tkr,
        resolved_market,
        window=window_mode,
        basis=basis_norm,
        preloaded_data=preloaded,
    )
    balance_payload = get_financials_balance_payload(
        tkr,
        resolved_market,
        window=window_mode,
        basis=basis_norm,
        preloaded_data=preloaded,
    )
    cashflow_payload = get_financials_cashflow_payload(
        tkr,
        resolved_market,
        window=window_mode,
        basis=basis_norm,
        preloaded_data=preloaded,
    )
    fundamentals_payload = get_fundamentals_payload(
        tkr,
        resolved_market,
        window=window_mode,
        basis=basis_norm,
        preloaded_data=preloaded,
    )
    valuation_payload = get_valuation_payload(
        tkr,
        resolved_market,
        window=window_mode,
        basis=basis_norm,
        preloaded_data=preloaded,
    )

    financial_charts = {
        "performance_overview": _trim_summary_chart(income_payload["charts"].get("is_perf_overview"), max_points),
        "balance_overview": _trim_summary_chart(balance_payload["charts"].get("bs_financial_status"), max_points),
        "cashflow_overview": _trim_summary_chart(cashflow_payload["charts"].get("cf_cashflow_overview"), max_points),
    }

    fundamentals_charts = {
        "dupont_roe": _trim_summary_chart(fundamentals_payload["charts"].get("fn_dupont"), max_points),
        "profitability": _trim_summary_chart(fundamentals_payload["charts"].get("fn_margins_combo"), max_points),
        "revenue_growth_yoy": _trim_summary_chart(fundamentals_payload["charts"].get("fn_profit_growth"), max_points),
    }

    valuation_charts = {
        "pbr_band": _trim_summary_chart(valuation_payload["charts"].get("val_band_pbr"), max_points),
        "per_price": _trim_summary_chart(valuation_payload["charts"].get("val_multiple_per"), max_points),
        "eps_price": _trim_summary_chart(valuation_payload["charts"].get("val_ps_eps"), max_points),
    }

    return {
        "ticker": tkr,
        "asof": str(asof).strip() if asof else None,
        "window": {"quarters": q_window},
        "snapshot": {
            "latest_period_end": latest_period_end,
            "latest_available_date": latest_available_date,
        },
        "charts": {
            "financial": financial_charts,
            "fundamentals": fundamentals_charts,
            "valuation": valuation_charts,
        },
    }


def _income_flow_columns() -> list[str]:
    return [
        "revenue",
        "cogs",
        "sga",
        "r_and_d",
        "gross_profit",
        "operating_income",
        "net_income",
        "pre_tax_income",
        "tax_expense",
        "other_gain",
        "financial_gain",
        "equity_method_gain",
        "other_income",
        "other_expense",
        "financial_income",
        "financial_expense",
        "non_op_total",
    ]


def _recompute_income_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    revenue = pd.to_numeric(out.get("revenue"), errors="coerce")
    cogs = pd.to_numeric(out.get("cogs"), errors="coerce")
    sga = pd.to_numeric(out.get("sga"), errors="coerce")
    op = pd.to_numeric(out.get("operating_income"), errors="coerce")
    net = pd.to_numeric(out.get("net_income"), errors="coerce")
    gross = pd.to_numeric(out.get("gross_profit"), errors="coerce")
    pre_tax = pd.to_numeric(out.get("pre_tax_income"), errors="coerce")
    tax = pd.to_numeric(out.get("tax_expense"), errors="coerce")

    out["gross_margin"] = gross / revenue.replace(0, np.nan) * 100.0
    out["op_margin"] = op / revenue.replace(0, np.nan) * 100.0
    out["net_margin"] = net / revenue.replace(0, np.nan) * 100.0
    out["pre_tax_margin"] = pre_tax / revenue.replace(0, np.nan) * 100.0
    out["cogs_ratio"] = cogs / revenue.replace(0, np.nan) * 100.0
    out["sga_ratio"] = sga / revenue.replace(0, np.nan) * 100.0
    out["total_cost_ratio"] = (cogs + sga) / revenue.replace(0, np.nan) * 100.0
    out["effective_tax_rate"] = np.where(pre_tax > 0, tax / pre_tax.replace(0, np.nan) * 100.0, np.nan)
    return out


def _period_labels(frame: pd.DataFrame, basis: str) -> list[str]:
    if frame is None or frame.empty:
        return []

    index = pd.DatetimeIndex(frame.index)
    if basis == "annual":
        fiscal_year = pd.to_numeric(frame.get("fiscal_year"), errors="coerce")
        if isinstance(fiscal_year, pd.Series) and fiscal_year.notna().any():
            return [
                str(int(fy)) if pd.notna(fy) else str(ts.year)
                for ts, fy in zip(index, fiscal_year, strict=False)
            ]
        return [str(ts.year) for ts in index]

    fiscal_label = frame.get("fiscal_label")
    if isinstance(fiscal_label, pd.Series) and fiscal_label.notna().any():
        labels: list[str] = []
        for ts, label in zip(index, fiscal_label, strict=False):
            txt = str(label).strip()
            labels.append(txt if txt and txt not in {"nan", "<NA>"} else f"{ts.year}Q{ts.quarter}")
        return labels
    return [f"{ts.year}Q{ts.quarter}" for ts in index]


def _build_income_base_quarter_frame(q_pit: pd.DataFrame, px: pd.DataFrame, ticker: str = "", market: str = "kr") -> pd.DataFrame:
    if q_pit.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=pd.to_datetime(q_pit["PeriodEnd"], errors="coerce"))
    out.index.name = "period_end"
    out = out.loc[~out.index.isna()].sort_index()

    revenue = _col(q_pit, "Revenue")
    cogs = _col(q_pit, "COGS")
    sga = _col(q_pit, "SG&A")
    r_and_d = _col(q_pit, "R&D")
    gross = _col(q_pit, "Gross Profit")
    revenue = _fill_revenue_from_components(revenue, cogs, gross)
    gross = gross.where(gross.notna(), revenue - cogs)
    op = _col(q_pit, "Operating Income")
    net = _col(q_pit, "Net Income")
    pre_tax = _col(q_pit, "Pretax Income")
    tax = _col(q_pit, "Tax")
    pre_tax, tax = _fill_pretax_and_tax(pre_tax, tax, net)

    # Non-op related columns can be sparse in SEC companyfacts.
    other_gain = _col(q_pit, "Other Gain")
    financial_gain = _col(q_pit, "Financial Gain")
    equity_gain = _col(q_pit, "Equity Method Gain")
    other_income = _col(q_pit, "Other Income")
    other_expense = _col(q_pit, "Other Expense")
    financial_income = _col(q_pit, "Financial Income")
    financial_expense = _col(q_pit, "Financial Expense")

    shares = _col(q_pit, "Shares").replace(0, np.nan)
    close_col = _pick_close_col(px)
    if close_col is not None and not px.empty:
        price = pd.to_numeric(px[close_col], errors="coerce").sort_index().reindex(out.index, method="ffill")
    else:
        price = pd.Series(np.nan, index=out.index, dtype=float)

    out["revenue"] = revenue
    out["cogs"] = cogs
    out["sga"] = sga
    out["r_and_d"] = r_and_d
    out["gross_profit"] = gross
    out["operating_income"] = op
    out["net_income"] = net
    out["pre_tax_income"] = pre_tax
    out["tax_expense"] = tax
    out["other_gain"] = other_gain
    out["financial_gain"] = financial_gain
    out["equity_method_gain"] = equity_gain
    out["other_income"] = other_income
    out["other_expense"] = other_expense
    out["financial_income"] = financial_income
    out["financial_expense"] = financial_expense
    out["shares"] = shares
    out["price"] = price
    out["market_cap"] = _resolve_market_cap(ticker, market, out.index)

    non_op_from_parts = other_gain.fillna(0.0) + financial_gain.fillna(0.0) + equity_gain.fillna(0.0)
    out["non_op_total"] = (pre_tax - op).where((pre_tax - op).notna(), non_op_from_parts)
    out = _recompute_income_ratios(out)
    out = _attach_fiscal_metadata(out, q_pit)
    return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _annualize_income_frame(quarter_frame: pd.DataFrame) -> pd.DataFrame:
    if quarter_frame.empty:
        return pd.DataFrame()
    flow_cols = [c for c in _income_flow_columns() if c in quarter_frame.columns]
    stock_cols = [c for c in ["shares", "price", "market_cap"] if c in quarter_frame.columns]
    rows: list[dict[str, Any]] = []
    fiscal_year = _period_year_series(quarter_frame)
    years = sorted({int(v) for v in fiscal_year.dropna().unique().tolist()})
    for year in years:
        chunk = quarter_frame.loc[fiscal_year == year]
        if chunk.empty:
            continue
        row: dict[str, Any] = {
            "period_end": pd.Timestamp(chunk.index.max()).normalize(),
            "fiscal_year": year,
            "fiscal_quarter": 4,
            "fiscal_label": str(year),
        }
        for col in flow_cols:
            row[col] = pd.to_numeric(chunk[col], errors="coerce").sum(min_count=1)
        for col in stock_cols:
            vals = pd.to_numeric(chunk[col], errors="coerce").dropna()
            row[col] = vals.iloc[-1] if not vals.empty else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("period_end").sort_index()
    out = _recompute_income_ratios(out)
    return out.replace([np.inf, -np.inf], np.nan)


def _drop_leading_fy_row(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop leading FY annual aggregate row from quarterly data.

    Detects: first row's revenue is >2.5x the median of the next 4 rows.
    """
    if frame.empty or len(frame) < 5:
        return frame
    rev_col = "revenue" if "revenue" in frame.columns else None
    if rev_col is None:
        return frame
    vals = pd.to_numeric(frame[rev_col], errors="coerce")
    first_val = vals.iloc[0]
    next_median = vals.iloc[1:5].median()
    if pd.notna(first_val) and pd.notna(next_median) and next_median > 0 and first_val > next_median * 2.5:
        return frame.iloc[1:].copy()
    return frame


def _rolling_ttm(series: pd.Series) -> pd.Series:
    """Rolling 4-quarter sum with annualized fill for leading partial periods."""
    rolling_sum = series.rolling(window=4, min_periods=1).sum()
    rolling_cnt = series.notna().rolling(window=4, min_periods=1).sum()
    result = rolling_sum.copy()
    # Annualize partial periods (1-3 quarters)
    partial = rolling_cnt < 4
    result[partial] = (rolling_sum[partial] / rolling_cnt[partial] * 4)
    return result.where(rolling_cnt > 0)


def _to_basis_income_frame(quarter_frame: pd.DataFrame, basis: str) -> tuple[pd.DataFrame, int]:
    if quarter_frame.empty:
        return pd.DataFrame(), 1

    if basis == "annual":
        annual = _annualize_income_frame(quarter_frame)
        return annual, 1

    if basis == "ttm":
        out = _drop_leading_fy_row(quarter_frame).copy()
        for col in _income_flow_columns():
            if col in out.columns:
                out[col] = _rolling_ttm(pd.to_numeric(out[col], errors="coerce"))
        out = _recompute_income_ratios(out)
        return out.replace([np.inf, -np.inf], np.nan), 4

    return _drop_leading_fy_row(quarter_frame).copy(), 4


def _normalize_window_spec(window: str | None) -> str:
    window_norm = str(window or "10y").strip().lower()
    if window_norm == "all":
        return "all"
    match = re.fullmatch(r"(\d+)y", window_norm)
    if not match:
        return "10y"
    years = max(1, min(40, int(match.group(1))))
    return f"{years}y"


def _apply_income_window(frame: pd.DataFrame, window: str, basis: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    window_norm = _normalize_window_spec(window)
    if window_norm == "all":
        return frame
    years = int(window_norm[:-1])
    size = years if basis == "annual" else years * 4
    return frame.tail(size)


def _drop_leading_annual_spike(frame: pd.DataFrame, basis: str) -> pd.DataFrame:
    """Drop the leading row if it's a Q4 with annual-cumulative values.

    When the first data point is Q4 and its revenue is ≥ 2.5× the median of
    the next 4 quarters, it's likely an annual cumulative that wasn't
    deaccumulated.  Drop it to prevent chart spikes.
    """
    if frame.empty or basis != "quarter" or len(frame) < 5:
        return frame
    fq = frame.get("fiscal_quarter")
    rev = pd.to_numeric(frame.get("revenue"), errors="coerce")
    if fq is None or not rev.notna().any():
        return frame
    first_q = fq.iloc[0]
    first_rev = rev.iloc[0]
    if first_q == 4 and pd.notna(first_rev) and first_rev > 0:
        next_median = rev.iloc[1:5].median()
        if pd.notna(next_median) and next_median > 0 and first_rev > next_median * 2.5:
            return frame.iloc[1:].copy()
    return frame


def _pct_change_series(
    series: pd.Series,
    lag: int,
    *,
    fiscal_year: pd.Series | None = None,
    fiscal_quarter: pd.Series | None = None,
) -> pd.Series:
    current = pd.to_numeric(series, errors="coerce")
    prev: pd.Series | None = None
    if fiscal_year is not None and fiscal_quarter is not None:
        fy = pd.to_numeric(fiscal_year, errors="coerce")
        fq = pd.to_numeric(fiscal_quarter, errors="coerce")
        if isinstance(fy, pd.Series) and isinstance(fq, pd.Series):
            unique_quarters = {int(v) for v in fq.dropna().unique().tolist()}
            if len(unique_quarters) > 1:
                key = pd.Series(pd.NA, index=current.index, dtype="Int64")
                mask = fy.notna() & fq.notna()
                if mask.any():
                    key.loc[mask] = (fy.loc[mask].astype(int) * 4 + fq.loc[mask].astype(int)).astype("Int64")
                    lookup = pd.DataFrame({"key": key, "value": current}).dropna(subset=["key"]).groupby("key")["value"].last()
                    prev_key = key - lag
                    prev = pd.Series(prev_key.map(lookup).to_numpy(), index=current.index, dtype=float)
    if prev is None:
        prev = current.shift(lag)
    out = (current / prev.where(prev > 0, np.nan) - 1.0) * 100.0
    return out.replace([np.inf, -np.inf], np.nan)


def _fill_pretax_and_tax(pre_tax: pd.Series, tax: pd.Series, net_income: pd.Series) -> tuple[pd.Series, pd.Series]:
    pre = pd.to_numeric(pre_tax, errors="coerce")
    tx = pd.to_numeric(tax, errors="coerce")
    net = pd.to_numeric(net_income, errors="coerce")
    pre = pre.where(pre.notna(), net + tx)
    tx = tx.where(tx.notna(), pre - net)
    return pre, tx


def _income_chart_payload(
    *,
    frame: pd.DataFrame,
    labels: list[str],
    title: str,
    unit_left: str,
    unit_right: str,
    notes: str,
    series_defs: list[dict[str, Any]],
) -> dict[str, Any]:
    series_payload: list[dict[str, Any]] = []
    missing_keys: list[str] = []
    non_empty = 0
    for sdef in series_defs:
        key = str(sdef["key"])
        if key in frame.columns:
            y_values = pd.to_numeric(frame[key], errors="coerce")
        else:
            y_values = pd.Series(np.nan, index=frame.index, dtype=float)
            missing_keys.append(key)
        payload = _chart_series(
            key=key,
            name=str(sdef["name"]),
            chart_type=str(sdef["type"]),
            y_axis=str(sdef["yAxis"]),
            x_values=labels,
            y_values=y_values,
            dashed=bool(sdef.get("dashed", False)),
        )
        if any(p.get("y") is not None for p in payload["data"]):
            non_empty += 1
        else:
            missing_keys.append(key)
        series_payload.append(payload)

    missing_reason = None
    # market_cap/price/shares are optional supplementary series —
    # don't flag them as missing since they depend on external price data
    _optional_keys = {"market_cap", "price", "shares"}
    _significant_missing = [k for k in missing_keys if k not in _optional_keys]
    if non_empty == 0:
        missing_reason = "데이터 없음(원천 컬럼 미지원)"
    elif _significant_missing:
        uniq = ", ".join(sorted(set(_significant_missing))[:6])
        missing_reason = f"일부 항목 누락: {uniq}"

    out = _build_chart_payload(
        title=title,
        unit_left=unit_left,
        unit_right=unit_right,
        notes=notes,
        categories=labels,
        series=series_payload,
    )
    out["missing_reason"] = missing_reason
    return out


def _apply_kr_rd_service_policy(
    basis_frame: pd.DataFrame,
    chart_defs: dict[str, dict[str, Any]],
    *,
    resolved_market: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if str(resolved_market).strip().lower() != "kr":
        return basis_frame, chart_defs

    out = basis_frame.copy()
    if "sga" in out.columns:
        sga_total = pd.to_numeric(out.get("sga"), errors="coerce")
    else:
        sga_total = pd.Series(np.nan, index=out.index, dtype=float)

    if "sga_growth" in out.columns:
        sga_growth = pd.to_numeric(out.get("sga_growth"), errors="coerce")
    else:
        sga_growth = pd.Series(np.nan, index=out.index, dtype=float)

    if "sga_ratio" in out.columns:
        sga_ratio = pd.to_numeric(out.get("sga_ratio"), errors="coerce")
    else:
        sga_ratio = pd.Series(np.nan, index=out.index, dtype=float)

    out["r_and_d_display"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["r_and_d_ratio"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["r_and_d_growth"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["sga_ex_r_and_d"] = sga_total
    out["sga_ex_r_and_d_ratio"] = sga_ratio
    out["sga_ex_r_and_d_growth"] = sga_growth

    defs = copy.deepcopy(chart_defs)

    def _rename_key(series_defs: list[dict[str, Any]], key: str, new_name: str) -> None:
        for item in series_defs:
            if str(item.get("key")) == key:
                item["name"] = new_name

    def _drop_key(series_defs: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        return [item for item in series_defs if str(item.get("key")) != key]

    for chart_id in ("is_perf_overview", "is_revenue_mix_stack", "is_sga_compare"):
        cfg = defs.get(chart_id)
        if not cfg:
            continue
        cfg["series"] = _drop_key(list(cfg.get("series", [])), "r_and_d_display")
        _rename_key(cfg["series"], "sga_ex_r_and_d", "판관비")
        cfg["notes"] = f"{cfg.get('notes', '')} | kr_r_and_d_hidden=true"

    for chart_id in ("is_revenue_mix_ratio", "is_cost_ratio", "is_cost_growth_lines", "is_sga_compare"):
        cfg = defs.get(chart_id)
        if not cfg:
            continue
        cfg["series"] = _drop_key(list(cfg.get("series", [])), "r_and_d_ratio")
        cfg["series"] = _drop_key(list(cfg.get("series", [])), "r_and_d_growth")
        _rename_key(cfg["series"], "sga_ex_r_and_d_ratio", "판매관리비율")
        _rename_key(cfg["series"], "sga_ex_r_and_d_growth", "판매관리비 증가율")
        cfg["notes"] = f"{cfg.get('notes', '')} | kr_r_and_d_hidden=true"

    return out, defs


def _apply_nonop_service_policy(
    basis_frame: pd.DataFrame,
    chart_defs: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    out = basis_frame.copy()
    for column in (
        "other_gain",
        "financial_gain",
        "equity_method_gain",
        "non_op_total",
        "other_income",
        "other_expense",
        "financial_income",
        "financial_expense",
    ):
        if column in out.columns:
            out[column] = pd.Series(np.nan, index=out.index, dtype=float)

    defs = copy.deepcopy(chart_defs)
    for chart_id in ("is_nonop_total", "is_other_income", "is_financial_income", "is_equity_method"):
        cfg = defs.get(chart_id)
        if not cfg:
            continue
        cfg["notes"] = f"{cfg.get('notes', '')} | nonop_hidden=true"
    return out, defs


def get_financials_income_payload(
    ticker: str,
    market: str = "auto",
    window: str = "10y",
    basis: str = "ttm",
    *,
    preloaded_data: _AnalysisPreloadedData | None = None,
) -> dict[str, Any]:
    tkr = str(ticker).strip().upper()
    window_norm = _normalize_window_spec(window)
    basis_norm = str(basis or "ttm").strip().lower()
    if basis_norm not in {"ttm", "quarter", "annual"}:
        basis_norm = "ttm"

    preloaded = preloaded_data or _build_preloaded_analysis_data(ticker=tkr, market=market)
    tkr = preloaded.ticker
    resolved_market = preloaded.resolved_market
    asof_ts = preloaded.asof_ts
    quarter = _preloaded_base_quarter_frame(preloaded, "income")
    basis_frame, growth_lag = _to_basis_income_frame(quarter, basis_norm)
    basis_frame = _apply_income_window(basis_frame, window_norm, basis_norm).copy()
    basis_frame = _drop_leading_annual_spike(basis_frame, basis_norm)

    if basis_frame.empty:
        empty_chart = _income_chart_payload(
            frame=pd.DataFrame(index=pd.DatetimeIndex([])),
            labels=[],
            title="데이터 없음",
            unit_left="",
            unit_right="",
            notes="해당 티커의 손익 PIT 데이터가 없습니다.",
            series_defs=[{"key": "revenue", "name": "매출액", "type": "line", "yAxis": "left"}],
        )
        return {
            "ticker": tkr,
            "window": window_norm,
            "basis": basis_norm,
            "periods": [],
            "table": {"kpis": []},
            "charts": {cid: empty_chart for cid in [
                "is_perf_overview",
                "is_revenue_line",
                "is_op_income_line",
                "is_revenue_mix_stack",
                "is_revenue_mix_ratio",
                "is_revenue_growth",
                "is_profitability_combo",
                "is_profit_growth_lines",
                "is_gross_profit",
                "is_gross_profit_growth",
                "is_op_income_barline",
                "is_op_income_growth",
                "is_net_income_barline",
                "is_net_income_growth",
                "is_cost_ratio",
                "is_cogs_compare",
                "is_sga_compare",
                "is_cost_growth_lines",
                "is_nonop_total",
                "is_other_income",
                "is_financial_income",
                "is_equity_method",
                "is_pretax",
                "is_tax",
            ]},
        }

    def _growth_series_for_chart(series: pd.Series) -> pd.Series:
        primary = _pct_change_series(
            series,
            growth_lag,
            fiscal_year=basis_frame.get("fiscal_year"),
            fiscal_quarter=basis_frame.get("fiscal_quarter"),
        )
        if basis_norm == "ttm" and primary.notna().sum() <= 1:
            return _pct_change_series(
                series,
                1,
                fiscal_year=basis_frame.get("fiscal_year"),
                fiscal_quarter=basis_frame.get("fiscal_quarter"),
            )
        return primary

    basis_frame["revenue_growth"] = _growth_series_for_chart(basis_frame["revenue"])
    basis_frame["gross_profit_growth"] = _growth_series_for_chart(basis_frame["gross_profit"])
    basis_frame["operating_income_growth"] = _growth_series_for_chart(basis_frame["operating_income"])
    basis_frame["net_income_growth"] = _growth_series_for_chart(basis_frame["net_income"])
    basis_frame["cogs_growth"] = _growth_series_for_chart(basis_frame["cogs"])
    basis_frame["sga_growth"] = _growth_series_for_chart(basis_frame["sga"])
    basis_frame["price_return"] = _growth_series_for_chart(basis_frame["price"])
    derived_basis = _preloaded_derived_frame(preloaded, basis_norm)
    basis_frame = _overlay_derived_columns(
        basis_frame,
        derived_basis,
        columns=[
            "gross_margin",
            "op_margin",
            "net_margin",
            "cogs_ratio",
            "sga_ratio",
            "total_cost_ratio",
            "revenue_growth",
            "gross_profit_growth",
            "operating_income_growth",
            "net_income_growth",
            "cogs_growth",
            "sga_growth",
            "price_return",
            "fcf",
            "ccr",
        ],
    )
    sga_total = pd.to_numeric(basis_frame.get("sga"), errors="coerce")
    r_and_d = pd.to_numeric(basis_frame.get("r_and_d"), errors="coerce")
    cogs_raw = pd.to_numeric(basis_frame.get("cogs"), errors="coerce")
    revenue = pd.to_numeric(basis_frame.get("revenue"), errors="coerce")
    oi_raw = pd.to_numeric(basis_frame.get("operating_income"), errors="coerce")

    # ── Detect "operating expense only" companies ──────────────────────────
    # Companies like Naver report only 영업수익/영업비용/영업이익 without
    # breaking down into COGS/SGA.  When COGS+SGA are mostly empty but
    # Revenue and OI are present, compute 영업비용 = Revenue - OI and use
    # that as the single stacked component instead of COGS+SGA+R&D.
    _has_cogs_sga = (cogs_raw.notna().sum() + sga_total.notna().sum()) / max(len(basis_frame), 1)
    _has_rev_oi = (revenue.notna() & oi_raw.notna()).sum() / max(len(basis_frame), 1)
    _opex_only = _has_cogs_sga < 0.3 and _has_rev_oi > 0.5

    if _opex_only:
        # Use 영업비용 (= Revenue - OI) as single stacked bar
        opex = revenue - oi_raw
        basis_frame["operating_expense"] = opex
        basis_frame["r_and_d_display"] = pd.Series(np.nan, index=basis_frame.index)
        basis_frame["sga_ex_r_and_d"] = pd.Series(np.nan, index=basis_frame.index)
        basis_frame["cogs"] = pd.Series(np.nan, index=basis_frame.index)
        basis_frame["opex_ratio"] = opex / revenue.replace(0, np.nan) * 100.0
    else:
        basis_frame["operating_expense"] = pd.Series(np.nan, index=basis_frame.index)

    if not _opex_only:
        basis_frame["r_and_d_display"], basis_frame["sga_ex_r_and_d"] = _derive_sga_split(
            gross_profit=pd.to_numeric(basis_frame.get("gross_profit"), errors="coerce"),
            sga=sga_total,
            r_and_d=r_and_d,
            operating_income=pd.to_numeric(basis_frame.get("operating_income"), errors="coerce"),
        )
    basis_frame["sga_ex_r_and_d_ratio"] = basis_frame["sga_ex_r_and_d"] / revenue.replace(0, np.nan) * 100.0
    basis_frame["r_and_d_ratio"] = basis_frame["r_and_d_display"] / revenue.replace(0, np.nan) * 100.0
    basis_frame["sga_ex_r_and_d_growth"] = _growth_series_for_chart(basis_frame["sga_ex_r_and_d"])
    basis_frame["r_and_d_growth"] = _growth_series_for_chart(basis_frame["r_and_d_display"])

    labels = _period_labels(basis_frame, basis_norm)
    change_lag = growth_lag
    kpi_rows: list[dict[str, Any]] = []
    rev_chg = _pct_change_series(
        basis_frame["revenue"],
        change_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    op_chg = _pct_change_series(
        basis_frame["operating_income"],
        change_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    ni_chg = _pct_change_series(
        basis_frame["net_income"],
        change_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    for i, label in enumerate(labels):
        idx = basis_frame.index[i]
        kpi_rows.append(
            {
                "period": label,
                "revenue": _safe_num(basis_frame.at[idx, "revenue"]),
                "revenue_chg_pct": _safe_num(rev_chg.iloc[i]),
                "op_income": _safe_num(basis_frame.at[idx, "operating_income"]),
                "op_income_chg_pct": _safe_num(op_chg.iloc[i]),
                "net_income": _safe_num(basis_frame.at[idx, "net_income"]),
                "net_income_chg_pct": _safe_num(ni_chg.iloc[i]),
            }
        )

    _ccy = "KRW" if resolved_market == "kr" else "USD"
    chart_defs: dict[str, dict[str, Any]] = {
        "is_perf_overview": {
            "title": "실적종합",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "stacked view for comparison: COGS+판매관리비+연구개발비+영업이익, 시가총액은 우측축",
            "series": [
                {"key": "revenue", "name": "매출액", "type": "line", "yAxis": "left"},
                {"key": "operating_income", "name": "영업이익", "type": "stackedBar", "yAxis": "left"},
                {"key": "r_and_d_display", "name": "연구개발비", "type": "stackedBar", "yAxis": "left"},
                {"key": "sga_ex_r_and_d", "name": "판관비(R&D제외)", "type": "stackedBar", "yAxis": "left"},
                {"key": "cogs", "name": "매출원가", "type": "stackedBar", "yAxis": "left"},
                {"key": "operating_expense", "name": "영업비용", "type": "stackedBar", "yAxis": "left"},
                {"key": "market_cap", "name": "시가총액", "type": "line", "yAxis": "right"},
            ],
        },
        "is_revenue_line": {
            "title": "매출액 증분",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "직전 표시기간 대비 매출액 증분 워터폴 + 시가총액",
            "series": [
                {"key": "revenue", "name": "매출액", "type": "bar", "yAxis": "left"},
                {"key": "market_cap", "name": "시가총액", "type": "line", "yAxis": "right"},
            ],
        },
        "is_op_income_line": {
            "title": "영업이익 증분",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "직전 표시기간 대비 영업이익 증분 워터폴 + 시가총액",
            "series": [
                {"key": "operating_income", "name": "영업이익", "type": "bar", "yAxis": "left"},
                {"key": "market_cap", "name": "시가총액", "type": "line", "yAxis": "right"},
            ],
        },
        "is_revenue_mix_stack": {
            "title": "매출액 구성",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": "COGS + 판매관리비 + 연구개발비 + 영업이익 누적",
            "series": [
                {"key": "operating_income", "name": "영업이익", "type": "stackedBar", "yAxis": "left"},
                {"key": "r_and_d_display", "name": "연구개발비", "type": "stackedBar", "yAxis": "left"},
                {"key": "sga_ex_r_and_d", "name": "판관비(R&D제외)", "type": "stackedBar", "yAxis": "left"},
                {"key": "cogs", "name": "매출원가", "type": "stackedBar", "yAxis": "left"},
                {"key": "operating_expense", "name": "영업비용", "type": "stackedBar", "yAxis": "left"},
            ],
        },
        "is_revenue_mix_ratio": {
            "title": "매출 구성비중",
            "unit_left": "%",
            "unit_right": "",
            "notes": "영업이익률/판매관리비율/연구개발비율/매출원가율",
            "series": [
                {"key": "op_margin", "name": "영업이익률", "type": "line", "yAxis": "left"},
                {"key": "sga_ex_r_and_d_ratio", "name": "판매관리비율", "type": "line", "yAxis": "left"},
                {"key": "r_and_d_ratio", "name": "연구개발비율", "type": "line", "yAxis": "left"},
                {"key": "cogs_ratio", "name": "매출원가율", "type": "line", "yAxis": "left"},
                {"key": "opex_ratio", "name": "영업비용율", "type": "line", "yAxis": "left"},
            ],
        },
        "is_revenue_growth": {
            "title": "매출액 성장률",
            "unit_left": "%",
            "unit_right": "%",
            "notes": "basis 기준 증감률 + 주가수익률",
            "series": [
                {"key": "revenue_growth", "name": "매출액 성장률", "type": "bar", "yAxis": "left"},
                {"key": "price_return", "name": "주가수익률", "type": "line", "yAxis": "right"},
            ],
        },
        "is_profitability_combo": {
            "title": "이익률",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "매출 + 이익률 3종",
            "series": [
                {"key": "revenue", "name": "매출액", "type": "bar", "yAxis": "left"},
                {"key": "gross_margin", "name": "매출총이익률", "type": "line", "yAxis": "right"},
                {"key": "op_margin", "name": "영업이익률", "type": "line", "yAxis": "right"},
                {"key": "net_margin", "name": "순이익률", "type": "line", "yAxis": "right"},
            ],
        },
        "is_profit_growth_lines": {
            "title": "이익성장률",
            "unit_left": "%",
            "unit_right": "",
            "notes": "매출/매출총이익/영업이익/순이익 성장률",
            "series": [
                {"key": "revenue_growth", "name": "매출액 성장률", "type": "line", "yAxis": "left"},
                {"key": "gross_profit_growth", "name": "매출총이익 성장률", "type": "line", "yAxis": "left"},
                {"key": "operating_income_growth", "name": "영업이익 성장률", "type": "line", "yAxis": "left"},
                {"key": "net_income_growth", "name": "순이익 성장률", "type": "line", "yAxis": "left"},
            ],
        },
        "is_gross_profit": {
            "title": "매출총이익",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "매출총이익 + 매출총이익률",
            "series": [
                {"key": "gross_profit", "name": "매출총이익", "type": "bar", "yAxis": "left"},
                {"key": "gross_margin", "name": "매출총이익률", "type": "line", "yAxis": "right"},
            ],
        },
        "is_gross_profit_growth": {
            "title": "매출총이익 성장률",
            "unit_left": "%",
            "unit_right": "%",
            "notes": "매출총이익 성장률 + 주가수익률",
            "series": [
                {"key": "gross_profit_growth", "name": "매출총이익 성장률", "type": "bar", "yAxis": "left"},
                {"key": "price_return", "name": "주가수익률", "type": "line", "yAxis": "right"},
            ],
        },
        "is_op_income_barline": {
            "title": "영업이익",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "영업이익 + 영업이익률",
            "series": [
                {"key": "operating_income", "name": "영업이익", "type": "bar", "yAxis": "left"},
                {"key": "op_margin", "name": "영업이익률", "type": "line", "yAxis": "right"},
            ],
        },
        "is_op_income_growth": {
            "title": "영업이익 성장률",
            "unit_left": "%",
            "unit_right": "%",
            "notes": "영업이익 성장률 + 주가수익률",
            "series": [
                {"key": "operating_income_growth", "name": "영업이익 성장률", "type": "bar", "yAxis": "left"},
                {"key": "price_return", "name": "주가수익률", "type": "line", "yAxis": "right"},
            ],
        },
        "is_net_income_barline": {
            "title": "순이익",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "순이익 + 순이익률",
            "series": [
                {"key": "net_income", "name": "순이익", "type": "bar", "yAxis": "left"},
                {"key": "net_margin", "name": "순이익률", "type": "line", "yAxis": "right"},
            ],
        },
        "is_net_income_growth": {
            "title": "순이익 성장률",
            "unit_left": "%",
            "unit_right": "%",
            "notes": "순이익 성장률 + 주가수익률",
            "series": [
                {"key": "net_income_growth", "name": "순이익 성장률", "type": "bar", "yAxis": "left"},
                {"key": "price_return", "name": "주가수익률", "type": "line", "yAxis": "right"},
            ],
        },
        "is_cost_ratio": {
            "title": "비용률",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "매출액 + 매출원가율/판매관리비율/연구개발비율/총비용률",
            "series": [
                {"key": "revenue", "name": "매출액", "type": "bar", "yAxis": "left"},
                {"key": "cogs_ratio", "name": "매출원가율", "type": "line", "yAxis": "right"},
                {"key": "sga_ex_r_and_d_ratio", "name": "판매관리비율", "type": "line", "yAxis": "right"},
                {"key": "r_and_d_ratio", "name": "연구개발비율", "type": "line", "yAxis": "right"},
                {"key": "total_cost_ratio", "name": "총비용률", "type": "line", "yAxis": "right"},
            ],
        },
        "is_cogs_compare": {
            "title": "매출액 vs 매출원가",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "비교막대 + 매출원가율",
            "series": [
                {"key": "revenue", "name": "매출액", "type": "bar", "yAxis": "left"},
                {"key": "cogs", "name": "매출원가", "type": "bar", "yAxis": "left"},
                {"key": "cogs_ratio", "name": "매출원가율", "type": "line", "yAxis": "right"},
            ],
        },
        "is_sga_compare": {
            "title": "매출액 vs 판매관리비/연구개발비",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "비교막대 + 판매관리비율 + 연구개발비율",
            "series": [
                {"key": "revenue", "name": "매출액", "type": "bar", "yAxis": "left"},
                {"key": "sga_ex_r_and_d", "name": "판관비(R&D제외)", "type": "bar", "yAxis": "left"},
                {"key": "r_and_d_display", "name": "연구개발비", "type": "bar", "yAxis": "left"},
                {"key": "sga_ex_r_and_d_ratio", "name": "판매관리비율", "type": "line", "yAxis": "right"},
                {"key": "r_and_d_ratio", "name": "연구개발비율", "type": "line", "yAxis": "right"},
            ],
        },
        "is_cost_growth_lines": {
            "title": "비용 증가율",
            "unit_left": "%",
            "unit_right": "",
            "notes": "매출액/매출원가/판매관리비/연구개발비 증가율",
            "series": [
                {"key": "revenue_growth", "name": "매출액 성장률", "type": "line", "yAxis": "left"},
                {"key": "cogs_growth", "name": "매출원가 증가율", "type": "line", "yAxis": "left"},
                {"key": "sga_ex_r_and_d_growth", "name": "판매관리비 증가율", "type": "line", "yAxis": "left"},
                {"key": "r_and_d_growth", "name": "연구개발비 증가율", "type": "line", "yAxis": "left"},
            ],
        },
        "is_nonop_total": {
            "title": "영업외손익 합계",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "기타손익/금융손익/지분법손익 누적 + 합계",
            "series": [
                {"key": "other_gain", "name": "기타손익", "type": "stackedBar", "yAxis": "left"},
                {"key": "financial_gain", "name": "금융손익", "type": "stackedBar", "yAxis": "left"},
                {"key": "equity_method_gain", "name": "지분법손익", "type": "stackedBar", "yAxis": "left"},
                {"key": "non_op_total", "name": "합계", "type": "line", "yAxis": "right"},
            ],
        },
        "is_other_income": {
            "title": "기타손익",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "기타손익 + 기타수익/기타비용",
            "series": [
                {"key": "other_gain", "name": "기타손익", "type": "bar", "yAxis": "left"},
                {"key": "other_income", "name": "기타수익", "type": "line", "yAxis": "right"},
                {"key": "other_expense", "name": "기타비용", "type": "line", "yAxis": "right"},
            ],
        },
        "is_financial_income": {
            "title": "금융손익",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "금융손익 + 금융수익/금융비용",
            "series": [
                {"key": "financial_gain", "name": "금융손익", "type": "bar", "yAxis": "left"},
                {"key": "financial_income", "name": "금융수익", "type": "line", "yAxis": "right"},
                {"key": "financial_expense", "name": "금융비용", "type": "line", "yAxis": "right"},
            ],
        },
        "is_equity_method": {
            "title": "지분법손익",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": "지분법손익",
            "series": [{"key": "equity_method_gain", "name": "지분법손익", "type": "bar", "yAxis": "left"}],
        },
        "is_pretax": {
            "title": "법인세차감전순손익",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "PretaxIncome + Pretax margin",
            "series": [
                {"key": "pre_tax_income", "name": "법인세차감전순손익", "type": "bar", "yAxis": "left"},
                {"key": "pre_tax_margin", "name": "법인세차감전 이익률", "type": "line", "yAxis": "right"},
            ],
        },
        "is_tax": {
            "title": "법인세비용",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "Tax expense + effective tax rate (Pretax<=0이면 null)",
            "series": [
                {"key": "tax_expense", "name": "법인세비용", "type": "bar", "yAxis": "left"},
                {"key": "effective_tax_rate", "name": "법인세율", "type": "line", "yAxis": "right"},
            ],
        },
    }

    basis_frame, chart_defs = _apply_kr_rd_service_policy(
        basis_frame,
        chart_defs,
        resolved_market=resolved_market,
    )
    basis_frame, chart_defs = _apply_nonop_service_policy(
        basis_frame,
        chart_defs,
    )

    charts = {
        chart_id: _income_chart_payload(
            frame=basis_frame,
            labels=labels,
            title=cfg["title"],
            unit_left=cfg["unit_left"],
            unit_right=cfg["unit_right"],
            notes=f"{cfg['notes']} | basis={basis_norm}, window={window_norm}",
            series_defs=cfg["series"],
        )
        for chart_id, cfg in chart_defs.items()
    }

    return {
        "ticker": tkr,
        "window": window_norm,
        "basis": basis_norm,
        "periods": labels,
        "table": {"kpis": kpi_rows},
        "charts": charts,
    }


def _pick_first_nonnull_col(frame: pd.DataFrame, *names: str) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(np.nan, dtype=float)
    for name in names:
        s = _col(frame, name)
        if pd.to_numeric(s, errors="coerce").notna().any():
            return pd.to_numeric(s, errors="coerce")
    fallback = _col(frame, names[0]) if names else pd.Series(np.nan, dtype=float)
    return pd.to_numeric(fallback, errors="coerce")


def _build_balance_base_quarter_frame(q_pit: pd.DataFrame, px: pd.DataFrame, ticker: str = "", market: str = "kr") -> pd.DataFrame:
    if q_pit.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=pd.to_datetime(q_pit["PeriodEnd"], errors="coerce"))
    out.index.name = "period_end"
    out = out.loc[~out.index.isna()].sort_index()

    assets = _pick_first_nonnull_col(q_pit, "Total Assets", "Assets")
    liabilities = _pick_first_nonnull_col(q_pit, "Total Liabilities", "Liabilities")
    equity = _pick_first_nonnull_col(q_pit, "Shareholders Equity", "Equity", "Stockholders Equity")
    current_assets = _pick_first_nonnull_col(q_pit, "Current Assets", "CurrentAssets")
    non_current_assets = _pick_first_nonnull_col(q_pit, "Non Current Assets", "NonCurrentAssets", "Noncurrent Assets")
    current_liabilities = _pick_first_nonnull_col(q_pit, "Current Liabilities", "CurrentLiabilities")
    non_current_liabilities = _pick_first_nonnull_col(q_pit, "Non Current Liabilities", "NonCurrentLiabilities", "Noncurrent Liabilities")

    ppe = _pick_first_nonnull_col(
        q_pit,
        "Property Plant And Equipment",
        "PropertyPlantAndEquipment",
        "PPE",
    )
    intangibles = _pick_first_nonnull_col(q_pit, "Intangible Assets", "Intangibles")
    subsidiary_inv = _pick_first_nonnull_col(
        q_pit,
        "Investments In Subsidiaries",
        "InvestmentsInSubsidiaries",
        "Subsidiary Investment",
    )

    ar = _pick_first_nonnull_col(q_pit, "Accounts Receivable", "AR", "Receivables")
    inventory = _pick_first_nonnull_col(q_pit, "Inventory")
    ap = _pick_first_nonnull_col(q_pit, "Accounts Payable", "AP")

    common_stock = _pick_first_nonnull_col(q_pit, "Common Stock", "CommonStock")
    apic = _pick_first_nonnull_col(q_pit, "Additional Paid In Capital", "APIC", "AdditionalPaidInCapital")
    retained_earnings = _pick_first_nonnull_col(q_pit, "Retained Earnings", "RetainedEarnings")
    aoci = _pick_first_nonnull_col(
        q_pit,
        "Accumulated Other Comprehensive Income",
        "AOCI",
        "Other Equity",
        "OtherEquity",
    )
    owner_equity = _pick_first_nonnull_col(
        q_pit,
        "Equity Attributable To Owners",
        "Owner Equity",
        "Owners Equity",
        "OwnerEquity",
    )

    current_fin_assets = _pick_first_nonnull_col(
        q_pit,
        "Current Financial Assets",
        "Current Investments",
        "Short Term Investments",
        "Marketable Securities",
    )
    non_current_fin_assets = _pick_first_nonnull_col(
        q_pit,
        "Non Current Financial Assets",
        "Long Term Investments",
        "NonCurrentFinancialAssets",
    )
    current_fin_liabilities = _pick_first_nonnull_col(
        q_pit,
        "Current Financial Liabilities",
        "Debt Short",
        "Short Term Debt",
        "CurrentFinancialLiabilities",
    )
    non_current_fin_liabilities = _pick_first_nonnull_col(
        q_pit,
        "Non Current Financial Liabilities",
        "Debt Long",
        "Long Term Debt",
        "NonCurrentFinancialLiabilities",
    )

    revenue = _pick_first_nonnull_col(q_pit, "Revenue", "Sales")
    cogs = _pick_first_nonnull_col(q_pit, "COGS", "Cost Of Revenue")
    ppe_capex = _pick_first_nonnull_col(
        q_pit,
        "PPE CAPEX",
        "PPECapex",
        "Property Plant And Equipment Additions",
    )
    intangible_capex = _pick_first_nonnull_col(
        q_pit,
        "Intangible CAPEX",
        "IntangibleCapex",
        "Capital Expenditure Intangible",
    )
    depreciation = _pick_first_nonnull_col(
        q_pit,
        "Depreciation",
        "Depreciation And Amortization",
        "D&A",
    )
    amortization = _pick_first_nonnull_col(q_pit, "Amortization")

    shares = _pick_first_nonnull_col(q_pit, "Shares", "Diluted Shares", "Basic Shares").replace(0, np.nan)
    close_col = _pick_close_col(px)
    if close_col is not None and not px.empty:
        price = pd.to_numeric(px[close_col], errors="coerce").sort_index().reindex(out.index, method="ffill")
    else:
        price = pd.Series(np.nan, index=out.index, dtype=float)

    out["assets"] = assets
    out["liabilities"] = liabilities
    out["equity"] = equity
    out["current_assets"] = current_assets
    out["non_current_assets"] = non_current_assets
    out["current_liabilities"] = current_liabilities
    out["non_current_liabilities"] = non_current_liabilities
    out["revenue"] = revenue
    out["cogs"] = cogs
    out["market_cap"] = _resolve_market_cap(ticker, market, out.index)

    out["ppe"] = ppe
    out["intangibles"] = intangibles
    out["subsidiary_investment"] = subsidiary_inv
    out["ppe_capex"] = ppe_capex
    out["intangible_capex"] = intangible_capex
    out["depreciation"] = depreciation
    out["amortization"] = amortization

    out["ar"] = ar
    out["inventory"] = inventory
    out["ap"] = ap

    out["common_stock"] = common_stock
    out["apic"] = apic
    out["retained_earnings"] = retained_earnings
    out["aoci"] = aoci
    out["owner_equity"] = owner_equity.where(owner_equity.notna(), equity)

    out["current_fin_assets"] = current_fin_assets
    out["non_current_fin_assets"] = non_current_fin_assets
    out["current_fin_liabilities"] = current_fin_liabilities
    out["non_current_fin_liabilities"] = non_current_fin_liabilities

    # Fallback component completion from totals.
    out["non_current_assets"] = out["non_current_assets"].where(
        out["non_current_assets"].notna(),
        out["assets"] - out["current_assets"],
    )
    out["non_current_liabilities"] = out["non_current_liabilities"].where(
        out["non_current_liabilities"].notna(),
        out["liabilities"] - out["current_liabilities"],
    )

    fin_assets_total = out["current_fin_assets"].fillna(0.0) + out["non_current_fin_assets"].fillna(0.0)
    fin_assets_valid = out["current_fin_assets"].notna() | out["non_current_fin_assets"].notna()
    fin_assets_total = fin_assets_total.where(fin_assets_valid, np.nan)

    fin_liab_total = out["current_fin_liabilities"].fillna(0.0) + out["non_current_fin_liabilities"].fillna(0.0)
    fin_liab_valid = out["current_fin_liabilities"].notna() | out["non_current_fin_liabilities"].notna()
    fin_liab_total = fin_liab_total.where(fin_liab_valid, np.nan)

    out["fin_assets_total"] = fin_assets_total
    out["fin_liabilities_total"] = fin_liab_total
    out["net_fin_assets"] = fin_assets_total - fin_liab_total
    out["net_fin_assets_delta"] = out["net_fin_assets"].diff()
    out["net_fin_assets_to_mcap"] = out["net_fin_assets"] / out["market_cap"].replace(0, np.nan) * 100.0

    out["debt_ratio"] = out["liabilities"] / out["equity"].replace(0, np.nan) * 100.0
    out["net_wc"] = out["ar"] + out["inventory"] - out["ap"]
    out["net_wc_delta"] = out["net_wc"].diff()

    out["ppe_capex_ratio"] = out["ppe_capex"] / out["ppe"].replace(0, np.nan) * 100.0
    out["depreciation_ratio"] = out["depreciation"] / out["ppe"].replace(0, np.nan) * 100.0
    out["intangible_capex_ratio"] = out["intangible_capex"] / out["intangibles"].replace(0, np.nan) * 100.0
    out["amortization_ratio"] = out["amortization"] / out["intangibles"].replace(0, np.nan) * 100.0
    out["retained_earnings_delta"] = out["retained_earnings"].diff()

    rev_ttm = pd.to_numeric(out["revenue"], errors="coerce").rolling(window=4, min_periods=4).sum()
    cogs_ttm = pd.to_numeric(out["cogs"], errors="coerce").rolling(window=4, min_periods=4).sum()
    avg_ar = (out["ar"] + out["ar"].shift(1)) / 2.0
    avg_inventory = (out["inventory"] + out["inventory"].shift(1)) / 2.0
    avg_ap = (out["ap"] + out["ap"].shift(1)) / 2.0
    out["ar_turnover"] = rev_ttm / avg_ar.replace(0, np.nan)
    out["inventory_turnover"] = cogs_ttm / avg_inventory.replace(0, np.nan)
    out["ap_turnover"] = cogs_ttm / avg_ap.replace(0, np.nan)

    out = _attach_fiscal_metadata(out, q_pit)
    return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _annualize_balance_frame(quarter_frame: pd.DataFrame) -> pd.DataFrame:
    if quarter_frame.empty:
        return pd.DataFrame()

    flow_cols = [
        c
        for c in ["revenue", "cogs", "ppe_capex", "intangible_capex", "depreciation", "amortization"]
        if c in quarter_frame.columns
    ]
    excluded = set(flow_cols) | {"fiscal_year", "fiscal_quarter", "fiscal_label"}
    stock_cols = [c for c in quarter_frame.columns if c not in excluded]

    rows: list[dict[str, Any]] = []
    fiscal_year = _period_year_series(quarter_frame)
    years = sorted({int(v) for v in fiscal_year.dropna().unique().tolist()})
    for year in years:
        chunk = quarter_frame.loc[fiscal_year == year]
        if chunk.empty:
            continue
        row: dict[str, Any] = {
            "period_end": pd.Timestamp(chunk.index.max()).normalize(),
            "fiscal_year": year,
            "fiscal_quarter": 4,
            "fiscal_label": str(year),
        }
        for col in flow_cols:
            row[col] = pd.to_numeric(chunk[col], errors="coerce").sum(min_count=1)
        for col in stock_cols:
            vals = pd.to_numeric(chunk[col], errors="coerce")
            row[col] = vals.iloc[-1] if vals.notna().any() else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("period_end").sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


def _to_basis_balance_frame(quarter_frame: pd.DataFrame, basis: str) -> tuple[pd.DataFrame, int]:
    if quarter_frame.empty:
        return pd.DataFrame(), 1

    flow_cols = [
        c
        for c in ["revenue", "cogs", "ppe_capex", "intangible_capex", "depreciation", "amortization"]
        if c in quarter_frame.columns
    ]
    if basis == "annual":
        annual = _annualize_balance_frame(quarter_frame)
        return annual, 1
    if basis == "ttm":
        out = _drop_leading_fy_row(quarter_frame).copy()
        for col in flow_cols:
            if col in out.columns:
                out[col] = _rolling_ttm(pd.to_numeric(out[col], errors="coerce"))
        return out.replace([np.inf, -np.inf], np.nan), 4
    return _drop_leading_fy_row(quarter_frame).copy(), 4


def _project_balance_turnover(quarter_frame: pd.DataFrame, basis_index: pd.Index, basis: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    if quarter_frame.empty:
        empty = pd.Series(np.nan, index=basis_index, dtype=float)
        return empty, empty, empty

    ar = pd.to_numeric(quarter_frame.get("ar"), errors="coerce")
    inventory = pd.to_numeric(quarter_frame.get("inventory"), errors="coerce")
    ap = pd.to_numeric(quarter_frame.get("ap"), errors="coerce")
    revenue = pd.to_numeric(quarter_frame.get("revenue"), errors="coerce")
    cogs = pd.to_numeric(quarter_frame.get("cogs"), errors="coerce")

    rev_ttm = revenue.rolling(window=4, min_periods=4).sum()
    cogs_ttm = cogs.rolling(window=4, min_periods=4).sum()
    avg_ar = (ar + ar.shift(1)) / 2.0
    avg_inv = (inventory + inventory.shift(1)) / 2.0
    avg_ap = (ap + ap.shift(1)) / 2.0
    ar_turn = rev_ttm / avg_ar.replace(0, np.nan)
    inv_turn = cogs_ttm / avg_inv.replace(0, np.nan)
    ap_turn = cogs_ttm / avg_ap.replace(0, np.nan)

    if basis == "annual":
        def _annual_last(series: pd.Series) -> pd.Series:
            temp = pd.DataFrame({"v": series})
            temp["year"] = _period_year_series(quarter_frame).reindex(temp.index).to_numpy()
            rows: list[tuple[pd.Timestamp, float | None]] = []
            for year, grp in temp.groupby("year", sort=True):
                if pd.isna(year):
                    continue
                val = pd.to_numeric(grp["v"], errors="coerce").dropna()
                rows.append((pd.Timestamp(grp.index.max()).normalize(), (float(val.iloc[-1]) if not val.empty else np.nan)))
            if not rows:
                return pd.Series(np.nan, index=basis_index, dtype=float)
            out = pd.Series({k: v for k, v in rows}, dtype=float).sort_index()
            return out.reindex(pd.DatetimeIndex(basis_index))

        return _annual_last(ar_turn), _annual_last(inv_turn), _annual_last(ap_turn)

    return (
        pd.Series(ar_turn.to_numpy(), index=pd.DatetimeIndex(quarter_frame.index)).reindex(pd.DatetimeIndex(basis_index)),
        pd.Series(inv_turn.to_numpy(), index=pd.DatetimeIndex(quarter_frame.index)).reindex(pd.DatetimeIndex(basis_index)),
        pd.Series(ap_turn.to_numpy(), index=pd.DatetimeIndex(quarter_frame.index)).reindex(pd.DatetimeIndex(basis_index)),
    )


def _recompute_balance_derived(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    out = frame.copy()

    def _num_col(name: str) -> pd.Series:
        ser = pd.to_numeric(out.get(name), errors="coerce")
        if isinstance(ser, pd.Series):
            return ser.reindex(out.index)
        return pd.Series(np.nan, index=out.index, dtype=float)

    liabilities = _num_col("liabilities")
    equity = _num_col("equity")
    out["liabilities"] = liabilities
    out["equity"] = equity
    out["debt_ratio"] = liabilities / equity.replace(0, np.nan) * 100.0

    current_fin_assets = _num_col("current_fin_assets")
    non_current_fin_assets = _num_col("non_current_fin_assets")
    current_fin_liabilities = _num_col("current_fin_liabilities")
    non_current_fin_liabilities = _num_col("non_current_fin_liabilities")

    fin_assets_total = current_fin_assets.fillna(0.0) + non_current_fin_assets.fillna(0.0)
    fin_assets_valid = current_fin_assets.notna() | non_current_fin_assets.notna()
    fin_assets_total = fin_assets_total.where(fin_assets_valid, np.nan)
    fin_liab_total = current_fin_liabilities.fillna(0.0) + non_current_fin_liabilities.fillna(0.0)
    fin_liab_valid = current_fin_liabilities.notna() | non_current_fin_liabilities.notna()
    fin_liab_total = fin_liab_total.where(fin_liab_valid, np.nan)
    out["fin_assets_total"] = fin_assets_total
    out["fin_liabilities_total"] = fin_liab_total
    out["net_fin_assets"] = fin_assets_total - fin_liab_total
    out["net_fin_assets_delta"] = _num_col("net_fin_assets").diff()
    out["market_cap"] = _num_col("market_cap")
    out["net_fin_assets_to_mcap"] = _num_col("net_fin_assets") / out["market_cap"].replace(0, np.nan) * 100.0

    out["ar"] = _num_col("ar")
    out["inventory"] = _num_col("inventory")
    out["ap"] = _num_col("ap")
    out["net_wc"] = out["ar"] + out["inventory"] - out["ap"]
    out["net_wc_delta"] = _num_col("net_wc").diff()

    out["ppe"] = _num_col("ppe")
    out["intangibles"] = _num_col("intangibles")
    out["ppe_capex"] = _num_col("ppe_capex")
    out["depreciation"] = _num_col("depreciation")
    out["intangible_capex"] = _num_col("intangible_capex")
    out["amortization"] = _num_col("amortization")
    out["ppe_capex_ratio"] = out["ppe_capex"] / out["ppe"].replace(0, np.nan) * 100.0
    out["depreciation_ratio"] = out["depreciation"] / out["ppe"].replace(0, np.nan) * 100.0
    out["intangible_capex_ratio"] = out["intangible_capex"] / out["intangibles"].replace(0, np.nan) * 100.0
    out["amortization_ratio"] = out["amortization"] / out["intangibles"].replace(0, np.nan) * 100.0
    out["retained_earnings"] = _num_col("retained_earnings")
    out["retained_earnings_delta"] = out["retained_earnings"].diff()
    owner_equity = _num_col("owner_equity")
    out["owner_equity"] = owner_equity.where(owner_equity.notna(), equity)
    return out.replace([np.inf, -np.inf], np.nan)


def get_financials_balance_payload(
    ticker: str,
    market: str = "auto",
    window: str = "10y",
    basis: str = "ttm",
    *,
    preloaded_data: _AnalysisPreloadedData | None = None,
) -> dict[str, Any]:
    tkr = str(ticker).strip().upper()
    window_norm = _normalize_window_spec(window)
    basis_norm = str(basis or "ttm").strip().lower()
    if basis_norm not in {"ttm", "quarter", "annual"}:
        basis_norm = "ttm"

    preloaded = preloaded_data or _build_preloaded_analysis_data(ticker=tkr, market=market)
    tkr = preloaded.ticker
    resolved_market = preloaded.resolved_market
    asof_ts = preloaded.asof_ts
    quarter = _preloaded_base_quarter_frame(preloaded, "balance")
    basis_frame, growth_lag = _to_basis_balance_frame(quarter, basis_norm)
    basis_frame = _recompute_balance_derived(basis_frame)
    basis_frame = _apply_income_window(basis_frame, window_norm, basis_norm).copy()
    basis_frame = _drop_leading_annual_spike(basis_frame, basis_norm)

    chart_ids = [
        "bs_assets_composition",
        "bs_revenue_equity_mktcap",
        "bs_financial_status",
        "bs_asset_allocation",
        "bs_ppe_detail",
        "bs_intangibles_detail",
        "bs_fin_assets_liab",
        "bs_net_fin_assets_trend",
        "bs_working_capital_components",
        "bs_net_working_capital",
        "bs_turnover",
        "bs_equity_breakdown",
        "bs_retained_earnings",
        "bs_owner_equity",
        "bs_liabilities_status",
    ]

    if basis_frame.empty:
        empty_chart = _income_chart_payload(
            frame=pd.DataFrame(index=pd.DatetimeIndex([])),
            labels=[],
            title="데이터 없음",
            unit_left="",
            unit_right="",
            notes="해당 티커의 재무상태표 PIT 데이터가 없습니다.",
            series_defs=[{"key": "assets", "name": "자산총계", "type": "line", "yAxis": "left"}],
        )
        return {
            "ticker": tkr,
            "window": window_norm,
            "basis": basis_norm,
            "periods": [],
            "table": {"kpis": []},
            "charts": {cid: empty_chart for cid in chart_ids},
        }

    ar_turn, inv_turn, ap_turn = _project_balance_turnover(quarter, basis_frame.index, basis_norm)
    basis_frame["ar_turnover"] = ar_turn
    basis_frame["inventory_turnover"] = inv_turn
    basis_frame["ap_turnover"] = ap_turn
    derived_basis = _preloaded_derived_frame(preloaded, basis_norm)
    basis_frame = _overlay_derived_columns(
        basis_frame,
        derived_basis,
        columns=[
            "debt_ratio",
            "current_ratio",
            "ar_turnover",
            "inventory_turnover",
            "ap_turnover",
            "dso",
            "dio",
            "dpo",
            "operating_cycle",
            "cash_cycle",
        ],
    )

    labels = _period_labels(basis_frame, basis_norm)
    change_lag = growth_lag

    assets_chg = _pct_change_series(
        basis_frame["assets"],
        change_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    equity_chg = _pct_change_series(
        basis_frame["equity"],
        change_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    liabilities_chg = _pct_change_series(
        basis_frame["liabilities"],
        change_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    debt_ratio_chg = _pct_change_series(
        basis_frame["debt_ratio"],
        change_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    kpi_rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        idx = basis_frame.index[i]
        kpi_rows.append(
            {
                "period": label,
                "assets": _safe_num(basis_frame.at[idx, "assets"]),
                "assets_chg_pct": _safe_num(assets_chg.iloc[i]),
                "equity": _safe_num(basis_frame.at[idx, "equity"]),
                "equity_chg_pct": _safe_num(equity_chg.iloc[i]),
                "liabilities": _safe_num(basis_frame.at[idx, "liabilities"]),
                "liabilities_chg_pct": _safe_num(liabilities_chg.iloc[i]),
                "debt_ratio": _safe_num(basis_frame.at[idx, "debt_ratio"]),
                "debt_ratio_chg_pct": _safe_num(debt_ratio_chg.iloc[i]),
            }
        )

    _ccy = "KRW" if resolved_market == "kr" else "USD"
    chart_defs: dict[str, dict[str, Any]] = {
        "bs_assets_composition": {
            "title": "자산 구성",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "유동/비유동 자산 누적 + 자산총계",
            "series": [
                {"key": "current_assets", "name": "유동자산", "type": "stackedBar", "yAxis": "left"},
                {"key": "non_current_assets", "name": "비유동자산", "type": "stackedBar", "yAxis": "left"},
                {"key": "assets", "name": "자산총계", "type": "line", "yAxis": "right"},
            ],
        },
        "bs_revenue_equity_mktcap": {
            "title": "매출액/자본/시가총액",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "매출은 basis에 맞춰 변환, 자본/시총은 stock 값",
            "series": [
                {"key": "revenue", "name": "매출액", "type": "bar", "yAxis": "left"},
                {"key": "equity", "name": "자본총계", "type": "line", "yAxis": "right"},
                {"key": "market_cap", "name": "시가총액", "type": "line", "yAxis": "right"},
            ],
        },
        "bs_financial_status": {
            "title": "재무현황",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "부채비율 = Liabilities / Equity * 100",
            "series": [
                {"key": "equity", "name": "자본총계", "type": "stackedBar", "yAxis": "left"},
                {"key": "liabilities", "name": "부채총계", "type": "stackedBar", "yAxis": "left"},
                {"key": "debt_ratio", "name": "부채비율", "type": "line", "yAxis": "right"},
            ],
        },
        "bs_asset_allocation": {
            "title": "자산 배치",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": "유형/무형/자회사투자 구성",
            "series": [
                {"key": "ppe", "name": "유형자산", "type": "stackedBar", "yAxis": "left"},
                {"key": "intangibles", "name": "무형자산", "type": "stackedBar", "yAxis": "left"},
                {"key": "subsidiary_investment", "name": "자회사투자", "type": "stackedBar", "yAxis": "left"},
            ],
        },
        "bs_ppe_detail": {
            "title": "유형자산",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "CAPEX ratio = PPE_CAPEX / PPE, Depreciation ratio = Depreciation / PPE",
            "series": [
                {"key": "ppe", "name": "유형자산", "type": "bar", "yAxis": "left"},
                {"key": "ppe_capex", "name": "유형자산 CAPEX", "type": "line", "yAxis": "left"},
                {"key": "ppe_capex_ratio", "name": "CAPEX 비율", "type": "line", "yAxis": "right"},
                {"key": "depreciation", "name": "유형자산 상각비", "type": "line", "yAxis": "left"},
                {"key": "depreciation_ratio", "name": "상각비 비율", "type": "line", "yAxis": "right"},
            ],
        },
        "bs_intangibles_detail": {
            "title": "무형자산",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "CAPEX ratio = Intangible_CAPEX / Intangibles, Amortization ratio = Amortization / Intangibles",
            "series": [
                {"key": "intangibles", "name": "무형자산", "type": "bar", "yAxis": "left"},
                {"key": "intangible_capex", "name": "무형자산 CAPEX", "type": "line", "yAxis": "left"},
                {"key": "intangible_capex_ratio", "name": "CAPEX 비율", "type": "line", "yAxis": "right"},
                {"key": "amortization", "name": "무형자산 상각비", "type": "line", "yAxis": "left"},
                {"key": "amortization_ratio", "name": "상각비 비율", "type": "line", "yAxis": "right"},
            ],
        },
        "bs_fin_assets_liab": {
            "title": "금융자산/부채",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "NetFinancialAssets = (Current+NonCurrent FinancialAssets) - (Current+NonCurrent FinancialLiabilities)",
            "series": [
                {"key": "current_fin_assets", "name": "유동금융자산", "type": "stackedBar", "yAxis": "left"},
                {"key": "non_current_fin_assets", "name": "비유동금융자산", "type": "stackedBar", "yAxis": "left"},
                {"key": "current_fin_liabilities", "name": "유동금융부채", "type": "stackedBar", "yAxis": "left"},
                {"key": "non_current_fin_liabilities", "name": "비유동금융부채", "type": "stackedBar", "yAxis": "left"},
                {"key": "net_fin_assets", "name": "순금융자산", "type": "line", "yAxis": "right"},
            ],
        },
        "bs_net_fin_assets_trend": {
            "title": "순금융자산 추이",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "시총대비 = NetFinancialAssets/MarketCap*100, 증감 = t - t-1",
            "series": [
                {"key": "net_fin_assets", "name": "순금융자산", "type": "line", "yAxis": "left"},
                {"key": "net_fin_assets_to_mcap", "name": "시총대비", "type": "line", "yAxis": "right"},
                {"key": "net_fin_assets_delta", "name": "전분기말대비증감", "type": "line", "yAxis": "left"},
            ],
        },
        "bs_working_capital_components": {
            "title": "운전자본 구성",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": "AR/Inventory/AP",
            "series": [
                {"key": "ar", "name": "매출채권", "type": "line", "yAxis": "left"},
                {"key": "inventory", "name": "재고자산", "type": "line", "yAxis": "left"},
                {"key": "ap", "name": "매입채무", "type": "line", "yAxis": "left"},
            ],
        },
        "bs_net_working_capital": {
            "title": "순운전자본",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "NetWC = AR + Inventory - AP, 증감 = t - t-1",
            "series": [
                {"key": "net_wc", "name": "순운전자본", "type": "line", "yAxis": "left"},
                {"key": "net_wc_delta", "name": "전분기말대비증감", "type": "bar", "yAxis": "right"},
            ],
        },
        "bs_turnover": {
            "title": "회전율",
            "unit_left": "x",
            "unit_right": "",
            "notes": "AR turnover=Revenue(TTM)/AvgAR, Inventory/AP turnover=COGS(TTM)/AvgBalance",
            "series": [
                {"key": "ar_turnover", "name": "매출채권회전율", "type": "line", "yAxis": "left"},
                {"key": "inventory_turnover", "name": "재고자산회전율", "type": "line", "yAxis": "left"},
                {"key": "ap_turnover", "name": "매입채무회전율", "type": "line", "yAxis": "left"},
            ],
        },
        "bs_equity_breakdown": {
            "title": "자본 구성",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "OwnerEquity 미존재 시 Equity 대체",
            "series": [
                {"key": "common_stock", "name": "자본금", "type": "stackedBar", "yAxis": "left"},
                {"key": "apic", "name": "자본잉여금", "type": "stackedBar", "yAxis": "left"},
                {"key": "retained_earnings", "name": "이익잉여금", "type": "stackedBar", "yAxis": "left"},
                {"key": "aoci", "name": "기타자본항", "type": "stackedBar", "yAxis": "left"},
                {"key": "owner_equity", "name": "지배주주자본총계", "type": "line", "yAxis": "right"},
            ],
        },
        "bs_retained_earnings": {
            "title": "이익잉여금",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "변동 = t - t-1",
            "series": [
                {"key": "retained_earnings_delta", "name": "이익잉여금 변동", "type": "bar", "yAxis": "left"},
                {"key": "retained_earnings", "name": "이익잉여금", "type": "line", "yAxis": "right"},
            ],
        },
        "bs_owner_equity": {
            "title": "자본총계",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": "OwnerEquity 미존재 시 Equity 동일값 표시",
            "series": [
                {"key": "equity", "name": "자본총계", "type": "line", "yAxis": "left"},
                {"key": "owner_equity", "name": "지배주주자본총계", "type": "line", "yAxis": "left"},
            ],
        },
        "bs_liabilities_status": {
            "title": "부채 현황",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "유동/비유동 부채 누적 + 총부채",
            "series": [
                {"key": "current_liabilities", "name": "유동부채", "type": "stackedBar", "yAxis": "left"},
                {"key": "non_current_liabilities", "name": "비유동부채", "type": "stackedBar", "yAxis": "left"},
                {"key": "liabilities", "name": "부채총계", "type": "line", "yAxis": "right"},
            ],
        },
    }

    charts = {
        chart_id: _income_chart_payload(
            frame=basis_frame,
            labels=labels,
            title=cfg["title"],
            unit_left=cfg["unit_left"],
            unit_right=cfg["unit_right"],
            notes=f"{cfg['notes']} | basis={basis_norm}, window={window_norm}",
            series_defs=cfg["series"],
        )
        for chart_id, cfg in chart_defs.items()
    }

    return {
        "ticker": tkr,
        "window": window_norm,
        "basis": basis_norm,
        "periods": labels,
        "table": {"kpis": kpi_rows},
        "charts": charts,
    }


def _build_cashflow_base_quarter_frame(q_pit: pd.DataFrame, px: pd.DataFrame, ticker: str = "", market: str = "kr") -> pd.DataFrame:
    if q_pit.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=pd.to_datetime(q_pit["PeriodEnd"], errors="coerce"))
    out.index.name = "period_end"
    out = out.loc[~out.index.isna()].sort_index()

    cfo = _pick_first_nonnull_col(q_pit, "Operating Cash Flow", "OperatingCashFlow", "CFO")
    cfi = _pick_first_nonnull_col(q_pit, "Investing Cash Flow", "InvestingCashFlow", "CFI")
    cff = _pick_first_nonnull_col(q_pit, "Financing Cash Flow", "FinancingCashFlow", "CFF")
    capex_raw = _pick_first_nonnull_col(
        q_pit,
        "Capital Expenditure",
        "CapitalExpenditures",
        "CAPEX",
        "Capital Expenditures",
    )
    operating_income = _pick_first_nonnull_col(q_pit, "Operating Income", "OperatingIncome")
    net_income = _pick_first_nonnull_col(q_pit, "Net Income", "NetIncome")
    revenue = _pick_first_nonnull_col(q_pit, "Revenue", "Sales")
    shares = _pick_first_nonnull_col(q_pit, "Shares", "Diluted Shares", "Basic Shares").replace(0, np.nan)

    close_col = _pick_close_col(px)
    if close_col is not None and not px.empty:
        price = pd.to_numeric(px[close_col], errors="coerce").sort_index().reindex(out.index, method="ffill")
    else:
        price = pd.Series(np.nan, index=out.index, dtype=float)

    capex_outflow = capex_raw.abs()
    fcf = cfo - capex_outflow
    ccr = np.where(pd.to_numeric(net_income, errors="coerce") > 0, cfo / net_income.replace(0, np.nan), np.nan)

    out["cfo"] = cfo
    out["cfi"] = cfi
    out["cff"] = cff
    out["capex_raw"] = capex_raw
    out["capex_outflow"] = capex_outflow
    out["fcf"] = fcf
    out["operating_income"] = operating_income
    out["net_income"] = net_income
    out["revenue"] = revenue
    out["shares"] = shares
    out["price"] = price
    out["market_cap"] = _resolve_market_cap(ticker, market, out.index)
    out["ccr"] = pd.to_numeric(ccr, errors="coerce")

    out = _attach_fiscal_metadata(out, q_pit)
    return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _annualize_cashflow_frame(quarter_frame: pd.DataFrame) -> pd.DataFrame:
    if quarter_frame.empty:
        return pd.DataFrame()

    flow_cols = [
        c
        for c in [
            "cfo",
            "cfi",
            "cff",
            "capex_raw",
            "capex_outflow",
            "fcf",
            "operating_income",
            "net_income",
            "revenue",
        ]
        if c in quarter_frame.columns
    ]
    stock_cols = [c for c in ["shares", "price", "market_cap"] if c in quarter_frame.columns]

    rows: list[dict[str, Any]] = []
    fiscal_year = _period_year_series(quarter_frame)
    years = sorted({int(v) for v in fiscal_year.dropna().unique().tolist()})
    for year in years:
        chunk = quarter_frame.loc[fiscal_year == year]
        if chunk.empty:
            continue
        row: dict[str, Any] = {
            "period_end": pd.Timestamp(chunk.index.max()).normalize(),
            "fiscal_year": year,
            "fiscal_quarter": 4,
            "fiscal_label": str(year),
        }
        for col in flow_cols:
            row[col] = pd.to_numeric(chunk[col], errors="coerce").sum(min_count=1)
        for col in stock_cols:
            vals = pd.to_numeric(chunk[col], errors="coerce")
            row[col] = vals.iloc[-1] if vals.notna().any() else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("period_end").sort_index()
    out["capex_outflow"] = pd.to_numeric(out.get("capex_outflow"), errors="coerce").where(
        pd.to_numeric(out.get("capex_outflow"), errors="coerce").notna(),
        pd.to_numeric(out.get("capex_raw"), errors="coerce").abs(),
    )
    out["fcf"] = pd.to_numeric(out.get("cfo"), errors="coerce") - pd.to_numeric(out.get("capex_outflow"), errors="coerce")
    out["ccr"] = np.where(
        pd.to_numeric(out.get("net_income"), errors="coerce") > 0,
        pd.to_numeric(out.get("cfo"), errors="coerce") / pd.to_numeric(out.get("net_income"), errors="coerce").replace(0, np.nan),
        np.nan,
    )
    return out.replace([np.inf, -np.inf], np.nan)


def _to_basis_cashflow_frame(quarter_frame: pd.DataFrame, basis: str) -> tuple[pd.DataFrame, int]:
    if quarter_frame.empty:
        return pd.DataFrame(), 1

    flow_cols = [
        c
        for c in [
            "cfo",
            "cfi",
            "cff",
            "capex_raw",
            "capex_outflow",
            "fcf",
            "operating_income",
            "net_income",
            "revenue",
        ]
        if c in quarter_frame.columns
    ]
    if basis == "annual":
        annual = _annualize_cashflow_frame(quarter_frame)
        return annual, 1
    if basis == "ttm":
        out = _drop_leading_fy_row(quarter_frame).copy()
        for col in flow_cols:
            if col in out.columns:
                out[col] = _rolling_ttm(pd.to_numeric(out[col], errors="coerce"))
        out["capex_outflow"] = pd.to_numeric(out.get("capex_outflow"), errors="coerce").where(
            pd.to_numeric(out.get("capex_outflow"), errors="coerce").notna(),
            pd.to_numeric(out.get("capex_raw"), errors="coerce").abs(),
        )
        out["fcf"] = pd.to_numeric(out.get("cfo"), errors="coerce") - pd.to_numeric(out.get("capex_outflow"), errors="coerce")
        out["ccr"] = np.where(
            pd.to_numeric(out.get("net_income"), errors="coerce") > 0,
            pd.to_numeric(out.get("cfo"), errors="coerce") / pd.to_numeric(out.get("net_income"), errors="coerce").replace(0, np.nan),
            np.nan,
        )
        return out.replace([np.inf, -np.inf], np.nan), 4
    return _drop_leading_fy_row(quarter_frame).copy(), 4


def get_financials_cashflow_payload(
    ticker: str,
    market: str = "auto",
    window: str = "10y",
    basis: str = "ttm",
    *,
    preloaded_data: _AnalysisPreloadedData | None = None,
) -> dict[str, Any]:
    tkr = str(ticker).strip().upper()
    window_norm = _normalize_window_spec(window)
    basis_norm = str(basis or "ttm").strip().lower()
    if basis_norm not in {"ttm", "quarter", "annual"}:
        basis_norm = "ttm"

    preloaded = preloaded_data or _build_preloaded_analysis_data(ticker=tkr, market=market)
    tkr = preloaded.ticker
    resolved_market = preloaded.resolved_market
    asof_ts = preloaded.asof_ts
    quarter = _preloaded_base_quarter_frame(preloaded, "cashflow")
    basis_frame, growth_lag = _to_basis_cashflow_frame(quarter, basis_norm)
    basis_frame = _apply_income_window(basis_frame, window_norm, basis_norm).copy()
    basis_frame = _drop_leading_annual_spike(basis_frame, basis_norm)
    derived_basis = _preloaded_derived_frame(preloaded, basis_norm)
    basis_frame = _overlay_derived_columns(
        basis_frame,
        derived_basis,
        columns=["fcf", "ccr"],
    )

    chart_ids = [
        "cf_cashflow_overview",
        "cf_cashflow_vs_earnings",
        "cf_cash_conversion",
        "cf_capex_vs_cashflow",
        "cf_capex_vs_performance",
    ]
    if basis_frame.empty:
        empty_chart = _income_chart_payload(
            frame=pd.DataFrame(index=pd.DatetimeIndex([])),
            labels=[],
            title="데이터 없음",
            unit_left="",
            unit_right="",
            notes="해당 티커의 현금흐름 PIT 데이터가 없습니다.",
            series_defs=[{"key": "cfo", "name": "영업현금흐름(CFO)", "type": "line", "yAxis": "left"}],
        )
        return {
            "ticker": tkr,
            "window": window_norm,
            "basis": basis_norm,
            "periods": [],
            "table": {"kpis": []},
            "charts": {cid: empty_chart for cid in chart_ids},
        }

    labels = _period_labels(basis_frame, basis_norm)
    change_lag = growth_lag
    cfo_chg = _pct_change_series(
        basis_frame["cfo"],
        change_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    capex_chg = _pct_change_series(
        basis_frame["capex_outflow"],
        change_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    fcf_chg = _pct_change_series(
        basis_frame["fcf"],
        change_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    kpi_rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        idx = basis_frame.index[i]
        kpi_rows.append(
            {
                "period": label,
                "cfo": _safe_num(basis_frame.at[idx, "cfo"]),
                "cfo_chg_pct": _safe_num(cfo_chg.iloc[i]),
                "capex": _safe_num(basis_frame.at[idx, "capex_outflow"]),
                "capex_chg_pct": _safe_num(capex_chg.iloc[i]),
                "fcf": _safe_num(basis_frame.at[idx, "fcf"]),
                "fcf_chg_pct": _safe_num(fcf_chg.iloc[i]),
            }
        )

    _ccy = "KRW" if resolved_market == "kr" else "USD"
    chart_defs: dict[str, dict[str, Any]] = {
        "cf_cashflow_overview": {
            "title": "현금흐름",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": "FCF = CFO - CAPEX_outflow, CAPEX displayed as outflow positive",
            "series": [
                {"key": "fcf", "name": "FCF", "type": "bar", "yAxis": "left"},
                {"key": "cfo", "name": "영업현금흐름(CFO)", "type": "line", "yAxis": "left"},
                {"key": "cfi", "name": "투자현금흐름(CFI)", "type": "line", "yAxis": "left"},
                {"key": "cff", "name": "재무현금흐름(CFF)", "type": "line", "yAxis": "left"},
            ],
        },
        "cf_cashflow_vs_earnings": {
            "title": "현금흐름과 실적",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": "CFO vs OperatingIncome vs NetIncome",
            "series": [
                {"key": "cfo", "name": "영업현금흐름(CFO)", "type": "line", "yAxis": "left"},
                {"key": "operating_income", "name": "영업이익", "type": "line", "yAxis": "left"},
                {"key": "net_income", "name": "순이익", "type": "line", "yAxis": "left"},
            ],
        },
        "cf_cash_conversion": {
            "title": "현금전환비율",
            "unit_left": _ccy,
            "unit_right": "x",
            "notes": "CCR = CFO / NetIncome, NetIncome<=0은 CCR null",
            "series": [
                {"key": "cfo", "name": "영업현금흐름(CFO)", "type": "line", "yAxis": "left"},
                {"key": "net_income", "name": "순이익", "type": "line", "yAxis": "left"},
                {"key": "ccr", "name": "CCR", "type": "line", "yAxis": "right"},
            ],
        },
        "cf_capex_vs_cashflow": {
            "title": "CAPEX vs 현금흐름",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "CAPEX displayed as outflow positive, stacked for magnitude comparison",
            "series": [
                {"key": "capex_outflow", "name": "CAPEX(지출)", "type": "stackedBar", "yAxis": "left"},
                {"key": "fcf", "name": "FCF", "type": "stackedBar", "yAxis": "left"},
                {"key": "cfo", "name": "영업현금흐름(CFO)", "type": "line", "yAxis": "right"},
            ],
        },
        "cf_capex_vs_performance": {
            "title": "CAPEX vs 실적",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "MarketCap = price_at_period_end * shares",
            "series": [
                {"key": "capex_outflow", "name": "CAPEX(지출)", "type": "bar", "yAxis": "left"},
                {"key": "operating_income", "name": "영업이익", "type": "line", "yAxis": "right"},
                {"key": "revenue", "name": "매출액", "type": "line", "yAxis": "right"},
                {"key": "market_cap", "name": "시가총액", "type": "line", "yAxis": "right"},
            ],
        },
    }

    charts = {
        chart_id: _income_chart_payload(
            frame=basis_frame,
            labels=labels,
            title=cfg["title"],
            unit_left=cfg["unit_left"],
            unit_right=cfg["unit_right"],
            notes=f"{cfg['notes']} | CAPEX_outflow=abs(CAPEX_raw), FCF=CFO-CAPEX_outflow, basis={basis_norm}, window={window_norm}",
            series_defs=cfg["series"],
        )
        for chart_id, cfg in chart_defs.items()
    }

    has_non_positive_ni = (pd.to_numeric(basis_frame.get("net_income"), errors="coerce") <= 0).fillna(False).any()
    if has_non_positive_ni:
        curr = charts["cf_cash_conversion"].get("missing_reason")
        if curr:
            charts["cf_cash_conversion"]["missing_reason"] = f"{curr}; NetIncome<=0 구간은 CCR null 처리"
        else:
            charts["cf_cash_conversion"]["missing_reason"] = "NetIncome<=0 구간은 CCR null 처리"

    return {
        "ticker": tkr,
        "window": window_norm,
        "basis": basis_norm,
        "periods": labels,
        "table": {"kpis": kpi_rows},
        "charts": charts,
    }


def _build_fundamentals_base_quarter_frame(q_pit: pd.DataFrame, px: pd.DataFrame, ticker: str = "", market: str = "kr") -> pd.DataFrame:
    if q_pit.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=pd.to_datetime(q_pit["PeriodEnd"], errors="coerce"))
    out.index.name = "period_end"
    out = out.loc[~out.index.isna()].sort_index()

    revenue = _pick_first_nonnull_col(q_pit, "Revenue", "Sales")
    cogs = _pick_first_nonnull_col(q_pit, "COGS", "Cost Of Revenue")
    sga = _pick_first_nonnull_col(q_pit, "SG&A", "Selling General And Administrative")
    r_and_d = _pick_first_nonnull_col(q_pit, "R&D", "Research And Development")
    gross_profit = _pick_first_nonnull_col(q_pit, "Gross Profit", "GrossProfit")
    revenue = _fill_revenue_from_components(revenue, cogs, gross_profit)
    gross_profit = gross_profit.where(gross_profit.notna(), revenue - cogs)
    operating_income = _pick_first_nonnull_col(q_pit, "Operating Income", "OperatingIncome")
    net_income = _pick_first_nonnull_col(q_pit, "Net Income", "NetIncome")
    assets = _pick_first_nonnull_col(q_pit, "Total Assets", "Assets")
    liabilities = _pick_first_nonnull_col(q_pit, "Total Liabilities", "Liabilities")
    equity = _pick_first_nonnull_col(q_pit, "Shareholders Equity", "Stockholders Equity", "Equity")
    current_assets = _pick_first_nonnull_col(q_pit, "Current Assets", "CurrentAssets")
    current_liabilities = _pick_first_nonnull_col(q_pit, "Current Liabilities", "CurrentLiabilities")
    ar = _pick_first_nonnull_col(q_pit, "Accounts Receivable", "AR", "Receivables")
    inventory = _pick_first_nonnull_col(q_pit, "Inventory")
    ap = _pick_first_nonnull_col(q_pit, "Accounts Payable", "AP")
    cfo = _pick_first_nonnull_col(q_pit, "Operating Cash Flow", "OperatingCashFlow", "CFO")
    capex_raw = _pick_first_nonnull_col(
        q_pit,
        "Capital Expenditure",
        "Capital Expenditures",
        "CapitalExpenditures",
        "CAPEX",
    )
    pre_tax_income = _pick_first_nonnull_col(q_pit, "Pretax Income", "Pre Tax Income")
    tax_expense = _pick_first_nonnull_col(q_pit, "Tax", "Tax Expense", "Income Tax Expense")
    debt_short = _pick_first_nonnull_col(q_pit, "Debt Short", "Short Term Debt")
    debt_long = _pick_first_nonnull_col(q_pit, "Debt Long", "Long Term Debt")

    owner_equity = _pick_first_nonnull_col(
        q_pit,
        "Equity Attributable To Owners",
        "Equity Attributable To Parent",
        "Owner Equity",
        "Owners Equity",
    )
    owner_net_income = _pick_first_nonnull_col(
        q_pit,
        "Net Income Attributable To Owners",
        "Net Income Attributable To Parent",
        "Owner Net Income",
    )

    shares = _pick_first_nonnull_col(q_pit, "Shares", "Diluted Shares", "Basic Shares").replace(0, np.nan)
    eps_raw = _pick_first_nonnull_col(q_pit, "EPS", "Diluted EPS")
    close_col = _pick_close_col(px)
    if close_col is not None and not px.empty:
        price = pd.to_numeric(px[close_col], errors="coerce").sort_index().reindex(out.index, method="ffill")
    else:
        price = pd.Series(np.nan, index=out.index, dtype=float)

    capex_outflow = capex_raw.abs()
    fcf = cfo - capex_outflow
    pre_tax_income, tax_expense = _fill_pretax_and_tax(pre_tax_income, tax_expense, net_income)

    out["revenue"] = revenue
    out["cogs"] = cogs
    out["sga"] = sga
    out["r_and_d"] = r_and_d
    out["gross_profit"] = gross_profit
    out["operating_income"] = operating_income
    out["net_income"] = net_income
    out["assets"] = assets
    out["liabilities"] = liabilities
    out["equity"] = equity
    out["current_assets"] = current_assets
    out["current_liabilities"] = current_liabilities
    out["ar"] = ar
    out["inventory"] = inventory
    out["ap"] = ap
    out["cfo"] = cfo
    out["capex_raw"] = capex_raw
    out["capex_outflow"] = capex_outflow
    out["fcf"] = fcf
    out["pre_tax_income"] = pre_tax_income
    out["tax_expense"] = tax_expense
    out["debt_short"] = debt_short
    out["debt_long"] = debt_long
    out["debt_total"] = debt_short.fillna(0.0) + debt_long.fillna(0.0)
    out["owner_equity"] = owner_equity
    out["owner_net_income"] = owner_net_income
    out["shares"] = shares
    out["eps_raw"] = eps_raw
    out["price"] = price
    out["market_cap"] = _resolve_market_cap(ticker, market, out.index)

    out = _attach_fiscal_metadata(out, q_pit)
    return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _derive_sga_split(
    gross_profit: pd.Series,
    sga: pd.Series,
    r_and_d: pd.Series,
    operating_income: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return (r_and_d_display, sga_ex_r_and_d) using an accounting fit check.

    If `Gross Profit - SG&A - R&D` matches `Operating Income` better than
    `Gross Profit - SG&A`, we treat SG&A and R&D as separately reported and keep
    SG&A as-is. Only when the embedded interpretation fits better do we subtract
    R&D from SG&A.
    """
    gross_profit = pd.to_numeric(gross_profit, errors="coerce")
    sga = pd.to_numeric(sga, errors="coerce")
    r_and_d = pd.to_numeric(r_and_d, errors="coerce")
    operating_income = pd.to_numeric(operating_income, errors="coerce")

    both_valid = sga.notna() & r_and_d.notna()
    can_subtract = both_valid & ((sga - r_and_d) >= -1e-9)
    identity_valid = gross_profit.notna() & sga.notna() & r_and_d.notna() & operating_income.notna()

    sep_resid = (gross_profit - sga - r_and_d - operating_income).abs()
    emb_resid = (gross_profit - sga - operating_income).abs()
    embedded_rows = identity_valid & can_subtract & (emb_resid + 1e-9 < sep_resid)

    if not bool(identity_valid.any()):
        n_valid = int(both_valid.sum())
        embedded_series = n_valid > 0 and int(can_subtract.sum()) / n_valid >= 0.7
        embedded_rows = can_subtract if embedded_series else pd.Series(False, index=sga.index)

    r_and_d_display = r_and_d
    sga_ex_r_and_d = sga.where(~embedded_rows, sga - r_and_d)
    return r_and_d_display, sga_ex_r_and_d


def _annualize_fundamentals_frame(quarter_frame: pd.DataFrame) -> pd.DataFrame:
    if quarter_frame.empty:
        return pd.DataFrame()

    flow_cols = [
        c
        for c in [
            "revenue",
            "cogs",
            "sga",
            "r_and_d",
            "gross_profit",
            "operating_income",
            "net_income",
            "owner_net_income",
            "cfo",
            "capex_raw",
            "capex_outflow",
            "fcf",
            "pre_tax_income",
            "tax_expense",
        ]
        if c in quarter_frame.columns
    ]
    excluded = set(flow_cols) | {"fiscal_year", "fiscal_quarter", "fiscal_label"}
    stock_cols = [c for c in quarter_frame.columns if c not in excluded]

    rows: list[dict[str, Any]] = []
    fiscal_year = _period_year_series(quarter_frame)
    years = sorted({int(v) for v in fiscal_year.dropna().unique().tolist()})
    for year in years:
        chunk = quarter_frame.loc[fiscal_year == year]
        if chunk.empty:
            continue
        row: dict[str, Any] = {
            "period_end": pd.Timestamp(chunk.index.max()).normalize(),
            "fiscal_year": year,
            "fiscal_quarter": 4,
            "fiscal_label": str(year),
        }
        for col in flow_cols:
            row[col] = pd.to_numeric(chunk[col], errors="coerce").sum(min_count=1)
        for col in stock_cols:
            vals = pd.to_numeric(chunk[col], errors="coerce")
            row[col] = vals.iloc[-1] if vals.notna().any() else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("period_end").sort_index()
    out["capex_outflow"] = pd.to_numeric(out.get("capex_outflow"), errors="coerce").where(
        pd.to_numeric(out.get("capex_outflow"), errors="coerce").notna(),
        pd.to_numeric(out.get("capex_raw"), errors="coerce").abs(),
    )
    out["fcf"] = pd.to_numeric(out.get("cfo"), errors="coerce") - pd.to_numeric(out.get("capex_outflow"), errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _to_basis_fundamentals_frame(quarter_frame: pd.DataFrame, basis: str) -> tuple[pd.DataFrame, int]:
    if quarter_frame.empty:
        return pd.DataFrame(), 4

    flow_cols = [
        c
        for c in [
            "revenue",
            "cogs",
            "sga",
            "r_and_d",
            "gross_profit",
            "operating_income",
            "net_income",
            "owner_net_income",
            "cfo",
            "capex_raw",
            "capex_outflow",
            "fcf",
            "pre_tax_income",
            "tax_expense",
        ]
        if c in quarter_frame.columns
    ]
    if basis == "annual":
        return _annualize_fundamentals_frame(quarter_frame), 1
    if basis == "ttm":
        out = _drop_leading_fy_row(quarter_frame).copy()
        for col in flow_cols:
            if col in out.columns:
                out[col] = _rolling_ttm(pd.to_numeric(out[col], errors="coerce"))
        out["capex_outflow"] = pd.to_numeric(out.get("capex_outflow"), errors="coerce").where(
            pd.to_numeric(out.get("capex_outflow"), errors="coerce").notna(),
            pd.to_numeric(out.get("capex_raw"), errors="coerce").abs(),
        )
        out["fcf"] = pd.to_numeric(out.get("cfo"), errors="coerce") - pd.to_numeric(out.get("capex_outflow"), errors="coerce")
        return out.replace([np.inf, -np.inf], np.nan), 4
    return _drop_leading_fy_row(quarter_frame).copy(), 4


def _project_fund_efficiency(
    quarter_frame: pd.DataFrame,
    basis_index: pd.Index,
    basis: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    if quarter_frame.empty:
        empty = pd.Series(np.nan, index=basis_index, dtype=float)
        return empty, empty, empty, empty, empty

    revenue = pd.to_numeric(quarter_frame.get("revenue"), errors="coerce")
    cogs = pd.to_numeric(quarter_frame.get("cogs"), errors="coerce")
    ar = pd.to_numeric(quarter_frame.get("ar"), errors="coerce")
    inventory = pd.to_numeric(quarter_frame.get("inventory"), errors="coerce")
    ap = pd.to_numeric(quarter_frame.get("ap"), errors="coerce")

    revenue_ttm = revenue.rolling(window=4, min_periods=4).sum()
    cogs_ttm = cogs.rolling(window=4, min_periods=4).sum()
    avg_ar = (ar + ar.shift(1)) / 2.0
    avg_inv = (inventory + inventory.shift(1)) / 2.0
    avg_ap = (ap + ap.shift(1)) / 2.0
    ar_turn = revenue_ttm / avg_ar.replace(0, np.nan)
    inv_turn = cogs_ttm / avg_inv.replace(0, np.nan)
    ap_turn = cogs_ttm / avg_ap.replace(0, np.nan)

    dso = 365.0 / ar_turn.replace(0, np.nan)
    dio = 365.0 / inv_turn.replace(0, np.nan)
    dpo = 365.0 / ap_turn.replace(0, np.nan)
    operating_cycle = dso + dio
    cash_cycle = operating_cycle - dpo

    if basis == "annual":
        def _annual_last(series: pd.Series) -> pd.Series:
            temp = pd.DataFrame({"v": series})
            temp["year"] = _period_year_series(quarter_frame).reindex(temp.index).to_numpy()
            rows: list[tuple[pd.Timestamp, float | None]] = []
            for year, grp in temp.groupby("year", sort=True):
                if pd.isna(year):
                    continue
                vals = pd.to_numeric(grp["v"], errors="coerce").dropna()
                rows.append((pd.Timestamp(grp.index.max()).normalize(), (float(vals.iloc[-1]) if not vals.empty else np.nan)))
            if not rows:
                return pd.Series(np.nan, index=basis_index, dtype=float)
            out = pd.Series({k: v for k, v in rows}, dtype=float).sort_index()
            return out.reindex(pd.DatetimeIndex(basis_index))

        return (
            _annual_last(ar_turn),
            _annual_last(inv_turn),
            _annual_last(ap_turn),
            _annual_last(operating_cycle),
            _annual_last(cash_cycle),
        )

    q_idx = pd.DatetimeIndex(quarter_frame.index)
    b_idx = pd.DatetimeIndex(basis_index)
    return (
        pd.Series(ar_turn.to_numpy(), index=q_idx).reindex(b_idx),
        pd.Series(inv_turn.to_numpy(), index=q_idx).reindex(b_idx),
        pd.Series(ap_turn.to_numpy(), index=q_idx).reindex(b_idx),
        pd.Series(operating_cycle.to_numpy(), index=q_idx).reindex(b_idx),
        pd.Series(cash_cycle.to_numpy(), index=q_idx).reindex(b_idx),
    )


def _build_fundamentals_table_rows(
    frame: pd.DataFrame,
    labels: list[str],
    metrics: list[tuple[str, str, str]],
    change_lag: int,
) -> list[dict[str, float | str | None]]:
    rows: list[dict[str, float | str | None]] = []
    for key, title, unit in metrics:
        row: dict[str, float | str | None] = {"metric": title, "unit": unit}
        if key in frame.columns:
            vals = pd.to_numeric(frame[key], errors="coerce")
        else:
            vals = pd.Series(np.nan, index=frame.index, dtype=float)
        chg = _pct_change_series(
            vals,
            change_lag,
            fiscal_year=frame.get("fiscal_year"),
            fiscal_quarter=frame.get("fiscal_quarter"),
        )
        for i, label in enumerate(labels):
            row[label] = _safe_num(vals.iloc[i])
            row[f"{label}_chg"] = _safe_num(chg.iloc[i])
        rows.append(row)
    return rows


def _fund_scatter_chart_payload(
    *,
    frame: pd.DataFrame,
    labels: list[str],
    title: str,
    unit_left: str,
    unit_right: str,
    notes: str,
    x_key: str,
    y_key: str,
    series_name: str,
) -> dict[str, Any]:
    x_vals = pd.to_numeric(frame.get(x_key), errors="coerce") if x_key in frame.columns else pd.Series(np.nan, index=frame.index, dtype=float)
    y_vals = pd.to_numeric(frame.get(y_key), errors="coerce") if y_key in frame.columns else pd.Series(np.nan, index=frame.index, dtype=float)
    points: list[dict[str, Any]] = []
    for label, x, y in zip(labels, x_vals, y_vals, strict=False):
        xv = _safe_num(x)
        yv = _safe_num(y)
        if xv is None or yv is None:
            continue
        points.append({"x": xv, "y": yv, "label": label})

    missing_reason = None
    if not points:
        missing_reason = f"데이터 없음: {x_key}/{y_key}"

    return {
        "meta": {
            "title": title,
            "unit_left": unit_left,
            "unit_right": unit_right,
            "notes": notes,
        },
        "series": [
            {
                "name": series_name,
                "type": "scatter",
                "yAxis": "left",
                "dashed": False,
                "data": points,
            }
        ],
        "missing_reason": missing_reason,
    }


def get_fundamentals_payload(
    ticker: str,
    market: str = "auto",
    window: str = "10y",
    basis: str = "ttm",
    *,
    preloaded_data: _AnalysisPreloadedData | None = None,
) -> dict[str, Any]:
    tkr = str(ticker).strip().upper()
    window_norm = _normalize_window_spec(window)
    basis_norm = str(basis or "ttm").strip().lower()
    if basis_norm not in {"ttm", "quarter", "annual"}:
        basis_norm = "ttm"

    preloaded = preloaded_data or _build_preloaded_analysis_data(ticker=tkr, market=market)
    tkr = preloaded.ticker
    resolved_market = preloaded.resolved_market
    asof_ts = preloaded.asof_ts
    quarter = _preloaded_base_quarter_frame(preloaded, "fundamentals")
    basis_frame, growth_lag = _to_basis_fundamentals_frame(quarter, basis_norm)
    basis_frame = _apply_income_window(basis_frame, window_norm, basis_norm).copy()
    basis_frame = _drop_leading_annual_spike(basis_frame, basis_norm)
    chart_ids = [
        "fn_dupont",
        "fn_roe_pbr_scatter",
        "fn_capital_efficiency",
        "fn_owner_roe",
        "fn_margins_combo",
        "fn_cost_ratios_combo",
        "fn_revenue_growth",
        "fn_profit_growth",
        "fn_eps_growth_per_scatter",
        "fn_cost_growth",
        "fn_capital_growth",
        "fn_debt_ratio",
        "fn_current_ratio",
        "fn_turnover_rates",
        "fn_operating_cycle",
        "fn_cash_cycle",
        "fn_cash_conversion_ratio",
    ]

    empty_tables = {
        "profitability": {"rows": []},
        "growth": {"rows": []},
        "stability": {"rows": []},
        "efficiency": {"rows": []},
    }
    if basis_frame.empty:
        empty_chart = _income_chart_payload(
            frame=pd.DataFrame(index=pd.DatetimeIndex([])),
            labels=[],
            title="데이터 없음",
            unit_left="",
            unit_right="",
            notes="해당 티커의 펀더멘탈 PIT 데이터가 없습니다.",
            series_defs=[{"key": "roe", "name": "ROE", "type": "line", "yAxis": "left"}],
        )
        return {
            "ticker": tkr,
            "window": window_norm,
            "basis": basis_norm,
            "periods": [],
            "tables": empty_tables,
            "charts": {cid: empty_chart for cid in chart_ids},
        }

    labels = _period_labels(basis_frame, basis_norm)
    change_lag = growth_lag

    basis_frame["capex_outflow"] = pd.to_numeric(basis_frame.get("capex_outflow"), errors="coerce").where(
        pd.to_numeric(basis_frame.get("capex_outflow"), errors="coerce").notna(),
        pd.to_numeric(basis_frame.get("capex_raw"), errors="coerce").abs(),
    )
    basis_frame["fcf"] = pd.to_numeric(basis_frame.get("cfo"), errors="coerce") - pd.to_numeric(basis_frame.get("capex_outflow"), errors="coerce")

    avg_assets = (pd.to_numeric(basis_frame.get("assets"), errors="coerce") + pd.to_numeric(basis_frame.get("assets"), errors="coerce").shift(1)) / 2.0
    avg_equity = (pd.to_numeric(basis_frame.get("equity"), errors="coerce") + pd.to_numeric(basis_frame.get("equity"), errors="coerce").shift(1)) / 2.0
    avg_owner_equity = (pd.to_numeric(basis_frame.get("owner_equity"), errors="coerce") + pd.to_numeric(basis_frame.get("owner_equity"), errors="coerce").shift(1)) / 2.0
    avg_debt = (pd.to_numeric(basis_frame.get("debt_total"), errors="coerce") + pd.to_numeric(basis_frame.get("debt_total"), errors="coerce").shift(1)) / 2.0

    revenue = pd.to_numeric(basis_frame.get("revenue"), errors="coerce")
    cogs = pd.to_numeric(basis_frame.get("cogs"), errors="coerce")
    sga = pd.to_numeric(basis_frame.get("sga"), errors="coerce")
    gross_profit = pd.to_numeric(basis_frame.get("gross_profit"), errors="coerce")
    operating_income = pd.to_numeric(basis_frame.get("operating_income"), errors="coerce")
    net_income = pd.to_numeric(basis_frame.get("net_income"), errors="coerce")
    owner_net_income = pd.to_numeric(basis_frame.get("owner_net_income"), errors="coerce")
    equity = pd.to_numeric(basis_frame.get("equity"), errors="coerce")
    liabilities = pd.to_numeric(basis_frame.get("liabilities"), errors="coerce")
    current_assets = pd.to_numeric(basis_frame.get("current_assets"), errors="coerce")
    current_liabilities = pd.to_numeric(basis_frame.get("current_liabilities"), errors="coerce")
    cfo = pd.to_numeric(basis_frame.get("cfo"), errors="coerce")
    shares = pd.to_numeric(basis_frame.get("shares"), errors="coerce").replace(0, np.nan)
    market_cap = pd.to_numeric(basis_frame.get("market_cap"), errors="coerce")
    price = pd.to_numeric(basis_frame.get("price"), errors="coerce")
    # KR fallback: derive shares from market_cap / price when Shares tag is missing
    if shares.isna().all() and market_cap.notna().any() and price.notna().any():
        shares = (market_cap / price.replace(0, np.nan)).replace(0, np.nan)
    pre_tax_income = pd.to_numeric(basis_frame.get("pre_tax_income"), errors="coerce")
    tax_expense = pd.to_numeric(basis_frame.get("tax_expense"), errors="coerce")

    basis_frame["gross_margin"] = gross_profit / revenue.replace(0, np.nan) * 100.0
    basis_frame["op_margin"] = operating_income / revenue.replace(0, np.nan) * 100.0
    basis_frame["net_margin"] = net_income / revenue.replace(0, np.nan) * 100.0
    basis_frame["cogs_ratio"] = cogs / revenue.replace(0, np.nan) * 100.0
    basis_frame["sga_ratio"] = sga / revenue.replace(0, np.nan) * 100.0
    basis_frame["total_cost_ratio"] = (cogs + sga) / revenue.replace(0, np.nan) * 100.0
    basis_frame["roe"] = net_income / avg_equity.replace(0, np.nan) * 100.0
    basis_frame["roa"] = net_income / avg_assets.replace(0, np.nan) * 100.0
    basis_frame["asset_turnover"] = revenue / avg_assets.replace(0, np.nan)
    basis_frame["leverage"] = avg_assets / avg_equity.replace(0, np.nan)
    basis_frame["gpa"] = gross_profit / avg_assets.replace(0, np.nan) * 100.0
    basis_frame["debt_ratio"] = liabilities / equity.replace(0, np.nan) * 100.0
    basis_frame["current_ratio"] = current_assets / current_liabilities.replace(0, np.nan) * 100.0
    basis_frame["ccr"] = np.where(net_income > 0, cfo / net_income.replace(0, np.nan), np.nan)

    tax_rate = np.where(pre_tax_income > 0, tax_expense / pre_tax_income.replace(0, np.nan), np.nan)
    invested_capital = avg_equity + avg_debt
    basis_frame["roic"] = (operating_income * (1.0 - pd.to_numeric(tax_rate, errors="coerce"))) / invested_capital.replace(0, np.nan) * 100.0

    basis_frame["owner_roe"] = owner_net_income / avg_owner_equity.replace(0, np.nan) * 100.0
    owner_supported = owner_net_income.notna().any() and pd.to_numeric(basis_frame.get("owner_equity"), errors="coerce").notna().any()

    basis_frame["eps"] = _basis_eps_series(basis_frame, basis=basis_norm, net_income=net_income, shares=shares)
    basis_frame["bps"] = _positive_div(equity, shares)
    basis_frame["sps"] = _positive_div(revenue, shares)
    basis_frame["ops"] = _positive_div(operating_income, shares)
    basis_frame["oofps"] = _positive_div(cfo, shares)
    basis_frame["fcfps"] = _positive_div(pd.to_numeric(basis_frame.get("fcf"), errors="coerce"), shares)

    eps_for_per = pd.to_numeric(basis_frame["eps"], errors="coerce")
    if basis_norm == "quarter":
        eps_for_per = _quarter_ttm_eps_series(quarter, basis_frame.index).combine_first(eps_for_per)
    basis_frame["eps_for_per"] = eps_for_per
    basis_frame["per"] = _positive_div(price, basis_frame["eps_for_per"])
    basis_frame["pbr"] = _positive_div(price, basis_frame["bps"])
    basis_frame["psr"] = _positive_div(price, basis_frame["sps"])
    basis_frame["por"] = _positive_div(price, basis_frame["ops"])
    basis_frame["pfcfr"] = _positive_div(price, basis_frame["fcfps"])
    basis_frame["eps_growth"] = _pct_change_series(
        basis_frame["eps_for_per"],
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    basis_frame["peg"] = _positive_div(
        pd.to_numeric(basis_frame["per"], errors="coerce"),
        basis_frame["eps_growth"].where(pd.to_numeric(basis_frame["eps_growth"], errors="coerce") > 0, np.nan),
    )
    basis_frame["price_return"] = _pct_change_series(
        price,
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    basis_frame["revenue_growth"] = _pct_change_series(
        revenue,
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    basis_frame["gross_profit_growth"] = _pct_change_series(
        gross_profit,
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    basis_frame["operating_income_growth"] = _pct_change_series(
        operating_income,
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    basis_frame["net_income_growth"] = _pct_change_series(
        net_income,
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    basis_frame["cogs_growth"] = _pct_change_series(
        cogs,
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    basis_frame["sga_growth"] = _pct_change_series(
        sga,
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )

    ar_turn, inv_turn, ap_turn, operating_cycle, cash_cycle = _project_fund_efficiency(quarter, basis_frame.index, basis_norm)
    basis_frame["ar_turnover"] = ar_turn
    basis_frame["inventory_turnover"] = inv_turn
    basis_frame["ap_turnover"] = ap_turn
    basis_frame["dso"] = 365.0 / ar_turn.replace(0, np.nan)
    basis_frame["dio"] = 365.0 / inv_turn.replace(0, np.nan)
    basis_frame["dpo"] = 365.0 / ap_turn.replace(0, np.nan)
    basis_frame["operating_cycle"] = operating_cycle
    basis_frame["cash_cycle"] = cash_cycle
    derived_basis = _preloaded_derived_frame(preloaded, basis_norm)
    basis_frame = _overlay_derived_columns(
        basis_frame,
        derived_basis,
        columns=[
            "roe",
            "roa",
            "roic",
            "gpa",
            "asset_turnover",
            "leverage",
            "debt_ratio",
            "current_ratio",
            "gross_margin",
            "op_margin",
            "net_margin",
            "cogs_ratio",
            "sga_ratio",
            "total_cost_ratio",
            "revenue_growth",
            "gross_profit_growth",
            "operating_income_growth",
            "net_income_growth",
            "cogs_growth",
            "sga_growth",
            "price_return",
            "ar_turnover",
            "inventory_turnover",
            "ap_turnover",
            "dso",
            "dio",
            "dpo",
            "operating_cycle",
            "cash_cycle",
            "ccr",
            "fcf",
        ],
    )
    sga_total = pd.to_numeric(basis_frame.get("sga"), errors="coerce")
    r_and_d = pd.to_numeric(basis_frame.get("r_and_d"), errors="coerce")
    basis_frame["r_and_d_display"], basis_frame["sga_ex_r_and_d"] = _derive_sga_split(
        gross_profit=pd.to_numeric(basis_frame.get("gross_profit"), errors="coerce"),
        sga=sga_total,
        r_and_d=r_and_d,
        operating_income=pd.to_numeric(basis_frame.get("operating_income"), errors="coerce"),
    )
    basis_frame["sga_ex_r_and_d_ratio"] = basis_frame["sga_ex_r_and_d"] / revenue.replace(0, np.nan) * 100.0
    basis_frame["r_and_d_ratio"] = basis_frame["r_and_d_display"] / revenue.replace(0, np.nan) * 100.0
    basis_frame["sga_ex_r_and_d_growth"] = _pct_change_series(
        basis_frame["sga_ex_r_and_d"],
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    basis_frame["r_and_d_growth"] = _pct_change_series(
        basis_frame["r_and_d_display"],
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )

    tables = {
        "profitability": {
            "rows": _build_fundamentals_table_rows(
                basis_frame,
                labels,
                [("roe", "ROE", "%"), ("op_margin", "영업이익률", "%"), ("net_margin", "순이익률", "%")],
                change_lag,
            )
        },
        "growth": {
            "rows": _build_fundamentals_table_rows(
                basis_frame,
                labels,
                [("revenue", "매출액", "USD"), ("operating_income", "영업이익", "USD"), ("net_income", "순이익", "USD"), ("equity", "자본", "USD")],
                change_lag,
            )
        },
        "stability": {
            "rows": _build_fundamentals_table_rows(
                basis_frame,
                labels,
                [("debt_ratio", "부채비율", "%"), ("current_ratio", "유동비율", "%")],
                change_lag,
            )
        },
        "efficiency": {
            "rows": _build_fundamentals_table_rows(
                basis_frame,
                labels,
                [("dso", "매출채권회전일수(DSO)", "days"), ("dio", "재고자산회전일수(DIO)", "days"), ("dpo", "매입채무회전일수(DPO)", "days")],
                change_lag,
            )
        },
    }

    _ccy = "KRW" if resolved_market == "kr" else "USD"
    chart_defs: dict[str, dict[str, Any]] = {
        "fn_dupont": {
            "title": "ROE 듀퐁분석",
            "unit_left": "%",
            "unit_right": "%/x",
            "notes": "ROE=NetIncome/AverageEquity, 순이익률=NetIncome/Revenue, 자산회전율=Revenue/AverageAssets, 레버리지=AverageAssets/AverageEquity",
            "series": [
                {"key": "roe", "name": "ROE", "type": "bar", "yAxis": "left"},
                {"key": "net_margin", "name": "순이익률", "type": "line", "yAxis": "right"},
                {"key": "asset_turnover", "name": "총자산회전율", "type": "line", "yAxis": "right"},
                {"key": "leverage", "name": "레버리지", "type": "line", "yAxis": "right"},
            ],
        },
        "fn_capital_efficiency": {
            "title": "자본효율성",
            "unit_left": "%",
            "unit_right": "",
            "notes": "ROA/ROE/ROIC/GP/A",
            "series": [
                {"key": "roa", "name": "ROA", "type": "line", "yAxis": "left"},
                {"key": "roe", "name": "ROE", "type": "line", "yAxis": "left"},
                {"key": "roic", "name": "ROIC", "type": "line", "yAxis": "left"},
                {"key": "gpa", "name": "GP/A", "type": "line", "yAxis": "left"},
            ],
        },
        "fn_owner_roe": {
            "title": "지배주주 ROE",
            "unit_left": "%",
            "unit_right": "",
            "notes": "지배주주 순이익/평균 지배주주자본",
            "series": [
                {"key": "owner_roe", "name": "지배주주 ROE", "type": "line", "yAxis": "left"},
                {"key": "roe", "name": "ROE", "type": "line", "yAxis": "left"},
            ],
        },
        "fn_margins_combo": {
            "title": "이익률",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "매출총이익률/영업이익률/순이익률 + 매출액",
            "series": [
                {"key": "revenue", "name": "매출액", "type": "bar", "yAxis": "left"},
                {"key": "gross_margin", "name": "매출총이익률", "type": "line", "yAxis": "right"},
                {"key": "op_margin", "name": "영업이익률", "type": "line", "yAxis": "right"},
                {"key": "net_margin", "name": "순이익률", "type": "line", "yAxis": "right"},
            ],
        },
        "fn_cost_ratios_combo": {
            "title": "비용율",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "매출원가율/판매관리비율/연구개발비율/총비용률 + 매출액",
            "series": [
                {"key": "revenue", "name": "매출액", "type": "bar", "yAxis": "left"},
                {"key": "cogs_ratio", "name": "매출원가율", "type": "line", "yAxis": "right"},
                {"key": "sga_ex_r_and_d_ratio", "name": "판매관리비율", "type": "line", "yAxis": "right"},
                {"key": "r_and_d_ratio", "name": "연구개발비율", "type": "line", "yAxis": "right"},
                {"key": "total_cost_ratio", "name": "총비용률", "type": "line", "yAxis": "right"},
            ],
        },
        "fn_revenue_growth": {
            "title": "매출액 성장률",
            "unit_left": "%",
            "unit_right": "%",
            "notes": "YoY 성장률 + 주가수익률",
            "series": [
                {"key": "revenue_growth", "name": "매출액 성장률", "type": "line", "yAxis": "left"},
                {"key": "gross_profit_growth", "name": "매출총이익 성장률", "type": "line", "yAxis": "left"},
                {"key": "price_return", "name": "주가수익률", "type": "line", "yAxis": "right"},
            ],
        },
        "fn_profit_growth": {
            "title": "이익성장률",
            "unit_left": "%",
            "unit_right": "",
            "notes": "매출액/매출총이익/영업이익/순이익 성장률",
            "series": [
                {"key": "revenue_growth", "name": "매출액 성장률", "type": "line", "yAxis": "left"},
                {"key": "gross_profit_growth", "name": "매출총이익 성장률", "type": "line", "yAxis": "left"},
                {"key": "operating_income_growth", "name": "영업이익 성장률", "type": "line", "yAxis": "left"},
                {"key": "net_income_growth", "name": "순이익 성장률", "type": "line", "yAxis": "left"},
            ],
        },
        "fn_cost_growth": {
            "title": "비용증가율",
            "unit_left": "%",
            "unit_right": "",
            "notes": "매출성장률/매출원가증가율/판매관리비증가율/연구개발비증가율",
            "series": [
                {"key": "revenue_growth", "name": "매출성장률", "type": "line", "yAxis": "left"},
                {"key": "cogs_growth", "name": "매출원가 증가율", "type": "line", "yAxis": "left"},
                {"key": "sga_ex_r_and_d_growth", "name": "판매관리비 증가율", "type": "line", "yAxis": "left"},
                {"key": "r_and_d_growth", "name": "연구개발비 증가율", "type": "line", "yAxis": "left"},
            ],
        },
        "fn_capital_growth": {
            "title": "자본성장률",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": "자산/부채/자본/지배주주자본",
            "series": [
                {"key": "assets", "name": "자산총계", "type": "line", "yAxis": "left"},
                {"key": "liabilities", "name": "부채총계", "type": "line", "yAxis": "left"},
                {"key": "equity", "name": "자본총계", "type": "line", "yAxis": "left"},
                {"key": "owner_equity", "name": "지배주주 자본총계", "type": "line", "yAxis": "left"},
            ],
        },
        "fn_debt_ratio": {
            "title": "부채비율",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "부채비율 = Liabilities / Equity * 100",
            "series": [
                {"key": "equity", "name": "자본총계", "type": "stackedBar", "yAxis": "left"},
                {"key": "liabilities", "name": "부채총계", "type": "stackedBar", "yAxis": "left"},
                {"key": "debt_ratio", "name": "부채비율", "type": "line", "yAxis": "right"},
            ],
        },
        "fn_current_ratio": {
            "title": "유동비율",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "유동비율 = CurrentAssets / CurrentLiabilities * 100",
            "series": [
                {"key": "current_assets", "name": "유동자산", "type": "stackedBar", "yAxis": "left"},
                {"key": "current_liabilities", "name": "유동부채", "type": "stackedBar", "yAxis": "left"},
                {"key": "current_ratio", "name": "유동비율", "type": "line", "yAxis": "right"},
            ],
        },
        "fn_turnover_rates": {
            "title": "회전율",
            "unit_left": "x",
            "unit_right": "",
            "notes": "매출채권/재고/매입채무 회전율",
            "series": [
                {"key": "ar_turnover", "name": "매출채권회전율", "type": "line", "yAxis": "left"},
                {"key": "inventory_turnover", "name": "재고자산회전율", "type": "line", "yAxis": "left"},
                {"key": "ap_turnover", "name": "매입채무회전율", "type": "line", "yAxis": "left"},
            ],
        },
        "fn_operating_cycle": {
            "title": "영업순환주기",
            "unit_left": "days",
            "unit_right": "days",
            "notes": "영업순환주기 = DSO + DIO",
            "series": [
                {"key": "operating_cycle", "name": "영업순환주기", "type": "bar", "yAxis": "left"},
                {"key": "dso", "name": "DSO", "type": "line", "yAxis": "right"},
                {"key": "dio", "name": "DIO", "type": "line", "yAxis": "right"},
            ],
        },
        "fn_cash_cycle": {
            "title": "현금회전일수",
            "unit_left": "days",
            "unit_right": "days",
            "notes": "현금회전일수 = DSO + DIO - DPO",
            "series": [
                {"key": "cash_cycle", "name": "현금회전일수", "type": "bar", "yAxis": "left"},
                {"key": "dpo", "name": "DPO", "type": "line", "yAxis": "right"},
                {"key": "operating_cycle", "name": "영업순환주기", "type": "line", "yAxis": "right"},
            ],
        },
        "fn_cash_conversion_ratio": {
            "title": "현금전환비율",
            "unit_left": _ccy,
            "unit_right": "x",
            "notes": "CCR = CFO / NetIncome (NetIncome<=0은 null)",
            "series": [
                {"key": "cfo", "name": "영업현금흐름(CFO)", "type": "line", "yAxis": "left"},
                {"key": "net_income", "name": "순이익", "type": "line", "yAxis": "left"},
                {"key": "ccr", "name": "CCR", "type": "line", "yAxis": "right"},
            ],
        },
    }

    charts = {
        chart_id: _income_chart_payload(
            frame=basis_frame,
            labels=labels,
            title=cfg["title"],
            unit_left=cfg["unit_left"],
            unit_right=cfg["unit_right"],
            notes=f"{cfg['notes']} | basis={basis_norm}, window={window_norm}",
            series_defs=cfg["series"],
        )
        for chart_id, cfg in chart_defs.items()
    }

    charts["fn_roe_pbr_scatter"] = _fund_scatter_chart_payload(
        frame=basis_frame,
        labels=labels,
        title="ROE-PBR",
        unit_left="ROE(%)",
        unit_right="PBR(x)",
        notes="x=ROE, y=PBR(MarketCap/Equity)",
        x_key="roe",
        y_key="pbr",
        series_name="ROE-PBR",
    )
    charts["fn_eps_growth_per_scatter"] = _fund_scatter_chart_payload(
        frame=basis_frame,
        labels=labels,
        title="EPS Growth-PER",
        unit_left="EPS Growth(%)",
        unit_right="PER(x)",
        notes="x=EPS Growth(YoY), y=PER(Price/EPS)",
        x_key="eps_growth",
        y_key="per",
        series_name="EPS Growth-PER",
    )

    if not owner_supported:
        base_msg = charts["fn_owner_roe"].get("missing_reason")
        charts["fn_owner_roe"]["missing_reason"] = "지배주주 ROE 미지원(지배주주 순이익/자본 데이터 부족)" if not base_msg else f"{base_msg}; 지배주주 ROE 미지원"

    if pd.to_numeric(tax_expense, errors="coerce").dropna().empty or pd.to_numeric(pre_tax_income, errors="coerce").dropna().empty:
        base_msg = charts["fn_capital_efficiency"].get("missing_reason")
        roic_msg = "ROIC 미지원(세전이익/법인세 데이터 부족)"
        charts["fn_capital_efficiency"]["missing_reason"] = roic_msg if not base_msg else f"{base_msg}; {roic_msg}"

    non_positive_ni = (pd.to_numeric(net_income, errors="coerce") <= 0).fillna(False).any()
    if non_positive_ni:
        base_msg = charts["fn_cash_conversion_ratio"].get("missing_reason")
        ni_msg = "NetIncome<=0 구간은 CCR null 처리"
        charts["fn_cash_conversion_ratio"]["missing_reason"] = ni_msg if not base_msg else f"{base_msg}; {ni_msg}"

    return {
        "ticker": tkr,
        "window": window_norm,
        "basis": basis_norm,
        "periods": labels,
        "tables": tables,
        "charts": charts,
    }


def _to_basis_valuation_frame(quarter_frame: pd.DataFrame, basis: str) -> tuple[pd.DataFrame, int]:
    if quarter_frame.empty:
        return pd.DataFrame(), 4

    flow_cols = [
        c
        for c in [
            "revenue",
            "cogs",
            "sga",
            "gross_profit",
            "operating_income",
            "net_income",
            "owner_net_income",
            "cfo",
            "capex_raw",
            "capex_outflow",
            "fcf",
            "pre_tax_income",
            "tax_expense",
        ]
        if c in quarter_frame.columns
    ]
    if basis == "annual":
        return _annualize_fundamentals_frame(quarter_frame), 1
    if basis == "ttm":
        out = _drop_leading_fy_row(quarter_frame).copy()
        for col in flow_cols:
            if col in out.columns:
                out[col] = _rolling_ttm(pd.to_numeric(out[col], errors="coerce"))
        out["capex_outflow"] = pd.to_numeric(out.get("capex_outflow"), errors="coerce").where(
            pd.to_numeric(out.get("capex_outflow"), errors="coerce").notna(),
            pd.to_numeric(out.get("capex_raw"), errors="coerce").abs(),
        )
        out["fcf"] = pd.to_numeric(out.get("cfo"), errors="coerce") - pd.to_numeric(out.get("capex_outflow"), errors="coerce")
        return out.replace([np.inf, -np.inf], np.nan), 4
    return _drop_leading_fy_row(quarter_frame).copy(), 4


def _positive_div(num: pd.Series, den: pd.Series) -> pd.Series:
    d = pd.to_numeric(den, errors="coerce")
    return pd.to_numeric(num, errors="coerce") / d.where(d > 0, np.nan)


def _basis_eps_series(
    frame: pd.DataFrame,
    *,
    basis: str,
    net_income: pd.Series,
    shares: pd.Series,
) -> pd.Series:
    eps_calc = _positive_div(net_income, shares)
    if basis != "quarter":
        return eps_calc
    eps_raw = pd.to_numeric(frame.get("eps_raw"), errors="coerce")
    return eps_raw.where(eps_raw.notna(), eps_calc)


def _quarter_ttm_eps_series(quarter_frame: pd.DataFrame, target_index: pd.Index) -> pd.Series:
    target = pd.DatetimeIndex(target_index)
    if quarter_frame is None or quarter_frame.empty:
        return pd.Series(np.nan, index=target, dtype=float)

    q_net_income = pd.to_numeric(quarter_frame.get("net_income"), errors="coerce").rolling(window=4, min_periods=4).sum()
    q_shares = pd.to_numeric(quarter_frame.get("shares"), errors="coerce").replace(0, np.nan)
    q_eps_ttm = _positive_div(q_net_income, q_shares)
    return pd.Series(q_eps_ttm.to_numpy(), index=pd.DatetimeIndex(quarter_frame.index)).reindex(target)


def _quantile_band_values(series: pd.Series, quantiles: list[float]) -> dict[float, float]:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    s = s[s > 0]
    if s.empty:
        return {}
    out: dict[float, float] = {}
    for q in quantiles:
        try:
            out[q] = float(s.quantile(q))
        except Exception:
            continue
    return out


def _build_valuation_table_rows(
    basis_frame: pd.DataFrame,
    *,
    band_qs: list[float],
    pbr_q: dict[float, float],
    psr_q: dict[float, float],
    per_q: dict[float, float],
    por_q: dict[float, float],
    pfcfr_q: dict[float, float],
) -> dict[str, Any]:
    latest = basis_frame.iloc[-1] if not basis_frame.empty else pd.Series(dtype=float)
    q_labels = [f"q{int(q * 100)}" for q in band_qs]
    band_rows: list[dict[str, Any]] = []
    for name, qmap in [("PBR", pbr_q), ("PSR", psr_q), ("PER", per_q), ("POR", por_q), ("PFCFR", pfcfr_q)]:
        row: dict[str, Any] = {"metric": name}
        for q, lbl in zip(band_qs, q_labels, strict=False):
            row[lbl] = _safe_num(qmap.get(q))
        band_rows.append(row)

    value_rows = [
        {"metric": "PBR", "latest": _safe_num(latest.get("pbr")), "median_q50": _safe_num(pbr_q.get(0.5))},
        {"metric": "PSR", "latest": _safe_num(latest.get("psr")), "median_q50": _safe_num(psr_q.get(0.5))},
        {"metric": "PER", "latest": _safe_num(latest.get("per")), "median_q50": _safe_num(per_q.get(0.5))},
        {"metric": "POR", "latest": _safe_num(latest.get("por")), "median_q50": _safe_num(por_q.get(0.5))},
        {"metric": "PEG", "latest": _safe_num(latest.get("peg")), "median_q50": None},
    ]

    per_share_rows = [
        {"metric": "BPS", "latest": _safe_num(latest.get("bps"))},
        {"metric": "SPS", "latest": _safe_num(latest.get("sps"))},
        {"metric": "OPS", "latest": _safe_num(latest.get("ops"))},
        {"metric": "EPS", "latest": _safe_num(latest.get("eps"))},
        {"metric": "OOFPS", "latest": _safe_num(latest.get("oofps"))},
        {"metric": "FCFPS", "latest": _safe_num(latest.get("fcfps"))},
    ]

    return {
        "band": {"rows": band_rows},
        "value": {"rows": value_rows},
        "per_share": {"rows": per_share_rows},
    }


def _valuation_chart_defs(
    currency: str,
    quantiles: list[float],
    reference_levels: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    pbr_refs = [float(v) for v in reference_levels.get("pbr", VALUATION_TTM_REFERENCE_LEVELS["pbr"])]
    psr_refs = [float(v) for v in reference_levels.get("psr", VALUATION_TTM_REFERENCE_LEVELS["psr"])]
    per_refs = [float(v) for v in reference_levels.get("per", VALUATION_TTM_REFERENCE_LEVELS["per"])]
    por_refs = [float(v) for v in reference_levels.get("por", VALUATION_TTM_REFERENCE_LEVELS["por"])]
    return {
        "val_band_pbr": {
            "title": "PBR 밴드",
            "unit_left": currency,
            "unit_right": currency,
            "notes": f"bandPrice=BPS*quantile, quantiles={quantiles}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "bps", "name": "BPS", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"pbr_band_q{int(q * 100)}", "name": f"PBR Q{int(q * 100)}", "type": "line", "yAxis": "left", "dashed": True}
                    for q in quantiles
                ],
            ],
        },
        "val_band_psr": {
            "title": "PSR 밴드",
            "unit_left": currency,
            "unit_right": currency,
            "notes": f"bandPrice=SPS*quantile, quantiles={quantiles}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "sps", "name": "SPS", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"psr_band_q{int(q * 100)}", "name": f"PSR Q{int(q * 100)}", "type": "line", "yAxis": "left", "dashed": True}
                    for q in quantiles
                ],
            ],
        },
        "val_band_per": {
            "title": "PER 밴드",
            "unit_left": currency,
            "unit_right": "",
            "notes": f"bandPrice=EPS*quantile, quantiles={quantiles}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                *[
                    {"key": f"per_band_q{int(q * 100)}", "name": f"PER Q{int(q * 100)}", "type": "line", "yAxis": "left", "dashed": True}
                    for q in quantiles
                ],
            ],
        },
        "val_band_por": {
            "title": "POR 밴드",
            "unit_left": currency,
            "unit_right": "",
            "notes": f"bandPrice=OPS*quantile, quantiles={quantiles}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                *[
                    {"key": f"por_band_q{int(q * 100)}", "name": f"POR Q{int(q * 100)}", "type": "line", "yAxis": "left", "dashed": True}
                    for q in quantiles
                ],
            ],
        },
        "val_band_pfcfr": {
            "title": "PFCFR 밴드",
            "unit_left": currency,
            "unit_right": "",
            "notes": f"bandPrice=FCFPS*quantile, quantiles={quantiles}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                *[
                    {"key": f"pfcfr_band_q{int(q * 100)}", "name": f"PFCFR Q{int(q * 100)}", "type": "line", "yAxis": "left", "dashed": True}
                    for q in quantiles
                ],
            ],
        },
        "val_multiple_pbr": {
            "title": "PBR",
            "unit_left": currency,
            "unit_right": "x",
            "notes": f"기준선={pbr_refs}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "pbr", "name": "PBR", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"pbr_ref_{str(ref).replace('.', '_')}", "name": f"기준 {ref:g}x", "type": "line", "yAxis": "right", "dashed": True}
                    for ref in pbr_refs
                ],
            ],
        },
        "val_multiple_psr": {
            "title": "PSR",
            "unit_left": currency,
            "unit_right": "x",
            "notes": f"기준선={psr_refs}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "psr", "name": "PSR", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"psr_ref_{str(ref).replace('.', '_')}", "name": f"기준 {ref:g}x", "type": "line", "yAxis": "right", "dashed": True}
                    for ref in psr_refs
                ],
            ],
        },
        "val_multiple_per": {
            "title": "PER",
            "unit_left": currency,
            "unit_right": "x",
            "notes": f"기준선={per_refs}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "per", "name": "PER", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"per_ref_{str(ref).replace('.', '_')}", "name": f"기준 {ref:g}x", "type": "line", "yAxis": "right", "dashed": True}
                    for ref in per_refs
                ],
            ],
        },
        "val_multiple_por": {
            "title": "POR",
            "unit_left": currency,
            "unit_right": "x",
            "notes": f"기준선={por_refs}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "por", "name": "POR", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"por_ref_{str(ref).replace('.', '_')}", "name": f"기준 {ref:g}x", "type": "line", "yAxis": "right", "dashed": True}
                    for ref in por_refs
                ],
            ],
        },
        "val_peg": {
            "title": "PEG",
            "unit_left": "x",
            "unit_right": "x / %",
            "notes": "PEG=PER/EPS_growth_yoy, EPS_growth<=0은 null",
            "series": [
                {"key": "peg", "name": "PEG", "type": "bar", "yAxis": "left"},
                {"key": "per", "name": "PER", "type": "line", "yAxis": "right"},
                {"key": "eps_growth", "name": "EPS 성장률", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_bps": {
            "title": "BPS",
            "unit_left": currency,
            "unit_right": currency,
            "notes": "BPS=Equity/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "bps", "name": "BPS", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_sps": {
            "title": "SPS",
            "unit_left": currency,
            "unit_right": currency,
            "notes": "SPS=Revenue/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "sps", "name": "SPS", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_ops": {
            "title": "OPS",
            "unit_left": currency,
            "unit_right": currency,
            "notes": "OPS=OperatingIncome/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "ops", "name": "OPS", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_eps": {
            "title": "EPS",
            "unit_left": currency,
            "unit_right": currency,
            "notes": "EPS=분기엔 SEC EPS 우선, TTM/연간은 NetIncome/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "eps", "name": "EPS", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_oofps": {
            "title": "OOFPS",
            "unit_left": currency,
            "unit_right": currency,
            "notes": "OOFPS=CFO/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "oofps", "name": "OOFPS", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_fcfps": {
            "title": "FCFPS",
            "unit_left": currency,
            "unit_right": currency,
            "notes": "FCFPS=(CFO-abs(CAPEX))/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "fcfps", "name": "FCFPS", "type": "line", "yAxis": "right"},
            ],
        },
    }


def _resolve_market_only(ticker: str, market: str) -> str:
    market_norm = str(market or "auto").strip().lower()
    if market_norm and market_norm != "auto":
        return market_norm
    return "kr" if is_kr_ticker(ticker) else "us"


def _build_precomputed_valuation_payload(
    ticker: str,
    market: str,
    window: str,
) -> dict[str, Any] | None:
    if window not in {"5y", "10y"}:
        return None

    resolved_market = _resolve_market_only(ticker, market)
    payload = _json_svc.load_valuation_ttm_data(ticker, resolved_market)
    if not isinstance(payload, dict):
        return None
    if str(payload.get("basis", "")).strip().lower() != "ttm":
        return None

    built = build_valuation_frame_from_precomputed(payload, window)
    if built is None:
        return None

    basis_frame, labels, tables, reference_levels = built
    if basis_frame.empty or not labels:
        return None

    quantiles = list(VALUATION_TTM_QUANTILES)
    currency = "KRW" if resolved_market == "kr" else "USD"
    chart_defs = _valuation_chart_defs(currency, quantiles, reference_levels)
    charts = {
        chart_id: _income_chart_payload(
            frame=basis_frame,
            labels=labels,
            title=cfg["title"],
            unit_left=cfg["unit_left"],
            unit_right=cfg["unit_right"],
            notes=(f"{cfg['notes']} | precomputed_ttm_json=true, window={window}"),
            series_defs=cfg["series"],
        )
        for chart_id, cfg in chart_defs.items()
    }

    for key in ["per", "por", "pfcfr", "peg"]:
        non_positive = (pd.to_numeric(basis_frame.get(key), errors="coerce") <= 0).fillna(False).any()
        if non_positive:
            target = "val_multiple_per" if key == "per" else "val_multiple_por" if key == "por" else "val_band_pfcfr" if key == "pfcfr" else "val_peg"
            msg = charts[target].get("missing_reason")
            add = f"{key.upper()} 일부 구간 null(분모<=0)"
            charts[target]["missing_reason"] = add if not msg else f"{msg}; {add}"

    return {
        "ticker": str(payload.get("ticker", ticker)).strip().upper(),
        "window": window,
        "basis": "ttm",
        "periods": labels,
        "tables": tables,
        "charts": charts,
    }


def get_valuation_payload(
    ticker: str,
    market: str = "auto",
    window: str = "10y",
    basis: str = "ttm",
    *,
    preloaded_data: _AnalysisPreloadedData | None = None,
) -> dict[str, Any]:
    tkr = str(ticker).strip().upper()
    window_norm = _normalize_window_spec(window)
    basis_norm = str(basis or "ttm").strip().lower()
    if basis_norm not in {"ttm", "quarter", "annual"}:
        basis_norm = "ttm"

    if basis_norm == "ttm":
        precomputed_payload = _build_precomputed_valuation_payload(
            ticker=tkr,
            market=market,
            window=window_norm,
        )
        if precomputed_payload is not None:
            return precomputed_payload

    preloaded = preloaded_data or _build_preloaded_analysis_data(ticker=tkr, market=market)
    tkr = preloaded.ticker
    resolved_market = preloaded.resolved_market
    asof_ts = preloaded.asof_ts
    quarter = _preloaded_base_quarter_frame(preloaded, "fundamentals")
    basis_frame, growth_lag = _to_basis_valuation_frame(quarter, basis_norm)
    basis_frame = _apply_income_window(basis_frame, window_norm, basis_norm).copy()
    basis_frame = _drop_leading_annual_spike(basis_frame, basis_norm)
    derived_basis = _preloaded_derived_frame(preloaded, basis_norm)

    chart_ids = [
        "val_band_pbr",
        "val_band_psr",
        "val_band_per",
        "val_band_por",
        "val_band_pfcfr",
        "val_multiple_pbr",
        "val_multiple_psr",
        "val_multiple_per",
        "val_multiple_por",
        "val_peg",
        "val_ps_bps",
        "val_ps_sps",
        "val_ps_ops",
        "val_ps_eps",
        "val_ps_oofps",
        "val_ps_fcfps",
    ]

    empty_tables = {"band": {"rows": []}, "value": {"rows": []}, "per_share": {"rows": []}}
    if basis_frame.empty:
        empty_chart = _income_chart_payload(
            frame=pd.DataFrame(index=pd.DatetimeIndex([])),
            labels=[],
            title="데이터 없음",
            unit_left="",
            unit_right="",
            notes="해당 티커의 밸류에이션 PIT 데이터가 없습니다.",
            series_defs=[{"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"}],
        )
        return {
            "ticker": tkr,
            "window": window_norm,
            "basis": basis_norm,
            "periods": [],
            "tables": empty_tables,
            "charts": {cid: empty_chart for cid in chart_ids},
        }

    labels = _period_labels(basis_frame, basis_norm)
    shares = pd.to_numeric(basis_frame.get("shares"), errors="coerce").replace(0, np.nan)
    revenue = pd.to_numeric(basis_frame.get("revenue"), errors="coerce")
    op_income = pd.to_numeric(basis_frame.get("operating_income"), errors="coerce")
    net_income = pd.to_numeric(basis_frame.get("net_income"), errors="coerce")
    equity = pd.to_numeric(basis_frame.get("equity"), errors="coerce")
    cfo = pd.to_numeric(basis_frame.get("cfo"), errors="coerce")
    capex_raw = pd.to_numeric(basis_frame.get("capex_raw"), errors="coerce")
    capex_outflow = pd.to_numeric(basis_frame.get("capex_outflow"), errors="coerce")
    capex_outflow = capex_outflow.where(capex_outflow.notna(), capex_raw.abs())
    fcf = cfo - capex_outflow
    price = pd.to_numeric(basis_frame.get("price"), errors="coerce")
    market_cap = pd.to_numeric(basis_frame.get("market_cap"), errors="coerce")
    # KR fallback: derive shares from market_cap / price when Shares tag is missing
    if shares.isna().all() and market_cap.notna().any() and price.notna().any():
        shares = (market_cap / price.replace(0, np.nan)).replace(0, np.nan)
    if market_cap.dropna().empty and price.notna().any():
        market_cap = price * shares

    eps = _basis_eps_series(basis_frame, basis=basis_norm, net_income=net_income, shares=shares)
    eps_for_per = eps
    if basis_norm == "quarter":
        eps_for_per = _quarter_ttm_eps_series(quarter, basis_frame.index).combine_first(pd.to_numeric(eps, errors="coerce"))
    bps = _positive_div(equity, shares)
    sps = _positive_div(revenue, shares)
    ops = _positive_div(op_income, shares)
    oofps = _positive_div(cfo, shares)
    fcfps = _positive_div(fcf, shares)

    basis_frame["price"] = price
    basis_frame["market_cap"] = market_cap
    basis_frame["bps"] = bps
    basis_frame["sps"] = sps
    basis_frame["ops"] = ops
    basis_frame["eps"] = eps
    basis_frame["eps_for_per"] = eps_for_per
    basis_frame["oofps"] = oofps
    basis_frame["fcfps"] = fcfps
    basis_frame["capex_outflow"] = capex_outflow
    basis_frame["fcf"] = fcf

    basis_frame["pbr"] = _positive_div(price, bps)
    basis_frame["psr"] = _positive_div(price, sps)
    basis_frame["per"] = _positive_div(price, eps_for_per)
    basis_frame["por"] = _positive_div(price, ops)
    basis_frame["pfcfr"] = _positive_div(price, fcfps)

    eps_growth = _pct_change_series(
        eps_for_per,
        growth_lag,
        fiscal_year=basis_frame.get("fiscal_year"),
        fiscal_quarter=basis_frame.get("fiscal_quarter"),
    )
    peg = pd.to_numeric(basis_frame["per"], errors="coerce") / eps_growth.where(eps_growth > 0, np.nan)
    basis_frame["eps_growth"] = eps_growth
    basis_frame["peg"] = peg

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    pbr_q = _quantile_band_values(basis_frame["pbr"], quantiles)
    psr_q = _quantile_band_values(basis_frame["psr"], quantiles)
    per_q = _quantile_band_values(basis_frame["per"], quantiles)
    por_q = _quantile_band_values(basis_frame["por"], quantiles)
    pfcfr_q = _quantile_band_values(basis_frame["pfcfr"], quantiles)

    band_metrics = [
        ("pbr", bps.where(bps > 0, np.nan), pbr_q),
        ("psr", sps.where(sps > 0, np.nan), psr_q),
        ("per", eps.where(eps > 0, np.nan), per_q),
        ("por", ops.where(ops > 0, np.nan), por_q),
        ("pfcfr", fcfps.where(fcfps > 0, np.nan), pfcfr_q),
    ]
    for name, base_series, qmap in band_metrics:
        for q, qv in qmap.items():
            qlabel = int(q * 100)
            basis_frame[f"{name}_band_q{qlabel}"] = base_series * float(qv)

    pbr_refs = [0.5, 1.0, 2.0, 3.0, 4.0]
    psr_refs = [0.5, 1.0, 2.0, 3.0, 4.0]
    per_refs = [5.0, 10.0, 15.0, 20.0, 25.0]
    por_refs = [5.0, 10.0, 15.0, 20.0]
    for ref in pbr_refs:
        basis_frame[f"pbr_ref_{str(ref).replace('.', '_')}"] = ref
    for ref in psr_refs:
        basis_frame[f"psr_ref_{str(ref).replace('.', '_')}"] = ref
    for ref in per_refs:
        basis_frame[f"per_ref_{str(ref).replace('.', '_')}"] = ref
    for ref in por_refs:
        basis_frame[f"por_ref_{str(ref).replace('.', '_')}"] = ref

    _ccy = "KRW" if resolved_market == "kr" else "USD"
    chart_defs: dict[str, dict[str, Any]] = {
        "val_band_pbr": {
            "title": "PBR 밴드",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": f"bandPrice=BPS*quantile, quantiles={quantiles}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "bps", "name": "BPS", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"pbr_band_q{int(q * 100)}", "name": f"PBR Q{int(q * 100)}", "type": "line", "yAxis": "left", "dashed": True}
                    for q in quantiles
                ],
            ],
        },
        "val_band_psr": {
            "title": "PSR 밴드",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": f"bandPrice=SPS*quantile, quantiles={quantiles}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "sps", "name": "SPS", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"psr_band_q{int(q * 100)}", "name": f"PSR Q{int(q * 100)}", "type": "line", "yAxis": "left", "dashed": True}
                    for q in quantiles
                ],
            ],
        },
        "val_band_per": {
            "title": "PER 밴드",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": f"bandPrice=EPS*quantile, quantiles={quantiles}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                *[
                    {"key": f"per_band_q{int(q * 100)}", "name": f"PER Q{int(q * 100)}", "type": "line", "yAxis": "left", "dashed": True}
                    for q in quantiles
                ],
            ],
        },
        "val_band_por": {
            "title": "POR 밴드",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": f"bandPrice=OPS*quantile, quantiles={quantiles}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                *[
                    {"key": f"por_band_q{int(q * 100)}", "name": f"POR Q{int(q * 100)}", "type": "line", "yAxis": "left", "dashed": True}
                    for q in quantiles
                ],
            ],
        },
        "val_band_pfcfr": {
            "title": "PFCFR 밴드",
            "unit_left": _ccy,
            "unit_right": "",
            "notes": f"bandPrice=FCFPS*quantile, quantiles={quantiles}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                *[
                    {"key": f"pfcfr_band_q{int(q * 100)}", "name": f"PFCFR Q{int(q * 100)}", "type": "line", "yAxis": "left", "dashed": True}
                    for q in quantiles
                ],
            ],
        },
        "val_multiple_pbr": {
            "title": "PBR",
            "unit_left": _ccy,
            "unit_right": "x",
            "notes": f"기준선={pbr_refs}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "pbr", "name": "PBR", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"pbr_ref_{str(ref).replace('.', '_')}", "name": f"기준 {ref:g}x", "type": "line", "yAxis": "right", "dashed": True}
                    for ref in pbr_refs
                ],
            ],
        },
        "val_multiple_psr": {
            "title": "PSR",
            "unit_left": _ccy,
            "unit_right": "x",
            "notes": f"기준선={psr_refs}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "psr", "name": "PSR", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"psr_ref_{str(ref).replace('.', '_')}", "name": f"기준 {ref:g}x", "type": "line", "yAxis": "right", "dashed": True}
                    for ref in psr_refs
                ],
            ],
        },
        "val_multiple_per": {
            "title": "PER",
            "unit_left": _ccy,
            "unit_right": "x",
            "notes": f"기준선={per_refs}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "per", "name": "PER", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"per_ref_{str(ref).replace('.', '_')}", "name": f"기준 {ref:g}x", "type": "line", "yAxis": "right", "dashed": True}
                    for ref in per_refs
                ],
            ],
        },
        "val_multiple_por": {
            "title": "POR",
            "unit_left": _ccy,
            "unit_right": "x",
            "notes": f"기준선={por_refs}",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "por", "name": "POR", "type": "line", "yAxis": "right"},
                *[
                    {"key": f"por_ref_{str(ref).replace('.', '_')}", "name": f"기준 {ref:g}x", "type": "line", "yAxis": "right", "dashed": True}
                    for ref in por_refs
                ],
            ],
        },
        "val_peg": {
            "title": "PEG",
            "unit_left": "x",
            "unit_right": "x / %",
            "notes": "PEG=PER/EPS_growth_yoy, EPS_growth<=0은 null",
            "series": [
                {"key": "peg", "name": "PEG", "type": "bar", "yAxis": "left"},
                {"key": "per", "name": "PER", "type": "line", "yAxis": "right"},
                {"key": "eps_growth", "name": "EPS 성장률", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_bps": {
            "title": "BPS",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "BPS=Equity/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "bps", "name": "BPS", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_sps": {
            "title": "SPS",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "SPS=Revenue/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "sps", "name": "SPS", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_ops": {
            "title": "OPS",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "OPS=OperatingIncome/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "ops", "name": "OPS", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_eps": {
            "title": "EPS",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "EPS=분기는 SEC EPS 우선, TTM/연간은 NetIncome/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "eps", "name": "EPS", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_oofps": {
            "title": "OOFPS",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "OOFPS=CFO/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "oofps", "name": "OOFPS", "type": "line", "yAxis": "right"},
            ],
        },
        "val_ps_fcfps": {
            "title": "FCFPS",
            "unit_left": _ccy,
            "unit_right": _ccy,
            "notes": "FCFPS=(CFO-abs(CAPEX))/SharesUsed",
            "series": [
                {"key": "price", "name": "수정주가", "type": "line", "yAxis": "left"},
                {"key": "fcfps", "name": "FCFPS", "type": "line", "yAxis": "right"},
            ],
        },
    }

    charts = {
        chart_id: _income_chart_payload(
            frame=basis_frame,
            labels=labels,
            title=cfg["title"],
            unit_left=cfg["unit_left"],
            unit_right=cfg["unit_right"],
            notes=(
                f"{cfg['notes']} | quantile_mode=positive_only, quantiles={quantiles}, "
                f"shares_source=Shares->Diluted/Basic fallback, basis={basis_norm}, window={window_norm}"
            ),
            series_defs=cfg["series"],
        )
        for chart_id, cfg in chart_defs.items()
    }

    for key in ["per", "por", "pfcfr", "peg"]:
        non_positive = (pd.to_numeric(basis_frame.get(key), errors="coerce") <= 0).fillna(False).any()
        if non_positive:
            target = "val_multiple_per" if key == "per" else "val_multiple_por" if key == "por" else "val_band_pfcfr" if key == "pfcfr" else "val_peg"
            msg = charts[target].get("missing_reason")
            add = f"{key.upper()} 일부 구간 null(분모<=0)"
            charts[target]["missing_reason"] = add if not msg else f"{msg}; {add}"

    tables = _build_valuation_table_rows(
        basis_frame,
        band_qs=quantiles,
        pbr_q=pbr_q,
        psr_q=psr_q,
        per_q=per_q,
        por_q=por_q,
        pfcfr_q=pfcfr_q,
    )

    return {
        "ticker": tkr,
        "window": window_norm,
        "basis": basis_norm,
        "periods": labels,
        "tables": tables,
        "charts": charts,
    }

def get_insights_payload(
    ticker: str,
    market: str = "auto",
    window: str = "10y",
    basis: str = "ttm",
) -> dict[str, Any]:
    def _aligned_q_metric(*names: str) -> pd.Series:
        if q_pit.empty:
            return pd.Series(np.nan, index=basis_frame.index, dtype=float)
        series = _pick_first_nonnull_col(q_pit, *names)
        series = pd.to_numeric(series, errors="coerce")
        period_index = pd.to_datetime(q_pit.get("PeriodEnd"), errors="coerce")
        aligned = pd.DataFrame({"period_end": period_index.to_numpy(), "value": series.to_numpy()}).dropna(subset=["period_end"])
        if aligned.empty:
            return pd.Series(np.nan, index=basis_frame.index, dtype=float)
        aligned = aligned.groupby("period_end", as_index=True)["value"].last()
        return aligned.reindex(basis_frame.index)

    def _append_chart_reason(chart: dict[str, Any], reason: str | None) -> None:
        text = str(reason or "").strip()
        if not text:
            return
        current = str(chart.get("missing_reason") or "").strip()
        if not current:
            chart["missing_reason"] = text
            return
        existing = {part.strip() for part in current.split(";") if part.strip()}
        if text not in existing:
            chart["missing_reason"] = f"{current}; {text}"

    def _load_pit_filings_frame() -> pd.DataFrame:
        raw = None
        if not _HAS_DUCKDB:
            return pd.DataFrame()
        try:
            raw = load_filings_from_db(tkr, resolved_market)
        except Exception:
            raw = None
        if raw is None or raw.empty:
            return pd.DataFrame()
        filings = raw.copy()
        for col in ("period_end", "report_date", "available_date", "filing_date", "accepted_at"):
            if col in filings.columns:
                filings[col] = _to_naive_series(filings[col])
        calc_date = _to_naive_series(filings.get("accepted_at")).dt.normalize()
        calc_date = calc_date.combine_first(_to_naive_series(filings.get("filing_date")))
        calc_date = calc_date.combine_first(_to_naive_series(filings.get("available_date")))
        filings = filings.loc[calc_date <= _to_naive_normalized_ts(asof_ts)].copy()
        if filings.empty:
            return filings
        return filings.sort_values(["period_end", "filing_date", "accepted_at"], na_position="last").reset_index(drop=True)

    tkr = str(ticker).strip().upper()
    window_norm = _normalize_window_spec(window)
    basis_norm = str(basis or "ttm").strip().lower()
    if basis_norm not in {"ttm", "quarter", "annual"}:
        basis_norm = "ttm"

    resolved_market, px = _resolve_market_and_price(tkr, market)
    asof_ts = _parse_asof(None)
    q_pit = _build_pit_quarterly_frame(tkr, resolved_market, asof_ts)
    derived_basis = _load_pit_derived_frame(tkr, resolved_market, asof_ts, basis=basis_norm)
    filings_pit = _load_pit_filings_frame()

    base = _build_fundamentals_base_quarter_frame(q_pit, px, tkr, resolved_market)
    basis_frame, _growth_lag = _to_basis_fundamentals_frame(base, basis_norm)
    basis_frame = _apply_income_window(basis_frame, window_norm, basis_norm).copy()
    basis_frame = _drop_leading_annual_spike(basis_frame, basis_norm)
    basis_frame = _overlay_derived_columns(basis_frame, derived_basis)
    basis_frame["dividends_paid"] = _aligned_q_metric("Dividends Paid").abs()
    basis_frame["share_repurchases"] = _aligned_q_metric("Repurchases").abs()
    basis_frame["sbc"] = _aligned_q_metric("SBC")
    basis_frame["shares_outstanding"] = _aligned_q_metric("Shares", "Diluted Shares", "Basic Shares")
    basis_frame["net_income_value"] = pd.to_numeric(basis_frame.get("net_income"), errors="coerce")
    basis_frame["cfo_value"] = pd.to_numeric(basis_frame.get("cfo"), errors="coerce")
    basis_frame["accruals_ratio_pct"] = pd.to_numeric(basis_frame.get("accruals_ratio"), errors="coerce") * 100.0
    _ni_for_ratio = basis_frame["net_income_value"].replace(0, np.nan)
    basis_frame["cfo_to_ni"] = np.where(
        _ni_for_ratio.gt(0),
        basis_frame["cfo_value"] / _ni_for_ratio,
        np.nan,
    )
    basis_frame["revenue_growth_yoy"] = pd.to_numeric(basis_frame.get("revenue_growth"), errors="coerce")
    _lag_series = pd.to_numeric(
        basis_frame["filing_lag_days"] if "filing_lag_days" in basis_frame.columns else pd.Series(np.nan, index=basis_frame.index),
        errors="coerce",
    )
    basis_frame["amendment_flag"] = _lag_series.where(
        pd.Series(basis_frame.get("is_amendment", False), index=basis_frame.index).fillna(False).astype(bool),
        np.nan,
    )
    basis_frame["nt_flag"] = _lag_series.where(
        pd.Series(basis_frame.get("is_nt", False), index=basis_frame.index).fillna(False).astype(bool),
        np.nan,
    )

    chart_ids = [
        "ins_1_shareholder_returns",
        "ins_2_accruals_quality",
        "ins_3_working_capital_vs_growth",
        "ins_4_filing_quality",
    ]

    if basis_frame.empty:
        empty_chart = _income_chart_payload(
            frame=pd.DataFrame(index=pd.DatetimeIndex([])),
            labels=[],
            title="데이터 없음",
            unit_left="",
            unit_right="",
            notes="해당 티커의 데이터가 없습니다.",
            series_defs=[{"key": "empty", "name": "데이터 없음", "type": "line", "yAxis": "left"}],
        )
        empty_chart["missing_reason"] = "데이터 없음"
        return {
            "ticker": tkr,
            "window": window_norm,
            "basis": basis_norm,
            "periods": [],
            "charts": {cid: empty_chart for cid in chart_ids},
        }

    periods = _period_labels(basis_frame, basis_norm)

    _ccy = "KRW" if resolved_market == "kr" else "USD"
    chart_defs = {
        "ins_1_shareholder_returns": {
            "title": "주주환원·희석",
            "unit_left": _ccy,
            "unit_right": "Shares",
            "notes": "배당/자사주매입은 현금유출 기준 절대값, SBC는 비용, 주식수는 분기말 shares 기준",
            "series": [
                {"key": "dividends_paid", "name": "배당지급", "type": "line", "yAxis": "left"},
                {"key": "share_repurchases", "name": "자사주매입", "type": "bar", "yAxis": "left"},
                {"key": "sbc", "name": "주식보상(SBC)", "type": "line", "yAxis": "left"},
                {"key": "shares_outstanding", "name": "유통주식수", "type": "line", "yAxis": "right"},
            ],
        },
        "ins_2_accruals_quality": {
            "title": "발생주의(Accruals) & 현금-이익 괴리",
            "unit_left": _ccy,
            "unit_right": "% / x",
            "notes": "왼축: CFO/NetIncome, 오른축: AccrualsRatio=(NetIncome-CFO)/TotalAssets 및 CFO/NI",
            "series": [
                {"key": "net_income_value", "name": "NetIncome", "type": "line", "yAxis": "left"},
                {"key": "cfo_value", "name": "CFO", "type": "line", "yAxis": "left", "dashed": True},
                {"key": "accruals_ratio_pct", "name": "Accruals Ratio (%)", "type": "line", "yAxis": "right"},
                {"key": "cfo_to_ni", "name": "CFO/NI 비율", "type": "line", "yAxis": "right"},
            ],
        },
        "ins_3_working_capital_vs_growth": {
            "title": "운전자본 변화 vs 매출 성장",
            "unit_left": _ccy,
            "unit_right": "%",
            "notes": "전분기 대비 AR/Inventory/AP 증감과 매출 YoY 성장률을 모두 선형으로 표시",
            "series": [
                {"key": "ar_delta", "name": "ΔAR", "type": "line", "yAxis": "left"},
                {"key": "inv_delta", "name": "ΔInventory", "type": "line", "yAxis": "left"},
                {"key": "ap_delta", "name": "ΔAP", "type": "line", "yAxis": "left"},
                {"key": "revenue_growth_yoy", "name": "Revenue YoY (%)", "type": "line", "yAxis": "right"},
            ],
        },
        "ins_4_filing_quality": {
            "title": "공시(Filing) 품질",
            "unit_left": "Days",
            "unit_right": "Score",
            "notes": "lag=AcceptedAt 우선, 없으면 FilingDate. 정정/NT는 점 마커, punctuality는 최근 4개 분기 기준",
            "series": [
                {"key": "filing_lag_days", "name": "공시 소요일", "type": "bar", "yAxis": "left"},
                {"key": "punctuality_score", "name": "Punctuality Score", "type": "line", "yAxis": "right"},
            ],
        },
    }

    if pd.to_numeric(basis_frame.get("amendment_flag"), errors="coerce").notna().any():
        chart_defs["ins_4_filing_quality"]["series"].append(
            {"key": "amendment_flag", "name": "정정공시", "type": "scatter", "yAxis": "left"}
        )
    if pd.to_numeric(basis_frame.get("nt_flag"), errors="coerce").notna().any():
        chart_defs["ins_4_filing_quality"]["series"].append(
            {"key": "nt_flag", "name": "지연공시(NT)", "type": "scatter", "yAxis": "left"}
        )

    charts: dict[str, Any] = {}
    for cid in chart_ids:
        cdef = chart_defs[cid]
        try:
            charts[cid] = _income_chart_payload(
                frame=basis_frame,
                labels=periods,
                title=cdef["title"],
                unit_left=cdef["unit_left"],
                unit_right=cdef["unit_right"],
                notes=cdef["notes"],
                series_defs=cdef["series"],
            )
        except Exception as e:
            empty = _income_chart_payload(
                frame=pd.DataFrame(index=pd.DatetimeIndex([])),
                labels=[],
                title=cdef["title"],
                unit_left="",
                unit_right="",
                notes=cdef["notes"],
                series_defs=[{"key": "empty", "name": "데이터 없음", "type": "line", "yAxis": "left"}],
            )
            empty["missing_reason"] = f"차트 렌더링 에러: {str(e)}"
            charts[cid] = empty

    if not basis_frame[["dividends_paid", "share_repurchases", "sbc", "shares_outstanding"]].notna().any().any():
        _append_chart_reason(charts["ins_1_shareholder_returns"], "SEC tag 미존재: dividends/share repurchases/SBC/shares")
    else:
        if not basis_frame[["dividends_paid", "share_repurchases"]].notna().any().any():
            _append_chart_reason(charts["ins_1_shareholder_returns"], "주주환원 현금흐름 태그 미존재")
        if not basis_frame["sbc"].notna().any():
            _append_chart_reason(charts["ins_1_shareholder_returns"], "ShareBasedCompensation 태그 미존재")
        if not basis_frame["shares_outstanding"].notna().any():
            _append_chart_reason(charts["ins_1_shareholder_returns"], "SharesOutstanding 계열 태그 미존재")

    if not basis_frame[["net_income_value", "cfo_value", "accruals_ratio_pct", "cfo_to_ni"]].notna().any().any():
        _append_chart_reason(charts["ins_2_accruals_quality"], "NetIncome/CFO/TotalAssets 태그 미존재")
    if (basis_frame["net_income_value"] <= 0).fillna(False).any():
        _append_chart_reason(charts["ins_2_accruals_quality"], "NetIncome<=0 구간은 CFO/NI null")

    if not basis_frame[["ar_delta", "inv_delta", "ap_delta"]].notna().any().any():
        _append_chart_reason(charts["ins_3_working_capital_vs_growth"], "AR/AP/Inventory 태그 미존재")
    if not basis_frame["revenue_growth_yoy"].notna().any():
        _append_chart_reason(charts["ins_3_working_capital_vs_growth"], "Revenue YoY 계산 불가(전년동기 비교 구간 부족)")

    if filings_pit.empty:
        _append_chart_reason(charts["ins_4_filing_quality"], "filings not found")
    if not _lag_series.notna().any():
        _append_chart_reason(charts["ins_4_filing_quality"], "accepted_at/filing_date 미존재")

    return {
        "ticker": tkr,
        "window": window_norm,
        "basis": basis_norm,
        "periods": periods,
        "charts": charts,
    }
