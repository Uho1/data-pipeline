from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from market_data.db_reader import load_financials_from_db
from market_data.financials_pit import load_financials_quarterly
from market_data.metrics_catalog import apply_metrics_catalog
from market_data.reader import load_price_dataframe
from market_data.sec_sector_proxy import align_sector_proxy_to_dates
from market_data.utils import coerce_series_naive
from market_data.valuation import build_bps_quarterly, build_eps_quarterly, build_eps_ttm

# ---------------------------------------------------------------------------
# Thread-local bulk-preload cache (populated by build_factor_panel when
# DuckDB is available; consumed by _prepare_price_frame /
# _prepare_quarterly_frame; cleared afterwards).
# ---------------------------------------------------------------------------
_preload = threading.local()


def _price_cache() -> dict[str, pd.DataFrame]:
    if not hasattr(_preload, "prices"):
        _preload.prices = {}
    return _preload.prices


def _fin_cache() -> dict[str, pd.DataFrame]:
    if not hasattr(_preload, "financials"):
        _preload.financials = {}
    return _preload.financials


def available_price_symbols(market: str = "us") -> list[str]:
    try:
        from market_data.db_router import db_available_for_market, get_prices_connection_for_market

        if db_available_for_market(market):
            con = get_prices_connection_for_market(market)
            rows = con.execute(
                "SELECT DISTINCT ticker FROM prices WHERE market = ? ORDER BY ticker",
                [str(market).strip().lower()],
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
    except Exception:
        return []
    return []


def _to_numeric_series(values: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if values is None:
        return pd.Series(index=index, dtype=float)
    out = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if index is not None:
        out = out.reindex(index)
    return out.astype(float)


def _to_string_series(values: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if values is None:
        return pd.Series("", index=index, dtype=str)
    out = pd.Series(values).astype(str)
    if index is not None:
        out = out.reindex(index).fillna("")
    return out


def _first_valid_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col not in df.columns:
            continue
        s = _to_numeric_series(df[col], index=df.index)
        if int(s.notna().sum()) > 0:
            return s
    return pd.Series(index=df.index, dtype=float)


def _align_quarterly_to_daily(series_q: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    s = _to_numeric_series(series_q)
    if s.empty:
        return pd.Series(index=daily_index, dtype=float)
    s = s.sort_index()
    s = s.loc[~s.index.duplicated(keep="last")]
    out = s.reindex(daily_index.union(s.index)).sort_index().ffill().reindex(daily_index)
    return _to_numeric_series(out, index=daily_index)


def _align_asof_statement_date(q_index: pd.DatetimeIndex, daily_index: pd.DatetimeIndex) -> pd.Series:
    if q_index.empty:
        return pd.Series(index=daily_index, dtype="datetime64[ns]")
    idx = pd.DatetimeIndex(q_index).sort_values().drop_duplicates()
    markers = pd.Series(idx, index=idx)
    out = markers.reindex(daily_index.union(idx)).sort_index().ffill().reindex(daily_index)
    out = pd.to_datetime(out, errors="coerce")
    return pd.Series(out.to_numpy(), index=daily_index, dtype="datetime64[ns]")


def _prepare_price_frame(ticker: str, market: str) -> pd.DataFrame:
    cached = _price_cache().get(ticker.upper())
    if cached is not None:
        df = cached
    else:
        df, _ = load_price_dataframe(ticker=ticker, market=market)
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()

    if "Open" not in out.columns or "Close" not in out.columns:
        raise ValueError(f"Price data missing Open/Close columns for {ticker}")

    out["open"] = pd.to_numeric(out["Open"], errors="coerce")
    out["close"] = pd.to_numeric(out["Close"], errors="coerce")
    if "Adj Close" in out.columns:
        out["adj_close"] = pd.to_numeric(out["Adj Close"], errors="coerce")
    else:
        out["adj_close"] = out["close"]
    out["high"] = pd.to_numeric(out.get("High"), errors="coerce")
    out["low"] = pd.to_numeric(out.get("Low"), errors="coerce")
    out["volume"] = pd.to_numeric(out.get("Volume"), errors="coerce")
    out = out[["open", "high", "low", "close", "adj_close", "volume"]]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _prepare_quarterly_frame(
    ticker: str,
    market: str,
    *,
    use_fundamentals_pit: bool = False,
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
    offline_mode: bool = False,
) -> pd.DataFrame:
    if use_fundamentals_pit:
        q = load_financials_quarterly(
            market=market,
            tickers=[ticker],
            statement_type="merged",
            use_next_trading_day_availability=use_next_trading_day_availability,
            availability_fallback=availability_fallback,
            fallback_q_days=fallback_q_days,
            fallback_k_days=fallback_k_days,
        )
        if q is None or q.empty:
            return pd.DataFrame()
        out = q.copy()
        out["PeriodEnd"] = pd.to_datetime(out.get("PeriodEnd"), errors="coerce")
        out = out.loc[~out["PeriodEnd"].isna()].copy()
        out = out.sort_values(["PeriodEnd", "AvailableDate", "FilingDate", "AcceptedAt"])
        out = out.set_index("PeriodEnd", drop=False)
        return out

    cached_fin = _fin_cache().get(ticker.upper())
    if cached_fin is not None:
        q = cached_fin
    else:
        try:
            loaded = load_financials_from_db(ticker=ticker, market=market)
            q = loaded if loaded is not None else pd.DataFrame()
        except Exception:
            q = pd.DataFrame()
    if q is None or q.empty:
        return pd.DataFrame()

    out = q.copy()
    date_col = None
    if "StatementDate" in out.columns:
        date_col = "StatementDate"
    elif "PeriodEnd" in out.columns:
        date_col = "PeriodEnd"
    if date_col is None:
        return pd.DataFrame()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.loc[~out[date_col].isna()].copy()
    out = out.sort_values(date_col)
    out = out.drop_duplicates(subset=[date_col], keep="last")
    out = out.set_index(date_col, drop=False)

    if out.empty:
        return pd.DataFrame()
    return out


def _compute_quarterly_metrics(q: pd.DataFrame) -> dict[str, pd.Series]:
    if q is None or q.empty:
        empty = pd.Series(dtype=float)
        return {
            "eps_ttm": empty,
            "bps": empty,
            "revenue_ttm": empty,
            "op_income_ttm": empty,
            "gross_profit_ttm": empty,
            "da_ttm": empty,
            "sbc_ttm": empty,
            "interest_expense_ttm": empty,
            "pretax_income_ttm": empty,
            "tax_ttm": empty,
            "ocf_ttm": empty,
            "fcf_ttm": empty,
            "ni_ttm": empty,
            "dividends_ttm": empty,
            "repurchases_ttm": empty,
            "avg_equity": empty,
            "equity": empty,
            "avg_assets": empty,
            "assets": empty,
            "liabilities": empty,
            "current_assets": empty,
            "current_liabilities": empty,
            "ar": empty,
            "ap": empty,
            "inventory": empty,
            "cash_and_equivalents": empty,
            "debt_short": empty,
            "debt_long": empty,
            "total_debt": empty,
            "deferred_revenue": empty,
            "goodwill": empty,
            "intangibles": empty,
            "ebitda_ttm": empty,
            "shares": empty,
            "asof_statement_date": pd.Series(dtype="datetime64[ns]"),
        }

    idx = pd.DatetimeIndex(q.index)

    eps_q, _ = build_eps_quarterly(q)
    eps_ttm = build_eps_ttm(eps_q)
    bps_q = build_bps_quarterly(q)

    revenue_q = _first_valid_series(q, ["Revenue", "Total Revenue", "Operating Revenue"])
    revenue_ttm = revenue_q.rolling(window=4, min_periods=1).sum()

    op_income_q = _first_valid_series(q, ["Operating Income", "EBIT", "Total Operating Income As Reported"])
    op_income_ttm = op_income_q.rolling(window=4, min_periods=1).sum()

    gross_profit_q = _first_valid_series(q, ["Gross Profit"])
    gross_profit_ttm = gross_profit_q.rolling(window=4, min_periods=1).sum()
    da_q = _first_valid_series(q, ["D&A", "Depreciation", "Depreciation And Amortization"])
    da_ttm = da_q.rolling(window=4, min_periods=1).sum()
    sbc_q = _first_valid_series(q, ["SBC", "Share Based Compensation"])
    sbc_ttm = sbc_q.rolling(window=4, min_periods=1).sum()
    interest_q = _first_valid_series(q, ["Interest", "Interest Expense"])
    interest_ttm = interest_q.rolling(window=4, min_periods=1).sum()
    pretax_q = _first_valid_series(q, ["Pretax Income"])
    pretax_ttm = pretax_q.rolling(window=4, min_periods=1).sum()
    tax_q = _first_valid_series(q, ["Tax"])
    tax_ttm = tax_q.rolling(window=4, min_periods=1).sum()

    ni_q = _first_valid_series(q, ["net_income_common", "Net Income Common", "Net Income"])
    ni_ttm = ni_q.rolling(window=4, min_periods=1).sum()

    ocf_q = _first_valid_series(
        q,
        ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Net Cash Provided By Operating Activities"],
    )
    ocf_ttm = ocf_q.rolling(window=4, min_periods=1).sum()
    capex_q = _first_valid_series(q, ["Capital Expenditure", "Capital Expenditure Reported"])
    fcf_q = ocf_q + capex_q
    fcf_ttm = fcf_q.rolling(window=4, min_periods=1).sum()
    dividends_q = _first_valid_series(q, ["Dividends Paid"])
    dividends_ttm = dividends_q.rolling(window=4, min_periods=1).sum().abs()
    repurchases_q = _first_valid_series(q, ["Repurchases"])
    repurchases_ttm = repurchases_q.rolling(window=4, min_periods=1).sum().abs()

    equity_q = _first_valid_series(
        q,
        [
            "Shareholders Equity",
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        ],
    )
    avg_equity = equity_q.rolling(window=4, min_periods=4).mean()

    assets_q = _first_valid_series(q, ["Total Assets", "Total Assets As Reported", "Total Assets Gross Minority Interest"])
    avg_assets = assets_q.rolling(window=4, min_periods=4).mean()
    liabilities_q = _first_valid_series(q, ["Total Liabilities", "Total Liabilities Net Minority Interest", "Current Liabilities"])
    current_assets_q = _first_valid_series(q, ["Current Assets", "Current Assets Total"])
    current_liabilities_q = _first_valid_series(q, ["Current Liabilities", "Current Liabilities Total"])
    ar_q = _first_valid_series(q, ["AR", "Accounts Receivable"])
    ap_q = _first_valid_series(q, ["AP", "Accounts Payable"])
    inventory_q = _first_valid_series(q, ["Inventory"])
    cash_q = _first_valid_series(q, ["Cash", "Cash And Cash Equivalents"])
    debt_short_q = _first_valid_series(q, ["Debt Short", "Short Term Debt"])
    debt_long_q = _first_valid_series(q, ["Debt Long", "Long Term Debt"])
    deferred_revenue_q = _first_valid_series(q, ["Deferred Revenue"])
    goodwill_q = _first_valid_series(q, ["Goodwill"])
    intangibles_q = _first_valid_series(q, ["Intangibles"])

    shares = _first_valid_series(q, ["Shares", "diluted_shares", "basic_shares", "Diluted Shares", "Basic Shares"])

    asof = pd.Series(idx, index=idx, dtype="datetime64[ns]")
    return {
        "eps_ttm": _to_numeric_series(eps_ttm, index=idx),
        "bps": _to_numeric_series(bps_q, index=idx),
        "revenue_ttm": _to_numeric_series(revenue_ttm, index=idx),
        "op_income_ttm": _to_numeric_series(op_income_ttm, index=idx),
        "gross_profit_ttm": _to_numeric_series(gross_profit_ttm, index=idx),
        "da_ttm": _to_numeric_series(da_ttm, index=idx),
        "sbc_ttm": _to_numeric_series(sbc_ttm, index=idx),
        "interest_expense_ttm": _to_numeric_series(interest_ttm, index=idx),
        "pretax_income_ttm": _to_numeric_series(pretax_ttm, index=idx),
        "tax_ttm": _to_numeric_series(tax_ttm, index=idx),
        "ocf_ttm": _to_numeric_series(ocf_ttm, index=idx),
        "fcf_ttm": _to_numeric_series(fcf_ttm, index=idx),
        "ni_ttm": _to_numeric_series(ni_ttm, index=idx),
        "dividends_ttm": _to_numeric_series(dividends_ttm, index=idx),
        "repurchases_ttm": _to_numeric_series(repurchases_ttm, index=idx),
        "avg_equity": _to_numeric_series(avg_equity, index=idx),
        "equity": _to_numeric_series(equity_q, index=idx),
        "avg_assets": _to_numeric_series(avg_assets, index=idx),
        "assets": _to_numeric_series(assets_q, index=idx),
        "liabilities": _to_numeric_series(liabilities_q, index=idx),
        "current_assets": _to_numeric_series(current_assets_q, index=idx),
        "current_liabilities": _to_numeric_series(current_liabilities_q, index=idx),
        "ar": _to_numeric_series(ar_q, index=idx),
        "ap": _to_numeric_series(ap_q, index=idx),
        "inventory": _to_numeric_series(inventory_q, index=idx),
        "cash_and_equivalents": _to_numeric_series(cash_q, index=idx),
        "debt_short": _to_numeric_series(debt_short_q, index=idx),
        "debt_long": _to_numeric_series(debt_long_q, index=idx),
        "total_debt": _to_numeric_series(debt_short_q, index=idx).fillna(0.0) + _to_numeric_series(debt_long_q, index=idx).fillna(0.0),
        "deferred_revenue": _to_numeric_series(deferred_revenue_q, index=idx),
        "goodwill": _to_numeric_series(goodwill_q, index=idx),
        "intangibles": _to_numeric_series(intangibles_q, index=idx),
        "ebitda_ttm": _to_numeric_series(op_income_ttm + da_ttm, index=idx),
        "shares": _to_numeric_series(shares, index=idx),
        "asof_statement_date": asof,
    }


def _row_more_recent(new_row: pd.Series, cur_row: pd.Series) -> bool:
    n_filed = pd.to_datetime(new_row.get("FilingDate"), errors="coerce")
    c_filed = pd.to_datetime(cur_row.get("FilingDate"), errors="coerce")
    if pd.notna(n_filed) and (pd.isna(c_filed) or n_filed > c_filed):
        return True
    if pd.notna(c_filed) and (pd.isna(n_filed) or n_filed < c_filed):
        return False
    n_acc = pd.to_datetime(new_row.get("AcceptedAt"), errors="coerce")
    c_acc = pd.to_datetime(cur_row.get("AcceptedAt"), errors="coerce")
    if pd.notna(n_acc) and (pd.isna(c_acc) or n_acc > c_acc):
        return True
    if pd.notna(c_acc) and (pd.isna(n_acc) or n_acc < c_acc):
        return False
    n_col = pd.to_datetime(new_row.get("CollectedAt"), errors="coerce")
    c_col = pd.to_datetime(cur_row.get("CollectedAt"), errors="coerce")
    if pd.notna(n_col) and (pd.isna(c_col) or n_col > c_col):
        return True
    return False


def _compute_daily_metrics_from_pit_events(q_events: pd.DataFrame, daily_index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    if q_events is None or q_events.empty:
        empty = pd.Series(index=daily_index, dtype=float)
        return {
            "eps_ttm": empty,
            "bps": empty,
            "revenue_ttm": empty,
            "op_income_ttm": empty,
            "gross_profit_ttm": empty,
            "da_ttm": empty,
            "sbc_ttm": empty,
            "interest_expense_ttm": empty,
            "pretax_income_ttm": empty,
            "tax_ttm": empty,
            "ocf_ttm": empty,
            "fcf_ttm": empty,
            "ni_ttm": empty,
            "dividends_ttm": empty,
            "repurchases_ttm": empty,
            "avg_equity": empty,
            "equity": empty,
            "avg_assets": empty,
            "assets": empty,
            "liabilities": empty,
            "current_assets": empty,
            "current_liabilities": empty,
            "ar": empty,
            "ap": empty,
            "inventory": empty,
            "cash_and_equivalents": empty,
            "debt_short": empty,
            "debt_long": empty,
            "total_debt": empty,
            "deferred_revenue": empty,
            "goodwill": empty,
            "intangibles": empty,
            "ebitda_ttm": empty,
            "shares": empty,
            "asof_statement_date": pd.Series(index=daily_index, dtype="datetime64[ns]"),
            "asof_available_date": pd.Series(index=daily_index, dtype="datetime64[ns]"),
            "availability_method": pd.Series(index=daily_index, dtype="object"),
            "fallback_used_rows": pd.Series(index=daily_index, dtype=float).fillna(0.0),
        }

    events = q_events.reset_index(drop=True).copy()
    events["PeriodEnd"] = pd.to_datetime(events.get("PeriodEnd"), errors="coerce")
    events["AvailableDate"] = pd.to_datetime(events.get("AvailableDate"), errors="coerce")
    events["FilingDate"] = pd.to_datetime(events.get("FilingDate"), errors="coerce")
    events["AcceptedAt"] = pd.to_datetime(events.get("AcceptedAt"), errors="coerce")
    events = events.dropna(subset=["PeriodEnd", "AvailableDate"]).copy()
    if events.empty:
        return _compute_daily_metrics_from_pit_events(pd.DataFrame(), daily_index)

    events = events.sort_values(["AvailableDate", "PeriodEnd", "FilingDate", "AcceptedAt"])
    state: dict[pd.Timestamp, pd.Series] = {}
    snapshots: list[dict[str, Any]] = []

    for available_date, group in events.groupby("AvailableDate", sort=True):
        for _, row in group.iterrows():
            period_end = pd.Timestamp(pd.to_datetime(row.get("PeriodEnd"), errors="coerce")).normalize()
            cur = state.get(period_end)
            if cur is None or _row_more_recent(row, cur):
                state[period_end] = row
        if not state:
            continue
        snap_df = pd.DataFrame(list(state.values()))
        if snap_df.empty:
            continue
        snap_df["StatementDate"] = pd.to_datetime(snap_df.get("PeriodEnd"), errors="coerce")
        snap_df = snap_df.dropna(subset=["StatementDate"]).sort_values("StatementDate")
        if snap_df.empty:
            continue
        snap_df = snap_df.set_index("StatementDate", drop=False)
        metrics = _compute_quarterly_metrics(snap_df)
        latest_period = pd.Timestamp(snap_df.index.max())
        latest_evt = snap_df.loc[latest_period]
        if isinstance(latest_evt, pd.DataFrame):
            latest_evt = latest_evt.iloc[-1]

        row: dict[str, Any] = {"event_date": pd.Timestamp(available_date).normalize()}
        for key in [
            "eps_ttm",
            "bps",
            "revenue_ttm",
            "op_income_ttm",
            "gross_profit_ttm",
            "da_ttm",
            "sbc_ttm",
            "interest_expense_ttm",
            "pretax_income_ttm",
            "tax_ttm",
            "ocf_ttm",
            "fcf_ttm",
            "ni_ttm",
            "dividends_ttm",
            "repurchases_ttm",
            "avg_equity",
            "equity",
            "avg_assets",
            "assets",
            "liabilities",
            "current_assets",
            "current_liabilities",
            "ar",
            "ap",
            "inventory",
            "cash_and_equivalents",
            "debt_short",
            "debt_long",
            "total_debt",
            "deferred_revenue",
            "goodwill",
            "intangibles",
            "ebitda_ttm",
            "shares",
        ]:
            val = float(pd.to_numeric(pd.Series([metrics[key].get(latest_period, np.nan)]), errors="coerce").iloc[0])
            if pd.isna(val) and key in latest_evt:
                val = float(pd.to_numeric(pd.Series([latest_evt[key]]), errors="coerce").iloc[0])
            row[key] = val
            
        row["asof_statement_date"] = latest_period
        row["asof_available_date"] = pd.Timestamp(available_date).normalize()
        row["availability_method"] = str(latest_evt.get("AvailabilityMethod", ""))
        
        # Robustly compute fallback_used_rows
        avail_col = snap_df.get("AvailabilityMethod")
        if avail_col is not None:
            row["fallback_used_rows"] = float(
                pd.Series(avail_col).astype(str).str.lower().str.startswith("fallback").sum()
            )
        else:
            row["fallback_used_rows"] = 0.0
        snapshots.append(row)

    if not snapshots:
        return _compute_daily_metrics_from_pit_events(pd.DataFrame(), daily_index)

    snap = pd.DataFrame(snapshots)
    snap["event_date"] = pd.to_datetime(snap["event_date"], errors="coerce")
    snap = snap.dropna(subset=["event_date"]).sort_values("event_date")
    snap = snap.drop_duplicates(subset=["event_date"], keep="last").set_index("event_date")

    out: dict[str, pd.Series] = {}
    for key in [
        "eps_ttm",
        "bps",
        "revenue_ttm",
        "op_income_ttm",
        "gross_profit_ttm",
        "da_ttm",
        "sbc_ttm",
        "interest_expense_ttm",
        "pretax_income_ttm",
        "tax_ttm",
        "ocf_ttm",
        "fcf_ttm",
        "ni_ttm",
        "dividends_ttm",
        "repurchases_ttm",
        "avg_equity",
        "equity",
        "avg_assets",
        "assets",
        "liabilities",
        "current_assets",
        "current_liabilities",
        "ar",
        "ap",
        "inventory",
        "cash_and_equivalents",
        "debt_short",
        "debt_long",
        "total_debt",
        "deferred_revenue",
        "goodwill",
        "intangibles",
        "ebitda_ttm",
        "shares",
    ]:
        vals = snap.get(key)
        if vals is None:
            out[key] = pd.Series(np.nan, index=daily_index, dtype=float)
            continue
        ser = _to_numeric_series(vals, index=pd.DatetimeIndex(snap.index))
        # Combine indexes, sort, ffill, then reindex to daily_index
        full_idx = pd.DatetimeIndex(daily_index.union(ser.index)).sort_values().unique()
        out[key] = ser.reindex(full_idx).ffill().reindex(daily_index)
    for key in ["asof_statement_date", "asof_available_date"]:
        vals = snap.get(key)
        if vals is None:
            out[key] = pd.Series(pd.NaT, index=daily_index, dtype="datetime64[ns]")
            continue
        ser = pd.Series(coerce_series_naive(vals).values, index=pd.DatetimeIndex(snap.index))
        # Combine indexes, sort, ffill, then reindex to daily_index
        full_idx = pd.DatetimeIndex(daily_index.union(ser.index)).sort_values().unique()
        out[key] = ser.reindex(full_idx).ffill().reindex(daily_index)
    method_ser = _to_string_series(snap.get("availability_method"), index=snap.index)
    out["availability_method"] = method_ser.reindex(daily_index.union(method_ser.index)).sort_index().ffill().reindex(daily_index)
    fallback_ser = pd.to_numeric(snap.get("fallback_used_rows"), errors="coerce").fillna(0.0)
    out["fallback_used_rows"] = fallback_ser.reindex(daily_index.union(fallback_ser.index)).sort_index().ffill().reindex(daily_index).fillna(0.0)
    return out


def build_ticker_factor_frame(
    ticker: str,
    market: str = "us",
    start: str | None = None,
    end: str | None = None,
    asof_mode: str = "quarter_end",
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
    offline_mode: bool = False,
) -> pd.DataFrame:
    asof_mode_norm = str(asof_mode).strip().lower()
    if asof_mode_norm not in {"quarter_end", "available_date", "filing_date"}:
        raise NotImplementedError(f"Unsupported asof_mode='{asof_mode}'")

    price = _prepare_price_frame(ticker=ticker, market=market)
    if price.empty:
        return pd.DataFrame()

    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.notna(start_ts):
        price = price.loc[price.index >= pd.Timestamp(start_ts).normalize()]
    if pd.notna(end_ts):
        price = price.loc[price.index <= pd.Timestamp(end_ts).normalize()]
    if price.empty:
        return pd.DataFrame()

    use_fundamentals_pit = asof_mode_norm in {"available_date", "filing_date"}
    q = _prepare_quarterly_frame(
        ticker=ticker,
        market=market,
        use_fundamentals_pit=use_fundamentals_pit,
        use_next_trading_day_availability=use_next_trading_day_availability,
        availability_fallback=availability_fallback,
        fallback_q_days=fallback_q_days,
        fallback_k_days=fallback_k_days,
        offline_mode=offline_mode,
    )
    q_metrics = _compute_quarterly_metrics(q) if not use_fundamentals_pit else {}

    # Ensure price index is timezone-naive
    if hasattr(price.index, "tz") and price.index.tz is not None:
        price.index = price.index.tz_localize(None)
    idx = pd.DatetimeIndex(price.index)

    # Ensure quarterly dates are timezone-naive
    for col in ["StatementDate", "PeriodEnd", "available_date"]:
        if col in q.columns:
            q[col] = coerce_series_naive(q[col]).dt.normalize()
    if use_fundamentals_pit:
        pit_metrics = _compute_daily_metrics_from_pit_events(q_events=q, daily_index=idx)
        eps_ttm_daily = _to_numeric_series(pit_metrics["eps_ttm"], index=idx)
        bps_daily = _to_numeric_series(pit_metrics["bps"], index=idx)
        revenue_ttm_daily = _to_numeric_series(pit_metrics["revenue_ttm"], index=idx)
        op_income_ttm_daily = _to_numeric_series(pit_metrics["op_income_ttm"], index=idx)
        gross_profit_ttm_daily = _to_numeric_series(pit_metrics["gross_profit_ttm"], index=idx)
        da_ttm_daily = _to_numeric_series(pit_metrics["da_ttm"], index=idx)
        sbc_ttm_daily = _to_numeric_series(pit_metrics["sbc_ttm"], index=idx)
        interest_expense_ttm_daily = _to_numeric_series(pit_metrics["interest_expense_ttm"], index=idx)
        pretax_income_ttm_daily = _to_numeric_series(pit_metrics["pretax_income_ttm"], index=idx)
        tax_ttm_daily = _to_numeric_series(pit_metrics["tax_ttm"], index=idx)
        ocf_ttm_daily = _to_numeric_series(pit_metrics["ocf_ttm"], index=idx)
        fcf_ttm_daily = _to_numeric_series(pit_metrics["fcf_ttm"], index=idx)
        ni_ttm_daily = _to_numeric_series(pit_metrics["ni_ttm"], index=idx)
        dividends_ttm_daily = _to_numeric_series(pit_metrics["dividends_ttm"], index=idx)
        repurchases_ttm_daily = _to_numeric_series(pit_metrics["repurchases_ttm"], index=idx)
        avg_equity_daily = _to_numeric_series(pit_metrics["avg_equity"], index=idx)
        equity_daily = _to_numeric_series(pit_metrics["equity"], index=idx)
        avg_assets_daily = _to_numeric_series(pit_metrics["avg_assets"], index=idx)
        assets_daily = _to_numeric_series(pit_metrics["assets"], index=idx)
        liabilities_daily = _to_numeric_series(pit_metrics["liabilities"], index=idx)
        current_assets_daily = _to_numeric_series(pit_metrics["current_assets"], index=idx)
        current_liabilities_daily = _to_numeric_series(pit_metrics["current_liabilities"], index=idx)
        ar_daily = _to_numeric_series(pit_metrics["ar"], index=idx)
        ap_daily = _to_numeric_series(pit_metrics["ap"], index=idx)
        inventory_daily = _to_numeric_series(pit_metrics["inventory"], index=idx)
        cash_daily = _to_numeric_series(pit_metrics["cash_and_equivalents"], index=idx)
        debt_short_daily = _to_numeric_series(pit_metrics["debt_short"], index=idx)
        debt_long_daily = _to_numeric_series(pit_metrics["debt_long"], index=idx)
        total_debt_daily = _to_numeric_series(pit_metrics["total_debt"], index=idx)
        deferred_revenue_daily = _to_numeric_series(pit_metrics["deferred_revenue"], index=idx)
        goodwill_daily = _to_numeric_series(pit_metrics["goodwill"], index=idx)
        intangibles_daily = _to_numeric_series(pit_metrics["intangibles"], index=idx)
        ebitda_ttm_daily = _to_numeric_series(pit_metrics["ebitda_ttm"], index=idx)
        shares_daily = _to_numeric_series(pit_metrics["shares"], index=idx)
        asof_statement_date = coerce_series_naive(pit_metrics["asof_statement_date"])
        asof_available_date = coerce_series_naive(pit_metrics["asof_available_date"])
        availability_method = _to_string_series(pit_metrics.get("availability_method"), index=idx)
        fallback_used_rows = _to_numeric_series(pit_metrics["fallback_used_rows"], index=idx).fillna(0.0)
    else:
        eps_ttm_daily = _align_quarterly_to_daily(q_metrics["eps_ttm"], idx)
        bps_daily = _align_quarterly_to_daily(q_metrics["bps"], idx)
        revenue_ttm_daily = _align_quarterly_to_daily(q_metrics["revenue_ttm"], idx)
        op_income_ttm_daily = _align_quarterly_to_daily(q_metrics["op_income_ttm"], idx)
        gross_profit_ttm_daily = _align_quarterly_to_daily(q_metrics["gross_profit_ttm"], idx)
        da_ttm_daily = _align_quarterly_to_daily(q_metrics["da_ttm"], idx)
        sbc_ttm_daily = _align_quarterly_to_daily(q_metrics["sbc_ttm"], idx)
        interest_expense_ttm_daily = _align_quarterly_to_daily(q_metrics["interest_expense_ttm"], idx)
        pretax_income_ttm_daily = _align_quarterly_to_daily(q_metrics["pretax_income_ttm"], idx)
        tax_ttm_daily = _align_quarterly_to_daily(q_metrics["tax_ttm"], idx)
        ocf_ttm_daily = _align_quarterly_to_daily(q_metrics["ocf_ttm"], idx)
        fcf_ttm_daily = _align_quarterly_to_daily(q_metrics["fcf_ttm"], idx)
        ni_ttm_daily = _align_quarterly_to_daily(q_metrics["ni_ttm"], idx)
        dividends_ttm_daily = _align_quarterly_to_daily(q_metrics["dividends_ttm"], idx)
        repurchases_ttm_daily = _align_quarterly_to_daily(q_metrics["repurchases_ttm"], idx)
        avg_equity_daily = _align_quarterly_to_daily(q_metrics["avg_equity"], idx)
        equity_daily = _align_quarterly_to_daily(q_metrics["equity"], idx)
        avg_assets_daily = _align_quarterly_to_daily(q_metrics["avg_assets"], idx)
        assets_daily = _align_quarterly_to_daily(q_metrics["assets"], idx)
        liabilities_daily = _align_quarterly_to_daily(q_metrics["liabilities"], idx)
        current_assets_daily = _align_quarterly_to_daily(q_metrics["current_assets"], idx)
        current_liabilities_daily = _align_quarterly_to_daily(q_metrics["current_liabilities"], idx)
        ar_daily = _align_quarterly_to_daily(q_metrics["ar"], idx)
        ap_daily = _align_quarterly_to_daily(q_metrics["ap"], idx)
        inventory_daily = _align_quarterly_to_daily(q_metrics["inventory"], idx)
        cash_daily = _align_quarterly_to_daily(q_metrics["cash_and_equivalents"], idx)
        debt_short_daily = _align_quarterly_to_daily(q_metrics["debt_short"], idx)
        debt_long_daily = _align_quarterly_to_daily(q_metrics["debt_long"], idx)
        total_debt_daily = _align_quarterly_to_daily(q_metrics["total_debt"], idx)
        deferred_revenue_daily = _align_quarterly_to_daily(q_metrics["deferred_revenue"], idx)
        goodwill_daily = _align_quarterly_to_daily(q_metrics["goodwill"], idx)
        intangibles_daily = _align_quarterly_to_daily(q_metrics["intangibles"], idx)
        ebitda_ttm_daily = _align_quarterly_to_daily(q_metrics["ebitda_ttm"], idx)
        shares_daily = _align_quarterly_to_daily(q_metrics["shares"], idx)
        if q.empty:
            asof_statement_date = pd.Series(index=idx, dtype="datetime64[ns]")
        else:
            asof_statement_date = _align_asof_statement_date(pd.DatetimeIndex(q.index), idx)
        asof_available_date = pd.Series(index=idx, dtype="datetime64[ns]")
        availability_method = pd.Series(index=idx, dtype="object").fillna("")
        fallback_used_rows = pd.Series(index=idx, dtype=float).fillna(0.0)

    close = _to_numeric_series(price["close"], index=idx)
    open_ = _to_numeric_series(price["open"], index=idx)
    high = _to_numeric_series(price["high"], index=idx)
    low = _to_numeric_series(price["low"], index=idx)
    adj_close = _to_numeric_series(price["adj_close"], index=idx)
    volume = _to_numeric_series(price["volume"], index=idx)

    market_cap = close * shares_daily
    adv20 = volume.rolling(window=20, min_periods=1).mean()
    dollar_volume_20d = (close * volume).rolling(window=20, min_periods=1).mean()

    pe = close / eps_ttm_daily.replace(0.0, np.nan)
    pe = pe.where(eps_ttm_daily > 0.0)
    ps = market_cap / revenue_ttm_daily.replace(0.0, np.nan)
    pb = close / bps_daily.replace(0.0, np.nan)
    pb = pb.where(bps_daily > 0.0)
    fcf_yield = fcf_ttm_daily / market_cap.replace(0.0, np.nan)
    roe = ni_ttm_daily / avg_equity_daily.replace(0.0, np.nan)

    out = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": adj_close,
            "volume": volume,
            "price": close,
            "adv20": adv20,
            "dollar_volume_20d": dollar_volume_20d,
            "market_cap": market_cap,
            "eps_ttm": eps_ttm_daily,
            "bps": bps_daily,
            "revenue_ttm": revenue_ttm_daily,
            "op_income_ttm": op_income_ttm_daily,
            "gross_profit_ttm": gross_profit_ttm_daily,
            "da_ttm": da_ttm_daily,
            "sbc_ttm": sbc_ttm_daily,
            "interest_expense_ttm": interest_expense_ttm_daily,
            "pretax_income_ttm": pretax_income_ttm_daily,
            "tax_ttm": tax_ttm_daily,
            "ocf_ttm": ocf_ttm_daily,
            "fcf_ttm": fcf_ttm_daily,
            "ni_ttm": ni_ttm_daily,
            "dividends_ttm": dividends_ttm_daily,
            "repurchases_ttm": repurchases_ttm_daily,
            "avg_equity": avg_equity_daily,
            "equity": equity_daily,
            "avg_assets": avg_assets_daily,
            "assets": assets_daily,
            "liabilities": liabilities_daily,
            "current_assets": current_assets_daily,
            "current_liabilities": current_liabilities_daily,
            "ar": ar_daily,
            "ap": ap_daily,
            "inventory": inventory_daily,
            "cash_and_equivalents": cash_daily,
            "debt_short": debt_short_daily,
            "debt_long": debt_long_daily,
            "total_debt": total_debt_daily,
            "deferred_revenue": deferred_revenue_daily,
            "goodwill": goodwill_daily,
            "intangibles": intangibles_daily,
            "ebitda_ttm": ebitda_ttm_daily,
            "shares": shares_daily,
            "pe": pe,
            "ps": ps,
            "pb": pb,
            "fcf_yield": fcf_yield,
            "roe": roe,
            "asof_statement_date": asof_statement_date,
            "asof_available_date": asof_available_date,
            "availability_method": availability_method,
            "fallback_used_rows": fallback_used_rows,
        },
        index=idx,
    )

    # Inject point-in-time sector proxy (SIC-based mapping + manual overrides) on daily index.
    try:
        sector_daily = align_sector_proxy_to_dates(ticker=ticker, dates=idx, market=market)
        if not sector_daily.empty:
            out = out.join(sector_daily, how="left")
    except Exception:
        # Keep factor build resilient even if identity cache/rules are temporarily missing.
        out["sec_sic"] = ""
        out["sec_sic_description"] = ""
        out["sector_l1_kr"] = "미분류"
        out["sector_l2_kr"] = "미분류"
        out["mapping_source"] = "fallback_unclassified"
        out["mapping_confidence"] = 0.0
        out["rule_id"] = "fallback_unclassified"
        out["mapping_version"] = "sec_sector_proxy_v1"
        out["source_filing_date"] = pd.NaT
        out["source_acceptance_datetime"] = pd.NaT
        out["sector_valid_from"] = pd.NaT
        out["sector_valid_to"] = pd.NaT

    out = out.replace([np.inf, -np.inf], np.nan)
    out["Ticker"] = str(ticker).strip().upper()
    return out


def build_factor_panel(
    symbols: list[str],
    market: str = "us",
    start: str | None = None,
    end: str | None = None,
    asof_mode: str = "quarter_end",
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
    offline_mode: bool = False,
    loading_callback: "Callable[[int, int, str], None] | None" = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    
    # Use the locally defined available_price_symbols
    avail_prices = set(available_price_symbols(market=market))
    
    def _norm(s: str) -> str:
        return str(s).strip().upper().replace(".", "").replace("-", "")
    
    price_map = {_norm(s): s for s in avail_prices}
    
    import logging
    logger = logging.getLogger(__name__)
    
    unique_targets = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    total = len(unique_targets)

    # ------------------------------------------------------------------
    # Bulk pre-load from DuckDB when available (dramatically reduces the
    # number of file opens for multi-ticker runs).
    # ------------------------------------------------------------------
    try:
        from market_data.db_router import db_available_for_market
        if db_available_for_market(market):
            from market_data.db_reader import (
                bulk_load_financials_quarterly,
                bulk_load_prices,
            )
            _price_cache().update(
                bulk_load_prices(unique_targets, market=market, start=start, end=end)
            )
            _fin_cache().update(
                bulk_load_financials_quarterly(unique_targets, market=market)
            )
    except Exception as _db_exc:
        logger.debug("DuckDB bulk pre-load skipped: %s", _db_exc)

    for idx, symbol in enumerate(unique_targets):
        if loading_callback is not None:
            try:
                loading_callback(idx + 1, total, symbol)
            except Exception:
                pass
        # Try direct match first, then normalized match
        target_sym = symbol
        if target_sym not in avail_prices:
            target_sym = price_map.get(_norm(symbol))
            if not target_sym:
                # logger.debug(f"Skipping {symbol}: no price data found in {market}")
                continue

        try:
            frame = build_ticker_factor_frame(
                ticker=target_sym,
                market=market,
                start=start,
                end=end,
                asof_mode=asof_mode,
                use_next_trading_day_availability=use_next_trading_day_availability,
                availability_fallback=availability_fallback,
                fallback_q_days=fallback_q_days,
                fallback_k_days=fallback_k_days,
                offline_mode=offline_mode,
            )
            if frame is not None and not frame.empty:
                # Ensure the ticker in the frame matches the requested symbol for the backtester
                frame["Ticker"] = symbol
                frames.append(frame)
            else:
                pass # logger.debug(f"Skipping {symbol}: build_ticker_factor_frame returned empty")
        except Exception as exc:
            logger.warning(f"Error building factor frame for {symbol} (using {target_sym}): {exc}")
            continue

    # Clear bulk-preload caches so stale data doesn't linger in the thread.
    _price_cache().clear()
    _fin_cache().clear()

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, axis=0, ignore_index=False)
    merged.index = pd.to_datetime(merged.index, errors="coerce")
    if hasattr(merged.index, "tz") and merged.index.tz is not None:
        merged.index = merged.index.tz_localize(None)
    merged = merged.loc[~merged.index.isna()].sort_index()

    merged = merged.reset_index().rename(columns={"index": "Date"})
    merged["Date"] = coerce_series_naive(merged["Date"])
    merged["Ticker"] = merged["Ticker"].astype(str).str.upper().str.strip()
    merged = merged.dropna(subset=["Date", "Ticker"])

    out = merged.set_index(["Date", "Ticker"]).sort_index()
    if hasattr(out.index.levels[0], "tz") and out.index.levels[0].tz is not None:
        new_levels = [out.index.levels[0].tz_localize(None), out.index.levels[1]]
        out.index = out.index.set_levels(new_levels)
    out, _ = apply_metrics_catalog(out)
    return out


def load_factor_panel_from_local(
    market: str = "us",
    start: str | None = None,
    end: str | None = None,
    asof_mode: str = "quarter_end",
    symbols: list[str] | None = None,
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
    offline_mode: bool = False,
) -> pd.DataFrame:
    if symbols:
        target = [str(s).strip().upper() for s in symbols if str(s).strip()]
    else:
        target = available_price_symbols(market=market)
    return build_factor_panel(
        symbols=target,
        market=market,
        start=start,
        end=end,
        asof_mode=asof_mode,
        use_next_trading_day_availability=use_next_trading_day_availability,
        availability_fallback=availability_fallback,
        fallback_q_days=fallback_q_days,
        fallback_k_days=fallback_k_days,
        offline_mode=offline_mode,
    )
