from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_data.utils import sanitize_ticker, coerce_series_naive

SUPPORTED_STATEMENT_TYPES = {
    "income": "income_quarterly",
    "balance": "balance_quarterly",
    "cashflow": "cashflow_quarterly",
    "income_quarterly": "income_quarterly",
    "balance_quarterly": "balance_quarterly",
    "cashflow_quarterly": "cashflow_quarterly",
    "merged": "merged",
    "all": "merged",
}


@dataclass(frozen=True)
class FinancialsPitOptions:
    use_next_trading_day_availability: bool = False
    availability_fallback: bool = True
    fallback_q_days: int = 45
    fallback_k_days: int = 90


def _normalize_statement_type(statement_type: str) -> str:
    key = str(statement_type or "merged").strip().lower()
    return SUPPORTED_STATEMENT_TYPES.get(key, "merged")


def _ensure_duckdb_available(market: str = "us") -> None:
    from market_data.db_router import db_available_for_market

    if not db_available_for_market(market):
        raise RuntimeError(
            "DuckDB financials source is required but unavailable. "
            "Run ingest first: python -m market_data ingest ..."
        )


def _infer_form_type(period_end: pd.Timestamp, current: str | None = None) -> str:
    form = str(current or "").strip().upper()
    if form in {"10-Q", "10-K", "10-Q/A", "10-K/A"}:
        return form
    if pd.isna(period_end):
        return "10-Q"
    return "10-K" if int(pd.Timestamp(period_end).month) == 12 else "10-Q"


def _load_trading_days(market: str) -> pd.DatetimeIndex:
    from market_data.db_router import db_available_for_market, get_prices_connection_for_market

    if not db_available_for_market(market):
        return pd.DatetimeIndex([])
    market_norm = str(market or "us").strip().lower()

    try:
        con = get_prices_connection_for_market(market_norm)
        preferred_tickers = ["005930", "000660", "035420"] if market_norm == "kr" else ["SPY", "IVV", "VOO", "AAPL", "MSFT"]
        for ref in preferred_tickers:
            rows = con.execute(
                "SELECT DISTINCT date FROM prices WHERE ticker = ? AND market = ? ORDER BY date",
                [ref, market_norm],
            ).fetchall()
            if rows:
                idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
                return idx.dropna().sort_values().unique()
        rows = con.execute(
            "SELECT DISTINCT date FROM prices WHERE market = ? ORDER BY date",
            [market_norm],
        ).fetchall()
    except Exception:
        # If DB is temporarily locked/unavailable, degrade to business-day fallback.
        return pd.DatetimeIndex([])
    if not rows:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return idx.dropna().sort_values().unique()


def _next_trading_day(ts: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    dt = pd.Timestamp(ts).normalize()
    if len(trading_days) == 0 or dt < trading_days[0]:
        # trading_days doesn't cover this date — fall back to simple business day
        return (dt + pd.offsets.BDay(1)).normalize()
    pos = int(np.searchsorted(trading_days.values, np.datetime64(dt), side="right"))
    if pos >= len(trading_days):
        return (dt + pd.offsets.BDay(1)).normalize()
    return pd.Timestamp(trading_days[pos]).normalize()


def _ensure_pit_columns(
    frame: pd.DataFrame,
    *,
    market: str,
    options: FinancialsPitOptions,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["StatementDate"] = coerce_series_naive(out.get("StatementDate"))
    if "PeriodEnd" in out.columns:
        out["PeriodEnd"] = coerce_series_naive(out.get("PeriodEnd"))
    else:
        out["PeriodEnd"] = out["StatementDate"]
    out["PeriodEnd"] = out["PeriodEnd"].fillna(out["StatementDate"])
    if "PeriodStart" in out.columns:
        out["PeriodStart"] = coerce_series_naive(out.get("PeriodStart"))
    else:
        out["PeriodStart"] = pd.NaT

    if "FormType" in out.columns:
        out["FormType"] = out["FormType"].astype(str).str.strip().str.upper()
    else:
        out["FormType"] = ""
    out["FormType"] = [
        _infer_form_type(period_end=pe, current=ft)
        for pe, ft in zip(coerce_series_naive(out["PeriodEnd"]), out["FormType"], strict=False)
    ]

    out["FilingDate"] = coerce_series_naive(out.get("FilingDate"))
    out["AcceptedAt"] = coerce_series_naive(out.get("AcceptedAt"))
    out["Source"] = out.get("Source", "yfinance")
    out["CollectedAt"] = out.get("CollectedAt", pd.Timestamp.now().isoformat())
    
    if "AvailabilityMethod" in out.columns:
        out["AvailabilityMethod"] = out["AvailabilityMethod"].astype(str)
    else:
        out["AvailabilityMethod"] = "missing"

    trading_days = _load_trading_days(market)
    q_lag = max(int(options.fallback_q_days), 0)
    k_lag = max(int(options.fallback_k_days), 0)

    available_values: list[pd.Timestamp | pd.NaT] = []
    method_values: list[str] = []
    for _, row in out.iterrows():
        filing = coerce_series_naive(pd.Series([row.get("FilingDate")])).iloc[0]
        accepted = coerce_series_naive(pd.Series([row.get("AcceptedAt")])).iloc[0]
        period_end = coerce_series_naive(pd.Series([row.get("PeriodEnd")])).iloc[0]
        form = _infer_form_type(period_end, row.get("FormType"))

        base = accepted.normalize() if pd.notna(accepted) else (filing.normalize() if pd.notna(filing) else pd.NaT)
        if pd.notna(base):
            if options.use_next_trading_day_availability:
                available_values.append(_next_trading_day(pd.Timestamp(base), trading_days))
                method_values.append("filed_next_trading_day")
            else:
                available_values.append(pd.Timestamp(base))
                method_values.append("filed")
            continue

        if not options.availability_fallback or pd.isna(period_end):
            available_values.append(pd.NaT)
            method_values.append("missing")
            continue

        lag_days = k_lag if "10-K" in form else q_lag
        fallback_date = (pd.Timestamp(period_end).normalize() + pd.Timedelta(days=lag_days)).normalize()
        if options.use_next_trading_day_availability:
            fallback_date = _next_trading_day(fallback_date, trading_days)
            method_values.append("fallback_next_trading_day")
        else:
            method_values.append("fallback")
        available_values.append(fallback_date)

    if "AvailableDate" in out.columns:
        out["AvailableDate"] = coerce_series_naive(out["AvailableDate"])
    else:
        out["AvailableDate"] = pd.NaT
    out["AvailableDate"] = coerce_series_naive(pd.Series(available_values)).fillna(out["AvailableDate"])
    out["AvailabilityMethod"] = [
        existing if str(existing).strip() and str(existing).strip().lower() != "missing" else computed
        for existing, computed in zip(out["AvailabilityMethod"], method_values, strict=False)
    ]
    out = out.loc[~out["PeriodEnd"].isna()].copy()
    out["PeriodEnd"] = coerce_series_naive(out["PeriodEnd"]).dt.normalize()
    out = out.sort_values(["PeriodEnd", "AvailableDate", "FilingDate", "AcceptedAt"])
    return out.reset_index(drop=True)


def _load_ticker_quarterly_merged(market: str, ticker: str) -> pd.DataFrame:
    _ensure_duckdb_available(market=market)
    from market_data.db_reader import load_financials_from_db

    try:
        df = load_financials_from_db(ticker, market=market)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load quarterly financials from DuckDB for ticker={sanitize_ticker(ticker)} market={market}."
        ) from exc
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _load_ticker_quarterly_statement(market: str, ticker: str, statement_type: str) -> pd.DataFrame:
    # DuckDB stores standardized wide quarterly facts in one table.
    # Keep statement_type API for backward compatibility by returning the same row set.
    _ = statement_type
    return _load_ticker_quarterly_merged(market=market, ticker=ticker)


def load_financials_quarterly(
    market: str,
    tickers: list[str],
    statement_type: str = "merged",
    *,
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
) -> pd.DataFrame:
    stmt = _normalize_statement_type(statement_type)
    _ensure_duckdb_available(market=market)
    options = FinancialsPitOptions(
        use_next_trading_day_availability=bool(use_next_trading_day_availability),
        availability_fallback=bool(availability_fallback),
        fallback_q_days=int(fallback_q_days),
        fallback_k_days=int(fallback_k_days),
    )
    rows: list[pd.DataFrame] = []
    for ticker in [str(t).strip().upper() for t in tickers if str(t).strip()]:
        if stmt == "merged":
            raw = _load_ticker_quarterly_merged(market=market, ticker=ticker)
        else:
            raw = _load_ticker_quarterly_statement(market=market, ticker=ticker, statement_type=stmt)
        if raw.empty:
            continue
        normalized = _ensure_pit_columns(raw, market=market, options=options)
        if normalized.empty:
            continue
        normalized["Ticker"] = ticker
        rows.append(normalized)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out = out.sort_values(["Ticker", "PeriodEnd", "AvailableDate", "FilingDate", "AcceptedAt"])
    return out.reset_index(drop=True)


def _pick_latest_row_asof(df: pd.DataFrame, asof_date: pd.Timestamp, prefer: str = "latest_filing") -> pd.Series | None:
    if df is None or df.empty:
        return None
    asof_naive = coerce_series_naive(pd.Series([asof_date])).iloc[0]
    if pd.isna(asof_naive):
        return None
    available = coerce_series_naive(df.get("AvailableDate"))
    eligible = df.loc[available <= asof_naive].copy()
    if eligible.empty:
        return None
    sort_cols = [c for c in ["PeriodEnd", "FilingDate", "AcceptedAt", "AvailableDate", "CollectedAt"] if c in eligible.columns]
    eligible = eligible.sort_values(sort_cols)
    latest_period = pd.to_datetime(eligible["PeriodEnd"], errors="coerce").max()
    latest = eligible.loc[pd.to_datetime(eligible["PeriodEnd"], errors="coerce") == latest_period].copy()
    if latest.empty:
        return None
    if str(prefer or "latest_filing").strip().lower() == "latest_filing":
        latest = latest.sort_values([c for c in ["FilingDate", "AcceptedAt", "AvailableDate", "CollectedAt"] if c in latest.columns])
    row = latest.iloc[-1]
    return row


def get_financials_asof(
    market: str,
    ticker: str,
    asof_date: str | pd.Timestamp,
    statement_type: str = "merged",
    prefer: str = "latest_filing",
    *,
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
) -> dict[str, Any] | None:
    asof_ts = pd.to_datetime(asof_date, errors="coerce")
    if pd.isna(asof_ts):
        return None
    frame = load_financials_quarterly(
        market=market,
        tickers=[ticker],
        statement_type=statement_type,
        use_next_trading_day_availability=use_next_trading_day_availability,
        availability_fallback=availability_fallback,
        fallback_q_days=fallback_q_days,
        fallback_k_days=fallback_k_days,
    )
    if frame.empty:
        return None
    row = _pick_latest_row_asof(frame, pd.Timestamp(asof_ts), prefer=prefer)
    if row is None:
        return None
    return {k: row[k] for k in row.index}


def get_financials_panel_asof(
    market: str,
    tickers: list[str],
    asof_date: str | pd.Timestamp,
    statement_type: str = "merged",
    prefer: str = "latest_filing",
    *,
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
) -> pd.DataFrame:
    asof_ts = pd.to_datetime(asof_date, errors="coerce")
    if pd.isna(asof_ts):
        return pd.DataFrame()
    asof_naive = coerce_series_naive(pd.Series([asof_ts])).iloc[0]
    if pd.isna(asof_naive):
        return pd.DataFrame()
    frame = load_financials_quarterly(
        market=market,
        tickers=tickers,
        statement_type=statement_type,
        use_next_trading_day_availability=use_next_trading_day_availability,
        availability_fallback=availability_fallback,
        fallback_q_days=fallback_q_days,
        fallback_k_days=fallback_k_days,
    )
    if frame.empty:
        return pd.DataFrame()
    frame["AvailableDate"] = coerce_series_naive(frame.get("AvailableDate"))
    frame = frame.loc[frame["AvailableDate"] <= asof_naive].copy()
    if frame.empty:
        return pd.DataFrame()

    sort_cols = [c for c in ["Ticker", "PeriodEnd", "FilingDate", "AcceptedAt", "AvailableDate", "CollectedAt"] if c in frame.columns]
    frame = frame.sort_values(sort_cols)
    latest_period = frame.groupby("Ticker")["PeriodEnd"].transform("max")
    latest = frame.loc[pd.to_datetime(frame["PeriodEnd"], errors="coerce") == pd.to_datetime(latest_period, errors="coerce")].copy()
    if latest.empty:
        return pd.DataFrame()
    if str(prefer or "latest_filing").strip().lower() == "latest_filing":
        latest = latest.sort_values([c for c in ["Ticker", "FilingDate", "AcceptedAt", "AvailableDate", "CollectedAt"] if c in latest.columns])
    out = latest.drop_duplicates(subset=["Ticker"], keep="last").set_index("Ticker")
    return out
