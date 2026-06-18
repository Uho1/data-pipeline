"""Bulk reader functions for price and financial data.

These functions keep the legacy reader interface while dispatching to the
active storage backend. When `MDL_STORAGE=parquet`, they use parquet-native
reads; otherwise they use DuckDB.

Usage (normal code does not call this directly; it is driven by
the bulk-preload path in factors.build_factor_panel):

    from market_data.db_reader import bulk_load_prices, bulk_load_financials_quarterly

    price_map  = bulk_load_prices(["AAPL", "MSFT"], market="us", start="2020-01-01")
    fin_map    = bulk_load_financials_quarterly(["AAPL", "MSFT"], market="us")
"""
from __future__ import annotations

import pandas as pd

from market_data import parquet_reader
from market_data.config import STORAGE_BACKEND
from market_data.db import get_connection
from market_data.db_prices import get_connection as get_prices_connection
from market_data.db_router import is_kr_market


def _use_parquet() -> bool:
    return STORAGE_BACKEND == "parquet"

# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def bulk_load_prices(
    tickers: list[str],
    market: str = "us",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load price DataFrames for multiple tickers in a single backend call.

    Returns:
        Dict  ticker (upper-case) → DataFrame with DatetimeIndex named "Date"
        and columns: Open, High, Low, Close, Adj Close, Volume, Dividends,
        Stock Splits  — same layout as load_price_dataframe().
    """
    if is_kr_market(market=market):
        from market_data import db_reader_kr

        return db_reader_kr.bulk_load_prices(tickers=tickers, market="kr", start=start, end=end)

    if _use_parquet():
        return parquet_reader.bulk_load_prices(tickers=tickers, market=market, start=start, end=end)

    if not tickers:
        return {}

    con = get_prices_connection()
    upper = [t.strip().upper() for t in tickers if t.strip()]
    if not upper:
        return {}

    ticker_list = ", ".join(f"'{t}'" for t in upper)
    clauses = [f"ticker IN ({ticker_list})", f"market = '{market.lower()}'"]
    if start:
        clauses.append(f"date >= '{str(start)[:10]}'")
    if end:
        clauses.append(f"date <= '{str(end)[:10]}'")
    where = " AND ".join(clauses)

    sql = f"""
        SELECT date, ticker, open, high, low, close, adj_close,
               volume, dividends, stock_splits
        FROM prices
        WHERE {where}
        ORDER BY ticker, date
    """
    df = con.execute(sql).df()
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"])

    result: dict[str, pd.DataFrame] = {}
    for ticker_val, grp in df.groupby("ticker", sort=False):
        out = grp.drop(columns=["ticker"]).copy()
        out = out.rename(columns={
            "date":         "Date",
            "open":         "Open",
            "high":         "High",
            "low":          "Low",
            "close":        "Close",
            "adj_close":    "Adj Close",
            "volume":       "Volume",
            "dividends":    "Dividends",
            "stock_splits": "Stock Splits",
        })
        out = out.set_index("Date").sort_index()
        result[str(ticker_val).upper()] = out
    return result


def bulk_load_price_close_frames(
    tickers: list[str],
    market: str = "us",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load only close/adjusted-close series for multiple tickers in one query."""
    if is_kr_market(market=market):
        from market_data import db_reader_kr

        return db_reader_kr.bulk_load_prices(tickers=tickers, market="kr", start=start, end=end)

    if _use_parquet():
        return parquet_reader.bulk_load_price_close_frames(tickers=tickers, market=market, start=start, end=end)

    if not tickers:
        return {}

    con = get_prices_connection()
    upper = [t.strip().upper() for t in tickers if t.strip()]
    if not upper:
        return {}

    ticker_list = ", ".join(f"'{t}'" for t in upper)
    clauses = [f"ticker IN ({ticker_list})", f"market = '{market.lower()}'"]
    if start:
        clauses.append(f"date >= '{str(start)[:10]}'")
    if end:
        clauses.append(f"date <= '{str(end)[:10]}'")
    where = " AND ".join(clauses)

    sql = f"""
        SELECT date, ticker, close, adj_close
        FROM prices
        WHERE {where}
        ORDER BY ticker, date
    """
    df = con.execute(sql).df()
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"])

    result: dict[str, pd.DataFrame] = {}
    for ticker_val, grp in df.groupby("ticker", sort=False):
        out = grp.drop(columns=["ticker"]).copy()
        out = out.rename(columns={"date": "Date", "close": "Close", "adj_close": "Adj Close"})
        out = out.set_index("Date").sort_index()
        keep_cols = [c for c in ("Adj Close", "Close") if c in out.columns and out[c].notna().any()]
        if not keep_cols:
            continue
        result[str(ticker_val).upper()] = out[keep_cols]
    return result


# ---------------------------------------------------------------------------
# Financials
# ---------------------------------------------------------------------------

def bulk_load_financials_quarterly(
    tickers: list[str],
    market: str = "us",
) -> dict[str, pd.DataFrame]:
    """Load quarterly financials for multiple tickers in a single backend call.

    Returns:
        Dict  ticker (upper-case) → DataFrame whose schema matches
        sec_companyfacts_quarterly.parquet (column names preserved).
        The 'symbol' column contains the ticker (upper-case string).
    """
    if is_kr_market(market=market):
        from market_data import db_reader_kr

        return db_reader_kr.bulk_load_financials_quarterly(tickers=tickers, market="kr")

    if _use_parquet():
        return parquet_reader.bulk_load_financials_quarterly(tickers=tickers, market=market)

    if not tickers:
        return {}

    con = get_connection()
    upper = [t.strip().upper() for t in tickers if t.strip()]
    if not upper:
        return {}

    ticker_list = ", ".join(f"'{t}'" for t in upper)
    sql = f"""
        SELECT *
        FROM financials_quarterly
        WHERE ticker IN ({ticker_list}) AND market = '{market.lower()}'
        ORDER BY ticker, "PeriodEnd"
    """
    df = con.execute(sql).df()
    if df.empty:
        return {}

    # Rename db column 'ticker' → 'symbol' to match original parquet convention
    if "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})

    # Coerce date columns to pandas datetime (DuckDB returns them as objects/date)
    for col in ("StatementDate", "PeriodEnd", "PeriodStart", "FilingDate", "AvailableDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    result: dict[str, pd.DataFrame] = {}
    for ticker_val, grp in df.groupby("symbol", sort=False):
        result[str(ticker_val).upper()] = grp.reset_index(drop=True)
    return result


def load_price_from_db(
    ticker: str,
    market: str | None = None,
) -> tuple[pd.DataFrame, str] | None:
    """Load price data for a single ticker from the active storage backend.

    Returns:
        (DataFrame with DatetimeIndex "Date", market_str)  or  None if not found.
        DataFrame columns match load_price_dataframe(): Open, High, Low, Close,
        Adj Close, Volume, Dividends, Stock Splits.
    """
    if is_kr_market(market=market, ticker=ticker):
        from market_data import db_reader_kr

        return db_reader_kr.load_price_from_db(ticker=ticker, market="kr")

    if _use_parquet():
        resolved_market = "us" if market in (None, "auto") else str(market).strip().lower()
        return parquet_reader.load_price(ticker=ticker, market=resolved_market)

    con = get_prices_connection()
    upper = ticker.strip().upper()
    clauses = [f"ticker = '{upper}'"]
    if market and market != "auto":
        clauses.append(f"market = '{market.lower()}'")
    where = " AND ".join(clauses)
    # Include market_cap if the column exists
    try:
        _cols = [c[0] for c in con.execute("DESCRIBE prices").fetchall()]
        _has_mcap = "market_cap" in _cols
    except Exception:
        _has_mcap = False
    _mcap_col = ", market_cap" if _has_mcap else ""
    sql = f"""
        SELECT date, ticker, market, open, high, low, close, adj_close,
               volume, dividends, stock_splits{_mcap_col}
        FROM prices
        WHERE {where}
        ORDER BY date
    """
    df = con.execute(sql).df()
    if df.empty:
        return None
    effective_market = str(df["market"].iloc[0])
    out = df.drop(columns=["ticker", "market"]).copy()
    _rename = {
        "date":         "Date",
        "open":         "Open",
        "high":         "High",
        "low":          "Low",
        "close":        "Close",
        "adj_close":    "Adj Close",
        "volume":       "Volume",
        "dividends":    "Dividends",
        "stock_splits": "Stock Splits",
    }
    if _has_mcap:
        _rename["market_cap"] = "MarketCap"
    out = out.rename(columns=_rename)
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.set_index("Date").sort_index()
    return out, effective_market


def load_financials_from_db(
    ticker: str,
    market: str = "us",
) -> pd.DataFrame | None:
    """Load quarterly financials for a single ticker from the active storage backend.

    Returns:
        DataFrame matching load_ticker_quarterly_cache() schema, or None if not found.
        The 'symbol' column contains the ticker (upper-case).
    """
    if is_kr_market(market=market, ticker=ticker):
        from market_data import db_reader_kr

        return db_reader_kr.load_financials_from_db(ticker=ticker, market="kr")

    if _use_parquet():
        resolved_market = "us" if market in (None, "auto") else str(market).strip().lower()
        df = parquet_reader.load_financials(ticker=ticker, market=resolved_market, include_extra=True)
        if df is None or df.empty:
            return None
        df = _drop_leading_fy_as_q4(df)
        return df.reset_index(drop=True)

    con = get_connection()
    upper = ticker.strip().upper()
    sql = f"""
        SELECT 
            q.*,
            e.owner_equity AS "Owner Equity",
            e.owner_net_income AS "Owner Net Income",
            e.common_stock AS "Common Stock",
            e.additional_paid_in_capital AS "Additional Paid In Capital",
            e.retained_earnings AS "Retained Earnings",
            e.aoci AS "AOCI",
            e.ppe AS "PPE",
            e.ppe_capex AS "PPE Capex",
            e.intangibles AS "Intangibles",
            e.intangible_capex AS "Intangible Capex",
            e.amortization AS "Amortization",
            e.other_gain AS "Other Gain",
            e.financial_gain AS "Financial Gain",
            e.equity_method_gain AS "Equity Method Gain",
            e.other_income AS "Other Income",
            e.other_expense AS "Other Expense",
            e.financial_income AS "Financial Income",
            e.financial_expense AS "Financial Expense",
            e.current_fin_assets AS "Current Fin Assets",
            e.non_current_fin_assets AS "Non Current Fin Assets",
            e.current_fin_liabilities AS "Current Fin Liabilities",
            e.non_current_fin_liabilities AS "Non Current Fin Liabilities"
        FROM financials_quarterly q
        LEFT JOIN financials_quarterly_extra e
          ON q.ticker = e.ticker
         AND q.market = e.market
         AND q."PeriodEnd" = e.period_end
         AND q."FormType" = e.form_type
         AND (q."FilingDate" = e.filing_date OR (q."FilingDate" IS NULL AND e.filing_date IS NULL))
        WHERE q.ticker = '{upper}' AND q.market = '{market.lower()}'
        ORDER BY q."PeriodEnd"
    """
    df = con.execute(sql).df()
    if df.empty:
        return None
    if "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    for col in ("StatementDate", "PeriodEnd", "PeriodStart", "FilingDate", "AvailableDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Fix: Remove FY rows masquerading as Q4 at start of series ---
    df = _drop_leading_fy_as_q4(df)

    return df.reset_index(drop=True)


def _drop_leading_fy_as_q4(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading FY-as-Q4 rows that contain annual totals."""
    if df.empty:
        return df
    ft_col = "FormType" if "FormType" in df.columns else None
    term_col = "term" if "term" in df.columns else None
    fq_col = "fiscal_quarter" if "fiscal_quarter" in df.columns else None
    rev_col = "Revenue" if "Revenue" in df.columns else None
    if not ft_col or (not term_col and not fq_col):
        return df

    drop_indices = []
    found_real_data = False
    for idx, row in df.iterrows():
        form = str(row.get(ft_col, "")).strip()
        term = str(row.get(term_col, "")).strip()
        fiscal_quarter = row.get(fq_col) if fq_col else None
        rev = row.get(rev_col) if rev_col else None
        if rev_col and (rev is None or pd.isna(rev)):
            continue
        is_q4 = term == "Q4" or term.endswith("Q4")
        if fiscal_quarter is not None and not pd.isna(fiscal_quarter):
            try:
                is_q4 = is_q4 or int(float(fiscal_quarter)) == 4
            except Exception:
                pass
        if form == "FY" and is_q4 and not found_real_data:
            drop_indices.append(idx)
        else:
            found_real_data = True
    if drop_indices:
        df = df.drop(index=drop_indices)
    return df


def load_derived_factors_from_db(
    ticker: str,
    market: str = "us",
    basis: str | None = None,
) -> pd.DataFrame | None:
    """Load materialized derived factors for a single ticker from the active storage backend."""
    if is_kr_market(market=market, ticker=ticker):
        from market_data import db_reader_kr

        return db_reader_kr.load_derived_factors_from_db(ticker=ticker, market="kr", basis=basis)

    if _use_parquet():
        resolved_market = "us" if market in (None, "auto") else str(market).strip().lower()
        return parquet_reader.load_derived_factors(ticker=ticker, market=resolved_market, basis=basis)

    con = get_connection()
    upper = ticker.strip().upper()
    clauses = [f"ticker = '{upper}'", f"market = '{market.lower()}'"]
    if basis:
        clauses.append(f"basis = '{str(basis).strip().lower()}'")
    where = " AND ".join(clauses)
    sql = f"""
        SELECT *
        FROM derived_factors_quarterly
        WHERE {where}
        ORDER BY basis, period_end, available_date
    """
    try:
        df = con.execute(sql).df()
    except Exception:
        return None
    if df.empty:
        return None
    for col in ("period_end", "available_date", "collected_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "basis" in df.columns:
        df["basis"] = df["basis"].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


def load_filings_from_db(
    ticker: str,
    market: str = "us",
) -> pd.DataFrame | None:
    """Load filing metadata rows for a single ticker from the active storage backend."""
    if is_kr_market(market=market, ticker=ticker):
        from market_data import db_reader_kr

        return db_reader_kr.load_filings_from_db(ticker=ticker, market="kr")

    if _use_parquet():
        resolved_market = "us" if market in (None, "auto") else str(market).strip().lower()
        return parquet_reader.load_filings(ticker=ticker, market=resolved_market)

    con = get_connection()
    upper = ticker.strip().upper()
    sql = f"""
        SELECT *
        FROM filings
        WHERE ticker = '{upper}' AND market = '{market.lower()}'
        ORDER BY filing_date DESC, accepted_at DESC
    """
    try:
        df = con.execute(sql).df()
    except Exception:
        return None
    if df.empty:
        return None
    for col in ("period_end", "report_date", "available_date", "filing_date", "accepted_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.reset_index(drop=True)
