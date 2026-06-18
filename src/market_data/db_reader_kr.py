"""Bulk reader functions for Korea market data."""
from __future__ import annotations

import pandas as pd

from market_data import parquet_reader
from market_data.config import STORAGE_BACKEND
from market_data.db_kr import get_connection
from market_data.db_kr_prices import get_connection as get_prices_connection
from market_data.db_router import normalize_kr_ticker


def _use_parquet() -> bool:
    return STORAGE_BACKEND == "parquet"


def bulk_load_prices(
    tickers: list[str],
    market: str = "kr",
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    if _use_parquet():
        return parquet_reader.bulk_load_prices(tickers=tickers, market=market, start=start, end=end)

    if not tickers:
        return {}
    con = get_prices_connection()
    upper = [normalize_kr_ticker(t) for t in tickers if str(t).strip()]
    if not upper:
        return {}
    ticker_list = ", ".join(f"'{ticker}'" for ticker in upper)
    clauses = [f"ticker IN ({ticker_list})", f"market = '{market.lower()}'"]
    if start:
        clauses.append(f"date >= '{str(start)[:10]}'")
    if end:
        clauses.append(f"date <= '{str(end)[:10]}'")
    where = " AND ".join(clauses)
    df = con.execute(
        f"""
        SELECT date, ticker, open, high, low, close, adj_close,
               volume, dividends, stock_splits
        FROM prices
        WHERE {where}
        ORDER BY ticker, date
        """
    ).df()
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"])
    result: dict[str, pd.DataFrame] = {}
    for ticker_value, grp in df.groupby("ticker", sort=False):
        out = grp.drop(columns=["ticker"]).rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adj_close": "Adj Close",
                "volume": "Volume",
                "dividends": "Dividends",
                "stock_splits": "Stock Splits",
            }
        )
        result[str(ticker_value)] = out.set_index("Date").sort_index()
    return result


def bulk_load_financials_quarterly(
    tickers: list[str],
    market: str = "kr",
) -> dict[str, pd.DataFrame]:
    if _use_parquet():
        return parquet_reader.bulk_load_financials_quarterly(tickers=tickers, market=market)

    if not tickers:
        return {}
    con = get_connection()
    upper = [normalize_kr_ticker(t) for t in tickers if str(t).strip()]
    if not upper:
        return {}
    ticker_list = ", ".join(f"'{ticker}'" for ticker in upper)
    df = con.execute(
        f"""
        SELECT *
        FROM financials_quarterly
        WHERE ticker IN ({ticker_list}) AND market = '{market.lower()}'
        ORDER BY ticker, "PeriodEnd"
        """
    ).df()
    if df.empty:
        return {}
    if "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    for column in ("StatementDate", "PeriodEnd", "PeriodStart", "FilingDate", "AvailableDate"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    result: dict[str, pd.DataFrame] = {}
    for ticker_value, grp in df.groupby("symbol", sort=False):
        result[str(ticker_value)] = grp.reset_index(drop=True)
    return result


def load_price_from_db(
    ticker: str,
    market: str | None = "kr",
) -> tuple[pd.DataFrame, str] | None:
    if _use_parquet():
        return parquet_reader.load_price(ticker=ticker, market=market or "kr")

    con = get_prices_connection()
    upper = normalize_kr_ticker(ticker)
    clauses = [f"ticker = '{upper}'"]
    if market and market != "auto":
        clauses.append(f"market = '{market.lower()}'")
    where = " AND ".join(clauses)
    # Include market_cap if the column exists in the table
    try:
        _cols = [c[0] for c in con.execute("DESCRIBE prices").fetchall()]
        _has_mcap = "market_cap" in _cols
    except Exception:
        _has_mcap = False
    _mcap_col = ", market_cap" if _has_mcap else ""
    df = con.execute(
        f"""
        SELECT date, ticker, market, open, high, low, close, adj_close,
               volume, dividends, stock_splits{_mcap_col}
        FROM prices
        WHERE {where}
        ORDER BY date
        """
    ).df()
    if df.empty:
        return None
    effective_market = str(df["market"].iloc[0])
    _rename = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj_close": "Adj Close",
        "volume": "Volume",
        "dividends": "Dividends",
        "stock_splits": "Stock Splits",
    }
    if _has_mcap:
        _rename["market_cap"] = "MarketCap"
    out = df.drop(columns=["ticker", "market"]).rename(columns=_rename)
    out["Date"] = pd.to_datetime(out["Date"])
    return out.set_index("Date").sort_index(), effective_market


def load_financials_from_db(
    ticker: str,
    market: str = "kr",
) -> pd.DataFrame | None:
    if _use_parquet():
        df = parquet_reader.load_financials(ticker=ticker, market=market, include_extra=False)
        if df is None or df.empty:
            return None
        df = _drop_leading_fy_as_q4(df)
        return df.reset_index(drop=True)

    con = get_connection()
    upper = normalize_kr_ticker(ticker)
    df = con.execute(
        f"""
        SELECT *
        FROM financials_quarterly
        WHERE ticker = '{upper}' AND market = '{market.lower()}'
        ORDER BY "PeriodEnd"
        """
    ).df()
    if df.empty:
        return None
    if "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    for column in ("StatementDate", "PeriodEnd", "PeriodStart", "FilingDate", "AvailableDate"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    # --- Fix: Remove FY rows masquerading as Q4 at the start of the series.
    # When the first data point has FormType='FY' but term='Q4', it contains
    # annual totals (not quarterly), causing chart spikes.
    df = _drop_leading_fy_as_q4(df)

    return df.reset_index(drop=True)


def _drop_leading_fy_as_q4(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading FY-as-Q4 rows that contain annual totals.

    Scans from the beginning, skipping NaN-Revenue rows, and drops any
    FY/Q4 rows encountered before the first genuine quarterly data.
    Also drops if the FY/Q4 value is >2.5x the next quarter's value.
    """
    if df.empty:
        return df
    ft_col = "FormType" if "FormType" in df.columns else None
    term_col = "term" if "term" in df.columns else None
    fq_col = "fiscal_quarter" if "fiscal_quarter" in df.columns else None
    rev_col = "Revenue" if "Revenue" in df.columns else None
    if not ft_col or (not term_col and not fq_col):
        return df

    def _row_anchor_value(row: pd.Series) -> float | None:
        for col in ("Revenue", "Operating Income", "Net Income"):
            if col in df.columns:
                value = pd.to_numeric(row.get(col), errors="coerce")
                if pd.notna(value):
                    return float(abs(value))
        return None

    def _should_drop_leading_fy_q4(position: int, row: pd.Series) -> bool:
        form = str(row.get(ft_col, "")).strip()
        term = str(row.get(term_col, "")).strip()
        fiscal_quarter = row.get(fq_col) if fq_col else None
        is_q4 = term == "Q4" or term.endswith("Q4")
        if fiscal_quarter is not None and not pd.isna(fiscal_quarter):
            try:
                is_q4 = is_q4 or int(float(fiscal_quarter)) == 4
            except Exception:
                pass
        if form != "FY" or not is_q4:
            return False

        current_anchor = _row_anchor_value(row)
        if current_anchor is None:
            return False

        tail = df.iloc[position + 1 :].copy()
        if tail.empty:
            return False
        next_anchor = None
        for _, next_row in tail.iterrows():
            next_anchor = _row_anchor_value(next_row)
            if next_anchor is not None:
                break
        if next_anchor is None or next_anchor <= 0:
            return False
        return current_anchor > next_anchor * 2.5

    drop_indices = []
    found_real_data = False

    for pos, (idx, row) in enumerate(df.iterrows()):
        if _should_drop_leading_fy_q4(pos, row) and not found_real_data:
            drop_indices.append(idx)
            continue
        found_real_data = True

    if drop_indices:
        df = df.drop(index=drop_indices)
    return df


def load_derived_factors_from_db(
    ticker: str,
    market: str = "kr",
    basis: str | None = None,
) -> pd.DataFrame | None:
    _ = (ticker, market, basis)
    return None


def load_filings_from_db(
    ticker: str,
    market: str = "kr",
) -> pd.DataFrame | None:
    if _use_parquet():
        return parquet_reader.load_filings(ticker=ticker, market=market)

    con = get_connection()
    upper = normalize_kr_ticker(ticker)
    df = con.execute(
        f"""
        SELECT *
        FROM filings
        WHERE ticker = '{upper}' AND market = '{market.lower()}'
        ORDER BY filing_date DESC, accession DESC
        """
    ).df()
    if df.empty:
        return None
    for column in ("period_end", "filing_date", "available_date", "accepted_at"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df.reset_index(drop=True)


def load_ticker_master_from_db(ticker: str) -> pd.DataFrame | None:
    if _use_parquet():
        return parquet_reader.load_ticker_master(ticker=ticker, market="kr")

    con = get_prices_connection()
    df = con.execute(
        "SELECT * FROM ticker_master WHERE ticker = ? LIMIT 1",
        [normalize_kr_ticker(ticker)],
    ).df()
    if df.empty:
        return None
    return df.reset_index(drop=True)


def load_ticker_master_all() -> pd.DataFrame:
    if _use_parquet():
        return parquet_reader.load_ticker_master_all(market="kr")

    con = get_prices_connection()
    return con.execute("SELECT * FROM ticker_master ORDER BY market_tier, ticker").df()


def load_dart_corp_master_all() -> pd.DataFrame:
    if _use_parquet():
        return parquet_reader.load_dart_corp_master_all(market="kr")

    con = get_connection()
    return con.execute("SELECT * FROM dart_corp_master ORDER BY corp_code").df()


def load_ksic_dim_all() -> pd.DataFrame:
    if _use_parquet():
        return parquet_reader.load_ksic_dim_all(market="kr")

    con = get_connection()
    return con.execute("SELECT * FROM ksic_dim ORDER BY depth, ksic_code").df()


def load_dart_financials_raw_all() -> pd.DataFrame:
    if _use_parquet():
        return parquet_reader.load_dart_financials_raw_all(market="kr")

    con = get_connection()
    return con.execute("SELECT * FROM dart_financials_raw ORDER BY ticker, bsns_year, reprt_code, account_key").df()


def load_dart_financials_raw_for_ticker(ticker: str) -> pd.DataFrame:
    if _use_parquet():
        return parquet_reader.load_dart_financials_raw_for_ticker(ticker=ticker, market="kr")

    con = get_connection()
    return con.execute(
        """
        SELECT *
        FROM dart_financials_raw
        WHERE ticker = ?
        ORDER BY bsns_year, reprt_code, account_key
        """,
        [normalize_kr_ticker(ticker)],
    ).df()


def load_dart_financials_raw_for_tickers(tickers: list[str]) -> pd.DataFrame:
    if _use_parquet():
        return parquet_reader.load_dart_financials_raw_for_tickers(tickers=tickers, market="kr")

    if not tickers:
        return pd.DataFrame()
    normalized = [normalize_kr_ticker(t) for t in tickers if str(t).strip()]
    if not normalized:
        return pd.DataFrame()
    placeholders = ", ".join(["?"] * len(normalized))
    con = get_connection()
    return con.execute(
        f"""
        SELECT *
        FROM dart_financials_raw
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, bsns_year, reprt_code, account_key
        """,
        normalized,
    ).df()


def load_capacity_ingest_done_tickers(market: str = "kr") -> set[str]:
    """Return tickers that have already been processed for capacity ingest."""
    try:
        con = get_connection()
        rows = con.execute(
            "SELECT ticker FROM capacity_ingest_tickers WHERE market = ?",
            [str(market).strip().lower()],
        ).fetchall()
        return {str(r[0]) for r in rows if r[0]}
    except Exception:
        return set()


def load_capacity_production_from_db(
    ticker: str,
    market: str = "kr",
) -> pd.DataFrame | None:
    """Load capacity/production/utilization data for one ticker."""
    con = get_connection()
    upper = normalize_kr_ticker(ticker)
    try:
        df = con.execute(
            """
            SELECT *
            FROM capacity_production_quarterly
            WHERE ticker = ? AND market = ?
            ORDER BY period_end DESC, section, product_name
            """,
            [upper, str(market).strip().lower()],
        ).df()
    except Exception:
        return None
    if df.empty:
        return None
    for col in ("period_end", "available_date", "filing_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_filings_all() -> pd.DataFrame:
    if _use_parquet():
        return parquet_reader.load_filings_all(market="kr")

    con = get_connection()
    return con.execute("SELECT * FROM filings ORDER BY ticker, filing_date, accession").df()


def load_filings_for_tickers(tickers: list[str]) -> pd.DataFrame:
    if _use_parquet():
        return parquet_reader.load_filings_for_tickers(tickers=tickers, market="kr")

    if not tickers:
        return pd.DataFrame()
    normalized = [normalize_kr_ticker(t) for t in tickers if str(t).strip()]
    if not normalized:
        return pd.DataFrame()
    placeholders = ", ".join(["?"] * len(normalized))
    con = get_connection()
    return con.execute(
        f"""
        SELECT *
        FROM filings
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, filing_date, accession
        """,
        normalized,
    ).df()


def load_investor_flows_from_db(
    ticker: str,
    market: str = "kr",
) -> pd.DataFrame | None:
    if _use_parquet():
        return parquet_reader.load_investor_flows(ticker=ticker, market=market)

    con = get_connection()
    df = con.execute(
        """
        SELECT *
        FROM investor_flows
        WHERE ticker = ? AND market = ?
        ORDER BY date, investor_type
        """,
        [normalize_kr_ticker(ticker), str(market).strip().lower()],
    ).df()
    if df.empty:
        return None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.reset_index(drop=True)


def load_index_prices_from_db(index_code: str) -> pd.DataFrame | None:
    if _use_parquet():
        return parquet_reader.load_index_prices(index_code=index_code, market="kr")

    con = get_prices_connection()
    df = con.execute(
        """
        SELECT *
        FROM index_prices
        WHERE index_code = ?
        ORDER BY date
        """,
        [str(index_code).strip()],
    ).df()
    if df.empty:
        return None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.reset_index(drop=True)
