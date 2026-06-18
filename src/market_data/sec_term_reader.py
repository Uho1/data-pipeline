from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_data.config import DATA_DIR
from market_data.sec_financials import SEC_DEFAULT_START_DATE, SEC_EXTRACTOR_VERSION, fetch_sec_quarterly_history
from market_data.utils import ensure_dir, now_utc_iso, sanitize_ticker

RAW_TERMS_DIR = DATA_DIR / "sec_term_cache" / "raw_companyfacts"
DERIVED_TICKER_DIR = DATA_DIR / "sec_term_cache" / "ticker_quarterly"

CANONICAL_COLUMN_ALIASES: dict[str, list[str]] = {
    "Revenue": ["Revenue", "Total Revenue", "Operating Revenue"],
    "COGS": ["COGS", "Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "SG&A": [
        "SG&A",
        "Selling General And Administration",
        "Selling And Marketing Expense",
        "General And Administrative Expense",
    ],
    "Operating Income": ["Operating Income", "EBIT", "Total Operating Income As Reported"],
    "Pretax Income": ["Pretax Income", "Pre Tax Income"],
    "Tax": ["Tax", "Tax Expense", "Income Tax Expense"],
    "Net Income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
    ],
    "net_income_common": [
        "net_income_common",
        "Net Income Common",
        "Net Income Common Stockholders",
        "Net Income",
    ],
    "EPS": ["EPS", "Diluted EPS", "Basic EPS"],
    "diluted_eps": ["diluted_eps", "Diluted EPS", "EPS"],
    "Shares": ["Shares", "Ordinary Shares Number", "Diluted Average Shares", "Basic Average Shares", "Share Issued"],
    "diluted_shares": ["diluted_shares", "Diluted Shares", "Shares", "Diluted Average Shares"],
    "basic_shares": ["basic_shares", "Basic Shares", "Shares", "Basic Average Shares"],
    "Total Liabilities": [
        "Total Liabilities",
        "Total Liabilities Net Minority Interest",
        "Current Liabilities",
    ],
    "Shareholders Equity": [
        "Shareholders Equity",
        "Stockholders Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest",
    ],
    "Total Assets": ["Total Assets", "Total Assets As Reported", "Total Assets Gross Minority Interest"],
    "Price": ["Price"],
    "end_date": ["end_date", "StatementDate"],
    "eps_source": ["eps_source"],
    "Operating Cash Flow": [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Net Cash Provided By Operating Activities",
    ],
    "Investing Cash Flow": [
        "Investing Cash Flow",
        "Cash Flow From Continuing Investing Activities",
        "Net Cash Provided By Investing Activities",
    ],
    "Financing Cash Flow": [
        "Financing Cash Flow",
        "Cash Flow From Continuing Financing Activities",
        "Net Cash Provided By Financing Activities",
    ],
    "Capital Expenditure": ["Capital Expenditure", "Capital Expenditure Reported"],
}

TICKER_CACHE_COLUMNS = [
    "symbol",
    "term",
    "end_date",
    "PeriodStart",
    "PeriodEnd",
    "FormType",
    "FilingDate",
    "AcceptedAt",
    "AvailableDate",
    "AvailabilityMethod",
    "Revenue",
    "COGS",
    "SG&A",
    "Operating Income",
    "Pretax Income",
    "Tax",
    "Net Income",
    "net_income_common",
    "EPS",
    "diluted_eps",
    "Shares",
    "diluted_shares",
    "basic_shares",
    "eps_source",
    "Total Assets",
    "Total Liabilities",
    "Shareholders Equity",
    "Price",
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Capital Expenditure",
    "sector",
    "industry",
    "name",
    "avg_volume",
    "RequestedStart",
    "ExtractorVersion",
]

REQUIRED_CACHE_COLUMNS = [
    "Revenue",
    "COGS",
    "SG&A",
    "Operating Income",
    "Pretax Income",
    "Tax",
    "Net Income",
    "Total Liabilities",
    "Shareholders Equity",
    "Shares",
    "diluted_shares",
    "basic_shares",
    "diluted_eps",
    "net_income_common",
    "end_date",
    "eps_source",
    "Price",
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Capital Expenditure",
]


def _normalize_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for canonical, candidates in CANONICAL_COLUMN_ALIASES.items():
        selected = None
        for cand in candidates:
            if cand in out.columns and out[cand].notna().sum() > 0:
                selected = out[cand]
                break
        if selected is None:
            if canonical not in out.columns:
                out[canonical] = pd.NA
            continue
        if canonical not in out.columns or out[canonical].notna().sum() == 0:
            out[canonical] = selected
    return out


def _standardize_cache(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=TICKER_CACHE_COLUMNS + ["StatementDate", "CollectedAt", "Source"])

    out = _normalize_alias_columns(df.copy())
    out["symbol"] = str(symbol).strip().upper()

    if "StatementDate" in out.columns:
        out["StatementDate"] = pd.to_datetime(out["StatementDate"], errors="coerce")
    else:
        out["StatementDate"] = pd.to_datetime(pd.Series(pd.NA, index=out.index), errors="coerce")
    if out["StatementDate"].isna().all() and "end_date" in out.columns:
        out["StatementDate"] = pd.to_datetime(out["end_date"], errors="coerce")
    if "end_date" in out.columns:
        out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    else:
        out["end_date"] = out["StatementDate"]

    if "term" in out.columns:
        out["term"] = out["term"].astype(str)
    else:
        out["term"] = out["StatementDate"].map(
            lambda dt: f"{dt.year}Q{((int(dt.month) - 1) // 3) + 1}" if pd.notna(dt) else ""
        )

    if "RequestedStart" in out.columns:
        out["RequestedStart"] = pd.to_datetime(out["RequestedStart"], errors="coerce")
    else:
        out["RequestedStart"] = pd.NaT
    if "ExtractorVersion" in out.columns:
        out["ExtractorVersion"] = pd.to_numeric(out["ExtractorVersion"], errors="coerce")
    else:
        out["ExtractorVersion"] = pd.NA
    if "eps_source" not in out.columns:
        out["eps_source"] = "none"
    out["eps_source"] = out["eps_source"].astype(str).replace({"<NA>": "none", "nan": "none"})

    out = out.loc[~out["StatementDate"].isna()]
    out = out.loc[out["StatementDate"] >= SEC_DEFAULT_START_DATE]

    for col in TICKER_CACHE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["CollectedAt"] = out.get("CollectedAt", now_utc_iso())
    out["Source"] = out.get("Source", "sec_companyfacts")

    keep = TICKER_CACHE_COLUMNS + ["StatementDate", "CollectedAt", "Source"]
    out = out[keep]
    out = out.sort_values(["StatementDate", "term"]).drop_duplicates(subset=["term"], keep="last")
    return out.reset_index(drop=True)


def rebuild_ticker_quarterly_cache(
    ticker: str,
    raw_dir: Path = RAW_TERMS_DIR,
    derived_dir: Path = DERIVED_TICKER_DIR,
    offline_mode: bool = False,
) -> pd.DataFrame:
    symbol = str(ticker).strip().upper()
    if not symbol:
        return pd.DataFrame()

    ensure_dir(raw_dir)
    ensure_dir(derived_dir)

    out_path = derived_dir / f"{sanitize_ticker(symbol)}.parquet"
    try:
        fresh = fetch_sec_quarterly_history(
            ticker=symbol,
            market="us",
            start=SEC_DEFAULT_START_DATE,
            force_refresh=False,
            raw_cache_dir=raw_dir,
            ticker_cache_dir=derived_dir,
            offline_mode=offline_mode,
        )
        standardized = _standardize_cache(fresh, symbol=symbol)
        # Write to DuckDB (primary) and parquet cache (fallback)
        try:
            from market_data import db_writer
            db_writer.init_schema()
            db_writer.upsert_financials(fresh, symbol, "us")
        except Exception:
            pass
        standardized.to_parquet(out_path, index=False)
        return standardized
    except Exception:
        if out_path.exists():
            try:
                cached = pd.read_parquet(out_path)
                return _standardize_cache(cached, symbol=symbol)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()


def _normalize_df_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    date_cols = [
        "StatementDate",
        "PeriodEnd",
        "available_date",
        "source_filing_date",
        "source_acceptance_datetime",
        "effective_date",
        "announcement_date",
        "valid_from",
        "valid_to",
    ]
    for col in date_cols:
        if col in out.columns:
            s = pd.to_datetime(out[col], errors="coerce")
            if hasattr(s.dt, "tz") and s.dt.tz is not None:
                s = s.dt.tz_localize(None)
            out[col] = s.dt.normalize()
    return out


def load_ticker_quarterly_cache(
    ticker: str,
    raw_dir: Path = RAW_TERMS_DIR,
    derived_dir: Path = DERIVED_TICKER_DIR,
    rebuild_if_stale: bool = True,
) -> pd.DataFrame:
    symbol = str(ticker).strip().upper()
    if not symbol:
        return pd.DataFrame()

    # Try DuckDB first if available
    try:
        from market_data.db import db_available
        if db_available():
            from market_data.db_reader import load_financials_from_db
            db_df = load_financials_from_db(symbol, market="us")
            if db_df is not None and not db_df.empty:
                return _normalize_df_dates(db_df)
    except Exception:
        pass

    ensure_dir(derived_dir)
    cache_path = derived_dir / f"{sanitize_ticker(symbol)}.parquet"

    if not cache_path.exists():
        if not rebuild_if_stale:
            return pd.DataFrame()
        out = rebuild_ticker_quarterly_cache(
            symbol, raw_dir=raw_dir, derived_dir=derived_dir, offline_mode=not rebuild_if_stale
        )
        return _normalize_df_dates(out)

    try:
        df = pd.read_parquet(cache_path)
        return _normalize_df_dates(df)
    except Exception:
        if not rebuild_if_stale:
            return pd.DataFrame()
        out = rebuild_ticker_quarterly_cache(
            symbol, raw_dir=raw_dir, derived_dir=derived_dir, offline_mode=not rebuild_if_stale
        )
        return _normalize_df_dates(out)

    if df is None or df.empty:
        return pd.DataFrame()

    out = _standardize_cache(df, symbol=symbol)
    if rebuild_if_stale:
        missing = [c for c in REQUIRED_CACHE_COLUMNS if c not in out.columns]
        needs_start_coverage = False
        needs_version_upgrade = False
        if not out.empty:
            requested_start = pd.NaT
            if "RequestedStart" in out.columns:
                requested = pd.to_datetime(out["RequestedStart"], errors="coerce").dropna()
                if not requested.empty:
                    requested_start = pd.Timestamp(requested.min()).normalize()
            if pd.notna(requested_start):
                needs_start_coverage = requested_start > SEC_DEFAULT_START_DATE
            else:
                earliest = pd.to_datetime(out["StatementDate"], errors="coerce").dropna()
                if not earliest.empty:
                    needs_start_coverage = pd.Timestamp(earliest.min()).normalize() > SEC_DEFAULT_START_DATE

            versions = pd.to_numeric(out.get("ExtractorVersion"), errors="coerce").dropna()
            if versions.empty:
                needs_version_upgrade = True
            else:
                needs_version_upgrade = int(versions.max()) < SEC_EXTRACTOR_VERSION

        if missing or needs_start_coverage or needs_version_upgrade:
            rebuilt = rebuild_ticker_quarterly_cache(
                symbol, raw_dir=raw_dir, derived_dir=derived_dir, offline_mode=not rebuild_if_stale
            )
            if not rebuilt.empty:
                return rebuilt

    return out.reset_index(drop=True)
