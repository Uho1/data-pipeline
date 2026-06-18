from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_data.config import DATA_DIR
from market_data.sec_financials import SEC_DEFAULT_START_DATE, fetch_sec_quarterly_history
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
    "Net Income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
    ],
    "EPS": ["EPS", "Diluted EPS", "Basic EPS"],
    "Shares": ["Shares", "Ordinary Shares Number", "Diluted Average Shares", "Basic Average Shares", "Share Issued"],
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
    "Revenue",
    "COGS",
    "SG&A",
    "Operating Income",
    "Net Income",
    "EPS",
    "Shares",
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
]

REQUIRED_CACHE_COLUMNS = [
    "Revenue",
    "COGS",
    "SG&A",
    "Operating Income",
    "Net Income",
    "Total Liabilities",
    "Shareholders Equity",
    "Shares",
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

    if "term" in out.columns:
        out["term"] = out["term"].astype(str)
    else:
        out["term"] = out["StatementDate"].map(
            lambda dt: f"{dt.year}Q{((int(dt.month) - 1) // 3) + 1}" if pd.notna(dt) else ""
        )

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
        )
        standardized = _standardize_cache(fresh, symbol=symbol)
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


def load_ticker_quarterly_cache(
    ticker: str,
    raw_dir: Path = RAW_TERMS_DIR,
    derived_dir: Path = DERIVED_TICKER_DIR,
    rebuild_if_stale: bool = True,
) -> pd.DataFrame:
    symbol = str(ticker).strip().upper()
    if not symbol:
        return pd.DataFrame()

    ensure_dir(derived_dir)
    cache_path = derived_dir / f"{sanitize_ticker(symbol)}.parquet"

    if not cache_path.exists():
        return rebuild_ticker_quarterly_cache(symbol, raw_dir=raw_dir, derived_dir=derived_dir)

    try:
        df = pd.read_parquet(cache_path)
    except Exception:
        return rebuild_ticker_quarterly_cache(symbol, raw_dir=raw_dir, derived_dir=derived_dir)

    if df is None or df.empty:
        return pd.DataFrame()

    out = _standardize_cache(df, symbol=symbol)
    if rebuild_if_stale:
        missing = [c for c in REQUIRED_CACHE_COLUMNS if c not in out.columns]
        if missing:
            rebuilt = rebuild_ticker_quarterly_cache(symbol, raw_dir=raw_dir, derived_dir=derived_dir)
            if not rebuilt.empty:
                return rebuilt

    return out.reset_index(drop=True)
