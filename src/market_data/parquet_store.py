"""Parquet-based storage for market data — replaces DuckDB read/write.

All data is stored as Parquet files under DATA_DIR/raw/{market}/.
Provides upsert semantics: new data replaces existing rows for the same key.

Thread-safe via file-level locks.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from pyarrow import dataset as ds

logger = logging.getLogger(__name__)

from market_data.config import PARQUET_DIR

_RAW_DIR = PARQUET_DIR

_lock = threading.Lock()
_PARTITIONED_TABLES = {
    "prices",
    "financials_quarterly",
    "financials_quarterly_extra",
    "derived_factors_quarterly",
    "filings",
    "sec_facts_raw_normalized",
    "dart_financials_raw",
}
_TICKER_PARTITIONED_TABLES = {
    "financials_quarterly",
    "financials_quarterly_extra",
    "derived_factors_quarterly",
    "filings",
    "sec_facts_raw_normalized",
}
_CORP_CODE_PARTITIONED_TABLES = {
    "dart_financials_raw",
}
_PRICE_PARTITION_COLUMN = "year"
_CORP_CODE_PARTITION_COLUMN = "corp_code"
_REMOVED_NONOP_FINANCIAL_COLUMNS = {
    "Other Gain",
    "Financial Gain",
    "Equity Method Gain",
    "Other Income",
    "Other Expense",
    "Financial Income",
    "Financial Expense",
}
_REMOVED_KR_FINANCIAL_COLUMNS = {
    "R&D",
    "Trading Gain",
    "Trading Loss",
    "Investment Gain/Loss",
    "Insurance Finance Income",
    "Insurance Finance Expense",
    "Reinsurance Finance Income",
    "Reinsurance Finance Expense",
    "Other Operating Income Component",
}


# ---------------------------------------------------------------------------
# Core read/write helpers
# ---------------------------------------------------------------------------

def _parquet_path(market: str, table: str) -> Path:
    """Get canonical path for a parquet table."""
    if table in _PARTITIONED_TABLES:
        return _RAW_DIR / market / table
    return _RAW_DIR / market / f"{table}.parquet"


def _legacy_parquet_file_path(market: str, table: str) -> Path:
    return _RAW_DIR / market / f"{table}.parquet"


def _read_partitioned_table(path: Path, *, table: str) -> pd.DataFrame:
    kwargs: dict[str, Any] = {"format": "parquet"}
    if table == "prices":
        kwargs["partitioning"] = ds.partitioning(
            pa.schema(
                [
                    ("ticker", pa.string()),
                    (_PRICE_PARTITION_COLUMN, pa.int32()),
                ]
            ),
            flavor="hive",
        )
    elif table in _TICKER_PARTITIONED_TABLES:
        kwargs["partitioning"] = ds.partitioning(
            pa.schema(
                [
                    ("ticker", pa.string()),
                ]
            ),
            flavor="hive",
        )
    elif table in _CORP_CODE_PARTITIONED_TABLES:
        kwargs["partitioning"] = ds.partitioning(
            pa.schema(
                [
                    (_CORP_CODE_PARTITION_COLUMN, pa.string()),
                ]
            ),
            flavor="hive",
        )
    else:
        kwargs["partitioning"] = "hive"
    dataset = ds.dataset(str(path), **kwargs)
    df = dataset.to_table().to_pandas()
    if _PRICE_PARTITION_COLUMN in df.columns:
        df = df.drop(columns=[_PRICE_PARTITION_COLUMN])
    return df


def _price_ticker_partition_root(market: str, ticker: str) -> Path:
    return _parquet_path(market, "prices") / f"ticker={str(ticker)}"


def _price_year_partition_dir(market: str, ticker: str, year: int) -> Path:
    return _price_ticker_partition_root(market, ticker) / f"{_PRICE_PARTITION_COLUMN}={int(year)}"


def _ticker_partition_dir(market: str, table: str, ticker: str) -> Path:
    return _parquet_path(market, table) / f"ticker={str(ticker)}"


def _corp_code_partition_dir(market: str, table: str, corp_code: str) -> Path:
    return _parquet_path(market, table) / f"{_CORP_CODE_PARTITION_COLUMN}={str(corp_code)}"


def _legacy_price_top_level_years(market: str) -> set[int]:
    root = _parquet_path(market, "prices")
    if not root.exists() or not root.is_dir():
        return set()
    years: set[int] = set()
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = str(child.name)
        if not name.startswith(f"{_PRICE_PARTITION_COLUMN}="):
            continue
        try:
            years.add(int(name.split("=", 1)[1]))
        except Exception:
            continue
    return years


def _existing_price_years_for_ticker(market: str, ticker: str) -> set[int]:
    root = _parquet_path(market, "prices")
    if not root.exists() or not root.is_dir():
        return set()
    years: set[int] = set()
    ticker_root = _price_ticker_partition_root(market, ticker)
    if ticker_root.exists():
        for child in ticker_root.iterdir():
            if not child.is_dir():
                continue
            name = str(child.name)
            if not name.startswith(f"{_PRICE_PARTITION_COLUMN}="):
                continue
            try:
                years.add(int(name.split("=", 1)[1]))
            except Exception:
                continue
    target = f"ticker={str(ticker)}"
    for year in _legacy_price_top_level_years(market):
        legacy_year_dir = _parquet_path(market, "prices") / f"{_PRICE_PARTITION_COLUMN}={int(year)}"
        if not legacy_year_dir.exists():
            continue
        for child in legacy_year_dir.iterdir():
            if child.is_dir() and child.name == target:
                years.add(int(year))
                break
    return years


def _normalize_ticker_partition_frame(df: pd.DataFrame, ticker: str, market: str) -> pd.DataFrame:
    out = df.copy()
    if "ticker" not in out.columns:
        out["ticker"] = str(ticker)
    if "market" not in out.columns:
        out["market"] = str(market)
    out["ticker"] = out["ticker"].astype(str)
    out["market"] = out["market"].astype(str)
    return out


def _normalize_corp_code_partition_frame(df: pd.DataFrame, corp_code: str) -> pd.DataFrame:
    out = df.copy()
    if _CORP_CODE_PARTITION_COLUMN not in out.columns:
        out[_CORP_CODE_PARTITION_COLUMN] = str(corp_code)
    out[_CORP_CODE_PARTITION_COLUMN] = out[_CORP_CODE_PARTITION_COLUMN].astype(str).str.strip()
    out = out.loc[out[_CORP_CODE_PARTITION_COLUMN] != ""].copy()
    return out


def _normalize_price_frame_for_partition(df: pd.DataFrame, ticker: str, market: str) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and "date" not in out.columns and "Date" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name or "index": "date"})
    out = out.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
            "CollectedAt": "collected_at",
            "MarketCap": "market_cap",
            "Ticker": "ticker",
            "Market": "market",
        }
    )
    if "ticker" not in out.columns:
        out["ticker"] = str(ticker)
    if "market" not in out.columns:
        out["market"] = str(market)
    if "date" not in out.columns:
        logger.warning("Skipping legacy prices rows without a usable date column during migration")
        return pd.DataFrame()
    date_series = pd.to_datetime(out["date"], errors="coerce")
    out = out.loc[~date_series.isna()].copy()
    if out.empty:
        return out
    out["date"] = date_series.loc[out.index].dt.date
    out[_PRICE_PARTITION_COLUMN] = date_series.loc[out.index].dt.year.astype(int)
    return out


def _write_price_partition(market: str, year: int, ticker: str, df: pd.DataFrame) -> None:
    partition_dir = _price_year_partition_dir(market, ticker, year)
    if df.empty:
        if partition_dir.exists():
            shutil.rmtree(partition_dir, ignore_errors=True)
        return
    partition_dir.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if _PRICE_PARTITION_COLUMN in out.columns:
        out = out.drop(columns=[_PRICE_PARTITION_COLUMN])
    if "ticker" in out.columns:
        out = out.drop(columns=["ticker"])
    out.to_parquet(str(partition_dir / "data.parquet"), index=False)


def _write_ticker_partition(market: str, table: str, ticker: str, df: pd.DataFrame) -> None:
    partition_dir = _ticker_partition_dir(market, table, ticker)
    if df.empty:
        if partition_dir.exists():
            shutil.rmtree(partition_dir, ignore_errors=True)
        return
    partition_dir.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "ticker" in out.columns:
        out = out.drop(columns=["ticker"])
    out.to_parquet(str(partition_dir / "data.parquet"), index=False)


def _write_corp_code_partition(market: str, table: str, corp_code: str, df: pd.DataFrame) -> None:
    partition_dir = _corp_code_partition_dir(market, table, corp_code)
    if df.empty:
        if partition_dir.exists():
            shutil.rmtree(partition_dir, ignore_errors=True)
        return
    partition_dir.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if _CORP_CODE_PARTITION_COLUMN in out.columns:
        out = out.drop(columns=[_CORP_CODE_PARTITION_COLUMN])
    out.to_parquet(str(partition_dir / "data.parquet"), index=False)


def _migrate_legacy_prices_file_if_needed(market: str) -> None:
    dataset_root = _parquet_path(market, "prices")
    legacy_path = _legacy_parquet_file_path(market, "prices")
    if legacy_path.exists():
        try:
            legacy = pd.read_parquet(str(legacy_path))
        except Exception as exc:
            logger.warning("Failed to migrate legacy prices parquet %s: %s", legacy_path, exc)
            return

        normalized = _normalize_price_frame_for_partition(legacy, ticker="", market=market)
        if normalized.empty:
            legacy_path.unlink(missing_ok=True)
            return

        dataset_root.mkdir(parents=True, exist_ok=True)
        for (year, ticker_value), chunk in normalized.groupby([_PRICE_PARTITION_COLUMN, "ticker"], sort=True):
            _write_price_partition(market, int(year), str(ticker_value), chunk)
        legacy_path.unlink(missing_ok=True)
        logger.info("Migrated legacy prices parquet to partitioned dataset: %s", dataset_root)
        return

    if not dataset_root.exists() or not dataset_root.is_dir():
        return

    migrated_any = False
    for year in sorted(_legacy_price_top_level_years(market)):
        year_dir = _parquet_path(market, "prices") / f"{_PRICE_PARTITION_COLUMN}={int(year)}"
        legacy_files = list(year_dir.glob("*.parquet"))
        if not legacy_files:
            continue
        for legacy_file in legacy_files:
            try:
                legacy = pd.read_parquet(str(legacy_file))
            except Exception as exc:
                logger.warning("Failed to migrate legacy year partition %s: %s", legacy_file, exc)
                continue
            normalized = _normalize_price_frame_for_partition(legacy, ticker="", market=market)
            if normalized.empty:
                legacy_file.unlink(missing_ok=True)
                continue
            for (part_year, ticker_value), chunk in normalized.groupby([_PRICE_PARTITION_COLUMN, "ticker"], sort=True):
                _write_price_partition(market, int(part_year), str(ticker_value), chunk)
            legacy_file.unlink(missing_ok=True)
            migrated_any = True
    if migrated_any:
        logger.info("Migrated legacy year-first prices partitions to ticker/year dataset: %s", dataset_root)


def _migrate_legacy_ticker_partitioned_table_if_needed(market: str, table: str) -> None:
    dataset_root = _parquet_path(market, table)
    legacy_path = _legacy_parquet_file_path(market, table)
    if dataset_root.exists() or not legacy_path.exists():
        return

    try:
        legacy = pd.read_parquet(str(legacy_path))
    except Exception as exc:
        logger.warning("Failed to migrate legacy %s parquet %s: %s", table, legacy_path, exc)
        return

    if legacy is None or legacy.empty or "ticker" not in legacy.columns:
        legacy_path.unlink(missing_ok=True)
        return

    normalized = _normalize_ticker_partition_frame(legacy, ticker="", market=market)
    dataset_root.mkdir(parents=True, exist_ok=True)
    for ticker_value, chunk in normalized.groupby("ticker", sort=True):
        _write_ticker_partition(market, table, str(ticker_value), chunk)
    legacy_path.unlink(missing_ok=True)
    logger.info("Migrated legacy %s parquet to partitioned dataset: %s", table, dataset_root)


def _migrate_legacy_corp_code_partitioned_table_if_needed(market: str, table: str) -> None:
    dataset_root = _parquet_path(market, table)
    legacy_path = _legacy_parquet_file_path(market, table)
    if dataset_root.exists() or not legacy_path.exists():
        return

    try:
        legacy = pd.read_parquet(str(legacy_path))
    except Exception as exc:
        logger.warning("Failed to migrate legacy %s parquet %s: %s", table, legacy_path, exc)
        return

    if legacy is None or legacy.empty or _CORP_CODE_PARTITION_COLUMN not in legacy.columns:
        legacy_path.unlink(missing_ok=True)
        return

    normalized = _normalize_corp_code_partition_frame(legacy, corp_code="")
    if normalized.empty:
        legacy_path.unlink(missing_ok=True)
        return

    dataset_root.mkdir(parents=True, exist_ok=True)
    for corp_code_value, chunk in normalized.groupby(_CORP_CODE_PARTITION_COLUMN, sort=True):
        _write_corp_code_partition(market, table, str(corp_code_value), chunk)
    legacy_path.unlink(missing_ok=True)
    logger.info("Migrated legacy %s parquet to corp_code-partitioned dataset: %s", table, dataset_root)


def read_parquet(market: str, table: str) -> pd.DataFrame | None:
    """Read a parquet table. Returns None if file doesn't exist."""
    with _lock:
        if table == "prices":
            _migrate_legacy_prices_file_if_needed(market)
        elif table in _TICKER_PARTITIONED_TABLES:
            _migrate_legacy_ticker_partitioned_table_if_needed(market, table)
        elif table in _CORP_CODE_PARTITIONED_TABLES:
            _migrate_legacy_corp_code_partitioned_table_if_needed(market, table)
    path = _parquet_path(market, table)
    legacy_path = _legacy_parquet_file_path(market, table)
    candidate = path if path.exists() else legacy_path if legacy_path.exists() else None
    if candidate is None:
        return None
    try:
        if candidate.is_dir():
            return _read_partitioned_table(candidate, table=table)
        return pd.read_parquet(str(candidate))
    except Exception as e:
        logger.warning(f"Failed to read {candidate}: {e}")
        return None


def write_parquet(df: pd.DataFrame, market: str, table: str) -> int:
    """Write a DataFrame to parquet (full replace). Returns rows written."""
    path = _parquet_path(market, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        if table == "prices":
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            normalized = _normalize_price_frame_for_partition(df, ticker="", market=market)
            if normalized.empty:
                return 0
            for (year, ticker_value), chunk in normalized.groupby([_PRICE_PARTITION_COLUMN, "ticker"], sort=True):
                _write_price_partition(market, int(year), str(ticker_value), chunk)
            logger.debug(f"Wrote {len(df)} rows to {path}")
            return len(df)
        if table in _TICKER_PARTITIONED_TABLES:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            normalized = _normalize_ticker_partition_frame(df, ticker="", market=market)
            if normalized.empty or "ticker" not in normalized.columns:
                return 0
            for ticker_value, chunk in normalized.groupby("ticker", sort=True):
                _write_ticker_partition(market, table, str(ticker_value), chunk)
            logger.debug(f"Wrote {len(df)} rows to {path}")
            return len(df)
        if table in _CORP_CODE_PARTITIONED_TABLES:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            normalized = _normalize_corp_code_partition_frame(df, corp_code="")
            if normalized.empty or _CORP_CODE_PARTITION_COLUMN not in normalized.columns:
                return 0
            for corp_code_value, chunk in normalized.groupby(_CORP_CODE_PARTITION_COLUMN, sort=True):
                _write_corp_code_partition(market, table, str(corp_code_value), chunk)
            logger.debug(f"Wrote {len(df)} rows to {path}")
            return len(df)
        df.to_parquet(str(path), index=False)
    logger.debug(f"Wrote {len(df)} rows to {path}")
    return len(df)


def upsert_parquet(
    df: pd.DataFrame,
    market: str,
    table: str,
    key_cols: list[str],
) -> int:
    """Upsert rows into a parquet table.

    Removes existing rows matching key_cols, then appends new rows.
    Returns total rows after upsert.
    """
    path = _parquet_path(market, table)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        existing = None
        if path.exists():
            try:
                existing = pd.read_parquet(str(path))
            except Exception:
                existing = None

        if existing is not None and not existing.empty:
            # Build mask to remove matching rows
            # For each key combination in new data, remove from existing
            merged_key = df[key_cols].drop_duplicates()
            mask = pd.Series(True, index=existing.index)
            for _, row in merged_key.iterrows():
                row_mask = pd.Series(True, index=existing.index)
                for col in key_cols:
                    row_mask &= existing[col].astype(str) == str(row[col])
                mask &= ~row_mask
            existing = existing[mask]
            result = pd.concat([existing, df], ignore_index=True)
        else:
            result = df

        result.to_parquet(str(path), index=False)

    logger.debug(f"Upserted {len(df)} rows into {path} (total: {len(result)})")
    return len(result)


# ---------------------------------------------------------------------------
# Financial-specific functions (replaces db_writer_kr DuckDB operations)
# ---------------------------------------------------------------------------

def upsert_dart_financials_raw(df: pd.DataFrame, *, corp_code: str, market: str = "kr") -> int:
    """Store raw DART financial data to Parquet.

    Replaces all rows for the given corp_code, then appends new data.
    """
    if df is None or df.empty:
        return 0

    table = "dart_financials_raw"
    dataset_root = _parquet_path(market, table)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        _migrate_legacy_corp_code_partitioned_table_if_needed(market, table)
        normalized = _normalize_corp_code_partition_frame(df, corp_code=corp_code)
        if normalized.empty:
            return 0
        target = normalized.loc[normalized[_CORP_CODE_PARTITION_COLUMN].astype(str) == str(corp_code)].copy()
        if target.empty:
            return 0
        _write_corp_code_partition(market, table, str(corp_code), target)

    logger.debug(f"Upserted {len(target)} raw financial rows for corp_code={corp_code}")
    return len(target)


def upsert_financials_quarterly(df: pd.DataFrame, *, ticker: str, market: str = "kr") -> int:
    """Store materialized quarterly financials to Parquet."""
    return _upsert_by_ticker(df, ticker=ticker, market=market, table="financials_quarterly")


def upsert_filings(df: pd.DataFrame, *, ticker: str, market: str = "kr") -> int:
    """Store filing metadata to Parquet."""
    if df is None or df.empty:
        return 0
    return _upsert_by_ticker(df, ticker=ticker, market=market, table="filings")


def upsert_prices(df: pd.DataFrame, *, ticker: str, market: str = "kr") -> int:
    """Store price data to Parquet."""
    if df is None or df.empty:
        return 0

    dataset_root = _parquet_path(market, "prices")
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        _migrate_legacy_prices_file_if_needed(market)
        normalized = _normalize_price_frame_for_partition(df, ticker=ticker, market=market)
        if normalized.empty:
            return 0

        existing_years = _existing_price_years_for_ticker(market, str(ticker))
        new_years = set(normalized[_PRICE_PARTITION_COLUMN].astype(int).tolist())
        affected_years = existing_years | new_years
        for year in sorted(affected_years):
            new_rows = normalized.loc[normalized[_PRICE_PARTITION_COLUMN].astype(int) == int(year)].copy()
            _write_price_partition(market, int(year), str(ticker), new_rows)

    return len(df)


# ---------------------------------------------------------------------------
# Generic ticker-based upsert (US + KR)
# ---------------------------------------------------------------------------

def _upsert_by_ticker(df: pd.DataFrame, *, ticker: str, market: str, table: str) -> int:
    """Generic upsert: remove all rows for ticker+market, append new."""
    if df is None or df.empty:
        return 0
    # Ensure ticker/market columns exist
    if "ticker" not in df.columns:
        df = df.copy()
        df["ticker"] = str(ticker)
    if "market" not in df.columns:
        df = df.copy()
        df["market"] = str(market)
    path = _parquet_path(market, table)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        if table in _TICKER_PARTITIONED_TABLES:
            _migrate_legacy_ticker_partitioned_table_if_needed(market, table)
            normalized = _normalize_ticker_partition_frame(df, ticker=ticker, market=market)
            if normalized.empty:
                return 0
            target = normalized[
                (normalized["ticker"].astype(str) == str(ticker))
                & (normalized["market"].astype(str) == str(market))
            ].copy()
            if target.empty:
                return 0
            _write_ticker_partition(market, table, str(ticker), target)
            logger.debug(f"[parquet] Upserted {len(target)} rows for {ticker}/{market} -> {table}")
            return len(target)

        existing = None
        if path.exists():
            try:
                existing = pd.read_parquet(str(path))
            except Exception:
                existing = None

        if existing is not None and not existing.empty:
            mask = (existing["ticker"].astype(str) != str(ticker))
            if "market" in existing.columns:
                mask = mask | (existing["market"].astype(str) != str(market))
                mask = (existing["ticker"].astype(str) != str(ticker)) | (existing["market"].astype(str) != str(market))
            existing = existing[mask]
            result = pd.concat([existing, df], ignore_index=True)
        else:
            result = df
        result.to_parquet(str(path), index=False)

    logger.debug(f"[parquet] Upserted {len(df)} rows for {ticker}/{market} → {table}")
    return len(df)


def _replace_all(df: pd.DataFrame, *, market: str, table: str) -> int:
    """Full replace of a table."""
    if df is None or df.empty:
        return 0
    path = _parquet_path(market, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        df.to_parquet(str(path), index=False)
    logger.debug(f"[parquet] Replaced {table} with {len(df)} rows")
    return len(df)


# -- US Financials --

def upsert_financials(df: pd.DataFrame, *, ticker: str, market: str) -> int:
    if df is None or df.empty:
        return 0
    out = df.copy()
    drop_cols = set(_REMOVED_NONOP_FINANCIAL_COLUMNS)
    if str(market).strip().lower() == "kr":
        drop_cols |= _REMOVED_KR_FINANCIAL_COLUMNS
    existing_drop = [column for column in out.columns if column in drop_cols]
    if existing_drop:
        out = out.drop(columns=existing_drop, errors="ignore")
    return _upsert_by_ticker(out, ticker=ticker, market=market, table="financials_quarterly")


def upsert_sec_issuer_registry(df: pd.DataFrame, *, ticker: str, market: str) -> int:
    return _upsert_by_ticker(df, ticker=ticker, market=market, table="sec_issuer_registry")


def replace_sec_issuer_registry_bulk(df: pd.DataFrame) -> int:
    return _replace_all(df, market="us", table="sec_issuer_registry")


def upsert_sec_facts_raw_normalized(df: pd.DataFrame, *, ticker: str, market: str) -> int:
    return _upsert_by_ticker(df, ticker=ticker, market=market, table="sec_facts_raw_normalized")


def upsert_financials_extra(df: pd.DataFrame, *, ticker: str, market: str) -> int:
    return _upsert_by_ticker(df, ticker=ticker, market=market, table="financials_quarterly_extra")


def upsert_derived_factors(df: pd.DataFrame, *, ticker: str, market: str) -> int:
    return _upsert_by_ticker(df, ticker=ticker, market=market, table="derived_factors_quarterly")



# -- KR specific --

def upsert_investor_flows(df: pd.DataFrame, *, ticker: str, market: str = "kr") -> int:
    return _upsert_by_ticker(df, ticker=ticker, market=market, table="investor_flows")


def upsert_index_prices(df: pd.DataFrame, *, index_code: str, market: str = "kr") -> int:
    """Upsert index prices by index_code."""
    if df is None or df.empty:
        return 0
    path = _parquet_path(market, "index_prices")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        existing = None
        if path.exists():
            try:
                existing = pd.read_parquet(str(path))
            except Exception:
                existing = None
        if existing is not None and not existing.empty:
            existing = existing[existing["index_code"].astype(str) != str(index_code)]
            result = pd.concat([existing, df], ignore_index=True)
        else:
            result = df
        result.to_parquet(str(path), index=False)
    return len(df)


def replace_ticker_master(df: pd.DataFrame, market: str = "kr") -> int:
    return _replace_all(df, market=market, table="ticker_master")


def replace_dart_corp_master(df: pd.DataFrame, market: str = "kr") -> int:
    return _replace_all(df, market=market, table="dart_corp_master")


def replace_ksic_dim(df: pd.DataFrame, market: str = "kr") -> int:
    return _replace_all(df, market=market, table="ksic_dim")


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(ticker: str, market: str, payload: dict) -> None:
    """Save ingest checkpoint for a ticker."""
    import json
    import datetime
    row = pd.DataFrame([{
        "ticker": ticker,
        "market": market,
        "completed_at": datetime.datetime.utcnow().isoformat(),
        "payload": json.dumps(payload),
    }])
    _upsert_by_ticker(row, ticker=ticker, market=market, table="ingest_checkpoints")


def get_checkpoint(ticker: str, market: str) -> dict | None:
    """Get ingest checkpoint for a ticker."""
    import json
    df = read_parquet(market, "ingest_checkpoints")
    if df is None or df.empty:
        return None
    match = df[(df["ticker"] == ticker) & (df["market"] == market)]
    if match.empty:
        return None
    row = match.iloc[-1]
    try:
        return json.loads(row["payload"])
    except Exception:
        return None


def get_fresh_tickers(market: str, fresh_days: int) -> set[str]:
    """Get tickers with recent checkpoints."""
    import datetime
    df = read_parquet(market, "ingest_checkpoints")
    if df is None or df.empty:
        return set()
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=fresh_days)).isoformat()
    recent = df[df["completed_at"] >= cutoff]
    return set(recent["ticker"].tolist())


def get_latest_price_date(ticker: str, market: str) -> str | None:
    """Get the latest price date for a ticker."""
    df = read_parquet(market, "prices")
    if df is None or df.empty:
        return None
    match = df[df["ticker"].astype(str) == str(ticker)]
    if match.empty:
        return None
    dates = pd.to_datetime(match["date"])
    return dates.max().strftime("%Y-%m-%d")


def update_market_cap(ticker: str, market: str, market_cap: float, date: str | None = None) -> int:
    """Update market_cap for a ticker's latest row (or specific date)."""
    # For parquet, this is a no-op or handled during price upsert
    return 0


def get_financial_null_rate_summary(market: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Return null % per column for financials_quarterly."""
    df = read_parquet(market, "financials_quarterly")
    if df is None or df.empty:
        return pd.DataFrame()
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    null_pct = df.isnull().mean() * 100
    return null_pct.to_frame("null_pct").reset_index().rename(columns={"index": "column"})


def init_dirs(market: str) -> None:
    """Ensure raw directories exist for a market."""
    (_RAW_DIR / market).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reader functions (replaces db_reader_kr DuckDB reads)
# ---------------------------------------------------------------------------

def load_dart_financials_raw_all(market: str = "kr") -> pd.DataFrame:
    """Load all raw DART financials from Parquet."""
    df = read_parquet(market, "dart_financials_raw")
    if df is None:
        return pd.DataFrame()
    return df.sort_values(["corp_code", "bsns_year", "reprt_code"], ignore_index=True)


def load_filings_all(market: str = "kr") -> pd.DataFrame:
    """Load all filing metadata from Parquet."""
    df = read_parquet(market, "filings")
    if df is None:
        return pd.DataFrame()
    return df


def load_financials_quarterly(ticker: str, market: str = "kr") -> pd.DataFrame | None:
    """Load materialized quarterly financials for a single ticker."""
    df = read_parquet(market, "financials_quarterly")
    if df is None or df.empty:
        return None
    result = df[(df["ticker"] == ticker) & (df["market"] == market)].copy()
    if result.empty:
        return None
    return result.sort_values("PeriodEnd", ignore_index=True)


def load_prices(ticker: str, market: str = "kr") -> pd.DataFrame | None:
    """Load price data for a single ticker."""
    df = read_parquet(market, "prices")
    if df is None or df.empty:
        return None
    result = df[df["ticker"] == ticker].copy()
    if result.empty:
        return None
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values("date").set_index("date")
