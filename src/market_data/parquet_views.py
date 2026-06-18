"""Register Parquet files as DuckDB views for transparent SQL access."""
from __future__ import annotations

import logging
from pathlib import Path

from market_data.config import PARQUET_DIR

log = logging.getLogger(__name__)

_KR_TICKER_PARTITIONED_TABLES = {
    "prices",
    "financials_quarterly",
    "financials_quarterly_extra",
    "derived_factors_quarterly",
    "filings",
    "sec_facts_raw_normalized",
}


def _classify_table(table_name: str, market: str) -> str:
    """Classify a table as 'prices' or 'main' db_type."""
    _ = market
    prices_tables = {
        "prices",
        "sec_issuer_registry",
        "ticker_market_cap",
        "ticker_master",
        "index_prices",
    }
    return "prices" if table_name in prices_tables else "main"


def register_parquet_views(conn, market: str, db_type: str) -> None:
    """Register Parquet files as views on an in-memory DuckDB connection."""
    base = PARQUET_DIR / market
    if not base.exists():
        log.warning("Parquet directory not found: %s", base)
        return

    registered = 0
    for item in base.iterdir():
        if item.is_file() and item.suffix == ".parquet":
            table_name = item.stem
            if _classify_table(table_name, market) != db_type:
                continue
            _create_view_single(conn, table_name, item)
            registered += 1
        elif item.is_dir():
            table_name = item.name
            if _classify_table(table_name, market) != db_type:
                continue
            _create_view_partitioned(conn, market, table_name, item)
            registered += 1

    log.info("Registered %d Parquet views for %s/%s", registered, market, db_type)


def _create_view_single(conn, table_name: str, parquet_path: Path) -> None:
    sql = f"""CREATE OR REPLACE VIEW "{table_name}" AS
              SELECT * FROM read_parquet('{parquet_path}')"""
    try:
        conn.execute(sql)
    except Exception as exc:
        log.warning("Failed to create view %s: %s", table_name, exc)


def _create_view_partitioned(conn, market: str, table_name: str, parquet_dir: Path) -> None:
    """Create a view over a hive-style partitioned Parquet directory."""
    if table_name == "prices":
        glob_pattern = str(parquet_dir / "*" / "*" / "*.parquet")
    else:
        glob_pattern = str(parquet_dir / "*" / "*.parquet")

    if table_name == "dart_financials_raw":
        sql = f"""CREATE OR REPLACE VIEW "{table_name}" AS
                  SELECT * EXCLUDE(year)
                  FROM read_parquet('{glob_pattern}', hive_partitioning=true)"""
    elif market == "kr" and table_name in _KR_TICKER_PARTITIONED_TABLES:
        excluded = "ticker, year" if table_name == "prices" else "ticker"
        sql = f"""CREATE OR REPLACE VIEW "{table_name}" AS
                  SELECT * EXCLUDE({excluded}),
                         LPAD(CAST(ticker AS VARCHAR), 6, '0') AS ticker
                  FROM read_parquet('{glob_pattern}', hive_partitioning=true)"""
    elif table_name == "prices":
        sql = f"""CREATE OR REPLACE VIEW "{table_name}" AS
                  SELECT * EXCLUDE(year)
                  FROM read_parquet('{glob_pattern}', hive_partitioning=true)"""
    else:
        sql = f"""CREATE OR REPLACE VIEW "{table_name}" AS
                  SELECT *
                  FROM read_parquet('{glob_pattern}', hive_partitioning=true)"""
    try:
        conn.execute(sql)
    except Exception as exc:
        log.warning("Failed to create partitioned view %s: %s", table_name, exc)
