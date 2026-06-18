"""Export DuckDB tables to standalone Parquet snapshot files.

Usage:
    python -m market_data.export_parquet [--market kr|us|all]

Writes to data/parquet_export/{market}/ with one .parquet per table.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPORT_DIR = Path(os.environ.get("MDL_PARQUET_EXPORT_DIR", Path(os.environ.get("MDL_EXPORT_DIR", _REPO_ROOT / "data")) / "parquet_export"))

# Table configs: (db_file_env_key, default_db_path, tables_to_export)
_KR_MAIN_DB = os.environ.get("MDL_DATA_DIR", str(_REPO_ROOT / "data")) + "/market_data_kr.duckdb"
_KR_PRICES_DB = os.environ.get("MDL_DATA_DIR", str(_REPO_ROOT / "data")) + "/market_data_kr_prices.duckdb"
_US_MAIN_DB = os.environ.get("MDL_DATA_DIR", str(_REPO_ROOT / "data")) + "/market_data.duckdb"
_US_PRICES_DB = os.environ.get("MDL_DATA_DIR", str(_REPO_ROOT / "data")) + "/market_data_prices.duckdb"

_KR_EXPORTS = [
    (_KR_MAIN_DB, [
        "dart_financials_raw",
        "financials_quarterly",
        "filings",
        "dart_corp_master",
    ]),
    (_KR_PRICES_DB, [
        "prices",
        "ticker_master",
        "index_prices",
    ]),
]

_US_EXPORTS = [
    (_US_MAIN_DB, [
        "financials_quarterly",
        "filings",
        "entity_master",
    ]),
    (_US_PRICES_DB, [
        "prices",
        "sec_issuer_registry",
    ]),
]


_START_DATE: str = "2013-06-01"

# Tables with date columns for filtering
_DATE_COLUMNS = {
    "prices": "date",
    "dart_financials_raw": "period_end",
    "financials_quarterly": '"PeriodEnd"',
    "filings": "filing_date",
    "index_prices": "date",
}


def _export_tables(db_path: str, tables: list[str], out_dir: Path) -> int:
    """Export tables from a DuckDB file to Parquet. Returns count of exported tables."""
    import duckdb

    if not Path(db_path).exists():
        print(f"  [SKIP] DB not found: {db_path}")
        return 0

    con = duckdb.connect(db_path, read_only=True)
    exported = 0
    try:
        existing = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
        for table in tables:
            if table not in existing:
                print(f"  [SKIP] {table} (not in DB)")
                continue

            # Apply date filter if possible
            date_col = _DATE_COLUMNS.get(table)
            if date_col and _START_DATE:
                query = f'SELECT * FROM "{table}" WHERE {date_col} >= \'{_START_DATE}\''
            else:
                query = f'SELECT * FROM "{table}"'

            out_path = out_dir / f"{table}.parquet"
            con.execute(f"COPY ({query}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            size_mb = out_path.stat().st_size / (1024 * 1024)
            count = con.execute(f"SELECT count(*) FROM ({query})").fetchone()[0]
            print(f"  [OK] {table}: {count:,} rows → {out_path.name} ({size_mb:.1f} MB)")
            exported += 1
    finally:
        con.close()
    return exported


def export_market(market: str) -> None:
    """Export all tables for a market to Parquet."""
    out_dir = _EXPORT_DIR / market
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = _KR_EXPORTS if market == "kr" else _US_EXPORTS
    total = 0
    for db_path, tables in configs:
        print(f"[{market}] Exporting from {Path(db_path).name}...")
        total += _export_tables(db_path, tables, out_dir)
    print(f"[{market}] Done: {total} tables exported to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DuckDB tables to Parquet")
    parser.add_argument("--market", default="all", choices=["kr", "us", "all"])
    args = parser.parse_args()

    if args.market in ("kr", "all"):
        export_market("kr")
    if args.market in ("us", "all"):
        export_market("us")


if __name__ == "__main__":
    main()
