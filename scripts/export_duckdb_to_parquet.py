#!/usr/bin/env python3
"""Export all DuckDB tables to Parquet files.

Usage:
    .venv/bin/python scripts/export_duckdb_to_parquet.py [--force]

Reads from 4 DuckDB files and writes to data/parquet/{us,kr}/ directory.
Prices tables are partitioned by year (hive style).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PARQUET_DIR = DATA_DIR / "parquet"

# DB configs: (market, db_type, duckdb_path)
DB_CONFIGS = [
    ("us", "prices", DATA_DIR / "market_data_prices.duckdb"),
    ("us", "main", DATA_DIR / "market_data.duckdb"),
    ("kr", "prices", DATA_DIR / "market_data_kr_prices.duckdb"),
    ("kr", "main", DATA_DIR / "market_data_kr.duckdb"),
]

# Tables to partition by year
PARTITION_BY_YEAR = {
    "prices": "date",  # partition column
}

# Large KR table to partition
KR_PARTITION_BY_YEAR = {
    "dart_financials_raw": "bsns_year",  # if column exists
}

# Skip these — they are views that will be recreated, not materialized
SKIP_IF_VIEW_EXISTS = set()  # We export materialized tables even if views exist


def export_db(market: str, db_type: str, db_path: Path, force: bool = False) -> dict:
    """Export all tables from one DuckDB file to Parquet."""
    if not db_path.exists():
        print(f"  SKIP {db_path} (not found)")
        return {}

    out_dir = PARQUET_DIR / market
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)

    # Get tables (not views) — for materialized data
    all_tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    views = {v[0] for v in con.execute(
        "SELECT view_name FROM duckdb_views() WHERE NOT internal"
    ).fetchall()}

    # Export tables that are real tables (exclude pure views)
    tables_to_export = [t for t in all_tables if t not in views]

    # Also export materialized view tables (tables that share name with views)
    # These contain pre-computed data we want to keep
    materialized = [t for t in all_tables if t in views]

    stats = {}
    for table in tables_to_export + materialized:
        count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if count == 0 and not force:
            print(f"  SKIP {market}/{table} (empty)")
            continue

        # Determine partitioning
        partition_col = None
        if table in PARTITION_BY_YEAR:
            partition_col = PARTITION_BY_YEAR[table]
        elif market == "kr" and table in KR_PARTITION_BY_YEAR:
            col = KR_PARTITION_BY_YEAR[table]
            # Check if column exists
            cols = [c[0] for c in con.execute(f'DESCRIBE "{table}"').fetchall()]
            if col in cols:
                partition_col = col

        if partition_col:
            rows = _export_partitioned(con, table, partition_col, out_dir, market)
        else:
            rows = _export_single(con, table, out_dir)

        stats[table] = rows
        print(f"  {market}/{table}: {rows:,} rows")

    con.close()
    return stats


def _export_single(con, table: str, out_dir: Path) -> int:
    """Export a table as a single Parquet file."""
    out_path = out_dir / f"{table}.parquet"

    # Use DuckDB's native COPY for efficiency
    con.execute(f"""
        COPY (SELECT * FROM "{table}")
        TO '{out_path}'
        (FORMAT PARQUET, COMPRESSION 'snappy')
    """)

    # Verify row count
    result = pq.read_metadata(str(out_path))
    return result.num_rows


def _export_partitioned(con, table: str, partition_col: str, out_dir: Path, market: str) -> int:
    """Export a table partitioned by year into hive-style directories."""
    table_dir = out_dir / table
    table_dir.mkdir(parents=True, exist_ok=True)

    # Get distinct years
    if partition_col == "date":
        years = con.execute(f"""
            SELECT DISTINCT EXTRACT(YEAR FROM "{partition_col}")::INTEGER AS yr
            FROM "{table}"
            WHERE "{partition_col}" IS NOT NULL
            ORDER BY yr
        """).fetchall()
    else:
        years = con.execute(f"""
            SELECT DISTINCT "{partition_col}"::INTEGER AS yr
            FROM "{table}"
            WHERE "{partition_col}" IS NOT NULL
            ORDER BY yr
        """).fetchall()

    total_rows = 0
    for (year,) in years:
        year_dir = table_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        out_path = year_dir / "data.parquet"

        if partition_col == "date":
            where = f"EXTRACT(YEAR FROM \"{partition_col}\") = {year}"
        else:
            where = f"\"{partition_col}\" = {year}"

        con.execute(f"""
            COPY (SELECT * FROM "{table}" WHERE {where})
            TO '{out_path}'
            (FORMAT PARQUET, COMPRESSION 'snappy')
        """)

        result = pq.read_metadata(str(out_path))
        total_rows += result.num_rows

    return total_rows


def main():
    parser = argparse.ArgumentParser(description="Export DuckDB tables to Parquet")
    parser.add_argument("--force", action="store_true", help="Export empty tables too")
    args = parser.parse_args()

    print(f"Exporting to {PARQUET_DIR}/")
    print()

    total_tables = 0
    total_rows = 0
    t0 = time.time()

    for market, db_type, db_path in DB_CONFIGS:
        print(f"=== {market}/{db_type} ({db_path.name}) ===")
        stats = export_db(market, db_type, db_path, force=args.force)
        total_tables += len(stats)
        total_rows += sum(stats.values())
        print()

    elapsed = time.time() - t0

    # Calculate total Parquet size
    total_size = sum(
        f.stat().st_size for f in PARQUET_DIR.rglob("*.parquet")
    )

    print(f"=== DONE ===")
    print(f"Tables exported: {total_tables}")
    print(f"Total rows: {total_rows:,}")
    print(f"Parquet size: {total_size / 1024 / 1024:.0f} MB")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
