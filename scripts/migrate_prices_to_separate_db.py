#!/usr/bin/env python3
"""Migrate prices-related tables from main DuckDB files to separate prices DBs.

Source → Destination:
  market_data_kr.duckdb  → market_data_kr_prices.duckdb  (prices, ticker_master, index_prices)
  market_data.duckdb     → market_data_prices.duckdb     (prices, sec_issuer_registry, ticker_market_cap)

After copying, the source tables are DROPped and the source DB is VACUUMed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"


def migrate_kr():
    import duckdb

    src_path = DATA_DIR / "market_data_kr.duckdb"
    dst_path = DATA_DIR / "market_data_kr_prices.duckdb"

    if not src_path.exists():
        print(f"[KR] Source not found: {src_path}")
        return

    tables = ["prices", "ticker_master", "index_prices"]

    print(f"[KR] Opening source: {src_path}")
    src = duckdb.connect(str(src_path))

    # Check which tables exist
    existing = {r[0] for r in src.execute("SHOW TABLES").fetchall()}
    tables_to_move = [t for t in tables if t in existing]
    if not tables_to_move:
        print("[KR] No tables to migrate")
        src.close()
        return

    print(f"[KR] Creating destination: {dst_path}")
    dst = duckdb.connect(str(dst_path))

    for table in tables_to_move:
        count = src.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"[KR] Copying {table} ({count:,} rows)...")

        # Export to parquet as intermediate (handles large tables efficiently)
        tmp_parquet = DATA_DIR / f"_migrate_{table}.parquet"
        src.execute(f"COPY {table} TO '{tmp_parquet}' (FORMAT PARQUET)")

        # Get CREATE TABLE DDL by describing columns
        cols = src.execute(f"DESCRIBE {table}").fetchall()
        col_defs = ", ".join(f'"{c[0]}" {c[1]}' for c in cols)
        dst.execute(f"DROP TABLE IF EXISTS {table}")
        dst.execute(f"CREATE TABLE {table} ({col_defs})")
        dst.execute(f"INSERT INTO {table} SELECT * FROM read_parquet('{tmp_parquet}')")

        verify = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"[KR] {table}: {verify:,} rows copied")
        assert verify == count, f"Row count mismatch: {verify} != {count}"

        tmp_parquet.unlink()

    dst.close()

    # Drop from source
    for table in tables_to_move:
        print(f"[KR] Dropping {table} from source...")
        src.execute(f"DROP TABLE {table}")

    print("[KR] Vacuuming source...")
    src.execute("VACUUM")
    src.close()
    print("[KR] Done!")


def migrate_us():
    import duckdb

    src_path = DATA_DIR / "market_data.duckdb"
    dst_path = DATA_DIR / "market_data_prices.duckdb"

    if not src_path.exists():
        print(f"[US] Source not found: {src_path}")
        return

    tables = ["prices", "sec_issuer_registry", "ticker_market_cap"]

    print(f"[US] Opening source: {src_path}")
    src = duckdb.connect(str(src_path))

    existing = {r[0] for r in src.execute("SHOW TABLES").fetchall()}
    tables_to_move = [t for t in tables if t in existing]
    if not tables_to_move:
        print("[US] No tables to migrate")
        src.close()
        return

    print(f"[US] Creating destination: {dst_path}")
    dst = duckdb.connect(str(dst_path))

    for table in tables_to_move:
        count = src.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"[US] Copying {table} ({count:,} rows)...")

        tmp_parquet = DATA_DIR / f"_migrate_{table}.parquet"
        src.execute(f"COPY {table} TO '{tmp_parquet}' (FORMAT PARQUET)")

        cols = src.execute(f"DESCRIBE {table}").fetchall()
        col_defs = ", ".join(f'"{c[0]}" {c[1]}' for c in cols)
        dst.execute(f"DROP TABLE IF EXISTS {table}")
        dst.execute(f"CREATE TABLE {table} ({col_defs})")
        dst.execute(f"INSERT INTO {table} SELECT * FROM read_parquet('{tmp_parquet}')")

        verify = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"[US] {table}: {verify:,} rows copied")
        assert verify == count, f"Row count mismatch: {verify} != {count}"

        tmp_parquet.unlink()

    dst.close()

    for table in tables_to_move:
        print(f"[US] Dropping {table} from source...")
        src.execute(f"DROP TABLE {table}")

    print("[US] Vacuuming source...")
    src.execute("VACUUM")
    src.close()
    print("[US] Done!")


if __name__ == "__main__":
    migrate_kr()
    print()
    migrate_us()
    print()
    print("=== Final DB sizes ===")
    for f in sorted(DATA_DIR.glob("market_data*.duckdb")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")
