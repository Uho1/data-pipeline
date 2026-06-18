#!/usr/bin/env python3
"""[MIGRATION TOOL] Build the DuckDB database from existing parquet files.

Use this script ONCE to migrate legacy parquet data into DuckDB.
After migration, new data is written directly to DuckDB via ingest.

    python scripts/build_duckdb.py [--market us|kr|all] [--force]

After migration you can delete data/prices/ and data/financials/ to free
disk space — all data will be in data/market_data.duckdb.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import duckdb
from market_data.config import DATA_DIR, FINANCIALS_DIR, PRICES_DIR
from market_data.db import DB_PATH
from market_data.db_prices import DB_PATH as PRICES_DB_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_size(path: Path) -> str:
    b = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def build_prices(con: duckdb.DuckDBPyConnection, market: str) -> int:
    prices_dir = PRICES_DIR / market
    if not prices_dir.exists():
        print(f"  [skip] {prices_dir} not found")
        return 0

    glob = str(prices_dir / "*.parquet")
    print(f"  Scanning {glob} …", flush=True)
    t0 = time.perf_counter()

    # DuckDB reads all files in one shot; union_by_name handles minor schema
    # differences between tickers (e.g. older files missing a column).
    con.execute(f"""
        CREATE OR REPLACE TABLE prices AS
        SELECT
            "Date"::DATE            AS date,
            "Ticker"                AS ticker,
            '{market}'              AS market,
            "Open"                  AS open,
            "High"                  AS high,
            "Low"                   AS low,
            "Close"                 AS close,
            "Adj Close"             AS adj_close,
            "Volume"::BIGINT        AS volume,
            "Dividends"             AS dividends,
            "Stock Splits"          AS stock_splits
        FROM read_parquet('{glob}', union_by_name = true)
        WHERE "Date" IS NOT NULL
          AND "Ticker" IS NOT NULL
        ORDER BY ticker, date
    """)

    rows = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    tickers = con.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
    elapsed = time.perf_counter() - t0
    print(f"  prices: {rows:,} rows · {tickers:,} tickers · {elapsed:.1f}s")
    return rows


_FIN_SELECT = """\
        SELECT
            "symbol"::VARCHAR                           AS ticker,
            '{market}'::VARCHAR                         AS market,
            TRY_CAST("term" AS VARCHAR)                 AS term,
            TRY_CAST("StatementDate" AS DATE)           AS "StatementDate",
            TRY_CAST("PeriodEnd" AS DATE)               AS "PeriodEnd",
            TRY_CAST("PeriodStart" AS DATE)             AS "PeriodStart",
            TRY_CAST("FormType" AS VARCHAR)             AS "FormType",
            TRY_CAST("FilingDate" AS DATE)              AS "FilingDate",
            TRY_CAST("AcceptedAt" AS TIMESTAMPTZ)       AS "AcceptedAt",
            TRY_CAST("AvailableDate" AS DATE)           AS "AvailableDate",
            TRY_CAST("AvailabilityMethod" AS VARCHAR)   AS "AvailabilityMethod",
            TRY_CAST("Revenue" AS DOUBLE)               AS "Revenue",
            TRY_CAST("COGS" AS DOUBLE)                  AS "COGS",
            TRY_CAST("Gross Profit" AS DOUBLE)          AS "Gross Profit",
            TRY_CAST("SG&A" AS DOUBLE)                  AS "SG&A",
            TRY_CAST("Operating Income" AS DOUBLE)      AS "Operating Income",
            TRY_CAST("Net Income" AS DOUBLE)            AS "Net Income",
            TRY_CAST("Net Income Common" AS DOUBLE)     AS "Net Income Common",
            TRY_CAST("EPS" AS DOUBLE)                   AS "EPS",
            TRY_CAST("Diluted EPS" AS DOUBLE)           AS "Diluted EPS",
            TRY_CAST("diluted_eps" AS DOUBLE)           AS diluted_eps,
            TRY_CAST("diluted_shares" AS DOUBLE)        AS diluted_shares,
            TRY_CAST("basic_shares" AS DOUBLE)          AS basic_shares,
            TRY_CAST("net_income_common" AS DOUBLE)     AS net_income_common,
            TRY_CAST("eps_source" AS VARCHAR)           AS eps_source,
            TRY_CAST("Operating Cash Flow" AS DOUBLE)   AS "Operating Cash Flow",
            TRY_CAST("Investing Cash Flow" AS DOUBLE)   AS "Investing Cash Flow",
            TRY_CAST("Financing Cash Flow" AS DOUBLE)   AS "Financing Cash Flow",
            TRY_CAST("Capital Expenditure" AS DOUBLE)   AS "Capital Expenditure",
            TRY_CAST("Total Assets" AS DOUBLE)          AS "Total Assets",
            TRY_CAST("Total Liabilities" AS DOUBLE)     AS "Total Liabilities",
            TRY_CAST("Shareholders Equity" AS DOUBLE)   AS "Shareholders Equity",
            TRY_CAST("Shares" AS DOUBLE)                AS "Shares",
            TRY_CAST("Diluted Shares" AS DOUBLE)        AS "Diluted Shares",
            TRY_CAST("Basic Shares" AS DOUBLE)          AS "Basic Shares",
            TRY_CAST("Price" AS DOUBLE)                 AS "Price",
            TRY_CAST("name" AS VARCHAR)                 AS "name",
            TRY_CAST("sector" AS VARCHAR)               AS "sector",
            TRY_CAST("industry" AS VARCHAR)             AS "industry",
            TRY_CAST("Source" AS VARCHAR)               AS "Source"
        FROM read_parquet('{path}', union_by_name = true)
        WHERE "symbol" IS NOT NULL
          AND "PeriodEnd" IS NOT NULL\
"""


def build_financials(con: duckdb.DuckDBPyConnection, market: str) -> int:
    fin_dir = FINANCIALS_DIR / market
    if not fin_dir.exists():
        print(f"  [skip] {fin_dir} not found")
        return 0

    files = sorted(fin_dir.glob("*/sec_companyfacts_quarterly.parquet"))
    if not files:
        print(f"  [skip] No sec_companyfacts_quarterly.parquet files found in {fin_dir}")
        return 0

    print(f"  Ingesting {len(files):,} financial parquet files …", flush=True)
    t0 = time.perf_counter()

    # Create the table from the first file so the schema is established.
    first = str(files[0])
    con.execute(
        f"CREATE OR REPLACE TABLE financials_quarterly AS "
        + _FIN_SELECT.format(market=market, path=first)
    )

    # Append remaining files one by one – avoids cross-file type-conflict errors.
    errors = 0
    for f in files[1:]:
        try:
            con.execute(
                "INSERT INTO financials_quarterly "
                + _FIN_SELECT.format(market=market, path=str(f))
            )
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"    [warn] {f.parent.name}: {exc}")

    # Sort for zone-map efficiency
    con.execute(
        "CREATE OR REPLACE TABLE financials_quarterly AS "
        "SELECT * FROM financials_quarterly ORDER BY ticker, \"PeriodEnd\""
    )

    rows = con.execute("SELECT COUNT(*) FROM financials_quarterly").fetchone()[0]
    tickers = con.execute(
        "SELECT COUNT(DISTINCT ticker) FROM financials_quarterly"
    ).fetchone()[0]
    elapsed = time.perf_counter() - t0
    err_note = f" · {errors} skipped" if errors else ""
    print(
        f"  financials_quarterly: {rows:,} rows · {tickers:,} tickers · "
        f"{elapsed:.1f}s{err_note}"
    )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build DuckDB market-data database")
    parser.add_argument("--market", default="us", choices=["us", "kr", "all"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing database and rebuild from scratch",
    )
    args = parser.parse_args()

    markets = ["us", "kr"] if args.market == "all" else [args.market]

    print(f"Target database : {DB_PATH}")
    print(f"Markets         : {', '.join(markets)}")
    print()

    if DB_PATH.exists():
        if args.force:
            print(f"Removing existing database …")
            DB_PATH.unlink()
        else:
            print(
                f"Database already exists ({_fmt_size(DB_PATH)}).\n"
                "Use --force to rebuild."
            )
            return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRICES_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    with duckdb.connect(str(DB_PATH)) as con, duckdb.connect(str(PRICES_DB_PATH)) as prices_con:
        for mkt in markets:
            print(f"=== Market: {mkt.upper()} ===")
            build_prices(prices_con, mkt)
            build_financials(con, mkt)
            print()

        # Summary
        for label, summary_con in [("main", con), ("prices", prices_con)]:
            tables = summary_con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchdf()["table_name"].tolist()

            print(f"=== Summary ({label}) ===")
            for tbl in tables:
                cnt = summary_con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                print(f"  {tbl:30s}: {cnt:>10,} rows")

    db_size = _fmt_size(DB_PATH)
    elapsed = time.perf_counter() - t_total
    print(f"\nDatabase size : {db_size}")
    print(f"Total time    : {elapsed:.1f}s")
    print(f"\nDone!  Use 'from market_data.db import db_available' to check availability.")


if __name__ == "__main__":
    main()
