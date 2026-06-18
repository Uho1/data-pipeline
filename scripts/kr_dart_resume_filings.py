from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_data.db_router import normalize_kr_ticker
from market_data.kr_dart.cli import KrDartOptions, run_kr_dart_command


DB_PATH = str(ROOT / "data" / "market_data_kr.duckdb")
START_DATE = "20130601"
END_DATE = "20260314"


def _remaining_tickers() -> list[str]:
    con = duckdb.connect(DB_PATH)
    try:
        query = """
        WITH eligible AS (
            SELECT ticker
            FROM ticker_master
            WHERE market = 'kr'
              AND COALESCE(NULLIF(TRIM(dart_corp_code), ''), '') <> ''
        ),
        filing_cov AS (
            SELECT ticker, COUNT(*) AS filing_count
            FROM filings
            GROUP BY 1
        )
        SELECT e.ticker
        FROM eligible e
        LEFT JOIN filing_cov f USING (ticker)
        WHERE COALESCE(f.filing_count, 0) = 0
        ORDER BY e.ticker
        """
        rows = con.execute(query).fetchall()
    finally:
        con.close()
    return [normalize_kr_ticker(str(row[0])) for row in rows if row and row[0]]


def main() -> int:
    remaining = _remaining_tickers()
    print(f"[resume-filings] remaining_tickers={len(remaining)}")
    if not remaining:
        return 0
    return run_kr_dart_command(
        KrDartOptions(
            command="filings",
            tickers=remaining,
            start_date=START_DATE,
            end_date=END_DATE,
            start_year=2013,
            end_year=2026,
            fs_div="CFS",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
