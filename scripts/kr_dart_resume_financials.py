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
START_YEAR = 2013
END_YEAR = 2026


def _remaining_tickers() -> list[str]:
    con = duckdb.connect(DB_PATH)
    try:
        query = """
        WITH raw_cov AS (
            SELECT
                ticker,
                MIN(bsns_year) AS min_year,
                MAX(bsns_year) AS max_year
            FROM dart_financials_raw
            WHERE ticker IS NOT NULL
            GROUP BY 1
        ),
        tm AS (
            SELECT
                ticker,
                YEAR(COALESCE(listed_date, DATE '2013-06-01')) AS listed_year
            FROM ticker_master
            WHERE market = 'kr'
        )
        SELECT tm.ticker
        FROM tm
        LEFT JOIN raw_cov r USING (ticker)
        WHERE r.ticker IS NULL
           OR r.min_year > GREATEST(2013, tm.listed_year)
           OR r.max_year < 2025
        ORDER BY tm.ticker
        """
        rows = con.execute(query).fetchall()
    finally:
        con.close()
    return [normalize_kr_ticker(str(row[0])) for row in rows if row and row[0]]


def main() -> int:
    remaining = _remaining_tickers()
    print(f"[resume] remaining_tickers={len(remaining)}")
    code = 0
    if remaining:
        code = run_kr_dart_command(
            KrDartOptions(
                command="financials",
                tickers=remaining,
                start_date="20130601",
                end_date=None,
                start_year=START_YEAR,
                end_year=END_YEAR,
                fs_div="CFS",
            )
        )
        if code not in {0, 75}:
            return code
    materialize_code = run_kr_dart_command(
        KrDartOptions(
            command="materialize",
            tickers=remaining,
            start_date="20130601",
            end_date=None,
            start_year=START_YEAR,
            end_year=END_YEAR,
            fs_div="CFS",
        )
    )
    if materialize_code != 0:
        return materialize_code
    return code


if __name__ == "__main__":
    raise SystemExit(main())
