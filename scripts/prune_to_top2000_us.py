"""
market_data.duckdb를 NYSE+NASDAQ 보통주 시총 Top 2000으로 축소.

기준:
- sec_issuer_registry에서 NYSE/NASDAQ + is_common_stock=TRUE
- prices 최신날짜(2026-03-12) 종가 × financials_quarterly 최신 shares = 시총
- WRDS prices_daily_canonical market_cap 우선 적용 (더 신뢰도 높음)
- 이상값 ($20T 초과) 제거
- 시총 기준 상위 2000 유지, 나머지 삭제

삭제 대상 테이블 (market='us'):
  prices, financials_quarterly, financials_quarterly_extra,
  derived_factors_quarterly, segment_revenue_quarterly,
  sec_facts_raw_normalized, filings, ingest_checkpoints,
  segment_facts_quarterly

실행:
  .venv/bin/python scripts/prune_to_top2000_us.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import duckdb

from market_data.db import DB_PATH
from market_data.db_prices import DB_PATH as PRICES_DB_PATH

# 프루닝할 (테이블, market 컬럼 유무) 목록
TABLES = [
    ("prices",                    "market"),
    ("financials_quarterly",      "market"),
    ("financials_quarterly_extra","market"),
    ("derived_factors_quarterly", "market"),
    ("segment_revenue_quarterly", "market"),
    ("sec_facts_raw_normalized",  "market"),
    ("filings",                   "market"),
    ("ingest_checkpoints",        "market"),
    ("segment_facts_quarterly",   "market"),
]


def compute_top2000(con: duckdb.DuckDBPyConnection) -> tuple[list[str], dict]:
    """Top 2000 ticker 목록과 메타 정보 반환."""
    con.execute(f"ATTACH '{PRICES_DB_PATH}' AS prices_db (READ_ONLY)")
    result = con.execute("""
        WITH eligible AS (
            SELECT DISTINCT ticker
            FROM prices_db.sec_issuer_registry
            WHERE is_common_stock = TRUE AND exchange IN ('NYSE', 'NASDAQ')
        ),
        latest_close AS (
            SELECT ticker, close AS price, date AS price_date
            FROM prices_db.prices
            WHERE market = 'us'
              AND date = (SELECT MAX(date) FROM prices_db.prices WHERE market = 'us')
        ),
        latest_shares AS (
            SELECT DISTINCT ON (ticker) ticker,
                COALESCE("Diluted Shares", "Basic Shares", Shares) AS shares,
                PeriodEnd AS shares_date
            FROM financials_quarterly
            WHERE market = 'us'
              AND COALESCE("Diluted Shares", "Basic Shares", Shares) > 0
            ORDER BY ticker, PeriodEnd DESC
        ),
        wrds_mc AS (
            SELECT ticker, market_cap AS wrds_market_cap
            FROM prices_daily_canonical
            WHERE trade_date = '2024-12-31'
              AND ticker IS NOT NULL
              AND exchange_code IN (1, 3)
        ),
        ranked AS (
            SELECT
                e.ticker,
                lc.price,
                lc.price_date,
                ls.shares,
                ls.shares_date,
                COALESCE(wm.wrds_market_cap, lc.price * ls.shares) AS mktcap,
                CASE WHEN wm.wrds_market_cap IS NOT NULL THEN 'WRDS' ELSE 'computed' END AS mktcap_source,
                ROW_NUMBER() OVER (
                    ORDER BY COALESCE(wm.wrds_market_cap, lc.price * ls.shares) DESC
                ) AS rnk
            FROM eligible e
            JOIN latest_close lc ON lc.ticker = e.ticker
            LEFT JOIN latest_shares ls ON ls.ticker = e.ticker
            LEFT JOIN wrds_mc wm ON wm.ticker = e.ticker
            WHERE ls.shares > 0
              AND (wm.wrds_market_cap IS NOT NULL
                   OR (lc.price * ls.shares) < 2e13)  -- $20T 이상 이상값 제거
        )
        SELECT ticker, price, price_date, shares, shares_date, mktcap, mktcap_source, rnk
        FROM ranked
        WHERE rnk <= 2000
        ORDER BY rnk
    """).fetchdf()
    con.execute("DETACH prices_db")

    tickers = result["ticker"].tolist()
    meta = {
        "price_date": str(result["price_date"].iloc[0].date()),
        "total_eligible": len(result),
        "mktcap_top1": result["mktcap"].iloc[0],
        "mktcap_top1_ticker": result["ticker"].iloc[0],
        "mktcap_2000": result["mktcap"].iloc[-1],
        "mktcap_2000_ticker": result["ticker"].iloc[-1],
    }
    return tickers, meta


# Tables in the prices DB (use prices_con for these)
PRICES_DB_TABLES = {"prices", "sec_issuer_registry"}


def main() -> int:
    parser = argparse.ArgumentParser(description="market_data.duckdb → Top 2000 US 보통주로 축소")
    parser.add_argument("--dry-run", action="store_true", help="삭제 없이 결과만 출력")
    parser.add_argument("--yes", action="store_true", help="확인 프롬프트 없이 바로 실행")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH))
    prices_con = duckdb.connect(str(PRICES_DB_PATH))

    print("Top 2000 목록 계산 중...")
    top2000, meta = compute_top2000(con)

    print(f"\n=== Top 2000 선정 결과 ===")
    print(f"  시총 기준일 (prices):    {meta['price_date']}")
    print(f"  1위: {meta['mktcap_top1_ticker']}  ${meta['mktcap_top1']:,.0f}")
    print(f"  2000위: {meta['mktcap_2000_ticker']}  ${meta['mktcap_2000']:,.0f}")
    print(f"  선정된 ticker 수: {len(top2000)}")

    # 현재 각 테이블의 US ticker 수 파악 (단일 쿼리 방식)
    ticker_list_sql_preview = ",".join(f"'{t}'" for t in top2000)
    print(f"\n=== 테이블별 삭제 예상 ===")
    for table, market_col in TABLES:
        try:
            tbl_con = prices_con if table in PRICES_DB_TABLES else con
            row = tbl_con.execute(f"""
                SELECT
                    COUNT(DISTINCT ticker) AS total,
                    COUNT(DISTINCT CASE WHEN ticker IN ({ticker_list_sql_preview}) THEN ticker END) AS keep
                FROM {table} WHERE {market_col}='us'
            """).fetchone()
            total, keep = row
            to_delete = total - keep
            print(f"  {table:40s}: {total:5d} tickers → {to_delete:5d} 삭제, {keep:5d} 유지")
        except Exception as e:
            print(f"  {table}: ERROR {e}")

    if args.dry_run:
        print("\n[dry-run] 삭제 실행 안 함.")
        return 0

    print()
    if not args.yes:
        resp = input("위 테이블에서 top 2000 외 ticker를 삭제하시겠습니까? [y/N] ").strip().lower()
        if resp != "y":
            print("취소.")
            return 0

    # ticker 목록을 임시 테이블로 (both connections need it)
    ticker_list_sql = ",".join(f"('{t}')" for t in top2000)
    con.execute("CREATE OR REPLACE TEMP TABLE top2000_tickers (ticker VARCHAR)")
    con.execute(f"INSERT INTO top2000_tickers VALUES {ticker_list_sql}")
    prices_con.execute("CREATE OR REPLACE TEMP TABLE top2000_tickers (ticker VARCHAR)")
    prices_con.execute(f"INSERT INTO top2000_tickers VALUES {ticker_list_sql}")

    total_start = time.time()
    print()

    for table, market_col in TABLES:
        start = time.time()
        tbl_con = prices_con if table in PRICES_DB_TABLES else con
        try:
            before = tbl_con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {market_col}='us'"
            ).fetchone()[0]

            tbl_con.execute(f"""
                DELETE FROM {table}
                WHERE {market_col} = 'us'
                  AND ticker NOT IN (SELECT ticker FROM top2000_tickers)
            """)

            after = tbl_con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {market_col}='us'"
            ).fetchone()[0]
            elapsed = time.time() - start
            print(f"  {table:40s}: {before:8,} → {after:8,} rows  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  {table}: ERROR {e}")

    # VACUUM (공간 회수)
    print("\nCHECKPOINT 실행 중 (WAL 정리)...")
    con.execute("CHECKPOINT")
    prices_con.execute("CHECKPOINT")

    elapsed_total = time.time() - total_start
    print(f"\n완료. 총 소요: {elapsed_total:.0f}초")
    print(f"참고: DuckDB 파일 크기 축소는 DB 재생성(EXPORT/IMPORT) 후 반영됩니다.")
    print(f"      일단 데이터는 삭제되었으며, 새 ingest 시 불필요한 확장은 없습니다.")

    # 기준 정보 저장 (메타 테이블에 기록)
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS universe_pruning_history (
                pruned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                universe_type VARCHAR,
                reference_date DATE,
                top_n INTEGER,
                mktcap_1st_ticker VARCHAR,
                mktcap_1st DOUBLE,
                mktcap_cutoff_ticker VARCHAR,
                mktcap_cutoff DOUBLE,
                ticker_count INTEGER
            )
        """)
        con.execute("""
            INSERT INTO universe_pruning_history VALUES (
                CURRENT_TIMESTAMP, 'NYSE+NASDAQ common stock top 2000',
                ?, 2000, ?, ?, ?, ?, ?
            )
        """, [
            meta["price_date"],
            meta["mktcap_top1_ticker"], meta["mktcap_top1"],
            meta["mktcap_2000_ticker"], meta["mktcap_2000"],
            len(top2000),
        ])
        print(f"\nuniverse_pruning_history 테이블에 기준 정보 저장 완료.")
    except Exception as e:
        print(f"메타 저장 오류 (무시 가능): {e}")

    prices_con.close()
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
