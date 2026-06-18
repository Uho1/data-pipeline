"""
전체 top 2000 복구 스크립트

상황:
  - F-Z 2000개 가격+재무 데이터 실수로 삭제됨
  - A-E 306개는 이미 복구됨 (prices + financials_quarterly)
  - sec_term_cache/ticker_quarterly/*.parquet로 F-Z 복구 가능

해결:
  1. F-Z NYSE/NASDAQ 공통주 가격 재-ingest (yfinance)
  2. F-Z 재무 데이터 parquet → financials_quarterly 복구
  3. db_writer.close() 후 top 2000 재선정 (연결 충돌 방지)
  4. universe 파일 업데이트

실행:
  .venv/bin/python scripts/restore_full_top2000.py [--skip-prices] [--skip-financials]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yfinance as yf
import duckdb

from market_data import db_writer
from market_data.db import DB_PATH
from market_data.db_prices import DB_PATH as PRICES_DB_PATH
CACHE_DIR = ROOT / "data" / "sec_term_cache" / "ticker_quarterly"


def get_fz_candidates() -> list[str]:
    """F-Z NYSE/NASDAQ 공통주 중 parquet 캐시 있는 것."""
    db_writer.close()  # 연결 닫고 안전하게 읽기
    prices_con = duckdb.connect(str(PRICES_DB_PATH))
    fz_eligible = prices_con.execute("""
        SELECT DISTINCT ticker FROM sec_issuer_registry
        WHERE is_common_stock=TRUE AND exchange IN ('NYSE','NASDAQ')
          AND LEFT(ticker,1) NOT IN ('A','B','C','D','E')
    """).fetchdf()["ticker"].tolist()
    prices_con.close()

    return [t for t in fz_eligible if (CACHE_DIR / f"{t}.parquet").exists()]


def ingest_prices_batch(tickers: list[str], label: str = "") -> dict[str, int]:
    """yfinance 배치 가격 다운로드 → DB upsert."""
    written = {}
    batch_size = 50
    total = len(tickers)
    for i in range(0, total, batch_size):
        batch = tickers[i: i + batch_size]
        print(f"  [{label}] 가격 [{i+1}~{min(i+batch_size, total)}/{total}]: {batch[0]}...", flush=True)
        try:
            raw = yf.download(
                tickers=batch,
                period="max",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for tk in batch:
                    try:
                        if tk not in raw.columns.get_level_values(1):
                            continue
                        df = raw.xs(tk, axis=1, level=1).copy()
                        if df.empty:
                            continue
                        rows = db_writer.upsert_prices(df, tk, "us")
                        written[tk] = rows
                    except Exception as e:
                        print(f"    {tk} 오류: {e}")
            else:
                tk = batch[0]
                rows = db_writer.upsert_prices(raw, tk, "us")
                written[tk] = rows
        except Exception as e:
            print(f"  배치 오류: {e}")
    return written


def restore_financials_batch(tickers: list[str], label: str = "") -> dict[str, int]:
    """parquet → financials_quarterly 복구."""
    written = {}
    for i, tk in enumerate(tickers):
        if i % 100 == 0:
            print(f"  [{label}] 재무 [{i}/{len(tickers)}]...", flush=True)
        path = CACHE_DIR / f"{tk}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            rows = db_writer.upsert_financials(df, tk, "us")
            written[tk] = rows
        except Exception as e:
            print(f"  {tk} 재무 오류: {e}")
    return written


def run_top2000_selection_and_prune() -> tuple[list[str], list[str]]:
    """db_writer 닫은 후 top 2000 재선정 + 하위 삭제."""
    # ★ 중요: db_writer 연결을 먼저 닫아야 새 연결이 최신 데이터를 볼 수 있음
    db_writer.close()
    time.sleep(1)

    con = duckdb.connect(str(DB_PATH))
    prices_con = duckdb.connect(str(PRICES_DB_PATH))
    con.execute(f"ATTACH '{PRICES_DB_PATH}' AS prices_db (READ_ONLY)")

    result = con.execute("""
        WITH eligible AS (
            SELECT DISTINCT sir.ticker
            FROM prices_db.sec_issuer_registry sir
            JOIN prices_db.prices p ON p.ticker=sir.ticker AND p.market='us'
            WHERE sir.is_common_stock=TRUE AND sir.exchange IN ('NYSE','NASDAQ')
        ),
        latest_close AS (
            SELECT ticker, close FROM prices_db.prices WHERE market='us'
            AND date=(SELECT MAX(date) FROM prices_db.prices WHERE market='us')
        ),
        latest_shares AS (
            SELECT DISTINCT ON (ticker) ticker,
                COALESCE("Diluted Shares","Basic Shares",Shares) AS shares
            FROM financials_quarterly WHERE market='us'
            AND COALESCE("Diluted Shares","Basic Shares",Shares) > 0
            ORDER BY ticker, PeriodEnd DESC
        ),
        wrds_mc AS (
            SELECT ticker, market_cap FROM prices_daily_canonical
            WHERE trade_date='2024-12-31' AND ticker IS NOT NULL AND exchange_code IN (1,3)
        )
        SELECT e.ticker,
            COALESCE(wm.market_cap, lc.close*ls.shares) AS mktcap,
            ROW_NUMBER() OVER (ORDER BY COALESCE(wm.market_cap, lc.close*ls.shares) DESC) AS rnk
        FROM eligible e
        JOIN latest_close lc ON lc.ticker=e.ticker
        LEFT JOIN latest_shares ls ON ls.ticker=e.ticker
        LEFT JOIN wrds_mc wm ON wm.ticker=e.ticker
        WHERE ls.shares > 0
          AND (wm.market_cap IS NOT NULL OR lc.close*ls.shares < 2e13)
        ORDER BY mktcap DESC
    """).fetchdf()
    con.execute("DETACH prices_db")

    total_eligible = len(result)
    n_top = min(2000, total_eligible)
    top2000 = set(result.head(n_top)["ticker"].tolist())
    to_remove = set(result.iloc[n_top:]["ticker"].tolist())

    # prices에 있지만 top2000 밖인 것도 제거
    cur_prices = set(
        prices_con.execute("SELECT DISTINCT ticker FROM prices WHERE market='us'").fetchdf()["ticker"].tolist()
    )
    to_remove = to_remove | (cur_prices - top2000)

    print(f"  전체 eligible: {total_eligible}, top {n_top}, 제거: {len(to_remove)}")

    TABLES = [
        ("prices",                     "market"),
        ("financials_quarterly",       "market"),
        ("financials_quarterly_extra", "market"),
        ("derived_factors_quarterly",  "market"),
        ("segment_revenue_quarterly",  "market"),
        ("filings",                    "market"),
        ("ingest_checkpoints",         "market"),
        ("segment_facts_quarterly",    "market"),
    ]
    PRICES_DB_TABLES = {"prices"}
    if to_remove:
        rm_sql = ",".join(f"'{t}'" for t in to_remove)
        for table, mcol in TABLES:
            try:
                tbl_con = prices_con if table in PRICES_DB_TABLES else con
                tbl_con.execute(f"DELETE FROM {table} WHERE {mcol}='us' AND ticker IN ({rm_sql})")
            except Exception as e:
                print(f"  {table} 오류: {e}")

    # universe_pruning_history 기록
    top_row = result.iloc[0]
    bot_row = result.iloc[n_top - 1]
    price_date = prices_con.execute("SELECT MAX(date) FROM prices WHERE market='us'").fetchone()[0]
    con.execute("""
        INSERT INTO universe_pruning_history VALUES (
            CURRENT_TIMESTAMP,
            'NYSE+NASDAQ common stock top 2000 (full A-Z restore)',
            ?, 2000, ?, ?, ?, ?, ?
        )
    """, [price_date, top_row["ticker"], float(top_row["mktcap"]),
          bot_row["ticker"], float(bot_row["mktcap"]), n_top])

    con.execute("CHECKPOINT")
    prices_con.execute("CHECKPOINT")
    prices_con.close()
    con.close()
    return list(top2000), list(to_remove)


def update_universe_files(top2000_tickers: list[str]) -> None:
    import json
    db_writer.close()
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"ATTACH '{PRICES_DB_PATH}' AS prices_db (READ_ONLY)")
    df = con.execute("""
        WITH latest_close AS (
            SELECT ticker, close FROM prices_db.prices WHERE market='us'
            AND date=(SELECT MAX(date) FROM prices_db.prices WHERE market='us')
        ),
        latest_shares AS (
            SELECT DISTINCT ON (ticker) ticker, name,
                COALESCE("Diluted Shares","Basic Shares",Shares) AS shares
            FROM financials_quarterly WHERE market='us'
            AND COALESCE("Diluted Shares","Basic Shares",Shares) > 0
            ORDER BY ticker, PeriodEnd DESC
        ),
        wrds_mc AS (
            SELECT ticker, market_cap FROM prices_daily_canonical
            WHERE trade_date='2024-12-31' AND ticker IS NOT NULL AND exchange_code IN (1,3)
        )
        SELECT ROW_NUMBER() OVER (ORDER BY COALESCE(wm.market_cap, lc.close*ls.shares) DESC) AS rank,
            lc.ticker, COALESCE(ls.name,'') AS name, lc.close AS price,
            COALESCE(wm.market_cap, lc.close*ls.shares) AS market_cap
        FROM latest_close lc
        LEFT JOIN latest_shares ls ON ls.ticker=lc.ticker
        LEFT JOIN wrds_mc wm ON wm.ticker=lc.ticker
        WHERE lc.ticker IN ({tickers})
        ORDER BY market_cap DESC
    """.format(tickers=",".join(f"'{t}'" for t in top2000_tickers))).fetchdf()
    con.execute("DETACH prices_db")
    con.close()

    out_dir = ROOT / "data" / "universe" / "top_2000"
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = df["ticker"].tolist()
    (out_dir / "tickers.txt").write_text("\n".join(tickers) + "\n")
    df.to_csv(out_dir / "universe.csv", index=False)
    meta = {
        "description": "NYSE+NASDAQ 보통주 시총 Top 2000 (A-Z 전체 복구 완료)",
        "reference_date": str(pd.Timestamp.now().date()),
        "ticker_count": len(tickers),
        "rank_1": {"ticker": df.iloc[0]["ticker"], "market_cap_usd": float(df.iloc[0]["market_cap"])},
        "rank_2000": {"ticker": df.iloc[-1]["ticker"], "market_cap_usd": float(df.iloc[-1]["market_cap"])},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"  universe 파일 업데이트: {len(tickers)}개 티커")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prices",     action="store_true", help="가격 ingest 건너뜀")
    parser.add_argument("--skip-financials", action="store_true", help="재무 복구 건너뜀")
    args = parser.parse_args()

    db_writer.init_schema()

    # F-Z 후보 목록
    print("F-Z 후보 목록 수집 중...")
    fz_tickers = get_fz_candidates()
    print(f"  F-Z 복구 대상: {len(fz_tickers)}개")

    # 1. 가격 재-ingest
    if not args.skip_prices:
        print(f"\n[1/3] F-Z 가격 재-ingest ({len(fz_tickers)}개)...")
        t0 = time.time()
        db_writer.init_schema()
        price_written = ingest_prices_batch(fz_tickers, "F-Z")
        print(f"  완료: {len(price_written)}개, {sum(price_written.values()):,} rows ({time.time()-t0:.0f}s)")
    else:
        print("[1/3] 가격 ingest 건너뜀")

    # 2. 재무 복구
    if not args.skip_financials:
        print(f"\n[2/3] F-Z 재무 데이터 복구 ({len(fz_tickers)}개)...")
        t0 = time.time()
        fin_written = restore_financials_batch(fz_tickers, "F-Z")
        print(f"  완료: {len(fin_written)}개, {sum(fin_written.values()):,} rows ({time.time()-t0:.0f}s)")
    else:
        print("[2/3] 재무 복구 건너뜀")

    # 3. Top 2000 재선정 (db_writer 닫은 후)
    print("\n[3/3] Top 2000 재선정...")
    t0 = time.time()
    top2000, removed = run_top2000_selection_and_prune()
    print(f"  Top 2000 확정: {len(top2000)}개, 제거: {len(removed)}개 ({time.time()-t0:.0f}s)")

    update_universe_files(top2000)

    print(f"\n=== 복구 완료 ===")
    print(f"  최종 top 2000: {len(top2000)}개 티커")
    return 0


if __name__ == "__main__":
    sys.exit(main())
