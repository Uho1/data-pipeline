"""
A-E 시작 누락 티커 복구 스크립트

문제: 원래 prices 테이블에 A-E 티커가 없어서 top 2000 선정에서 제외됨
      (AAPL $4T, AVGO $1.6T, COST $453B 등 270개 대형주 누락)

해결:
  1. A-E NYSE/NASDAQ 공통주 중 시총 상위 → 추가 대상 선정
  2. yfinance로 가격 데이터 재-ingest
  3. sec_term_cache/ticker_quarterly/*.parquet로 재무 데이터 복구
  4. 전체 top 2000 재선정 → F-Z 하위 티커 제거

실행:
  .venv/bin/python scripts/restore_ae_tickers.py [--dry-run]
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


def get_ae_candidates() -> list[tuple[str, float]]:
    """A-E 시작 NYSE/NASDAQ 공통주 중 parquet 기반 시총 계산 후 반환."""
    prices_con = duckdb.connect(str(PRICES_DB_PATH))
    ae_eligible = prices_con.execute("""
        SELECT DISTINCT ticker FROM sec_issuer_registry
        WHERE is_common_stock=TRUE AND exchange IN ('NYSE','NASDAQ')
          AND LEFT(ticker,1) IN ('A','B','C','D','E')
    """).fetchdf()["ticker"].tolist()
    prices_con.close()

    results = []
    for t in ae_eligible:
        path = CACHE_DIR / f"{t}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            if df.empty:
                continue
            last = df.sort_values("PeriodEnd").iloc[-1]
            shares = (last.get("Diluted Shares") or last.get("Basic Shares")
                      or last.get("Shares"))
            price = last.get("Price")
            if shares and price and shares > 0 and price > 0:
                results.append((t, float(shares * price)))
        except Exception:
            continue

    return sorted(results, key=lambda x: -x[1])


def get_current_fz_bottom_mktcap(con) -> float:
    """현재 F-Z 2000번째 시총 (하위 컷오프)."""
    con.execute(f"ATTACH '{PRICES_DB_PATH}' AS prices_db (READ_ONLY)")
    row = con.execute("""
        WITH latest_close AS (
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
        ranked AS (
            SELECT lc.ticker, lc.close * ls.shares AS mktcap,
                ROW_NUMBER() OVER (ORDER BY lc.close * ls.shares DESC) AS rnk
            FROM latest_close lc JOIN latest_shares ls ON ls.ticker=lc.ticker
        )
        SELECT mktcap FROM ranked WHERE rnk = 2000
    """).fetchone()
    con.execute("DETACH prices_db")
    return float(row[0]) if row else 0.0


def ingest_prices_batch(tickers: list[str]) -> dict[str, int]:
    """yfinance로 가격 일괄 다운로드 후 DB upsert. {ticker: rows_written} 반환."""
    written = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  가격 다운로드 [{i+1}~{min(i+batch_size, len(tickers))}/{len(tickers)}]: {batch[:3]}...")
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
            # multi-ticker 결과 분리
            if isinstance(raw.columns, pd.MultiIndex):
                for tk in batch:
                    try:
                        df = raw.xs(tk, axis=1, level=1).copy() if tk in raw.columns.get_level_values(1) else pd.DataFrame()
                        if df.empty:
                            continue
                        rows = db_writer.upsert_prices(df, tk, "us")
                        written[tk] = rows
                    except Exception as e:
                        print(f"    {tk} 가격 오류: {e}")
            else:
                # 단일 ticker
                tk = batch[0]
                rows = db_writer.upsert_prices(raw, tk, "us")
                written[tk] = rows
        except Exception as e:
            print(f"  배치 오류: {e}")
    return written


def restore_financials(tickers: list[str]) -> dict[str, int]:
    """parquet 캐시에서 financials_quarterly 복구. {ticker: rows_written} 반환."""
    written = {}
    for tk in tickers:
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


def prune_to_top2000() -> tuple[list[str], list[str]]:
    """전체 재선정: top 2000 유지, 나머지 삭제. (added, removed) 반환."""
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

    n_top = min(2000, len(result))
    top2000 = set(result.head(n_top)["ticker"].tolist())
    to_remove = set(result.iloc[n_top:]["ticker"].tolist())

    # prices에 있지만 top2000 밖인 티커도 제거
    cur_tickers = set(
        prices_con.execute("SELECT DISTINCT ticker FROM prices WHERE market='us'").fetchdf()["ticker"].tolist()
    )
    to_remove = to_remove | (cur_tickers - top2000)

    TABLES = [
        ("prices",                    "market"),
        ("financials_quarterly",      "market"),
        ("financials_quarterly_extra","market"),
        ("derived_factors_quarterly", "market"),
        ("segment_revenue_quarterly", "market"),
        ("filings",                   "market"),
        ("ingest_checkpoints",        "market"),
        ("segment_facts_quarterly",   "market"),
    ]

    PRICES_DB_TABLES = {"prices"}
    if to_remove:
        rm_sql = ",".join(f"'{t}'" for t in to_remove)
        for table, mcol in TABLES:
            try:
                tbl_con = prices_con if table in PRICES_DB_TABLES else con
                tbl_con.execute(f"DELETE FROM {table} WHERE {mcol}='us' AND ticker IN ({rm_sql})")
            except Exception as e:
                print(f"  {table} 삭제 오류: {e}")

    # universe_pruning_history 업데이트
    top_row = result.iloc[0]
    bot_row = result.iloc[n_top - 1]
    price_date = prices_con.execute(
        "SELECT MAX(date) FROM prices WHERE market='us'"
    ).fetchone()[0]
    con.execute("""
        INSERT INTO universe_pruning_history VALUES (
            CURRENT_TIMESTAMP, 'NYSE+NASDAQ common stock top 2000 (A-E restored)',
            ?, 2000, ?, ?, ?, ?, ?
        )
    """, [price_date, top_row["ticker"], float(top_row["mktcap"]),
          bot_row["ticker"], float(bot_row["mktcap"]), len(top2000)])

    con.execute("CHECKPOINT")
    prices_con.execute("CHECKPOINT")
    prices_con.close()
    con.close()
    return list(top2000), list(to_remove)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_writer.init_schema()

    # 1. 후보 선정
    print("A-E 후보 선정 중...")
    con_r = duckdb.connect(str(DB_PATH))
    fz_bottom = get_current_fz_bottom_mktcap(con_r)
    con_r.close()
    print(f"  현재 F-Z 2000위 시총 하한: ${fz_bottom:,.0f}")

    candidates = get_ae_candidates()
    # 현재 2000위 시총보다 큰 것만 추가 (여유있게 $100M 이하까지 포함)
    to_add = [(t, mc) for t, mc in candidates if mc > 100_000_000]
    print(f"  추가 대상 A-E 티커: {len(to_add)}개")
    print(f"  Top 5: {[(t, f'${mc:.2e}') for t, mc in to_add[:5]]}")
    print(f"  Bottom 5: {[(t, f'${mc:.2e}') for t, mc in to_add[-5:]]}")

    if args.dry_run:
        print("\n[dry-run] 실제 변경 없음.")
        return 0

    add_tickers = [t for t, _ in to_add]

    # 2. 가격 데이터 ingest
    print(f"\n[1/3] 가격 데이터 ingest ({len(add_tickers)}개)...")
    t0 = time.time()
    price_written = ingest_prices_batch(add_tickers)
    print(f"  완료: {len(price_written)}개 ticker, {sum(price_written.values()):,} rows ({time.time()-t0:.0f}s)")

    # 3. 재무 데이터 복구
    print(f"\n[2/3] 재무 데이터 복구 ({len(add_tickers)}개)...")
    t0 = time.time()
    fin_written = restore_financials(add_tickers)
    print(f"  완료: {len(fin_written)}개 ticker, {sum(fin_written.values()):,} rows ({time.time()-t0:.0f}s)")

    # 4. Top 2000 재선정 + 하위 제거
    print(f"\n[3/3] Top 2000 재선정 및 하위 티커 제거...")
    t0 = time.time()
    top2000, removed = prune_to_top2000()
    print(f"  Top 2000 확정: {len(top2000)}개")
    print(f"  제거된 티커: {len(removed)}개")
    print(f"  소요: {time.time()-t0:.0f}s")

    # universe CSV 업데이트
    _update_universe_files(top2000)

    print("\n=== 완료 ===")
    print(f"  추가: {len(price_written)}개 A-E 티커")
    print(f"  제거: {len(removed)}개 하위 티커")
    return 0


def _update_universe_files(top2000_tickers: list[str]) -> None:
    """data/universe/top_2000/ 파일 업데이트."""
    import json
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"ATTACH '{PRICES_DB_PATH}' AS prices_db (READ_ONLY)")
    df = con.execute("""
        WITH latest_close AS (
            SELECT ticker, close FROM prices_db.prices WHERE market='us'
            AND date=(SELECT MAX(date) FROM prices_db.prices WHERE market='us')
        ),
        latest_shares AS (
            SELECT DISTINCT ON (ticker) ticker,
                COALESCE("Diluted Shares","Basic Shares",Shares) AS shares,
                name
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
        WHERE lc.ticker IN ({})
        ORDER BY market_cap DESC
    """.format(",".join(f"'{t}'" for t in top2000_tickers))).fetchdf()
    con.execute("DETACH prices_db")
    con.close()

    out_dir = ROOT / "data" / "universe" / "top_2000"
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = df["ticker"].tolist()
    price_date = str(pd.Timestamp.now().date())
    (out_dir / "tickers.txt").write_text("\n".join(tickers) + "\n")
    df.to_csv(out_dir / "universe.csv", index=False)
    meta = {
        "description": "NYSE+NASDAQ 보통주 시총 Top 2000 (A-E 복구 완료)",
        "reference_date": price_date,
        "ticker_count": len(tickers),
        "rank_1": {"ticker": df.iloc[0]["ticker"], "market_cap_usd": float(df.iloc[0]["market_cap"])},
        "rank_2000": {"ticker": df.iloc[-1]["ticker"], "market_cap_usd": float(df.iloc[-1]["market_cap"])},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"  universe 파일 업데이트 완료 ({len(tickers)}개 티커)")


if __name__ == "__main__":
    sys.exit(main())
