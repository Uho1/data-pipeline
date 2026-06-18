"""
KR DART 배치 materialization 스크립트

- ticker 단위로 배치 처리 (메모리 절약)
- 진행상황 실시간 출력
- --all: 기존 materialized 티커 포함 전체 재실행
- --batch-size: 배치 크기 (기본 50)
- --tickers: 특정 티커만 실행 (콤마 구분)
- --start-from: 특정 티커부터 재개
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_data import db_reader_kr, db_writer_kr
from market_data.kr_dart.materialize import materialize_financials_quarterly


def get_ticker_list(con: duckdb.DuckDBPyConnection, *, include_existing: bool) -> list[str]:
    all_tickers = con.execute(
        "SELECT DISTINCT ticker FROM dart_financials_raw ORDER BY ticker"
    ).fetchdf()["ticker"].tolist()
    if include_existing:
        return all_tickers
    materialized = set(
        con.execute("SELECT DISTINCT ticker FROM financials_quarterly WHERE market='kr'")
        .fetchdf()["ticker"].tolist()
    )
    return [t for t in all_tickers if t not in materialized]


def main() -> int:
    parser = argparse.ArgumentParser(description="KR DART 배치 materialization")
    parser.add_argument("--batch-size", type=int, default=50, help="배치당 티커 수 (기본 50)")
    parser.add_argument("--all", dest="include_existing", action="store_true",
                        help="기존 materialized 티커 포함 전체 재실행")
    parser.add_argument("--tickers", default=None,
                        help="특정 티커만 (콤마 구분, 예: 005930,000660)")
    parser.add_argument("--start-from", default=None,
                        help="이 티커부터 재개 (이전 티커 건너뜀)")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    db_writer_kr.init_schema()

    # 처리할 티커 목록 결정 — db_reader_kr 연결 재사용 (별도 read_only 연결 열지 않음)
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        # db_reader_kr의 연결을 통해 티커 목록 조회
        from market_data.db_kr import get_connection
        con = get_connection()
        all_tickers = con.execute(
            "SELECT DISTINCT ticker FROM dart_financials_raw ORDER BY ticker"
        ).fetchdf()["ticker"].tolist()
        if args.include_existing:
            tickers = all_tickers
        else:
            materialized = set(
                con.execute("SELECT DISTINCT ticker FROM financials_quarterly WHERE market='kr'")
                .fetchdf()["ticker"].tolist()
            )
            tickers = [t for t in all_tickers if t not in materialized]

    if args.start_from:
        idx = next((i for i, t in enumerate(tickers) if t >= args.start_from), 0)
        tickers = tickers[idx:]
        print(f"[재개] {args.start_from}부터 시작 ({len(tickers)}개 남음)")

    ticker_master = db_reader_kr.load_ticker_master_all()

    total = len(tickers)
    batch_size = args.batch_size
    total_batches = (total + batch_size - 1) // batch_size

    mode = "전체 재실행" if args.include_existing else "미materialized만"
    print(f"[시작] 처리 대상: {total}개 티커, 배치 크기: {batch_size}, 배치 수: {total_batches} ({mode})")
    print("-" * 70)

    rows_total = 0
    tickers_done = 0
    start_time = time.time()

    for batch_idx in range(total_batches):
        batch = tickers[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_start = time.time()

        try:
            raw = db_reader_kr.load_dart_financials_raw_for_tickers(batch)
            filings = db_reader_kr.load_filings_for_tickers(batch)

            if raw is None or raw.empty:
                print(f"[배치 {batch_idx+1:4d}/{total_batches}] {batch[0]}~{batch[-1]} | raw 없음 — SKIP")
                tickers_done += len(batch)
                continue

            frame = materialize_financials_quarterly(
                raw, filings=filings, ticker_master=ticker_master
            )

            rows_written = 0
            if frame is not None and not frame.empty:
                for ticker, chunk in frame.groupby("ticker", sort=False):
                    rows_written += db_writer_kr.upsert_financials(
                        chunk.reset_index(drop=True), str(ticker), "kr"
                    )

            tickers_done += len(batch)
            rows_total += rows_written
            elapsed = time.time() - start_time
            batch_elapsed = time.time() - batch_start
            eta_sec = (elapsed / tickers_done) * (total - tickers_done) if tickers_done else 0

            pct = 100 * tickers_done / total
            print(
                f"[배치 {batch_idx+1:4d}/{total_batches}] "
                f"{batch[0]}~{batch[-1]} | "
                f"rows={rows_written:5d} | "
                f"진행 {tickers_done:4d}/{total} ({pct:5.1f}%) | "
                f"배치 {batch_elapsed:.1f}s | "
                f"ETA {eta_sec/60:.1f}m",
                flush=True,
            )

        except Exception as exc:
            print(f"[오류] 배치 {batch_idx+1} ({batch[0]}~{batch[-1]}): {exc}", flush=True)
            tickers_done += len(batch)
            continue

    elapsed_total = time.time() - start_time
    print("-" * 70)
    print(f"[완료] 총 {rows_total:,}행 upsert | {tickers_done}개 티커 처리 | {elapsed_total:.0f}초")

    db_writer_kr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
