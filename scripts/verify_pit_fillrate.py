"""
PIT 컬럼 fill-rate 검증 스크립트
====================================
ingest --force 완료 후 실행.
산출물:
  logs/pit_verification/sec_companyfacts_pit_fillrate.csv
  logs/pit_verification/sec_companyfacts_availability_method_hist.csv

실행:
  ./.venv/bin/python scripts/verify_pit_fillrate.py [--max-tickers N]
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUT_DIR = Path("logs/pit_verification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_FILE = Path("data/universe/symbols_nasdaq_stock_only.csv")
FINANCIALS_DIR = Path("data/financials/us")
PIT_COLS = ["AcceptedAt", "AvailableDate", "AvailabilityMethod"]


def main(max_tickers: int = 0) -> None:
    tickers = pd.read_csv(UNIVERSE_FILE)["Symbol"].tolist()
    if max_tickers > 0:
        tickers = tickers[:max_tickers]

    print(f"검증 대상: {len(tickers)}개 티커")

    rows = []
    method_counts: dict[str, int] = {}

    for i, ticker in enumerate(tickers):
        parquet_path = FINANCIALS_DIR / ticker / "sec_companyfacts_quarterly.parquet"
        if not parquet_path.exists():
            rows.append({
                "ticker": ticker, "status": "no_parquet", "n_rows": 0,
                "extractor_version": None,
                "has_AcceptedAt": False, "has_AvailableDate": False, "has_AvailabilityMethod": False,
                "accepted_at_fill_pct": 0.0, "available_date_fill_pct": 0.0,
                "fallback_pct": 100.0, "accepted_pct": 0.0,
            })
            continue

        try:
            df = pd.read_parquet(parquet_path)
            n = len(df)
            has_accepted = "AcceptedAt" in df.columns
            has_avail = "AvailableDate" in df.columns
            has_method = "AvailabilityMethod" in df.columns
            ev = int(df["ExtractorVersion"].max()) if "ExtractorVersion" in df.columns else None

            acc_fill = float(df["AcceptedAt"].notna().sum() / n * 100) if has_accepted and n > 0 else 0.0
            avail_fill = float(df["AvailableDate"].notna().sum() / n * 100) if has_avail and n > 0 else 0.0

            fallback_pct = 100.0
            accepted_pct = 0.0
            if has_method and n > 0:
                vc = df["AvailabilityMethod"].astype(str).value_counts()
                for method, cnt in vc.items():
                    method_counts[method] = method_counts.get(method, 0) + cnt
                fallback_pct = float(
                    df["AvailabilityMethod"].astype(str).str.lower().str.startswith("fallback").sum() / n * 100
                )
                accepted_pct = float(
                    df["AvailabilityMethod"].astype(str).str.lower().str.startswith("accepted").sum() / n * 100
                )

            rows.append({
                "ticker": ticker, "status": "ok", "n_rows": n,
                "extractor_version": ev,
                "has_AcceptedAt": has_accepted, "has_AvailableDate": has_avail,
                "has_AvailabilityMethod": has_method,
                "accepted_at_fill_pct": round(acc_fill, 2),
                "available_date_fill_pct": round(avail_fill, 2),
                "fallback_pct": round(fallback_pct, 2),
                "accepted_pct": round(accepted_pct, 2),
            })
        except Exception as e:
            rows.append({
                "ticker": ticker, "status": f"error:{e}", "n_rows": -1,
                "extractor_version": None,
                "has_AcceptedAt": False, "has_AvailableDate": False, "has_AvailabilityMethod": False,
                "accepted_at_fill_pct": 0.0, "available_date_fill_pct": 0.0,
                "fallback_pct": 100.0, "accepted_pct": 0.0,
            })

        if (i + 1) % 200 == 0:
            print(f"  진행: {i+1}/{len(tickers)}")

    df_out = pd.DataFrame(rows)

    # --- fill-rate CSV ---
    fillrate_path = OUT_DIR / "sec_companyfacts_pit_fillrate.csv"
    df_out.to_csv(fillrate_path, index=False)
    print(f"\n저장: {fillrate_path}")

    # --- AvailabilityMethod histogram CSV ---
    method_df = pd.DataFrame([
        {"AvailabilityMethod": k, "count": v}
        for k, v in sorted(method_counts.items(), key=lambda x: -x[1])
    ])
    hist_path = OUT_DIR / "sec_companyfacts_availability_method_hist.csv"
    method_df.to_csv(hist_path, index=False)
    print(f"저장: {hist_path}")

    # --- 요약 출력 ---
    ok = df_out[df_out["status"] == "ok"]
    n_ok = len(ok)
    n_total = len(df_out)
    print(f"\n=== 요약 ({n_ok}/{n_total}개 파일 정상) ===")
    print(f"AcceptedAt 컬럼 보유: {ok['has_AcceptedAt'].sum()}/{n_ok}")
    print(f"AvailableDate 컬럼 보유: {ok['has_AvailableDate'].sum()}/{n_ok}")
    print(f"AvailabilityMethod 컬럼 보유: {ok['has_AvailabilityMethod'].sum()}/{n_ok}")
    print(f"AcceptedAt 평균 fill%: {ok['accepted_at_fill_pct'].mean():.1f}%")
    print(f"AvailableDate 평균 fill%: {ok['available_date_fill_pct'].mean():.1f}%")
    print(f"fallback% 평균: {ok['fallback_pct'].mean():.1f}%")
    print(f"accepted% 평균: {ok['accepted_pct'].mean():.1f}%")
    print(f"\nExtractorVersion 분포:\n{ok['extractor_version'].value_counts().sort_index()}")
    print(f"\nAvailabilityMethod 분포:\n{method_df.to_string(index=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tickers", type=int, default=0,
                        help="0=전체, N=처음 N개만 (테스트용)")
    args = parser.parse_args()
    main(max_tickers=args.max_tickers)
