"""
NASDAQ Healthy PIT 유니버스 구성
=================================
조건:
  (a) 가격 정상: Close/Adj Close가 all-NaN 아님
  (b) SEC companyfacts 정상: sec_companyfacts_quarterly.parquet 존재 + row > 0
  (c) common stock only: suffix 필터 (W, R, U, WS 등 제외)

산출물:
  data/universe/symbols_nasdaq_healthy_pit.csv
  logs/pit_verification/nasdaq_healthy_filter_stats.md

실행:
  ./.venv/bin/python scripts/build_healthy_universe.py
"""
from __future__ import annotations
import sys
import re
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

UNIVERSE_FILE = Path("data/universe/symbols_nasdaq_stock_only.csv")
PRICES_DIR = Path("data/prices/us")
FINANCIALS_DIR = Path("data/financials/us")
OUT_CSV = Path("data/universe/symbols_nasdaq_healthy_pit.csv")
OUT_STATS = Path("logs/pit_verification/nasdaq_healthy_filter_stats.md")
OUT_STATS.parent.mkdir(parents=True, exist_ok=True)

# Non-common-stock suffix 패턴
# 워런트, 유닛, 권리, 프리퍼드, SPAC관련 등
EXCLUDE_SUFFIX_PATTERN = re.compile(
    r"(W|WS|WI|R|U|UN|WT|RT|WU|A|B|C|D|E|P|PR|PA|PB|PC|PD|PE|PF|PG|PH|PI|PJ|PK|PL|PM)"
    r"$",
    re.IGNORECASE,
)
# 단, 1~2글자 suffix만 체크 — 뒤에 숫자 없는 경우만
EXCLUDE_SUFFIX_STRICT = re.compile(
    r"(W|WS|WI|WT|WU|WW)$|"         # warrants
    r"(R|RT|RI)$|"                    # rights
    r"(U|UN|UNIT|UNITS)$|"            # units
    r"([A-Z]\d*)(\.WS|\.W|\.U|\.R)$",# dot-suffix variants
    re.IGNORECASE,
)
# Security name 키워드 필터 (대소문자 무관)
EXCLUDE_NAME_KEYWORDS = [
    "warrant", "warrants", "right ", "rights ", " unit ", " units",
    "depositary share", "depositary receipt", "adr",
    "preferred", "preference share",
    "acquisition corp", "blank check",
    "spac", "special purpose",
]


def is_non_common(symbol: str, security_name: str = "") -> tuple[bool, str]:
    """True이면 제외. 이유도 반환."""
    sym = symbol.strip()
    name = security_name.lower() if security_name else ""

    # ticker suffix 체크
    if EXCLUDE_SUFFIX_STRICT.search(sym):
        return True, f"suffix:{sym}"

    # 이름 키워드 체크
    for kw in EXCLUDE_NAME_KEYWORDS:
        if kw in name:
            return True, f"name_keyword:{kw}"

    return False, ""


def check_price(ticker: str) -> tuple[bool, str]:
    p = PRICES_DIR / f"{ticker}.parquet"
    if not p.exists():
        return False, "no_price_file"
    try:
        df = pd.read_parquet(p, columns=["Close"])
        if df["Close"].notna().any():
            return True, "ok"
        return False, "all_nan_close"
    except Exception as e:
        return False, f"price_err:{e}"


def check_financials(ticker: str) -> tuple[bool, str]:
    p = FINANCIALS_DIR / ticker / "sec_companyfacts_quarterly.parquet"
    if not p.exists():
        return False, "no_companyfacts"
    try:
        df = pd.read_parquet(p)
        if len(df) > 0:
            return True, "ok"
        return False, "empty_companyfacts"
    except Exception as e:
        return False, f"financials_err:{e}"


def main() -> None:
    src = pd.read_csv(UNIVERSE_FILE)
    tickers_all = src["Symbol"].tolist()
    security_map = dict(zip(src["Symbol"], src.get("Security", pd.Series(dtype=str)).fillna("")))

    n_total = len(tickers_all)
    print(f"입력 티커: {n_total}개")

    results = []
    for i, ticker in enumerate(tickers_all):
        security_name = security_map.get(ticker, "")
        exc, exc_reason = is_non_common(ticker, security_name)
        if exc:
            results.append({"ticker": ticker, "pass_c": False, "reason_c": exc_reason,
                             "pass_a": None, "reason_a": None,
                             "pass_b": None, "reason_b": None, "healthy": False})
            if (i+1) % 500 == 0:
                print(f"  진행: {i+1}/{n_total}")
            continue

        ok_a, reason_a = check_price(ticker)
        ok_b, reason_b = check_financials(ticker)
        healthy = ok_a and ok_b
        results.append({
            "ticker": ticker, "pass_c": True, "reason_c": "",
            "pass_a": ok_a, "reason_a": reason_a,
            "pass_b": ok_b, "reason_b": reason_b,
            "healthy": healthy,
        })
        if (i+1) % 500 == 0:
            print(f"  진행: {i+1}/{n_total}")

    df = pd.DataFrame(results)

    # 통계
    n_fail_c = int((df["pass_c"] == False).sum())
    n_pass_c = n_total - n_fail_c
    df_postc = df[df["pass_c"] == True]
    n_fail_a = int((df_postc["pass_a"] == False).sum())
    n_pass_a = int((df_postc["pass_a"] == True).sum())
    n_fail_b = int((df_postc[df_postc["pass_a"] == True]["pass_b"] == False).sum())
    n_pass_b = int((df_postc[df_postc["pass_a"] == True]["pass_b"] == True).sum())
    n_healthy = int(df["healthy"].sum())

    # 저장
    healthy_tickers = df[df["healthy"]]["ticker"].tolist()
    healthy_df = src[src["Symbol"].isin(healthy_tickers)].copy()
    healthy_df.to_csv(OUT_CSV, index=False)
    print(f"\n저장: {OUT_CSV} ({n_healthy}개 티커)")

    # 통계 MD
    reason_a_counts = df_postc[df_postc["pass_a"] == False]["reason_a"].value_counts().head(10)
    reason_b_counts = df_postc[df_postc["pass_a"] == True][df_postc["pass_b"] == False]["reason_b"].value_counts().head(10)
    reason_c_counts = df[df["pass_c"] == False]["reason_c"].str.extract(r"^([^:]+)")[0].value_counts().head(5)

    lines = [
        "# NASDAQ Healthy PIT 유니버스 필터 통계",
        f"\n생성일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## 필터 단계별 통계\n",
        f"| 단계 | 설명 | 입력 | 통과 | 실패 |",
        f"|------|------|------|------|------|",
        f"| 전체 입력 | symbols_nasdaq_stock_only.csv | {n_total} | — | — |",
        f"| (c) Common stock | suffix/name 키워드 필터 | {n_total} | {n_pass_c} | {n_fail_c} |",
        f"| (a) 가격 정상 | Close not all-NaN | {n_pass_c} | {n_pass_a} | {n_fail_a} |",
        f"| (b) SEC companyfacts 정상 | parquet 존재 + row>0 | {n_pass_a} | {n_pass_b} | {n_fail_b} |",
        f"| **최종 healthy** | 3조건 모두 통과 | — | **{n_healthy}** | — |",
        "",
        "## (c) Common stock 필터 제외 사유 TOP",
        "",
        reason_c_counts.to_string() if not reason_c_counts.empty else "(없음)",
        "",
        "## (a) 가격 실패 사유 TOP",
        "",
        reason_a_counts.to_string() if not reason_a_counts.empty else "(없음)",
        "",
        "## (b) SEC companyfacts 실패 사유 TOP",
        "",
        reason_b_counts.to_string() if not reason_b_counts.empty else "(없음)",
        "",
        f"## 출력 파일\n\n- `{OUT_CSV}` ({n_healthy}개 티커)",
    ]

    OUT_STATS.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {OUT_STATS}")

    print(f"\n=== 결과 ===")
    print(f"전체 입력: {n_total}")
    print(f"(c) 통과(common stock): {n_pass_c}, 실패: {n_fail_c}")
    print(f"(a) 통과(가격 정상): {n_pass_a}, 실패: {n_fail_a}")
    print(f"(b) 통과(companyfacts): {n_pass_b}, 실패: {n_fail_b}")
    print(f"최종 healthy: {n_healthy}")


if __name__ == "__main__":
    main()
