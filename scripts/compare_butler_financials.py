#!/usr/bin/env python3
"""
Butler Reference vs DB 재무제표 비교 스크립트.

Butler reference JSON (scripts/butler_reference/*.json) 데이터와
market_data_kr.duckdb의 financials_quarterly 테이블을 비교하여
매치율을 계산하고 불일치 패턴을 분석한다.

Usage:
    .venv/bin/python scripts/compare_butler_financials.py [--verbose] [--ticker TICKER]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

# ── 경로 설정 ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "market_data_kr.duckdb"
REF_DIR = ROOT / "scripts" / "butler_reference"

# ── Butler metric → DB column 매핑 ──────────────────────────────
# Butler CAPEX = PPE 취득만 (유형자산의 취득). 무형자산/투자부동산/사용권자산 미포함.
# DB "Capital Expenditure" = comprehensive (PPE+Intangible+InvestmentProperty+ROU).
# "PPE CapEx" 우선 사용, NULL이면 "Capital Expenditure" 폴백.
METRIC_MAP = {
    "매출액": "Revenue",
    "영업이익": "Operating Income",
    "순이익": "Net Income",
    "자산총계": "Total Assets",
    "부채총계": "Total Liabilities",
    "자본총계": "Shareholders Equity",
    "영업현금흐름": "Operating Cash Flow",
    "자본지출(CAPEX)": "Capital Expenditure",
}

# Additional columns to load for fallback/comparison
_EXTRA_DB_COLS = ["PPE CapEx", "Net Income Common"]

TOLERANCES = [0.05, 0.10, 0.15]

# Flow metrics where Butler might use TTM (Trailing Twelve Months) instead of quarterly
# BS items (Total Assets, Total Liabilities, Shareholders Equity) are always point-in-time.
_FLOW_METRICS = {"Revenue", "Operating Income", "Net Income", "Operating Cash Flow",
                 "Capital Expenditure", "PPE CapEx"}


def butler_period_to_date(p: str) -> str | None:
    """'2024.03' → '2024-03-31' (말일). Also handles '2016.03(16Q1)' format."""
    import calendar
    import re
    # Strip parenthetical suffix like "(16Q1)"
    clean = re.sub(r"\(.*?\)", "", p).strip()
    parts = clean.split(".")
    if len(parts) != 2:
        return None
    try:
        y, m = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    last_day = calendar.monthrange(y, m)[1]
    return f"{y:04d}-{m:02d}-{last_day:02d}"


def load_butler_refs(ref_dir: Path, ticker_filter: str | None = None) -> dict:
    """Load all butler reference JSONs. Returns {ticker: data}."""
    refs = {}
    for fp in sorted(ref_dir.glob("*.json")):
        ticker = fp.stem
        if ticker_filter and ticker != ticker_filter:
            continue
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        # Skip if no financials data
        if not data.get("financials", {}).get("p"):
            continue
        refs[ticker] = data
    return refs


def load_db_financials(con: duckdb.DuckDBPyConnection, tickers: list[str]) -> pd.DataFrame:
    """Load financials_quarterly for given tickers."""
    cols = ["ticker", "PeriodEnd"] + list(dict.fromkeys(list(METRIC_MAP.values()) + _EXTRA_DB_COLS))
    col_str = ", ".join(f'"{c}"' for c in cols)
    placeholders = ",".join(["?"] * len(tickers))
    sql = f"""
        SELECT {col_str}
        FROM financials_quarterly
        WHERE market = 'kr' AND ticker IN ({placeholders})
        ORDER BY ticker, "PeriodEnd"
    """
    df = con.execute(sql, tickers).fetchdf()
    df["PeriodEnd"] = pd.to_datetime(df["PeriodEnd"])
    return df


def _build_ttm_lookup(db_df: pd.DataFrame, ticker: str, db_col: str) -> dict[str, float]:
    """Build TTM (Trailing Twelve Months) lookup for a ticker/metric.

    Returns {period_end_str: ttm_value_in_원} where TTM = sum of last 4 quarters.
    """
    tk_df = db_df[db_df["ticker"] == ticker].sort_values("PeriodEnd").copy()
    tk_df["_val"] = pd.to_numeric(tk_df[db_col], errors="coerce")
    tk_df["_ttm"] = tk_df["_val"].rolling(window=4, min_periods=4).sum()
    result = {}
    for _, row in tk_df.iterrows():
        if pd.notna(row["_ttm"]):
            result[row["PeriodEnd"].strftime("%Y-%m-%d")] = row["_ttm"]
    return result


def detect_butler_mode(butler_data: dict, db_df: pd.DataFrame, ticker: str) -> str:
    """Detect if butler uses TTM or quarterly for flow metrics.

    Compares multiple flow metrics (Revenue, OCF, CAPEX) against both quarterly
    and TTM DB values, using majority vote. Returns 'ttm' or 'quarterly'.
    """
    import numpy as np

    fin = butler_data["financials"]
    periods = fin["p"]

    db_rows = db_df[db_df["ticker"] == ticker].copy()
    db_rows_str = db_rows.copy()
    db_rows_str["PeriodEnd"] = db_rows_str["PeriodEnd"].dt.strftime("%Y-%m-%d")
    db_quarterly = db_rows_str.set_index("PeriodEnd")

    # Test multiple flow metrics for more robust detection
    _detection_metrics = [
        ("매출액", "Revenue"),
        ("영업현금흐름", "Operating Cash Flow"),
        ("자본지출(CAPEX)", "Capital Expenditure"),
    ]

    ttm_votes = 0
    quarterly_votes = 0

    for butler_key, db_col in _detection_metrics:
        butler_vals = fin["d"].get(butler_key, [])
        if not butler_vals or len(butler_vals) != len(periods):
            continue

        ttm_lookup = _build_ttm_lookup(db_df, ticker, db_col)
        quarterly_errors = []
        ttm_errors = []

        for i, period_str in enumerate(periods):
            butler_val = butler_vals[i]
            if butler_val is None or butler_val == 0:
                continue
            date_str = butler_period_to_date(period_str)
            if date_str is None:
                continue

            # Quarterly
            if date_str in db_quarterly.index:
                db_val_raw = db_quarterly.loc[date_str, db_col] if db_col in db_quarterly.columns else pd.NA
                if isinstance(db_val_raw, pd.Series):
                    db_val_raw = db_val_raw.iloc[0]
                if pd.notna(db_val_raw):
                    quarterly_errors.append(abs(db_val_raw / 1e8 / butler_val - 1.0))

            # TTM
            if date_str in ttm_lookup:
                ttm_errors.append(abs(ttm_lookup[date_str] / 1e8 / butler_val - 1.0))

        if not quarterly_errors or not ttm_errors:
            continue

        q_median = np.median(quarterly_errors)
        t_median = np.median(ttm_errors)

        if t_median < q_median * 0.5 and t_median < 0.15:
            ttm_votes += 1
        elif q_median < t_median * 0.5 and q_median < 0.15:
            quarterly_votes += 1

    if ttm_votes > quarterly_votes:
        return "ttm"
    return "quarterly"


def load_db_segments(con: duckdb.DuckDBPyConnection, tickers: list[str]) -> pd.DataFrame:
    """Load segment revenue from derived view."""
    placeholders = ",".join(["?"] * len(tickers))
    sql = f"""
        SELECT ticker, period_end, segment_type, segment_name, metric, value
        FROM segment_revenue_quarterly_derived
        WHERE market = 'kr'
          AND ticker IN ({placeholders})
          AND metric = 'revenue'
        ORDER BY ticker, period_end, segment_type, segment_name
    """
    df = con.execute(sql, tickers).fetchdf()
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.strftime("%Y-%m-%d")
    return df


def compare_value(butler_val, db_val, tol: float) -> bool:
    """Compare two values within tolerance."""
    if butler_val is None or db_val is None:
        return False
    if pd.isna(butler_val) or pd.isna(db_val):
        return False
    if butler_val == 0 and db_val == 0:
        return True
    if butler_val == 0:
        return abs(db_val) < 1  # 억원 단위에서 1 미만
    ratio = db_val / butler_val
    return abs(ratio - 1.0) <= tol


def _compare_metric_single_mode(
    butler_values: list, periods: list[str], db_col: str,
    db_rows: pd.DataFrame, ttm_lookup: dict[str, float],
    ppe_fallback_ttm: dict[str, float],
    use_ttm: bool,
) -> dict:
    """Compare a single metric in either quarterly or TTM mode."""
    total = 0
    matches = {t: 0 for t in TOLERANCES}
    mismatches = []

    for i, period_str in enumerate(periods):
        butler_val = butler_values[i]
        if butler_val is None:
            continue
        date_str = butler_period_to_date(period_str)
        if date_str is None:
            continue

        if use_ttm:
            ttm_val = ttm_lookup.get(date_str)
            if ttm_val is None and ppe_fallback_ttm:
                ttm_val = ppe_fallback_ttm.get(date_str)
            if ttm_val is None:
                if date_str not in db_rows.index:
                    total += 1
                    mismatches.append((period_str, butler_val, None, "MISSING"))
                else:
                    total += 1
                    mismatches.append((period_str, butler_val, None, "TTM_INSUFF"))
                continue
            db_val = ttm_val / 1e8
        else:
            if date_str not in db_rows.index:
                total += 1
                mismatches.append((period_str, butler_val, None, "MISSING"))
                continue
            db_val_raw = db_rows.loc[date_str, db_col]
            if isinstance(db_val_raw, pd.Series):
                db_val_raw = db_val_raw.iloc[0]
            if pd.isna(db_val_raw) and db_col == "Capital Expenditure":
                db_val_raw = db_rows.loc[date_str, "PPE CapEx"] if "PPE CapEx" in db_rows.columns else pd.NA
                if isinstance(db_val_raw, pd.Series):
                    db_val_raw = db_val_raw.iloc[0]
            if pd.isna(db_val_raw):
                total += 1
                mismatches.append((period_str, butler_val, None, "DB_NULL"))
                continue
            db_val = db_val_raw / 1e8

        total += 1
        matched_any = False
        for tol in TOLERANCES:
            if compare_value(butler_val, db_val, tol):
                matches[tol] += 1
                matched_any = True
        if not matched_any:
            if butler_val != 0:
                err_pct = (db_val / butler_val - 1) * 100
            else:
                err_pct = float("inf")
            mismatches.append((period_str, butler_val, db_val, f"{err_pct:+.1f}%"))

    return {"total": total, "matches": matches, "mismatches": mismatches}


def compare_financials(butler_data: dict, db_df: pd.DataFrame, ticker: str, mode: str = "quarterly"):
    """Compare financials for one ticker. Returns per-metric results.

    For flow metrics, tries both quarterly and TTM modes and picks whichever
    achieves a higher match rate at ±10%.
    """
    fin = butler_data["financials"]
    periods = fin["p"]
    metrics_data = fin["d"]

    # Build DB lookup
    db_rows_ts = db_df[db_df["ticker"] == ticker].copy()
    db_rows_str = db_rows_ts.copy()
    db_rows_str["PeriodEnd"] = db_rows_str["PeriodEnd"].dt.strftime("%Y-%m-%d")
    db_rows = db_rows_str.set_index("PeriodEnd")

    # Pre-build TTM lookups for all flow metrics (always, for best-fit selection)
    ttm_lookups: dict[str, dict[str, float]] = {}
    for butler_key, db_col in METRIC_MAP.items():
        if db_col in _FLOW_METRICS:
            ttm_lookups[db_col] = _build_ttm_lookup(db_df, ticker, db_col)
    if "PPE CapEx" not in ttm_lookups:
        ttm_lookups["PPE CapEx"] = _build_ttm_lookup(db_df, ticker, "PPE CapEx")

    results = {}

    for butler_key, db_col in METRIC_MAP.items():
        if butler_key not in metrics_data:
            continue
        values = metrics_data[butler_key]
        if len(values) != len(periods):
            continue

        is_flow = db_col in _FLOW_METRICS
        ppe_fallback = ttm_lookups.get("PPE CapEx", {}) if db_col == "Capital Expenditure" else {}

        if is_flow:
            # Try both modes and pick better one
            q_result = _compare_metric_single_mode(
                values, periods, db_col, db_rows,
                {}, {}, use_ttm=False,
            )
            t_result = _compare_metric_single_mode(
                values, periods, db_col, db_rows,
                ttm_lookups.get(db_col, {}), ppe_fallback, use_ttm=True,
            )
            # Pick mode with more ±10% matches
            q_rate = q_result["matches"][0.10] / q_result["total"] if q_result["total"] else 0
            t_rate = t_result["matches"][0.10] / t_result["total"] if t_result["total"] else 0
            results[butler_key] = t_result if t_rate > q_rate else q_result
        else:
            # BS metrics: always quarterly (point-in-time)
            results[butler_key] = _compare_metric_single_mode(
                values, periods, db_col, db_rows,
                {}, {}, use_ttm=False,
            )

    return results


def _normalize_seg_name(name: str) -> str:
    """Normalize segment name for fuzzy matching."""
    import re
    s = str(name).strip()
    # Remove trailing whitespace/tabs
    s = s.strip()
    # Remove HTML artifacts (&cr = line break in DART XML)
    s = re.sub(r"&cr", "", s, flags=re.IGNORECASE)
    # Remove common suffixes: 사업부, 부문, 사업, 매출
    s = re.sub(r"(사업부|부문|사업|매출|부분)$", "", s)
    # Remove all whitespace
    s = re.sub(r"\s+", "", s)
    # Normalize punctuation: ㆍ → ·, remove dots/commas
    s = s.replace("ㆍ", "").replace("·", "").replace(",", "").replace(".", "")
    # Remove parenthetical suffixes
    s = re.sub(r"\(.*?\)$", "", s)
    # Remove leading parenthetical prefixes: (석유화학) → 석유화학
    s = re.sub(r"^\((.+?)\)", r"\1", s)
    # Common typo normalization
    s = s.replace("비지니스", "비즈니스")
    s = s.replace("컨텐츠", "콘텐츠")
    # Remove company prefixes: LG, SK, 삼성, 현대 etc.
    s = re.sub(r"^(lg|sk|삼성|현대|한화|롯데|포스코|cj)", "", s, flags=re.IGNORECASE)
    # Normalize "및" and "and" to empty (라인및기타 → 라인기타)
    s = s.replace("및", "").replace("and", "")
    # Remove trailing 등 (TV, 모니터등 → TV,모니터)
    s = re.sub(r"등$", "", s)
    # Lowercase for English names (NAND Flash → nandflash)
    s = s.lower()
    return s.strip()


def compare_segments(butler_data: dict, db_seg_df: pd.DataFrame, ticker: str):
    """Compare business segment revenue for one ticker."""
    biz = butler_data.get("business", {}).get("매출액")
    if not biz or not biz.get("p") or not biz.get("s"):
        return None

    periods = biz["p"]
    segments = biz["s"]  # [[name, [values]], ...]

    # DB segments for this ticker — search all segment types, not just business,
    # because some segments may be classified as product/geographic in DB
    db_biz = db_seg_df[
        (db_seg_df["ticker"] == ticker)
        & (db_seg_df["segment_type"].isin(["business", "product"]))
    ]

    # Build DB lookup: exact name → value, normalized name → value
    db_lookup = {}
    db_norm_lookup = {}  # (period, normalized_name) → value
    for _, row in db_biz.iterrows():
        key = (row["period_end"], row["segment_name"])
        db_lookup[key] = row["value"]
        norm_key = (row["period_end"], _normalize_seg_name(row["segment_name"]))
        # Keep first (or largest) value for normalized key
        if norm_key not in db_norm_lookup:
            db_norm_lookup[norm_key] = row["value"]

    # Get all DB segment names per period
    db_names_by_period = defaultdict(set)
    for _, row in db_biz.iterrows():
        db_names_by_period[row["period_end"]].add(row["segment_name"])

    total = 0
    matches = {t: 0 for t in TOLERANCES}
    mismatches = []
    name_mismatches = 0

    for seg_name, seg_values in segments:
        if seg_name in ("합계", "Total", "계"):
            continue
        butler_norm = _normalize_seg_name(seg_name)
        for i, period_str in enumerate(periods):
            if i >= len(seg_values):
                break
            butler_val = seg_values[i]
            if butler_val is None or butler_val == 0:
                continue

            date_str = butler_period_to_date(period_str)
            if date_str is None:
                continue

            # Try exact name match first, then normalized match, then partial match
            db_val_raw = db_lookup.get((date_str, seg_name))
            if db_val_raw is None:
                db_val_raw = db_norm_lookup.get((date_str, butler_norm))
            if db_val_raw is None:
                # Partial match: Butler norm is substring of DB norm or vice versa
                # Require minimum 3 chars to avoid over-matching (기타→기타금융)
                for (d, dn), dv in db_norm_lookup.items():
                    if d != date_str:
                        continue
                    if len(butler_norm) >= 3 and len(dn) >= 3:
                        if butler_norm in dn or dn in butler_norm:
                            db_val_raw = dv
                            break

            if db_val_raw is None:
                total += 1
                name_mismatches += 1
                continue

            db_val = db_val_raw / 1e8  # 원 → 억원
            total += 1

            matched_any = False
            for tol in TOLERANCES:
                if compare_value(butler_val, db_val, tol):
                    matches[tol] += 1
                    matched_any = True

            if not matched_any:
                if butler_val != 0:
                    err_pct = (db_val / butler_val - 1) * 100
                else:
                    err_pct = float("inf")
                mismatches.append((period_str, seg_name, butler_val, db_val, f"{err_pct:+.1f}%"))

    return {
        "total": total,
        "matches": matches,
        "mismatches": mismatches,
        "name_mismatches": name_mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description="Butler vs DB financials comparison")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--ticker", "-t", type=str, default=None)
    args = parser.parse_args()

    # Load butler references
    print("Loading butler reference files...")
    refs = load_butler_refs(REF_DIR, args.ticker)
    print(f"  → {len(refs)} tickers with financial data")

    if not refs:
        print("No butler reference files found.")
        sys.exit(1)

    tickers = sorted(refs.keys())

    # Connect to DB
    print(f"Connecting to {DB_PATH}...")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Load DB data
    print("Loading DB financials...")
    db_fin = load_db_financials(con, tickers)
    db_tickers = set(db_fin["ticker"].unique())
    print(f"  → {len(db_tickers)} tickers found in DB")

    print("Loading DB segments...")
    db_seg = load_db_segments(con, tickers)
    print(f"  → {len(db_seg)} segment rows loaded")

    con.close()

    # ── Detect butler mode per ticker ─────────────────────────
    print("\nDetecting butler data mode (quarterly vs TTM)...")
    ticker_modes: dict[str, str] = {}
    for ticker in tickers:
        if ticker not in db_tickers:
            continue
        mode = detect_butler_mode(refs[ticker], db_fin, ticker)
        ticker_modes[ticker] = mode

    ttm_count = sum(1 for m in ticker_modes.values() if m == "ttm")
    quarterly_count = sum(1 for m in ticker_modes.values() if m == "quarterly")
    print(f"  → TTM: {ttm_count} tickers, Quarterly: {quarterly_count} tickers")
    print("  (Note: compare_financials also tries both modes per-metric for best fit)")

    # ── Compare financials ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINANCIALS COMPARISON")
    print("=" * 80)

    # Aggregated stats
    agg_metric = defaultdict(lambda: {"total": 0, "matches": {t: 0 for t in TOLERANCES}})
    agg_period = defaultdict(lambda: {"total": 0, "matches": {t: 0 for t in TOLERANCES}})
    ticker_scores = {}  # ticker → overall match rate at 10%
    low_match_tickers = []

    # Cache results to avoid double computation
    cached_results: dict[str, dict] = {}

    for ticker in tickers:
        if ticker not in db_tickers:
            continue

        mode = ticker_modes.get(ticker, "quarterly")
        result = compare_financials(refs[ticker], db_fin, ticker, mode=mode)
        cached_results[ticker] = result

        # Per-ticker aggregate
        tk_total = 0
        tk_match_15 = 0
        tk_match_10 = 0
        tk_match_5 = 0

        for metric_kr, mdata in result.items():
            agg_metric[metric_kr]["total"] += mdata["total"]
            for tol in TOLERANCES:
                agg_metric[metric_kr]["matches"][tol] += mdata["matches"][tol]

            tk_total += mdata["total"]
            tk_match_5 += mdata["matches"][0.05]
            tk_match_10 += mdata["matches"][0.10]
            tk_match_15 += mdata["matches"][0.15]

            # Period aggregation
            for mm in mdata["mismatches"]:
                period_str = mm[0]
                agg_period[period_str]["total"] += 1

        if tk_total > 0:
            rate_15 = tk_match_15 / tk_total * 100
            rate_10 = tk_match_10 / tk_total * 100
            ticker_scores[ticker] = rate_10

            if rate_15 < 80:
                corp_name = refs[ticker].get("corp_name", "")
                low_match_tickers.append((ticker, corp_name, rate_15, rate_10, tk_total, result))

    # ── Mismatch type breakdown ───────────────────────────���──
    miss_type_counts = defaultdict(lambda: defaultdict(int))  # metric → {MISSING, DB_NULL, VALUE} → count
    for ticker in tickers:
        if ticker not in db_tickers:
            continue
        result = cached_results.get(ticker, {})
        for metric_kr, mdata in result.items():
            for mm in mdata["mismatches"]:
                tag = mm[3] if len(mm) == 4 else "?"
                if tag in ("MISSING", "TTM_INSUFF"):
                    miss_type_counts[metric_kr]["MISSING"] += 1
                elif tag == "DB_NULL":
                    miss_type_counts[metric_kr]["DB_NULL"] += 1
                else:
                    miss_type_counts[metric_kr]["VALUE"] += 1

    # ── Print per-metric summary ──────────────────────────────
    print(f"\n{'Metric':<20} {'Total':>7} {'±5%':>10} {'±10%':>10} {'±15%':>10}")
    print("-" * 60)
    overall_total = 0
    overall_matches = {t: 0 for t in TOLERANCES}

    for metric_kr in METRIC_MAP.keys():
        if metric_kr not in agg_metric:
            continue
        d = agg_metric[metric_kr]
        total = d["total"]
        overall_total += total
        row = f"{metric_kr:<20} {total:>7}"
        for tol in TOLERANCES:
            m = d["matches"][tol]
            overall_matches[tol] += m
            pct = m / total * 100 if total > 0 else 0
            row += f" {pct:>8.1f}%"
        print(row)

    print("-" * 60)
    row = f"{'OVERALL':<20} {overall_total:>7}"
    for tol in TOLERANCES:
        m = overall_matches[tol]
        pct = m / overall_total * 100 if overall_total > 0 else 0
        row += f" {pct:>8.1f}%"
    print(row)

    # ── Mismatch type breakdown table ──────────────────────────
    total_missing = sum(v["MISSING"] for v in miss_type_counts.values())
    total_null = sum(v["DB_NULL"] for v in miss_type_counts.values())
    total_value = sum(v["VALUE"] for v in miss_type_counts.values())
    total_mismatch = total_missing + total_null + total_value

    print(f"\n--- Mismatch breakdown (total: {total_mismatch}) ---")
    print(f"  {'Metric':<20} {'MISSING':>8} {'DB_NULL':>8} {'VALUE':>8}")
    print(f"  {'-'*46}")
    for metric_kr in METRIC_MAP.keys():
        mc = miss_type_counts.get(metric_kr, {})
        print(f"  {metric_kr:<20} {mc.get('MISSING',0):>8} {mc.get('DB_NULL',0):>8} {mc.get('VALUE',0):>8}")
    print(f"  {'-'*46}")
    print(f"  {'TOTAL':<20} {total_missing:>8} {total_null:>8} {total_value:>8}")
    if total_mismatch > 0:
        print(f"  MISSING={total_missing} ({total_missing/total_mismatch*100:.1f}%)  DB_NULL={total_null} ({total_null/total_mismatch*100:.1f}%)  VALUE={total_value} ({total_value/total_mismatch*100:.1f}%)")
    else:
        print(f"  MISSING={total_missing}  DB_NULL={total_null}  VALUE={total_value}  (no mismatches)")

    # ── Value-only match rates (excluding MISSING/DB_NULL) ────
    print(f"\n--- Value-only match rates (excluding MISSING & DB_NULL periods) ---")
    print(f"{'Metric':<20} {'Total':>7} {'±5%':>10} {'±10%':>10} {'±15%':>10}")
    print("-" * 60)
    vo_overall_total = 0
    vo_overall_matches = {t: 0 for t in TOLERANCES}
    for metric_kr in METRIC_MAP.keys():
        if metric_kr not in agg_metric:
            continue
        d = agg_metric[metric_kr]
        mc = miss_type_counts.get(metric_kr, {})
        excluded = mc.get("MISSING", 0) + mc.get("DB_NULL", 0)
        total = d["total"] - excluded
        vo_overall_total += total
        row = f"{metric_kr:<20} {total:>7}"
        for tol in TOLERANCES:
            m = d["matches"][tol]
            vo_overall_matches[tol] += m
            pct = m / total * 100 if total > 0 else 0
            row += f" {pct:>8.1f}%"
        print(row)
    print("-" * 60)
    row = f"{'OVERALL (val-only)':<20} {vo_overall_total:>7}"
    for tol in TOLERANCES:
        m = vo_overall_matches[tol]
        pct = m / vo_overall_total * 100 if vo_overall_total > 0 else 0
        row += f" {pct:>8.1f}%"
    print(row)

    # ── Ticker distribution ─────────────────────────────────────
    print(f"\n--- Ticker match rate distribution (±10%, {len(ticker_scores)} tickers) ---")
    buckets = [0, 50, 60, 70, 80, 90, 95, 100, 101]
    bucket_labels = ["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-95%", "95-100%", "100%"]
    bucket_counts = [0] * len(bucket_labels)
    for score in ticker_scores.values():
        for i in range(len(buckets) - 1):
            if buckets[i] <= score < buckets[i + 1]:
                bucket_counts[i] += 1
                break
    for label, count in zip(bucket_labels, bucket_counts):
        bar = "#" * (count // 2)
        print(f"  {label:>10}: {count:>4}  {bar}")

    # ── Low-match tickers ───────────────────────────────────────
    print(f"\n--- Low-match tickers (<80% at ±15%): {len(low_match_tickers)} tickers ---")
    low_match_tickers.sort(key=lambda x: x[2])

    for ticker, corp_name, rate_15, rate_10, total, result in low_match_tickers[:50]:
        print(f"\n  [{ticker}] {corp_name}  ±10%={rate_10:.1f}%  ±15%={rate_15:.1f}%  (n={total})")
        for metric_kr, mdata in result.items():
            if not mdata["mismatches"]:
                continue
            mt = mdata["total"]
            m15 = mdata["matches"][0.15]
            pct = m15 / mt * 100 if mt > 0 else 0
            if pct < 80:
                # Show first few mismatches
                print(f"    {metric_kr}: {m15}/{mt} ({pct:.0f}%)")
                for mm in mdata["mismatches"][:3]:
                    if len(mm) == 4:
                        period, bval, dval, tag = mm
                        if dval is not None:
                            print(f"      {period}: butler={bval:,.0f} db={dval:,.0f} ({tag})")
                        else:
                            print(f"      {period}: butler={bval:,.0f} ({tag})")
                if len(mdata["mismatches"]) > 3:
                    print(f"      ... +{len(mdata['mismatches']) - 3} more")

    if len(low_match_tickers) > 50:
        print(f"\n  ... +{len(low_match_tickers) - 50} more tickers not shown")

    # ── Top failure patterns ────────────────────────────────────
    print("\n--- Most common failure metrics (by mismatch count) ---")
    metric_fail_count = defaultdict(int)
    for ticker in tickers:
        if ticker not in db_tickers:
            continue
        result = cached_results.get(ticker, {})
        for metric_kr, mdata in result.items():
            metric_fail_count[metric_kr] += len(mdata["mismatches"])

    for metric_kr, count in sorted(metric_fail_count.items(), key=lambda x: -x[1]):
        print(f"  {metric_kr:<25} {count:>6} mismatches")

    # ── Period failure analysis ─────────────────────────────────
    print("\n--- Top 20 periods with most mismatches ---")
    period_fail = sorted(agg_period.items(), key=lambda x: -x[1]["total"])[:20]
    for period_str, pdata in period_fail:
        print(f"  {period_str}: {pdata['total']} mismatches")

    # ── Segment comparison ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("BUSINESS SEGMENT COMPARISON (매출액)")
    print("=" * 80)

    seg_total = 0
    seg_matches = {t: 0 for t in TOLERANCES}
    seg_name_mismatches = 0
    seg_tickers_compared = 0
    seg_low_tickers = []

    for ticker in tickers:
        seg_result = compare_segments(refs[ticker], db_seg, ticker)
        if seg_result is None or seg_result["total"] == 0:
            continue

        seg_tickers_compared += 1
        seg_total += seg_result["total"]
        seg_name_mismatches += seg_result["name_mismatches"]
        for tol in TOLERANCES:
            seg_matches[tol] += seg_result["matches"][tol]

        # Track low match tickers
        rate_15 = seg_result["matches"][0.15] / seg_result["total"] * 100 if seg_result["total"] > 0 else 0
        if rate_15 < 60 and seg_result["total"] >= 5:
            corp_name = refs[ticker].get("corp_name", "")
            seg_low_tickers.append((ticker, corp_name, rate_15, seg_result))

    print(f"\nTickers compared: {seg_tickers_compared}")
    print(f"Total comparisons: {seg_total}")
    print(f"Name mismatches (seg name not in DB): {seg_name_mismatches}")

    if seg_total > 0:
        print(f"\n{'Metric':<20} {'Total':>7} {'±5%':>10} {'±10%':>10} {'±15%':>10}")
        print("-" * 60)
        row = f"{'Segment Revenue':<20} {seg_total:>7}"
        for tol in TOLERANCES:
            m = seg_matches[tol]
            pct = m / seg_total * 100
            row += f" {pct:>8.1f}%"
        print(row)

        # Name match vs value match breakdown
        value_comparisons = seg_total - seg_name_mismatches
        if value_comparisons > 0:
            print(f"\n  (Excluding name mismatches: {value_comparisons} comparisons)")
            row = f"  {'Value-only':<18} {value_comparisons:>7}"
            for tol in TOLERANCES:
                m = seg_matches[tol]
                pct = m / value_comparisons * 100
                row += f" {pct:>8.1f}%"
            print(row)

    # Low segment match tickers
    if seg_low_tickers:
        print(f"\n--- Low segment match tickers (<60% at ±15%): {len(seg_low_tickers)} ---")
        seg_low_tickers.sort(key=lambda x: x[2])
        for ticker, corp_name, rate_15, seg_result in seg_low_tickers[:20]:
            n = seg_result["total"]
            nm = seg_result["name_mismatches"]
            print(f"  [{ticker}] {corp_name}  ±15%={rate_15:.0f}%  n={n}  name_miss={nm}")
            for mm in seg_result["mismatches"][:3]:
                period, seg_name, bval, dval, tag = mm
                print(f"    {period} {seg_name}: butler={bval:,.0f} db={dval:,.0f} ({tag})")

    # ── Final summary ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Butler refs loaded:    {len(refs)}")
    print(f"Tickers in DB:         {len(db_tickers)}")
    print(f"Tickers compared:      {len(ticker_scores)}")

    if overall_total > 0:
        for tol in TOLERANCES:
            pct = overall_matches[tol] / overall_total * 100
            print(f"Overall match ±{tol*100:.0f}%:    {pct:.1f}% ({overall_matches[tol]}/{overall_total})")

    if ticker_scores:
        n_good = sum(1 for s in ticker_scores.values() if s >= 80)
        print(f"Tickers ≥80% at ±10%:  {n_good}/{len(ticker_scores)} ({n_good/len(ticker_scores)*100:.1f}%)")
    print(f"Low-match tickers:     {len(low_match_tickers)} (<80% at ±15%)")


if __name__ == "__main__":
    main()
