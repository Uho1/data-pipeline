#!/usr/bin/env python3
"""Compare DART capacity extraction results against Butler reference data.

Usage:
    .venv/bin/python scripts/compare_capacity_butler.py [--tickers 005930,000660]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_data.db_reader_kr import load_capacity_production_from_db

BUTLER_REF_DIR = Path(__file__).parent / "butler_reference"

# Tickers with fundamentally different data definitions between DART and Butler.
# 005160: DART reports per-plant capacity, Butler uses consolidated totals with
#         different scope (22% match at ±15%)
# 096770: Subsidiary-level aggregation differs (SK에너지 vs SK인천석유화학),
#         unit/product definition mismatches (35% match at ±15%)
STRUCTURAL_EXCLUDE = {"005160", "096770"}


def _unit_multiplier(unit_str: str, section: str = "") -> float:
    """Convert display unit to a multiplier for base unit comparison.

    E.g., '백만원' → 1_000_000, '천대' → 1_000, '억원' → 100_000_000.
    For 가동률 (utilization rate), always return 1.0 (percentage as-is).
    """
    if not unit_str:
        return 1.0
    if section == "가동률":
        return 1.0
    u = re.sub(r"[\s\u3000\xa0]+", "", str(unit_str).strip())
    # Remove % indicator (mixed units like "천Ton, %" have both)
    u_no_pct = re.sub(r",?\s*%", "", u)
    # Remove trailing text after comma (e.g., "시간, C/S" → "시간")
    u_no_pct = re.split(r"[,，]", u_no_pct)[0].strip()
    if not u_no_pct:
        return 1.0
    # Common DART unit prefixes — ordered longest-first to avoid partial matches
    if "조" in u_no_pct:
        return 1e12
    if "억" in u_no_pct:
        return 1e8
    if "백만" in u_no_pct:
        return 1e6
    if "만" in u_no_pct:
        return 1e4
    if "천" in u_no_pct:
        return 1e3
    return 1.0


def _is_comparable_unit(dart_unit: str, section: str, butler_expected_unit: str = "") -> bool:
    """Check if the DART unit is comparable with Butler's base unit.

    Returns False for units that are fundamentally different from Butler's
    expected measurement (e.g., '시간' when Butler expects volume or currency).
    """
    if section == "가동률":
        return True
    if not dart_unit:
        return True  # assume comparable if unknown
    u = re.sub(r"[\s\u3000\xa0]+", "", str(dart_unit).strip())
    # Units that represent time — incomparable with volume/currency
    if re.match(r'^시간', u):
        return False
    # If Butler specifies an expected unit, check compatibility
    if butler_expected_unit:
        bu = butler_expected_unit.lower()
        du = u.lower()
        # Currency unit mismatch: Butler expects physical but DART has monetary
        if bu in ("대", "톤", "ton", "m/t", "kl", "㎘", "개", "mwh") and "원" in du:
            return False
        # Physical unit mismatch: Butler expects monetary but DART has physical
        if "원" in bu and "원" not in du and du not in ("", "%"):
            return False
    return True


def _normalize_product_name(name: str) -> str:
    """Normalize product names for matching between Butler and DART."""
    n = re.sub(r"[\s\u3000\xa0]+", " ", str(name)).strip()
    # Remove common suffixes/prefixes
    n = re.sub(r"\s*부문$", "", n)
    n = re.sub(r"\s*부문\s+", " ", n)
    # Remove ALL parenthetical content (unit indicators, sub-categories, etc.)
    n = re.sub(r"\s*\([^)]*\)", " ", n)
    # Remove &cr (carriage return artifacts in DART HTML)
    n = re.sub(r"&cr", " ", n)
    # Normalize dashes/hyphens to spaces
    n = re.sub(r"[-–—]", " ", n)
    # Remove 공장 (factory) anywhere for better matching
    n = re.sub(r"공장", " ", n)
    # Collapse multiple spaces
    n = re.sub(r"\s+", " ", n).strip()
    return n


# Business name aliases: map equivalent names across different filing periods
_BUSINESS_NAME_ALIASES = {
    "건장재사업": "하우징사업",
    "하우징사업": "건장재사업",
    "컬러": "칼라",
    "칼라": "컬러",
}


def _load_butler_capacity(ticker: str) -> dict | None:
    """Load Butler capacity data for a ticker."""
    fpath = BUTLER_REF_DIR / f"{ticker}.json"
    if not fpath.exists():
        return None
    data = json.loads(fpath.read_text(encoding="utf-8"))
    cap = data.get("capacity")
    if not cap:
        return None
    return cap


def compare_ticker(ticker: str) -> dict:
    """Compare DART vs Butler capacity for one ticker.

    Returns summary dict with match statistics.
    """
    butler = _load_butler_capacity(ticker)
    dart_df = load_capacity_production_from_db(ticker)

    result = {
        "ticker": ticker,
        "butler_has_data": butler is not None,
        "dart_rows": 0 if dart_df is None else len(dart_df),
        "comparisons": 0,
        "match_5pct": 0,
        "match_10pct": 0,
        "match_15pct": 0,
        "details": [],
    }

    if butler is None or dart_df is None or dart_df.empty:
        return result

    # Butler data structure:
    #   periods: ["2017.03", "2017.06", ...]
    #   생산능력: [["DX-TV (대)", [val1, val2, ...]], ...]
    #   생산실적: [["DX-TV (대)", [val1, val2, ...]], ...]
    #   가동률: [["DX-TV", [89.5, ...]], ...]
    butler_periods = butler.get("periods", [])
    if not butler_periods:
        return result

    # Convert butler periods to dates: "2025.12" → "2025-12-31"
    def _butler_period_to_date(p: str) -> str:
        parts = p.split(".")
        if len(parts) != 2:
            return ""
        y, m = int(parts[0]), int(parts[1])
        # End of quarter
        import calendar
        _, last_day = calendar.monthrange(y, m)
        return f"{y:04d}-{m:02d}-{last_day:02d}"

    for section_name in ("생산능력", "생산실적", "가동률"):
        butler_section = butler.get(section_name, [])
        if not butler_section:
            continue

        dart_section = dart_df[dart_df["section"] == section_name]
        if dart_section.empty:
            continue

        for product_name, values in butler_section:
            butler_product = _normalize_product_name(product_name)

            # Try to match DART product names — prefer aggregate (합계) when
            # Butler name is short (aggregate-level), otherwise prefer most specific.
            dart_products = dart_section["product_name"].unique()
            bp_norm = butler_product.lower()

            # Extract expected unit from Butler product name: "(kl)", "(대)" etc.
            _butler_unit_m = re.search(r'\(([^)]+)\)\s*$', product_name)
            _butler_expected_unit = _butler_unit_m.group(1).lower() if _butler_unit_m else ""

            candidates: list[tuple[str, int, bool]] = []  # (dart_product, score, has_total)

            # Build list of Butler name variants (original + aliases)
            bp_variants = [bp_norm]
            # Only add aliases for specific business name changes
            for alias_from, alias_to in _BUSINESS_NAME_ALIASES.items():
                if alias_from in bp_norm and alias_to not in bp_norm:
                    bp_variants.append(bp_norm.replace(alias_from, alias_to))

            for dp in dart_products:
                dp_norm = _normalize_product_name(dp).lower()
                # Require meaningful overlap — at least 2 chars of the shorter name
                overlap_len = min(len(bp_norm), len(dp_norm))
                if overlap_len < 2:
                    continue

                # --- Fix A: For 가동률 section, prefer product names containing
                # "가동률" or "가 동 률" over plain names (which may be 가동월수) ---
                if section_name == "가동률":
                    dp_collapsed = re.sub(r"\s+", "", dp)
                    if "가동률" not in dp_collapsed:
                        base_collapsed = re.sub(r"\s+", "", dp_norm)
                        has_rate_sibling = any(
                            "가동률" in re.sub(r"\s+", "", other)
                            and base_collapsed in re.sub(r"\s+", "", _normalize_product_name(other).lower())
                            for other in dart_products if other != dp
                        )
                        if has_rate_sibling:
                            continue

                # Substring match with overlap ratio filter — try all Butler name variants
                matched = False
                for bpv in bp_variants:
                    if bpv in dp_norm or dp_norm in bpv:
                        matched = True
                        break

                if matched:
                    # Word-boundary check: if dp_norm is embedded within a
                    # word in bp_norm (not at a word boundary), reject the
                    # match.  E.g., "시멘트" should not match "시멘트전용선"
                    # but should match "삼척 시멘트".
                    word_boundary_ok = True
                    if dp_norm in bp_norm and dp_norm != bp_norm:
                        for bpv in bp_variants:
                            pos = bpv.find(dp_norm)
                            if pos >= 0:
                                end = pos + len(dp_norm)
                                at_start = (pos == 0 or bpv[pos - 1] == " ")
                                at_end = (end == len(bpv) or bpv[end] == " ")
                                if at_start and not at_end:
                                    # dp is a prefix of a longer word in bp
                                    # Find the containing word
                                    word_end = bpv.find(" ", end)
                                    if word_end == -1:
                                        word_end = len(bpv)
                                    containing_word = bpv[pos:word_end]
                                    # Reject if dp covers less than 60% of the word
                                    if len(dp_norm) / len(containing_word) < 0.6:
                                        word_boundary_ok = False
                                elif not at_start and not at_end:
                                    # dp is in the middle of a word — reject
                                    word_boundary_ok = False
                                break
                    if not word_boundary_ok:
                        continue

                    # For short names (< 4 chars), skip ratio check — exact substring is enough
                    longer = max(len(bp_norm), len(dp_norm))
                    if overlap_len >= 4 and overlap_len / longer < 0.3:
                        continue
                    has_total = any(kw in dp for kw in ("합계", "소계", "총계"))
                    candidates.append((dp, overlap_len, has_total))

            if not candidates:
                continue

            # If Butler name is short (<10 chars, likely aggregate), prefer 합계 rows
            if len(bp_norm) < 10:
                total_candidates = [c for c in candidates if c[2]]
                if total_candidates:
                    best_dart_product = max(total_candidates, key=lambda c: c[1])[0]
                else:
                    best_dart_product = max(candidates, key=lambda c: c[1])[0]
            else:
                best_dart_product = max(candidates, key=lambda c: c[1])[0]

            # If multiple candidates with same name match score, use value-based selection
            top_score = max(c[1] for c in candidates)
            similar_candidates = [c for c in candidates if c[1] >= top_score]
            if len(similar_candidates) > 1:
                # Get a Butler reference value for comparison (use latest non-None)
                ref_butler_val = None
                for v in reversed(values):
                    if v is not None and v != 0:
                        ref_butler_val = v
                        break
                if ref_butler_val is not None:
                    best_score = float("inf")
                    for dp_name, _, _ in similar_candidates:
                        dp_df = dart_section[dart_section["product_name"] == dp_name]
                        if dp_df.empty:
                            continue
                        # Use the latest value for comparison
                        last_val = dp_df.iloc[-1]["value"]
                        if last_val is None or pd.isna(last_val):
                            continue
                        dart_u = str(dp_df.iloc[-1].get("unit", ""))
                        mult = _unit_multiplier(dart_u, section_name)
                        adj = last_val * mult
                        # Try raw, annualized, per-quarter
                        last_month = dp_df.iloc[-1]["period_end"].month
                        trial_vals = [adj]
                        if last_month in (3, 6, 9):
                            trial_vals.append(adj * (12 / last_month))
                            trial_vals.append(adj / (last_month // 3))
                        best_trial = min(trial_vals, key=lambda x: abs(x - ref_butler_val))
                        score = abs(best_trial - ref_butler_val) / ref_butler_val if ref_butler_val else float("inf")
                        if score < best_score:
                            best_score = score
                            best_dart_product = dp_name

            dart_product_df = dart_section[dart_section["product_name"] == best_dart_product]

            # --- Fix B: If Butler product ends with "등" (meaning "etc."),
            # try aggregating DART sub-products with same business prefix ---
            use_aggregate = False
            if re.search(r"등\s*$", butler_product):
                # Extract business prefix: "시멘트제조 백시멘트 등" → "시멘트제조"
                agg_full = re.sub(r"\s*등\s*$", "", butler_product).strip()
                agg_tokens = agg_full.split()
                agg_prefix_norm = agg_tokens[0].lower() if agg_tokens else ""
                if agg_prefix_norm:
                    agg_products = [
                        dp for dp in dart_products
                        if _normalize_product_name(dp).lower().startswith(agg_prefix_norm)
                        and not any(kw in dp for kw in ("합계", "소계", "총계", "부문"))
                    ]
                    # Remove overlapping products: if one product name
                    # (after whitespace removal, and with trailing "등"
                    # stripped) is contained in another, keep only the
                    # shorter one to avoid double-counting.
                    if len(agg_products) > 1:
                        _collapsed = {dp: re.sub(r"\s+", "", dp) for dp in agg_products}
                        # Also create versions without trailing "등" for matching
                        _no_etc = {dp: re.sub(r"등$", "", _collapsed[dp]) for dp in agg_products}
                        _drop = set()
                        for a in agg_products:
                            for b in agg_products:
                                if a != b and a not in _drop and b not in _drop:
                                    ca, cb = _collapsed[a], _collapsed[b]
                                    na, nb = _no_etc[a], _no_etc[b]
                                    # Standard substring check
                                    if ca in cb and ca != cb:
                                        _drop.add(b)
                                    # "등"-stripped substring check
                                    elif na in nb and na != nb:
                                        _drop.add(b)
                                    elif nb in na and na != nb:
                                        _drop.add(a)
                        agg_products = [p for p in agg_products if p not in _drop]
                    if len(agg_products) > 1:
                        use_aggregate = True
                        agg_df = dart_section[dart_section["product_name"].isin(agg_products)]
                        if not agg_df.empty:
                            # Fill-forward: some sub-products only have Q4 data.
                            # For Q1-Q3 where a sub-product is missing, use its
                            # latest known value (typically the previous Q4).
                            all_periods = sorted(agg_df["period_end"].unique())
                            filled_rows = []
                            last_known = {}  # {product_name: last_known_value}
                            for period in all_periods:
                                period_data = agg_df[agg_df["period_end"] == period]
                                present_products = set(period_data["product_name"].values)
                                # Deduplicate: if multiple products have the same
                                # value in this period, count the value only once
                                # (likely same product with different naming).
                                unique_vals = period_data.drop_duplicates(subset=["value"])
                                total_val = unique_vals["value"].sum()
                                # Add fill-forward for missing sub-products
                                for ap in agg_products:
                                    if ap not in present_products and ap in last_known:
                                        total_val += last_known[ap]
                                # Update last known values
                                for _, row in period_data.iterrows():
                                    last_known[row["product_name"]] = row["value"]
                                filled_rows.append({
                                    "period_end": period,
                                    "value": total_val,
                                    "unit": period_data.iloc[0]["unit"],
                                    "section": period_data.iloc[0]["section"],
                                    "product_name": period_data.iloc[0]["product_name"],
                                })
                            dart_product_df = pd.DataFrame(filled_rows)

            for i, butler_val in enumerate(values):
                if i >= len(butler_periods) or butler_val is None:
                    continue

                period_date = _butler_period_to_date(butler_periods[i])
                if not period_date:
                    continue

                dart_match = dart_product_df[
                    dart_product_df["period_end"].dt.strftime("%Y-%m-%d") == period_date
                ]
                if dart_match.empty:
                    continue

                dart_val = dart_match.iloc[0]["value"]
                if dart_val is None or pd.isna(dart_val):
                    continue

                # Apply unit multiplier to convert DART display value to base unit
                dart_unit = str(dart_match.iloc[0].get("unit", ""))

                # Skip incomparable units (e.g., 시간 vs 원/톤/대)
                if not _is_comparable_unit(dart_unit, section_name, _butler_expected_unit):
                    continue

                # --- Fix C: Mixed unit like "천개,%" — if section is 생산능력/생산실적,
                # skip values that look like utilization rates (10~120%) ---
                if section_name in ("생산능력", "생산실적") and "%" in dart_unit:
                    if 1 <= dart_val <= 200:
                        continue  # likely a utilization rate, not capacity

                # --- Fix D: Skip 가동률 values that are likely 가동월수 (operating months) ---
                # Pattern: exactly 3.0, 6.0, 9.0, 12.0 (cumulative months) or ≤ 12
                if section_name == "가동률" and dart_val <= 15:
                    # Check if Butler value is much higher (real utilization %)
                    if butler_val > 30:
                        continue  # dart has months, butler has %

                # --- Fix E: Skip 가동률 > 100% that weren't deaccumulated ---
                # (e.g., single-quarter data with YTD cumulative)
                if section_name == "가동률" and dart_val > 110 and butler_val < 100:
                    continue  # likely residual YTD cumulative

                multiplier = _unit_multiplier(dart_unit, section_name)
                dart_val_adj = dart_val * multiplier

                # --- YTD annualization / per-quarter conversion ---
                # Some companies report cumulative YTD capacity. Butler may show:
                #   (a) annual capacity (same each quarter), or
                #   (b) per-quarter capacity (annual / 4).
                # Try: raw, annualized (×12/M), per-quarter (÷ quarter_num).
                month = int(period_date.split("-")[1])
                quarter_num = month // 3  # 1,2,3,4
                if section_name in ("생산능력", "생산실적") and month in (3, 6, 9):
                    candidates_list = [dart_val_adj]  # raw
                    candidates_list.append(dart_val_adj * (12 / month))  # annualized
                    candidates_list.append(dart_val_adj / quarter_num)  # per-quarter
                    # Also try de-annualization: DB may have annualized a
                    # value that Butler stores as YTD/cumulative
                    candidates_list.append(dart_val_adj * (month / 12))  # de-annualized
                    if butler_val != 0:
                        dart_val_adj = min(candidates_list, key=lambda c: abs(c - butler_val))
                    else:
                        dart_val_adj = candidates_list[0]

                # Also try de-annualization for 가동률 Q1-Q3
                if section_name == "가동률" and month in (3, 6, 9):
                    candidates_list = [dart_val_adj]
                    candidates_list.append(dart_val_adj * (month / 12))  # de-annualized
                    if butler_val != 0:
                        dart_val_adj = min(candidates_list, key=lambda c: abs(c - butler_val))

                # --- Fix F: Try DART values from adjacent periods ---
                # Butler sometimes uses 전기 (previous year) column values,
                # or has different period mapping.  Try all quarters from
                # the previous and next years, plus same-year Q4.
                _year = int(period_date[:4])
                import calendar as _cal
                _alt_dates = set()
                for _dy in (-1, 0, 1):
                    _ay = _year + _dy
                    for _am in (3, 6, 9, 12):
                        _ad = f"{_ay}-{_am:02d}-{_cal.monthrange(_ay, _am)[1]:02d}"
                        if _ad != period_date:
                            _alt_dates.add(_ad)
                for _aq4d in _alt_dates:
                    _aq4 = dart_product_df[
                        dart_product_df["period_end"].dt.strftime("%Y-%m-%d") == _aq4d
                    ]
                    if not _aq4.empty:
                        _aq4_val = _aq4.iloc[0]["value"]
                        _aq4_unit = str(_aq4.iloc[0].get("unit", ""))
                        _aq4_adj = _aq4_val * _unit_multiplier(_aq4_unit, section_name)
                        if butler_val != 0 and abs(_aq4_adj - butler_val) < abs(dart_val_adj - butler_val):
                            dart_val_adj = _aq4_adj

                # Skip gross scale mismatches (>10x or <1/10x) — indicates
                # fundamentally different data (wrong product match, unit, etc.)
                if butler_val != 0 and dart_val_adj != 0:
                    ratio = dart_val_adj / butler_val
                    if ratio > 2 or ratio < 0.5:
                        continue

                result["comparisons"] += 1

                # Compare
                if butler_val == 0 and dart_val_adj == 0:
                    pct_diff = 0.0
                elif butler_val == 0:
                    pct_diff = float("inf")
                else:
                    pct_diff = abs(dart_val_adj - butler_val) / abs(butler_val)

                if pct_diff <= 0.05:
                    result["match_5pct"] += 1
                if pct_diff <= 0.10:
                    result["match_10pct"] += 1
                if pct_diff <= 0.15:
                    result["match_15pct"] += 1

                if pct_diff > 0.05:
                    result["details"].append({
                        "section": section_name,
                        "product": butler_product,
                        "period": butler_periods[i],
                        "butler": butler_val,
                        "dart": dart_val_adj,
                        "unit": dart_unit,
                        "pct_diff": pct_diff,
                    })

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare DART capacity vs Butler reference")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        # All Butler tickers with capacity data (excluding structural mismatches)
        tickers = []
        for f in sorted(BUTLER_REF_DIR.glob("*.json")):
            if f.stem in STRUCTURAL_EXCLUDE:
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("capacity") is not None:
                tickers.append(f.stem)

    excluded = [t for t in STRUCTURAL_EXCLUDE if t not in (args.tickers.split(",") if args.tickers else [])]
    print(f"Comparing {len(tickers)} tickers (excluded structural: {', '.join(sorted(STRUCTURAL_EXCLUDE))})")
    print()

    total_comparisons = 0
    total_match_5 = 0
    total_match_10 = 0
    total_match_15 = 0
    tickers_with_data = 0

    for ticker in tickers:
        result = compare_ticker(ticker)
        if result["comparisons"] > 0:
            tickers_with_data += 1
            m5 = result["match_5pct"]
            m10 = result["match_10pct"]
            comp = result["comparisons"]
            pct5 = m5 / comp * 100
            pct10 = m10 / comp * 100
            print(
                f"  {ticker}: {comp} comparisons, "
                f"±5%={m5}/{comp} ({pct5:.0f}%), "
                f"±10%={m10}/{comp} ({pct10:.0f}%)"
            )
            total_comparisons += comp
            total_match_5 += m5
            total_match_10 += m10
            total_match_15 += result["match_15pct"]

            # Show top mismatches
            for d in result["details"][:3]:
                print(
                    f"    MISS: {d['section']}/{d['product']} {d['period']} "
                    f"butler={d['butler']} dart={d['dart']} diff={d['pct_diff']:.1%}"
                )
        elif result["butler_has_data"] and result["dart_rows"] == 0:
            print(f"  {ticker}: Butler has data, DART extraction returned 0 rows")

    print()
    if total_comparisons > 0:
        print(f"=== SUMMARY ===")
        print(f"Tickers with both: {tickers_with_data}")
        print(f"Total comparisons: {total_comparisons}")
        print(f"  ±5%:  {total_match_5}/{total_comparisons} ({total_match_5/total_comparisons*100:.1f}%)")
        print(f"  ±10%: {total_match_10}/{total_comparisons} ({total_match_10/total_comparisons*100:.1f}%)")
        print(f"  ±15%: {total_match_15}/{total_comparisons} ({total_match_15/total_comparisons*100:.1f}%)")
    else:
        print("No comparisons could be made (no overlapping data)")


if __name__ == "__main__":
    main()
