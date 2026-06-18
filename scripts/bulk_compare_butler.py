"""Bulk compare our segment data vs Butler reference for all tickers.

Value-based matching: instead of relying on name matching, find the best
segment pair by comparing quarterly revenue values across overlapping periods.

Usage:
    python scripts/bulk_compare_butler.py [--limit 50] [--verbose]
"""
from __future__ import annotations

import sys
import os
import json
import calendar
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["MDL_DATA_DIR"] = str(Path(__file__).resolve().parents[1] / "data")

from market_data.kr_dart.segment_store import load_segment_facts
from market_data.kr_dart.segment_normalization import normalize_segment_dataframe


BUTLER_DIR = Path(__file__).resolve().parents[0] / "butler_reference"


def load_butler_segments(ticker: str) -> dict | None:
    """Load Butler segment data for a ticker."""
    path = BUTLER_DIR / f"{ticker}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sb = data.get("summary_biz_segments", {})
    if not sb.get("p") or not sb.get("d"):
        return None
    return sb


def load_our_segments(ticker: str) -> dict | None:
    """Load and normalize our segment data."""
    try:
        df = load_segment_facts(ticker, market="kr")
        if df is None or df.empty:
            return None
        norm = normalize_segment_dataframe(df, ticker, "kr")
        if norm.empty:
            return None

        # Pivot to {segment_name: {period: value}} for ALL segment types (business+geo+product)
        biz = norm[
            norm["metric"].str.contains("revenue|매출", case=False, na=False)
        ]
        if biz.empty:
            return None

        result = {}
        for seg_name in biz["segment_name"].unique():
            seg_df = biz[biz["segment_name"] == seg_name]
            period_vals = {}
            for _, r in seg_df.iterrows():
                p = pd.Timestamp(r["period_end"])
                key = f"{p.year}.{p.month:02d}"
                period_vals[key] = r["value"]
            if period_vals:
                result[seg_name] = period_vals
        return result if result else None
    except Exception:
        return None


def value_match_score(our_vals: dict, butler_vals: list, butler_periods: list) -> tuple[float, int, int]:
    """Compute match score between our segment and butler segment.

    Uses both direct value comparison (±15%) and ratio consistency
    (handles different unit scales like 원 vs 백만원 vs 억원).

    Returns (match_rate, matched_count, compared_count).
    """
    import numpy as np

    compared = 0
    matched = 0

    for bi, bp in enumerate(butler_periods):
        bval = butler_vals[bi] if bi < len(butler_vals) else None
        if bval is None or bval == 0:
            continue
        bval_won = bval * 1e8

        our_val = our_vals.get(bp)
        if our_val is None:
            continue

        compared += 1
        diff = abs(our_val - bval_won) / abs(bval_won) * 100
        if diff <= 15:
            matched += 1

    # Also check ratio consistency (handles scale differences)
    ratios = []
    for bi, bp in enumerate(butler_periods):
        bval = butler_vals[bi] if bi < len(butler_vals) else None
        if bval is None or bval == 0: continue
        our_val = our_vals.get(bp)
        if our_val is None or our_val == 0: continue
        ratios.append(our_val / (bval * 1e8))

    ratio_score = 0.0
    if len(ratios) >= 3:
        mean_r = np.mean(ratios)
        if mean_r > 0:
            for scale in [1.0, 0.01, 0.001, 0.000001, 100, 1000, 1000000]:
                sr = [r / scale for r in ratios]
                sm = np.mean(sr)
                if sm <= 0: continue
                cv = np.std(sr) / sm
                if cv < 0.2:
                    ratio_score = 100.0
                    break
                if cv < 0.35:
                    ratio_score = 80.0
                    break

    rate = max(
        matched / compared * 100 if compared > 0 else 0,
        ratio_score,
    )
    return rate, matched, compared


def find_best_matches(our_segs: dict, butler_sb: dict) -> list[dict]:
    """Find best segment matches using value-based comparison."""
    bp = butler_sb["p"]
    bd = butler_sb["d"]
    matches = []

    for butler_name, butler_vals in bd.items():
        # Skip sub-segments and adjustments
        if any(x in butler_name for x in ["조정", "-"]):
            continue
        if not any(v is not None and v != 0 for v in butler_vals):
            continue

        best_match = None
        best_rate = 0
        best_matched = 0
        best_compared = 0

        for our_name, our_vals in our_segs.items():
            rate, matched, compared = value_match_score(our_vals, butler_vals, bp)
            if compared >= 3 and rate > best_rate:
                best_rate = rate
                best_match = our_name
                best_matched = matched
                best_compared = compared

        matches.append({
            "butler_name": butler_name,
            "our_name": best_match,
            "match_rate": best_rate,
            "matched": best_matched,
            "compared": best_compared,
        })

    return matches


def run_bulk_compare(limit: int = 0, verbose: bool = False):
    """Run comparison across all tickers."""
    butler_tickers = sorted([f.stem for f in BUTLER_DIR.glob("*.json") if f.stem.isdigit()])
    if limit > 0:
        butler_tickers = butler_tickers[:limit]

    print(f"Comparing {len(butler_tickers)} tickers...")

    # Track results
    ticker_results = []
    all_mismatches = []
    noise_patterns = defaultdict(int)
    no_data = 0
    total_segments_compared = 0
    total_segments_matched = 0

    for i, ticker in enumerate(butler_tickers):
        butler_sb = load_butler_segments(ticker)
        if butler_sb is None:
            continue

        our_segs = load_our_segments(ticker)
        if our_segs is None:
            no_data += 1
            continue

        matches = find_best_matches(our_segs, butler_sb)
        if not matches:
            continue

        # Calculate ticker-level stats
        ticker_matched = sum(1 for m in matches if m["match_rate"] >= 80)
        ticker_total = len(matches)
        ticker_rate = ticker_matched / ticker_total * 100 if ticker_total else 0

        total_segments_compared += sum(m["compared"] for m in matches)
        total_segments_matched += sum(m["matched"] for m in matches)

        ticker_results.append({
            "ticker": ticker,
            "rate": ticker_rate,
            "matched": ticker_matched,
            "total": ticker_total,
            "details": matches,
        })

        # Track mismatches
        for m in matches:
            if m["match_rate"] < 50 and m["compared"] >= 3:
                all_mismatches.append({
                    "ticker": ticker,
                    "butler_name": m["butler_name"],
                    "our_name": m["our_name"],
                    "rate": m["match_rate"],
                })

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(butler_tickers)}] processed...")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(ticker_results)} tickers compared")
    print(f"{'='*60}")

    if not ticker_results:
        print("No results!")
        return

    rates = [r["rate"] for r in ticker_results]
    period_rate = total_segments_matched / total_segments_compared * 100 if total_segments_compared else 0

    print(f"  No segment data: {no_data} tickers")
    print(f"  Period-level match: {total_segments_matched}/{total_segments_compared} = {period_rate:.1f}%")
    print(f"  Avg ticker match rate: {np.mean(rates):.1f}%")
    print(f"  Tickers >= 90%: {sum(1 for r in rates if r >= 90)}/{len(rates)} ({sum(1 for r in rates if r >= 90)/len(rates)*100:.0f}%)")
    print(f"  Tickers >= 80%: {sum(1 for r in rates if r >= 80)}/{len(rates)}")
    print(f"  Tickers >= 50%: {sum(1 for r in rates if r >= 50)}/{len(rates)}")
    print(f"  Tickers < 50%: {sum(1 for r in rates if r < 50)}/{len(rates)}")

    # Distribution
    print(f"\n  Match rate distribution:")
    for threshold in [100, 90, 80, 70, 60, 50, 30, 0]:
        count = sum(1 for r in rates if r >= threshold)
        print(f"    >= {threshold}%: {count} tickers ({count/len(rates)*100:.0f}%)")

    # Worst tickers
    worst = sorted(ticker_results, key=lambda x: x["rate"])[:10]
    print(f"\n  Worst 10 tickers:")
    for r in worst:
        segs = ", ".join(f'{m["butler_name"]}→{m["our_name"] or "?"}'
                         for m in r["details"][:3])
        print(f"    {r['ticker']}: {r['rate']:.0f}% ({r['matched']}/{r['total']}) [{segs}]")

    # Common unmatched Butler names
    unmatched_names = defaultdict(int)
    for m in all_mismatches:
        unmatched_names[m["butler_name"]] += 1
    if unmatched_names:
        print(f"\n  Most common unmatched Butler segments:")
        for name, count in sorted(unmatched_names.items(), key=lambda x: -x[1])[:15]:
            print(f"    \"{name}\": {count} tickers")

    # Save results
    output_path = Path(__file__).parent / "butler_compare_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_tickers": len(ticker_results),
            "period_match_rate": period_rate,
            "avg_ticker_rate": float(np.mean(rates)),
            "tickers_above_90": sum(1 for r in rates if r >= 90),
            "results": ticker_results,
            "mismatches": all_mismatches[:100],
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_bulk_compare(limit=args.limit, verbose=args.verbose)
