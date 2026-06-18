#!/usr/bin/env python3
"""Benchmark: DuckDB bulk reads vs individual parquet file reads.

Tests two scenarios:
  - Small set  (10 well-known tickers)
  - Large set  (100 random tickers, simulating a real backtest universe)

Metrics measured:
  - Price loading time
  - Financials loading time
  - Combined (factor panel) loading time for N tickers
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

# ── Benchmark configuration ──────────────────────────────────────────────────
MARKET = "us"
SMALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "BRK-B"]
RUNS = 5  # how many repetitions per benchmark (best-of reported)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _timed(fn, *args, runs: int = RUNS, **kwargs):
    """Run fn(*args, **kwargs) `runs` times, return (best_ms, last_result)."""
    best = float("inf")
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best * 1000, result


def _row(label: str, parquet_ms: float, duck_ms: float) -> None:
    speedup = parquet_ms / duck_ms if duck_ms > 0 else float("inf")
    print(
        f"  {label:<40}  parquet {parquet_ms:7.1f}ms  "
        f"duckdb {duck_ms:7.1f}ms  "
        f"speedup {speedup:.1f}×"
    )


# ── Parquet baselines ─────────────────────────────────────────────────────────

def parquet_price(tickers: list[str]) -> dict:
    from market_data.reader import load_price_dataframe
    out = {}
    for t in tickers:
        try:
            df, _ = load_price_dataframe(t, MARKET)
            out[t] = df
        except Exception:
            pass
    return out


def parquet_financials(tickers: list[str]) -> dict:
    from market_data.sec_term_reader import load_ticker_quarterly_cache
    out = {}
    for t in tickers:
        try:
            df = load_ticker_quarterly_cache(t, rebuild_if_stale=False)
            if df is not None and not df.empty:
                out[t] = df
        except Exception:
            pass
    return out


# ── DuckDB ───────────────────────────────────────────────────────────────────

def duckdb_price(tickers: list[str]) -> dict:
    from market_data.db_reader import bulk_load_prices
    return bulk_load_prices(tickers, MARKET)


def duckdb_financials(tickers: list[str]) -> dict:
    from market_data.db_reader import bulk_load_financials_quarterly
    return bulk_load_financials_quarterly(tickers, MARKET)


# ── Factor panel (end-to-end) ────────────────────────────────────────────────

def parquet_factor_panel(tickers: list[str]) -> pd.DataFrame:
    """Factor panel WITHOUT DuckDB (patches db.db_available to return False)."""
    from unittest.mock import patch
    from market_data.backtest.factors import build_factor_panel
    with patch("market_data.db.db_available", return_value=False):
        return build_factor_panel(
            tickers,
            market=MARKET,
            start="2022-01-01",
            end="2024-12-31",
            offline_mode=True,
        )


def duckdb_factor_panel(tickers: list[str]) -> pd.DataFrame:
    """Factor panel WITH DuckDB bulk preload."""
    from market_data.backtest.factors import build_factor_panel
    return build_factor_panel(
        tickers,
        market=MARKET,
        start="2022-01-01",
        end="2024-12-31",
        offline_mode=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def run_suite(label: str, tickers: list[str]) -> None:
    print(f"\n{'─'*70}")
    print(f"  Suite: {label}  ({len(tickers)} tickers, best of {RUNS} runs)")
    print(f"{'─'*70}")

    # Price
    p_ms, p_res = _timed(parquet_price, tickers)
    d_ms, d_res = _timed(duckdb_price, tickers)
    _row("Price loading", p_ms, d_ms)
    print(f"    parquet rows: {sum(len(v) for v in p_res.values()):,}  "
          f"duckdb rows: {sum(len(v) for v in d_res.values()):,}")

    # Financials
    p_ms, p_res = _timed(parquet_financials, tickers)
    d_ms, d_res = _timed(duckdb_financials, tickers)
    _row("Financials loading", p_ms, d_ms)
    print(f"    parquet tickers: {len(p_res)}  duckdb tickers: {len(d_res)}")

    # Factor panel (end-to-end, offline)
    p_ms, p_panel = _timed(parquet_factor_panel, tickers, runs=3)
    d_ms, d_panel = _timed(duckdb_factor_panel, tickers, runs=3)
    _row("Factor panel (build_factor_panel)", p_ms, d_ms)
    p_rows = len(p_panel) if p_panel is not None and not p_panel.empty else 0
    d_rows = len(d_panel) if d_panel is not None and not d_panel.empty else 0
    print(f"    parquet panel rows: {p_rows:,}  duckdb panel rows: {d_rows:,}")


def main() -> None:
    from market_data.db import db_available, DB_PATH
    from market_data.backtest.factors import available_price_symbols

    print("=" * 70)
    print("  Market-data-lake  ·  DuckDB vs Parquet Benchmark")
    print("=" * 70)
    print(f"\n  DuckDB available : {db_available()}")
    if db_available():
        size_mb = DB_PATH.stat().st_size / 1024**2
        print(f"  Database path    : {DB_PATH}")
        print(f"  Database size    : {size_mb:.0f} MB")

    if not db_available():
        print("\n  ERROR: DuckDB database not found.  Run build_duckdb.py first.")
        sys.exit(1)

    # Build large ticker set (100 random from available price symbols)
    all_syms = available_price_symbols(MARKET)
    random.seed(42)
    large_tickers = random.sample(
        [s for s in all_syms if s not in SMALL_TICKERS],
        min(100, len(all_syms) - len(SMALL_TICKERS)),
    )

    run_suite("Small set (10 tickers)", SMALL_TICKERS)
    run_suite("Large set (100 tickers)", large_tickers)

    print(f"\n{'='*70}")
    print("  Done.")


if __name__ == "__main__":
    main()
