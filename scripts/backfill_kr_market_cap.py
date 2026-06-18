#!/usr/bin/env python3
"""Backfill market_cap + shares_outstanding for KR prices using yfinance.

Usage:
    .venv/bin/python scripts/backfill_kr_market_cap.py [--limit 50] [--ticker 005930]
"""
from __future__ import annotations

import argparse
import sys
import time

import duckdb
import pandas as pd

DB_PATH = "data/market_data_kr_prices.duckdb"


def _fetch_shares(ticker_code: str) -> pd.Series | None:
    """Fetch shares outstanding from yfinance."""
    try:
        import yfinance as yf

        for suffix in (".KS", ".KQ"):
            t = yf.Ticker(f"{ticker_code}{suffix}")
            shares = t.get_shares_full(start="2010-01-01")
            if shares is not None and not shares.empty:
                shares.index = shares.index.tz_localize(None)
                shares = shares[~shares.index.duplicated(keep="last")].sort_index()
                return shares
    except Exception:
        pass
    return None


def backfill_ticker(con: duckdb.DuckDBPyConnection, ticker: str) -> int:
    """Backfill market_cap for one ticker. Returns number of rows updated."""
    # Get price dates for this ticker
    prices = con.execute(
        "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
        [ticker],
    ).fetchdf()
    if prices.empty:
        return 0

    shares = _fetch_shares(ticker)
    if shares is None or shares.empty:
        return 0

    # Build daily shares by forward-fill
    price_dates = pd.DatetimeIndex(prices["date"])
    shares_daily = shares.reindex(price_dates, method="ffill")

    # Compute market cap
    close = pd.to_numeric(prices["close"], errors="coerce").values
    mcap = close * shares_daily.values
    shares_vals = shares_daily.values

    # Update in batches
    updated = 0
    for i in range(len(prices)):
        mc = mcap[i]
        sv = shares_vals[i]
        if pd.notna(mc) and mc > 0:
            con.execute(
                "UPDATE prices SET market_cap = ?, shares_outstanding = ? "
                "WHERE ticker = ? AND date = ?",
                [float(mc), float(sv), ticker, prices["date"].iloc[i]],
            )
            updated += 1

    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max tickers to process (0=all)")
    parser.add_argument("--ticker", default=None, help="Single ticker to backfill")
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH)

    if args.ticker:
        tickers = [args.ticker]
    else:
        # Get tickers that have no market_cap data
        tickers = [
            r[0]
            for r in con.execute(
                """
                SELECT DISTINCT ticker FROM prices
                WHERE (market_cap IS NULL OR market_cap = 0)
                AND ticker NOT IN (
                    SELECT DISTINCT ticker FROM prices WHERE market_cap > 0
                )
                ORDER BY ticker
                """
            ).fetchall()
        ]

    if args.limit > 0:
        tickers = tickers[: args.limit]

    print(f"Backfilling market_cap for {len(tickers)} tickers...")
    total_updated = 0
    failed = 0

    for i, ticker in enumerate(tickers, 1):
        try:
            n = backfill_ticker(con, ticker)
            total_updated += n
            status = f"{n} rows" if n > 0 else "no shares data"
            print(f"  [{i}/{len(tickers)}] {ticker}: {status}")
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(tickers)}] {ticker}: ERROR {e}")

        # Rate limit: yfinance can throttle
        if i % 10 == 0:
            time.sleep(1)

    print(f"\nDone. Updated {total_updated:,} rows, {failed} failures.")
    con.close()


if __name__ == "__main__":
    main()
