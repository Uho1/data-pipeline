#!/usr/bin/env python3
"""Backfill market_cap for US prices using yfinance shares_outstanding.

Usage:
    .venv/bin/python scripts/backfill_us_market_cap.py [--limit 50] [--ticker AAPL]
"""
from __future__ import annotations

import argparse
import time

import duckdb
import pandas as pd

DB_PATH = "data/market_data_prices.duckdb"


def _fetch_shares(ticker: str) -> pd.Series | None:
    """Fetch shares outstanding from yfinance for US ticker."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        shares = t.get_shares_full(start="2010-01-01")
        if shares is not None and not shares.empty:
            shares.index = shares.index.tz_localize(None)
            shares = shares[~shares.index.duplicated(keep="last")].sort_index()
            return shares
    except Exception:
        pass
    return None


def backfill_ticker(con: duckdb.DuckDBPyConnection, ticker: str) -> int:
    """Backfill market_cap for one ticker. Returns rows updated."""
    prices = con.execute(
        "SELECT date, close FROM prices WHERE ticker = ? AND market = 'us' ORDER BY date",
        [ticker],
    ).fetchdf()
    if prices.empty:
        return 0

    shares = _fetch_shares(ticker)
    if shares is None or shares.empty:
        return 0

    price_dates = pd.DatetimeIndex(prices["date"])
    shares_daily = shares.reindex(price_dates, method="ffill")

    close = pd.to_numeric(prices["close"], errors="coerce").values
    mcap = close * shares_daily.values

    updated = 0
    for i in range(len(prices)):
        mc = mcap[i]
        if pd.notna(mc) and mc > 0:
            con.execute(
                "UPDATE prices SET market_cap = ? WHERE ticker = ? AND date = ? AND market = 'us'",
                [float(mc), ticker, prices["date"].iloc[i]],
            )
            updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ticker", default=None)
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH)

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [
            r[0] for r in con.execute(
                """
                SELECT DISTINCT ticker FROM prices
                WHERE market = 'us'
                AND (market_cap IS NULL OR market_cap = 0)
                AND ticker NOT IN (
                    SELECT DISTINCT ticker FROM prices WHERE market = 'us' AND market_cap > 0
                )
                ORDER BY ticker
                """
            ).fetchall()
        ]

    if args.limit > 0:
        tickers = tickers[:args.limit]

    print(f"Backfilling market_cap for {len(tickers)} US tickers...")
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
        if i % 10 == 0:
            time.sleep(1)

    print(f"\nDone. Updated {total_updated:,} rows, {failed} failures.")
    con.close()


if __name__ == "__main__":
    main()
