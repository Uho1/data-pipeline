from __future__ import annotations

import pandas as pd

from market_data.reader import load_price_dataframe


def run_sample(ticker: str, market: str | None) -> int:
    try:
        df, effective_market = load_price_dataframe(ticker=ticker, market=market)
    except FileNotFoundError:
        print(f"[SAMPLE] no price data found for ticker={ticker} market={market}")
        return 1

    print(f"[SAMPLE] ticker={ticker} market={effective_market} rows={len(df)} cols={len(df.columns)}")
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"[SAMPLE] price_range={df.index.min()} -> {df.index.max()}")
    print(df.reset_index().head(3).to_string(index=False))

    # Show financials summary from DuckDB
    try:
        from market_data.db_router import db_available_for_market, get_connection_for_market

        if db_available_for_market(effective_market, ticker=ticker):
            con = get_connection_for_market(effective_market, ticker=ticker)
            ticker_upper = str(ticker).strip().upper()
            mkt = str(effective_market).strip().lower()
            rows = con.execute(
                'SELECT COUNT(*) FROM financials_quarterly WHERE ticker = ? AND market = ?',
                [ticker_upper, mkt],
            ).fetchone()
            count = rows[0] if rows else 0
            print(f"[SAMPLE] financials_quarterly: {count} rows in DB")
    except Exception:
        pass

    return 0
