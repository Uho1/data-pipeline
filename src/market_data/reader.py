from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_data.config import PRICES_DIR
from market_data.db_router import (
    db_available_for_market,
    get_connection_for_market,
    get_prices_connection_for_market,
    is_kr_market,
    normalize_market,
    normalize_ticker_for_market,
)
from market_data.utils import sanitize_ticker


def load_price_dataframe(ticker: str, market: str | None = None) -> tuple[pd.DataFrame, Path]:
    """Load price DataFrame for a ticker from local storage.

    Returns (df, synthetic_path). The synthetic path preserves backward
    compatibility for callers that inspect ``source.parent.name`` to infer market.
    """
    resolved_market = normalize_market(market=market, ticker=ticker)
    if not db_available_for_market(resolved_market, ticker=ticker):
        raise RuntimeError(
            "Local price storage is unavailable. "
            "Run ingest first: python -m market_data ingest ..."
        )

    try:
        from market_data.db_reader import load_price_from_db

        result = load_price_from_db(normalize_ticker_for_market(ticker, resolved_market), market=resolved_market or market)
    except Exception as exc:
        raise RuntimeError(
            "Failed to read price data from local storage."
        ) from exc

    if result is None:
        raise FileNotFoundError(f"No local price data found for ticker={ticker} market={market}")

    df, effective_market = result
    synthetic_ticker = normalize_ticker_for_market(ticker, effective_market)
    synthetic_path = PRICES_DIR / effective_market / f"{sanitize_ticker(synthetic_ticker)}.parquet"
    return df, synthetic_path


def available_tickers(market: str | None = None) -> list[str]:
    """Return ticker symbols available in local storage."""
    resolved = normalize_market(market=market)
    if resolved is not None and not db_available_for_market(resolved):
        raise RuntimeError(
            "Local storage is unavailable for ticker discovery. "
            "Run ingest first: python -m market_data ingest ..."
        )

    try:
        if resolved is not None:
            con = get_prices_connection_for_market(resolved)
            rows = con.execute(
                "SELECT DISTINCT ticker FROM prices WHERE market = ? ORDER BY ticker",
                [str(resolved).strip().lower()],
            ).fetchall()
            return [r[0] for r in rows]

        combined: set[str] = set()
        for market_name in ("us", "kr"):
            if not db_available_for_market(market_name):
                continue
            con = get_prices_connection_for_market(market_name)
            rows = con.execute(
                "SELECT DISTINCT ticker FROM prices WHERE market = ? ORDER BY ticker",
                [market_name],
            ).fetchall()
            combined.update(str(row[0]) for row in rows if row and row[0])
        return sorted(combined)
    except Exception as exc:
        raise RuntimeError(
            "Failed to list tickers from local storage."
        ) from exc
