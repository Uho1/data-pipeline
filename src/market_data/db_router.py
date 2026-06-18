"""Market-aware DuckDB routing helpers."""
from __future__ import annotations

import re
from pathlib import Path


_KR_TICKER_RE = re.compile(r"^(?P<code>\d{6})(?:\.(?:KS|KQ))?$", re.IGNORECASE)


def is_kr_ticker(value: str | None) -> bool:
    if value is None:
        return False
    return _KR_TICKER_RE.match(str(value).strip().upper()) is not None


def normalize_kr_ticker(value: str) -> str:
    text = str(value or "").strip().upper()
    match = _KR_TICKER_RE.match(text)
    if match is None:
        raise ValueError(f"Invalid KR ticker/code: {value}")
    return str(match.group("code"))


def normalize_market(market: str | None, ticker: str | None = None) -> str | None:
    if market is None:
        return "kr" if is_kr_ticker(ticker) else None
    norm = str(market).strip().lower()
    if norm in {"", "none"}:
        return "kr" if is_kr_ticker(ticker) else None
    if norm == "auto":
        return "kr" if is_kr_ticker(ticker) else "us"
    return norm


def normalize_ticker_for_market(ticker: str, market: str | None) -> str:
    resolved = normalize_market(market=market, ticker=ticker)
    if resolved == "kr":
        return normalize_kr_ticker(ticker)
    return str(ticker or "").strip().upper()


def is_kr_market(market: str | None, ticker: str | None = None) -> bool:
    return normalize_market(market=market, ticker=ticker) == "kr"


def db_available_for_market(market: str | None, ticker: str | None = None) -> bool:
    if is_kr_market(market=market, ticker=ticker):
        from market_data.db_kr import db_available

        return db_available()
    from market_data.db import db_available

    return db_available()


def get_connection_for_market(market: str | None, ticker: str | None = None):
    if is_kr_market(market=market, ticker=ticker):
        from market_data.db_kr import get_connection

        return get_connection()
    from market_data.db import get_connection

    return get_connection()


def close_connection_for_market(market: str | None, ticker: str | None = None) -> None:
    if is_kr_market(market=market, ticker=ticker):
        from market_data.db_kr import close_connection

        close_connection()
        return
    from market_data.db import close_connection

    close_connection()


def get_db_path_for_market(market: str | None, ticker: str | None = None) -> Path:
    if is_kr_market(market=market, ticker=ticker):
        from market_data.db_kr import DB_PATH

        return DB_PATH
    from market_data.db import DB_PATH

    return DB_PATH


# --- Prices DB helpers ---------------------------------------------------


def get_prices_connection_for_market(market: str | None, ticker: str | None = None):
    """Return a connection to the prices-specific DuckDB for the given market."""
    if is_kr_market(market=market, ticker=ticker):
        from market_data.db_kr_prices import get_connection

        return get_connection()
    from market_data.db_prices import get_connection

    return get_connection()


def get_prices_db_path_for_market(market: str | None, ticker: str | None = None) -> Path:
    if is_kr_market(market=market, ticker=ticker):
        from market_data.db_kr_prices import DB_PATH

        return DB_PATH
    from market_data.db_prices import DB_PATH

    return DB_PATH


def prices_db_available_for_market(market: str | None, ticker: str | None = None) -> bool:
    if is_kr_market(market=market, ticker=ticker):
        from market_data.db_kr_prices import db_available

        return db_available()
    from market_data.db_prices import db_available

    return db_available()
