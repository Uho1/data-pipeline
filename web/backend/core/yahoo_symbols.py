"""Shared Yahoo Finance symbol conversion with KOSDAQ/KOSPI awareness."""
from __future__ import annotations

import logging
from threading import Lock

log = logging.getLogger(__name__)

_kosdaq_tickers: set[str] | None = None
_lock = Lock()


def _load_kosdaq_set() -> set[str]:
    """Build KOSDAQ ticker set from ticker_master_kr.json (lazy, one-time)."""
    try:
        from web.backend.services.json_data_service import load_ticker_master

        items = load_ticker_master("kr")
        return {
            item["ticker"]
            for item in items
            if item.get("market_tier") == "KOSDAQ" and item.get("ticker")
        }
    except Exception as exc:
        log.warning("Failed to load KOSDAQ ticker set: %s", exc)
        return set()


def _get_kosdaq_set() -> set[str]:
    global _kosdaq_tickers
    if _kosdaq_tickers is None:
        with _lock:
            if _kosdaq_tickers is None:
                _kosdaq_tickers = _load_kosdaq_set()
    return _kosdaq_tickers


def reload() -> None:
    """Force-reload the KOSDAQ ticker set (e.g. after data update)."""
    global _kosdaq_tickers
    with _lock:
        _kosdaq_tickers = _load_kosdaq_set()


def to_yahoo_symbol(ticker: str, market: str) -> str:
    """Convert internal ticker to Yahoo Finance symbol.

    Korean KOSDAQ tickers get .KQ suffix, KOSPI tickers get .KS.
    """
    if market == "kr":
        t = ticker.replace(".KS", "").replace(".KQ", "").lstrip("0").zfill(6)
        suffix = ".KQ" if t in _get_kosdaq_set() else ".KS"
        return f"{t}{suffix}"
    return ticker
