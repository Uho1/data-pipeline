from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable

import pandas as pd
import requests

from market_data.db_router import normalize_kr_ticker
from market_data.utils import now_utc_iso, retry_call

_log = logging.getLogger(__name__)
_REQUEST_TIMEOUT_PATCH_LOCK = threading.Lock()
_REQUEST_TIMEOUT_PATCHED = False


def _default_krx_request_timeout() -> int:
    raw = str(os.getenv("MARKET_DATA_KRX_REQUEST_TIMEOUT_SEC", "20")).strip()
    try:
        timeout = int(raw)
    except (TypeError, ValueError):
        timeout = 20
    return max(5, timeout)


def _default_krx_request_retries() -> int:
    raw = str(os.getenv("MARKET_DATA_KRX_REQUEST_RETRIES", "2")).strip()
    try:
        retries = int(raw)
    except (TypeError, ValueError):
        retries = 2
    return max(0, retries)


def _wrap_session_request_with_default_timeout(
    request_fn: Callable[..., Any],
    *,
    timeout: int,
) -> Callable[..., Any]:
    def _wrapped(self, method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return request_fn(self, method, url, **kwargs)

    return _wrapped


def _ensure_requests_default_timeout(timeout: int | None = None) -> None:
    global _REQUEST_TIMEOUT_PATCHED
    if _REQUEST_TIMEOUT_PATCHED:
        return
    with _REQUEST_TIMEOUT_PATCH_LOCK:
        if _REQUEST_TIMEOUT_PATCHED:
            return
        effective_timeout = int(timeout or _default_krx_request_timeout())
        original = requests.sessions.Session.request
        requests.sessions.Session.request = _wrap_session_request_with_default_timeout(
            original,
            timeout=effective_timeout,
        )
        _REQUEST_TIMEOUT_PATCHED = True
        _log.info("Applied default requests timeout=%ss for KRX/pykrx fetches", effective_timeout)


def _require_pykrx():
    try:
        from pykrx import stock
    except ImportError as exc:
        raise RuntimeError(
            "pykrx is required for KRX ingest. Install it first: pip install pykrx"
        ) from exc
    return stock


def _fetch_shares_yfinance(ticker_code: str, start: str) -> pd.Series | None:
    """Fetch shares outstanding from yfinance as fallback for KRX API.

    yfinance shares data can lag behind by several months, so we fetch
    from 2 years before the requested start to ensure we have data to
    forward-fill from.
    """
    try:
        import yfinance as yf

        # Fetch from 2 years before start to ensure coverage for ffill
        early_start = (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

        for suffix in (".KS", ".KQ"):
            yf_ticker = f"{ticker_code}{suffix}"
            t = yf.Ticker(yf_ticker)
            shares = t.get_shares_full(start=early_start)
            if shares is not None and not shares.empty:
                shares.index = shares.index.tz_localize(None)
                return shares
    except Exception as exc:
        _log.debug("yfinance shares fetch failed for %s: %s", ticker_code, exc)
    return None


def _rename_first(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [mapping.get(str(col), str(col)) for col in out.columns]
    return out


def _yearly_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """Split a date range into 1-year chunks for KRX API reliability."""
    from datetime import date, timedelta
    s = date.fromisoformat(start) if "-" in start else date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    e = date.fromisoformat(end) if "-" in end else date(int(end[:4]), int(end[4:6]), int(end[6:8]))
    ranges = []
    while s < e:
        chunk_end = min(date(s.year, 12, 31), e)
        ranges.append((s.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        s = date(s.year + 1, 1, 1)
    return ranges


def _fetch_concat(fetcher, start: str, end: str, ticker_code: str, **kwargs) -> pd.DataFrame:
    """Call a pykrx fetcher in 1-year chunks and concat."""
    _ensure_requests_default_timeout()
    chunks = []
    retries = _default_krx_request_retries()
    for s, e in _yearly_ranges(start, end):
        try:
            label = f"pykrx:{getattr(fetcher, '__name__', 'fetch')}:{ticker_code}:{s}-{e}"
            df = retry_call(
                lambda: fetcher(s, e, ticker_code, **kwargs),
                retries=retries,
                backoff_base=1.0,
                label=label,
            )
            if df is not None and not df.empty:
                chunks.append(df)
        except Exception as exc:
            _log.warning("pykrx chunk %s-%s %s failed: %s", s, e, ticker_code, exc)
    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks)
    return result[~result.index.duplicated(keep="last")].sort_index()


def fetch_price_frame(
    *,
    ticker: str,
    start: str,
    end: str,
    ticker_name: str | None = None,
    market_tier: str | None = None,
) -> pd.DataFrame:
    stock = _require_pykrx()
    ticker_code = normalize_kr_ticker(ticker)

    ohlcv = _fetch_concat(stock.get_market_ohlcv_by_date, start, end, ticker_code)
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()
    cap = _fetch_concat(stock.get_market_cap_by_date, start, end, ticker_code)
    fundamental = _fetch_concat(stock.get_market_fundamental_by_date, start, end, ticker_code)

    out = _rename_first(
        ohlcv,
        {
            "시가": "Open",
            "고가": "High",
            "저가": "Low",
            "종가": "Close",
            "거래량": "Volume",
            "등락률": "PctChange",
        },
    )
    if cap is not None and not cap.empty:
        cap = _rename_first(
            cap,
            {
                "시가총액": "MarketCap",
                "거래대금": "TradedValue",
                "상장주식수": "SharesOutstanding",
                "거래량": "CapVolume",
            },
        )
        out = out.join(cap[[col for col in ("MarketCap", "TradedValue", "SharesOutstanding") if col in cap.columns]], how="left")

    # Fallback: if KRX market cap is missing, compute from yfinance shares
    if "MarketCap" not in out.columns or out["MarketCap"].isna().all():
        shares = _fetch_shares_yfinance(ticker_code, start)
        if shares is not None and not shares.empty:
            # Deduplicate and sort before reindex
            shares = shares[~shares.index.duplicated(keep="last")].sort_index()
            shares_daily = shares.reindex(out.index, method="ffill")
            close = pd.to_numeric(out.get("Close"), errors="coerce")
            out["MarketCap"] = close * shares_daily
            out["SharesOutstanding"] = shares_daily
    if fundamental is not None and not fundamental.empty:
        fundamental = _rename_first(
            fundamental,
            {
                "BPS": "BPS",
                "PER": "PER",
                "PBR": "PBR",
                "EPS": "EPS",
                "DIV": "DividendYield",
                "DPS": "DPS",
            },
        )
        out = out.join(
            fundamental[
                [col for col in ("BPS", "PER", "PBR", "EPS", "DividendYield", "DPS") if col in fundamental.columns]
            ],
            how="left",
        )

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    if out.empty:
        return out

    out.index.name = "Date"
    out["Ticker"] = ticker_code
    out["TickerName"] = ticker_name
    out["MarketTier"] = market_tier
    out["Adj Close"] = pd.to_numeric(out.get("Close"), errors="coerce")
    out["PriceChange"] = pd.to_numeric(out.get("Close"), errors="coerce").diff()
    out["PctChange"] = pd.to_numeric(out.get("PctChange"), errors="coerce")
    out["Dividends"] = 0.0
    # TODO: populate stock split history when a reliable KRX source is added.
    out["Stock Splits"] = 0.0
    out["CollectedAt"] = now_utc_iso()
    return out

