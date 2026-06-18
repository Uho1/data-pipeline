"""Price data service — serves from per-ticker JSON files.

Primary:  data/tickers/{market}/{ticker}.json  →  JSON file read
Fallback: synthetic OHLCV with random walk (used when JSON file unavailable)
"""
from __future__ import annotations

import logging
import math
import random
from datetime import date, timedelta

import pandas as pd

log = logging.getLogger(__name__)


def _resolve_market(ticker: str, market: str) -> str:
    """Resolve 'auto' market to 'kr' or 'us'."""
    if market and market != "auto":
        return market.strip().lower()
    return "kr" if ticker.isdigit() and len(ticker) == 6 else "us"


def get_price_bars(
    ticker: str,
    market: str,
    start: str | None = None,
    end: str | None = None,
) -> tuple[list[dict], str, bool]:
    """Return (bars, resolved_market, is_mock).

    bars = [{"time": "YYYY-MM-DD", "open": .., "high": .., "low": .., "close": .., "volume": ..}]
    is_mock = True when JSON file is unavailable.
    """
    resolved_market = _resolve_market(ticker, market)
    try:
        from web.backend.services.json_data_service import get_price_bars as json_price_bars

        bars = json_price_bars(ticker.upper() if resolved_market == "us" else ticker, resolved_market, start, end)
        if bars:
            return bars, resolved_market, False
        log.warning("No JSON price data for %s/%s — returning mock", ticker, resolved_market)
    except Exception as exc:
        log.warning("JSON price load failed (%s) — returning mock", exc)

    return _mock_bars(ticker), resolved_market, True


def get_available_tickers(market: str) -> tuple[list[str], bool]:
    """Return (tickers, is_mock) from JSON meta files."""
    try:
        from web.backend.services.json_data_service import list_available_tickers
        tickers = list_available_tickers(market.lower())
        if tickers:
            return tickers, False
    except Exception as exc:
        log.warning("list_available_tickers failed (%s) — returning mock list", exc)

    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY"], True


def get_available_ticker_items(market: str) -> tuple[list[dict[str, str]], bool]:
    """Return ticker list with company names from JSON meta files."""
    try:
        from web.backend.services.json_data_service import load_ticker_master

        resolved_market = market.strip().lower() if market else "us"
        items_raw = load_ticker_master(resolved_market)
        if items_raw:
            items = [
                {
                    "ticker": item.get("ticker", ""),
                    "company_name": item.get("name", ""),
                    "name_kr": item.get("name_kr", ""),
                }
                for item in items_raw
                if item.get("ticker")
            ]
            return items, False
    except Exception as exc:
        log.warning("load_ticker_master failed (%s) — falling back", exc)

    tickers, is_mock = get_available_tickers(market)
    return [{"ticker": ticker, "company_name": ""} for ticker in tickers], is_mock


def get_valuation_points(
    ticker: str,
    market: str,
    start: str | None = None,
    end: str | None = None,
) -> tuple[list[dict], float | None, float | None, str, bool, list[str]]:
    """Return valuation-aligned daily points for PER/PBR band overlays.

    points = [{"time","eps","bps","per","pbr"}], default_per, default_pbr, is_mock, warnings
    """
    warnings: list[str] = []
    resolved_market = _resolve_market(ticker, market)

    # Load prices from JSON
    try:
        from web.backend.services.json_data_service import load_ticker_data

        data = load_ticker_data(ticker.upper() if resolved_market == "us" else ticker, resolved_market)
        if data is None or "prices" not in data:
            warnings.append(f"가격 데이터 없음: {ticker.upper()}")
            return [], None, None, resolved_market, True, warnings

        prices = data["prices"]
        dates = prices.get("dates", [])
        closes = prices.get("close", [])

        if not dates or not closes:
            warnings.append("가격 데이터 비어있음")
            return [], None, None, resolved_market, True, warnings

        # Build simple PER/PBR from financials if available
        points: list[dict] = []
        fin = data.get("financials", {})
        fin_periods = fin.get("periods", [])
        fin_eps_list = fin.get("income", {}).get("eps", [])

        # Create period→EPS/BPS lookup
        eps_lookup: dict[str, float | None] = {}
        bps_lookup: dict[str, float | None] = {}
        shares_list = fin.get("shares_outstanding", [])
        equity_list = fin.get("balance", {}).get("shareholders_equity", [])

        for i, p in enumerate(fin_periods):
            if i < len(fin_eps_list):
                eps_lookup[str(p)] = fin_eps_list[i]
            if i < len(equity_list) and i < len(shares_list):
                eq = equity_list[i]
                sh = shares_list[i]
                if eq is not None and sh is not None and sh > 0:
                    bps_lookup[str(p)] = eq / sh

        # For each price date, find the latest available EPS/BPS
        current_eps: float | None = None
        current_bps: float | None = None
        fin_period_idx = 0

        for i, d in enumerate(dates):
            if start and d < start:
                continue
            if end and d > end:
                continue

            # Advance financial period pointer
            while fin_period_idx < len(fin_periods) and str(fin_periods[fin_period_idx]) <= d:
                p = str(fin_periods[fin_period_idx])
                if p in eps_lookup and eps_lookup[p] is not None:
                    current_eps = eps_lookup[p]
                if p in bps_lookup and bps_lookup[p] is not None:
                    current_bps = bps_lookup[p]
                fin_period_idx += 1

            close_val = closes[i] if i < len(closes) else None
            per_val = None
            pbr_val = None
            if close_val and current_eps and current_eps != 0:
                per_val = round(close_val / current_eps, 2)
            if close_val and current_bps and current_bps != 0:
                pbr_val = round(close_val / current_bps, 2)

            points.append({
                "time": d,
                "eps": _nullable_float(current_eps),
                "bps": _nullable_float(current_bps),
                "per": _nullable_float(per_val),
                "pbr": _nullable_float(pbr_val),
            })

        default_per = points[-1]["per"] if points else None
        default_pbr = points[-1]["pbr"] if points else None
        return points, default_per, default_pbr, resolved_market, False, warnings

    except Exception as exc:
        log.warning("valuation points failed (%s)", exc)
        warnings.append(f"밸류에이션 계산 실패: {exc}")
        return [], None, None, resolved_market, True, warnings


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_bars(ticker: str, days: int = 504) -> list[dict]:
    """Generate ~2 years of synthetic OHLCV data (deterministic per ticker)."""
    rng = random.Random(sum(ord(c) for c in ticker))
    price = rng.uniform(50, 500)
    today = date.today()
    start = today - timedelta(days=days)
    bars: list[dict] = []
    d = start
    while d <= today:
        if d.weekday() < 5:  # Mon-Fri
            change = rng.gauss(0.0003, 0.015)
            price = max(1.0, price * (1 + change))
            noise = price * 0.012
            o = price + rng.uniform(-noise, noise)
            h = max(o, price) + abs(rng.gauss(0, noise / 2))
            l = min(o, price) - abs(rng.gauss(0, noise / 2))
            bars.append({
                "time":   str(d),
                "open":   round(o, 2),
                "high":   round(h, 2),
                "low":    round(max(0.01, l), 2),
                "close":  round(price, 2),
                "volume": int(rng.uniform(1e6, 10e6)),
            })
        d += timedelta(days=1)
    return bars


def _safe_float(val) -> float:
    try:
        f = float(val)
        return f if not math.isnan(f) else 0.0
    except Exception:
        return 0.0


def _nullable_float(val) -> float | None:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def _last_valid_float(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    s = pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
    if s.empty:
        return None
    return _nullable_float(s.iloc[-1])
