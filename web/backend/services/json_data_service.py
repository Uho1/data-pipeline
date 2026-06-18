"""JSON-file-based data service for serving per-ticker data.

Reads from local data/tickers/{market}/{ticker}.json first,
falls back to Cloudflare R2 if local file is missing.
Uses an LRU cache to avoid repeated reads for the same ticker.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from web.backend.core.config import settings

_DATA_DIR = Path(settings.data_dir)
_TICKERS_DIR = _DATA_DIR / "tickers"
_META_DIR = _DATA_DIR / "meta"

# R2 public URL — from settings (MDL_R2_PUBLIC_URL env var or .env)
_R2_PUBLIC_URL = (settings.r2_public_url or os.environ.get("R2_PUBLIC_URL", "")).strip().rstrip("/")


# ---------------------------------------------------------------------------
# Cache & Loading
# ---------------------------------------------------------------------------

def _fetch_from_r2(relative_path: str) -> dict | None:
    """Fetch JSON from R2 public URL. Returns None on failure."""
    if not _R2_PUBLIC_URL:
        return None
    url = f"{_R2_PUBLIC_URL}/{relative_path}"
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "StocksGram/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _cache_version_token(path: Path) -> int | None:
    """Return a cache-busting token that changes when the local file changes."""
    try:
        if path.exists():
            return path.stat().st_mtime_ns
    except Exception:
        return None
    return None


@lru_cache(maxsize=256)
def _load_json_file(path: str, version_token: int | None = None) -> dict | None:
    """Load and cache a JSON file. Local first, R2 fallback.

    The cache key includes a local-file version token so updated JSON files
    are automatically reloaded without requiring a backend restart.
    """
    p = Path(path)
    # Local first
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    # R2 fallback
    try:
        rel = p.relative_to(_DATA_DIR)
        return _fetch_from_r2(str(rel).replace("\\", "/"))
    except (ValueError, Exception):
        return None


def invalidate_cache() -> None:
    """Clear the JSON file cache (e.g., after data update)."""
    _load_json_file.cache_clear()


def invalidate_ticker(ticker: str, market: str = "kr") -> None:
    """Remove a specific ticker from cache by clearing all (LRU doesn't support key removal)."""
    # LRU cache doesn't support per-key eviction; clear all
    _load_json_file.cache_clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ticker_data(ticker: str, market: str = "kr") -> dict | None:
    """Load full ticker JSON data. Returns None if file doesn't exist."""
    file_path = _TICKERS_DIR / market / f"{ticker}.json"
    return _load_json_file(str(file_path), _cache_version_token(file_path))


def load_ticker_master(market: str = "kr") -> list[dict]:
    """Load ticker master list for a market."""
    file_path = _META_DIR / f"ticker_master_{market}.json"
    data = _load_json_file(str(file_path), _cache_version_token(file_path))
    if data is None:
        return []
    return data.get("items", [])


def load_last_updated() -> dict:
    """Load last_updated metadata."""
    file_path = _META_DIR / "last_updated.json"
    return _load_json_file(str(file_path), _cache_version_token(file_path)) or {}


def load_valuation_ttm_data(ticker: str, market: str = "kr") -> dict | None:
    """Load precomputed valuation TTM sidecar JSON for a ticker."""
    file_path = _TICKERS_DIR / market / f"{ticker}.valuation_ttm.json"
    return _load_json_file(str(file_path), _cache_version_token(file_path))


# ---------------------------------------------------------------------------
# Data extraction helpers (from loaded ticker JSON)
# ---------------------------------------------------------------------------

def get_price_bars(ticker: str, market: str = "kr", start: str | None = None, end: str | None = None) -> list[dict]:
    """Extract price bars from ticker JSON in the format expected by PriceResponse."""
    data = load_ticker_data(ticker, market)
    if data is None or "prices" not in data:
        return []

    prices = data["prices"]
    dates = prices.get("dates", [])
    opens = prices.get("open", [])
    highs = prices.get("high", [])
    lows = prices.get("low", [])
    closes = prices.get("close", [])
    volumes = prices.get("volume", [])

    bars = []
    for i, d in enumerate(dates):
        if start and d < start:
            continue
        if end and d > end:
            continue
        bars.append({
            "time": d,
            "open": opens[i] if i < len(opens) else None,
            "high": highs[i] if i < len(highs) else None,
            "low": lows[i] if i < len(lows) else None,
            "close": closes[i] if i < len(closes) else None,
            "volume": volumes[i] if i < len(volumes) else None,
        })
    return bars


def get_financials_dataframe(ticker: str, market: str = "kr") -> pd.DataFrame | None:
    """Convert the ticker JSON financials block back to a DataFrame.

    This provides backward compatibility with existing ticker_analysis_service
    that expects a DataFrame with columns like 'Revenue', 'Operating Income', etc.
    """
    data = load_ticker_data(ticker, market)
    if data is None or "financials" not in data:
        return None

    fin = data["financials"]
    periods = fin.get("periods", [])
    if not periods:
        return None

    rows: dict[str, list] = {"PeriodEnd": periods}
    for key in ("term", "fiscal_year", "fiscal_quarter", "fiscal_label"):
        values = fin.get(key)
        if isinstance(values, list) and len(values) == len(periods):
            rows[key] = values

    # Flatten income/balance/cashflow into column names matching _FIN_SCHEMA
    _KEY_TO_COL = {
        # income
        "revenue": "Revenue",
        "cogs": "COGS",
        "gross_profit": "Gross Profit",
        "sga": "SG&A",
        "rd": "R&D",
        "operating_income": "Operating Income",
        "net_income": "Net Income",
        "eps": "EPS",
        "diluted_eps": "Diluted EPS",
        "da": "D&A",
        "sbc": "SBC",
        "interest": "Interest",
        "pretax_income": "Pretax Income",
        "tax": "Tax",
        # balance
        "total_assets": "Total Assets",
        "total_liabilities": "Total Liabilities",
        "shareholders_equity": "Shareholders Equity",
        "current_assets": "Current Assets",
        "current_liabilities": "Current Liabilities",
        "ar": "AR",
        "ap": "AP",
        "inventory": "Inventory",
        "cash": "Cash",
        "debt_short": "Debt Short",
        "debt_long": "Debt Long",
        "deferred_revenue": "Deferred Revenue",
        "goodwill": "Goodwill",
        "intangibles": "Intangibles",
        # cashflow
        "cfo": "Operating Cash Flow",
        "cfi": "Investing Cash Flow",
        "cff": "Financing Cash Flow",
        "capex": "Capital Expenditure",
        "ppe_capex": "PPE CapEx",
        "dividends_paid": "Dividends Paid",
        "repurchases": "Repurchases",
    }

    n_periods = len(periods)
    for section in ("income", "balance", "cashflow"):
        section_data = fin.get(section, {})
        for key, values in section_data.items():
            if len(values) != n_periods:
                continue  # skip mismatched arrays
            col_name = _KEY_TO_COL.get(key, key)
            rows[col_name] = values

    # Extra top-level financial fields
    if "market_cap" in fin and len(fin["market_cap"]) == n_periods:
        rows["MarketCap"] = fin["market_cap"]
    if "shares_outstanding" in fin and len(fin["shares_outstanding"]) == n_periods:
        rows["Shares"] = fin["shares_outstanding"]

    df = pd.DataFrame(rows)
    df["PeriodEnd"] = pd.to_datetime(df["PeriodEnd"])
    df["ticker"] = ticker
    df["market"] = market
    return df


def get_financials_block(ticker: str, market: str = "kr", basis: str = "quarter") -> dict | None:
    """Get pre-computed financials block directly.

    basis: "quarter" (raw), "ttm" (trailing 4Q rolling), "annual" (fiscal year sum)
    """
    data = load_ticker_data(ticker, market)
    if data is None:
        return None

    if basis == "ttm":
        return data.get("financials_ttm")
    elif basis == "annual":
        return data.get("financials_annual")
    else:
        return data.get("financials")


def get_ticker_info(ticker: str, market: str = "kr") -> dict:
    """Get ticker metadata from the JSON file."""
    data = load_ticker_data(ticker, market)
    if data is None:
        return {"ticker": ticker, "market": market}
    result = {
        "ticker": data.get("ticker", ticker),
        "market": data.get("market", market),
        "company_name": data.get("name", ""),
        "sector": data.get("sector", ""),
        "industry": data.get("industry", ""),
        "market_tier": "",
    }
    # Add Korean name from ticker master for US stocks
    if market == "us":
        master = load_ticker_master("us")
        for item in master:
            if item.get("ticker") == ticker:
                result["name_kr"] = item.get("name_kr", "")
                break
    return result


def list_available_tickers(market: str = "kr") -> list[str]:
    """List all tickers that have JSON files."""
    ticker_dir = _TICKERS_DIR / market
    if not ticker_dir.exists():
        return []
    return sorted(p.stem for p in ticker_dir.glob("*.json"))
