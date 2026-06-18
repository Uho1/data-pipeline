"""Universe Builder — 보통주 중 재무데이터 보유 + 시가총액 상위 N 종목 선별.

Usage:
    market-data build-universe --market kr --top-n 2000

Reads from existing per-ticker JSON files (data/tickers/{market}/*.json)
to extract latest market cap and financial data availability, then produces:
    data/meta/universe_kr.json   (or universe_us.json)

The universe list can then drive ingest/export pipelines so that only
selected tickers are processed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = Path(os.environ.get("MDL_EXPORT_DIR", _REPO_ROOT / "data"))
_TICKERS_DIR = _DATA_ROOT / "tickers"
_META_DIR = _DATA_ROOT / "meta"


# ---------------------------------------------------------------------------
# Core: scan existing JSON files and rank by market cap
# ---------------------------------------------------------------------------

def _get_latest_market_cap(prices: dict) -> float | None:
    """Extract latest non-null market_cap from prices block."""
    mcaps = prices.get("market_cap", [])
    for v in reversed(mcaps):
        if v is not None and v > 0:
            return float(v)
    return None


def _has_financial_data(data: dict) -> bool:
    """Check if ticker JSON has meaningful financial data."""
    fin = data.get("financials")
    if not fin:
        return False
    periods = fin.get("periods", [])
    if not periods:
        return False
    # Must have at least one statement type with data
    for key in ("income", "balance", "cashflow"):
        stmt = fin.get(key, {})
        if stmt and any(v for values in stmt.values() for v in values if v is not None):
            return True
    return False


def _is_common_stock_by_name(name: str) -> bool:
    """Quick heuristic to filter out preferred stocks etc. from name.

    Korean preferred stocks typically have suffixes like 우, 우B, 우C,
    or contain keywords like 신주인수권, 전환사채 etc.
    """
    if not name:
        return True  # If no name, don't filter out
    name = name.strip()
    # Preferred stock suffixes
    if name.endswith(("우", "우B", "우C", "(우)", "1우", "2우", "3우")):
        return False
    # Preferred stock patterns
    preferred_keywords = ["우선주", "신주인수권", "전환사채", "스팩", "SPAC"]
    for kw in preferred_keywords:
        if kw in name:
            return False
    return True


def scan_tickers(
    market: str = "kr",
    *,
    require_financials: bool = True,
    require_market_cap: bool = True,
    common_stock_only: bool = True,
) -> pd.DataFrame:
    """Scan all per-ticker JSON files and return a DataFrame with metadata.

    Returns DataFrame with columns:
        ticker, name, sector, industry, market_tier,
        has_prices, has_financials, has_segments,
        latest_market_cap, financial_periods_count
    """
    ticker_dir = _TICKERS_DIR / market
    if not ticker_dir.exists():
        print(f"[universe] No ticker directory: {ticker_dir}")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    files = sorted(f for f in os.listdir(ticker_dir) if f.endswith(".json"))

    for fname in files:
        fpath = ticker_dir / fname
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        ticker = data.get("ticker", fname.replace(".json", ""))
        name = data.get("name", "")

        # Common stock filter
        if common_stock_only and not _is_common_stock_by_name(name):
            continue

        prices = data.get("prices", {})
        financials = data.get("financials", {})
        segments = data.get("segments", {})

        has_prices = bool(prices.get("dates"))
        has_fin = _has_financial_data(data)
        has_segs = bool(segments)
        latest_mcap = _get_latest_market_cap(prices) if has_prices else None
        fin_periods = len(financials.get("periods", []))

        # Apply filters
        if require_financials and not has_fin:
            continue
        if require_market_cap and (latest_mcap is None or latest_mcap <= 0):
            continue

        rows.append({
            "ticker": ticker,
            "name": name,
            "sector": data.get("sector", ""),
            "industry": data.get("industry", ""),
            "market_tier": "",  # Will be enriched from ticker_master
            "has_prices": has_prices,
            "has_financials": has_fin,
            "has_segments": has_segs,
            "latest_market_cap": latest_mcap,
            "financial_periods_count": fin_periods,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Enrich market_tier from ticker_master JSON or DuckDB
    _enrich_market_tier(df, market)

    return df


def _enrich_market_tier(df: pd.DataFrame, market: str) -> None:
    """Fill in market_tier from ticker_master meta or DuckDB."""
    if df.empty:
        return

    # First try: ticker_master JSON meta
    master_path = _META_DIR / f"ticker_master_{market}.json"
    tier_map: dict[str, str] = {}
    if master_path.exists():
        try:
            with open(master_path, "r", encoding="utf-8") as fh:
                master = json.load(fh)
            for item in master.get("items", []):
                tier = item.get("market_tier", "")
                if tier:
                    tier_map[item["ticker"]] = tier
        except Exception:
            pass

    # Second try: DuckDB ticker_master (has more reliable market_tier data)
    if market == "kr":
        try:
            from market_data import db_reader_kr
            master_df = db_reader_kr.load_ticker_master_all()
            if master_df is not None and not master_df.empty:
                for _, row in master_df.iterrows():
                    ticker = str(row.get("ticker", ""))
                    tier = str(row.get("market_tier", "") or "")
                    if tier and ticker:
                        tier_map[ticker] = tier
        except Exception:
            pass

    # Third try: fetch KRX corp list directly (most reliable for market_tier)
    missing_tickers = [t for t in df["ticker"] if t not in tier_map or not tier_map[t]]
    if missing_tickers and market == "kr":
        try:
            from market_data.krx.universe import fetch_krx_corp_list
            corp_list = fetch_krx_corp_list()
            if not corp_list.empty:
                for _, row in corp_list.iterrows():
                    ticker = str(row.get("ticker", ""))
                    tier = str(row.get("market_tier", "") or "")
                    if tier and ticker:
                        tier_map[ticker] = tier
                print(f"[universe] Enriched market_tier from KRX corp list ({len(tier_map)} tickers)")
        except Exception as exc:
            print(f"[universe] WARN: KRX corp list enrichment failed: {exc}")

    if tier_map:
        df["market_tier"] = df["ticker"].map(tier_map).fillna(df["market_tier"])


def build_universe(
    market: str = "kr",
    top_n: int = 2000,
    *,
    require_financials: bool = True,
    require_market_cap: bool = True,
    common_stock_only: bool = True,
    min_market_cap: float = 0,
) -> pd.DataFrame:
    """Build universe: scan tickers, filter, rank by market cap, take top N.

    Args:
        market: "kr" or "us"
        top_n: Maximum number of tickers (0 = all that pass filters)
        require_financials: Only include tickers with financial data
        require_market_cap: Only include tickers with market cap data
        common_stock_only: Filter out preferred stocks by name
        min_market_cap: Minimum market cap threshold (in raw currency units)

    Returns:
        DataFrame sorted by market_cap descending, limited to top_n
    """
    print(f"[universe] Scanning {market} tickers...")
    df = scan_tickers(
        market,
        require_financials=require_financials,
        require_market_cap=require_market_cap,
        common_stock_only=common_stock_only,
    )

    if df.empty:
        print("[universe] No tickers found matching criteria")
        return df

    print(f"[universe] {len(df)} tickers pass initial filters")

    # Apply minimum market cap filter
    if min_market_cap > 0:
        df = df[df["latest_market_cap"] >= min_market_cap].copy()
        print(f"[universe] {len(df)} tickers above min market cap {min_market_cap:,.0f}")

    # Sort by market cap descending
    df = df.sort_values("latest_market_cap", ascending=False).reset_index(drop=True)

    # Apply top N
    if top_n > 0 and len(df) > top_n:
        df = df.head(top_n).reset_index(drop=True)
        print(f"[universe] Trimmed to top {top_n}")

    print(f"[universe] Final universe: {len(df)} tickers")
    return df


# ---------------------------------------------------------------------------
# Save / Load universe
# ---------------------------------------------------------------------------

def save_universe(df: pd.DataFrame, market: str = "kr") -> Path:
    """Save universe to data/meta/universe_{market}.json."""
    _META_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _META_DIR / f"universe_{market}.json"

    items = []
    for _, row in df.iterrows():
        items.append({
            "ticker": str(row["ticker"]),
            "name": str(row.get("name", "")),
            "sector": str(row.get("sector", "")),
            "industry": str(row.get("industry", "")),
            "market_tier": str(row.get("market_tier", "")),
            "latest_market_cap": row.get("latest_market_cap"),
            "has_financials": bool(row.get("has_financials", False)),
            "has_segments": bool(row.get("has_segments", False)),
            "financial_periods_count": int(row.get("financial_periods_count", 0)),
        })

    payload = {
        "market": market,
        "count": len(items),
        "top_n": len(items),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "filters": {
            "common_stock_only": True,
            "require_financials": True,
            "require_market_cap": True,
        },
        "items": items,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[universe] Saved to {out_path}")
    return out_path


def load_universe(market: str = "kr") -> list[str]:
    """Load universe ticker list from data/meta/universe_{market}.json.

    Returns list of ticker strings. If no universe file exists, returns empty list.
    """
    path = _META_DIR / f"universe_{market}.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [item["ticker"] for item in data.get("items", [])]
    except (json.JSONDecodeError, KeyError, OSError):
        return []


def load_universe_df(market: str = "kr") -> pd.DataFrame:
    """Load universe as DataFrame. Returns empty DataFrame if not found."""
    path = _META_DIR / f"universe_{market}.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data.get("items", []))
    except (json.JSONDecodeError, KeyError, OSError):
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _update_ticker_master_tiers(market: str, tier_map: dict[str, str]) -> None:
    """Update ticker_master_{market}.json with enriched market_tier values."""
    master_path = _META_DIR / f"ticker_master_{market}.json"
    if not master_path.exists() or not tier_map:
        return
    try:
        with open(master_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        updated = 0
        for item in data.get("items", []):
            ticker = item.get("ticker", "")
            if ticker in tier_map and tier_map[ticker]:
                if item.get("market_tier", "") != tier_map[ticker]:
                    item["market_tier"] = tier_map[ticker]
                    updated += 1
        if updated > 0:
            with open(master_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[universe] Updated {updated} market_tier entries in ticker_master_{market}.json")
    except Exception as exc:
        print(f"[universe] WARN: Failed to update ticker_master: {exc}")


def run_build_universe(
    market: str = "kr",
    top_n: int = 2000,
    min_market_cap: float = 0,
) -> None:
    """CLI entry: build and save universe."""
    df = build_universe(
        market=market,
        top_n=top_n,
        min_market_cap=min_market_cap,
    )
    if df.empty:
        print("[universe] Empty universe — nothing to save")
        return

    # Also update ticker_master with enriched market_tier data
    if not df.empty and "market_tier" in df.columns:
        tier_map = dict(zip(df["ticker"], df["market_tier"]))
        _update_ticker_master_tiers(market, tier_map)

    path = save_universe(df, market=market)

    # Print summary
    print("\n=== Universe Summary ===")
    print(f"Market: {market}")
    print(f"Total tickers: {len(df)}")
    if "market_tier" in df.columns:
        tier_counts = df["market_tier"].value_counts()
        for tier, count in tier_counts.items():
            if tier:
                print(f"  {tier}: {count}")
    if "latest_market_cap" in df.columns:
        mcaps = df["latest_market_cap"].dropna()
        if not mcaps.empty:
            print(f"Market cap range: {mcaps.min():,.0f} ~ {mcaps.max():,.0f}")
            print(f"Market cap median: {mcaps.median():,.0f}")
    print(f"With financials: {df['has_financials'].sum()}")
    print(f"With segments: {df['has_segments'].sum()}")
    print(f"Saved to: {path}")
