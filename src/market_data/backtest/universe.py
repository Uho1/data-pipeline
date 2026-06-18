from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import pandas as pd

from market_data.backtest.models import UniverseConfig, UniverseFilterConfig
from market_data.backtest.rules import evaluate_rule
from market_data.sp500_pit import load_sp500_constituents_pit


def load_symbols_from_file(path: str | Path) -> list[str]:
    p = Path(path).expanduser()
    return list(_load_symbols_from_file_cached(str(p)))


@lru_cache(maxsize=32)
def _load_symbols_from_file_cached(path: str) -> tuple[str, ...]:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"symbols file not found: {p}")
    df = pd.read_csv(p)
    if df.empty:
        return tuple()
    col = df.columns[0]
    out = [str(v).strip().upper() for v in df[col].dropna().tolist() if str(v).strip()]
    return tuple(dict.fromkeys(out))


def _is_adr_symbol(symbol: str) -> bool:
    s = str(symbol).strip().upper()
    # Common ADR tickers in US often end with Y/F (OTC) with 5 letters.
    return (len(s) == 5 and s.endswith(("Y", "F"))) or "." in s


def _is_preferred_symbol(symbol: str) -> bool:
    s = str(symbol).strip().upper()
    if "-" in s:
        return True
    return bool(re.search(r"(PR|PFD|PREF)$", s))


def _is_non_common_stock_ticker(symbol: str) -> bool:
    s = str(symbol).strip().upper()
    # Category 1: Units, Warrants, Rights (U, W, R suffixes)
    # Nasdaq often uses 5th character or suffix after a space/dash
    if len(s) > 3 and s[-1] in ("U", "W", "R"):
        return True
    
    # Category 1: Preferred shares, secondary issues (., -, PR, P)
    if "." in s or "-" in s:
        return True
    if "PR" in s:
        return True
    # If ends with P and length > 3 (e.g., AAPLP)
    if len(s) > 3 and s.endswith("P"):
        return True
        
    return False


@lru_cache(maxsize=65536)
def _has_financial_data(symbol: str, market: str = "us") -> bool:
    sym = str(symbol).strip().upper()
    if not sym:
        return False

    fin_tickers = _load_financial_ticker_set(str(market).strip().lower())
    return sym in fin_tickers


@lru_cache(maxsize=8)
def _load_exchange_symbols_from_nasdaqtraded(exchange_codes: tuple[str, ...]) -> tuple[str, ...]:
    raw_path = Path("data/universe/nasdaqtraded_raw.txt")
    if not raw_path.exists():
        return tuple()
    try:
        df = pd.read_csv(raw_path, sep="|", dtype=str)
    except Exception:
        return tuple()
    if df.empty or "Listing Exchange" not in df.columns or "Symbol" not in df.columns:
        return tuple()

    exch_set = {str(x).strip().upper() for x in exchange_codes if str(x).strip()}
    if not exch_set:
        return tuple()

    ex = df["Listing Exchange"].astype(str).str.strip().str.upper()
    syms = df["Symbol"].astype(str).str.strip().str.upper()
    mask = ex.isin(exch_set)
    if "Test Issue" in df.columns:
        mask &= df["Test Issue"].astype(str).str.strip().str.upper().ne("Y")
    # Strip footer/noise rows (e.g. "File Creation Time ...").
    mask &= syms.str.fullmatch(r"[A-Z0-9.\-]+").fillna(False)
    out = syms.loc[mask].tolist()
    return tuple(dict.fromkeys(out))


def _resolve_nyse_symbols() -> list[str]:
    try:
        return load_symbols_from_file("data/universe/symbols_nyse.csv")
    except Exception:
        pass
    # NasdaqTrader symdir codes: N=NYSE, A=NYSE American, P=NYSE Arca.
    return list(_load_exchange_symbols_from_nasdaqtraded(("N", "A", "P")))


def _filter_financial_symbols(symbols: list[str], market: str) -> list[str]:
    mkt = str(market).strip().lower() or "us"
    return [s for s in symbols if _has_financial_data(s, market=mkt)]


@lru_cache(maxsize=8)
def _load_financial_ticker_set(market: str = "us") -> frozenset[str]:
    mkt = str(market).strip().lower() or "us"

    # DuckDB path (fast): one query, reused across all rebalance steps.
    try:
        from market_data.db_router import db_available_for_market, get_connection_for_market

        if db_available_for_market(mkt):
            con = get_connection_for_market(mkt)
            rows = con.execute(
                "SELECT DISTINCT ticker FROM financials_quarterly WHERE market = ?",
                [mkt],
            ).fetchall()
            if rows:
                return frozenset(str(r[0]).strip().upper() for r in rows if r and r[0])
    except Exception:
        return frozenset()

    return frozenset()


def resolve_universe_symbols(config: UniverseConfig, available_symbols: list[str]) -> list[str]:
    available = [str(s).strip().upper() for s in available_symbols if str(s).strip()]
    avail_set = set(available)

    source = str(config.source or "").strip().lower()
    if source == "sp500":
        try:
            symbols = load_symbols_from_file("data/universe/symbols_sp500.csv")
        except Exception:
            symbols = available
    elif source == "sp500_financial_only":
        try:
            symbols = _filter_financial_symbols(load_symbols_from_file("data/universe/symbols_sp500.csv"), config.market)
        except Exception:
            symbols = _filter_financial_symbols(available, config.market)
    elif source == "nasdaq_all":
        try:
            symbols = load_symbols_from_file("data/universe/symbols_nasdaq.csv")
        except Exception:
            symbols = available
    elif source == "nyse_all":
        symbols = _resolve_nyse_symbols() or available
    elif source == "nyse_stock_only_financial":
        symbols = _resolve_nyse_symbols()
        if symbols:
            symbols = [s for s in symbols if not _is_non_common_stock_ticker(s)]
            symbols = _filter_financial_symbols(symbols, config.market)
        else:
            symbols = []
    elif source == "sp500_pit":
        try:
            pit = load_sp500_constituents_pit()
            if pit is None or pit.empty:
                symbols = available
            else:
                symbols = pit["ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
        except Exception:
            symbols = available
    elif source == "sp500_pit_financial_only":
        try:
            pit = load_sp500_constituents_pit()
            if pit is None or pit.empty:
                symbols = _filter_financial_symbols(available, config.market)
            else:
                base = pit["ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
                symbols = _filter_financial_symbols(base, config.market)
        except Exception:
            symbols = _filter_financial_symbols(available, config.market)
    elif source == "nasdaq_stock_only_financial":
        try:
            # 1. Load from the provided nasdaq stock only file
            raw_list = load_symbols_from_file("data/universe/symbols_nasdaq_stock_only.csv")
            
            # 2. Filter by ticker pattern (Only clean common stocks)
            common_only = [s for s in raw_list if not _is_non_common_stock_ticker(s)]
            
            # 3. Filter by actual data existence
            symbols = _filter_financial_symbols(common_only, config.market)
        except Exception:
            symbols = []
    elif config.symbols:
        symbols = [s for s in [str(x).strip().upper() for x in config.symbols] if s]
    elif config.symbols_file:
        symbols = load_symbols_from_file(config.symbols_file)
    else:
        symbols = available

    symbols = [s for s in symbols if s in avail_set]
    return list(dict.fromkeys(symbols))


def _market_cap_bucket(value: float) -> str:
    if not pd.notna(value):
        return "unknown"
    v = float(value)
    if v >= 200_000_000_000:
        return "mega"
    if v >= 10_000_000_000:
        return "large"
    if v >= 2_000_000_000:
        return "mid"
    if v >= 300_000_000:
        return "small"
    return "micro"


def apply_universe_filters(
    frame_t: pd.DataFrame,
    config: UniverseConfig,
    filter_config: UniverseFilterConfig | None = None,
) -> pd.DataFrame:
    if frame_t is None or frame_t.empty:
        return pd.DataFrame()

    out = frame_t.copy()

    if config.symbols:
        allowed = {str(s).strip().upper() for s in config.symbols if str(s).strip()}
        out = out.loc[out.index.astype(str).str.upper().isin(allowed)]

    if config.symbols_file:
        try:
            allowed = set(load_symbols_from_file(config.symbols_file))
            out = out.loc[out.index.astype(str).str.upper().isin(allowed)]
        except Exception:
            # For optional file mode we ignore missing/invalid files in runtime filters.
            pass

    if config.source == "screen" and config.screen:
        mask = evaluate_rule(config.screen, out, na_policy="fail")
        out = out.loc[mask]

    fcfg = filter_config or UniverseFilterConfig()

    if config.min_market_cap is not None and "market_cap" in out.columns:
        out = out.loc[pd.to_numeric(out["market_cap"], errors="coerce") >= float(config.min_market_cap)]

    if config.min_dollar_volume_20d is not None and "dollar_volume_20d" in out.columns:
        out = out.loc[
            pd.to_numeric(out["dollar_volume_20d"], errors="coerce") >= float(config.min_dollar_volume_20d)
        ]

    idx_upper = out.index.astype(str).str.upper()
    if config.exclude_adr:
        adr_mask = idx_upper.map(_is_adr_symbol)
        out = out.loc[~adr_mask]

    if config.exclude_preferred:
        pref_mask = idx_upper.map(_is_preferred_symbol)
        out = out.loc[~pref_mask]

    if config.exclude_distress and "price" in out.columns:
        out = out.loc[pd.to_numeric(out["price"], errors="coerce") >= 1.0]

    merged_excludes = {str(s).strip().upper() for s in (config.exclude_tickers + fcfg.exclude_tickers) if str(s).strip()}
    if merged_excludes:
        out = out.loc[~out.index.astype(str).str.upper().isin(merged_excludes)]

    exchange_filters = [str(x).strip().upper() for x in (config.exchanges + fcfg.exchanges) if str(x).strip()]
    if exchange_filters:
        exchange_col = None
        for name in ["exchange", "listing_exchange", "primary_exchange"]:
            if name in out.columns:
                exchange_col = name
                break
        if exchange_col is not None:
            ex_s = out[exchange_col].astype(str).str.upper()
            out = out.loc[ex_s.isin(set(exchange_filters))]

    size_filters = [str(x).strip().lower() for x in (config.size_buckets + fcfg.size_buckets) if str(x).strip()]
    if size_filters and "market_cap" in out.columns:
        buckets = pd.to_numeric(out["market_cap"], errors="coerce").map(_market_cap_bucket)
        out = out.loc[buckets.isin(set(size_filters))]

    if "sector_l1_kr" in out.columns:
        sector_s = out["sector_l1_kr"].astype(str)
        include_sectors = config.include_sectors + fcfg.include_sectors
        exclude_sectors = config.exclude_sectors + fcfg.exclude_sectors
        if include_sectors:
            allow = {str(x).strip() for x in include_sectors if str(x).strip()}
            out = out.loc[sector_s.isin(allow)]
        if exclude_sectors:
            deny = {str(x).strip() for x in exclude_sectors if str(x).strip()}
            out = out.loc[~sector_s.isin(deny)]

    if "sector_l2_kr" in out.columns:
        subsector_s = out["sector_l2_kr"].astype(str)
        include_sub = config.include_subsectors + fcfg.include_subsectors
        exclude_sub = config.exclude_subsectors + fcfg.exclude_subsectors
        if include_sub:
            allow = {str(x).strip() for x in include_sub if str(x).strip()}
            out = out.loc[subsector_s.isin(allow)]
        if exclude_sub:
            deny = {str(x).strip() for x in exclude_sub if str(x).strip()}
            out = out.loc[~subsector_s.isin(deny)]

    if fcfg.financial_data_only:
        idx_up = out.index.astype(str).str.upper()
        fin_tickers = _load_financial_ticker_set(str(config.market).strip().lower())
        if fin_tickers:
            out = out.loc[idx_up.isin(fin_tickers)]
        else:
            fin_mask = idx_up.map(lambda s: _has_financial_data(s, market=config.market))
            out = out.loc[fin_mask]

    return out
