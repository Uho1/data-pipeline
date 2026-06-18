from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import pandas as pd
import yfinance as yf

from market_data.config import PRICE_REQUIRED_COLUMNS
from market_data import db_writer
from market_data import db_writer_kr
from market_data.db_router import db_available_for_market, get_connection_for_market, get_prices_connection_for_market
from market_data.krx.prices import fetch_price_frame as fetch_kr_price_frame
from market_data.utils import coerce_datetime_index, now_utc_iso


@dataclass
class PriceUpdateOptions:
    market: str = "us"
    tickers: list[str] | None = None
    start_default: str = "2000-01-01"
    interval: str = "1d"
    force_full: bool = False


def _normalize_ticker_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for t in values:
        s = str(t).strip().upper()
        if s:
            out.append(s)
    return list(dict.fromkeys(out))


def _discover_existing_tickers(market: str) -> list[str]:
    """Return tickers that have data in DuckDB."""
    try:
        if db_available_for_market(market):
            con = get_prices_connection_for_market(market)
            rows = con.execute(
                "SELECT DISTINCT ticker FROM prices WHERE market = ? ORDER BY ticker",
                [str(market).strip().lower()],
            ).fetchall()
            return [r[0] for r in rows]
    except Exception:
        pass
    return []


def _extract_single_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.get_level_values(-1)
        if ticker in levels:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _prepare_price_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = _extract_single_ticker_frame(raw, ticker)
    df = coerce_datetime_index(df)
    for col in PRICE_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col in {"Dividends", "Stock Splits"} else pd.NA
    out = df[PRICE_REQUIRED_COLUMNS].copy()
    out.index.name = "Date"
    out["Ticker"] = ticker
    out["CollectedAt"] = now_utc_iso()
    return out


def _fetch_shares_series(ticker: str, start: str) -> pd.Series | None:
    """Fetch historical shares outstanding from yfinance and forward-fill to daily."""
    try:
        t = yf.Ticker(ticker)
        shares = t.get_shares_full(start=start)
        if shares is None or shares.empty:
            return None
        # De-dup and sort, remove timezone for alignment with price index
        shares = shares[~shares.index.duplicated(keep="last")].sort_index()
        shares.index = shares.index.tz_localize(None)
        return shares
    except Exception:
        return None


def _download_price(ticker: str, start: str, end_exclusive: str, interval: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        start=start,
        end=end_exclusive,
        interval=interval,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = _prepare_price_frame(raw, ticker)

    # Compute daily MarketCap = Close × shares_outstanding
    shares = _fetch_shares_series(ticker, start)
    if shares is not None and not shares.empty:
        # Reindex shares to price dates with forward-fill
        shares_daily = shares.reindex(out.index, method="ffill")
        close = pd.to_numeric(out.get("Close") if "Close" in out.columns else out.get("Adj Close"), errors="coerce")
        out["MarketCap"] = close * shares_daily
    else:
        out["MarketCap"] = pd.NA

    return out


def _next_day_string(ts: pd.Timestamp) -> str:
    return (pd.Timestamp(ts).normalize() + pd.Timedelta(days=1)).date().isoformat()


def run_price_update(opts: PriceUpdateOptions, log_cb: Callable[[str], None] | None = None) -> int:
    def log(msg: str) -> None:
        if log_cb is None:
            print(msg)
        else:
            log_cb(msg)

    market = (opts.market or "us").strip().lower()
    if market not in {"us", "kr"}:
        log(f"[ERROR] invalid market: {market}. use us/kr")
        return 2

    # Ensure DB schema exists
    if market == "kr":
        db_writer_kr.init_schema()
    else:
        db_writer.init_schema()

    tickers = _normalize_ticker_list(opts.tickers)
    if not tickers:
        tickers = _discover_existing_tickers(market)
    if not tickers:
        log("[ERROR] no tickers to update (run ingest first or provide tickers)")
        return 2

    end_exclusive = (date.today() + timedelta(days=1)).isoformat()
    ok = 0
    skipped = 0
    failed = 0

    log(
        f"[RUN] price update market={market} tickers={len(tickers)} "
        f"mode={'full' if opts.force_full else 'append'} interval={opts.interval}"
    )

    for ticker in tickers:
        try:
            start = opts.start_default
            if not opts.force_full:
                latest = db_writer_kr.get_latest_price_date(ticker, market) if market == "kr" else db_writer.get_latest_price_date(ticker, market)
                if latest is not None:
                    start = _next_day_string(latest)

            if pd.to_datetime(start) >= pd.to_datetime(end_exclusive):
                log(f"{ticker}...SKIP (up-to-date)")
                skipped += 1
                continue

            if market == "kr":
                new_df = fetch_kr_price_frame(
                    ticker=ticker,
                    start=start.replace("-", ""),
                    end=(pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)).strftime("%Y%m%d"),
                )
            else:
                new_df = _download_price(
                    ticker=ticker,
                    start=start,
                    end_exclusive=end_exclusive,
                    interval=opts.interval,
                )
            if new_df.empty:
                log(f"{ticker}...SKIP (no new rows)")
                skipped += 1
                continue

            if opts.force_full:
                rows_written = db_writer_kr.upsert_prices(new_df, ticker, market) if market == "kr" else db_writer.upsert_prices(new_df, ticker, market)
            else:
                if market == "kr":
                    rows_written = db_writer_kr.upsert_prices(new_df, ticker, market)
                else:
                    rows_written = db_writer.append_prices(new_df, ticker, market)

            idx = pd.to_datetime(new_df.index, errors="coerce")
            rng = f"{idx.min().date()}->{idx.max().date()}" if len(idx) else "n/a"
            log(f"{ticker}...OK (added={rows_written}, range={rng})")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            log(f"{ticker}...FAIL ({exc})")
            failed += 1

    log(f"[DONE] ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2
