from __future__ import annotations

from dataclasses import dataclass

from market_data.sec_financials import SEC_DEFAULT_START_DATE, build_sec_cache_for_price_tickers, price_tickers


@dataclass
class SecCacheOptions:
    market: str = "us"
    start: str = str(SEC_DEFAULT_START_DATE.date())
    user_agent: str | None = None
    retries: int = 3
    backoff_base: float = 1.0
    force: bool = False


def run_sec_cache(opts: SecCacheOptions) -> int:
    market = str(opts.market or "us").strip().lower()
    if market != "us":
        print("[ERROR] SEC cache currently supports only market=us")
        return 2

    tickers = price_tickers(market=market)
    if not tickers:
        print("[ERROR] No price parquet tickers found under data/prices/us")
        return 2

    print(f"[RUN] SEC cache update start tickers={len(tickers)} market={market} start={opts.start}")

    def _progress(ticker: str, status: str) -> None:
        print(f"{ticker}...{status}")

    ok, skipped, failed = build_sec_cache_for_price_tickers(
        market=market,
        start=opts.start,
        user_agent=opts.user_agent,
        force_refresh=opts.force,
        retries=max(0, int(opts.retries)),
        backoff=max(float(opts.backoff_base), 0.1),
        progress_cb=_progress,
    )

    print(f"[DONE] sec-cache ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2
