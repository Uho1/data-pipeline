from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from market_data.config import DATA_DIR, KOSPI_EXTERNAL_DEFAULT_URL, KOSPI_TOP_N_DEFAULT
from market_data.finterstellar_financials import fetch_consolidated_universe_history
from market_data.universe import build_universe
from market_data.utils import ensure_dir, now_utc_iso


@dataclass
class FinterstellarCacheOptions:
    universe: str
    tickers_file: str | None
    start: str
    end: str | None
    kospi_external_url: str
    kospi_top_n: int | None
    otp: str | None
    retries: int
    backoff_base: float
    vol: int
    study: str
    years: int
    cache_dir: str | None
    force: bool


def _default_cache_dir(universe: str, market: str) -> Path:
    return DATA_DIR / "finterstellar_term_cache" / market / universe


def run_finterstellar_cache(opts: FinterstellarCacheOptions) -> int:
    otp = (opts.otp or "").strip()
    if not otp:
        print("[ERROR] finterstellar OTP is required. Use --finterstellar-otp or FINTERSTELLAR_OTP.")
        return 2

    symbols, market, universe_path = build_universe(
        universe=opts.universe,
        tickers_file=opts.tickers_file,
        kospi_external_url=opts.kospi_external_url or KOSPI_EXTERNAL_DEFAULT_URL,
        kospi_top_n=opts.kospi_top_n if opts.kospi_top_n is not None else KOSPI_TOP_N_DEFAULT,
    )
    symbols = list(dict.fromkeys([str(s).strip().upper() for s in symbols if str(s).strip()]))
    if not symbols:
        print("[ERROR] Universe is empty.")
        return 2

    root = Path(opts.cache_dir).expanduser() if opts.cache_dir else _default_cache_dir(opts.universe, market)
    terms_dir = root / "terms"
    ensure_dir(terms_dir)

    print(f"[UNIVERSE] {opts.universe} symbols={len(symbols)} source={universe_path}")
    print(f"[CACHE DIR] {root}")
    print("[RUN] finterstellar quarterly universe fetch started")

    def _progress(term: str, status: str) -> None:
        print(f"{term}...{status}")

    history = fetch_consolidated_universe_history(
        otp=otp,
        symbols=symbols,
        start=opts.start,
        end=opts.end,
        history_years=opts.years,
        retries=opts.retries,
        backoff=opts.backoff_base,
        vol=opts.vol,
        study=opts.study,
        cache_dir=terms_dir,
        force=opts.force,
        progress_cb=_progress,
    )

    if history.empty:
        print("[DONE] No rows were returned for the selected universe/period.")
        return 2

    history = history.sort_values(["symbol", "StatementDate", "term"]).reset_index(drop=True)
    out_all = root / "universe_filtered.parquet"
    out_symbols = root / "symbols_in_cache.csv"
    out_manifest = root / "manifest.json"
    history.to_parquet(out_all, index=False)
    pd.DataFrame({"Symbol": sorted(history["symbol"].astype(str).unique())}).to_csv(out_symbols, index=False)

    manifest = {
        "created_at": now_utc_iso(),
        "universe": opts.universe,
        "market": market,
        "universe_source": str(universe_path),
        "symbols_requested": len(symbols),
        "symbols_returned": int(history["symbol"].astype(str).nunique()),
        "rows": int(len(history)),
        "start": opts.start,
        "end": opts.end,
        "history_years": opts.years,
        "term_cache_dir": str(terms_dir),
        "output_file": str(out_all),
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SAVE] {out_all}")
    print(f"[SAVE] {out_symbols}")
    print(f"[SAVE] {out_manifest}")
    print(
        f"[DONE] rows={manifest['rows']} symbols={manifest['symbols_returned']} "
        f"cache_terms_dir={terms_dir}"
    )
    return 0
