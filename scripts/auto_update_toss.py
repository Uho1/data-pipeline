"""Post-auto-update hook that regenerates toss-app static JSON for the
tickers that changed during the current auto-update run.

Designed to run right after `scripts/auto_update.py` inside the CI job:

    python scripts/auto_update.py ...
    python scripts/auto_update_toss.py       # <-- this script
    python scripts/upload_toss_r2.py         # uploads to r2://.../toss/

What it does:
  1. Reads `data/meta/auto_update_state.json` to find the tickers that
     changed in the most recent run.
  2. If no tickers changed, exits 0 (no-op).
  3. Ensures meta files (`ticker_master_{kr,us}.json`) are present on
     the runner — downloads from R2 if missing.
  4. Invokes `scripts/export_toss_data.py` once per market with
     `--ticker X` flags to regenerate just those tickers (skipping the
     index so we avoid redundant work).
  5. Rebuilds the per-market search indexes once at the end.

Every external dependency is failure-tolerant — if something breaks
we return non-zero but log clearly so the CI step can decide to either
halt or continue-on-error.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"
_META_DIR = _DATA_DIR / "meta"
_STATE_FILE = _META_DIR / "auto_update_state.json"
_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "export_toss_data.py"

# Meta files that `export_toss_data.py::_build_info` and `build_index`
# rely on. They're uploaded to R2 under `meta/` by `upload_r2.py`, so
# we can pull them back if the runner doesn't have them yet.
_REQUIRED_META = {
    "kr": "ticker_master_kr.json",
    "us": "ticker_master_us.json",
}

# Additional meta files we fetch so we can rebuild a broken ticker_master
# from the (more authoritative) universe file if needed.
_OPTIONAL_META = {
    "kr": ("universe_kr.json", "ticker_name_kr_us.json"),
    "us": ("universe_us.json", "ticker_name_kr_us.json"),
}

# Below this count, we consider `ticker_master_{market}.json` to be
# broken and rebuild it from universe. The main pipeline has produced
# incomplete masters in the past (e.g. only the 17 top-market-cap
# tickers) and we don't want toss data to inherit that regression.
_MASTER_HEALTH_MIN = {
    "kr": 1500,
    "us": 1000,
}


def _log(msg: str) -> None:
    print(f"[auto_update_toss] {msg}", flush=True)


def _r2_client():
    try:
        import boto3
        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env")
        account_id = os.environ.get("R2_ACCOUNT_ID")
        if not account_id:
            return None, None
        client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        return client, os.environ["R2_BUCKET_NAME"]
    except Exception as exc:  # noqa: BLE001
        _log(f"could not init R2 client: {exc}")
        return None, None


def _download_meta_from_r2() -> None:
    """Pull `meta/*.json` files from R2 if they're missing on the runner.

    Downloads both the required master files and the optional universe
    files that we use to rebuild masters when the master file is broken.
    """
    client, bucket = _r2_client()
    if client is None:
        _log("R2 credentials not set, cannot download meta files")
        return

    _META_DIR.mkdir(parents=True, exist_ok=True)

    wanted: set[str] = set()
    wanted.update(_REQUIRED_META.values())
    for pair in _OPTIONAL_META.values():
        wanted.update(pair)

    for name in sorted(wanted):
        local_path = _META_DIR / name
        if local_path.exists():
            continue
        r2_key = f"meta/{name}"
        try:
            client.download_file(bucket, r2_key, str(local_path))
            _log(f"downloaded {r2_key} → {local_path}")
        except Exception as exc:  # noqa: BLE001
            _log(f"[warn] could not download {r2_key}: {exc}")


def _master_count(market: str) -> int:
    """Return how many tickers are in the local master file (0 if absent)."""
    path = _META_DIR / _REQUIRED_META[market]
    if not path.exists():
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return len(data.get("items", []) or [])
    except Exception:  # noqa: BLE001
        return 0


def _repair_master_if_broken(market: str) -> None:
    """Rebuild `ticker_master_{market}.json` from `universe_{market}.json`
    when the master file is suspiciously small. Writes both fields that
    `export_toss_data._build_info` looks at: name, market_tier, sector,
    industry, name_kr.

    The main StocksGram pipeline has shipped broken masters before
    (e.g. 17-ticker US master); this keeps the toss pipeline immune
    without requiring a main pipeline fix.
    """
    count = _master_count(market)
    threshold = _MASTER_HEALTH_MIN[market]
    if count >= threshold:
        return

    universe_name = _OPTIONAL_META[market][0]
    universe_path = _META_DIR / universe_name
    if not universe_path.exists():
        _log(
            f"[warn] {market.upper()} master has only {count} tickers but "
            f"{universe_name} is missing — cannot repair"
        )
        return

    try:
        with open(universe_path, encoding="utf-8") as f:
            universe = json.load(f)
        uni_items = universe.get("items", []) or []
    except Exception as exc:  # noqa: BLE001
        _log(f"[warn] could not parse {universe_name}: {exc}")
        return

    if len(uni_items) < threshold:
        _log(
            f"[warn] {market.upper()} universe also too small "
            f"({len(uni_items)}), skipping repair"
        )
        return

    # Korean name overlay (US only; KR names are already in `name`)
    name_kr_map: dict[str, str] = {}
    kr_map_name = _OPTIONAL_META[market][1]
    kr_map_path = _META_DIR / kr_map_name
    if kr_map_path.exists():
        try:
            with open(kr_map_path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                name_kr_map = {k: v for k, v in raw.items() if isinstance(v, str)}
        except Exception as exc:  # noqa: BLE001
            _log(f"[warn] could not parse {kr_map_name}: {exc}")

    default_tier = "NYSE/NASDAQ" if market == "us" else "KOSPI"
    items = []
    for u in uni_items:
        ticker = (u.get("ticker") or "").strip()
        if not ticker:
            continue
        items.append(
            {
                "ticker": ticker,
                "name": u.get("name") or ticker,
                "market_tier": u.get("market_tier") or default_tier,
                "sector": u.get("sector") or "",
                "industry": u.get("industry") or "",
                "name_kr": name_kr_map.get(ticker, "") or "",
            }
        )
    items.sort(key=lambda x: x["ticker"])

    master = {
        "market": market,
        "count": len(items),
        "items": items,
    }
    out_path = _META_DIR / _REQUIRED_META[market]
    out_path.write_text(
        json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(
        f"[repair] {market.upper()} master rebuilt from universe: "
        f"{count} → {len(items)} tickers"
    )


def _run_export(args: list[str]) -> int:
    """Invoke `scripts/export_toss_data.py` with the given args."""
    cmd = [sys.executable, "-u", str(_EXPORT_SCRIPT), *args]
    _log(f"$ python {' '.join(str(a) for a in cmd[2:])}")
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    return result.returncode


def _chunks(seq: list[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> int:
    if not _STATE_FILE.exists():
        _log(f"state file not found: {_STATE_FILE}")
        _log("skipping (nothing to export)")
        return 0

    with open(_STATE_FILE, encoding="utf-8") as f:
        state = json.load(f)

    updated_kr: list[str] = state.get("updated_tickers_kr") or []
    updated_us: list[str] = state.get("updated_tickers_us") or []

    _log(f"KR changed: {len(updated_kr)}, US changed: {len(updated_us)}")

    if not updated_kr and not updated_us:
        _log("no tickers changed in this run — skipping toss export")
        return 0

    # Make sure ticker_master_{market}.json is available so _build_info
    # and build_index can enrich the output. Pull from R2 first, then
    # repair locally if the master turns out to be broken (e.g. the
    # main pipeline shipped an incomplete top-N-only master).
    _download_meta_from_r2()
    for market in ("kr", "us"):
        _repair_master_if_broken(market)

    # Re-export each changed ticker per market. We pass --skip-index so
    # we only touch the per-ticker files; the search indexes get rebuilt
    # once at the end.
    for market, tickers in (("kr", updated_kr), ("us", updated_us)):
        if not tickers:
            continue
        _log(f"re-exporting {len(tickers)} {market.upper()} tickers")
        # Batch to keep argv lengths reasonable. Linux allows millions of
        # bytes but Windows developers running this script locally have a
        # tighter limit, so keep batches small.
        for batch in _chunks(tickers, 200):
            args = ["--market", market, "--skip-index"]
            for t in batch:
                args.extend(["--ticker", t])
            rc = _run_export(args)
            if rc != 0:
                _log(f"[error] export_toss_data.py failed for {market} batch")
                return rc

    # Rebuild indexes last so the new/removed tickers in the universe
    # are reflected in `index/{market}.json`.
    for market in ("kr", "us"):
        rc = _run_export(["--market", market, "--index-only"])
        if rc != 0:
            _log(f"[error] index rebuild failed for {market}")
            return rc

    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
