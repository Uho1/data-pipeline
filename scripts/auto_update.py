"""Automated ingest + JSON export pipeline.

Detects new quarterly filings (DART for KR, Yahoo earnings calendar for US),
ingests only changed tickers, exports JSON, and syncs to R2.

Usage:
    PYTHONPATH=src python scripts/auto_update.py              # full run
    PYTHONPATH=src python scripts/auto_update.py --dry-run    # check only, no changes
    PYTHONPATH=src python scripts/auto_update.py --market kr  # KR only
    PYTHONPATH=src python scripts/auto_update.py --skip-r2    # skip R2 sync
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"
_META_DIR = _DATA_DIR / "meta"
_STATE_FILE = _META_DIR / "auto_update_state.json"


def _log(msg: str) -> None:
    """Print with timestamp and flush immediately (for CI live output)."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# DART report types we care about (quarterly/semi-annual/annual)
_KR_REPORT_KEYWORDS = ("사업보고서", "반기보고서", "분기보고서")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if _STATE_FILE.exists():
        with open(_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(state: dict) -> None:
    _META_DIR.mkdir(parents=True, exist_ok=True)
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------

def _load_universe_kr() -> set[str]:
    """Load KR ticker universe (6-digit codes)."""
    universe_path = _DATA_DIR / "universe" / "kr_2000_universe.csv"
    if not universe_path.exists():
        universe_path = _REPO_ROOT / ".codex_tmp" / "kr_2000_universe.csv"
    if universe_path.exists():
        with open(universe_path, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip() and not line.strip().startswith("ticker")}
    # Fallback: read from ticker_master parquet
    try:
        sys.path.insert(0, str(_REPO_ROOT / "src"))
        from market_data import db_reader_kr
        master = db_reader_kr.load_ticker_master_all()
        if master is not None and not master.empty:
            return set(master["ticker"].astype(str).tolist())
    except Exception:
        pass
    return set()


def _load_universe_us() -> set[str]:
    """Load US ticker universe (1786 tickers)."""
    universe_path = _DATA_DIR / "universe" / "us_1786_universe.csv"
    if universe_path.exists():
        with open(universe_path, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip() and not line.strip().startswith("ticker")}
    return set()


# ---------------------------------------------------------------------------
# KR: Check new filings via DART API (bulk, no corp_code)
# ---------------------------------------------------------------------------

def check_new_filings_kr(since_date: str) -> list[str]:
    """Query DART for all recent filings and return KR tickers with new reports.

    Args:
        since_date: YYYYMMDD format start date

    Returns:
        List of 6-digit KR ticker codes with new quarterly filings.
    """
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from market_data.kr_dart.client import DartClient

    client = DartClient()
    universe = _load_universe_kr()
    if not universe:
        _log("[kr] WARNING: empty universe, skipping KR check")
        return []

    changed_tickers: set[str] = set()
    page = 1
    total_pages = 1

    while page <= total_pages:
        try:
            result = client.list_filings(
                bgn_de=since_date,
                end_de=date.today().strftime("%Y%m%d"),
                page_no=page,
                page_count=100,
            )
        except Exception as e:
            _log(f"[kr] DART API error on page {page}: {e}")
            break

        status = str(result.get("status", ""))
        if status == "013":
            # No data
            break
        if status != "000":
            _log(f"[kr] DART API status={status} message={result.get('message')}")
            break

        items = result.get("list", [])
        for item in items:
            report_nm = str(item.get("report_nm", ""))
            stock_code = str(item.get("stock_code", "")).strip()

            # Only care about quarterly/semi-annual/annual reports
            if not any(kw in report_nm for kw in _KR_REPORT_KEYWORDS):
                continue

            # Only care about tickers in our universe
            if stock_code and stock_code in universe:
                changed_tickers.add(stock_code)

        total_count = int(result.get("total_count", 0))
        page_count = int(result.get("page_count", 100))
        total_pages = (total_count + page_count - 1) // page_count if total_count > 0 else 1
        page += 1

    _log(f"[kr] Found {len(changed_tickers)} tickers with new filings since {since_date}")
    return sorted(changed_tickers)


# ---------------------------------------------------------------------------
# US: Check new filings via Yahoo earnings calendar
# ---------------------------------------------------------------------------

def check_new_filings_us(since_date: str) -> list[str]:
    """Detect US tickers with new 10-K/10-Q filings via SEC EDGAR EFTS API.

    Args:
        since_date: YYYY-MM-DD format

    Returns:
        List of US ticker symbols with new filings, filtered by universe.
    """
    import re

    universe = _load_universe_us()
    if not universe:
        _log("[us] WARNING: empty universe, skipping US check")
        return []

    changed_tickers: set[str] = set()

    try:
        import requests

        end_date = date.today().isoformat()
        _log(f"[us] Querying SEC EFTS API for 10-K/10-Q filings {since_date} ~ {end_date}...")

        # EFTS search-index returns up to 100 results per request.
        # Paginate with 'from' parameter to get all results.
        page_size = 200
        start_from = 0
        total_hits = None

        while True:
            resp = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "forms": "10-Q,10-K",
                    "dateRange": "custom",
                    "startdt": since_date,
                    "enddt": end_date,
                    "_source": "display_names,form,file_date",
                    "size": page_size,
                    "from": start_from,
                },
                headers={"User-Agent": "StocksGram admin@stocksgram.com"},
                timeout=30,
            )
            if not resp.ok:
                _log(f"[us] EFTS API error: {resp.status_code}")
                break

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if total_hits is None:
                total_hits = data["hits"]["total"]["value"]
                _log(f"[us] EFTS total results: {total_hits}")

            if not hits:
                break

            for h in hits:
                src = h["_source"]
                form = src.get("form", "") or ""
                # Only actual 10-K/10-Q forms (not exhibits, amendments ok)
                if form not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                    continue
                for name in src.get("display_names", []):
                    # Format: "COMPANY NAME  (TICKER)  (CIK 0001234567)"
                    m = re.search(r"\(([A-Z]{1,5})\)\s+\(CIK", name)
                    if m:
                        ticker = m.group(1)
                        if ticker in universe:
                            changed_tickers.add(ticker)

            start_from += len(hits)
            if start_from >= total_hits:
                break

    except ImportError:
        _log("[us] requests not installed, skipping US check")
    except Exception as e:
        _log(f"[us] EFTS error: {e}")

    _log(f"[us] Found {len(changed_tickers)} tickers with new filings since {since_date}")
    return sorted(changed_tickers)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def _run_cli(args: list[str], label: str) -> bool:
    """Run a market_data CLI command via subprocess, streaming output."""
    cmd = [sys.executable, "-u", "-m", "market_data"] + args
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    _log(f"[{label}] Running: {' '.join(args)}")
    t0 = datetime.now()
    result = subprocess.run(
        cmd, env=env, cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    # Print subprocess output with label prefix
    out = (result.stdout or "").strip()
    if out:
        for line in out.splitlines():
            _log(f"[{label}]   {line}")
    else:
        _log(f"[{label}]   (no output, exit={result.returncode})")
    if result.returncode != 0:
        _log(f"[{label}] FAILED (exit {result.returncode}, {elapsed:.0f}s)")
        return False
    _log(f"[{label}] Done ({elapsed:.0f}s)")
    return True


def ingest_tickers_kr(tickers: list[str], since_date: str) -> bool:
    """Ingest KR tickers: filings → financials → materialize."""
    _log(f"[kr-ingest] {len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
    tickers_str = ",".join(tickers)
    year = date.today().year

    ok = True
    _log("[kr-ingest] Phase 1/3: fetching DART filings...")
    ok &= _run_cli([
        "kr-dart", "filings",
        "--tickers", tickers_str,
        "--start-date", since_date,
    ], "kr-ingest-filings")

    _log("[kr-ingest] Phase 2/3: fetching DART financials...")
    ok &= _run_cli([
        "kr-dart", "financials",
        "--tickers", tickers_str,
        "--start-year", str(year - 1),
        "--end-year", str(year),
    ], "kr-ingest-financials")

    _log("[kr-ingest] Phase 3/3: materializing to parquet...")
    ok &= _run_cli([
        "kr-dart", "materialize",
        "--tickers", tickers_str,
        "--start-year", str(year - 1),
        "--end-year", str(year),
    ], "kr-materialize")

    return ok


def ingest_tickers_us(tickers: list[str]) -> bool:
    """Ingest US tickers: SEC financials only (no prices needed)."""
    _log(f"[us-ingest] {len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
    # Write tickers to temp file (CLI only accepts --tickers-file, not inline list)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, dir=str(_REPO_ROOT)) as f:
        f.write("ticker\n")
        for t in tickers:
            f.write(f"{t}\n")
        tickers_file = f.name
    try:
        return _run_cli([
            "ingest",
            "--universe", "custom",
            "--tickers-file", tickers_file,
            "--sec-financials-only",
            "--force",
        ], "us-ingest")
    finally:
        try:
            os.unlink(tickers_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Export JSON
# ---------------------------------------------------------------------------

def export_tickers(tickers: list[str], market: str) -> bool:
    """Export JSON for specific tickers."""
    tickers_str = ",".join(tickers)
    return _run_cli([
        "export-json",
        "--market", market,
        "--tickers", tickers_str,
    ], f"{market}-export")


# ---------------------------------------------------------------------------
# R2 sync
# ---------------------------------------------------------------------------

def sync_json_to_r2(market_filter: str | None = None) -> bool:
    """Upload changed JSON files to R2."""
    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "upload_r2.py")]
    if market_filter:
        cmd += ["--market", market_filter]
    env = {**os.environ}
    _log(f"[r2] Uploading JSON to R2...")
    result = subprocess.run(cmd, env=env, cwd=str(_REPO_ROOT))
    return result.returncode == 0


def sync_parquet_to_r2(changed: dict[str, list[str]]) -> bool:
    """Backup parquet for changed tickers to R2."""
    try:
        client, bucket = _get_r2_client()
        if client is None:
            _log("[r2-parquet] No R2 credentials, skipping parquet backup")
            return False

        total_tickers = sum(len(t) for t in changed.values())
        _log(f"[r2-parquet] Uploading parquet for {total_tickers} tickers...")
        uploaded = 0

        _PARTITIONED_TABLES = ("financials_quarterly", "filings")
        _KR_FLAT_FILES = ("ingest_checkpoints.parquet", "ticker_master.parquet",
                          "dart_corp_master.parquet")
        _US_FLAT_FILES = ("ingest_checkpoints.parquet", "sec_issuer_registry.parquet")

        for market, tickers in changed.items():
            if not tickers:
                continue

            # Upload flat metadata files
            flat_files = _KR_FLAT_FILES if market == "kr" else _US_FLAT_FILES
            for flat in flat_files:
                local_path = _DATA_DIR / "parquet" / market / flat
                if local_path.is_file():
                    client.upload_file(str(local_path), bucket, f"parquet/{market}/{flat}")
                    uploaded += 1

            # KR: resolve ticker → corp_code for dart_financials_raw
            corp_code_map: dict[str, str] = {}
            if market == "kr":
                try:
                    import pandas as pd
                    master_path = _DATA_DIR / "parquet" / "kr" / "dart_corp_master.parquet"
                    if master_path.exists():
                        master = pd.read_parquet(master_path)
                        for _, row in master[master["ticker"].isin(tickers)].iterrows():
                            corp_code_map[str(row["ticker"])] = str(row["corp_code"])
                except Exception:
                    pass

            # Upload ticker-partitioned data
            parquet_base = _DATA_DIR / "parquet"
            for i, t in enumerate(tickers, 1):
                _log(f"[r2-parquet] {market.upper()}: [{i}/{len(tickers)}] {t}")
                for table in _PARTITIONED_TABLES:
                    ticker_dir = parquet_base / market / table / f"ticker={t}"
                    if not ticker_dir.exists():
                        continue
                    for pf in ticker_dir.rglob("*.parquet"):
                        rel = pf.relative_to(parquet_base)
                        r2_key = f"parquet/{rel.as_posix()}"
                        client.upload_file(str(pf), bucket, r2_key)
                        uploaded += 1
                # KR: also upload dart_financials_raw by corp_code
                if market == "kr" and t in corp_code_map:
                    cc = corp_code_map[t]
                    cc_dir = parquet_base / "kr" / "dart_financials_raw" / f"corp_code={cc}"
                    if cc_dir.exists():
                        for pf in cc_dir.rglob("*.parquet"):
                            rel = pf.relative_to(parquet_base)
                            r2_key = f"parquet/{rel.as_posix()}"
                            client.upload_file(str(pf), bucket, r2_key)
                            uploaded += 1

        _log(f"[r2-parquet] Done: {uploaded} files for {total_tickers} tickers")
        return True
    except Exception as e:
        _log(f"[r2-parquet] Error: {e}")
        return False


def _get_r2_client():
    """Create and return (client, bucket) for R2 access."""
    import boto3
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")

    account_id = os.environ.get("R2_ACCOUNT_ID", "")
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


def _download_r2_prefix(client, bucket: str, prefix: str) -> int:
    """Download all objects under a given R2 prefix. Returns count downloaded."""
    paginator = client.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            r2_key = obj["Key"]
            local_rel = r2_key.replace("parquet/", "", 1)
            local_path = _DATA_DIR / "parquet" / local_rel
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                continue

            client.download_file(bucket, r2_key, str(local_path))
            downloaded += 1
    return downloaded


def _download_r2_file(client, bucket: str, r2_key: str) -> bool:
    """Download a single file from R2. Returns True if downloaded."""
    local_rel = r2_key.replace("parquet/", "", 1)
    local_path = _DATA_DIR / "parquet" / local_rel
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, r2_key, str(local_path))
        return True
    except Exception:
        return False


def restore_parquet_for_tickers(changed: dict[str, list[str]]) -> bool:
    """Restore only the parquet partitions needed for changed tickers."""
    try:
        client, bucket = _get_r2_client()
        if client is None:
            _log("[r2-restore] No R2 credentials, skipping restore")
            return False

        total_tickers = sum(len(t) for t in changed.values())
        _log(f"[r2-restore] Restoring parquet for {total_tickers} tickers...")
        downloaded = 0

        # Ticker-partitioned tables to restore
        _PARTITIONED_TABLES = ("financials_quarterly", "filings")
        # Flat metadata files (small, always needed)
        _KR_FLAT_FILES = ("ingest_checkpoints.parquet", "ticker_master.parquet",
                          "dart_corp_master.parquet")
        _US_FLAT_FILES = ("ingest_checkpoints.parquet", "sec_issuer_registry.parquet")

        for market, tickers in changed.items():
            if not tickers:
                continue

            # Download flat metadata files
            flat_files = _KR_FLAT_FILES if market == "kr" else _US_FLAT_FILES
            _log(f"[r2-restore] {market.upper()}: downloading metadata files...")
            for flat in flat_files:
                if _download_r2_file(client, bucket, f"parquet/{market}/{flat}"):
                    downloaded += 1

            # KR: resolve ticker → corp_code for dart_financials_raw
            corp_code_map: dict[str, str] = {}
            if market == "kr":
                try:
                    import pandas as pd
                    master_path = _DATA_DIR / "parquet" / "kr" / "dart_corp_master.parquet"
                    if master_path.exists():
                        master = pd.read_parquet(master_path)
                        for _, row in master[master["ticker"].isin(tickers)].iterrows():
                            corp_code_map[str(row["ticker"])] = str(row["corp_code"])
                except Exception:
                    pass

            # Download ticker-partitioned data
            for i, t in enumerate(tickers, 1):
                _log(f"[r2-restore] {market.upper()}: [{i}/{len(tickers)}] {t}")
                for table in _PARTITIONED_TABLES:
                    prefix = f"parquet/{market}/{table}/ticker={t}/"
                    downloaded += _download_r2_prefix(client, bucket, prefix)
                # KR: also download dart_financials_raw by corp_code
                if market == "kr" and t in corp_code_map:
                    cc = corp_code_map[t]
                    prefix = f"parquet/kr/dart_financials_raw/corp_code={cc}/"
                    downloaded += _download_r2_prefix(client, bucket, prefix)

        _log(f"[r2-restore] Done: {downloaded} files for {total_tickers} tickers")
        return True
    except Exception as e:
        _log(f"[r2-restore] Error: {e}")
        return False


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> int:
    # Force unbuffered stdout so CI logs stream in real time
    os.environ["PYTHONUNBUFFERED"] = "1"

    parser = argparse.ArgumentParser(description="Auto-update: detect new filings → ingest → export → R2")
    parser.add_argument("--dry-run", action="store_true", help="Check for new filings only, don't ingest/export")
    parser.add_argument("--market", choices=["kr", "us"], help="Process only one market")
    parser.add_argument("--skip-r2", action="store_true", help="Skip R2 sync steps")
    parser.add_argument("--skip-restore", action="store_true", help="Skip parquet restore from R2 (use local data)")
    parser.add_argument("--lookback-days", type=int, default=7, help="How many days back to check for new filings (default: 7)")
    args = parser.parse_args()

    _log(f"{'=' * 60}")
    _log(f"Auto-Update Pipeline")
    _log(f"Markets: {args.market or 'both'} | Lookback: {args.lookback_days}d | Dry-run: {args.dry_run}")
    _log(f"{'=' * 60}")

    state = _load_state()
    markets = [args.market] if args.market else ["kr", "us"]

    # Step 1: Check for new filings (no R2 needed — uses DART API / SEC RSS)
    since_days = args.lookback_days
    changed: dict[str, list[str]] = {}

    if "kr" in markets:
        last_kr = state.get("last_run_kr", "")
        since_kr = last_kr if last_kr else (date.today() - timedelta(days=since_days)).strftime("%Y%m%d")
        _log(f"[step 1] Checking KR filings since {since_kr}...")
        changed["kr"] = check_new_filings_kr(since_kr)

    if "us" in markets:
        last_us = state.get("last_run_us", "")
        since_us = last_us if last_us else (date.today() - timedelta(days=since_days)).isoformat()
        _log(f"[step 1] Checking US filings since {since_us}...")
        changed["us"] = check_new_filings_us(since_us)

    # Summary
    total_changed = sum(len(v) for v in changed.values())
    _log(f"[summary] {total_changed} tickers with new filings")
    for market, tickers in changed.items():
        if tickers:
            preview = tickers[:10]
            suffix = f" ... (+{len(tickers) - 10})" if len(tickers) > 10 else ""
            _log(f"  {market.upper()}: {', '.join(preview)}{suffix}")

    if args.dry_run:
        _log("[dry-run] Exiting without changes.")
        return 0

    if total_changed == 0:
        _log("[done] No new filings detected. Nothing to do.")
        # Update state even when nothing changed. Clear the changed-ticker
        # lists so downstream hooks (e.g. auto_update_toss.py) don't
        # re-process stale values from a previous run.
        if "kr" in markets:
            state["last_run_kr"] = date.today().strftime("%Y%m%d")
            state["updated_tickers_kr"] = []
        if "us" in markets:
            state["last_run_us"] = date.today().isoformat()
            state["updated_tickers_us"] = []
        _save_state(state)
        return 0

    # Step 1.5: Restore parquet for changed tickers only (CI)
    if not args.skip_restore and not args.skip_r2:
        is_ci = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
        if is_ci:
            _log(f"[step 1.5] Restoring parquet for {total_changed} changed tickers from R2...")
            t_restore = datetime.now()
            restore_parquet_for_tickers(changed)
            _log(f"[step 1.5] Restore complete ({(datetime.now() - t_restore).total_seconds():.0f}s)")

    # Step 2: Ingest changed tickers
    _log(f"[step 2] Ingesting changed tickers...")
    t_ingest = datetime.now()
    if changed.get("kr"):
        since_kr_fmt = state.get("last_run_kr", (date.today() - timedelta(days=since_days)).strftime("%Y%m%d"))
        ingest_tickers_kr(changed["kr"], since_kr_fmt)

    if changed.get("us"):
        ingest_tickers_us(changed["us"])
    _log(f"[step 2] Ingest complete ({(datetime.now() - t_ingest).total_seconds():.0f}s)")

    # Debug: check parquet state after ingest
    for market in changed:
        pq_dir = _DATA_DIR / "parquet" / market / "financials_quarterly"
        if pq_dir.exists():
            ticker_dirs = [d for d in pq_dir.iterdir() if d.is_dir()]
            _log(f"[debug] {market}/financials_quarterly: {len(ticker_dirs)} ticker dirs")
            # Check first ticker's parquet file size
            if ticker_dirs:
                first = ticker_dirs[0]
                pfiles = list(first.glob("*.parquet"))
                if pfiles:
                    _log(f"[debug]   sample: {first.name}/{pfiles[0].name} ({pfiles[0].stat().st_size} bytes)")
                else:
                    _log(f"[debug]   sample: {first.name}/ has NO parquet files")
        else:
            _log(f"[debug] {market}/financials_quarterly: DOES NOT EXIST")
        raw_dir = _DATA_DIR / "parquet" / market / "dart_financials_raw"
        if raw_dir.exists():
            raw_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
            _log(f"[debug] {market}/dart_financials_raw: {len(raw_dirs)} corp dirs")
        # Check ticker_master
        tm = _DATA_DIR / "parquet" / market / "ticker_master.parquet"
        _log(f"[debug] {market}/ticker_master.parquet: {'exists' if tm.exists() else 'MISSING'}"
             + (f" ({tm.stat().st_size} bytes)" if tm.exists() else ""))
        # Check dart_corp_master
        dcm = _DATA_DIR / "parquet" / market / "dart_corp_master.parquet"
        if dcm.exists():
            _log(f"[debug] {market}/dart_corp_master.parquet: {dcm.stat().st_size} bytes")

    # Step 3: Export JSON
    _log(f"[step 3] Exporting JSON for changed tickers...")
    t_export = datetime.now()
    for market, tickers in changed.items():
        if tickers:
            _log(f"[step 3] {market.upper()}: exporting {len(tickers)} tickers...")
            export_tickers(tickers, market)

    # Debug: check if JSON files were created
    for market in changed:
        tickers_dir = _DATA_DIR / "tickers" / market
        if tickers_dir.exists():
            json_files = list(tickers_dir.glob("*.json"))
            _log(f"[debug] {market}/tickers: {len(json_files)} JSON files")
        else:
            _log(f"[debug] {market}/tickers: DOES NOT EXIST")

    _log(f"[step 3] Export complete ({(datetime.now() - t_export).total_seconds():.0f}s)")

    # Step 4: Sync to R2
    if not args.skip_r2:
        _log(f"[step 4] Syncing JSON to R2...")
        t_r2 = datetime.now()
        for market in changed:
            if changed[market]:
                sync_json_to_r2(market)
        _log(f"[step 4] R2 sync complete ({(datetime.now() - t_r2).total_seconds():.0f}s)")

        # Backup parquet to R2
        _log(f"[step 5] Backing up parquet to R2...")
        t_backup = datetime.now()
        sync_parquet_to_r2(changed)
        _log(f"[step 5] Backup complete ({(datetime.now() - t_backup).total_seconds():.0f}s)")

    # Update state
    if "kr" in markets:
        state["last_run_kr"] = date.today().strftime("%Y%m%d")
        state["updated_tickers_kr"] = changed.get("kr", [])
    if "us" in markets:
        state["last_run_us"] = date.today().isoformat()
        state["updated_tickers_us"] = changed.get("us", [])
    state["last_run_at"] = datetime.now().isoformat(timespec="seconds")
    _save_state(state)

    total_elapsed = (datetime.now() - t_ingest).total_seconds()
    _log(f"{'=' * 60}")
    _log(f"Auto-Update Complete - {total_changed} tickers updated ({total_elapsed:.0f}s total)")
    _log(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
