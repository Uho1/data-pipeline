"""Upload pre-computed toss-app JSON data to Cloudflare R2.

Uses the same R2 credentials as `upload_r2.py` (StocksGram) but writes to
a separate `toss/` prefix inside the bucket so the two apps stay isolated.

Final object keys look like::

    toss/index/kr.json
    toss/index/us.json
    toss/tickers/kr/005930/main.json
    toss/tickers/kr/005930/detail/income.json
    toss/tickers/us/AAPL/main.json
    ...

Uploads happen in parallel (default 16 workers) because the dataset is
~17,000 files and sequential uploads would take hours. Unchanged files
are skipped via MD5/ETag comparison, so reruns after a partial upload
are fast.

Usage::

    python scripts/upload_toss_r2.py                # incremental
    python scripts/upload_toss_r2.py --all          # force re-upload
    python scripts/upload_toss_r2.py --dry-run      # preview only
    python scripts/upload_toss_r2.py --workers 32   # faster (more bandwidth)

Required environment variables (shared with upload_r2.py):
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET_NAME
    R2_PUBLIC_URL   (optional — used for the summary line)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load .env from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

# Local source — the folder the export script writes to.
_TOSS_DATA_DIR = _REPO_ROOT / "toss-app" / "public" / "toss-data"

# Remote prefix — lives inside the same bucket as the StocksGram data
# but under an isolated top-level folder.
_REMOTE_PREFIX = "toss"


def _get_r2_client(max_pool: int):
    """Build a boto3 S3 client tuned for high parallelism against R2."""
    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        # Default pool is 10 — bump so 16+ workers don't starve.
        config=Config(max_pool_connections=max(max_pool * 2, 20)),
    )


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_remote_etags(client, bucket: str, prefix: str) -> dict[str, str]:
    """Return {key: etag} for every object under the given prefix.

    R2's S3 API supports paginated list_objects_v2 just like AWS. For
    ~17k objects this takes a few seconds once and then we skip every
    unchanged file on the local side.
    """
    etags: dict[str, str] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            etags[obj["Key"]] = obj["ETag"].strip('"')
    return etags


def _iter_local_files(root: Path) -> list[Path]:
    """Walk the toss-data tree and return every file we want to upload."""
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith("_"):
            files.append(path)
    return files


def _key_for(local: Path, root: Path) -> str:
    """Map a local file path to its R2 object key under the toss prefix."""
    rel = local.relative_to(root).as_posix()
    return f"{_REMOTE_PREFIX}/{rel}"


# Counters shared across worker threads
_counter_lock = threading.Lock()


class _Progress:
    uploaded = 0
    skipped = 0
    failed = 0
    bytes_uploaded = 0


def _upload_one(
    client, bucket: str, path: Path, key: str, dry_run: bool
) -> tuple[str, int, Exception | None]:
    """Upload a single file and return (key, size, error_or_None)."""
    size = path.stat().st_size
    if dry_run:
        return key, size, None
    try:
        client.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={
                "ContentType": "application/json",
                # Short cache — we replace files nightly via reruns.
                "CacheControl": "public, max-age=3600",
            },
        )
        return key, size, None
    except ClientError as exc:
        return key, size, exc
    except Exception as exc:  # noqa: BLE001
        return key, size, exc


def run(force: bool, dry_run: bool, workers: int) -> int:
    bucket = os.environ["R2_BUCKET_NAME"]
    public_url = os.environ.get("R2_PUBLIC_URL", "").strip().rstrip("/")

    if not _TOSS_DATA_DIR.exists():
        print(f"[error] {_TOSS_DATA_DIR} does not exist.", file=sys.stderr)
        print("Run `python scripts/export_toss_data.py` first.", file=sys.stderr)
        return 1

    client = _get_r2_client(workers)

    print(f"[scan] {_TOSS_DATA_DIR}")
    files = _iter_local_files(_TOSS_DATA_DIR)
    print(f"[scan] {len(files)} local files found")

    # Fetch all existing ETags in one paginated listing so each worker
    # can decide to skip without a HEAD round-trip.
    remote: dict[str, str] = {}
    if not force:
        print(f"[remote] listing existing objects under r2://{bucket}/{_REMOTE_PREFIX}/")
        try:
            remote = _list_remote_etags(client, bucket, _REMOTE_PREFIX)
        except ClientError as exc:
            print(f"[warn] could not list remote: {exc}", file=sys.stderr)
        print(f"[remote] {len(remote)} existing objects")

    # Build the work list, skipping unchanged files up front
    todo: list[tuple[Path, str]] = []
    prog = _Progress()
    for path in files:
        key = _key_for(path, _TOSS_DATA_DIR)
        if not force and remote.get(key) == _md5(path):
            prog.skipped += 1
            continue
        todo.append((path, key))

    if not todo:
        print("[done] nothing to upload — all files up to date.")
        return 0

    total_todo = len(todo)
    print(f"[upload] {total_todo} files to upload ({prog.skipped} skipped)")
    if dry_run:
        print(f"[dry-run] would upload to r2://{bucket}/{_REMOTE_PREFIX}/")

    start = time.time()
    reported = 0
    report_every = max(total_todo // 20, 50)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_upload_one, client, bucket, path, key, dry_run): key
            for path, key in todo
        }
        for future in as_completed(futures):
            key, size, err = future.result()
            with _counter_lock:
                if err is not None:
                    prog.failed += 1
                    print(f"[error] {key}: {err}", file=sys.stderr)
                else:
                    prog.uploaded += 1
                    prog.bytes_uploaded += size
                done = prog.uploaded + prog.failed
                if done - reported >= report_every:
                    reported = done
                    elapsed = time.time() - start
                    mb = prog.bytes_uploaded / (1024 * 1024)
                    rate = mb / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[progress] {done}/{total_todo} "
                        f"({prog.uploaded} ok, {prog.failed} fail) "
                        f"{mb:.1f} MB · {rate:.1f} MB/s · {elapsed:.0f}s"
                    )

    elapsed = time.time() - start
    mb = prog.bytes_uploaded / (1024 * 1024)
    action = "would upload" if dry_run else "uploaded"
    print()
    print(f"[done] {action} {prog.uploaded} files "
          f"({mb:.1f} MB) in {elapsed:.1f}s")
    print(f"[done] skipped {prog.skipped} unchanged, failed {prog.failed}")

    if public_url and not dry_run:
        sample_key = f"{_REMOTE_PREFIX}/index/kr.json"
        print(f"[hint] sample URL: {public_url}/{sample_key}")
        print("[hint] set this as the frontend base URL:")
        print(f"       NEXT_PUBLIC_DATA_BASE={public_url}/{_REMOTE_PREFIX}")

    return 0 if prog.failed == 0 else 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload toss-app static JSON data to Cloudflare R2 "
        "under the `toss/` prefix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="force",
        help="Force re-upload every file (ignore ETag match)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without touching R2",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel upload workers (default 16)",
    )
    args = parser.parse_args()

    required = [
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(
            f"[error] Missing environment variables: {', '.join(missing)}",
            file=sys.stderr,
        )
        print("Set them in .env or your shell.", file=sys.stderr)
        sys.exit(1)

    code = run(force=args.force, dry_run=args.dry_run, workers=args.workers)
    sys.exit(code)


if __name__ == "__main__":
    main()
