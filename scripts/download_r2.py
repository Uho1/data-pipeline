"""Mirror JSON data from Cloudflare R2 down to the local `data/` tree.

The inverse of `scripts/upload_r2.py`. Used when the local copy has
drifted out of date (e.g. a laptop that's been offline for a few days
while GitHub Actions kept ingesting new filings) and you want to pull
the authoritative version back.

Downloads are parallel (default 16 workers) and incremental — any file
that already matches the remote ETag is left untouched.

Usage::

    python scripts/download_r2.py                     # default: tickers + meta
    python scripts/download_r2.py --prefix tickers/kr # only one subtree
    python scripts/download_r2.py --all               # force re-download
    python scripts/download_r2.py --dry-run           # preview only
    python scripts/download_r2.py --workers 32

Required env (same as upload_r2.py):
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET_NAME
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

# Local mirror root — matches `upload_r2.py` UPLOAD_TARGETS so keys
# round-trip: upload writes `tickers/kr/foo.json` from `data/tickers/kr/foo.json`,
# and download writes the same key back to the same local path.
_DATA_DIR = _REPO_ROOT / "data"

# Default set of prefixes to mirror
_DEFAULT_PREFIXES: list[str] = ["tickers/kr", "tickers/us", "meta"]


def _get_r2_client(max_pool: int):
    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(max_pool_connections=max(max_pool * 2, 20)),
    )


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_prefix(client, bucket: str, prefix: str) -> list[dict]:
    """Return every {Key, ETag, Size} object under the given prefix."""
    out: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append(
                {
                    "Key": obj["Key"],
                    "ETag": obj["ETag"].strip('"'),
                    "Size": obj["Size"],
                }
            )
    return out


def _local_path(key: str) -> Path:
    """Map an R2 key back to its local filesystem path."""
    return _DATA_DIR / key


_counter_lock = threading.Lock()


class _Progress:
    downloaded = 0
    skipped = 0
    failed = 0
    bytes_downloaded = 0


def _download_one(
    client,
    bucket: str,
    obj: dict,
    dry_run: bool,
) -> tuple[str, int, Exception | None]:
    key = obj["Key"]
    size = int(obj["Size"])
    local = _local_path(key)

    if dry_run:
        return key, size, None

    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(local))
        return key, size, None
    except ClientError as exc:
        return key, size, exc
    except Exception as exc:  # noqa: BLE001
        return key, size, exc


def run(
    prefixes: list[str],
    force: bool,
    dry_run: bool,
    workers: int,
) -> int:
    bucket = os.environ["R2_BUCKET_NAME"]
    client = _get_r2_client(workers)

    # Aggregate the work list across every requested prefix
    print(f"[list] fetching object metadata from r2://{bucket}")
    total_objects = 0
    all_objects: list[dict] = []
    for prefix in prefixes:
        # 단일 파일(.parquet/.json)은 "/"를 붙이면 디렉토리로 스캔돼 0건이 됨
        list_prefix = prefix if prefix.endswith((".parquet", ".json")) else prefix + "/"
        print(f"[list] scanning prefix: {list_prefix}")
        objs = _list_prefix(client, bucket, list_prefix)
        print(f"[list]   {len(objs)} objects")
        total_objects += len(objs)
        all_objects.extend(objs)

    if not all_objects:
        print("[done] nothing to download — prefixes are empty")
        return 0

    # Decide which files we actually need to fetch
    prog = _Progress()
    todo: list[dict] = []
    for obj in all_objects:
        local = _local_path(obj["Key"])
        if not force and local.exists():
            # Fast path: size match then MD5 match
            if local.stat().st_size == obj["Size"]:
                if _md5(local) == obj["ETag"]:
                    prog.skipped += 1
                    continue
        todo.append(obj)

    if not todo:
        print(
            f"[done] all {total_objects} files already up to date "
            f"({prog.skipped} skipped)"
        )
        return 0

    total_todo = len(todo)
    print(f"[download] {total_todo} files to download ({prog.skipped} skipped)")
    if dry_run:
        print("[dry-run] would download:")

    start = time.time()
    reported = 0
    report_every = max(total_todo // 20, 50)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, client, bucket, obj, dry_run): obj["Key"]
            for obj in todo
        }
        for future in as_completed(futures):
            key, size, err = future.result()
            with _counter_lock:
                if err is not None:
                    prog.failed += 1
                    print(f"[error] {key}: {err}", file=sys.stderr)
                else:
                    prog.downloaded += 1
                    prog.bytes_downloaded += size
                done = prog.downloaded + prog.failed
                if done - reported >= report_every:
                    reported = done
                    elapsed = time.time() - start
                    mb = prog.bytes_downloaded / (1024 * 1024)
                    rate = mb / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[progress] {done}/{total_todo} "
                        f"({prog.downloaded} ok, {prog.failed} fail) "
                        f"{mb:.1f} MB · {rate:.1f} MB/s · {elapsed:.0f}s"
                    )

    elapsed = time.time() - start
    mb = prog.bytes_downloaded / (1024 * 1024)
    action = "would download" if dry_run else "downloaded"
    print()
    print(
        f"[done] {action} {prog.downloaded} files "
        f"({mb:.1f} MB) in {elapsed:.1f}s"
    )
    print(
        f"[done] skipped {prog.skipped} already-current, "
        f"failed {prog.failed}"
    )
    return 0 if prog.failed == 0 else 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download JSON data from Cloudflare R2 to local data/.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        help="R2 prefix to download (repeatable). Default: tickers/kr, tickers/us, meta",
    )
    parser.add_argument(
        "--all",
        dest="force",
        action="store_true",
        help="Force re-download every file (ignore ETag match)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without touching the disk",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel download workers (default 16)",
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
        sys.exit(1)

    prefixes = args.prefix if args.prefix else _DEFAULT_PREFIXES
    code = run(
        prefixes=prefixes,
        force=args.force,
        dry_run=args.dry_run,
        workers=args.workers,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
