"""Upload local JSON data files to Cloudflare R2.

Usage:
    python scripts/upload_r2.py               # upload all changed files
    python scripts/upload_r2.py --all         # force upload all files
    python scripts/upload_r2.py --market kr   # upload only kr tickers
    python scripts/upload_r2.py --dry-run     # show what would be uploaded

Required environment variables (in .env or shell):
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET_NAME
    R2_PUBLIC_URL   (e.g. https://pub-xxxx.r2.dev)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load .env from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

_DATA_DIR = _REPO_ROOT / "data"

UPLOAD_TARGETS: list[tuple[Path, str]] = [
    (_DATA_DIR / "tickers" / "kr", "tickers/kr"),
    (_DATA_DIR / "tickers" / "us", "tickers/us"),
    (_DATA_DIR / "meta", "meta"),
]


def _get_r2_client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_remote_etags(client, bucket: str, prefix: str) -> dict[str, str]:
    """Return {key: etag} for all objects under prefix."""
    etags: dict[str, str] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            # R2 ETag is MD5 hex (no quotes for simple uploads)
            etags[obj["Key"]] = obj["ETag"].strip('"')
    return etags


def upload(force: bool = False, market_filter: str | None = None, dry_run: bool = False) -> None:
    bucket = os.environ["R2_BUCKET_NAME"]
    client = _get_r2_client()

    total_uploaded = 0
    total_skipped = 0
    total_size = 0

    for local_dir, r2_prefix in UPLOAD_TARGETS:
        if not local_dir.exists():
            print(f"[skip] {local_dir} does not exist")
            continue

        # Apply market filter
        if market_filter:
            if "kr" in r2_prefix and market_filter != "kr":
                continue
            if "us" in r2_prefix and market_filter != "us":
                continue

        print(f"\n[scan] {local_dir} → r2://{bucket}/{r2_prefix}/")

        # Fetch existing ETags from R2 for incremental uploads
        remote_etags: dict[str, str] = {}
        if not force:
            try:
                remote_etags = _get_remote_etags(client, bucket, r2_prefix)
            except ClientError as e:
                print(f"  [warn] could not list remote: {e}")

        files = sorted(local_dir.glob("*.json"))
        print(f"  {len(files)} local files found")

        for path in files:
            r2_key = f"{r2_prefix}/{path.name}"
            local_md5 = _md5(path)
            remote_etag = remote_etags.get(r2_key, "")

            if not force and local_md5 == remote_etag:
                total_skipped += 1
                continue

            size = path.stat().st_size
            total_size += size

            if dry_run:
                print(f"  [dry-run] would upload {r2_key} ({size / 1024:.1f} KB)")
                total_uploaded += 1
                continue

            try:
                client.upload_file(
                    str(path),
                    bucket,
                    r2_key,
                    ExtraArgs={
                        "ContentType": "application/json",
                        "CacheControl": "public, max-age=3600",
                    },
                )
                total_uploaded += 1
                print(f"  [upload] {r2_key} ({size / 1024:.1f} KB)")
            except ClientError as e:
                print(f"  [error] {r2_key}: {e}")
                sys.exit(1)

    action = "would upload" if dry_run else "uploaded"
    print(
        f"\nDone: {action} {total_uploaded} files "
        f"({total_size / 1024 / 1024:.1f} MB), "
        f"skipped {total_skipped} unchanged."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload JSON data to Cloudflare R2")
    parser.add_argument("--all", action="store_true", dest="force", help="Force upload all files")
    parser.add_argument("--market", choices=["kr", "us"], help="Upload only one market")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    required = ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"[error] Missing environment variables: {', '.join(missing)}")
        print("Set them in .env or as shell env vars.")
        sys.exit(1)

    upload(force=args.force, market_filter=args.market, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
