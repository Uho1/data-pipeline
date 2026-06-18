"""DuckDB connection manager for Korea prices data (market_data_kr_prices.duckdb)."""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
from pathlib import Path

from market_data.config import DATA_DIR, PARQUET_DIR, STORAGE_BACKEND

DB_PATH: Path = DATA_DIR / "market_data_kr_prices.duckdb"

_local = threading.local()
_snapshot_lock = threading.Lock()
_snapshot_sig: tuple[int, int] | None = None
_SNAPSHOT_RE = re.compile(r"^market_data_kr_prices_readonly_(\d+)\.duckdb(?:\..+)?$")


def _db_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _snapshot_path() -> Path:
    return Path(tempfile.gettempdir()) / f"market_data_kr_prices_readonly_{os.getpid()}.duckdb"


def _snapshot_pid(path: Path) -> int | None:
    match = _SNAPSHOT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale_snapshot_copies() -> int:
    removed = 0
    temp_dir = Path(tempfile.gettempdir())
    for candidate in temp_dir.glob("market_data_kr_prices_readonly_*.duckdb*"):
        pid = _snapshot_pid(candidate)
        if pid is None or _is_pid_alive(pid):
            continue
        try:
            candidate.unlink()
            removed += 1
        except (FileNotFoundError, IsADirectoryError):
            continue
    return removed


def _ensure_snapshot_copy(path: Path, sig: tuple[int, int]) -> Path:
    global _snapshot_sig

    snapshot = _snapshot_path()
    with _snapshot_lock:
        cleanup_stale_snapshot_copies()
        if snapshot.exists() and _snapshot_sig == sig:
            return snapshot

        tmp = snapshot.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(path, tmp)
        tmp.replace(snapshot)
        _snapshot_sig = sig
        return snapshot


def db_available() -> bool:
    try:
        import duckdb  # noqa: F401
    except ImportError:
        return False
    if STORAGE_BACKEND == "parquet":
        return (PARQUET_DIR / "kr").exists()
    return DB_PATH.exists()


def get_connection():
    import duckdb

    if STORAGE_BACKEND == "parquet":
        return _get_parquet_connection()

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"KR Prices DuckDB not found at {DB_PATH}. "
            "Run KR ingest first or run the migration script."
        )

    conn = getattr(_local, "con", None)
    mode = getattr(_local, "con_mode", None)
    sig = _db_signature(DB_PATH)

    if conn is not None:
        if mode == "direct":
            return conn
        if mode == "snapshot" and getattr(_local, "con_sig", None) == sig:
            return conn
        close_connection()

    try:
        _local.con = duckdb.connect(str(DB_PATH))
        _local.con_mode = "direct"
        _local.con_sig = sig
        return _local.con
    except duckdb.IOException as exc:
        message = str(exc)
        if "Could not set lock on file" not in message:
            raise
        snapshot = _ensure_snapshot_copy(DB_PATH, sig)
        _local.con = duckdb.connect(str(snapshot), read_only=True)
        _local.con_mode = "snapshot"
        _local.con_sig = sig
        return _local.con


def _get_parquet_connection():
    import duckdb
    from market_data.parquet_views import register_parquet_views

    conn = getattr(_local, "con", None)
    mode = getattr(_local, "con_mode", None)
    if conn is not None and mode == "parquet":
        return conn

    close_connection()
    conn = duckdb.connect()
    register_parquet_views(conn, market="kr", db_type="prices")
    _local.con = conn
    _local.con_mode = "parquet"
    _local.con_sig = None
    return conn


def close_connection() -> None:
    conn = getattr(_local, "con", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.con = None
        _local.con_mode = None
        _local.con_sig = None
