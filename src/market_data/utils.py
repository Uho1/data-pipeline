from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import pandas as pd

T = TypeVar("T")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_ticker(ticker: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {".", "_", "-"} else "_" for ch in ticker)


def is_file_fresh(path: Path, fresh_days: int | None) -> bool:
    if not path.exists():
        return False
    if fresh_days is None:
        return True
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds <= fresh_days * 86400


def retry_call(
    func: Callable[[], T],
    retries: int = 3,
    backoff_base: float = 1.0,
    max_backoff: float = 30.0,
    retriable_exceptions: tuple[type[Exception], ...] = (Exception,),
    non_retriable_exceptions: tuple[type[Exception], ...] = (),
    label: str = "operation",
) -> T:
    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except non_retriable_exceptions:
            raise
        except retriable_exceptions as exc:
            if attempt > retries:
                raise RuntimeError(f"{label} failed after {retries} retries") from exc
            sleep_for = min(max_backoff, backoff_base * (2 ** (attempt - 1)))
            print(f"[RETRY] {label} attempt={attempt}/{retries} wait={sleep_for:.1f}s reason={exc}")
            time.sleep(sleep_for)


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: Iterable[str]) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def coerce_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()]
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def coerce_series_naive(s: Any) -> pd.Series:
    if s is None:
        return pd.Series(dtype="datetime64[ns]")
    
    # If it's already a Series, keep it, otherwise wrap it
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
        
    out = pd.to_datetime(s, errors="coerce")
    
    # Try to access .dt safely
    try:
        if hasattr(out, "dt") and out.dt is not None:
            if hasattr(out.dt, "tz") and out.dt.tz is not None:
                return out.dt.tz_localize(None)
    except (AttributeError, ValueError):
        pass
        
    return out
