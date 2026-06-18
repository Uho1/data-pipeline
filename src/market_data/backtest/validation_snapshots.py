from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_REQUIRED_SNAPSHOT_COLUMNS = [
    "ticker",
    "in_universe",
    "filter_pass",
    "selected",
    "rank",
    "rank_score",
]


def _resolve_snapshot_path(raw_path: str, base_dir: Path | None) -> Path:
    p = Path(str(raw_path)).expanduser()
    if p.is_absolute():
        return p
    if base_dir is not None:
        return (base_dir / p).resolve()
    return p.resolve()


def validate_rebalance_snapshots(
    index_path: str | Path | pd.DataFrame | None,
    *,
    rebalance_log_df: pd.DataFrame | None = None,
    required_columns: list[str] | None = None,
    base_dir: str | Path | None = None,
    validation_mode: str = "warn",
) -> dict[str, Any]:
    mode = str(validation_mode or "warn").strip().lower()
    if mode not in {"off", "warn", "fail"}:
        mode = "warn"
    if mode == "off":
        return {
            "status": "pass",
            "summary": {"note": "snapshot validation disabled"},
            "missing_files": [],
            "bad_rows": [],
            "counts": {
                "rebalance_rows": 0,
                "snapshot_index_rows": 0,
                "existing_snapshot_files": 0,
                "missing_snapshot_files": 0,
            },
        }

    req_cols = list(required_columns or DEFAULT_REQUIRED_SNAPSHOT_COLUMNS)
    base = Path(base_dir).expanduser().resolve() if base_dir is not None else None
    errors: list[str] = []
    warnings: list[str] = []
    missing_files: list[str] = []
    bad_rows: list[dict[str, Any]] = []

    idx_exists = True
    if isinstance(index_path, pd.DataFrame):
        idx_df = index_path.copy()
    elif index_path is None:
        idx_exists = False
        idx_df = pd.DataFrame()
    else:
        p = Path(index_path).expanduser()
        idx_exists = p.exists()
        if idx_exists:
            try:
                idx_df = pd.read_csv(p)
            except Exception:
                idx_df = pd.DataFrame()
                errors.append("snapshot index file is unreadable")
        else:
            idx_df = pd.DataFrame()

    rebalance_rows = int(len(rebalance_log_df)) if rebalance_log_df is not None else 0
    snapshot_index_rows = int(len(idx_df))
    existing_snapshot_files = 0

    if not idx_exists:
        errors.append("rebalance_snapshots_index.csv missing")
    if idx_df.empty:
        errors.append("rebalance snapshots index is empty")
    if not idx_df.empty and "snapshot_path" not in idx_df.columns:
        errors.append("snapshot_path column missing in index")

    if not idx_df.empty and "snapshot_path" in idx_df.columns:
        for i, row in idx_df.iterrows():
            raw = str(row.get("snapshot_path", "") or "").strip()
            if not raw:
                bad_rows.append({"row": int(i), "issue": "empty_snapshot_path"})
                continue
            path = _resolve_snapshot_path(raw, base)
            if not path.exists():
                missing_files.append(str(path))
                bad_rows.append({"row": int(i), "issue": "missing_snapshot_file", "snapshot_path": str(path)})
                continue
            existing_snapshot_files += 1
            try:
                snap_df = pd.read_csv(path)
            except Exception:
                warnings.append(f"snapshot unreadable: {path}")
                bad_rows.append({"row": int(i), "issue": "unreadable_snapshot", "snapshot_path": str(path)})
                continue
            missing_cols = [c for c in req_cols if c not in snap_df.columns]
            if missing_cols:
                warnings.append(f"snapshot missing required columns: {path}")
                bad_rows.append(
                    {
                        "row": int(i),
                        "issue": "missing_snapshot_columns",
                        "snapshot_path": str(path),
                        "missing_columns": missing_cols,
                    }
                )

    if rebalance_rows > 0 and snapshot_index_rows > 0 and rebalance_rows != snapshot_index_rows:
        warnings.append(
            f"rebalance rows({rebalance_rows}) != snapshot index rows({snapshot_index_rows})"
        )
    if snapshot_index_rows > 0 and existing_snapshot_files < snapshot_index_rows and existing_snapshot_files > 0:
        warnings.append(
            f"some snapshot files missing ({snapshot_index_rows - existing_snapshot_files}/{snapshot_index_rows})"
        )
    if snapshot_index_rows > 0 and existing_snapshot_files == 0:
        errors.append("all snapshot files missing")

    status = "pass"
    if errors:
        status = "fail"
    elif warnings:
        status = "warn"

    if mode == "fail" and status == "warn":
        status = "fail"
        warnings.append("strict snapshot validation mode escalated warn->fail")

    return {
        "status": status,
        "summary": {
            "errors": errors,
            "warnings": warnings,
        },
        "missing_files": missing_files,
        "bad_rows": bad_rows[:50],
        "counts": {
            "rebalance_rows": rebalance_rows,
            "snapshot_index_rows": snapshot_index_rows,
            "existing_snapshot_files": existing_snapshot_files,
            "missing_snapshot_files": int(max(snapshot_index_rows - existing_snapshot_files, 0)),
        },
    }
