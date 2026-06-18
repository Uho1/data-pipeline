from __future__ import annotations

from typing import Any

import pandas as pd

from market_data.sp500_pit import get_sp500_constituents_asof, load_sp500_constituents_pit


def detect_sp500_pit_membership_inconsistency(
    trades_df: pd.DataFrame,
    *,
    pit_df: pd.DataFrame | None = None,
    pit_path: str | None = None,
    mode_label: str | None = None,
    min_confidence: float = 0.0,
    low_confidence_threshold: float | None = None,
    low_confidence_result: str = "warn",
) -> dict[str, Any]:
    cols = [
        "mode_label",
        "ticker",
        "trade_date",
        "check_result",
        "check_type",
        "note",
    ]
    if trades_df is None or trades_df.empty:
        return {
            "status": "pass",
            "issues": pd.DataFrame(columns=cols),
            "summary": {"total_issues": 0, "warn_count": 0, "fail_count": 0},
        }

    out = trades_df.copy()
    if "ticker" not in out.columns:
        return {
            "status": "warn",
            "issues": pd.DataFrame(columns=cols),
            "summary": {"total_issues": 0, "warn_count": 1, "fail_count": 0, "note": "missing ticker column"},
        }
    date_col = "signal_date" if "signal_date" in out.columns else ("exec_date" if "exec_date" in out.columns else None)
    if date_col is None:
        return {
            "status": "warn",
            "issues": pd.DataFrame(columns=cols),
            "summary": {"total_issues": 0, "warn_count": 1, "fail_count": 0, "note": "missing trade date column"},
        }

    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["trade_date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=["ticker", "trade_date"]).copy()
    if out.empty:
        return {
            "status": "pass",
            "issues": pd.DataFrame(columns=cols),
            "summary": {"total_issues": 0, "warn_count": 0, "fail_count": 0},
        }

    pit = pit_df if isinstance(pit_df, pd.DataFrame) else load_sp500_constituents_pit(pit_path=pit_path)
    if pit is None or pit.empty:
        issues_df = pd.DataFrame(
            [
                {
                    "mode_label": mode_label or "",
                    "ticker": "",
                    "trade_date": pd.NaT,
                    "check_result": "warn",
                    "check_type": "missing_sp500_pit",
                    "note": "sp500 PIT intervals unavailable",
                }
            ],
            columns=cols,
        )
        return {
            "status": "warn",
            "issues": issues_df,
            "summary": {"total_issues": 1, "warn_count": 1, "fail_count": 0},
        }

    low_conf_res = str(low_confidence_result or "warn").strip().lower()
    if low_conf_res not in {"warn", "fail"}:
        low_conf_res = "warn"
    low_threshold = None
    if low_confidence_threshold is not None:
        try:
            low_threshold = float(low_confidence_threshold)
        except Exception:
            low_threshold = None
        if low_threshold is not None and low_threshold < 0:
            low_threshold = 0.0

    date_to_members: dict[pd.Timestamp, dict[str, float]] = {}
    for d in sorted(out["trade_date"].unique()):
        snap = get_sp500_constituents_asof(
            pd.Timestamp(d),
            pit_df=pit,
            min_confidence=0.0,
            strict=False,
        )
        members_map: dict[str, float] = {}
        if isinstance(snap, pd.DataFrame) and not snap.empty:
            tickers = snap.get("ticker", pd.Series(dtype=str)).astype(str).str.upper()
            conf = pd.to_numeric(snap.get("confidence", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            for sym, c in zip(tickers.tolist(), conf.tolist()):
                if not sym:
                    continue
                members_map[str(sym)] = float(c)
        if min_confidence > 0.0:
            members_map = {k: v for k, v in members_map.items() if float(v) >= float(min_confidence)}
        date_to_members[pd.Timestamp(d)] = members_map

    issues: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        td = pd.Timestamp(row["trade_date"]).normalize()
        sym = str(row["ticker"]).upper()
        members = date_to_members.get(td, {})
        if sym not in members:
            issues.append(
                {
                    "mode_label": mode_label or "",
                    "ticker": sym,
                    "trade_date": td,
                    "check_result": "fail",
                    "check_type": "not_in_sp500_asof",
                    "note": f"ticker not found in SP500 PIT snapshot on {td.date().isoformat()}",
                }
            )
            continue
        if low_threshold is not None and float(members.get(sym, 0.0)) < float(low_threshold):
            conf_val = float(members.get(sym, 0.0))
            issues.append(
                {
                    "mode_label": mode_label or "",
                    "ticker": sym,
                    "trade_date": td,
                    "check_result": low_conf_res,
                    "check_type": "low_confidence_membership",
                    "note": (
                        f"membership confidence {conf_val:.4f} below threshold "
                        f"{float(low_threshold):.4f} on {td.date().isoformat()}"
                    ),
                }
            )

    issues_df = pd.DataFrame(issues, columns=cols)
    fail_count = int((issues_df.get("check_result", pd.Series(dtype=str)) == "fail").sum()) if not issues_df.empty else 0
    warn_count = int((issues_df.get("check_result", pd.Series(dtype=str)) == "warn").sum()) if not issues_df.empty else 0
    status = "fail" if fail_count > 0 else ("warn" if warn_count > 0 else "pass")
    return {
        "status": status,
        "issues": issues_df,
        "summary": {
            "total_issues": int(len(issues_df)),
            "warn_count": warn_count,
            "fail_count": fail_count,
        },
    }


def summarize_sp500_pit_issues(
    detected: dict[str, Any],
    *,
    check_name: str = "sp500_pit_membership_consistency",
    max_examples: int = 5,
) -> dict[str, Any]:
    issues_df = detected.get("issues")
    if not isinstance(issues_df, pd.DataFrame):
        issues_df = pd.DataFrame()
    status = str(detected.get("status", "pass")).lower()
    if status not in {"pass", "warn", "fail"}:
        status = "warn"
    examples: list[dict[str, Any]] = []
    if not issues_df.empty:
        for _, row in issues_df.head(max_examples).iterrows():
            examples.append(
                {
                    "mode_label": row.get("mode_label", ""),
                    "ticker": row.get("ticker", ""),
                    "trade_date": (
                        pd.Timestamp(row.get("trade_date")).date().isoformat()
                        if pd.notna(pd.to_datetime(row.get("trade_date"), errors="coerce"))
                        else ""
                    ),
                    "check_type": row.get("check_type", ""),
                    "note": row.get("note", ""),
                }
            )
    summary = detected.get("summary", {}) if isinstance(detected.get("summary"), dict) else {}
    return {
        "name": check_name,
        "status": status,
        "details": (
            f"issues={summary.get('total_issues', len(issues_df))}, "
            f"warn={summary.get('warn_count', 0)}, fail={summary.get('fail_count', 0)}"
        ),
        "examples": examples,
        "summary": summary,
    }


__all__ = [
    "detect_sp500_pit_membership_inconsistency",
    "summarize_sp500_pit_issues",
]
