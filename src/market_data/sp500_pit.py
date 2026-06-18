from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_data.config import DATA_DIR
from market_data.providers import (
    GitHubSecondaryConfig,
    GitHubSecondarySP500Provider,
    SPDJIAnnouncementsProvider,
    WRDSSP500Provider,
    SnapshotSeedSP500Provider,
)
from market_data.providers.sp500_constituents_provider import normalize_sp500_event_frame
from market_data.utils import ensure_dir, now_utc_iso

LOGGER = logging.getLogger(__name__)

SP500_PIT_DIR = DATA_DIR / "index_identity_cache" / "sp500_pit"
SP500_EVENTS_RAW_DIR = SP500_PIT_DIR / "events_raw"
SP500_EVENTS_PATH = SP500_PIT_DIR / "sp500_pit_events.csv"
SP500_PIT_INTERVALS_PATH = SP500_PIT_DIR / "sp500_constituents_pit_intervals.parquet"
SP500_PIT_TABLE_PATH = SP500_PIT_DIR / "sp500_constituents_pit.parquet"
SP500_PIT_ISSUES_PATH = SP500_PIT_DIR / "sp500_pit_issues.csv"
SP500_PIT_COVERAGE_REPORT_PATH = SP500_PIT_DIR / "sp500_pit_coverage_report.json"
SP500_PIT_DAILY_COUNTS_PATH = SP500_PIT_DIR / "sp500_pit_daily_counts.csv"
SP500_PIT_SUMMARY_PATH = SP500_PIT_DIR / "sp500_pit_summary.md"

SP500_MANUAL_EVENTS_PATH = Path("config") / "sp500_manual_events.csv"
SP500_MANUAL_TEMPLATE_DEFAULT_PATH = DATA_DIR / "reference" / "sp500_manual_events.template.csv"
SP500_MANUAL_OVERRIDE_PATH = Path("config") / "sp500_constituent_overrides.csv"

SP500_NORMALIZED_EVENT_COLUMNS = [
    "event_id",
    "index_code",
    "effective_date",
    "announcement_date",
    "action",
    "ticker",
    "ticker_new",
    "company_name",
    "company_name_new",
    "reason",
    "identifier_type",
    "identifier_value",
    "source_name",
    "source_type",  # official | licensed | secondary | manual
    "source_tier",  # official | licensed | secondary | manual_override
    "source_ref",
    "source_url",
    "source_doc_id",
    "source_row_hash",
    "provenance_text",
    "evidence_text",
    "confidence",
    "asof_loaded_at",
    "note",
]

SP500_INTERVAL_COLUMNS = [
    "index_code",
    "ticker",
    "security_name",
    "identifier_type",
    "identifier_value",
    "valid_from",
    "valid_to",
    "event_type",
    "effective_date",
    "announcement_date",
    "event_id",
    "source_type",
    "source_tier",
    "source_name",
    "source_ref",
    "source_url",
    "source_doc_id",
    "source_row_hash",
    "provenance_text",
    "evidence_text",
    "confidence",
    "mapping_note",
    "reason",
    "created_at",
    "updated_at",
]

SP500_SNAPSHOT_REQUIRED_COLUMNS = [
    "ticker",
    "sp500_pit_member",
    "sp500_pit_source",
    "sp500_pit_confidence",
    "sp500_pit_valid_from",
    "sp500_pit_valid_to",
    "sp500_pit_provenance",
]

PROVIDER_ORDER_DEFAULT = ["wrds", "spdji", "manual", "secondary"]
VALID_SOURCE_TYPES = {"official", "licensed", "secondary", "manual"}
VALID_SOURCE_TIERS = {"official", "licensed", "secondary", "manual_override"}
VALID_ACTIONS = {
    "add",
    "remove",
    "replace",
    "rename",
    "seed",
    "unknown",
    "spin_merge_note",
    "spin_merge_adjust",
}
SEED_POLICIES = {"allow", "official_only", "manual_only", "disable_secondary"}


def _coerce_ts(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if hasattr(ts, "tz") and ts.tz is not None:
        ts = ts.tz_localize(None)
    return pd.Timestamp(ts).normalize()


def _safe_float(value: Any, default: float = 0.0) -> float:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(v):
        return float(default)
    return float(v)


def _sha1_text(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8"), usedforsecurity=False).hexdigest()


def _source_type_to_tier(source_type: str) -> str:
    st = str(source_type or "").strip().lower()
    if st in {"official", "licensed", "secondary"}:
        return st
    if st in {"manual", "manual_override"}:
        return "manual_override"
    return "secondary"


def _normalize_source_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in VALID_SOURCE_TYPES:
        return text
    if text in {"manual_override", "manual"}:
        return "manual"
    if text in {"official", "licensed", "secondary"}:
        return text
    return "secondary"


def _normalize_action(value: Any) -> str:
    text = str(value or "").strip().lower()
    alias = {
        "delete": "remove",
        "drop": "remove",
        "in": "add",
        "out": "remove",
        "merge": "spin_merge_note",
        "spin": "spin_merge_note",
    }
    text = alias.get(text, text)
    if text in VALID_ACTIONS:
        return text
    return "unknown"


def _normalize_provider_order(provider_order: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if provider_order is None:
        return PROVIDER_ORDER_DEFAULT.copy()
    if isinstance(provider_order, str):
        items = [x.strip().lower() for x in provider_order.split(",") if x.strip()]
    else:
        items = [str(x).strip().lower() for x in provider_order if str(x).strip()]
    out: list[str] = []
    for x in items:
        if x in {"wrds", "spdji", "manual", "github", "secondary"} and x not in out:
            out.append(x)
    return out or PROVIDER_ORDER_DEFAULT.copy()


def _normalize_seed_policy(seed_policy: str | None) -> str:
    text = str(seed_policy or "allow").strip().lower()
    return text if text in SEED_POLICIES else "allow"


def _event_rank(action: str) -> int:
    mapping = {
        "seed": 10,
        "add": 20,
        "rename": 30,
        "replace": 35,
        "spin_merge_note": 40,
        "spin_merge_adjust": 40,
        "unknown": 50,
        "remove": 90,
    }
    return int(mapping.get(str(action).strip().lower(), 50))


def _source_rank(source_type: str) -> int:
    mapping = {
        "official": 10,
        "licensed": 20,
        "manual": 30,
        "secondary": 40,
    }
    return int(mapping.get(str(source_type).strip().lower(), 40))


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=SP500_NORMALIZED_EVENT_COLUMNS)


def _normalize_events_frame(
    events: pd.DataFrame | None,
    *,
    source_name: str,
    source_type: str,
    default_action: str = "unknown",
) -> pd.DataFrame:
    if events is None or events.empty:
        return _empty_events()

    out = events.copy()
    # Backward compatibility with previous schema names.
    rename_map = {
        "event_type": "action",
        "security_name": "company_name",
        "mapping_note": "note",
        "evidence_text": "provenance_text",
        "source_tier": "source_type",
        "source_url": "source_ref",
    }
    # Carefully merge legacy columns to avoid creating duplicated column names.
    for old, new in rename_map.items():
        if old not in out.columns:
            continue
        if new in out.columns:
            try:
                new_series = out[new]
                old_series = out[old]
                if isinstance(new_series, pd.DataFrame):
                    new_series = new_series.iloc[:, 0]
                if isinstance(old_series, pd.DataFrame):
                    old_series = old_series.iloc[:, 0]
                new_text = new_series.astype(str).str.strip()
                fill_mask = new_series.isna() | new_text.eq("")
                out.loc[fill_mask, new] = old_series.loc[fill_mask]
                out = out.drop(columns=[old])
            except Exception:
                out = out.drop(columns=[old])
        else:
            out = out.rename(columns={old: new})

    defaults = {
        "index_code": "SP500",
        "effective_date": pd.NaT,
        "announcement_date": pd.NaT,
        "action": default_action,
        "ticker": "",
        "ticker_new": "",
        "company_name": "",
        "company_name_new": "",
        "reason": "",
        "identifier_type": "ticker",
        "identifier_value": "",
        "source_name": source_name,
        "source_type": source_type,
        "source_ref": "",
        "source_url": "",
        "source_doc_id": "",
        "source_row_hash": "",
        "provenance_text": "",
        "evidence_text": "",
        "confidence": np.nan,
        "asof_loaded_at": now_utc_iso(),
        "note": "",
        "event_id": "",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out["index_code"] = out["index_code"].fillna("SP500").astype(str).str.strip().str.upper().replace("", "SP500")
    out["effective_date"] = pd.to_datetime(out["effective_date"], errors="coerce").dt.normalize()
    out["announcement_date"] = pd.to_datetime(out["announcement_date"], errors="coerce").dt.normalize()
    out["action"] = out["action"].map(_normalize_action)
    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip().str.upper()
    out["ticker_new"] = out["ticker_new"].fillna("").astype(str).str.strip().str.upper()
    out["company_name"] = out["company_name"].fillna("").astype(str).str.strip()
    out["company_name_new"] = out["company_name_new"].fillna("").astype(str).str.strip()
    out["reason"] = out["reason"].fillna("").astype(str).str.strip()
    out["identifier_type"] = out["identifier_type"].fillna("ticker").astype(str).str.strip().str.lower().replace("", "ticker")
    out["identifier_value"] = out["identifier_value"].fillna("").astype(str).str.strip()
    id_empty = out["identifier_value"].eq("")
    out.loc[id_empty, "identifier_value"] = out.loc[id_empty, "ticker"]

    out["source_name"] = out["source_name"].fillna(source_name).astype(str).str.strip().replace("", source_name)
    out["source_type"] = out["source_type"].fillna(source_type).map(_normalize_source_type)
    out["source_tier"] = out["source_type"].map(_source_type_to_tier)
    out["source_ref"] = out["source_ref"].fillna("").astype(str).str.strip()
    # Keep legacy source_url as separate convenience column if URL-like.
    out["source_url"] = out["source_url"].fillna("").astype(str).str.strip()
    missing_url = out["source_url"].eq("") & out["source_ref"].str.startswith(("http://", "https://"))
    out.loc[missing_url, "source_url"] = out.loc[missing_url, "source_ref"]
    out["source_doc_id"] = out["source_doc_id"].fillna("").astype(str).str.strip()
    out["provenance_text"] = out["provenance_text"].fillna("").astype(str).str.strip()
    out["evidence_text"] = out["evidence_text"].fillna(out["provenance_text"]).astype(str).str.strip()
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(np.nan)
    default_conf = {
        "official": 0.95,
        "licensed": 0.92,
        "manual": 0.75,
        "secondary": 0.35,
    }
    conf_missing = out["confidence"].isna()
    out.loc[conf_missing, "confidence"] = out.loc[conf_missing, "source_type"].map(default_conf).fillna(0.35)
    out["confidence"] = out["confidence"].clip(lower=0.0, upper=1.0)
    out["asof_loaded_at"] = out["asof_loaded_at"].fillna(now_utc_iso()).astype(str)
    out["note"] = out["note"].fillna("").astype(str).str.strip()

    # Basic validity filters.
    out = out.loc[out["effective_date"].notna()].copy()
    action_needs_old = out["action"].isin(["add", "remove", "seed", "unknown", "spin_merge_note", "spin_merge_adjust"])
    out = out.loc[(~action_needs_old) | out["ticker"].ne("")].copy()

    # Build row hash / stable event id.
    def _row_hash(row: pd.Series) -> str:
        base = {
            "index_code": row.get("index_code"),
            "effective_date": str(pd.Timestamp(row.get("effective_date")).date()) if pd.notna(row.get("effective_date")) else "",
            "announcement_date": str(pd.Timestamp(row.get("announcement_date")).date()) if pd.notna(row.get("announcement_date")) else "",
            "action": row.get("action"),
            "ticker": row.get("ticker"),
            "ticker_new": row.get("ticker_new"),
            "source_name": row.get("source_name"),
            "source_ref": row.get("source_ref"),
            "reason": row.get("reason"),
            "note": row.get("note"),
        }
        return _sha1_text(json.dumps(base, ensure_ascii=False, sort_keys=True))

    out["source_row_hash"] = out.apply(_row_hash, axis=1)
    out["event_id"] = out["event_id"].fillna("").astype(str).str.strip()
    missing_event_id = out["event_id"].eq("")
    out.loc[missing_event_id, "event_id"] = out.loc[missing_event_id, "source_row_hash"].map(lambda x: f"evt_{x[:16]}")

    return out[SP500_NORMALIZED_EVENT_COLUMNS].reset_index(drop=True)


def _validate_normalized_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events is None or events.empty:
        return _empty_events(), pd.DataFrame(columns=["row", "issue_type", "details", "severity"])

    issues: list[dict[str, Any]] = []
    out = events.copy()

    # action constraints
    bad_action = ~out["action"].isin(VALID_ACTIONS)
    for idx, row in out.loc[bad_action].iterrows():
        issues.append({"row": int(idx), "issue_type": "invalid_action", "details": str(row.get("action", "")), "severity": "fail"})

    needs_ticker_new = out["action"].isin(["replace", "rename"])
    missing_ticker_new = needs_ticker_new & out["ticker_new"].astype(str).str.strip().eq("")
    for idx, _ in out.loc[missing_ticker_new].iterrows():
        issues.append({"row": int(idx), "issue_type": "missing_ticker_new", "details": "replace/rename requires ticker_new", "severity": "fail"})

    missing_ticker = out["ticker"].astype(str).str.strip().eq("")
    for idx, _ in out.loc[missing_ticker].iterrows():
        issues.append({"row": int(idx), "issue_type": "missing_ticker", "details": "ticker required", "severity": "fail"})

    missing_eff = out["effective_date"].isna()
    for idx, _ in out.loc[missing_eff].iterrows():
        issues.append({"row": int(idx), "issue_type": "missing_effective_date", "details": "effective_date required", "severity": "fail"})

    dup_event_id = out["event_id"].duplicated(keep=False)
    for idx, row in out.loc[dup_event_id].iterrows():
        issues.append({"row": int(idx), "issue_type": "duplicate_event_id", "details": str(row.get("event_id", "")), "severity": "warn"})

    issues_df = pd.DataFrame(issues, columns=["row", "issue_type", "details", "severity"])
    hard_fail_rows = set(issues_df.loc[issues_df["severity"].eq("fail"), "row"].tolist()) if not issues_df.empty else set()
    if hard_fail_rows:
        out = out.drop(index=list(hard_fail_rows), errors="ignore").reset_index(drop=True)
    return out, issues_df


def ensure_sp500_pit_reference_files(
    *,
    manual_events_path: str | Path | None = None,
    events_raw_dir: str | Path | None = None,
) -> None:
    ensure_dir(SP500_PIT_DIR)
    ensure_dir(events_raw_dir or SP500_EVENTS_RAW_DIR)

    p = Path(manual_events_path).expanduser() if manual_events_path is not None else SP500_MANUAL_EVENTS_PATH
    ensure_dir(p.parent)
    if not p.exists():
        create_sp500_manual_template(p)


def create_sp500_manual_template(out_path: str | Path | None = None) -> Path:
    p = Path(out_path).expanduser() if out_path is not None else SP500_MANUAL_TEMPLATE_DEFAULT_PATH
    ensure_dir(p.parent)
    df = pd.DataFrame(
        columns=[
            "event_id",
            "effective_date",
            "announcement_date",
            "action",
            "ticker",
            "ticker_new",
            "company_name",
            "company_name_new",
            "reason",
            "identifier_type",
            "identifier_value",
            "source_ref",
            "source_doc_id",
            "provenance_text",
            "confidence",
            "note",
        ]
    )
    df.to_csv(p, index=False, encoding="utf-8")
    return p


def ingest_sp500_manual_events(
    events_file: str | Path,
    *,
    append: bool = True,
    replace: bool = False,
    target_path: str | Path | None = None,
    strict: bool = True,
    require_source_ref: bool = False,
    require_source_doc_id: bool = False,
    require_provenance_text: bool = False,
    require_confidence: bool = False,
    normalize_complex_actions: bool = True,
    rebuild_after_ingest: bool = False,
    rebuild_start: str = "2000-01-01",
    rebuild_end: str | None = None,
    rebuild_provider_order: str | list[str] | tuple[str, ...] | None = "manual,secondary",
    rebuild_seed_policy: str = "allow",
    rebuild_fail_closed: bool = False,
    rebuild_min_confidence: float = 0.7,
    rebuild_pit_dir: str | Path | None = None,
) -> dict[str, Any]:
    src = Path(events_file).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"manual events file not found: {src}")

    target = Path(target_path).expanduser() if target_path is not None else SP500_MANUAL_EVENTS_PATH
    ensure_dir(target.parent)

    raw = pd.read_csv(src)
    normalized = _normalize_events_frame(raw, source_name="manual_csv", source_type="manual", default_action="unknown")
    if normalize_complex_actions:
        normalized = _expand_complex_events(normalized)
    normalized, issues_df = _validate_normalized_events(normalized)

    required_issues: list[dict[str, Any]] = []
    if not normalized.empty:
        if require_source_ref:
            mask = normalized["source_ref"].astype(str).str.strip().eq("")
            for idx, _ in normalized.loc[mask].iterrows():
                required_issues.append(
                    {"row": int(idx), "issue_type": "missing_source_ref", "details": "source_ref is required", "severity": "fail"}
                )
        if require_source_doc_id:
            mask = normalized["source_doc_id"].astype(str).str.strip().eq("")
            for idx, _ in normalized.loc[mask].iterrows():
                required_issues.append(
                    {"row": int(idx), "issue_type": "missing_source_doc_id", "details": "source_doc_id is required", "severity": "fail"}
                )
        if require_provenance_text:
            mask = normalized["provenance_text"].astype(str).str.strip().eq("")
            for idx, _ in normalized.loc[mask].iterrows():
                required_issues.append(
                    {"row": int(idx), "issue_type": "missing_provenance_text", "details": "provenance_text is required", "severity": "fail"}
                )
        if require_confidence:
            mask = pd.to_numeric(normalized["confidence"], errors="coerce").isna()
            for idx, _ in normalized.loc[mask].iterrows():
                required_issues.append(
                    {"row": int(idx), "issue_type": "missing_confidence", "details": "confidence is required", "severity": "fail"}
                )
    if required_issues:
        req_df = pd.DataFrame(required_issues, columns=["row", "issue_type", "details", "severity"])
        issues_df = pd.concat([issues_df, req_df], ignore_index=True, sort=False) if not issues_df.empty else req_df

    if strict and not issues_df.empty and (issues_df["severity"] == "fail").any():
        sample = issues_df.loc[issues_df["severity"] == "fail"].head(5).to_dict("records")
        raise ValueError(f"manual events validation failed: {sample}")

    if replace:
        merged = normalized.copy()
    else:
        existing = pd.DataFrame(columns=SP500_NORMALIZED_EVENT_COLUMNS)
        if target.exists():
            try:
                existing = pd.read_csv(target)
                existing = _normalize_events_frame(existing, source_name="manual_csv", source_type="manual")
            except Exception:
                existing = pd.DataFrame(columns=SP500_NORMALIZED_EVENT_COLUMNS)
        if append:
            merged = pd.concat([existing, normalized], ignore_index=True, sort=False)
        else:
            merged = normalized.copy()

    if "event_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["event_id"], keep="last")
    logical_keys = [
        k
        for k in ["effective_date", "action", "ticker", "ticker_new", "source_name", "source_doc_id"]
        if k in merged.columns
    ]
    if logical_keys:
        merged = merged.drop_duplicates(subset=logical_keys, keep="last")
    merged = merged.sort_values(["effective_date", "action", "ticker", "event_id"]).reset_index(drop=True)
    merged.to_csv(target, index=False, encoding="utf-8")

    rebuild_summary: dict[str, Any] | None = None
    if rebuild_after_ingest:
        rebuild_summary = build_sp500_pit(
            start=rebuild_start,
            end=rebuild_end,
            provider_order=rebuild_provider_order,
            seed_policy=rebuild_seed_policy,
            strict=bool(strict),
            fail_closed=bool(rebuild_fail_closed),
            min_confidence=float(rebuild_min_confidence),
            force_refresh=True,
            pit_dir=rebuild_pit_dir,
            manual_events_path=target,
        )

    out = {
        "status": "ok",
        "target_path": str(target),
        "rows_written": int(len(merged)),
        "ingested_rows": int(len(normalized)),
        "issue_count": int(len(issues_df)),
        "issues": issues_df.to_dict("records")[:20],
    }
    if rebuild_summary is not None:
        out["rebuild"] = rebuild_summary
    return out


def import_sp500_github_secondary(
    *,
    out_path: str | Path,
    repo_url: str = "fja05680/sp500",
    raw_url: str | None = None,
    cache_dir: str | Path | None = None,
    confidence_default: float = 0.7,
    append: bool = True,
    replace: bool = False,
    target_path: str | Path | None = None,
    strict: bool = False,
    fail_closed: bool = False,
    dry_run: bool = False,
    force_refresh: bool = False,
    since_date: str | None = None,
    until_date: str | None = None,
    parse_fail_rate_threshold: float = 0.2,
    missing_ticker_rate_threshold: float = 0.05,
    rebuild_after_import: bool = False,
    rebuild_start: str = "2000-01-01",
    rebuild_end: str | None = None,
    rebuild_provider_order: str | list[str] | tuple[str, ...] | None = "manual,secondary",
    rebuild_seed_policy: str = "allow",
    rebuild_fail_closed: bool = False,
    rebuild_min_confidence: float = 0.7,
    pit_dir: str | Path | None = None,
) -> dict[str, Any]:
    out_csv = Path(out_path).expanduser()
    ensure_dir(out_csv.parent)

    provider = GitHubSecondarySP500Provider(
        GitHubSecondaryConfig(
            repo_url=repo_url,
            raw_url=raw_url,
            cache_dir=Path(cache_dir).expanduser() if cache_dir else None,
            confidence_default=float(confidence_default),
            strict=bool(strict),
            force_refresh=bool(force_refresh),
            since_date=since_date,
            until_date=until_date,
        )
    )
    start = since_date or "1900-01-01"
    end = until_date or pd.Timestamp.now().date().isoformat()
    events = provider.fetch_events(start=start, end=end)
    issues = list(getattr(provider, "last_issues", []) or [])
    stats = dict(getattr(provider, "last_stats", {}) or {})

    schema_drift = bool(stats.get("schema_drift_detected", False))
    if not schema_drift:
        for it in issues:
            issue_type = str(it.get("issue_type", "")).lower()
            message = str(it.get("message", "")).lower()
            if "schema_drift" in issue_type or "schema_drift" in message:
                schema_drift = True
                break
    rows_scanned = int(stats.get("rows", 0) or 0)
    parse_fail_count = int(stats.get("parse_fail_count", 0) or 0)
    parse_fail_rate = float(parse_fail_count / rows_scanned) if rows_scanned > 0 else 0.0
    missing_ticker_rate = float(events["ticker"].astype(str).str.strip().eq("").mean()) if not events.empty else 0.0

    fail_reasons: list[str] = []
    if schema_drift:
        fail_reasons.append("schema_drift_detected")
    if parse_fail_rate > float(parse_fail_rate_threshold):
        fail_reasons.append(
            f"parse_fail_rate {parse_fail_rate:.2%} > threshold {float(parse_fail_rate_threshold):.2%}"
        )
    if missing_ticker_rate > float(missing_ticker_rate_threshold):
        fail_reasons.append(
            f"missing_ticker_rate {missing_ticker_rate:.2%} > threshold {float(missing_ticker_rate_threshold):.2%}"
        )
    if events.empty:
        fail_reasons.append("no_events_parsed")

    if fail_reasons and (strict or fail_closed):
        raise RuntimeError(
            "github secondary import failed in strict mode: " + "; ".join(fail_reasons)
        )

    if not dry_run:
        events.to_csv(out_csv, index=False, encoding="utf-8")

    ingest_summary: dict[str, Any] | None = None
    if not dry_run and not events.empty:
        ingest_summary = ingest_sp500_manual_events(
            out_csv,
            append=bool(append and not replace),
            replace=bool(replace),
            target_path=target_path,
            strict=bool(strict),
            require_source_ref=True,
            require_source_doc_id=True,
            require_provenance_text=True,
            require_confidence=True,
            normalize_complex_actions=True,
            rebuild_after_ingest=bool(rebuild_after_import),
            rebuild_start=rebuild_start,
            rebuild_end=rebuild_end,
            rebuild_provider_order=rebuild_provider_order,
            rebuild_seed_policy=rebuild_seed_policy,
            rebuild_fail_closed=bool(rebuild_fail_closed),
            rebuild_min_confidence=float(rebuild_min_confidence),
            rebuild_pit_dir=pit_dir,
        )

    return {
        "status": "ok" if not fail_reasons else "warn",
        "out_path": str(out_csv),
        "events_rows": int(len(events)),
        "issues": issues[:50],
        "stats": {
            **stats,
            "schema_drift_detected": schema_drift,
            "parse_fail_rate": parse_fail_rate,
            "missing_ticker_rate": missing_ticker_rate,
            "duplicated_events_dropped_count": int(events.duplicated(subset=["event_id"]).sum()) if not events.empty else 0,
            "github_secondary_import_count": int(len(events)),
        },
        "fail_reasons": fail_reasons,
        "dry_run": bool(dry_run),
        "ingest_summary": ingest_summary,
    }


def _load_manual_events(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path).expanduser() if path is not None else SP500_MANUAL_EVENTS_PATH
    if not p.exists():
        return _empty_events()
    try:
        raw = pd.read_csv(p)
    except Exception:
        return _empty_events()
    out = _normalize_events_frame(raw, source_name="manual_csv", source_type="manual")
    out, _ = _validate_normalized_events(out)
    return out


def _load_legacy_manual_override_events(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path).expanduser() if path is not None else SP500_MANUAL_OVERRIDE_PATH
    if not p.exists():
        return _empty_events()
    try:
        raw = pd.read_csv(p)
    except Exception:
        return _empty_events()
    # Legacy schema support.
    out = _normalize_events_frame(raw, source_name="manual_override_legacy", source_type="manual")
    out, _ = _validate_normalized_events(out)
    return out


def _collect_provider_events(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    provider_order: list[str],
    seed_policy: str,
    events_raw_dir: Path,
    manual_events_path: str | Path | None = None,
    manual_override_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    provider_stats: dict[str, Any] = {}
    collected: list[pd.DataFrame] = []

    start_ts = _coerce_ts(start)
    end_ts = _coerce_ts(end)

    wrds_path = Path(str(events_raw_dir / "wrds_sp500_events.csv"))
    spdji_path = Path(str(events_raw_dir / "spdji_announcements_events.csv"))
    github_cache_dir = Path(str(events_raw_dir / "github_cache"))

    wrds_provider = WRDSSP500Provider(local_csv=wrds_path)
    spdji_provider = SPDJIAnnouncementsProvider(local_csv=spdji_path)
    github_provider = GitHubSecondarySP500Provider(GitHubSecondaryConfig(cache_dir=github_cache_dir))
    secondary_provider = SnapshotSeedSP500Provider(snapshot_csv=None, seed_effective_date=str(start))

    provider_map = {
        "wrds": (wrds_provider, "licensed", "wrds_compustat"),
        "spdji": (spdji_provider, "official", "spdji_announcements"),
        "github": (github_provider, "secondary", "github_fja05680_sp500"),
        "secondary": (secondary_provider, "secondary", "snapshot_seed_secondary"),
    }

    effective_order = provider_order.copy()
    policy = _normalize_seed_policy(seed_policy)
    if policy == "disable_secondary":
        effective_order = [x for x in effective_order if x != "secondary"]
    elif policy == "official_only":
        effective_order = [x for x in effective_order if x in {"wrds", "spdji"}]
    elif policy == "manual_only":
        effective_order = ["manual"]

    for key in effective_order:
        if key == "manual":
            manual_df = _load_manual_events(manual_events_path)
            legacy_df = _load_legacy_manual_override_events(manual_override_path)
            provider_stats["manual"] = {
                "rows": int(len(manual_df) + len(legacy_df)),
                "status": "ok" if (not manual_df.empty or not legacy_df.empty) else "empty",
            }
            if not manual_df.empty:
                collected.append(manual_df)
            if not legacy_df.empty:
                collected.append(legacy_df)
            if manual_df.empty and legacy_df.empty:
                issues.append({"provider": "manual", "issue_type": "empty", "message": "no manual events"})
            continue

        entry = provider_map.get(key)
        if entry is None:
            continue
        provider, src_type, src_name = entry
        try:
            raw_df = provider.fetch_events(start=start, end=end)
        except Exception as exc:  # noqa: BLE001
            issues.append({"provider": key, "issue_type": "fetch_error", "message": str(exc)})
            provider_stats[key] = {"rows": 0, "status": "fetch_error", "error": str(exc)}
            continue
        if raw_df is None or raw_df.empty:
            issues.append({"provider": key, "issue_type": "empty", "message": "no events"})
            provider_stats[key] = {"rows": 0, "status": "empty"}
            extra_issues = getattr(provider, "last_issues", [])
            if isinstance(extra_issues, list):
                issues.extend(extra_issues)
            extra_stats = getattr(provider, "last_stats", None)
            if isinstance(extra_stats, dict):
                provider_stats[key]["stats"] = extra_stats
            continue

        # provider output may be legacy schema.
        # Only coerce legacy format when `action` is missing; some modern providers
        # include source_url while already emitting the new action-based schema.
        if "action" not in raw_df.columns and set(raw_df.columns).intersection({"event_type", "security_name", "source_tier", "source_url"}):
            raw_df = normalize_sp500_event_frame(raw_df, source_tier=src_type, source_name=src_name)

        norm = _normalize_events_frame(raw_df, source_name=src_name, source_type=src_type)
        norm, provider_issues = _validate_normalized_events(norm)
        if not provider_issues.empty:
            for _, r in provider_issues.head(20).iterrows():
                issues.append({
                    "provider": key,
                    "issue_type": str(r.get("issue_type", "validation")),
                    "message": str(r.get("details", "")),
                })
        if norm.empty:
            issues.append({"provider": key, "issue_type": "normalized_empty", "message": "rows dropped after normalize"})
            provider_stats[key] = {"rows": 0, "status": "normalized_empty"}
            continue
        collected.append(norm)
        provider_stats[key] = {
            "rows": int(len(norm)),
            "status": "ok",
            "source_type": src_type,
            "source_name": src_name,
        }
        extra_issues = getattr(provider, "last_issues", [])
        if isinstance(extra_issues, list):
            issues.extend(extra_issues)
        extra_stats = getattr(provider, "last_stats", None)
        if isinstance(extra_stats, dict):
            provider_stats[key]["stats"] = extra_stats

    if not collected:
        return _empty_events(), issues, provider_stats

    events = pd.concat(collected, ignore_index=True, sort=False)
    events = events.drop_duplicates(subset=["event_id"], keep="last")

    if end_ts is not None:
        events = events.loc[events["effective_date"] <= end_ts].copy()
    if start_ts is not None:
        # Keep some earlier events so as-of at start can be built; no lower truncation by default.
        pass

    events = events.sort_values(["effective_date", "announcement_date", "source_type", "event_id"], na_position="last").reset_index(drop=True)
    return events, issues, provider_stats


def _expand_complex_events(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return _empty_events()

    rows: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        action = str(row.get("action", "unknown")).strip().lower()
        ticker = str(row.get("ticker", "")).strip().upper()
        ticker_new = str(row.get("ticker_new", "")).strip().upper()

        if action in {"replace", "rename"} and ticker_new:
            left = row.to_dict()
            left["action"] = "remove"
            left["note"] = (str(left.get("note", "")) + f" | normalized_from_{action}_old").strip(" |")
            left["event_id"] = str(left.get("event_id", "")) + "_old"
            rows.append(left)

            right = row.to_dict()
            right["ticker"] = ticker_new
            right["identifier_value"] = ticker_new
            if str(right.get("company_name_new", "")).strip():
                right["company_name"] = str(right.get("company_name_new", "")).strip()
            right["action"] = "add"
            right["note"] = (str(right.get("note", "")) + f" | normalized_from_{action}_new").strip(" |")
            right["event_id"] = str(right.get("event_id", "")) + "_new"
            rows.append(right)
            continue

        rows.append(row.to_dict())

    out = pd.DataFrame(rows)
    out = _normalize_events_frame(out, source_name="normalized", source_type="manual", default_action="unknown")
    return out


def _build_interval_row(
    open_event: pd.Series,
    *,
    valid_from: pd.Timestamp,
    valid_to: pd.Timestamp | None,
    reason: str = "",
) -> dict[str, Any]:
    src_ref = str(open_event.get("source_ref") or "").strip()
    src_url = str(open_event.get("source_url") or "").strip()
    if not src_url and src_ref.startswith(("http://", "https://")):
        src_url = src_ref
    eff_ts = _coerce_ts(open_event.get("effective_date"))
    ann_ts = _coerce_ts(open_event.get("announcement_date"))
    return {
        "index_code": str(open_event.get("index_code") or "SP500").strip().upper() or "SP500",
        "ticker": str(open_event.get("ticker") or "").strip().upper(),
        "security_name": str(open_event.get("company_name") or "").strip(),
        "identifier_type": str(open_event.get("identifier_type") or "ticker").strip().lower() or "ticker",
        "identifier_value": str(open_event.get("identifier_value") or open_event.get("ticker") or "").strip(),
        "valid_from": pd.Timestamp(valid_from).normalize(),
        "valid_to": (pd.Timestamp(valid_to).normalize() if valid_to is not None else pd.NaT),
        "event_type": str(open_event.get("action") or "unknown").strip().lower(),
        "effective_date": eff_ts if eff_ts is not None else pd.NaT,
        "announcement_date": ann_ts if ann_ts is not None else pd.NaT,
        "event_id": str(open_event.get("event_id") or "").strip(),
        "source_type": _normalize_source_type(open_event.get("source_type")),
        "source_tier": _source_type_to_tier(open_event.get("source_type")),
        "source_name": str(open_event.get("source_name") or "").strip(),
        "source_ref": src_ref,
        "source_url": src_url,
        "source_doc_id": str(open_event.get("source_doc_id") or "").strip(),
        "source_row_hash": str(open_event.get("source_row_hash") or "").strip(),
        "provenance_text": str(open_event.get("provenance_text") or "").strip(),
        "evidence_text": str(open_event.get("evidence_text") or open_event.get("provenance_text") or "").strip(),
        "confidence": float(_safe_float(open_event.get("confidence"), 0.0)),
        "mapping_note": str(open_event.get("note") or "").strip(),
        "reason": str(reason or open_event.get("reason") or "").strip(),
        "created_at": now_utc_iso(),
        "updated_at": now_utc_iso(),
    }


def build_sp500_intervals_from_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events is None or events.empty:
        return pd.DataFrame(columns=SP500_INTERVAL_COLUMNS), pd.DataFrame(columns=["ticker", "issue_type", "details", "severity"])

    df = _expand_complex_events(events)
    df, norm_issues = _validate_normalized_events(df)

    df["source_rank"] = df["source_type"].map(_source_rank).fillna(40).astype(int)
    df["event_rank"] = df["action"].map(_event_rank).fillna(50).astype(int)
    df = df.sort_values(
        ["ticker", "effective_date", "announcement_date", "event_rank", "source_rank", "confidence", "event_id"],
        ascending=[True, True, True, True, True, False, True],
        na_position="last",
    )

    interval_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for ticker, g in df.groupby("ticker", sort=True):
        open_event: pd.Series | None = None
        open_from: pd.Timestamp | None = None

        for _, row in g.iterrows():
            action = str(row.get("action") or "unknown").strip().lower()
            eff = _coerce_ts(row.get("effective_date"))
            if eff is None:
                issues.append({"ticker": ticker, "issue_type": "missing_effective_date", "details": f"event_id={row.get('event_id','')}", "severity": "fail"})
                continue

            if action in {"add", "seed", "unknown", "spin_merge_note", "spin_merge_adjust"}:
                if open_event is None:
                    open_event = row
                    open_from = eff
                    continue
                close_to = eff - pd.Timedelta(days=1)
                if open_from is not None and close_to >= open_from:
                    interval_rows.append(
                        _build_interval_row(
                            open_event,
                            valid_from=open_from,
                            valid_to=close_to,
                            reason="auto_closed_by_subsequent_add",
                        )
                    )
                    issues.append(
                        {
                            "ticker": ticker,
                            "issue_type": "duplicate_add_open",
                            "details": f"closed previous interval at {close_to.date().isoformat()} due to add event_id={row.get('event_id','')}",
                            "severity": "warn",
                        }
                    )
                else:
                    issues.append(
                        {
                            "ticker": ticker,
                            "issue_type": "invalid_auto_close",
                            "details": f"open_from={open_from} close_to={close_to}",
                            "severity": "warn",
                        }
                    )
                open_event = row
                open_from = eff
                continue

            if action == "remove":
                if open_event is None or open_from is None:
                    issues.append(
                        {
                            "ticker": ticker,
                            "issue_type": "remove_without_add",
                            "details": f"event_id={row.get('event_id','')} effective_date={eff.date().isoformat()}",
                            "severity": "warn",
                        }
                    )
                    continue
                close_to = eff - pd.Timedelta(days=1)
                if close_to < open_from:
                    issues.append(
                        {
                            "ticker": ticker,
                            "issue_type": "remove_before_add",
                            "details": f"open_from={open_from.date().isoformat()} remove_effective_date={eff.date().isoformat()}",
                            "severity": "fail",
                        }
                    )
                    open_event = None
                    open_from = None
                    continue
                interval_rows.append(
                    _build_interval_row(open_event, valid_from=open_from, valid_to=close_to, reason="closed_by_remove")
                )
                open_event = None
                open_from = None
                continue

            issues.append(
                {
                    "ticker": ticker,
                    "issue_type": "unsupported_action",
                    "details": f"action={action} event_id={row.get('event_id','')}",
                    "severity": "warn",
                }
            )

        if open_event is not None and open_from is not None:
            interval_rows.append(_build_interval_row(open_event, valid_from=open_from, valid_to=None, reason="open_interval"))

    intervals = pd.DataFrame(interval_rows, columns=SP500_INTERVAL_COLUMNS)
    if not intervals.empty:
        intervals["valid_from"] = pd.to_datetime(intervals["valid_from"], errors="coerce").dt.normalize()
        intervals["valid_to"] = pd.to_datetime(intervals["valid_to"], errors="coerce").dt.normalize()
        intervals["effective_date"] = pd.to_datetime(intervals["effective_date"], errors="coerce").dt.normalize()
        intervals["announcement_date"] = pd.to_datetime(intervals["announcement_date"], errors="coerce").dt.normalize()
        intervals["confidence"] = pd.to_numeric(intervals["confidence"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        intervals = intervals.drop_duplicates(subset=["ticker", "valid_from", "valid_to", "event_id"], keep="last")
        intervals = intervals.sort_values(["ticker", "valid_from", "valid_to", "confidence"], ascending=[True, True, True, False], na_position="last").reset_index(drop=True)

    issues_df = pd.DataFrame(issues, columns=["ticker", "issue_type", "details", "severity"])
    if not norm_issues.empty:
        norm_issues2 = norm_issues.copy()
        norm_issues2["ticker"] = ""
        norm_issues2 = norm_issues2[["ticker", "issue_type", "details", "severity"]]
        issues_df = pd.concat([issues_df, norm_issues2], ignore_index=True, sort=False)
    return intervals, issues_df


def _build_daily_counts(intervals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if start > end:
        start, end = end, start
    days = pd.date_range(start=start, end=end, freq="B")
    if len(days) == 0:
        return pd.DataFrame(columns=["date", "constituent_count"])
    if intervals is None or intervals.empty:
        return pd.DataFrame({"date": days, "constituent_count": 0})

    work = intervals.copy()
    work["valid_from"] = pd.to_datetime(work["valid_from"], errors="coerce").dt.normalize()
    work["valid_to"] = pd.to_datetime(work["valid_to"], errors="coerce").dt.normalize()

    counts: list[dict[str, Any]] = []
    for day in days:
        mask = work["valid_from"].notna() & (work["valid_from"] <= day)
        mask &= work["valid_to"].isna() | (work["valid_to"] >= day)
        n = int(work.loc[mask, "ticker"].astype(str).str.upper().nunique())
        counts.append({"date": pd.Timestamp(day), "constituent_count": n})
    return pd.DataFrame(counts)


def _sample_asof_quality(intervals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    if intervals.empty:
        return []
    days = pd.date_range(start=start, end=end, freq="2YS")
    if len(days) == 0:
        days = pd.DatetimeIndex([start, end])
    out: list[dict[str, Any]] = []
    for d in days[:8]:
        snap = get_sp500_constituents_asof(d, pit_df=intervals, strict=False)
        out.append(
            {
                "date": pd.Timestamp(d).date().isoformat(),
                "constituent_count": int(len(snap)),
                "low_confidence_count": int((pd.to_numeric(snap.get("confidence"), errors="coerce") < 0.7).sum()) if not snap.empty else 0,
            }
        )
    return out


def validate_sp500_pit(
    intervals: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    events: pd.DataFrame | None = None,
    issues_df: pd.DataFrame | None = None,
    min_confidence: float = 0.7,
    max_missing_provenance_rate: float = 0.2,
    max_low_confidence_rate: float = 0.5,
    min_coverage_ratio: float = 0.95,
    strict: bool = False,
    fail_closed: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    issues: list[dict[str, Any]] = []

    if intervals is None or intervals.empty:
        errors.append("sp500 pit intervals are empty")
        out = {
            "status": "fail",
            "summary": {
                "errors": errors,
                "warnings": warnings,
                "interval_rows": 0,
                "ticker_count": 0,
                "coverage_ratio": 0.0,
                "low_confidence_rate": 1.0,
                "missing_provenance_rate": 1.0,
            },
            "issues": pd.DataFrame(columns=["ticker", "issue_type", "details", "severity"]),
            "daily_counts": pd.DataFrame(columns=["date", "constituent_count"]),
            "event_stats": {},
            "sample_asof": [],
        }
        if fail_closed:
            raise RuntimeError("SP500 PIT validation failed: empty intervals")
        return out

    df = intervals.copy()
    for col in SP500_INTERVAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["valid_from"] = pd.to_datetime(df["valid_from"], errors="coerce").dt.normalize()
    df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce").dt.normalize()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    df["source_type"] = df["source_type"].map(_normalize_source_type)
    df["source_tier"] = df["source_tier"].astype(str).str.strip().str.lower()

    bad_rev = df.loc[df["valid_to"].notna() & (df["valid_to"] < df["valid_from"])]
    for _, row in bad_rev.iterrows():
        issues.append({
            "ticker": row.get("ticker", ""),
            "issue_type": "reversed_interval",
            "details": f"valid_from={row.get('valid_from')} valid_to={row.get('valid_to')}",
            "severity": "fail",
        })
    if not bad_rev.empty:
        errors.append(f"reversed intervals: {len(bad_rev)}")

    overlap_n = 0
    for ticker, g in df.sort_values(["ticker", "valid_from", "valid_to"], na_position="last").groupby("ticker"):
        prev_to: pd.Timestamp | None = None
        for _, row in g.iterrows():
            cur_from = _coerce_ts(row.get("valid_from"))
            cur_to = _coerce_ts(row.get("valid_to"))
            if cur_from is None:
                issues.append({"ticker": ticker, "issue_type": "missing_valid_from", "details": "", "severity": "fail"})
                continue
            if prev_to is not None and cur_from <= prev_to:
                overlap_n += 1
                issues.append(
                    {
                        "ticker": ticker,
                        "issue_type": "overlap_interval",
                        "details": f"cur_from={cur_from.date().isoformat()} prev_to={prev_to.date().isoformat()}",
                        "severity": "warn",
                    }
                )
            if prev_to is None:
                prev_to = cur_to
            elif prev_to is not pd.NaT:
                if cur_to is None:
                    prev_to = max(prev_to, cur_from)
                else:
                    prev_to = max(prev_to, cur_to)
    if overlap_n > 0:
        warnings.append(f"overlap intervals detected: {overlap_n}")

    invalid_tier = sorted(set(df.loc[~df["source_tier"].isin(VALID_SOURCE_TIERS), "source_tier"].astype(str)))
    if invalid_tier:
        warnings.append("unknown source_tier values: " + ",".join(invalid_tier))

    missing_prov_mask = (
        df["source_ref"].astype(str).str.strip().eq("")
        | df["provenance_text"].astype(str).str.strip().eq("")
    )
    missing_prov_rate = float(missing_prov_mask.mean()) if len(df) > 0 else 1.0

    low_conf_mask = df["confidence"] < float(min_confidence)
    low_conf_rate = float(low_conf_mask.mean()) if len(df) > 0 else 1.0

    start_ts = _coerce_ts(start if start is not None else df["valid_from"].min())
    end_ts = _coerce_ts(end if end is not None else pd.Timestamp.now().normalize())
    if start_ts is None or end_ts is None:
        errors.append("invalid date range for coverage")
        coverage_ratio = 0.0
        daily_counts = pd.DataFrame(columns=["date", "constituent_count"])
        sample_asof = []
    else:
        daily_counts = _build_daily_counts(df, start=start_ts, end=end_ts)
        coverage_ratio = float((daily_counts["constituent_count"] > 0).mean()) if not daily_counts.empty else 0.0
        sample_asof = _sample_asof_quality(df, start=start_ts, end=end_ts)

    if missing_prov_rate > max_missing_provenance_rate:
        msg = f"missing provenance rate {missing_prov_rate:.2%} exceeds threshold {max_missing_provenance_rate:.2%}"
        (errors if strict else warnings).append(msg)
    if low_conf_rate > max_low_confidence_rate:
        msg = f"low confidence rate {low_conf_rate:.2%} exceeds threshold {max_low_confidence_rate:.2%}"
        (errors if strict else warnings).append(msg)
    if coverage_ratio < min_coverage_ratio:
        msg = f"coverage ratio {coverage_ratio:.2%} below threshold {min_coverage_ratio:.2%}"
        (errors if strict else warnings).append(msg)

    event_stats: dict[str, Any] = {}
    if isinstance(events, pd.DataFrame) and not events.empty:
        ev = events.copy()
        ev["action"] = ev["action"].astype(str).str.lower()
        ev["source_type"] = ev["source_type"].map(_normalize_source_type)
        if "source_tier" in ev.columns:
            ev["source_tier"] = ev["source_tier"].astype(str).str.strip().str.lower()
        event_stats = {
            "event_rows": int(len(ev)),
            "by_action": {str(k): int(v) for k, v in ev.groupby("action").size().to_dict().items()},
            "by_source_type": {str(k): int(v) for k, v in ev.groupby("source_type").size().to_dict().items()},
        }
        if "source_tier" in ev.columns:
            event_stats["by_source_tier"] = {str(k): int(v) for k, v in ev.groupby("source_tier").size().to_dict().items()}

    if isinstance(issues_df, pd.DataFrame) and not issues_df.empty:
        for _, row in issues_df.iterrows():
            issues.append(
                {
                    "ticker": row.get("ticker", ""),
                    "issue_type": row.get("issue_type", ""),
                    "details": row.get("details", ""),
                    "severity": row.get("severity", "warn"),
                }
            )

    issues_out = pd.DataFrame(issues, columns=["ticker", "issue_type", "details", "severity"])
    fail_from_issues = int((issues_out.get("severity", pd.Series(dtype=str)) == "fail").sum()) if not issues_out.empty else 0
    if fail_from_issues > 0:
        errors.append(f"critical issue rows: {fail_from_issues}")

    interval_per_ticker = df.groupby("ticker", as_index=True).size() if not df.empty else pd.Series(dtype=int)
    multi_interval_ratio = float((interval_per_ticker > 1).mean()) if not interval_per_ticker.empty else 0.0
    issue_counts_by_type = (
        {str(k): int(v) for k, v in issues_out.groupby("issue_type").size().to_dict().items()}
        if not issues_out.empty
        else {}
    )
    source_tier_mix = (
        {str(k): int(v) for k, v in df.groupby("source_tier").size().to_dict().items()}
        if not df.empty
        else {}
    )
    confidence_series = pd.to_numeric(df.get("confidence"), errors="coerce").dropna()
    confidence_summary = {
        "min": float(confidence_series.min()) if not confidence_series.empty else 0.0,
        "max": float(confidence_series.max()) if not confidence_series.empty else 0.0,
        "mean": float(confidence_series.mean()) if not confidence_series.empty else 0.0,
        "median": float(confidence_series.median()) if not confidence_series.empty else 0.0,
        "p10": float(confidence_series.quantile(0.10)) if not confidence_series.empty else 0.0,
        "p90": float(confidence_series.quantile(0.90)) if not confidence_series.empty else 0.0,
    }

    status = "pass"
    if errors:
        status = "fail"
    elif warnings:
        status = "warn"

    out = {
        "status": status,
        "summary": {
            "errors": errors,
            "warnings": warnings,
            "interval_rows": int(len(df)),
            "ticker_count": int(df["ticker"].nunique()),
            "coverage_ratio": float(coverage_ratio),
            "low_confidence_rate": float(low_conf_rate),
            "missing_provenance_rate": float(missing_prov_rate),
            "multi_interval_ticker_ratio": float(multi_interval_ratio),
            "source_type_mix": {str(k): int(v) for k, v in df.groupby("source_type").size().to_dict().items()},
            "source_tier_mix": source_tier_mix,
            "issue_count": int(len(issues_out)),
            "issue_counts_by_type": issue_counts_by_type,
            "confidence_summary": confidence_summary,
        },
        "issues": issues_out,
        "daily_counts": daily_counts,
        "event_stats": event_stats,
        "sample_asof": sample_asof,
    }
    if fail_closed and status != "pass":
        raise RuntimeError(
            "SP500 PIT validation failed in fail-closed mode: "
            f"status={status} errors={len(errors)} warnings={len(warnings)}"
        )
    return out


def load_sp500_events(
    *,
    events_path: str | Path | None = None,
) -> pd.DataFrame:
    p = Path(events_path).expanduser() if events_path is not None else SP500_EVENTS_PATH
    if not p.exists():
        return _empty_events()
    try:
        raw = pd.read_csv(p)
    except Exception:
        return _empty_events()
    return _normalize_events_frame(raw, source_name="cached_events", source_type="manual")


def load_sp500_constituents_pit(
    *,
    pit_path: str | Path | None = None,
) -> pd.DataFrame:
    p = Path(pit_path).expanduser() if pit_path is not None else SP500_PIT_INTERVALS_PATH
    if not p.exists():
        return pd.DataFrame(columns=SP500_INTERVAL_COLUMNS)
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=SP500_INTERVAL_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=SP500_INTERVAL_COLUMNS)
    if "source_type" not in df.columns:
        if "source_tier" in df.columns:
            df["source_type"] = df["source_tier"].map(_normalize_source_type)
        else:
            df["source_type"] = "secondary"
    if "source_tier" not in df.columns:
        df["source_tier"] = df["source_type"].map(_source_type_to_tier)
    for col in ["valid_from", "valid_to", "effective_date", "announcement_date"]:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce")
            if hasattr(s.dt, "tz") and s.dt.tz is not None:
                s = s.dt.tz_localize(None)
            df[col] = s.dt.normalize()
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    return df


def get_sp500_constituents_asof(
    asof_date: str | pd.Timestamp,
    *,
    pit_df: pd.DataFrame | None = None,
    pit_path: str | Path | None = None,
    min_confidence: float = 0.0,
    strict: bool = False,
) -> pd.DataFrame:
    asof = _coerce_ts(asof_date)
    if asof is None:
        if strict:
            raise ValueError(f"Invalid asof date: {asof_date}")
        return pd.DataFrame(columns=SP500_INTERVAL_COLUMNS)

    df = pit_df.copy() if isinstance(pit_df, pd.DataFrame) else load_sp500_constituents_pit(pit_path=pit_path)
    if df.empty:
        if strict:
            raise FileNotFoundError("SP500 PIT intervals are not available")
        return pd.DataFrame(columns=SP500_INTERVAL_COLUMNS)

    work = df.copy()
    if "source_type" not in work.columns:
        if "source_tier" in work.columns:
            work["source_type"] = work["source_tier"].map(_normalize_source_type)
        else:
            work["source_type"] = "secondary"
    if "source_tier" not in work.columns:
        work["source_tier"] = work["source_type"].map(_source_type_to_tier)
    for col in ["valid_from", "valid_to"]:
        if col in work.columns:
            s = pd.to_datetime(work[col], errors="coerce")
            if hasattr(s.dt, "tz") and s.dt.tz is not None:
                s = s.dt.tz_localize(None)
            work[col] = s.dt.normalize()
    work["confidence"] = pd.to_numeric(work.get("confidence"), errors="coerce").fillna(0.0)

    # Use a more robust mask that handles NaT explicitly
    v_from = work["valid_from"]
    v_to = work["valid_to"]
    
    mask = v_from.notna() & (v_from <= asof)
    mask &= (v_to.isna()) | (v_to >= asof)
    if float(min_confidence) > 0.0:
        mask &= work["confidence"] >= float(min_confidence)
    out = work.loc[mask].copy()

    if out.empty and strict:
        raise RuntimeError(
            "SP500 PIT as-of lookup returned empty set in strict mode: "
            f"asof={asof.date().isoformat()} min_confidence={float(min_confidence):.2f}"
        )
    if out.empty:
        return pd.DataFrame(columns=SP500_INTERVAL_COLUMNS)

    # pick most recent interval per ticker; if conflict prefer higher confidence and source rank
    out["source_rank"] = out["source_type"].map(_source_rank).fillna(40)
    out = out.sort_values(["ticker", "valid_from", "confidence", "source_rank"], ascending=[True, False, False, True])
    out = out.drop_duplicates(subset=["ticker"], keep="first")
    return out.reset_index(drop=True)


def get_sp500_constituent_tickers_asof(
    asof_date: str | pd.Timestamp,
    *,
    pit_df: pd.DataFrame | None = None,
    pit_path: str | Path | None = None,
    min_confidence: float = 0.0,
    strict: bool = False,
) -> list[str]:
    snap = get_sp500_constituents_asof(
        asof_date,
        pit_df=pit_df,
        pit_path=pit_path,
        min_confidence=min_confidence,
        strict=strict,
    )
    if snap.empty:
        return []
    return sorted(snap["ticker"].astype(str).str.upper().unique().tolist())


def get_sp500_symbol_universe_for_period(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    pit_df: pd.DataFrame | None = None,
    pit_path: str | Path | None = None,
    min_confidence: float = 0.0,
) -> list[str]:
    start_ts = _coerce_ts(start)
    end_ts = _coerce_ts(end) if end is not None else pd.Timestamp.now().normalize()
    if start_ts is None or end_ts is None:
        return []
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    df = pit_df.copy() if isinstance(pit_df, pd.DataFrame) else load_sp500_constituents_pit(pit_path=pit_path)
    if df.empty:
        return []
    work = df.copy()
    for col in ["valid_from", "valid_to"]:
        if col in work.columns:
            s = pd.to_datetime(work[col], errors="coerce")
            if hasattr(s.dt, "tz") and s.dt.tz is not None:
                s = s.dt.tz_localize(None)
            work[col] = s.dt.normalize()
    work["confidence"] = pd.to_numeric(work.get("confidence"), errors="coerce").fillna(0.0)

    mask = work["valid_from"].notna() & (work["valid_from"] <= end_ts)
    mask &= work["valid_to"].isna() | (work["valid_to"] >= start_ts)
    if float(min_confidence) > 0.0:
        mask &= work["confidence"] >= float(min_confidence)
    out = work.loc[mask, "ticker"].astype(str).str.upper().str.strip()
    return sorted(set(x for x in out.tolist() if x))


def sp500_pit_diff(
    *,
    date_a: str,
    date_b: str,
    pit_df: pd.DataFrame | None = None,
    pit_path: str | Path | None = None,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    a = _coerce_ts(date_a)
    b = _coerce_ts(date_b)
    if a is None or b is None:
        raise ValueError("date_a/date_b must be parseable dates")
    snap_a = get_sp500_constituent_tickers_asof(a, pit_df=pit_df, pit_path=pit_path, min_confidence=min_confidence, strict=False)
    snap_b = get_sp500_constituent_tickers_asof(b, pit_df=pit_df, pit_path=pit_path, min_confidence=min_confidence, strict=False)
    set_a = set(snap_a)
    set_b = set(snap_b)
    added = sorted(set_b - set_a)
    removed = sorted(set_a - set_b)
    return {
        "date_a": a.date().isoformat(),
        "date_b": b.date().isoformat(),
        "count_a": len(set_a),
        "count_b": len(set_b),
        "added_count": len(added),
        "removed_count": len(removed),
        "added": added,
        "removed": removed,
    }


def _write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    event_stats = payload.get("event_stats", {}) if isinstance(payload, dict) else {}
    provider_stats = payload.get("provider_stats", {}) if isinstance(payload, dict) else {}

    lines: list[str] = []
    lines.append("# SP500 PIT Summary")
    lines.append("")
    lines.append(f"- status: `{payload.get('status')}`")
    lines.append(f"- interval_rows: `{summary.get('interval_rows', 0)}`")
    lines.append(f"- ticker_count: `{summary.get('ticker_count', 0)}`")
    lines.append(f"- coverage_ratio: `{summary.get('coverage_ratio', 0.0):.2%}`")
    lines.append(f"- low_confidence_rate: `{summary.get('low_confidence_rate', 0.0):.2%}`")
    lines.append(f"- missing_provenance_rate: `{summary.get('missing_provenance_rate', 0.0):.2%}`")
    lines.append(f"- multi_interval_ticker_ratio: `{summary.get('multi_interval_ticker_ratio', 0.0):.2%}`")
    lines.append(f"- issue_count: `{summary.get('issue_count', 0)}`")
    lines.append("")
    lines.append("## Source Mix")
    source_mix = summary.get("source_type_mix", {}) if isinstance(summary.get("source_type_mix"), dict) else {}
    if source_mix:
        for k, v in source_mix.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- n/a")
    lines.append("")
    lines.append("## Source Tier Mix")
    source_tier_mix = summary.get("source_tier_mix", {}) if isinstance(summary.get("source_tier_mix"), dict) else {}
    if source_tier_mix:
        for k, v in source_tier_mix.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- n/a")

    lines.append("")
    lines.append("## Event Stats")
    if event_stats:
        by_action = event_stats.get("by_action", {})
        by_source = event_stats.get("by_source_type", {})
        lines.append(f"- event_rows: {event_stats.get('event_rows', 0)}")
        lines.append("- by_action:")
        for k, v in by_action.items():
            lines.append(f"  - {k}: {v}")
        lines.append("- by_source_type:")
        for k, v in by_source.items():
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("- n/a")

    lines.append("")
    lines.append("## Provider Stats")
    if isinstance(provider_stats, dict) and provider_stats:
        for k, v in provider_stats.items():
            if isinstance(v, dict):
                lines.append(f"- {k}: rows={v.get('rows', 0)} status={v.get('status', 'n/a')}")
                stats = v.get("stats", {})
                if isinstance(stats, dict):
                    if "schema_drift_detected" in stats:
                        lines.append(f"  - schema_drift_detected: {stats.get('schema_drift_detected')}")
                    if "parse_fail_count" in stats:
                        lines.append(f"  - parse_fail_count: {stats.get('parse_fail_count')}")
                    if "events_emitted" in stats:
                        lines.append(f"  - events_emitted: {stats.get('events_emitted')}")
            else:
                lines.append(f"- {k}: {v}")
    else:
        lines.append("- n/a")

    errs = summary.get("errors", []) or []
    warns = summary.get("warnings", []) or []
    lines.append("")
    lines.append("## Errors")
    lines.extend([f"- {x}" for x in errs] if errs else ["- none"])
    lines.append("")
    lines.append("## Warnings")
    lines.extend([f"- {x}" for x in warns] if warns else ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_sp500_pit(
    *,
    start: str = "2000-01-01",
    end: str | None = None,
    provider_order: str | list[str] | tuple[str, ...] | None = None,
    seed_policy: str = "allow",
    strict: bool = False,
    fail_closed: bool = False,
    min_confidence: float = 0.7,
    force_refresh: bool = False,
    pit_dir: str | Path | None = None,
    manual_events_path: str | Path | None = None,
    manual_override_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(pit_dir).expanduser() if pit_dir is not None else SP500_PIT_DIR
    events_raw_dir = root / "events_raw"
    events_path = root / "sp500_pit_events.csv"
    intervals_path = root / "sp500_constituents_pit_intervals.parquet"
    pit_table_path = root / "sp500_constituents_pit.parquet"
    coverage_path = root / "sp500_pit_coverage_report.json"
    issues_path = root / "sp500_pit_issues.csv"
    daily_path = root / "sp500_pit_daily_counts.csv"
    summary_md_path = root / "sp500_pit_summary.md"

    ensure_dir(root)
    ensure_sp500_pit_reference_files(manual_events_path=manual_events_path, events_raw_dir=events_raw_dir)

    if intervals_path.exists() and events_path.exists() and not force_refresh:
        pit = load_sp500_constituents_pit(pit_path=intervals_path)
        events = load_sp500_events(events_path=events_path)
        issue_df = pd.read_csv(issues_path) if issues_path.exists() else pd.DataFrame(columns=["ticker", "issue_type", "details", "severity"])
        report = validate_sp500_pit(
            pit,
            start=start,
            end=end,
            events=events,
            issues_df=issue_df,
            min_confidence=min_confidence,
            strict=strict,
            fail_closed=bool(fail_closed),
        )
        payload = {
            "status": report.get("status", "warn"),
            "provider_order": _normalize_provider_order(provider_order),
            "seed_policy": _normalize_seed_policy(seed_policy),
            "summary": report.get("summary", {}),
            "event_stats": report.get("event_stats", {}),
            "paths": {
                "events_path": str(events_path),
                "intervals_path": str(intervals_path),
                "pit_table_path": str(pit_table_path),
                "coverage_report": str(coverage_path),
                "issues_path": str(issues_path),
                "daily_counts_path": str(daily_path),
                "summary_md_path": str(summary_md_path),
            },
            "used_cache": True,
            "generated_at": now_utc_iso(),
        }
        coverage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_summary_markdown(summary_md_path, payload)
        return payload

    provider_keys = _normalize_provider_order(provider_order)
    seed_policy_norm = _normalize_seed_policy(seed_policy)
    end_value = end or pd.Timestamp.now().date().isoformat()

    events, provider_issues, provider_stats = _collect_provider_events(
        start=start,
        end=end_value,
        provider_order=provider_keys,
        seed_policy=seed_policy_norm,
        events_raw_dir=events_raw_dir,
        manual_events_path=manual_events_path,
        manual_override_path=manual_override_path,
    )
    if events.empty:
        msg = "No S&P500 PIT events collected from providers"
        if strict or fail_closed:
            raise RuntimeError(msg)
        LOGGER.warning(msg)

    # Save normalized combined events for audit/replay.
    events.to_csv(events_path, index=False, encoding="utf-8")

    intervals, interval_issues = build_sp500_intervals_from_events(events)
    report = validate_sp500_pit(
        intervals,
        start=start,
        end=end,
        events=events,
        issues_df=interval_issues,
        min_confidence=min_confidence,
        strict=strict,
        fail_closed=bool(fail_closed),
    )

    intervals.to_parquet(intervals_path, index=False)
    intervals.to_parquet(pit_table_path, index=False)

    all_issues = report.get("issues")
    if not isinstance(all_issues, pd.DataFrame):
        all_issues = pd.DataFrame(columns=["ticker", "issue_type", "details", "severity"])
    if provider_issues:
        provider_df = pd.DataFrame(provider_issues)
        provider_df["ticker"] = ""
        provider_df["severity"] = np.where(provider_df["issue_type"].astype(str).eq("fetch_error"), "warn", "warn")
        provider_df["details"] = provider_df.apply(
            lambda x: f"provider={x.get('provider', '')}; message={x.get('message', '')}",
            axis=1,
        )
        provider_df["issue_type"] = "provider_" + provider_df["issue_type"].astype(str)
        provider_df = provider_df[["ticker", "issue_type", "details", "severity"]]
        all_issues = pd.concat([all_issues, provider_df], ignore_index=True, sort=False)
    all_issues.to_csv(issues_path, index=False, encoding="utf-8")

    daily_counts = report.get("daily_counts")
    if isinstance(daily_counts, pd.DataFrame):
        daily_counts.to_csv(daily_path, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["date", "constituent_count"]).to_csv(daily_path, index=False, encoding="utf-8")

    payload = {
        "status": report.get("status", "warn"),
        "provider_order": provider_keys,
        "seed_policy": seed_policy_norm,
        "interval_rows": int(len(intervals)),
        "ticker_count": int(intervals["ticker"].nunique()) if not intervals.empty else 0,
        "raw_events_rows": int(len(events)),
        "provider_issues": provider_issues,
        "provider_stats": provider_stats,
        "summary": report.get("summary", {}),
        "event_stats": report.get("event_stats", {}),
        "sample_asof": report.get("sample_asof", []),
        "paths": {
            "events_path": str(events_path),
            "intervals_path": str(intervals_path),
            "pit_table_path": str(pit_table_path),
            "coverage_report": str(coverage_path),
            "issues_path": str(issues_path),
            "daily_counts_path": str(daily_path),
            "summary_md_path": str(summary_md_path),
        },
        "generated_at": now_utc_iso(),
    }
    coverage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_markdown(summary_md_path, payload)

    if (strict or fail_closed) and str(payload.get("status", "")).lower() == "fail":
        raise RuntimeError(
            "SP500 PIT build failed in strict mode; see coverage report: "
            f"{coverage_path}"
        )
    return payload


def validate_sp500_pit_cache(
    *,
    start: str = "2000-01-01",
    end: str | None = None,
    pit_dir: str | Path | None = None,
    min_confidence: float = 0.7,
    strict: bool = False,
    fail_closed: bool = False,
) -> dict[str, Any]:
    root = Path(pit_dir).expanduser() if pit_dir is not None else SP500_PIT_DIR
    intervals_path = root / "sp500_constituents_pit_intervals.parquet"
    events_path = root / "sp500_pit_events.csv"
    issues_path = root / "sp500_pit_issues.csv"
    coverage_path = root / "sp500_pit_coverage_report.json"
    daily_path = root / "sp500_pit_daily_counts.csv"
    summary_md_path = root / "sp500_pit_summary.md"

    pit = load_sp500_constituents_pit(pit_path=intervals_path)
    events = load_sp500_events(events_path=events_path)
    issue_df = pd.read_csv(issues_path) if issues_path.exists() else pd.DataFrame(columns=["ticker", "issue_type", "details", "severity"])
    report = validate_sp500_pit(
        pit,
        start=start,
        end=end,
        events=events,
        issues_df=issue_df,
        min_confidence=min_confidence,
        strict=strict,
        fail_closed=fail_closed,
    )

    payload = {
        "status": report.get("status", "warn"),
        "summary": report.get("summary", {}),
        "event_stats": report.get("event_stats", {}),
        "sample_asof": report.get("sample_asof", []),
        "generated_at": now_utc_iso(),
        "paths": {
            "events_path": str(events_path),
            "intervals_path": str(intervals_path),
            "coverage_report": str(coverage_path),
            "issues_path": str(issues_path),
            "daily_counts_path": str(daily_path),
            "summary_md_path": str(summary_md_path),
        },
    }
    coverage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    issue_out = report.get("issues")
    if isinstance(issue_out, pd.DataFrame):
        issue_out.to_csv(issues_path, index=False, encoding="utf-8")
    daily_counts = report.get("daily_counts")
    if isinstance(daily_counts, pd.DataFrame):
        daily_counts.to_csv(daily_path, index=False, encoding="utf-8")
    _write_summary_markdown(summary_md_path, payload)
    return payload


def report_sp500_pit(
    *,
    pit_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(pit_dir).expanduser() if pit_dir is not None else SP500_PIT_DIR
    payload_path = root / "sp500_pit_coverage_report.json"
    if not payload_path.exists():
        return {"status": "warn", "message": f"coverage report not found: {payload_path}"}
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"status": "fail", "message": f"coverage report read failed: {exc}"}
    return payload


__all__ = [
    "SP500_PIT_DIR",
    "SP500_EVENTS_RAW_DIR",
    "SP500_EVENTS_PATH",
    "SP500_PIT_INTERVALS_PATH",
    "SP500_PIT_TABLE_PATH",
    "SP500_PIT_ISSUES_PATH",
    "SP500_PIT_COVERAGE_REPORT_PATH",
    "SP500_PIT_DAILY_COUNTS_PATH",
    "SP500_PIT_SUMMARY_PATH",
    "SP500_MANUAL_EVENTS_PATH",
    "SP500_MANUAL_OVERRIDE_PATH",
    "SP500_NORMALIZED_EVENT_COLUMNS",
    "SP500_INTERVAL_COLUMNS",
    "SP500_SNAPSHOT_REQUIRED_COLUMNS",
    "create_sp500_manual_template",
    "ingest_sp500_manual_events",
    "import_sp500_github_secondary",
    "ensure_sp500_pit_reference_files",
    "build_sp500_intervals_from_events",
    "build_sp500_pit",
    "validate_sp500_pit",
    "validate_sp500_pit_cache",
    "report_sp500_pit",
    "load_sp500_events",
    "load_sp500_constituents_pit",
    "get_sp500_constituents_asof",
    "get_sp500_constituent_tickers_asof",
    "get_sp500_symbol_universe_for_period",
    "sp500_pit_diff",
]
