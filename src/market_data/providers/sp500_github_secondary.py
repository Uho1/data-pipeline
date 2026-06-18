from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from market_data.providers.sp500_constituents_provider import BaseSP500ConstituentsProvider
from market_data.utils import ensure_dir, now_utc_iso

LOGGER = logging.getLogger(__name__)

DEFAULT_REPO = "fja05680/sp500"
DEFAULT_API_CONTENTS = "https://api.github.com/repos/{repo}/contents"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_ticker_token(token: str) -> str:
    t = str(token or "").strip().upper()
    if not t:
        return ""
    # Some historical rows store token as SYMBOL-YYYYMM (or YYYYMMDD) metadata.
    if "-" in t:
        left, right = t.rsplit("-", 1)
        if re.fullmatch(r"\d{4,8}", right):
            t = left.strip().upper()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[^A-Z0-9\.\-]", "", t)
    return t


def _split_tickers(raw: Any) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    out: list[str] = []
    for part in re.split(r"[,\|;]", text):
        sym = _normalize_ticker_token(part)
        if sym:
            out.append(sym)
    return sorted(set(out))


def _detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    cols = [str(c) for c in df.columns]
    lower_map = {str(c).lower().strip(): str(c) for c in cols}
    date_col = None
    tickers_col = None

    for cand in ["date", "as_of_date", "day", "snapshot_date"]:
        if cand in lower_map:
            date_col = lower_map[cand]
            break
    for cand in ["tickers", "components", "constituents", "members", "symbols"]:
        if cand in lower_map:
            tickers_col = lower_map[cand]
            break
    if date_col is None:
        # fuzzy
        for c in cols:
            lc = c.lower()
            if "date" in lc:
                date_col = c
                break
    if tickers_col is None:
        for c in cols:
            lc = c.lower()
            if any(k in lc for k in ["ticker", "component", "constituent", "member", "symbol"]):
                tickers_col = c
                break
    return date_col, tickers_col


def parse_github_historical_components_csv(
    raw_df: pd.DataFrame,
    *,
    source_ref: str,
    source_doc_id: str,
    confidence_default: float = 0.7,
    strict: bool = False,
    since_date: str | None = None,
    until_date: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), issues, {"parse_fail_count": 0, "schema_drift_detected": False, "rows": 0}

    df = raw_df.copy()
    date_col, tickers_col = _detect_columns(df)
    if date_col is None or tickers_col is None:
        msg = f"schema_drift: required columns not found (date={date_col}, tickers={tickers_col})"
        issues.append(
            {
                "provider": "github_secondary",
                "issue_type": "schema_drift",
                "message": msg,
            }
        )
        if strict:
            raise ValueError(msg)
        return pd.DataFrame(), issues, {"parse_fail_count": len(df), "schema_drift_detected": True, "rows": int(len(df))}

    work = df[[date_col, tickers_col]].rename(columns={date_col: "date", tickers_col: "tickers_raw"}).copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    parse_fail_count = int(work["date"].isna().sum())
    work = work.loc[work["date"].notna()].copy()
    if since_date:
        since_ts = pd.to_datetime(since_date, errors="coerce")
        if pd.notna(since_ts):
            work = work.loc[work["date"] >= pd.Timestamp(since_ts).normalize()]
    if until_date:
        until_ts = pd.to_datetime(until_date, errors="coerce")
        if pd.notna(until_ts):
            work = work.loc[work["date"] <= pd.Timestamp(until_ts).normalize()]
    work = work.sort_values("date").reset_index(drop=True)

    if work.empty:
        return pd.DataFrame(), issues, {"parse_fail_count": parse_fail_count, "schema_drift_detected": False, "rows": 0}

    events: list[dict[str, Any]] = []
    prev_set: set[str] | None = None
    for idx, row in work.iterrows():
        dt = pd.Timestamp(row["date"]).date().isoformat()
        cur_set = set(_split_tickers(row["tickers_raw"]))
        if prev_set is None:
            # seed first day as adds.
            for sym in sorted(cur_set):
                evt_base = f"github_secondary|{dt}|add|{sym}|{source_doc_id}"
                events.append(
                    {
                        "event_id": f"evt_{_sha1(evt_base)[:16]}",
                        "index_code": "SP500",
                        "effective_date": dt,
                        "announcement_date": dt,
                        "action": "add",
                        "ticker": sym,
                        "company_name": "",
                        "reason": "initial_membership_snapshot",
                        "source_name": "github_fja05680_sp500",
                        "source_type": "secondary",
                        "source_ref": source_ref,
                        "source_url": source_ref,
                        "source_doc_id": source_doc_id,
                        "provenance_text": (
                            "Parsed from fja05680/sp500 historical components CSV; "
                            f"row_index={idx}; derived_by_daily_set_diff"
                        ),
                        "evidence_text": f"date={dt}; action=add; ticker={sym}",
                        "confidence": float(confidence_default),
                        "note": "github_secondary_initial_seed",
                    }
                )
            prev_set = cur_set
            continue

        added = sorted(cur_set - prev_set)
        removed = sorted(prev_set - cur_set)
        for sym in added:
            evt_base = f"github_secondary|{dt}|add|{sym}|{source_doc_id}"
            events.append(
                {
                    "event_id": f"evt_{_sha1(evt_base)[:16]}",
                    "index_code": "SP500",
                    "effective_date": dt,
                    "announcement_date": dt,
                    "action": "add",
                    "ticker": sym,
                    "company_name": "",
                    "reason": "derived_add_from_daily_membership_diff",
                    "source_name": "github_fja05680_sp500",
                    "source_type": "secondary",
                    "source_ref": source_ref,
                    "source_url": source_ref,
                    "source_doc_id": source_doc_id,
                    "provenance_text": (
                        "Parsed from fja05680/sp500 historical components CSV; "
                        f"row_index={idx}; derived_by_daily_set_diff"
                    ),
                    "evidence_text": f"date={dt}; action=add; ticker={sym}",
                    "confidence": float(confidence_default),
                    "note": "github_secondary_derived",
                }
            )
        for sym in removed:
            evt_base = f"github_secondary|{dt}|remove|{sym}|{source_doc_id}"
            events.append(
                {
                    "event_id": f"evt_{_sha1(evt_base)[:16]}",
                    "index_code": "SP500",
                    "effective_date": dt,
                    "announcement_date": dt,
                    "action": "remove",
                    "ticker": sym,
                    "company_name": "",
                    "reason": "derived_remove_from_daily_membership_diff",
                    "source_name": "github_fja05680_sp500",
                    "source_type": "secondary",
                    "source_ref": source_ref,
                    "source_url": source_ref,
                    "source_doc_id": source_doc_id,
                    "provenance_text": (
                        "Parsed from fja05680/sp500 historical components CSV; "
                        f"row_index={idx}; derived_by_daily_set_diff"
                    ),
                    "evidence_text": f"date={dt}; action=remove; ticker={sym}",
                    "confidence": float(confidence_default),
                    "note": "github_secondary_derived",
                }
            )
        prev_set = cur_set

    ev_df = pd.DataFrame(events)
    if not ev_df.empty:
        ev_df = ev_df.drop_duplicates(subset=["event_id"], keep="last").reset_index(drop=True)
    stats = {
        "parse_fail_count": parse_fail_count,
        "schema_drift_detected": False,
        "rows": int(len(work)),
        "events_emitted": int(len(ev_df)),
        "input_columns": [str(c) for c in raw_df.columns],
    }
    return ev_df, issues, stats


@dataclass
class GitHubSecondaryConfig:
    repo_url: str = DEFAULT_REPO
    raw_url: str | None = None
    cache_dir: Path | None = None
    confidence_default: float = 0.7
    strict: bool = False
    force_refresh: bool = False
    since_date: str | None = None
    until_date: str | None = None


class GitHubSecondarySP500Provider(BaseSP500ConstituentsProvider):
    def __init__(self, cfg: GitHubSecondaryConfig | None = None) -> None:
        super().__init__(name="github_fja05680_sp500", source_tier="secondary")
        self.cfg = cfg or GitHubSecondaryConfig()
        self.last_issues: list[dict[str, Any]] = []
        self.last_stats: dict[str, Any] = {}

    def _cache_dir(self) -> Path:
        if self.cfg.cache_dir is not None:
            return Path(self.cfg.cache_dir).expanduser()
        return Path("data/index_identity_cache/sp500_pit/github_cache")

    def _meta_path(self) -> Path:
        return self._cache_dir() / "github_source_meta.json"

    def _raw_path(self) -> Path:
        return self._cache_dir() / "github_source_raw.csv"

    def _read_cached(self) -> pd.DataFrame | None:
        p = self._raw_path()
        if not p.exists():
            return None
        try:
            return pd.read_csv(p)
        except Exception:
            return None

    def _resolve_download(self) -> tuple[str, str]:
        if self.cfg.raw_url:
            return self.cfg.raw_url, "manual_raw_override"
        url = DEFAULT_API_CONTENTS.format(repo=self.cfg.repo_url.strip())
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError("unexpected github API response format")
        candidates = []
        for item in data:
            name = str(item.get("name", ""))
            if "Historical Components" in name and name.lower().endswith(".csv"):
                candidates.append(item)
        if not candidates:
            raise RuntimeError("target historical components CSV not found in repo")

        def _score(it: dict[str, Any]) -> tuple[int, str]:
            n = str(it.get("name", ""))
            # prefer dated file variant
            return (1 if "(" in n and ")" in n else 0, n)

        best = sorted(candidates, key=_score, reverse=True)[0]
        raw = str(best.get("download_url") or "").strip()
        if not raw:
            raise RuntimeError("github download_url missing")
        doc_id = f"{best.get('name','file')}:{best.get('sha','')}"
        return raw, doc_id

    def _download_csv(self) -> tuple[pd.DataFrame, str, str]:
        ensure_dir(self._cache_dir())
        meta_path = self._meta_path()
        cached_meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                cached_meta = {}

        raw_url, source_doc_id = self._resolve_download()
        headers: dict[str, str] = {}
        if not self.cfg.force_refresh:
            if cached_meta.get("raw_url") == raw_url:
                if cached_meta.get("etag"):
                    headers["If-None-Match"] = str(cached_meta["etag"])
                if cached_meta.get("last_modified"):
                    headers["If-Modified-Since"] = str(cached_meta["last_modified"])

        # local file override path
        if raw_url.startswith("file://"):
            local = Path(raw_url.replace("file://", "", 1)).expanduser()
            if not local.exists():
                raise FileNotFoundError(f"github raw file override not found: {local}")
            payload = local.read_bytes()
            self._raw_path().write_bytes(payload)
            sha = _sha256_bytes(payload)
            cached_meta = {
                "raw_url": raw_url,
                "source_doc_id": source_doc_id,
                "content_sha256": sha,
                "downloaded_at": now_utc_iso(),
            }
            meta_path.write_text(json.dumps(cached_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return pd.read_csv(self._raw_path()), raw_url, source_doc_id or f"github_raw:{sha[:16]}"

        try:
            resp = requests.get(raw_url, headers=headers, timeout=30)
            if resp.status_code == 304:
                cached_df = self._read_cached()
                if cached_df is not None:
                    return cached_df, raw_url, str(cached_meta.get("source_doc_id") or source_doc_id)
            resp.raise_for_status()
            payload = resp.content
            self._raw_path().write_bytes(payload)
            sha = _sha256_bytes(payload)
            cached_meta = {
                "raw_url": raw_url,
                "etag": resp.headers.get("ETag", ""),
                "last_modified": resp.headers.get("Last-Modified", ""),
                "source_doc_id": source_doc_id or f"github_raw:{sha[:16]}",
                "content_sha256": sha,
                "downloaded_at": now_utc_iso(),
            }
            meta_path.write_text(json.dumps(cached_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return pd.read_csv(self._raw_path()), raw_url, str(cached_meta.get("source_doc_id"))
        except Exception as exc:  # noqa: BLE001
            cached_df = self._read_cached()
            if cached_df is not None:
                self.last_issues.append(
                    {
                        "provider": "github_secondary",
                        "issue_type": "network_error_using_cache",
                        "message": str(exc),
                    }
                )
                return cached_df, raw_url, str(cached_meta.get("source_doc_id") or source_doc_id)
            raise

    def fetch_events(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        self.last_issues = []
        self.last_stats = {}
        try:
            raw_df, source_ref, source_doc_id = self._download_csv()
        except Exception as exc:  # noqa: BLE001
            self.last_issues.append(
                {"provider": "github_secondary", "issue_type": "fetch_error", "message": str(exc)}
            )
            LOGGER.warning("github secondary provider fetch failed: %s", exc)
            return pd.DataFrame()

        try:
            events, issues, stats = parse_github_historical_components_csv(
                raw_df,
                source_ref=source_ref,
                source_doc_id=source_doc_id,
                confidence_default=float(self.cfg.confidence_default),
                strict=bool(self.cfg.strict),
                since_date=self.cfg.since_date,
                until_date=self.cfg.until_date,
            )
        except Exception as exc:  # noqa: BLE001
            self.last_issues.append(
                {"provider": "github_secondary", "issue_type": "parse_error", "message": str(exc)}
            )
            LOGGER.warning("github secondary provider parse failed: %s", exc)
            return pd.DataFrame()

        self.last_issues.extend(issues)
        self.last_stats = stats
        if not events.empty:
            events["source_name"] = "github_fja05680_sp500"
            events["source_type"] = "secondary"
        return events


__all__ = [
    "GitHubSecondaryConfig",
    "GitHubSecondarySP500Provider",
    "parse_github_historical_components_csv",
]

