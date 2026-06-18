from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SP500_EVENT_COLUMNS = [
    "index_code",
    "ticker",
    "security_name",
    "identifier_type",
    "identifier_value",
    "event_type",
    "effective_date",
    "announcement_date",
    "valid_from",
    "valid_to",
    "source_tier",
    "source_name",
    "source_url",
    "source_doc_id",
    "evidence_text",
    "confidence",
    "mapping_note",
]


def _coerce_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def normalize_sp500_event_frame(
    events: pd.DataFrame | None,
    *,
    source_tier: str,
    source_name: str,
    default_event_type: str = "unknown",
) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(columns=SP500_EVENT_COLUMNS)

    out = events.copy()
    for col in SP500_EVENT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["index_code"] = out["index_code"].fillna("SP500").astype(str).str.strip().str.upper()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out = out.loc[out["ticker"] != ""].copy()
    out["security_name"] = out["security_name"].fillna("").astype(str)
    out["identifier_type"] = out["identifier_type"].fillna("ticker").astype(str).str.strip().str.lower()
    out["identifier_value"] = out["identifier_value"].fillna(out["ticker"]).astype(str).str.strip()
    out["event_type"] = (
        out["event_type"]
        .fillna(default_event_type)
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", default_event_type)
    )
    out["effective_date"] = _coerce_date(out["effective_date"])
    out["announcement_date"] = _coerce_date(out["announcement_date"])
    out["valid_from"] = _coerce_date(out["valid_from"])
    out["valid_to"] = _coerce_date(out["valid_to"])
    out["source_tier"] = out["source_tier"].fillna(source_tier).astype(str).str.strip().str.lower()
    out["source_name"] = out["source_name"].fillna(source_name).astype(str).str.strip()
    out["source_url"] = out["source_url"].fillna("").astype(str).str.strip()
    out["source_doc_id"] = out["source_doc_id"].fillna("").astype(str).str.strip()
    out["evidence_text"] = out["evidence_text"].fillna("").astype(str).str.strip()
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    out["mapping_note"] = out["mapping_note"].fillna("").astype(str).str.strip()

    # Accept event rows with either explicit interval(valid_from) or effective_date.
    mask_valid = out["valid_from"].notna() | out["effective_date"].notna()
    out = out.loc[mask_valid].copy()
    return out[SP500_EVENT_COLUMNS].reset_index(drop=True)


@dataclass
class BaseSP500ConstituentsProvider:
    name: str
    source_tier: str

    def fetch_events(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError

    def _read_csv_if_exists(self, path: str | Path) -> pd.DataFrame:
        p = Path(path).expanduser()
        if not p.exists():
            return pd.DataFrame(columns=SP500_EVENT_COLUMNS)
        try:
            raw = pd.read_csv(p)
        except Exception:
            return pd.DataFrame(columns=SP500_EVENT_COLUMNS)
        return normalize_sp500_event_frame(
            raw,
            source_tier=self.source_tier,
            source_name=self.name,
        )

