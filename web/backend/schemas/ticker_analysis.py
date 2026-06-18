"""Schemas for tab-level ticker analysis endpoints (Phase 2)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SeriesPoint(BaseModel):
    date: str
    value: float | None = None


class SnapshotInfo(BaseModel):
    period_end: str | None = None
    available_date: str | None = None


class TabMeta(BaseModel):
    units: dict[str, str]
    notes: dict[str, str]
    missing: dict[str, str]


class TickerTabResponse(BaseModel):
    ticker: str
    asof: str
    snapshot: SnapshotInfo
    series: dict[str, list[SeriesPoint]]
    meta: TabMeta
    available_snapshot: dict[str, Any] | None = None
    business: dict[str, Any] | None = None
    product: dict[str, Any] | None = None
    geography: dict[str, Any] | None = None
    missing: dict[str, str] | None = None
    # optional diagnostics
    extra: dict[str, Any] | None = None
