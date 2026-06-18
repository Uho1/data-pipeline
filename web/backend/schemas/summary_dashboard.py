from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SummaryDashboardResponse(BaseModel):
    ticker: str
    asof: str | None = None
    window: dict[str, int]
    snapshot: dict[str, str | None]
    charts: dict[str, Any]
