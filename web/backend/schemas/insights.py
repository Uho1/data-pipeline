from __future__ import annotations
from typing import Any
from pydantic import BaseModel
from web.backend.schemas.fundamentals import FundamentalsChartPayload

class InsightsResponse(BaseModel):
    ticker: str
    window: str
    basis: str
    periods: list[str]
    charts: dict[str, FundamentalsChartPayload]
