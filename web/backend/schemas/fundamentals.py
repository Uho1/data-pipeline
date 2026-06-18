"""Schemas for fundamentals tab endpoint."""
from __future__ import annotations

from pydantic import BaseModel


class FundamentalsSeriesPoint(BaseModel):
    x: str | float
    y: float | None = None
    label: str | None = None


class FundamentalsChartSeries(BaseModel):
    key: str = ""
    name: str
    type: str  # line | bar | stackedBar | scatter
    yAxis: str  # left | right
    dashed: bool = False
    data: list[FundamentalsSeriesPoint]


class FundamentalsChartMeta(BaseModel):
    title: str
    unit_left: str = ""
    unit_right: str = ""
    notes: str = ""


class FundamentalsChartPayload(BaseModel):
    meta: FundamentalsChartMeta
    series: list[FundamentalsChartSeries]
    missing_reason: str | None = None


class FundamentalsTable(BaseModel):
    rows: list[dict[str, float | str | None]]


class FundamentalsTables(BaseModel):
    profitability: FundamentalsTable
    growth: FundamentalsTable
    stability: FundamentalsTable
    efficiency: FundamentalsTable


class FundamentalsResponse(BaseModel):
    ticker: str
    window: str
    basis: str
    periods: list[str]
    tables: FundamentalsTables
    charts: dict[str, FundamentalsChartPayload]
