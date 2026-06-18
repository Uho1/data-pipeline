"""Schemas for valuation tab endpoint."""
from __future__ import annotations

from pydantic import BaseModel


class ValuationSeriesPoint(BaseModel):
    x: str
    y: float | None = None


class ValuationChartSeries(BaseModel):
    key: str = ""
    name: str
    type: str  # line | bar | scatter
    yAxis: str  # left | right
    dashed: bool = False
    data: list[ValuationSeriesPoint]


class ValuationChartMeta(BaseModel):
    title: str
    unit_left: str = ""
    unit_right: str = ""
    notes: str = ""


class ValuationChartPayload(BaseModel):
    meta: ValuationChartMeta
    series: list[ValuationChartSeries]
    missing_reason: str | None = None


class ValuationTable(BaseModel):
    rows: list[dict[str, float | str | None]]


class ValuationTables(BaseModel):
    band: ValuationTable
    value: ValuationTable
    per_share: ValuationTable


class ValuationTabResponse(BaseModel):
    ticker: str
    window: str
    basis: str
    periods: list[str]
    tables: ValuationTables
    charts: dict[str, ValuationChartPayload]
