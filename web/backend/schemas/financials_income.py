"""Schemas for income-statement specific financial tab endpoint."""
from __future__ import annotations

from pydantic import BaseModel


class IncomeSeriesPoint(BaseModel):
    x: str
    y: float | None = None


class IncomeChartSeries(BaseModel):
    key: str = ""
    name: str
    type: str  # line | bar | stackedBar
    yAxis: str  # left | right
    dashed: bool = False
    data: list[IncomeSeriesPoint]


class IncomeChartMeta(BaseModel):
    title: str
    unit_left: str = ""
    unit_right: str = ""
    notes: str = ""


class IncomeChartPayload(BaseModel):
    meta: IncomeChartMeta
    series: list[IncomeChartSeries]
    missing_reason: str | None = None


class IncomeKpiRow(BaseModel):
    period: str
    revenue: float | None = None
    revenue_chg_pct: float | None = None
    op_income: float | None = None
    op_income_chg_pct: float | None = None
    net_income: float | None = None
    net_income_chg_pct: float | None = None


class IncomeKpiTable(BaseModel):
    kpis: list[IncomeKpiRow]


class FinancialsIncomeResponse(BaseModel):
    ticker: str
    window: str
    basis: str
    periods: list[str]
    table: IncomeKpiTable
    charts: dict[str, IncomeChartPayload]
