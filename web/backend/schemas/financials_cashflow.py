"""Schemas for cashflow-statement specific financial tab endpoint."""
from __future__ import annotations

from pydantic import BaseModel


class CashflowSeriesPoint(BaseModel):
    x: str
    y: float | None = None


class CashflowChartSeries(BaseModel):
    key: str = ""
    name: str
    type: str  # line | bar | stackedBar
    yAxis: str  # left | right
    dashed: bool = False
    data: list[CashflowSeriesPoint]


class CashflowChartMeta(BaseModel):
    title: str
    unit_left: str = ""
    unit_right: str = ""
    notes: str = ""


class CashflowChartPayload(BaseModel):
    meta: CashflowChartMeta
    series: list[CashflowChartSeries]
    missing_reason: str | None = None


class CashflowKpiRow(BaseModel):
    period: str
    cfo: float | None = None
    cfo_chg_pct: float | None = None
    capex: float | None = None
    capex_chg_pct: float | None = None
    fcf: float | None = None
    fcf_chg_pct: float | None = None


class CashflowKpiTable(BaseModel):
    kpis: list[CashflowKpiRow]


class FinancialsCashflowResponse(BaseModel):
    ticker: str
    window: str
    basis: str
    periods: list[str]
    table: CashflowKpiTable
    charts: dict[str, CashflowChartPayload]
