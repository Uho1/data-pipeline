"""Schemas for balance-sheet specific financial tab endpoint."""
from __future__ import annotations

from pydantic import BaseModel


class BalanceSeriesPoint(BaseModel):
    x: str
    y: float | None = None


class BalanceChartSeries(BaseModel):
    key: str = ""
    name: str
    type: str  # line | bar | stackedBar
    yAxis: str  # left | right
    dashed: bool = False
    data: list[BalanceSeriesPoint]


class BalanceChartMeta(BaseModel):
    title: str
    unit_left: str = ""
    unit_right: str = ""
    notes: str = ""


class BalanceChartPayload(BaseModel):
    meta: BalanceChartMeta
    series: list[BalanceChartSeries]
    missing_reason: str | None = None


class BalanceKpiRow(BaseModel):
    period: str
    assets: float | None = None
    assets_chg_pct: float | None = None
    equity: float | None = None
    equity_chg_pct: float | None = None
    liabilities: float | None = None
    liabilities_chg_pct: float | None = None
    debt_ratio: float | None = None
    debt_ratio_chg_pct: float | None = None


class BalanceKpiTable(BaseModel):
    kpis: list[BalanceKpiRow]


class FinancialsBalanceResponse(BaseModel):
    ticker: str
    window: str
    basis: str
    periods: list[str]
    table: BalanceKpiTable
    charts: dict[str, BalanceChartPayload]
