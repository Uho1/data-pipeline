"""Pydantic schemas for Financial Statement and Fundamental Analysis endpoints."""
from __future__ import annotations

from pydantic import BaseModel


class SeriesSpec(BaseModel):
    label: str       # Korean display label
    col: str         # data column name
    chart_type: str  # "bar" or "line"
    color: str       # hex color string


class ChartSpec(BaseModel):
    id: str
    section: str
    title: str
    left: list[SeriesSpec]
    right: list[SeriesSpec]
    left_percent: bool
    right_percent: bool


class KpiTable(BaseModel):
    rows: list[str]                        # row labels (Korean metric names)
    columns: list[str]                     # period labels (e.g. "24Q1")
    data: list[list[float | None]]         # rows × columns
    format_hints: dict[str, str]           # row_name -> "currency" | "percent"


class ChartDataset(BaseModel):
    labels: list[str]                      # period labels in chronological order
    series: dict[str, list[float | None]]  # col_name -> values array


class MetaInfo(BaseModel):
    ticker: str
    resolved_market: str
    loaded_at: str
    warnings: list[str]


class ControlsEcho(BaseModel):
    statement: str | None = None
    category: str | None = None
    mode: str
    horizon: str
    start: str | None
    end: str | None
    offline_mode: bool


class FinancialsViewResponse(BaseModel):
    meta: MetaInfo
    controls: ControlsEcho
    kpi_table: KpiTable
    chart_specs: list[ChartSpec]
    chart_datasets: dict[str, ChartDataset]


class FundamentalsViewResponse(BaseModel):
    meta: MetaInfo
    controls: ControlsEcho
    kpi_table: KpiTable
    chart_specs: list[ChartSpec]
    chart_datasets: dict[str, ChartDataset]


class UIConfigOption(BaseModel):
    value: str
    label: str


class UIConfigResponse(BaseModel):
    statements: list[UIConfigOption]
    categories: list[UIConfigOption]
    modes: list[UIConfigOption]
    horizons: list[UIConfigOption]
    markets: list[UIConfigOption]
