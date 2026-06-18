"""Service layer for financial statement and fundamental analysis views.

Delegates computation to the existing gui.py pure-data functions.
Tkinter is NOT imported at module level in gui.py, so this import is safe.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

# Lazy import: gui.py pulls in matplotlib/backtest which are not available in deploy.
# These are only needed if the legacy /financials/view endpoint is actually called.
def _lazy_gui():
    from market_data.gui import (
        _build_financial_view_frame,
        _build_fundamental_view_frame,
        _chart_specs_for_statement,
        _fundamental_chart_specs,
        _parse_financial_bound,
    )
    return (
        _build_financial_view_frame,
        _build_fundamental_view_frame,
        _chart_specs_for_statement,
        _fundamental_chart_specs,
        _parse_financial_bound,
    )
from web.backend.schemas.financials import (
    ChartDataset,
    ChartSpec,
    ControlsEcho,
    FinancialsViewResponse,
    FundamentalsViewResponse,
    KpiTable,
    MetaInfo,
    SeriesSpec,
    UIConfigOption,
    UIConfigResponse,
)

# ---------------------------------------------------------------------------
# Constants / UI metadata
# ---------------------------------------------------------------------------

_STATEMENTS = [
    UIConfigOption(value="is", label="손익계산서"),
    UIConfigOption(value="bs", label="재무상태표"),
    UIConfigOption(value="cf", label="현금흐름표"),
]
_CATEGORIES = [
    UIConfigOption(value="profit",     label="수익성"),
    UIConfigOption(value="growth",     label="성장성"),
    UIConfigOption(value="stability",  label="안정성"),
    UIConfigOption(value="efficiency", label="효율성"),
    UIConfigOption(value="valuation",  label="밸류에이션"),
]
_MODES = [
    UIConfigOption(value="ttm",     label="TTM"),
    UIConfigOption(value="quarter", label="분기"),
    UIConfigOption(value="year",    label="연간"),
]
_HORIZONS = [
    UIConfigOption(value="all", label="전체"),
    UIConfigOption(value="10y", label="10년"),
    UIConfigOption(value="5y",  label="5년"),
]
_MARKETS = [
    UIConfigOption(value="auto", label="Auto"),
    UIConfigOption(value="us",   label="US"),
    UIConfigOption(value="kr",   label="KR"),
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _format_hint(row: str) -> str:
    """Return 'percent' for YoY / ratio / margin rows, otherwise 'currency'."""
    pct_keywords = ("yoy", "YoY", "률", "비율", "비중", "margin", "roe", "roa",
                    "ROE", "ROA", "PER", "PBR", "PSR")
    if any(kw in row for kw in pct_keywords):
        return "percent"
    return "currency"


def _build_kpi_response(kpi_df: pd.DataFrame) -> KpiTable:
    if kpi_df.empty:
        return KpiTable(rows=[], columns=[], data=[], format_hints={})

    rows = kpi_df.index.tolist()
    columns = kpi_df.columns.tolist()
    data: list[list[float | None]] = []
    for row in rows:
        row_vals: list[float | None] = []
        for col in columns:
            row_vals.append(_safe_float(kpi_df.loc[row, col]))
        data.append(row_vals)

    format_hints = {r: _format_hint(r) for r in rows}
    return KpiTable(rows=rows, columns=columns, data=data, format_hints=format_hints)


def _build_chart_spec(spec: dict) -> ChartSpec:
    def _series(items: list) -> list[SeriesSpec]:
        return [
            SeriesSpec(label=lbl, col=col, chart_type=ct, color=clr)
            for lbl, col, ct, clr in items
        ]

    return ChartSpec(
        id=spec["id"],
        section=spec["section"],
        title=spec["title"],
        left=_series(spec["left"]),
        right=_series(spec["right"]),
        left_percent=bool(spec["left_percent"]),
        right_percent=bool(spec["right_percent"]),
    )


def _build_dataset(df: pd.DataFrame, cols: set[str]) -> ChartDataset:
    labels: list[str] = (
        df["label"].astype(str).tolist()
        if "label" in df.columns
        else [str(i) for i in df.index]
    )
    series: dict[str, list[float | None]] = {}
    for col in cols:
        if col not in df.columns:
            series[col] = [None] * len(df)
        else:
            series[col] = [_safe_float(v) for v in pd.to_numeric(df[col], errors="coerce")]
    return ChartDataset(labels=labels, series=series)


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

def get_financials_view(
    ticker: str,
    market: str,
    statement: str,
    mode: str,
    horizon: str,
    start: str | None,
    end: str | None,
    offline_mode: bool,
) -> FinancialsViewResponse:
    (
        _build_financial_view_frame,
        _build_fundamental_view_frame,
        _chart_specs_for_statement,
        _fundamental_chart_specs,
        _parse_financial_bound,
    ) = _lazy_gui()
    start_bound = _parse_financial_bound(start, is_end=False) if start else None
    end_bound = _parse_financial_bound(end, is_end=True) if end else None

    warnings: list[str] = []
    try:
        datasets, kpi_df = _build_financial_view_frame(
            ticker=ticker,
            market=market,
            start_bound=start_bound,
            end_bound=end_bound,
            statement=statement,
            mode=mode,
            horizon=horizon,
            offline_mode=offline_mode,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"데이터 로드 실패: {exc}")
        datasets, kpi_df = {}, pd.DataFrame()

    if not datasets:
        warnings.append(f"데이터 없음: {ticker.upper()}")

    specs = _chart_specs_for_statement(statement)
    chart_specs = [_build_chart_spec(s) for s in specs]
    chart_datasets: dict[str, ChartDataset] = {}
    for spec in specs:
        sid = spec["id"]
        if sid in datasets:
            cols = {col for _, col, _, _ in spec["left"] + spec["right"]}
            chart_datasets[sid] = _build_dataset(datasets[sid], cols)

    return FinancialsViewResponse(
        meta=MetaInfo(
            ticker=ticker.upper(),
            resolved_market=market,
            loaded_at=datetime.now().isoformat(),
            warnings=warnings,
        ),
        controls=ControlsEcho(
            statement=statement,
            mode=mode,
            horizon=horizon,
            start=start,
            end=end,
            offline_mode=offline_mode,
        ),
        kpi_table=_build_kpi_response(kpi_df),
        chart_specs=chart_specs,
        chart_datasets=chart_datasets,
    )


def get_fundamentals_view(
    ticker: str,
    market: str,
    category: str,
    mode: str,
    horizon: str,
    start: str | None,
    end: str | None,
    offline_mode: bool,
) -> FundamentalsViewResponse:
    (
        _build_financial_view_frame,
        _build_fundamental_view_frame,
        _chart_specs_for_statement,
        _fundamental_chart_specs,
        _parse_financial_bound,
    ) = _lazy_gui()
    start_bound = _parse_financial_bound(start, is_end=False) if start else None
    end_bound = _parse_financial_bound(end, is_end=True) if end else None

    warnings: list[str] = []
    try:
        datasets, kpi_df = _build_fundamental_view_frame(
            ticker=ticker,
            market=market,
            start_bound=start_bound,
            end_bound=end_bound,
            category=category,
            mode=mode,
            horizon=horizon,
            offline_mode=offline_mode,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"데이터 로드 실패: {exc}")
        datasets, kpi_df = {}, pd.DataFrame()

    if not datasets:
        warnings.append(f"데이터 없음: {ticker.upper()}")

    specs = _fundamental_chart_specs(category)
    chart_specs = [_build_chart_spec(s) for s in specs]
    chart_datasets: dict[str, ChartDataset] = {}
    for spec in specs:
        sid = spec["id"]
        if sid in datasets:
            cols = {col for _, col, _, _ in spec["left"] + spec["right"]}
            chart_datasets[sid] = _build_dataset(datasets[sid], cols)

    return FundamentalsViewResponse(
        meta=MetaInfo(
            ticker=ticker.upper(),
            resolved_market=market,
            loaded_at=datetime.now().isoformat(),
            warnings=warnings,
        ),
        controls=ControlsEcho(
            category=category,
            mode=mode,
            horizon=horizon,
            start=start,
            end=end,
            offline_mode=offline_mode,
        ),
        kpi_table=_build_kpi_response(kpi_df),
        chart_specs=chart_specs,
        chart_datasets=chart_datasets,
    )


def get_ui_config() -> UIConfigResponse:
    return UIConfigResponse(
        statements=_STATEMENTS,
        categories=_CATEGORIES,
        modes=_MODES,
        horizons=_HORIZONS,
        markets=_MARKETS,
    )
