from __future__ import annotations

from dataclasses import dataclass
import re
import threading

import numpy as np
import pandas as pd

from market_data.sec_term_reader import rebuild_ticker_quarterly_cache, load_ticker_quarterly_cache
from market_data.reader import load_price_dataframe
from market_data.ui_style import (
    BORDER,
    PANEL_BG,
    PRIMARY,
    TEXT_MUTED,
    TEXT_SUB,
    UI_BG,
    apply_theme as _apply_theme,
    configure_ttk_style as _configure_ttk_style,
    create_header_toggle,
    create_segmented_control,
    get_palette,
)
from market_data.valuation import load_valuation_series
from market_data.chart_style import (
    CARD_CHART_MIN_HEIGHT_PX,
    CARD_GAP_PX,
    CARD_MIN_HEIGHT_PX,
    add_axis_unit_label,
    apply_chart_style,
    apply_matplotlib_theme as _configure_matplotlib_theme,
    apply_secondary_axis_style,
    format_legend as _format_chart_legend,
    get_figure_size_px,
    get_layout_mode,
    make_figure_for_card,
)
from market_data.backtest.simulator import run_screen_backtest, run_strategy_backtest_from_config
from market_data.backtest.ui_backtest import mount_backtest_tab


@dataclass
class GuiState:
    ticker: str = "AAPL"
    market: str = "us"
    chart_type: str = "candles"
    offline_mode: bool = True


FINANCIAL_MIN_DATE = pd.Timestamp("2000-01-01")

def _series_from_candidates(df: pd.DataFrame | None, candidates: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    if "StatementDate" not in df.columns:
        return None

    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() <= 0:
            continue
        out = pd.Series(values.to_numpy(dtype=float), index=df["StatementDate"])
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out[~out.index.duplicated(keep="last")]
        out = out.sort_index()
        return out
    return None

def _quarter_label(dt: pd.Timestamp) -> str:
    q = int(((int(dt.month) - 1) // 3) + 1)
    yy = int(dt.year) % 100
    return f"{yy:02d}Q{q}"

def _choose_scale(values: np.ndarray) -> tuple[float, str]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0, ""
    max_abs = float(np.max(np.abs(finite)))
    if max_abs >= 1_000_000_000_000:
        return 1_000_000_000_000.0, "T"
    if max_abs >= 1_000_000_000:
        return 1_000_000_000.0, "B"
    if max_abs >= 1_000_000:
        return 1_000_000.0, "M"
    if max_abs >= 1_000:
        return 1_000.0, "K"
    return 1.0, ""


def _parse_financial_bound(raw: str, is_end: bool) -> pd.Timestamp | None:
    text = str(raw or "").strip().upper()
    if not text:
        return None

    q_match = re.fullmatch(r"(\d{4})[- ]?Q([1-4])", text)
    if q_match:
        year = int(q_match.group(1))
        q = int(q_match.group(2))
        p = pd.Period(f"{year}Q{q}", freq="Q")
        return p.to_timestamp(how="end").normalize() if is_end else p.to_timestamp(how="start").normalize()

    y_match = re.fullmatch(r"\d{4}", text)
    if y_match:
        year = int(text)
        return pd.Timestamp(year=year, month=12, day=31) if is_end else pd.Timestamp(year=year, month=1, day=1)

    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Invalid range value: {raw}")
    return pd.Timestamp(dt).normalize()


_FLOW_METRIC_COLUMNS = [
    "revenue",
    "operating_income",
    "net_income",
    "cogs",
    "sga",
    "gross_profit",
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
]
_STOCK_METRIC_COLUMNS = [
    "equity",
    "liabilities",
    "total_assets",
    "shares",
    "price",
    "market_cap",
    "debt_ratio",
]
_YOY_TARGET_COLUMNS = [
    "revenue",
    "operating_income",
    "net_income",
    "gross_profit",
    "market_cap",
    "price",
    "operating_cash_flow",
    "free_cash_flow",
]


def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = pd.to_numeric(numer, errors="coerce") / pd.to_numeric(denom, errors="coerce").replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _pct_change_percent(series: pd.Series, periods: int) -> pd.Series:
    base = pd.to_numeric(series, errors="coerce")
    prev = base.shift(periods)
    out = ((base / prev) - 1.0) * 100.0
    return out.replace([np.inf, -np.inf], np.nan)


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _recompute_financial_metrics(frame: pd.DataFrame, yoy_lag: int) -> pd.DataFrame:
    out = frame.copy()
    out = _ensure_columns(out, _FLOW_METRIC_COLUMNS + _STOCK_METRIC_COLUMNS)

    out["market_cap"] = pd.to_numeric(out["shares"], errors="coerce") * pd.to_numeric(out["price"], errors="coerce")
    out["market_cap"] = out["market_cap"].replace([np.inf, -np.inf], np.nan)

    out["gross_profit"] = pd.to_numeric(out["revenue"], errors="coerce") - pd.to_numeric(out["cogs"], errors="coerce")
    out["debt_ratio"] = _safe_divide(out["liabilities"], out["equity"]) * 100.0
    out["gross_margin"] = _safe_divide(out["gross_profit"], out["revenue"]) * 100.0
    out["operating_margin"] = _safe_divide(out["operating_income"], out["revenue"]) * 100.0
    out["net_margin"] = _safe_divide(out["net_income"], out["revenue"]) * 100.0
    out["cogs_ratio"] = _safe_divide(out["cogs"], out["revenue"]) * 100.0
    out["sga_ratio"] = _safe_divide(out["sga"], out["revenue"]) * 100.0
    out["free_cash_flow"] = pd.to_numeric(out["operating_cash_flow"], errors="coerce") + pd.to_numeric(
        out["capital_expenditure"], errors="coerce"
    )

    for col in _YOY_TARGET_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
        out[f"{col}_yoy"] = _pct_change_percent(out[col], periods=max(int(yoy_lag), 1))

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _prepare_financial_quarterly_frame(term_df: pd.DataFrame | None) -> pd.DataFrame:
    if term_df is None or term_df.empty or "StatementDate" not in term_df.columns:
        return pd.DataFrame()

    prepared = term_df.copy()
    prepared["StatementDate"] = pd.to_datetime(prepared["StatementDate"], errors="coerce")
    prepared = prepared.loc[~prepared["StatementDate"].isna()].sort_values("StatementDate")
    prepared = prepared.loc[prepared["StatementDate"] >= FINANCIAL_MIN_DATE]
    prepared = prepared.drop_duplicates(subset=["StatementDate"], keep="last")
    if prepared.empty:
        return pd.DataFrame()

    base_index = pd.DatetimeIndex(prepared["StatementDate"]).sort_values().drop_duplicates()
    out = pd.DataFrame(index=base_index)
    series_candidates: dict[str, list[str]] = {
        "revenue": ["Revenue", "Total Revenue", "Operating Revenue"],
        "operating_income": ["Operating Income", "EBIT", "Total Operating Income As Reported"],
        "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"],
        "cogs": ["COGS", "Cost Of Revenue", "Reconciled Cost Of Revenue"],
        "sga": [
            "SG&A",
            "Selling General And Administration",
            "Selling And Marketing Expense",
            "General And Administrative Expense",
        ],
        "equity": [
            "Shareholders Equity",
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        ],
        "liabilities": [
            "Total Liabilities",
            "Total Liabilities Net Minority Interest",
            "Current Liabilities",
        ],
        "total_assets": ["Total Assets", "Total Assets As Reported", "Total Assets Gross Minority Interest"],
        "shares": ["Shares", "Ordinary Shares Number", "Diluted Average Shares", "Basic Average Shares", "Share Issued"],
        "price": ["Price"],
        "operating_cash_flow": [
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
            "Net Cash Provided By Operating Activities",
        ],
        "investing_cash_flow": [
            "Investing Cash Flow",
            "Cash Flow From Continuing Investing Activities",
            "Net Cash Provided By Investing Activities",
        ],
        "financing_cash_flow": [
            "Financing Cash Flow",
            "Cash Flow From Continuing Financing Activities",
            "Net Cash Provided By Financing Activities",
        ],
        "capital_expenditure": ["Capital Expenditure", "Capital Expenditure Reported"],
    }
    for key, candidates in series_candidates.items():
        values = _series_from_candidates(prepared, candidates)
        out[key] = values.reindex(out.index) if values is not None else np.nan

    out = _recompute_financial_metrics(out, yoy_lag=4)
    out["label"] = out.index.to_series().map(_quarter_label).to_numpy()
    return out


def _convert_financial_mode(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    mode_key = str(mode).strip().lower()
    prepared = frame.sort_index().copy()
    prepared = _ensure_columns(prepared, _FLOW_METRIC_COLUMNS + _STOCK_METRIC_COLUMNS)

    if mode_key == "ttm":
        out = prepared.copy()
        for col in _FLOW_METRIC_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").rolling(4, min_periods=4).sum()
        out = _recompute_financial_metrics(out, yoy_lag=4)
        out["label"] = out.index.to_series().map(_quarter_label).to_numpy()
        return out

    if mode_key == "year":
        year_idx = pd.Index(prepared.index.year, name="year")
        flow = prepared[_FLOW_METRIC_COLUMNS].groupby(year_idx).sum(min_count=1)
        stock = prepared[_STOCK_METRIC_COLUMNS].groupby(year_idx).last()
        out = flow.join(stock, how="outer")
        out.index = pd.DatetimeIndex([pd.Timestamp(year=int(y), month=12, day=31) for y in out.index])
        out = out.sort_index()
        out = _recompute_financial_metrics(out, yoy_lag=1)
        out["label"] = out.index.to_series().dt.year.astype(str).to_numpy()
        return out

    out = _recompute_financial_metrics(prepared, yoy_lag=4)
    out["label"] = out.index.to_series().map(_quarter_label).to_numpy()
    return out


def _apply_financial_horizon(frame: pd.DataFrame, mode: str, horizon: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    mode_key = str(mode).strip().lower()
    horizon_key = str(horizon).strip().lower()

    if horizon_key in {"all", "full", "전체"}:
        return frame

    if mode_key == "year":
        limit = 10 if horizon_key == "10y" else 5
    else:
        limit = 40 if horizon_key == "10y" else 20
    return frame.tail(max(int(limit), 1))


def _build_kpi_table(frame: pd.DataFrame, max_periods: int = 5) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    subset = frame.tail(max(int(max_periods), 1)).iloc[::-1]
    labels = subset["label"].astype(str).tolist() if "label" in subset.columns else [str(i) for i in subset.index]
    out = pd.DataFrame(index=["매출액", "매출액 YoY", "영업이익", "영업이익 YoY", "순이익", "순이익 YoY"], columns=labels, dtype=float)
    metric_map = [
        ("매출액", "revenue"),
        ("매출액 YoY", "revenue_yoy"),
        ("영업이익", "operating_income"),
        ("영업이익 YoY", "operating_income_yoy"),
        ("순이익", "net_income"),
        ("순이익 YoY", "net_income_yoy"),
    ]
    for label, col in metric_map:
        if col not in subset.columns:
            out.loc[label, :] = np.nan
            continue
        values = pd.to_numeric(subset[col], errors="coerce").to_numpy(dtype=float)
        out.loc[label, :] = values
    return out


def _chart_specs_for_statement(statement: str) -> list[dict]:
    st = str(statement).strip().lower()
    if st == "bs":
        return [
            {
                "id": "bs_overview",
                "section": "재무상태",
                "title": "재무상태 종합",
                "left": [("자본총계", "equity", "bar", "#7ea1e3"), ("부채총계", "liabilities", "bar", "#6fb38f")],
                "right": [("부채비율(우)", "debt_ratio", "line", "#f0c44c")],
                "left_percent": False,
                "right_percent": True,
            },
            {
                "id": "bs_market",
                "section": "재무상태",
                "title": "시가총액/주가",
                "left": [("시가총액", "market_cap", "line", "#6f7487")],
                "right": [("주가(우)", "price", "line", "#b291d2")],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "bs_equity",
                "section": "재무상태",
                "title": "자본총계",
                "left": [("자본총계", "equity", "bar", "#7ea1e3")],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "bs_liabilities",
                "section": "재무상태",
                "title": "부채총계",
                "left": [("부채총계", "liabilities", "bar", "#6fb38f")],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "bs_debt_ratio",
                "section": "재무상태",
                "title": "부채비율",
                "left": [("부채비율", "debt_ratio", "line", "#f0c44c")],
                "right": [],
                "left_percent": True,
                "right_percent": False,
            },
            {
                "id": "bs_value_mix",
                "section": "재무상태",
                "title": "자본/부채 구성비",
                "left": [("자본비중", "equity_ratio", "line", "#7ea1e3"), ("부채비중", "liability_ratio", "line", "#6fb38f")],
                "right": [],
                "left_percent": True,
                "right_percent": False,
            },
        ]

    if st == "cf":
        return [
            {
                "id": "cf_overview",
                "section": "현금흐름",
                "title": "현금흐름",
                "left": [
                    ("FCF", "free_cash_flow", "bar", "#6a8fe0"),
                    ("영업현금흐름", "operating_cash_flow", "line", "#e889a2"),
                    ("투자현금흐름", "investing_cash_flow", "line", "#e7bf58"),
                    ("재무현금흐름", "financing_cash_flow", "line", "#a3c92f"),
                ],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "cf_vs_earnings",
                "section": "현금흐름",
                "title": "현금흐름 vs 실적",
                "left": [
                    ("영업현금흐름", "operating_cash_flow", "line", "#e889a2"),
                    ("영업이익", "operating_income", "line", "#e7bf58"),
                    ("순이익", "net_income", "line", "#a3c92f"),
                ],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "cf_fcf",
                "section": "현금흐름 상세",
                "title": "잉여현금흐름",
                "left": [("잉여현금흐름", "free_cash_flow", "line", "#6a8fe0")],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "cf_ocf",
                "section": "현금흐름 상세",
                "title": "영업현금흐름",
                "left": [("영업현금흐름", "operating_cash_flow", "bar", "#6aa6c8")],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "cf_icf",
                "section": "현금흐름 상세",
                "title": "투자현금흐름",
                "left": [("투자현금흐름", "investing_cash_flow", "bar", "#ef8a62")],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "cf_fincf",
                "section": "현금흐름 상세",
                "title": "재무현금흐름",
                "left": [("재무현금흐름", "financing_cash_flow", "bar", "#9b8fcb")],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "cf_capex",
                "section": "현금흐름 상세",
                "title": "자본적지출(CAPEX)",
                "left": [("CAPEX", "capital_expenditure", "bar", "#c9a227")],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
        ]

    return [
        {
            "id": "earnings_overview",
            "section": "실적",
            "title": "실적 종합",
            "left": [
                ("영업이익", "operating_income", "bar", "#dd6d87"),
                ("판관비", "sga", "bar", "#6a8fe0"),
                ("매출원가", "cogs", "bar", "#9abf38"),
                ("매출액", "revenue", "line", "#b291d2"),
            ],
            "right": [("시가총액(우)", "market_cap", "line", "#6f7487")],
            "left_percent": False,
            "right_percent": False,
        },
        {
            "id": "earnings_revenue",
            "section": "실적",
            "title": "매출액",
            "left": [("매출액", "revenue", "bar", "#b291d2")],
            "right": [],
            "left_percent": False,
            "right_percent": False,
        },
        {
            "id": "earnings_operating_income",
            "section": "실적",
            "title": "영업이익",
            "left": [("영업이익", "operating_income", "bar", "#dd6d87")],
            "right": [],
            "left_percent": False,
            "right_percent": False,
        },
        {
            "id": "earnings_net_income",
            "section": "실적",
            "title": "순이익",
            "left": [("순이익", "net_income", "bar", "#7b8fd6")],
            "right": [],
            "left_percent": False,
            "right_percent": False,
        },
        {
            "id": "revenue_mix",
            "section": "매출액",
            "title": "매출액 구성",
            "left": [
                ("매출총이익", "gross_profit", "bar", "#f3c969"),
                ("매출원가", "cogs", "bar", "#9abf38"),
                ("판관비", "sga", "bar", "#6a8fe0"),
            ],
            "right": [("매출액(우)", "revenue", "line", "#b291d2")],
            "left_percent": False,
            "right_percent": False,
        },
        {
            "id": "revenue_mix_ratio",
            "section": "매출액",
            "title": "매출액 구성비중",
            "left": [
                ("매출총이익률", "gross_margin", "line", "#f3c969"),
                ("매출원가율", "cogs_ratio", "line", "#9abf38"),
                ("판관비율", "sga_ratio", "line", "#6a8fe0"),
            ],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
        {
            "id": "revenue_growth",
            "section": "매출액",
            "title": "매출액 성장률(YoY)",
            "left": [("매출액 YoY", "revenue_yoy", "bar", "#f0c44c")],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
        {
            "id": "market_price",
            "section": "매출액",
            "title": "시가총액/주가",
            "left": [("시가총액", "market_cap", "line", "#6f7487")],
            "right": [("주가(우)", "price", "line", "#b291d2")],
            "left_percent": False,
            "right_percent": False,
        },
        {
            "id": "profit_margin",
            "section": "이익",
            "title": "이익률",
            "left": [
                ("매출총이익률", "gross_margin", "line", "#f3c969"),
                ("영업이익률", "operating_margin", "line", "#dd6d87"),
                ("순이익률", "net_margin", "line", "#7b8fd6"),
            ],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
        {
            "id": "profit_growth",
            "section": "이익",
            "title": "이익 성장률(YoY)",
            "left": [
                ("매출총이익 YoY", "gross_profit_yoy", "line", "#f3c969"),
                ("영업이익 YoY", "operating_income_yoy", "line", "#dd6d87"),
                ("순이익 YoY", "net_income_yoy", "line", "#7b8fd6"),
            ],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
        {
            "id": "gross_profit",
            "section": "이익",
            "title": "매출총이익",
            "left": [("매출총이익", "gross_profit", "bar", "#f3c969")],
            "right": [],
            "left_percent": False,
            "right_percent": False,
        },
        {
            "id": "gross_profit_growth",
            "section": "이익",
            "title": "매출총이익 성장률(YoY)",
            "left": [("매출총이익 YoY", "gross_profit_yoy", "line", "#f3c969")],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
        {
            "id": "op_income_detail",
            "section": "이익",
            "title": "영업이익",
            "left": [("영업이익", "operating_income", "bar", "#dd6d87")],
            "right": [],
            "left_percent": False,
            "right_percent": False,
        },
        {
            "id": "op_income_growth",
            "section": "이익",
            "title": "영업이익 성장률(YoY)",
            "left": [("영업이익 YoY", "operating_income_yoy", "line", "#dd6d87")],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
    ]


_FUNDAMENTAL_CATEGORY_LABELS = {
    "profit": "수익성",
    "growth": "성장성",
    "stability": "안정성",
    "efficiency": "효율성",
    "valuation": "밸류에이션",
}


def _fundamental_chart_specs(category: str) -> list[dict]:
    cat = str(category).strip().lower()
    if cat == "growth":
        return [
            {
                "id": "growth_main",
                "section": "성장성",
                "title": "성장률(YoY)",
                "left": [
                    ("매출 성장률", "revenue_yoy", "line", "#b291d2"),
                    ("영업이익 성장률", "operating_income_yoy", "line", "#dd6d87"),
                    ("순이익 성장률", "net_income_yoy", "line", "#9abf38"),
                ],
                "right": [],
                "left_percent": True,
                "right_percent": False,
            },
            {
                "id": "growth_cash",
                "section": "성장성",
                "title": "현금흐름 성장률(YoY)",
                "left": [
                    ("영업현금흐름 성장률", "operating_cash_flow_yoy", "line", "#6aa6c8"),
                    ("잉여현금흐름 성장률", "free_cash_flow_yoy", "line", "#0f766e"),
                ],
                "right": [],
                "left_percent": True,
                "right_percent": False,
            },
        ]

    if cat == "stability":
        return [
            {
                "id": "stability_debt",
                "section": "안정성",
                "title": "부채비율",
                "left": [("부채비율", "debt_ratio", "line", "#f0c44c")],
                "right": [],
                "left_percent": True,
                "right_percent": False,
            },
            {
                "id": "stability_mix",
                "section": "안정성",
                "title": "자본/부채 비중",
                "left": [("자본비중", "equity_ratio", "line", "#7ea1e3"), ("부채비중", "liability_ratio", "line", "#6fb38f")],
                "right": [],
                "left_percent": True,
                "right_percent": False,
            },
        ]

    if cat == "efficiency":
        return [
            {
                "id": "efficiency_turnover",
                "section": "효율성",
                "title": "자산회전율",
                "left": [("자산회전율", "asset_turnover", "line", "#6f7487")],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "efficiency_leverage",
                "section": "효율성",
                "title": "재무레버리지",
                "left": [("자산/자본", "equity_multiplier", "line", "#9b8fcb")],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
        ]

    if cat == "valuation":
        return [
            {
                "id": "valuation_per",
                "section": "밸류에이션",
                "title": "PER/PBR/PSR (Proxy)",
                "left": [
                    ("PER", "per_proxy", "line", "#6f7487"),
                    ("PBR", "pbr_proxy", "line", "#b291d2"),
                    ("PSR", "psr_proxy", "line", "#6aa6c8"),
                ],
                "right": [],
                "left_percent": False,
                "right_percent": False,
            },
            {
                "id": "valuation_market",
                "section": "밸류에이션",
                "title": "시가총액/주가",
                "left": [("시가총액", "market_cap", "line", "#6f7487")],
                "right": [("주가(우)", "price", "line", "#b291d2")],
                "left_percent": False,
                "right_percent": False,
            },
        ]

    # Default: profit
    return [
        {
            "id": "profit_roe",
            "section": "수익성",
            "title": "ROE",
            "left": [("ROE", "roe_pct", "line", "#8b7ad3")],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
        {
            "id": "profit_op_margin",
            "section": "수익성",
            "title": "영업이익률",
            "left": [("영업이익률", "operating_margin", "line", "#dd6d87")],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
        {
            "id": "profit_net_margin",
            "section": "수익성",
            "title": "순이익률",
            "left": [("순이익률", "net_margin", "line", "#9abf38")],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
        {
            "id": "profit_gross_margin",
            "section": "수익성",
            "title": "매출총이익률",
            "left": [("매출총이익률", "gross_margin", "line", "#f3c969")],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
        {
            "id": "profit_roa",
            "section": "수익성",
            "title": "ROA",
            "left": [("ROA", "roa_pct", "line", "#6aa6c8")],
            "right": [],
            "left_percent": True,
            "right_percent": False,
        },
    ]


def _series_or_nan(frame: pd.DataFrame, col: str) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _build_fundamental_metrics(view: pd.DataFrame, quarter_frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = view.copy()
    mode_key = str(mode).strip().lower()
    lag = 1 if mode_key == "year" else 4

    revenue = _series_or_nan(out, "revenue")
    op_income = _series_or_nan(out, "operating_income")
    net_income = _series_or_nan(out, "net_income")
    equity = _series_or_nan(out, "equity")
    liabilities = _series_or_nan(out, "liabilities")
    total_assets = _series_or_nan(out, "total_assets")
    assets_base = total_assets.copy()
    fallback_assets = (equity + liabilities).replace([np.inf, -np.inf], np.nan)
    assets_base = assets_base.where(assets_base.notna(), fallback_assets)

    # ROE/ROA in TTM uses average balance sheet denominator when possible.
    if mode_key == "ttm":
        q_equity = _series_or_nan(quarter_frame, "equity")
        q_liabilities = _series_or_nan(quarter_frame, "liabilities")
        q_total_assets = _series_or_nan(quarter_frame, "total_assets")
        q_assets_base = q_total_assets.where(q_total_assets.notna(), (q_equity + q_liabilities))

        avg_equity = q_equity.rolling(4, min_periods=4).mean().reindex(out.index)
        avg_assets = q_assets_base.rolling(4, min_periods=4).mean().reindex(out.index)
        roe_denom = avg_equity.where(avg_equity.notna(), equity)
        roa_denom = avg_assets.where(avg_assets.notna(), assets_base)
    else:
        roe_denom = equity
        roa_denom = assets_base

    out["roe_pct"] = _safe_divide(net_income, roe_denom) * 100.0
    out["roa_pct"] = _safe_divide(net_income, roa_denom) * 100.0
    out["operating_margin"] = _safe_divide(op_income, revenue) * 100.0
    out["net_margin"] = _safe_divide(net_income, revenue) * 100.0
    out["gross_margin"] = _safe_divide(_series_or_nan(out, "gross_profit"), revenue) * 100.0

    out["asset_turnover"] = _safe_divide(revenue, assets_base)
    out["equity_multiplier"] = _safe_divide(assets_base, equity)

    out["per_proxy"] = _safe_divide(_series_or_nan(out, "market_cap"), net_income)
    out["pbr_proxy"] = _safe_divide(_series_or_nan(out, "market_cap"), equity)
    out["psr_proxy"] = _safe_divide(_series_or_nan(out, "market_cap"), revenue)

    out["operating_cash_flow_yoy"] = _pct_change_percent(_series_or_nan(out, "operating_cash_flow"), periods=lag)
    out["free_cash_flow_yoy"] = _pct_change_percent(_series_or_nan(out, "free_cash_flow"), periods=lag)
    out["roe_pct_yoy"] = _pct_change_percent(out["roe_pct"], periods=lag)
    out["operating_margin_yoy"] = _pct_change_percent(out["operating_margin"], periods=lag)
    out["net_margin_yoy"] = _pct_change_percent(out["net_margin"], periods=lag)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _build_fundamental_kpi_table(frame: pd.DataFrame, category: str, max_periods: int = 5) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    subset = frame.tail(max(int(max_periods), 1)).iloc[::-1]
    labels = subset["label"].astype(str).tolist() if "label" in subset.columns else [str(i) for i in subset.index]

    cat = str(category).strip().lower()
    rows: list[tuple[str, str]] = [
        ("ROE", "roe_pct"),
        ("ROE YoY", "roe_pct_yoy"),
        ("영업이익률", "operating_margin"),
        ("영업이익률 YoY", "operating_margin_yoy"),
        ("순이익률", "net_margin"),
        ("순이익률 YoY", "net_margin_yoy"),
    ]
    if cat == "growth":
        rows = [
            ("매출 성장률", "revenue_yoy"),
            ("영업이익 성장률", "operating_income_yoy"),
            ("순이익 성장률", "net_income_yoy"),
            ("영업현금흐름 성장률", "operating_cash_flow_yoy"),
            ("잉여현금흐름 성장률", "free_cash_flow_yoy"),
        ]
    elif cat == "stability":
        rows = [("부채비율", "debt_ratio"), ("자본비중", "equity_ratio"), ("부채비중", "liability_ratio")]
    elif cat == "efficiency":
        rows = [("자산회전율", "asset_turnover"), ("재무레버리지", "equity_multiplier")]
    elif cat == "valuation":
        rows = [("PER", "per_proxy"), ("PBR", "pbr_proxy"), ("PSR", "psr_proxy")]

    out = pd.DataFrame(index=[r[0] for r in rows], columns=labels, dtype=float)
    for label, col in rows:
        if col not in subset.columns:
            out.loc[label, :] = np.nan
            continue
        out.loc[label, :] = pd.to_numeric(subset[col], errors="coerce").to_numpy(dtype=float)
    return out


def _build_fundamental_view_frame(
    ticker: str,
    market: str,
    start_bound: pd.Timestamp | None = None,
    end_bound: pd.Timestamp | None = None,
    category: str = "profit",
    mode: str = "ttm",
    horizon: str = "all",
    offline_mode: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    _ = market
    term_df = load_ticker_quarterly_cache(ticker=ticker, rebuild_if_stale=not offline_mode)
    quarter = _prepare_financial_quarterly_frame(term_df)
    if quarter.empty:
        return {}, pd.DataFrame()

    view = _convert_financial_mode(quarter, mode=mode)
    view = _build_fundamental_metrics(view, quarter_frame=quarter, mode=mode)
    if view.empty:
        return {}, pd.DataFrame()

    view = view.loc[view.index >= FINANCIAL_MIN_DATE]
    if start_bound is not None:
        view = view.loc[view.index >= start_bound]
    if end_bound is not None:
        view = view.loc[view.index <= end_bound]
    view = _apply_financial_horizon(view, mode=mode, horizon=horizon)
    if view.empty:
        return {}, pd.DataFrame()

    view = view.sort_index()
    if mode == "year":
        view["label"] = view.index.to_series().dt.year.astype(str).to_numpy()
    else:
        view["label"] = view.index.to_series().map(_quarter_label).to_numpy()

    specs = _fundamental_chart_specs(category)
    datasets: dict[str, pd.DataFrame] = {}
    for spec in specs:
        cols = {col for _, col, _, _ in spec["left"] + spec["right"]}
        available_cols = [c for c in cols if c in view.columns]
        if not available_cols:
            continue
        card = view[["label"] + available_cols].copy()
        has_data = any(pd.to_numeric(card[c], errors="coerce").notna().sum() > 0 for c in available_cols)
        if not has_data:
            continue
        datasets[spec["id"]] = card
    datasets["base"] = view.copy()

    kpi_df = _build_fundamental_kpi_table(view, category=category, max_periods=5)
    return datasets, kpi_df


@dataclass
class ChartCard:
    section: str
    spec: dict
    data: pd.DataFrame


def build_two_column_grid(cards: list[ChartCard], columns: int = 2) -> list[dict]:
    rows: list[dict] = []
    if not cards:
        return rows
    cols = 1 if int(columns) <= 1 else 2

    current_section = ""
    bucket: list[ChartCard] = []
    for card in cards:
        if card.section != current_section:
            if bucket:
                for i in range(0, len(bucket), cols):
                    rows.append({"type": "cards", "cards": bucket[i : i + cols]})
                bucket = []
            current_section = card.section
            rows.append({"type": "section", "title": current_section})
        bucket.append(card)
    if bucket:
        for i in range(0, len(bucket), cols):
            rows.append({"type": "cards", "cards": bucket[i : i + cols]})
    return rows


def _build_financial_view_frame(
    ticker: str,
    market: str,
    start_bound: pd.Timestamp | None = None,
    end_bound: pd.Timestamp | None = None,
    statement: str = "is",
    mode: str = "ttm",
    horizon: str = "10y",
    offline_mode: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    _ = market
    term_df = load_ticker_quarterly_cache(ticker=ticker, rebuild_if_stale=not offline_mode)
    if str(statement).strip().lower() == "cf" and term_df is not None and not term_df.empty:
        cf_cols = ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Capital Expenditure"]
        cf_non_na = 0
        for col in cf_cols:
            if col not in term_df.columns:
                continue
            values = pd.to_numeric(term_df[col], errors="coerce")
            cf_non_na += int(values.notna().sum())
        if cf_non_na == 0:
            # Older ticker cache files may have empty CF columns; rebuild from raw terms once.
            term_df = rebuild_ticker_quarterly_cache(ticker=ticker, offline_mode=offline_mode)

    base_quarter = _prepare_financial_quarterly_frame(term_df)
    if base_quarter.empty:
        return {}, pd.DataFrame()

    view = _convert_financial_mode(base_quarter, mode=mode)
    view = view.loc[view.index >= FINANCIAL_MIN_DATE]
    if start_bound is not None:
        view = view.loc[view.index >= start_bound]
    if end_bound is not None:
        view = view.loc[view.index <= end_bound]
    view = _apply_financial_horizon(view, mode=mode, horizon=horizon)
    if view.empty:
        return {}, pd.DataFrame()

    view = view.sort_index()
    if mode == "year":
        view["label"] = view.index.to_series().dt.year.astype(str).to_numpy()
    else:
        view["label"] = view.index.to_series().map(_quarter_label).to_numpy()

    view["equity_ratio"] = _safe_divide(view["equity"], view["equity"] + view["liabilities"]) * 100.0
    view["liability_ratio"] = _safe_divide(view["liabilities"], view["equity"] + view["liabilities"]) * 100.0

    specs = _chart_specs_for_statement(statement)
    datasets: dict[str, pd.DataFrame] = {}
    for spec in specs:
        cols = {col for _, col, _, _ in spec["left"] + spec["right"]}
        cols_list = ["label"] + [c for c in cols if c in view.columns]
        datasets[spec["id"]] = view[cols_list].copy()
    datasets["base"] = view.copy()

    kpi_df = _build_kpi_table(view, max_periods=5)
    return datasets, kpi_df


def run_gui(
    initial_ticker: str | None = None,
    market: str | None = None,
    valuation_price_field: str = "adjclose",
    valuation_per_negative: str = "nan",
    valuation_band_window: str = "all",
    valuation_band_quantiles: str = "0.1,0.3,0.5,0.7,0.9",
    valuation_outlier: str = "none",
) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Tkinter is not available on this Python runtime") from exc

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch, Patch

    _configure_matplotlib_theme(plt)

    state = GuiState(
        ticker=initial_ticker or "AAPL",
        market=(market or "auto"),
        chart_type="candles",
        offline_mode=True,
    )
    valuation_price_field = (valuation_price_field or "adjclose").strip().lower()
    if valuation_price_field not in {"adjclose", "close"}:
        valuation_price_field = "adjclose"
    valuation_per_negative = (valuation_per_negative or "nan").strip().lower()
    if valuation_per_negative not in {"nan", "allow"}:
        valuation_per_negative = "nan"
    valuation_band_window = (valuation_band_window or "all").strip().lower()
    if valuation_band_window not in {"all", "10y", "5y"}:
        valuation_band_window = "all"
    valuation_band_quantiles = (valuation_band_quantiles or "0.1,0.3,0.5,0.7,0.9").strip()
    valuation_outlier = (valuation_outlier or "none").strip().lower()
    if valuation_outlier not in {"none", "winsorize-1-99"}:
        valuation_outlier = "none"

    root = tk.Tk()
    root.title("Market Data Viewer")
    root.geometry("1440x920")
    root.minsize(1100, 720)
    root.configure(bg=UI_BG)
    ui_font_family = _configure_ttk_style(root, ttk)

    # Global market and theme state (shared across all tabs)
    market_var = tk.StringVar(value=state.market if state.market in ("us", "kr") else "us")
    theme_var  = tk.StringVar(value="light")

    info_var = tk.StringVar(value="")

    # ------------------------------------------------------------------
    # App header bar
    # ------------------------------------------------------------------
    _header = tk.Frame(root, bg=PRIMARY, height=52)
    _header.pack(fill="x", side="top")
    _header.pack_propagate(False)

    _title_lbl = tk.Label(
        _header, text="  Market Data Viewer",
        bg=PRIMARY, fg="#ffffff",
        font=(ui_font_family, 15, "bold"), anchor="w",
    )
    _title_lbl.pack(side="left", fill="y", padx=(8, 0))

    _sub_lbl = tk.Label(
        _header, text="퀀트 분석 · 백테스트 플랫폼",
        bg=PRIMARY, fg="#c7d9ff",
        font=(ui_font_family, 10), anchor="w",
    )
    _sub_lbl.pack(side="left", fill="y", padx=(10, 0))

    _ver_lbl = tk.Label(
        _header, text="v0.1",
        bg=PRIMARY, fg="#93b4f3",
        font=(ui_font_family, 10), anchor="e",
    )
    _ver_lbl.pack(side="right", fill="y", padx=(0, 16))

    # Theme toggle  (Light | Dark)
    _theme_sep = tk.Label(_header, text="|", bg=PRIMARY, fg="#5589e8",
                          font=(ui_font_family, 12))
    _theme_sep.pack(side="right", fill="y", padx=(0, 4))
    create_header_toggle(
        _header, theme_var,
        [("Light", "light"), ("Dark", "dark")],
        font_family=ui_font_family,
    ).pack(side="right", fill="y", pady=8, padx=(0, 4))

    # Market toggle  (US | KR)
    _mkt_sep = tk.Label(_header, text="|", bg=PRIMARY, fg="#5589e8",
                        font=(ui_font_family, 12))
    _mkt_sep.pack(side="right", fill="y", padx=(0, 4))
    create_header_toggle(
        _header, market_var,
        [("US", "us"), ("KR", "kr")],
        font_family=ui_font_family,
    ).pack(side="right", fill="y", pady=8, padx=(0, 4))

    # Theme-aware widget registry: (widget, bg_palette_key, fg_palette_key|None)
    _theme_widgets: list[tuple[tk.Widget, str, str | None]] = []

    def _do_apply_theme(*_args) -> None:
        theme = theme_var.get()
        p = _apply_theme(ttk.Style(root), root, theme, ui_font_family)
        for widget, bg_key, fg_key in _theme_widgets:
            try:
                kw: dict = {"bg": p[bg_key]}
                if fg_key:
                    kw["fg"] = p[fg_key]
                widget.configure(**kw)
            except Exception:
                pass

    theme_var.trace_add("write", _do_apply_theme)

    # ------------------------------------------------------------------
    # Notebook (tabs)
    # ------------------------------------------------------------------
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=(6, 0))

    chart_tab = ttk.Frame(notebook, style="Surface.TFrame")
    financial_tab = ttk.Frame(notebook, style="Surface.TFrame")
    fundamental_tab = ttk.Frame(notebook, style="Surface.TFrame")
    backtest_tab = ttk.Frame(notebook, style="Surface.TFrame")
    notebook.add(chart_tab,       text="  차트  ")
    notebook.add(financial_tab,   text="  재무제표  ")
    notebook.add(fundamental_tab, text="  팩터 분석  ")
    notebook.add(backtest_tab,    text="  백테스트  ")
    mount_backtest_tab(backtest_tab)

    # Keep legacy backtest UI objects detached from notebook to avoid duplicate visible tabs.
    backtest_tab = ttk.Frame(notebook, style="Card.TFrame")

    chart_controls = ttk.Frame(chart_tab, padding=(12, 10, 12, 8), style="Card.TFrame")
    chart_controls.pack(fill="x", padx=10, pady=(10, 6))

    chart_main_row = ttk.Frame(chart_controls, style="TFrame")
    chart_main_row.pack(fill="x", pady=(0, 4))

    ttk.Label(chart_main_row, text="Ticker", style="Dim.TLabel").pack(side="left")
    ticker_var = tk.StringVar(value=state.ticker)
    ticker_entry = ttk.Entry(chart_main_row, width=16, textvariable=ticker_var)
    ticker_entry.pack(side="left", padx=6)

    ttk.Label(chart_main_row, text="Type", style="Dim.TLabel").pack(side="left", padx=(10, 0))
    chart_type_var = tk.StringVar(value=state.chart_type)
    chart_type_combo = ttk.Combobox(chart_main_row, width=10, state="readonly", textvariable=chart_type_var)
    chart_type_combo["values"] = ("candles", "line")
    chart_type_combo.pack(side="left", padx=6)

    offline_mode_var = tk.StringVar(value="offline" if state.offline_mode else "online")
    create_segmented_control(
        chart_main_row,
        offline_mode_var,
        [("Online", "online"), ("Offline", "offline")],
        font_family=ui_font_family,
    ).pack(side="left", padx=(10, 0))

    chart_load_btn = ttk.Button(chart_main_row, text="Load Data", style="Accent.TButton")
    chart_load_btn.pack(side="left", padx=(10, 0))

    chart_overlay_row = ttk.Frame(chart_controls, style="TFrame")
    chart_overlay_row.pack(fill="x", pady=(0, 4))
    open_tv_btn = ttk.Button(chart_overlay_row, text="📈  TradingView 차트 열기", style="Accent.TButton")
    open_tv_btn.pack(side="left")
    tv_chart_status_var = tk.StringVar(value="")
    ttk.Label(chart_overlay_row, textvariable=tv_chart_status_var, style="Dim.TLabel").pack(side="left", padx=(8, 0))

    chart_range_row = ttk.Frame(chart_controls, style="TFrame")
    chart_range_row.pack(fill="x")
    ttk.Label(chart_range_row, text="Financial Range", style="Dim.TLabel").pack(side="left")
    fin_from_var = tk.StringVar(value="2000-01-01")
    fin_to_var = tk.StringVar(value="")
    ttk.Label(chart_range_row, text="From", style="Dim.TLabel").pack(side="left", padx=(8, 2))
    fin_from_entry = ttk.Entry(chart_range_row, width=14, textvariable=fin_from_var)
    fin_from_entry.pack(side="left")
    ttk.Label(chart_range_row, text="To", style="Dim.TLabel").pack(side="left", padx=(8, 2))
    fin_to_entry = ttk.Entry(chart_range_row, width=14, textvariable=fin_to_var)
    fin_to_entry.pack(side="left")
    apply_fin_range_btn = ttk.Button(chart_range_row, text="Apply", style="Accent.TButton")
    apply_fin_range_btn.pack(side="left", padx=(8, 0))
    ttk.Label(chart_range_row, text="(YYYY / YYYYQn / YYYY-MM-DD)", style="Dim.TLabel").pack(side="left", padx=(8, 0))

    # Chart info panel (replaces matplotlib canvas)
    from market_data.ui_style import SURFACE_BG, TEXT_MAIN  # noqa: PLC0415

    chart_panel = ttk.Frame(chart_tab, style="Card.TFrame")
    chart_panel.pack(fill="both", expand=True, padx=10, pady=8)

    chart_panel_inner = ttk.Frame(chart_panel, padding=(28, 22, 28, 22), style="Card.TFrame")
    chart_panel_inner.pack(fill="both", expand=True)

    # Ticker headline
    chart_ticker_var = tk.StringVar(value="")
    chart_price_var = tk.StringVar(value="")
    chart_stats_var = tk.StringVar(value="데이터를 로드하려면 'Load Data'를 클릭하세요.")

    chart_ticker_lbl = tk.Label(
        chart_panel_inner, textvariable=chart_ticker_var,
        bg=SURFACE_BG, fg=TEXT_MAIN,
        font=(ui_font_family, 22, "bold"), anchor="w",
    )
    chart_ticker_lbl.pack(anchor="w")

    chart_price_lbl = tk.Label(
        chart_panel_inner, textvariable=chart_price_var,
        bg=SURFACE_BG, fg="#1fa774",
        font=(ui_font_family, 15), anchor="w",
    )
    chart_price_lbl.pack(anchor="w", pady=(2, 0))

    chart_stats_lbl = tk.Label(
        chart_panel_inner, textvariable=chart_stats_var,
        bg=SURFACE_BG, fg=TEXT_SUB,
        font=(ui_font_family, 11), anchor="w",
    )
    chart_stats_lbl.pack(anchor="w", pady=(4, 0))

    ttk.Separator(chart_panel_inner).pack(fill="x", pady=18)

    # TV chart launch hint
    hint_lines = [
        "📈  위 [TradingView 차트 열기] 버튼을 클릭하면 별도 창이 열립니다.",
        "",
        "  ·  부드러운 줌 / 팬 (트랙패드 핀치, 마우스 휠)",
        "  ·  전문 캔들스틱 렌더링 + 볼륨 바",
        "  ·  크로스헤어 + OHLCV 툴팁",
        "  ·  기술적 지표 (이동평균, 볼린저 밴드, 이치모쿠)",
        "  ·  그리기 도구 (추세선, 수평선, 피보나치 등)",
        "  ·  다크 테마 (TradingView 기본 스타일)",
    ]
    for line in hint_lines:
        tk.Label(
            chart_panel_inner, text=line,
            bg=SURFACE_BG, fg=TEXT_MUTED,
            font=(ui_font_family, 11), anchor="w",
        ).pack(anchor="w", pady=1)

    def build_ticker_loader_row(
        parent,
        default_ticker: str,
        default_market: str,
        on_load_callback,
        allow_sync_option: bool = True,
    ) -> dict[str, object]:  # type: ignore[no-untyped-def]
        row_card = ttk.Frame(parent, padding=(12, 8, 12, 6), style="Card.TFrame")
        row_card.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(row_card, text="Ticker", style="Dim.TLabel").pack(side="left")
        row_ticker_var = tk.StringVar(value=str(default_ticker or "AAPL").strip().upper())
        row_ticker_entry = ttk.Entry(row_card, width=14, textvariable=row_ticker_var)
        row_ticker_entry.pack(side="left", padx=(6, 8))

        row_use_chart_var = tk.BooleanVar(value=allow_sync_option)
        if allow_sync_option:
            ttk.Checkbutton(row_card, text="Use Chart tab settings", variable=row_use_chart_var).pack(side="left", padx=(0, 8))

        def _sync_from_chart(*_args) -> None:
            if not allow_sync_option:
                return
            use_chart = bool(row_use_chart_var.get())
            if use_chart:
                row_ticker_var.set(ticker_var.get().strip().upper())
            row_ticker_entry.configure(state="disabled" if use_chart else "normal")

        def _do_load() -> None:
            if allow_sync_option and bool(row_use_chart_var.get()):
                ticker = ticker_var.get().strip().upper()
            else:
                ticker = row_ticker_var.get().strip().upper()
            market_name = market_var.get().strip().lower() or "us"
            if not ticker:
                messagebox.showerror("Input Error", "Ticker is required")
                return
            on_load_callback(ticker, market_name)

        load_btn = ttk.Button(row_card, text="Load", style="Accent.TButton", command=_do_load)
        load_btn.pack(side="left", padx=(0, 6))

        if allow_sync_option:
            def _sync_to_chart() -> None:
                ticker = row_ticker_var.get().strip().upper()
                if ticker:
                    ticker_var.set(ticker)
                try:
                    render_chart()
                except Exception:
                    pass

            ttk.Button(row_card, text="Sync To Chart", command=_sync_to_chart).pack(side="left", padx=(0, 6))
            row_use_chart_var.trace_add("write", _sync_from_chart)
            ticker_var.trace_add("write", _sync_from_chart)
            _sync_from_chart()

        row_ticker_entry.bind("<Return>", lambda _event: _do_load())

        return {
            "ticker_var": row_ticker_var,
            "market_var": market_var,   # now global
            "use_chart_var": row_use_chart_var,
            "load": _do_load,
        }

    financial_loader = build_ticker_loader_row(
        parent=financial_tab,
        default_ticker=state.ticker,
        default_market=state.market,
        on_load_callback=lambda ticker, market_name: _load_financial_from_controls(ticker, market_name),
        allow_sync_option=True,
    )

    fin_controls = ttk.Frame(financial_tab, padding=(12, 10, 12, 8), style="Card.TFrame")
    fin_controls.pack(fill="x", padx=10, pady=(10, 6))

    fin_statement_var = tk.StringVar(value="is")
    fin_mode_var = tk.StringVar(value="ttm")
    fin_horizon_var = tk.StringVar(value="all")

    control_row = ttk.Frame(fin_controls, style="TFrame")
    control_row.pack(fill="x", pady=(2, 2))
    create_segmented_control(
        control_row,
        fin_statement_var,
        [("손익계산서", "is"), ("재무상태표", "bs"), ("현금흐름표", "cf")],
        font_family=ui_font_family,
    ).pack(side="left", padx=(0, 10))
    create_segmented_control(
        control_row,
        fin_mode_var,
        [("4분기누적", "ttm"), ("분기", "quarter"), ("연도", "year")],
        font_family=ui_font_family,
    ).pack(side="left", padx=(0, 10))
    create_segmented_control(
        control_row,
        fin_horizon_var,
        [("전체", "all"), ("10년", "10y"), ("5년", "5y")],
        font_family=ui_font_family,
    ).pack(side="left")

    fin_scroll_canvas = tk.Canvas(financial_tab, bg=PANEL_BG, highlightthickness=0, bd=0)
    fin_scrollbar = ttk.Scrollbar(financial_tab, orient="vertical", command=fin_scroll_canvas.yview)
    fin_scroll_canvas.configure(yscrollcommand=fin_scrollbar.set)
    fin_scrollbar.pack(side="right", fill="y")
    fin_scroll_canvas.pack(side="left", fill="both", expand=True)
    fin_inner = ttk.Frame(fin_scroll_canvas)
    fin_inner_window = fin_scroll_canvas.create_window((0, 0), window=fin_inner, anchor="nw")

    def _on_fin_inner_configure(_event) -> None:  # type: ignore[no-untyped-def]
        fin_scroll_canvas.configure(scrollregion=fin_scroll_canvas.bbox("all"))

    def _on_fin_canvas_configure(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal financial_layout_mode, financial_last_canvas_width, financial_resize_job, financial_is_rendering
        fin_scroll_canvas.itemconfigure(fin_inner_window, width=event.width)
        width = max(int(getattr(event, "width", 0)), 0)
        if width <= 0:
            return
        next_mode = get_layout_mode(width - 24)
        mode_changed = next_mode != financial_layout_mode
        width_changed = abs(width - financial_last_canvas_width) >= 240
        financial_last_canvas_width = width
        if mode_changed:
            financial_layout_mode = next_mode
        if not (mode_changed or width_changed):
            return
        if financial_cached_datasets is None or financial_cached_kpi is None or financial_is_rendering:
            return
        if financial_resize_job is not None:
            try:
                root.after_cancel(financial_resize_job)
            except Exception:
                pass

        def _rerender_later() -> None:
            nonlocal financial_resize_job
            financial_resize_job = None
            if financial_cached_datasets is None or financial_cached_kpi is None or financial_is_rendering:
                return
            _draw_financial_data(financial_cached_datasets, financial_cached_kpi)

        financial_resize_job = root.after(140, _rerender_later)

    fin_inner.bind("<Configure>", _on_fin_inner_configure)
    fin_scroll_canvas.bind("<Configure>", _on_fin_canvas_configure)

    def _on_fin_mousewheel(event) -> None:  # type: ignore[no-untyped-def]
        if hasattr(event, "num") and event.num in (4, 5):
            units = -1 if event.num == 4 else 1
        else:
            delta = int(getattr(event, "delta", 0))
            if delta == 0:
                return
            units = -1 if delta > 0 else 1
        fin_scroll_canvas.yview_scroll(units, "units")

    kpi_card = ttk.Frame(fin_inner, style="Card.TFrame", padding=(14, 12, 14, 12))
    kpi_card.pack(fill="x", padx=12, pady=(10, 10))
    ttk.Label(kpi_card, text="KPI 미니 테이블 (최근 5개 기간)", style="Dim.TLabel").pack(anchor="w")
    kpi_table = ttk.Treeview(kpi_card, show="headings", height=6)
    kpi_table.pack(fill="x", pady=(4, 0))

    fig_fin = plt.Figure(figsize=(11.0, 10.0))
    canvas_fin = FigureCanvasTkAgg(fig_fin, master=fin_inner)
    canvas_fin_widget = canvas_fin.get_tk_widget()
    canvas_fin_widget.pack(fill="x", expand=False, padx=10, pady=(6, 12))

    fundamental_loader = build_ticker_loader_row(
        parent=fundamental_tab,
        default_ticker=state.ticker,
        default_market=state.market,
        on_load_callback=lambda ticker, market_name: _load_fundamental_from_controls(ticker, market_name),
        allow_sync_option=True,
    )

    fund_controls = ttk.Frame(fundamental_tab, padding=(12, 10, 12, 8), style="Card.TFrame")
    fund_controls.pack(fill="x", padx=10, pady=(10, 6))

    fund_category_var = tk.StringVar(value="profit")
    fund_mode_var = tk.StringVar(value="ttm")
    fund_horizon_var = tk.StringVar(value="all")

    fund_control_row = ttk.Frame(fund_controls, style="TFrame")
    fund_control_row.pack(fill="x", pady=(2, 2))
    create_segmented_control(
        fund_control_row,
        fund_category_var,
        [("수익성", "profit"), ("성장성", "growth"), ("안정성", "stability"), ("효율성", "efficiency"), ("밸류에이션", "valuation")],
        font_family=ui_font_family,
    ).pack(side="left", padx=(0, 10))
    create_segmented_control(
        fund_control_row,
        fund_mode_var,
        [("4분기누적", "ttm"), ("분기", "quarter"), ("연도", "year")],
        font_family=ui_font_family,
    ).pack(side="left", padx=(0, 10))
    create_segmented_control(
        fund_control_row,
        fund_horizon_var,
        [("전체", "all"), ("10년", "10y"), ("5년", "5y")],
        font_family=ui_font_family,
    ).pack(side="left")

    fund_scroll_canvas = tk.Canvas(fundamental_tab, bg=PANEL_BG, highlightthickness=0, bd=0)
    fund_scrollbar = ttk.Scrollbar(fundamental_tab, orient="vertical", command=fund_scroll_canvas.yview)
    fund_scroll_canvas.configure(yscrollcommand=fund_scrollbar.set)
    fund_scrollbar.pack(side="right", fill="y")
    fund_scroll_canvas.pack(side="left", fill="both", expand=True)
    fund_inner = ttk.Frame(fund_scroll_canvas)
    fund_inner_window = fund_scroll_canvas.create_window((0, 0), window=fund_inner, anchor="nw")

    def _on_fund_inner_configure(_event) -> None:  # type: ignore[no-untyped-def]
        fund_scroll_canvas.configure(scrollregion=fund_scroll_canvas.bbox("all"))

    def _on_fund_canvas_configure(event) -> None:  # type: ignore[no-untyped-def]
        nonlocal fundamental_layout_mode, fundamental_last_canvas_width, fundamental_resize_job, fundamental_is_rendering
        fund_scroll_canvas.itemconfigure(fund_inner_window, width=event.width)
        width = max(int(getattr(event, "width", 0)), 0)
        if width <= 0:
            return
        next_mode = get_layout_mode(width - 24)
        mode_changed = next_mode != fundamental_layout_mode
        width_changed = abs(width - fundamental_last_canvas_width) >= 240
        fundamental_last_canvas_width = width
        if mode_changed:
            fundamental_layout_mode = next_mode
        if not (mode_changed or width_changed):
            return
        if fundamental_cached_datasets is None or fundamental_cached_kpi is None or fundamental_is_rendering:
            return
        if fundamental_resize_job is not None:
            try:
                root.after_cancel(fundamental_resize_job)
            except Exception:
                pass

        def _rerender_later() -> None:
            nonlocal fundamental_resize_job
            fundamental_resize_job = None
            if fundamental_cached_datasets is None or fundamental_cached_kpi is None or fundamental_is_rendering:
                return
            _draw_fundamental_data(fundamental_cached_datasets, fundamental_cached_kpi)

        fundamental_resize_job = root.after(140, _rerender_later)

    fund_inner.bind("<Configure>", _on_fund_inner_configure)
    fund_scroll_canvas.bind("<Configure>", _on_fund_canvas_configure)

    def _on_fund_mousewheel(event) -> None:  # type: ignore[no-untyped-def]
        if hasattr(event, "num") and event.num in (4, 5):
            units = -1 if event.num == 4 else 1
        else:
            delta = int(getattr(event, "delta", 0))
            if delta == 0:
                return
            units = -1 if delta > 0 else 1
        fund_scroll_canvas.yview_scroll(units, "units")

    fund_kpi_card = ttk.Frame(fund_inner, style="Card.TFrame", padding=(14, 12, 14, 12))
    fund_kpi_card.pack(fill="x", padx=12, pady=(10, 10))
    ttk.Label(fund_kpi_card, text="Fundamental KPI (최근 5개 기간)", style="Dim.TLabel").pack(anchor="w")
    fund_kpi_table = ttk.Treeview(fund_kpi_card, show="headings", height=6)
    fund_kpi_table.pack(fill="x", pady=(4, 0))

    fig_fund = plt.Figure(figsize=(11.0, 10.0))
    canvas_fund = FigureCanvasTkAgg(fig_fund, master=fund_inner)
    canvas_fund_widget = canvas_fund.get_tk_widget()
    canvas_fund_widget.pack(fill="x", expand=False, padx=10, pady=(6, 12))

    backtest_controls = ttk.Frame(backtest_tab, padding=(12, 10, 12, 8), style="Card.TFrame")
    backtest_controls.pack(fill="x", padx=10, pady=(10, 6))

    bt_strategy_path_var = tk.StringVar(value="strategies/per_ps_q_rebal.json")
    bt_screen_var = tk.StringVar(value="pe <= 10 and ps <= 3")
    bt_freq_var = tk.StringVar(value="Q")
    bt_start_var = tk.StringVar(value="2000-01-01")
    bt_end_var = tk.StringVar(value="")
    bt_holdings_var = tk.StringVar(value="3")
    bt_market_var = market_var  # use global market selector

    row1 = ttk.Frame(backtest_controls, style="TFrame")
    row1.pack(fill="x", pady=(2, 4))
    ttk.Label(row1, text="Strategy Config", style="Dim.TLabel").pack(side="left")
    bt_strategy_entry = ttk.Entry(row1, width=48, textvariable=bt_strategy_path_var)
    bt_strategy_entry.pack(side="left", padx=(6, 12))
    ttk.Label(row1, text="Screen Rule", style="Dim.TLabel").pack(side="left")
    bt_screen_entry = ttk.Entry(row1, width=36, textvariable=bt_screen_var)
    bt_screen_entry.pack(side="left", padx=(6, 0))

    row2 = ttk.Frame(backtest_controls, style="TFrame")
    row2.pack(fill="x", pady=(2, 2))
    ttk.Label(row2, text="Freq", style="Dim.TLabel").pack(side="left")
    bt_freq_combo = ttk.Combobox(row2, width=6, state="readonly", textvariable=bt_freq_var)
    bt_freq_combo["values"] = ("W", "M", "Q")
    bt_freq_combo.pack(side="left", padx=(6, 10))
    ttk.Label(row2, text="Start", style="Dim.TLabel").pack(side="left")
    bt_start_entry = ttk.Entry(row2, width=14, textvariable=bt_start_var)
    bt_start_entry.pack(side="left", padx=(6, 10))
    ttk.Label(row2, text="End", style="Dim.TLabel").pack(side="left")
    bt_end_entry = ttk.Entry(row2, width=14, textvariable=bt_end_var)
    bt_end_entry.pack(side="left", padx=(6, 10))
    ttk.Label(row2, text="Holdings", style="Dim.TLabel").pack(side="left")
    bt_holdings_entry = ttk.Entry(row2, width=7, textvariable=bt_holdings_var)
    bt_holdings_entry.pack(side="left", padx=(6, 10))

    create_segmented_control(
        row2,
        offline_mode_var,
        [("Online", "online"), ("Offline", "offline")],
        font_family=ui_font_family,
    ).pack(side="left", padx=(0, 10))

    bt_run_screen_btn = ttk.Button(row2, text="Run Screen", style="TButton")
    bt_run_screen_btn.pack(side="left", padx=(4, 6))
    bt_run_strategy_btn = ttk.Button(row2, text="Run Strategy", style="TButton")
    bt_run_strategy_btn.pack(side="left", padx=(0, 6))
    bt_compare_btn = ttk.Button(row2, text="Compare", style="Accent.TButton")
    bt_compare_btn.pack(side="left")

    bt_summary_var = tk.StringVar(value="Backtest not started")
    bt_summary_label = ttk.Label(backtest_tab, textvariable=bt_summary_var, style="Dim.TLabel")
    bt_summary_label.pack(fill="x", padx=16, pady=(0, 6))

    fig_bt = plt.Figure(figsize=(12.0, 7.8))
    ax_bt_eq = fig_bt.add_subplot(2, 1, 1)
    ax_bt_dd = fig_bt.add_subplot(2, 1, 2, sharex=ax_bt_eq)
    fig_bt.subplots_adjust(hspace=0.12, top=0.96, bottom=0.09, left=0.07, right=0.97)
    canvas_bt = FigureCanvasTkAgg(fig_bt, master=backtest_tab)
    canvas_bt_widget = canvas_bt.get_tk_widget()
    canvas_bt_widget.pack(fill="both", expand=True, padx=10, pady=(4, 8))

    bt_yearly_card = ttk.Frame(backtest_tab, style="Card.TFrame", padding=(14, 10, 14, 12))
    bt_yearly_card.pack(fill="x", padx=10, pady=(0, 10))
    ttk.Label(bt_yearly_card, text="Yearly Returns (%)", style="Dim.TLabel").pack(anchor="w")
    bt_yearly_table = ttk.Treeview(bt_yearly_card, show="headings", height=6)
    bt_yearly_table.pack(fill="x", pady=(4, 0))

    current_summary = "Enter ticker and click Load"
    _chart_loaded_df: pd.DataFrame | None = None     # last loaded OHLCV DataFrame
    _chart_loaded_valuation = None                   # last loaded ValuationResult
    current_financial_summary = "financials=not-loaded"
    financial_hidden_labels_by_chart: dict[str, set[str]] = {}
    financial_pick_map: dict[object, tuple[str, str]] = {}
    financial_axes_by_chart: dict[str, object] = {}
    financial_cards_by_chart: dict[str, ChartCard] = {}
    financial_right_axes_by_chart: dict[str, object] = {}
    financial_cached_datasets: dict[str, pd.DataFrame] | None = None
    financial_cached_kpi: pd.DataFrame | None = None
    financial_bundle_cache: dict[tuple[str, str, str, str, str, str, str], tuple[dict[str, pd.DataFrame], pd.DataFrame]] = {}
    financial_cached_market = ""
    financial_last_ticker: str | None = None
    financial_last_market: str | None = None
    financial_layout_mode = "two_col"
    financial_last_canvas_width = 0
    financial_resize_job = None
    financial_is_rendering = False
    current_fundamental_summary = "fundamental=not-loaded"
    fundamental_hidden_labels_by_chart: dict[str, set[str]] = {}
    fundamental_pick_map: dict[object, tuple[str, str]] = {}
    fundamental_axes_by_chart: dict[str, object] = {}
    fundamental_cards_by_chart: dict[str, ChartCard] = {}
    fundamental_right_axes_by_chart: dict[str, object] = {}
    fundamental_cached_datasets: dict[str, pd.DataFrame] | None = None
    fundamental_cached_kpi: pd.DataFrame | None = None
    fundamental_bundle_cache: dict[tuple[str, str, str, str, str, str, str], tuple[dict[str, pd.DataFrame], pd.DataFrame]] = {}
    fundamental_cached_market = ""
    fundamental_last_ticker: str | None = None
    fundamental_last_market: str | None = None
    fundamental_layout_mode = "two_col"
    fundamental_last_canvas_width = 0
    fundamental_resize_job = None
    fundamental_is_rendering = False
    last_loaded_ticker: str | None = None
    last_loaded_market: str | None = None
    backtest_screen_result = None
    backtest_strategy_result = None
    _chart_gen = 0
    _fin_gen = 0
    _fund_gen = 0
    _chart_debounce_job: object | None = None
    _fin_debounce_job: object | None = None
    _fund_debounce_job: object | None = None

    def _apply_fin_axis_limits(ax, arrays: list[np.ndarray], include_zero: bool = True) -> None:
        finite_parts: list[np.ndarray] = []
        for arr in arrays:
            valid = arr[np.isfinite(arr)]
            if valid.size > 0:
                finite_parts.append(valid)
        if not finite_parts:
            ax.set_ylim(0.0, 1.0)
            return
        flat = np.concatenate(finite_parts)
        lo = float(np.min(flat))
        hi = float(np.max(flat))
        if flat.size >= 12:
            q02 = float(np.quantile(flat, 0.02))
            q98 = float(np.quantile(flat, 0.98))
            if np.isfinite(q02):
                lo = q02 if q02 < lo else lo
            if np.isfinite(q98) and hi > q98 * 1.35:
                hi = q98 * 1.12
        if include_zero:
            lo = min(lo, 0.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            span = max(abs(hi), 1.0)
            ax.set_ylim(lo - span * 0.2, hi + span * 0.2)
            return
        pad = max((hi - lo) * 0.08, 1e-6)
        ax.set_ylim(lo - pad * 0.25, hi + pad)

    def _legend_handles(legend) -> list:  # type: ignore[no-untyped-def]
        handles = getattr(legend, "legendHandles", None)
        if handles is None:
            handles = getattr(legend, "legend_handles", [])
        return list(handles or [])

    def _format_kpi_cell(metric: str, value: float) -> str:
        if not np.isfinite(value):
            return "미지원"
        if "YoY" in metric:
            return f"{value:+.1f}%"
        return f"{value:,.1f}"

    def _render_kpi_table(kpi_df: pd.DataFrame) -> None:
        columns = ["항목"]
        if not kpi_df.empty:
            columns += [str(c) for c in kpi_df.columns]

        kpi_table.configure(columns=columns)
        for col in columns:
            width = 120 if col == "항목" else 92
            anchor = "w" if col == "항목" else "center"
            kpi_table.heading(col, text=col)
            kpi_table.column(col, width=width, anchor=anchor, stretch=False)

        for row_id in kpi_table.get_children():
            kpi_table.delete(row_id)

        if kpi_df.empty:
            kpi_table.insert("", "end", values=("데이터 없음",))
            return

        for metric in kpi_df.index:
            values = [str(metric)]
            for raw in pd.to_numeric(kpi_df.loc[metric], errors="coerce").to_numpy(dtype=float):
                values.append(_format_kpi_cell(str(metric), float(raw)))
            kpi_table.insert("", "end", values=values)

    def _format_fund_kpi_cell(metric: str, value: float) -> str:
        if not np.isfinite(value):
            return "미지원"
        metric_text = str(metric)
        is_pct_metric = (
            ("YoY" in metric_text)
            or ("률" in metric_text)
            or ("비율" in metric_text)
            or ("비중" in metric_text)
            or ("ROE" in metric_text.upper())
            or ("ROA" in metric_text.upper())
        )
        if is_pct_metric:
            if "YoY" in metric_text:
                return f"{value:+.1f}%"
            return f"{value:.1f}%"
        return f"{value:,.2f}"

    def _render_fund_kpi_table(kpi_df: pd.DataFrame) -> None:
        columns = ["항목"]
        if not kpi_df.empty:
            columns += [str(c) for c in kpi_df.columns]

        fund_kpi_table.configure(columns=columns)
        for col in columns:
            width = 128 if col == "항목" else 94
            anchor = "w" if col == "항목" else "center"
            fund_kpi_table.heading(col, text=col)
            fund_kpi_table.column(col, width=width, anchor=anchor, stretch=False)

        for row_id in fund_kpi_table.get_children():
            fund_kpi_table.delete(row_id)

        if kpi_df.empty:
            fund_kpi_table.insert("", "end", values=("데이터 없음",))
            return

        for metric in kpi_df.index:
            values = [str(metric)]
            for raw in pd.to_numeric(kpi_df.loc[metric], errors="coerce").to_numpy(dtype=float):
                values.append(_format_fund_kpi_cell(str(metric), float(raw)))
            fund_kpi_table.insert("", "end", values=values)

    def _render_backtest_yearly_table() -> None:
        nonlocal backtest_screen_result, backtest_strategy_result

        screen_df = (
            backtest_screen_result.yearly_returns.copy()
            if backtest_screen_result is not None and not backtest_screen_result.yearly_returns.empty
            else pd.DataFrame(columns=["year", "return_pct"])
        )
        strategy_df = (
            backtest_strategy_result.yearly_returns.copy()
            if backtest_strategy_result is not None and not backtest_strategy_result.yearly_returns.empty
            else pd.DataFrame(columns=["year", "return_pct"])
        )

        if not screen_df.empty:
            screen_df = screen_df.rename(columns={"return_pct": "Screen"})
        else:
            screen_df = pd.DataFrame(columns=["year", "Screen"])
        if not strategy_df.empty:
            strategy_df = strategy_df.rename(columns={"return_pct": "Strategy"})
        else:
            strategy_df = pd.DataFrame(columns=["year", "Strategy"])

        merged = screen_df.merge(strategy_df, on="year", how="outer").sort_values("year")
        merged = merged.reset_index(drop=True)

        columns = ["Year", "Screen", "Strategy"]
        bt_yearly_table.configure(columns=columns)
        bt_yearly_table.heading("Year", text="Year")
        bt_yearly_table.heading("Screen", text="Screen")
        bt_yearly_table.heading("Strategy", text="Strategy")
        bt_yearly_table.column("Year", width=90, anchor="center", stretch=False)
        bt_yearly_table.column("Screen", width=120, anchor="center", stretch=False)
        bt_yearly_table.column("Strategy", width=120, anchor="center", stretch=False)

        for row_id in bt_yearly_table.get_children():
            bt_yearly_table.delete(row_id)

        if merged.empty:
            bt_yearly_table.insert("", "end", values=("N/A", "N/A", "N/A"))
            return

        for _, row in merged.iterrows():
            year_val = str(int(row["year"])) if pd.notna(row["year"]) else ""
            s_val = pd.to_numeric(pd.Series([row.get("Screen")]), errors="coerce").iloc[0]
            t_val = pd.to_numeric(pd.Series([row.get("Strategy")]), errors="coerce").iloc[0]
            s_txt = f"{float(s_val):+.2f}%" if pd.notna(s_val) else "-"
            t_txt = f"{float(t_val):+.2f}%" if pd.notna(t_val) else "-"
            bt_yearly_table.insert("", "end", values=(year_val, s_txt, t_txt))

    def _render_backtest_plot() -> None:
        nonlocal backtest_screen_result, backtest_strategy_result
        ax_bt_eq.clear()
        ax_bt_dd.clear()

        plotted = 0
        summary_parts: list[str] = []

        for label, result, color in [
            ("Screen", backtest_screen_result, "#2f7ed8"),
            ("Strategy", backtest_strategy_result, "#d94f4f"),
        ]:
            if result is None or result.equity_curve is None or result.equity_curve.empty:
                continue
            eq_df = result.equity_curve.copy()
            eq_df["date"] = pd.to_datetime(eq_df["date"], errors="coerce")
            eq_df = eq_df.dropna(subset=["date"]).sort_values("date")
            if eq_df.empty:
                continue

            eq = pd.to_numeric(eq_df["equity"], errors="coerce")
            ax_bt_eq.plot(eq_df["date"], eq, label=label, color=color, linewidth=1.5)

            dd_df = result.drawdown_curve.copy() if result.drawdown_curve is not None else pd.DataFrame()
            if not dd_df.empty and "date" in dd_df.columns and "drawdown" in dd_df.columns:
                dd_df["date"] = pd.to_datetime(dd_df["date"], errors="coerce")
                dd_df = dd_df.dropna(subset=["date"]).sort_values("date")
                dd_vals = pd.to_numeric(dd_df["drawdown"], errors="coerce") * 100.0
                ax_bt_dd.plot(dd_df["date"], dd_vals, label=label, color=color, linewidth=1.2)

            metrics = result.metrics or {}
            cagr = float(metrics.get("cagr", 0.0)) * 100.0
            mdd = float(metrics.get("mdd", 0.0)) * 100.0
            sharpe = float(metrics.get("sharpe", 0.0))
            summary_parts.append(f"{label}: CAGR {cagr:+.2f}% | MDD {mdd:.2f}% | Sharpe {sharpe:.2f}")
            plotted += 1

        if plotted == 0:
            ax_bt_eq.text(0.5, 0.5, "Run screen/strategy backtest", ha="center", va="center", transform=ax_bt_eq.transAxes)
            ax_bt_dd.text(0.5, 0.5, "Drawdown view", ha="center", va="center", transform=ax_bt_dd.transAxes)
        else:
            ax_bt_eq.legend(loc="upper left", fontsize=8)
            ax_bt_dd.legend(loc="lower left", fontsize=8)

        ax_bt_eq.grid(alpha=0.2)
        ax_bt_dd.grid(alpha=0.2)
        ax_bt_eq.set_ylabel("Equity", fontsize=8)
        ax_bt_dd.set_ylabel("Drawdown %", fontsize=8)
        ax_bt_dd.set_xlabel("Date", fontsize=8)
        ax_bt_eq.set_title("Backtest: Equity Curve", fontsize=10)
        ax_bt_dd.set_title("Backtest: Drawdown", fontsize=10)
        canvas_bt.draw_idle()

        bt_summary_var.set(" | ".join(summary_parts) if summary_parts else "Backtest not started")
        _render_backtest_yearly_table()

    def _parse_backtest_inputs() -> tuple[str, str, str, str | None, int, str]:
        screen_expr = bt_screen_var.get().strip()
        freq = bt_freq_var.get().strip().upper() or "Q"
        start = bt_start_var.get().strip() or "2000-01-01"
        end_raw = bt_end_var.get().strip()
        end = end_raw if end_raw else None
        holdings = max(1, int(bt_holdings_var.get().strip() or "3"))
        market = bt_market_var.get().strip().lower() or "us"
        return screen_expr, freq, start, end, holdings, market

    def _run_backtest_screen() -> None:
        nonlocal backtest_screen_result
        try:
            screen_expr, freq, start, end, holdings, market = _parse_backtest_inputs()
            if not screen_expr:
                messagebox.showerror("Backtest Error", "Screen rule is required")
                return
            backtest_screen_result = run_screen_backtest(
                screen_expr=screen_expr,
                freq=freq,
                start=start,
                end=end,
                holdings=holdings,
                sizing="equal",
                market=market,
                out_dir=None,
                offline_mode=(offline_mode_var.get() == "offline"),
            )
            backtest_screen_result.name = "Screen"
            _render_backtest_plot()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Screen Backtest Error", str(exc))

    def _run_backtest_strategy() -> None:
        nonlocal backtest_strategy_result
        try:
            cfg_path = bt_strategy_path_var.get().strip()
            if not cfg_path:
                messagebox.showerror("Backtest Error", "Strategy config path is required")
                return
            backtest_strategy_result = run_strategy_backtest_from_config(
                cfg_path,
                offline_mode_override=(offline_mode_var.get() == "offline"),
            )
            backtest_strategy_result.name = "Strategy"
            _render_backtest_plot()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Strategy Backtest Error", str(exc))

    def _run_backtest_compare() -> None:
        _run_backtest_screen()
        _run_backtest_strategy()

    def _draw_single_metric_chart(
        ax,
        spec: dict,
        data: pd.DataFrame,
        layout_mode: str,
        chart_id: str,
        hidden_labels_by_chart: dict[str, set[str]],
        pick_map: dict[object, tuple[str, str]],
        right_axes_by_chart: dict[str, object],
        legend_inside: bool = False,
    ) -> None:  # type: ignore[no-untyped-def]
        prev_right_ax = right_axes_by_chart.pop(chart_id, None)
        if prev_right_ax is not None:
            try:
                prev_right_ax.remove()
            except Exception:
                pass

        ax.clear()
        apply_chart_style(ax)
        ax_right = None
        hidden_labels = hidden_labels_by_chart.get(chart_id, set())
        card_box = FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            facecolor="#ffffff",
            edgecolor="#dbe2ec",
            linewidth=0.9,
            zorder=-5,
            clip_on=False,
        )
        ax.add_patch(card_box)
        if data.empty:
            ax.text(0.5, 0.5, "미지원 데이터", ha="center", va="center", transform=ax.transAxes, fontsize=6)
            ax.set_title(spec["title"], fontsize=13, pad=12)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        x = np.arange(len(data), dtype=float)
        labels = data["label"].astype(str).tolist() if "label" in data.columns else [str(i) for i in range(len(data))]
        tick_step = max(1, int(np.ceil(len(labels) / 10)))
        ticks = list(range(0, len(labels), tick_step))
        if (len(labels) - 1) not in ticks:
            ticks.append(len(labels) - 1)
        tick_labels = [labels[i] for i in ticks]

        left_defs = spec.get("left", [])
        right_defs = spec.get("right", [])
        left_percent = bool(spec.get("left_percent", False))
        right_percent = bool(spec.get("right_percent", False))

        left_values_raw: list[np.ndarray] = []
        for legend_label, col, _kind, _color in left_defs:
            if legend_label in hidden_labels or col not in data.columns:
                continue
            arr = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(arr).sum() > 0:
                left_values_raw.append(arr)

        if left_values_raw:
            left_scale, left_unit = (1.0, "%") if left_percent else _choose_scale(np.concatenate(left_values_raw))
        else:
            left_scale, left_unit = (1.0, "%" if left_percent else "")

        left_arrays: list[np.ndarray] = []
        bar_base = np.zeros(len(x), dtype=float)
        legend_handles: list = []
        legend_labels: list[str] = []
        for legend_label, col, kind, color in left_defs:
            if col not in data.columns:
                continue
            arr_raw = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
            arr = arr_raw / left_scale
            valid_count = int(np.isfinite(arr).sum())
            if valid_count <= 0:
                continue
            if legend_label in hidden_labels:
                proxy = Line2D([0], [0], color=color, marker="o", linewidth=1.3, markersize=3.6) if kind == "line" else Patch(facecolor=color, edgecolor=color)
                proxy.set_alpha(0.22)
                legend_handles.append(proxy)
                legend_labels.append(legend_label)
                continue

            if kind == "bar":
                values = np.where(np.isfinite(arr), arr, 0.0)
                ax.bar(x, values, bottom=bar_base, width=0.62, color=color, alpha=0.83)
                bar_base = bar_base + values
                left_arrays.append(bar_base.copy())
                handle = Patch(facecolor=color, edgecolor=color)
            else:
                ax.plot(x, arr, color=color, linewidth=1.45, marker="o", markersize=2.4)
                left_arrays.append(arr)
                handle = Line2D([0], [0], color=color, marker="o", linewidth=1.4, markersize=3.6)
            legend_handles.append(handle)
            legend_labels.append(legend_label)

        right_arrays: list[np.ndarray] = []
        if right_defs:
            right_values_raw: list[np.ndarray] = []
            for legend_label, col, _kind, _color in right_defs:
                if legend_label in hidden_labels or col not in data.columns:
                    continue
                arr = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
                if np.isfinite(arr).sum() > 0:
                    right_values_raw.append(arr)
            if right_values_raw:
                right_scale, right_unit = (1.0, "%") if right_percent else _choose_scale(np.concatenate(right_values_raw))
            else:
                right_scale, right_unit = (1.0, "%" if right_percent else "")

            ax_right = ax.twinx()
            apply_secondary_axis_style(ax_right)
            for legend_label, col, kind, color in right_defs:
                if col not in data.columns:
                    continue
                arr_raw = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
                arr = arr_raw / right_scale
                if np.isfinite(arr).sum() <= 0:
                    continue
                if legend_label in hidden_labels:
                    proxy = Line2D([0], [0], color=color, marker="o", linewidth=1.3, markersize=3.6) if kind == "line" else Patch(facecolor=color, edgecolor=color)
                    proxy.set_alpha(0.22)
                    legend_handles.append(proxy)
                    legend_labels.append(legend_label)
                    continue
                if kind == "bar":
                    ax_right.bar(x, np.where(np.isfinite(arr), arr, 0.0), width=0.38, color=color, alpha=0.36)
                    right_arrays.append(arr)
                    handle = Patch(facecolor=color, edgecolor=color)
                else:
                    ax_right.plot(x, arr, color=color, linewidth=1.35, marker="o", markersize=2.2)
                    right_arrays.append(arr)
                    handle = Line2D([0], [0], color=color, marker="o", linewidth=1.3, markersize=3.6)
                legend_handles.append(handle)
                legend_labels.append(legend_label)

            y_right = f"({right_unit})" if right_unit else ""
            ax_right.set_ylabel("")
            add_axis_unit_label(ax_right, y_right, side="right")
            ax_right.tick_params(axis="y", labelsize=9, colors="#667085")
            if right_arrays:
                _apply_fin_axis_limits(ax_right, right_arrays, include_zero=not right_percent)
            else:
                ax_right.set_yticks([])
                ax_right.set_ylabel("")

        if left_arrays:
            _apply_fin_axis_limits(ax, left_arrays, include_zero=not left_percent)
        else:
            ax.set_ylim(0.0, 1.0)
            if not right_arrays:
                ax.text(0.5, 0.5, "미지원 데이터", ha="center", va="center", transform=ax.transAxes, fontsize=5.5, color="#6b7280")

        unit_label = f"({left_unit})" if left_unit else ""
        ax.set_ylabel("")
        add_axis_unit_label(ax, unit_label, side="left")
        ax.set_xlim(-0.7, len(x) - 0.3)
        ax.set_xticks(ticks, tick_labels)
        ax.set_xticks(x, minor=True)
        ax.tick_params(axis="x", which="major", rotation=0, labelsize=9, pad=0)
        ax.tick_params(axis="x", which="minor", length=1.8, width=0.7, color="#9aa7ba")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=1.0)
        ax.grid(axis="x", which="minor", alpha=0.06)
        ax.set_xlabel("")

        ax.set_title(spec["title"], fontsize=14, pad=12, loc="center")
        if legend_handles:
            if legend_inside:
                lg = ax.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper left",
                    bbox_to_anchor=(0.01, 0.99),
                    ncol=max(1, len(legend_labels)),
                    fontsize=9.5,
                    handletextpad=0.4,
                    columnspacing=0.75,
                    borderaxespad=0.0,
                    frameon=False,
                )
            else:
                lg = ax.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.045),
                    ncol=max(1, len(legend_labels)),
                    fontsize=10,
                    handletextpad=0.45,
                    columnspacing=0.8,
                    borderaxespad=0.0,
                    frameon=False,
                )
            _format_chart_legend(lg)
            for h, label in zip(_legend_handles(lg), legend_labels):
                h.set_picker(5)
                pick_map[h] = (chart_id, label)
            for t, label in zip(lg.get_texts(), legend_labels):
                if label in hidden_labels:
                    t.set_alpha(0.35)
                t.set_picker(True)
                pick_map[t] = (chart_id, label)

        if ax_right is not None:
            right_axes_by_chart[chart_id] = ax_right

    def _draw_single_fin_chart(
        ax,
        spec: dict,
        data: pd.DataFrame,
        layout_mode: str,
        chart_id: str,
    ) -> None:  # type: ignore[no-untyped-def]
        _draw_single_metric_chart(
            ax=ax,
            spec=spec,
            data=data,
            layout_mode=layout_mode,
            chart_id=chart_id,
            hidden_labels_by_chart=financial_hidden_labels_by_chart,
            pick_map=financial_pick_map,
            right_axes_by_chart=financial_right_axes_by_chart,
            legend_inside=False,
        )

    def _draw_single_fund_chart(
        ax,
        spec: dict,
        data: pd.DataFrame,
        layout_mode: str,
        chart_id: str,
    ) -> None:  # type: ignore[no-untyped-def]
        _draw_single_metric_chart(
            ax=ax,
            spec=spec,
            data=data,
            layout_mode=layout_mode,
            chart_id=chart_id,
            hidden_labels_by_chart=fundamental_hidden_labels_by_chart,
            pick_map=fundamental_pick_map,
            right_axes_by_chart=fundamental_right_axes_by_chart,
            legend_inside=True,
        )

    def _draw_financial_data(datasets: dict[str, pd.DataFrame], kpi_df: pd.DataFrame) -> None:
        nonlocal current_financial_summary, financial_pick_map, financial_cached_market
        nonlocal financial_layout_mode, financial_last_canvas_width, financial_axes_by_chart
        nonlocal financial_cards_by_chart, financial_right_axes_by_chart, financial_is_rendering, financial_resize_job
        financial_pick_map = {}
        financial_axes_by_chart = {}
        financial_cards_by_chart = {}
        financial_right_axes_by_chart = {}
        _render_kpi_table(kpi_df)

        if financial_resize_job is not None:
            try:
                root.after_cancel(financial_resize_job)
            except Exception:
                pass
            financial_resize_job = None

        financial_is_rendering = True
        try:
            fig_fin.clear()
            if not datasets:
                ax = fig_fin.add_subplot(111)
                ax.text(0.5, 0.5, "No financial data in selected range", ha="center", va="center", transform=ax.transAxes, fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                canvas_fin.draw_idle()
                fin_scroll_canvas.configure(scrollregion=fin_scroll_canvas.bbox("all"))
                current_financial_summary = "financials=none"
                return

            specs = _chart_specs_for_statement(fin_statement_var.get())
            cards = [
                ChartCard(
                    section=str(spec.get("section", "")).strip(),
                    spec=spec,
                    data=datasets.get(spec["id"], pd.DataFrame()),
                )
                for spec in specs
            ]
            canvas_width = int(fin_scroll_canvas.winfo_width())
            if canvas_width <= 1:
                canvas_width = max(int(root.winfo_width()) - 36, 900)
            financial_last_canvas_width = canvas_width
            financial_layout_mode = get_layout_mode(canvas_width - 24)
            columns = 2 if financial_layout_mode == "two_col" else 1

            target_w_px, target_h_px = get_figure_size_px(financial_layout_mode)
            probe_fig, _ = make_figure_for_card(financial_layout_mode)
            dpi = int(probe_fig.get_dpi())
            plt.close(probe_fig)

            if columns == 2:
                usable_width = max(canvas_width - 24, 2 * 620 + CARD_GAP_PX)
                card_w_px = max(620, int((usable_width - CARD_GAP_PX) / 2))
            else:
                usable_width = max(canvas_width - 24, 760)
                card_w_px = max(min(target_w_px, usable_width), 760)

            chart_h_px = max(CARD_CHART_MIN_HEIGHT_PX, target_h_px)
            card_h_px = max(CARD_MIN_HEIGHT_PX, chart_h_px + 80)

            layout_rows = build_two_column_grid(cards, columns=columns)
            if not layout_rows:
                ax = fig_fin.add_subplot(111)
                ax.text(0.5, 0.5, "표시할 카드가 없습니다", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.axis("off")
                canvas_fin.draw_idle()
                fin_scroll_canvas.configure(scrollregion=fin_scroll_canvas.bbox("all"))
                return

            section_row_count = sum(1 for row in layout_rows if row["type"] == "section")
            card_row_count = sum(1 for row in layout_rows if row["type"] == "cards")
            section_h_px = 36
            row_gap_px = 18
            fig_h_px = int(24 + section_row_count * section_h_px + card_row_count * card_h_px + max(len(layout_rows) - 1, 0) * row_gap_px + 28)
            fig_w_px = int(columns * card_w_px + (CARD_GAP_PX if columns == 2 else 0) + 28)
            fig_fin.set_dpi(dpi)
            fig_fin.set_size_inches(fig_w_px / dpi, fig_h_px / dpi, forward=True)
            canvas_fin_widget.configure(width=fig_w_px, height=fig_h_px)

            height_ratios = [section_h_px if row["type"] == "section" else card_h_px for row in layout_rows]
            grid = fig_fin.add_gridspec(
                len(layout_rows),
                columns,
                height_ratios=height_ratios,
                hspace=0.34,
                wspace=0.16 if columns == 2 else 0.0,
                left=0.05,
                right=0.95,
                top=0.985,
                bottom=0.04,
            )

            for row_idx, row in enumerate(layout_rows):
                if row["type"] == "section":
                    ax_title = fig_fin.add_subplot(grid[row_idx, :])
                    ax_title.axis("off")
                    ax_title.text(
                        0.0,
                        0.46,
                        str(row["title"]),
                        fontsize=16,
                        fontweight="bold",
                        color="#111827",
                        ha="left",
                        va="center",
                        transform=ax_title.transAxes,
                    )
                    continue

                row_cards = row["cards"]
                for col in range(columns):
                    if col >= len(row_cards):
                        ax_empty = fig_fin.add_subplot(grid[row_idx, col])
                        ax_empty.axis("off")
                        continue
                    card = row_cards[col]
                    ax = fig_fin.add_subplot(grid[row_idx, col])
                    chart_id = str(card.spec.get("id", f"card_{row_idx}_{col}"))
                    financial_axes_by_chart[chart_id] = ax
                    financial_cards_by_chart[chart_id] = card
                    _draw_single_fin_chart(ax, card.spec, card.data, financial_layout_mode, chart_id)

            canvas_fin.draw_idle()
            fin_scroll_canvas.configure(scrollregion=fin_scroll_canvas.bbox("all"))
            statement_name = {"is": "손익", "bs": "재무상태", "cf": "현금흐름"}.get(fin_statement_var.get(), fin_statement_var.get())
            current_financial_summary = f"financial_rows={len(datasets.get('base', pd.DataFrame()))} market={financial_cached_market} stmt={statement_name} mode={fin_mode_var.get()} horizon={fin_horizon_var.get()}"
        finally:
            financial_is_rendering = False

    def render_financial_tab(ticker: str, resolved_market: str) -> None:
        nonlocal financial_cached_datasets, financial_cached_kpi, financial_cached_market
        nonlocal financial_last_ticker, financial_last_market, _fin_gen

        start_bound = _parse_financial_bound(fin_from_var.get(), is_end=False)
        end_bound = _parse_financial_bound(fin_to_var.get(), is_end=True)
        if start_bound is None or start_bound < FINANCIAL_MIN_DATE:
            start_bound = FINANCIAL_MIN_DATE
        if start_bound is not None and end_bound is not None and end_bound < start_bound:
            messagebox.showerror("Financial Range Error", "Financial range end must be >= start")
            return

        statement = fin_statement_var.get()
        mode = fin_mode_var.get()
        horizon = fin_horizon_var.get()
        offline_mode = (offline_mode_var.get() == "offline")
        cache_key = (
            str(ticker).upper(),
            str(resolved_market),
            statement,
            mode,
            horizon,
            str(start_bound) if start_bound is not None else "",
            str(end_bound) if end_bound is not None else "",
            offline_mode,
        )
        bundle = financial_bundle_cache.get(cache_key)
        if bundle is not None:
            # Fast path: already cached, draw immediately on main thread
            financial_cached_datasets, financial_cached_kpi = bundle
            financial_cached_market = resolved_market
            financial_last_ticker = ticker
            financial_last_market = resolved_market
            _draw_financial_data(financial_cached_datasets, financial_cached_kpi)
            return

        # Slow path: load bundle in background thread
        _fin_gen += 1
        gen = _fin_gen
        info_var.set(f"[{ticker}] 재무제표 로딩 중...")

        def _fin_bg() -> None:
            try:
                result_bundle = _build_financial_view_frame(
                    ticker=ticker,
                    market=resolved_market,
                    start_bound=start_bound,
                    end_bound=end_bound,
                    statement=statement,
                    mode=mode,
                    horizon=horizon,
                    offline_mode=offline_mode,
                )
            except Exception as exc:  # noqa: BLE001
                root.after(0, lambda e=exc: messagebox.showerror("Financial Render Error", str(e)))
                return
            if gen != _fin_gen:
                return
            root.after(0, lambda b=result_bundle, k=cache_key, g=gen: _fin_apply(b, k, ticker, resolved_market, g))

        def _fin_apply(
            result_bundle: tuple,
            key: tuple,
            t: str,
            rm: str,
            g: int,
        ) -> None:
            nonlocal financial_cached_datasets, financial_cached_kpi, financial_cached_market
            nonlocal financial_last_ticker, financial_last_market
            if g != _fin_gen:
                return
            financial_bundle_cache[key] = result_bundle
            financial_cached_datasets, financial_cached_kpi = result_bundle
            financial_cached_market = rm
            financial_last_ticker = t
            financial_last_market = rm
            _draw_financial_data(financial_cached_datasets, financial_cached_kpi)

        threading.Thread(target=_fin_bg, daemon=True).start()

    def _resolve_market_for_tab_load(ticker: str, market_name: str) -> tuple[str, str]:
        resolved_ticker = str(ticker).strip().upper()
        resolved_market_name = str(market_name).strip().lower() or "auto"
        _, source = load_price_dataframe(ticker=resolved_ticker, market=resolved_market_name)
        if hasattr(source, "parent") and getattr(source, "parent") is not None:
            return resolved_ticker, str(source.parent.name)
        return resolved_ticker, ("us" if resolved_market_name == "auto" else resolved_market_name)

    def _load_financial_from_controls(ticker: str, market_name: str) -> None:
        nonlocal _fin_gen
        _fin_gen += 1
        gen = _fin_gen
        info_var.set(f"[{ticker}] 재무제표 로딩 중...")

        def _resolve_bg() -> None:
            try:
                resolved_ticker, resolved_market = _resolve_market_for_tab_load(ticker, market_name)
            except Exception as exc:  # noqa: BLE001
                root.after(0, lambda e=exc: messagebox.showerror("Financial Render Error", str(e)))
                return
            if gen != _fin_gen:
                return
            root.after(0, lambda t=resolved_ticker, m=resolved_market: render_financial_tab(ticker=t, resolved_market=m))

        threading.Thread(target=_resolve_bg, daemon=True).start()

    def _purge_fin_pick_map(chart_id: str) -> None:
        nonlocal financial_pick_map
        for artist, mapped in list(financial_pick_map.items()):
            if mapped[0] == chart_id:
                financial_pick_map.pop(artist, None)

    def _redraw_single_fin_chart(chart_id: str) -> None:
        card = financial_cards_by_chart.get(chart_id)
        ax = financial_axes_by_chart.get(chart_id)
        if card is None or ax is None:
            if financial_cached_datasets is not None and financial_cached_kpi is not None:
                _draw_financial_data(financial_cached_datasets, financial_cached_kpi)
            return
        _purge_fin_pick_map(chart_id)
        _draw_single_fin_chart(ax, card.spec, card.data, financial_layout_mode, chart_id)
        canvas_fin.draw_idle()
        fin_scroll_canvas.configure(scrollregion=fin_scroll_canvas.bbox("all"))

    def _toggle_financial_label(chart_id: str, label: str) -> None:
        nonlocal financial_hidden_labels_by_chart
        hidden_labels = set(financial_hidden_labels_by_chart.get(chart_id, set()))
        if label in hidden_labels:
            hidden_labels.remove(label)
        else:
            hidden_labels.add(label)
        financial_hidden_labels_by_chart[chart_id] = hidden_labels
        _redraw_single_fin_chart(chart_id)

    def on_fin_pick(event) -> None:  # type: ignore[no-untyped-def]
        picked = financial_pick_map.get(event.artist)
        if picked is None:
            return
        chart_id, label = picked
        _toggle_financial_label(chart_id, label)

    def on_fin_click(event) -> None:  # type: ignore[no-untyped-def]
        if not financial_pick_map:
            return
        for artist, picked in list(financial_pick_map.items()):
            try:
                contains, _ = artist.contains(event)
            except Exception:
                continue
            if contains:
                chart_id, label = picked
                _toggle_financial_label(chart_id, label)
                return

    def on_fin_control_change(*_args) -> None:
        nonlocal _fin_debounce_job
        if _fin_debounce_job is not None:
            try:
                root.after_cancel(_fin_debounce_job)
            except Exception:
                pass
        def _do_fin_load() -> None:
            nonlocal _fin_debounce_job
            _fin_debounce_job = None
            if bool(financial_loader["use_chart_var"].get()):
                _ticker = ticker_var.get().strip().upper()
                _mkt = market_var.get().strip().lower() or "auto"
            else:
                _ticker = financial_loader["ticker_var"].get().strip().upper()
                _mkt = financial_loader["market_var"].get().strip().lower() or "auto"
            if not _ticker:
                return
            _load_financial_from_controls(_ticker, _mkt)
        _fin_debounce_job = root.after(250, _do_fin_load)

    def _draw_fundamental_data(datasets: dict[str, pd.DataFrame], kpi_df: pd.DataFrame) -> None:
        nonlocal current_fundamental_summary, fundamental_pick_map, fundamental_cached_market
        nonlocal fundamental_layout_mode, fundamental_last_canvas_width, fundamental_axes_by_chart
        nonlocal fundamental_cards_by_chart, fundamental_right_axes_by_chart, fundamental_is_rendering, fundamental_resize_job
        fundamental_pick_map = {}
        fundamental_axes_by_chart = {}
        fundamental_cards_by_chart = {}
        fundamental_right_axes_by_chart = {}
        _render_fund_kpi_table(kpi_df)

        if fundamental_resize_job is not None:
            try:
                root.after_cancel(fundamental_resize_job)
            except Exception:
                pass
            fundamental_resize_job = None

        fundamental_is_rendering = True
        try:
            fig_fund.clear()
            if not datasets:
                ax = fig_fund.add_subplot(111)
                ax.text(0.5, 0.5, "No fundamental data in selected range", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                canvas_fund.draw_idle()
                fund_scroll_canvas.configure(scrollregion=fund_scroll_canvas.bbox("all"))
                current_fundamental_summary = "fundamental=none"
                return

            specs = _fundamental_chart_specs(fund_category_var.get())
            cards: list[ChartCard] = []
            for spec in specs:
                frame = datasets.get(spec["id"])
                if frame is None or frame.empty:
                    continue
                cards.append(
                    ChartCard(
                        section=str(spec.get("section", "")).strip(),
                        spec=spec,
                        data=frame,
                    )
                )
            if not cards:
                ax = fig_fund.add_subplot(111)
                ax.text(0.5, 0.5, "표시할 카드가 없습니다", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.axis("off")
                canvas_fund.draw_idle()
                fund_scroll_canvas.configure(scrollregion=fund_scroll_canvas.bbox("all"))
                current_fundamental_summary = "fundamental=none"
                return

            canvas_width = int(fund_scroll_canvas.winfo_width())
            if canvas_width <= 1:
                canvas_width = max(int(root.winfo_width()) - 36, 900)
            fundamental_last_canvas_width = canvas_width
            fundamental_layout_mode = get_layout_mode(canvas_width - 24)
            columns = 2 if fundamental_layout_mode == "two_col" else 1

            target_w_px, target_h_px = get_figure_size_px(fundamental_layout_mode)
            probe_fig, _ = make_figure_for_card(fundamental_layout_mode)
            dpi = int(probe_fig.get_dpi())
            plt.close(probe_fig)

            if columns == 2:
                usable_width = max(canvas_width - 24, 2 * 620 + CARD_GAP_PX)
                card_w_px = max(620, int((usable_width - CARD_GAP_PX) / 2))
            else:
                usable_width = max(canvas_width - 24, 760)
                card_w_px = max(min(target_w_px, usable_width), 760)

            chart_h_px = max(CARD_CHART_MIN_HEIGHT_PX, target_h_px)
            card_h_px = max(CARD_MIN_HEIGHT_PX, chart_h_px + 80)

            layout_rows = build_two_column_grid(cards, columns=columns)
            section_row_count = sum(1 for row in layout_rows if row["type"] == "section")
            card_row_count = sum(1 for row in layout_rows if row["type"] == "cards")
            section_h_px = 36
            row_gap_px = 18
            fig_h_px = int(24 + section_row_count * section_h_px + card_row_count * card_h_px + max(len(layout_rows) - 1, 0) * row_gap_px + 28)
            fig_w_px = int(columns * card_w_px + (CARD_GAP_PX if columns == 2 else 0) + 28)
            fig_fund.set_dpi(dpi)
            fig_fund.set_size_inches(fig_w_px / dpi, fig_h_px / dpi, forward=True)
            canvas_fund_widget.configure(width=fig_w_px, height=fig_h_px)

            height_ratios = [section_h_px if row["type"] == "section" else card_h_px for row in layout_rows]
            grid = fig_fund.add_gridspec(
                len(layout_rows),
                columns,
                height_ratios=height_ratios,
                hspace=0.34,
                wspace=0.16 if columns == 2 else 0.0,
                left=0.05,
                right=0.95,
                top=0.985,
                bottom=0.04,
            )

            for row_idx, row in enumerate(layout_rows):
                if row["type"] == "section":
                    ax_title = fig_fund.add_subplot(grid[row_idx, :])
                    ax_title.axis("off")
                    ax_title.text(
                        0.0,
                        0.46,
                        str(row["title"]),
                        fontsize=16,
                        fontweight="bold",
                        color="#111827",
                        ha="left",
                        va="center",
                        transform=ax_title.transAxes,
                    )
                    continue

                row_cards = row["cards"]
                for col in range(columns):
                    if col >= len(row_cards):
                        ax_empty = fig_fund.add_subplot(grid[row_idx, col])
                        ax_empty.axis("off")
                        continue
                    card = row_cards[col]
                    ax = fig_fund.add_subplot(grid[row_idx, col])
                    chart_id = str(card.spec.get("id", f"fund_{row_idx}_{col}"))
                    fundamental_axes_by_chart[chart_id] = ax
                    fundamental_cards_by_chart[chart_id] = card
                    _draw_single_fund_chart(ax, card.spec, card.data, fundamental_layout_mode, chart_id)

            canvas_fund.draw_idle()
            fund_scroll_canvas.configure(scrollregion=fund_scroll_canvas.bbox("all"))
            category_label = _FUNDAMENTAL_CATEGORY_LABELS.get(fund_category_var.get(), fund_category_var.get())
            current_fundamental_summary = (
                f"fund_rows={len(datasets.get('base', pd.DataFrame()))} market={fundamental_cached_market} "
                f"cat={category_label} mode={fund_mode_var.get()} horizon={fund_horizon_var.get()}"
            )
        finally:
            fundamental_is_rendering = False

    def render_fundamental_tab(ticker: str, resolved_market: str) -> None:
        nonlocal fundamental_cached_datasets, fundamental_cached_kpi, fundamental_cached_market
        nonlocal fundamental_last_ticker, fundamental_last_market, _fund_gen

        start_bound = _parse_financial_bound(fin_from_var.get(), is_end=False)
        end_bound = _parse_financial_bound(fin_to_var.get(), is_end=True)
        if start_bound is None or start_bound < FINANCIAL_MIN_DATE:
            start_bound = FINANCIAL_MIN_DATE
        if start_bound is not None and end_bound is not None and end_bound < start_bound:
            messagebox.showerror("Financial Range Error", "Financial range end must be >= start")
            return

        category = fund_category_var.get()
        mode = fund_mode_var.get()
        horizon = fund_horizon_var.get()
        offline_mode = (offline_mode_var.get() == "offline")
        cache_key = (
            str(ticker).upper(),
            str(resolved_market),
            category,
            mode,
            horizon,
            str(start_bound) if start_bound is not None else "",
            str(end_bound) if end_bound is not None else "",
            offline_mode,
        )
        bundle = fundamental_bundle_cache.get(cache_key)
        if bundle is not None:
            # Fast path: already cached, draw immediately on main thread
            fundamental_cached_datasets, fundamental_cached_kpi = bundle
            fundamental_cached_market = resolved_market
            fundamental_last_ticker = ticker
            fundamental_last_market = resolved_market
            _draw_fundamental_data(fundamental_cached_datasets, fundamental_cached_kpi)
            return

        # Slow path: load bundle in background thread
        _fund_gen += 1
        gen = _fund_gen
        info_var.set(f"[{ticker}] 팩터 분석 로딩 중...")

        def _fund_bg() -> None:
            try:
                result_bundle = _build_fundamental_view_frame(
                    ticker=ticker,
                    market=resolved_market,
                    start_bound=start_bound,
                    end_bound=end_bound,
                    category=category,
                    mode=mode,
                    horizon=horizon,
                    offline_mode=offline_mode,
                )
            except Exception as exc:  # noqa: BLE001
                root.after(0, lambda e=exc: messagebox.showerror("Fundamental Render Error", str(e)))
                return
            if gen != _fund_gen:
                return
            root.after(0, lambda b=result_bundle, k=cache_key, g=gen: _fund_apply(b, k, ticker, resolved_market, g))

        def _fund_apply(
            result_bundle: tuple,
            key: tuple,
            t: str,
            rm: str,
            g: int,
        ) -> None:
            nonlocal fundamental_cached_datasets, fundamental_cached_kpi, fundamental_cached_market
            nonlocal fundamental_last_ticker, fundamental_last_market
            if g != _fund_gen:
                return
            fundamental_bundle_cache[key] = result_bundle
            fundamental_cached_datasets, fundamental_cached_kpi = result_bundle
            fundamental_cached_market = rm
            fundamental_last_ticker = t
            fundamental_last_market = rm
            _draw_fundamental_data(fundamental_cached_datasets, fundamental_cached_kpi)

        threading.Thread(target=_fund_bg, daemon=True).start()

    def _load_fundamental_from_controls(ticker: str, market_name: str) -> None:
        nonlocal _fund_gen
        _fund_gen += 1
        gen = _fund_gen
        info_var.set(f"[{ticker}] 팩터 분석 로딩 중...")

        def _resolve_bg() -> None:
            try:
                resolved_ticker, resolved_market = _resolve_market_for_tab_load(ticker, market_name)
            except Exception as exc:  # noqa: BLE001
                root.after(0, lambda e=exc: messagebox.showerror("Fundamental Render Error", str(e)))
                return
            if gen != _fund_gen:
                return
            root.after(0, lambda t=resolved_ticker, m=resolved_market: render_fundamental_tab(ticker=t, resolved_market=m))

        threading.Thread(target=_resolve_bg, daemon=True).start()

    def _purge_fund_pick_map(chart_id: str) -> None:
        nonlocal fundamental_pick_map
        for artist, mapped in list(fundamental_pick_map.items()):
            if mapped[0] == chart_id:
                fundamental_pick_map.pop(artist, None)

    def _redraw_single_fund_chart(chart_id: str) -> None:
        card = fundamental_cards_by_chart.get(chart_id)
        ax = fundamental_axes_by_chart.get(chart_id)
        if card is None or ax is None:
            if fundamental_cached_datasets is not None and fundamental_cached_kpi is not None:
                _draw_fundamental_data(fundamental_cached_datasets, fundamental_cached_kpi)
            return
        _purge_fund_pick_map(chart_id)
        _draw_single_fund_chart(ax, card.spec, card.data, fundamental_layout_mode, chart_id)
        canvas_fund.draw_idle()
        fund_scroll_canvas.configure(scrollregion=fund_scroll_canvas.bbox("all"))

    def _toggle_fundamental_label(chart_id: str, label: str) -> None:
        nonlocal fundamental_hidden_labels_by_chart
        hidden_labels = set(fundamental_hidden_labels_by_chart.get(chart_id, set()))
        if label in hidden_labels:
            hidden_labels.remove(label)
        else:
            hidden_labels.add(label)
        fundamental_hidden_labels_by_chart[chart_id] = hidden_labels
        _redraw_single_fund_chart(chart_id)

    def on_fund_pick(event) -> None:  # type: ignore[no-untyped-def]
        picked = fundamental_pick_map.get(event.artist)
        if picked is None:
            return
        chart_id, label = picked
        _toggle_fundamental_label(chart_id, label)

    def on_fund_click(event) -> None:  # type: ignore[no-untyped-def]
        if not fundamental_pick_map:
            return
        for artist, picked in list(fundamental_pick_map.items()):
            try:
                contains, _ = artist.contains(event)
            except Exception:
                continue
            if contains:
                chart_id, label = picked
                _toggle_fundamental_label(chart_id, label)
                return

    def on_fund_control_change(*_args) -> None:
        nonlocal _fund_debounce_job
        if _fund_debounce_job is not None:
            try:
                root.after_cancel(_fund_debounce_job)
            except Exception:
                pass
        def _do_fund_load() -> None:
            nonlocal _fund_debounce_job
            _fund_debounce_job = None
            if bool(fundamental_loader["use_chart_var"].get()):
                _ticker = ticker_var.get().strip().upper()
                _mkt = market_var.get().strip().lower() or "auto"
            else:
                _ticker = fundamental_loader["ticker_var"].get().strip().upper()
                _mkt = fundamental_loader["market_var"].get().strip().lower() or "auto"
            if not _ticker:
                return
            _load_fundamental_from_controls(_ticker, _mkt)
        _fund_debounce_job = root.after(250, _do_fund_load)

    def render_chart() -> None:
        nonlocal _chart_gen, _chart_loaded_df, _chart_loaded_valuation

        ticker = ticker_var.get().strip().upper()
        market_name = market_var.get().strip() or "auto"
        offline = (offline_mode_var.get() == "offline")
        use_fin = bool(financial_loader["use_chart_var"].get())
        use_fund = bool(fundamental_loader["use_chart_var"].get())

        if not ticker:
            messagebox.showerror("Input Error", "Ticker is required")
            return

        _chart_gen += 1
        gen = _chart_gen
        info_var.set(f"[{ticker}] 데이터 로딩 중...")
        chart_load_btn.configure(state="disabled")
        chart_ticker_var.set("")
        chart_price_var.set("")
        chart_stats_var.set(f"{ticker} 데이터 로딩 중...")
        tv_chart_status_var.set("")

        def _bg_load() -> None:
            try:
                df, source = load_price_dataframe(ticker=ticker, market=market_name)
                out = df.copy()
                for col in ["Open", "High", "Low", "Close"]:
                    if col in out.columns:
                        out[col] = pd.to_numeric(out[col], errors="coerce")
                out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
                if out.empty:
                    raise ValueError("가격 데이터가 없습니다")
                if gen != _chart_gen:
                    root.after(0, lambda: chart_load_btn.configure(state="normal"))
                    return
                val_close = out["Close"]
                if valuation_price_field == "adjclose" and "Adj Close" in out.columns:
                    val_close = pd.to_numeric(out["Adj Close"], errors="coerce").fillna(out["Close"])
                valuation = load_valuation_series(
                    ticker=ticker,
                    market=source.parent.name,
                    price_index=out.index,
                    close_series=val_close,
                    stock_splits_series=out["Stock Splits"] if "Stock Splits" in out.columns else None,
                    price_field=valuation_price_field,
                    per_negative=valuation_per_negative,
                    band_window=valuation_band_window,
                    band_quantiles=valuation_band_quantiles,
                    outlier=valuation_outlier,
                    offline_mode=offline,
                )
            except Exception as exc:  # noqa: BLE001
                def _on_err(e: Exception) -> None:
                    chart_load_btn.configure(state="normal")
                    chart_stats_var.set(f"오류: {e}")
                    messagebox.showerror("Load Error", str(e))
                root.after(0, lambda e=exc: _on_err(e))
                return
            if gen != _chart_gen:
                root.after(0, lambda: chart_load_btn.configure(state="normal"))
                return
            result = {
                "df": out, "source": source, "valuation": valuation,
                "ticker": ticker, "market_name": market_name,
                "use_fin": use_fin, "use_fund": use_fund,
            }
            root.after(0, lambda r=result, g=gen: _apply_chart(r, g))

        def _apply_chart(result: dict, gen: int) -> None:
            nonlocal current_summary, last_loaded_ticker, last_loaded_market
            nonlocal _chart_loaded_df, _chart_loaded_valuation

            chart_load_btn.configure(state="normal")
            if gen != _chart_gen:
                return

            out = result["df"]
            source = result["source"]
            valuation = result["valuation"]
            r_ticker = result["ticker"]
            r_market_name = result["market_name"]
            r_use_fin = result["use_fin"]
            r_use_fund = result["use_fund"]

            _chart_loaded_df = out
            _chart_loaded_valuation = valuation
            last_loaded_ticker = r_ticker
            last_loaded_market = r_market_name

            # Update info panel
            resolved_market = source.parent.name
            close_col = "Adj Close" if "Adj Close" in out.columns else "Close"
            close_clean = pd.to_numeric(out[close_col], errors="coerce").dropna()
            last_price = float(close_clean.iloc[-1]) if not close_clean.empty else float("nan")
            prev_price = float(close_clean.iloc[-2]) if len(close_clean) >= 2 else last_price
            chg = last_price - prev_price
            chg_pct = (chg / prev_price * 100) if prev_price else 0.0
            chg_sign = "▲" if chg >= 0 else "▼"
            chart_price_lbl.configure(fg="#1fa774" if chg >= 0 else "#d84a4a")

            chart_ticker_var.set(f"{r_ticker}  ·  {resolved_market.upper()}")
            chart_price_var.set(
                f"{last_price:,.2f}   {chg_sign} {abs(chg):,.2f} ({chg_pct:+.2f}%)"
            )

            date_min = out.index.min().date()
            date_max = out.index.max().date()
            vol_clean = pd.to_numeric(out["Volume"], errors="coerce").dropna() if "Volume" in out.columns else pd.Series([], dtype=float)
            vol_last = int(vol_clean.iloc[-1]) if not vol_clean.empty else 0
            chart_stats_var.set(
                f"기간: {date_min} → {date_max}  ·  {len(out):,} 봉  ·  "
                f"최근 거래량: {vol_last:,}  ·  평가: {valuation.valuation_source}"
            )
            tv_chart_status_var.set("로드 완료 — TradingView 차트 열기 버튼을 클릭하세요.")

            current_summary = (
                f"{r_ticker} rows={len(out)} | "
                f"range={date_min} -> {date_max} | "
                f"valuation={valuation.valuation_source}"
            )
            info_var.set(current_summary)

            if r_use_fin:
                financial_loader["ticker_var"].set(r_ticker)
                financial_loader["market_var"].set(r_market_name)
                render_financial_tab(ticker=r_ticker, resolved_market=resolved_market)
            if r_use_fund:
                fundamental_loader["ticker_var"].set(r_ticker)
                fundamental_loader["market_var"].set(r_market_name)
                render_fundamental_tab(ticker=r_ticker, resolved_market=resolved_market)

        threading.Thread(target=_bg_load, daemon=True).start()

    def _do_open_tv_chart() -> None:
        """Open (or refresh) the TradingView chart window."""
        from market_data.tv_chart import open_tv_chart as _open_tv_chart  # noqa: PLC0415

        if _chart_loaded_df is None:
            messagebox.showinfo("차트 없음", "먼저 'Load Data'를 클릭해 데이터를 로드하세요.")
            return

        ticker = last_loaded_ticker or ticker_var.get().strip().upper()
        market_name = last_loaded_market or market_var.get().strip() or "auto"
        chart_type = chart_type_var.get().strip() or "candles"

        valuation = _chart_loaded_valuation

        tv_chart_status_var.set("차트 창 여는 중...")
        open_tv_btn.configure(state="disabled")

        def _do_open() -> None:
            try:
                _open_tv_chart(
                    df=_chart_loaded_df,
                    ticker=ticker,
                    market=market_name,
                    chart_type=chart_type,
                    indicator_mode="none",
                    valuation=valuation,
                )
                root.after(500, lambda: (
                    open_tv_btn.configure(state="normal"),
                    tv_chart_status_var.set("차트 창이 열렸습니다."),
                ))
            except Exception as exc:  # noqa: BLE001
                root.after(0, lambda e=exc: (
                    messagebox.showerror("TV Chart Error", str(e)),
                    open_tv_btn.configure(state="normal"),
                    tv_chart_status_var.set(f"오류: {e}"),
                ))

        threading.Thread(target=_do_open, daemon=True).start()

    fig_fin.canvas.mpl_connect("pick_event", on_fin_pick)
    fig_fin.canvas.mpl_connect("button_press_event", on_fin_click)
    fig_fund.canvas.mpl_connect("pick_event", on_fund_pick)
    fig_fund.canvas.mpl_connect("button_press_event", on_fund_click)
    chart_load_btn.configure(command=render_chart)
    open_tv_btn.configure(command=_do_open_tv_chart)
    apply_fin_range_btn.configure(command=render_chart)
    bt_run_screen_btn.configure(command=_run_backtest_screen)
    bt_run_strategy_btn.configure(command=_run_backtest_strategy)
    bt_compare_btn.configure(command=_run_backtest_compare)

    offline_mode_var.trace_add("write", lambda *_args: render_chart())
    fin_statement_var.trace_add("write", on_fin_control_change)
    fin_mode_var.trace_add("write", on_fin_control_change)
    fin_horizon_var.trace_add("write", on_fin_control_change)
    fund_category_var.trace_add("write", on_fund_control_change)
    fund_mode_var.trace_add("write", on_fund_control_change)
    fund_horizon_var.trace_add("write", on_fund_control_change)

    for widget in (fin_scroll_canvas, canvas_fin_widget, kpi_table):
        widget.bind("<MouseWheel>", _on_fin_mousewheel)
        widget.bind("<Button-4>", _on_fin_mousewheel)
        widget.bind("<Button-5>", _on_fin_mousewheel)

    for widget in (fund_scroll_canvas, canvas_fund_widget, fund_kpi_table):
        widget.bind("<MouseWheel>", _on_fund_mousewheel)
        widget.bind("<Button-4>", _on_fund_mousewheel)
        widget.bind("<Button-5>", _on_fund_mousewheel)

    ticker_entry.bind("<Return>", lambda _event: render_chart())
    market_var.trace_add("write", lambda *_: render_chart())
    fin_from_entry.bind("<Return>", lambda _event: render_chart())
    fin_to_entry.bind("<Return>", lambda _event: render_chart())
    bt_strategy_entry.bind("<Return>", lambda _event: _run_backtest_strategy())
    bt_screen_entry.bind("<Return>", lambda _event: _run_backtest_screen())
    bt_start_entry.bind("<Return>", lambda _event: _run_backtest_screen())
    bt_end_entry.bind("<Return>", lambda _event: _run_backtest_screen())
    bt_holdings_entry.bind("<Return>", lambda _event: _run_backtest_screen())
    # ------------------------------------------------------------------
    # Status bar (bottom)
    # ------------------------------------------------------------------
    _status_bar = tk.Frame(root, bg=BORDER, height=26)
    _status_bar.pack(fill="x", side="bottom")
    _status_bar.pack_propagate(False)

    _status_info = tk.Label(
        _status_bar, textvariable=info_var,
        bg=BORDER, fg=TEXT_SUB,
        font=(ui_font_family, 10), anchor="w",
    )
    _status_info.pack(side="left", fill="y", padx=(10, 0))

    _status_right = tk.Label(
        _status_bar, text="Market Data Viewer  |  오프라인 모드",
        bg=BORDER, fg=TEXT_MUTED,
        font=(ui_font_family, 10), anchor="e",
    )
    _status_right.pack(side="right", fill="y", padx=(0, 12))

    # Register non-ttk widgets for dynamic theme recolouring
    _theme_widgets.extend([
        (_status_bar,    "BORDER",     None),
        (_status_info,   "BORDER",     "TEXT_SUB"),
        (_status_right,  "BORDER",     "TEXT_MUTED"),
        (chart_ticker_lbl, "SURFACE_BG", "TEXT_MAIN"),
        (chart_price_lbl,  "SURFACE_BG", None),
        (chart_stats_lbl,  "SURFACE_BG", "TEXT_SUB"),
        (fin_scroll_canvas,  "PANEL_BG", None),
        (fund_scroll_canvas, "PANEL_BG", None),
    ])

    ticker_entry.focus_set()
    render_chart()
    root.mainloop()
    return 0
