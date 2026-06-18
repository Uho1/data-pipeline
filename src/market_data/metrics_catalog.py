from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


MetricComputeFn = Callable[[pd.DataFrame], pd.Series]
MetricFormatFn = Callable[[float | int | None], str]


@dataclass(frozen=True)
class MetricDefinition:
    id: str
    label: str
    category: str
    direction_default: str
    unit: str
    required_columns: tuple[str, ...]
    filterable: bool
    rankable: bool
    compute_fn: MetricComputeFn
    formatter: MetricFormatFn
    description: str = ""
    formula: str = ""
    data_source: str = "factor_panel"
    missing_policy: str = "nan"


def _fmt_ratio(v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{float(v):.2f}"


def _fmt_pct(v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{float(v) * 100.0:.2f}%"


def _fmt_usd(v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{float(v):,.0f}"


def _series(panel: pd.DataFrame, col: str) -> pd.Series:
    if col not in panel.columns:
        return pd.Series(np.nan, index=panel.index, dtype=float)
    return pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = pd.to_numeric(numer, errors="coerce") / pd.to_numeric(denom, errors="coerce").replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _by_ticker(panel: pd.DataFrame, col: str, fn: Callable[[pd.Series], pd.Series]) -> pd.Series:
    src = _series(panel, col)
    if isinstance(panel.index, pd.MultiIndex) and panel.index.nlevels >= 2:
        return src.groupby(level=1, sort=False).transform(fn)
    return fn(src)


def _compute_per(panel: pd.DataFrame) -> pd.Series:
    ni = _series(panel, "ni_ttm")
    out = _safe_divide(_series(panel, "market_cap"), ni)
    return out.where(ni > 0.0)


def _compute_pbr(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "market_cap"), _series(panel, "equity"))


def _compute_psr(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "market_cap"), _series(panel, "revenue_ttm"))


def _compute_pcr(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "market_cap"), _series(panel, "ocf_ttm"))


def _compute_ev(panel: pd.DataFrame) -> pd.Series:
    return _series(panel, "market_cap") + _series(panel, "total_debt") - _series(panel, "cash_and_equivalents")


def _compute_ev_ebitda(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_compute_ev(panel), _series(panel, "ebitda_ttm"))


def _compute_fcf_yield(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "fcf_ttm"), _series(panel, "market_cap"))


def _compute_earnings_yield(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "ni_ttm"), _series(panel, "market_cap"))


def _compute_dividend_yield(panel: pd.DataFrame) -> pd.Series:
    if "dividend_yield" in panel.columns:
        return _series(panel, "dividend_yield")
    return _safe_divide(_series(panel, "dividends_ttm"), _series(panel, "market_cap"))


def _compute_roe(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "ni_ttm"), _series(panel, "avg_equity"))


def _compute_roa(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "ni_ttm"), _series(panel, "avg_assets"))


def _compute_gpa(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "gross_profit_ttm"), _series(panel, "avg_assets"))


def _compute_roic(panel: pd.DataFrame) -> pd.Series:
    tax_rate = 0.21
    nopat = _series(panel, "op_income_ttm") * (1.0 - tax_rate)
    if "cash_and_equivalents" in panel.columns:
        cash = _series(panel, "cash_and_equivalents").fillna(0.0)
    else:
        cash = pd.Series(0.0, index=panel.index, dtype=float)
    invested_capital = _series(panel, "equity") + _series(panel, "liabilities") - cash
    return _safe_divide(nopat, invested_capital)


def _compute_gross_margin(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "gross_profit_ttm"), _series(panel, "revenue_ttm"))


def _compute_op_margin(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "op_income_ttm"), _series(panel, "revenue_ttm"))


def _compute_net_margin(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "ni_ttm"), _series(panel, "revenue_ttm"))


def _compute_accruals(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "ni_ttm") - _series(panel, "ocf_ttm"), _series(panel, "avg_assets"))


def _compute_debt_to_equity(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "liabilities"), _series(panel, "equity"))


def _compute_debt_to_assets(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "liabilities"), _series(panel, "assets"))


def _compute_interest_coverage(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "op_income_ttm"), _series(panel, "interest_expense_ttm"))


def _compute_current_ratio(panel: pd.DataFrame) -> pd.Series:
    return _safe_divide(_series(panel, "current_assets"), _series(panel, "current_liabilities"))


def _compute_momentum(panel: pd.DataFrame, periods: int, skip_recent: int = 0) -> pd.Series:
    close = _series(panel, "close")
    if isinstance(panel.index, pd.MultiIndex) and panel.index.nlevels >= 2:
        by_ticker = close.groupby(level=1, sort=False)
        if skip_recent > 0:
            left = by_ticker.transform(lambda s: s.shift(skip_recent))
        else:
            left = close
        base = by_ticker.transform(lambda s: s.shift(periods))
        return _safe_divide(left, base) - 1.0

    left = close.shift(skip_recent) if skip_recent > 0 else close
    base = close.shift(periods)
    return _safe_divide(left, base) - 1.0


def _compute_vol(panel: pd.DataFrame, window: int) -> pd.Series:
    close = _series(panel, "close")
    if isinstance(panel.index, pd.MultiIndex) and panel.index.nlevels >= 2:
        by_ticker = close.groupby(level=1, sort=False)
        ret = by_ticker.transform(lambda s: s.pct_change())
        return by_ticker.transform(lambda s: s.pct_change().rolling(window=window, min_periods=max(5, window // 4)).std()) * np.sqrt(252.0)
    ret = close.pct_change()
    return ret.rolling(window=window, min_periods=max(5, window // 4)).std() * np.sqrt(252.0)


def _compute_sma(panel: pd.DataFrame, window: int) -> pd.Series:
    close = _series(panel, "close")
    if isinstance(panel.index, pd.MultiIndex) and panel.index.nlevels >= 2:
        by_ticker = close.groupby(level=1, sort=False)
        return by_ticker.transform(lambda s: s.rolling(window=window, min_periods=max(5, window // 4)).mean())
    return close.rolling(window=window, min_periods=max(5, window // 4)).mean()


def _compute_rsi(panel: pd.DataFrame, window: int) -> pd.Series:
    close = _series(panel, "close")

    def _rsi_series(s: pd.Series) -> pd.Series:
        d = s.diff()
        up = d.clip(lower=0.0)
        down = (-d).clip(lower=0.0)
        up_ma = up.rolling(window=window, min_periods=window).mean()
        down_ma = down.rolling(window=window, min_periods=window).mean()
        rs = _safe_divide(up_ma, down_ma)
        return 100.0 - (100.0 / (1.0 + rs))

    if isinstance(panel.index, pd.MultiIndex) and panel.index.nlevels >= 2:
        return close.groupby(level=1, sort=False).transform(_rsi_series)
    return _rsi_series(close)


def _compute_boll_z(panel: pd.DataFrame) -> pd.Series:
    close = _series(panel, "close")

    def _bz(s: pd.Series) -> pd.Series:
        ma = s.rolling(window=20, min_periods=10).mean()
        sd = s.rolling(window=20, min_periods=10).std()
        return _safe_divide(s - ma, 2.0 * sd)

    if isinstance(panel.index, pd.MultiIndex) and panel.index.nlevels >= 2:
        return close.groupby(level=1, sort=False).transform(_bz)
    return _bz(close)


def _compute_piotroski_f(panel: pd.DataFrame) -> pd.Series:
    ni = _series(panel, "ni_ttm")
    ocf = _series(panel, "ocf_ttm")
    assets = _series(panel, "assets")
    liabilities = _series(panel, "liabilities")
    equity = _series(panel, "equity")
    revenue = _series(panel, "revenue_ttm")
    gp = _series(panel, "gross_profit_ttm")
    shares = _series(panel, "shares")

    roa = _safe_divide(ni, assets)
    cfo = ocf
    leverage = _safe_divide(liabilities, assets)
    current_ratio = _safe_divide(assets, liabilities)
    gross_margin = _safe_divide(gp, revenue)
    asset_turnover = _safe_divide(revenue, assets)

    if isinstance(panel.index, pd.MultiIndex) and panel.index.nlevels >= 2:
        by_ticker = panel.index.get_level_values(1)
        roa_prev = roa.groupby(by_ticker, sort=False).shift(252)
        lev_prev = leverage.groupby(by_ticker, sort=False).shift(252)
        cr_prev = current_ratio.groupby(by_ticker, sort=False).shift(252)
        gm_prev = gross_margin.groupby(by_ticker, sort=False).shift(252)
        at_prev = asset_turnover.groupby(by_ticker, sort=False).shift(252)
        shares_prev = shares.groupby(by_ticker, sort=False).shift(252)
    else:
        roa_prev = roa.shift(252)
        lev_prev = leverage.shift(252)
        cr_prev = current_ratio.shift(252)
        gm_prev = gross_margin.shift(252)
        at_prev = asset_turnover.shift(252)
        shares_prev = shares.shift(252)

    conds = [
        roa > 0.0,
        cfo > 0.0,
        roa > roa_prev,
        cfo > ni,
        leverage < lev_prev,
        current_ratio > cr_prev,
        shares <= shares_prev,
        gross_margin > gm_prev,
        asset_turnover > at_prev,
    ]

    score = pd.Series(0.0, index=panel.index, dtype=float)
    for c in conds:
        score = score + c.fillna(False).astype(float)
    return score


def _compute_market_cap(panel: pd.DataFrame) -> pd.Series:
    return _series(panel, "market_cap")


def _compute_dollar_volume_20d(panel: pd.DataFrame) -> pd.Series:
    return _series(panel, "dollar_volume_20d")


def _compute_price(panel: pd.DataFrame) -> pd.Series:
    if "price" in panel.columns:
        return _series(panel, "price")
    return _series(panel, "close")


def _compute_named(panel: pd.DataFrame, col: str) -> pd.Series:
    return _series(panel, col)


def _compute_price_above_sma200(panel: pd.DataFrame) -> pd.Series:
    close = _series(panel, "close")
    sma200 = _compute_sma(panel, 200)
    return (close > sma200).astype(float)


def _compute_sma50_above_sma200(panel: pd.DataFrame) -> pd.Series:
    sma50 = _compute_sma(panel, 50)
    sma200 = _compute_sma(panel, 200)
    return (sma50 > sma200).astype(float)


def _compute_ma_gap(panel: pd.DataFrame) -> pd.Series:
    close = _series(panel, "close")
    sma200 = _compute_sma(panel, 200)
    return _safe_divide(close, sma200) - 1.0


METRICS: dict[str, MetricDefinition] = {
    "per": MetricDefinition("per", "PER", "valuation", "asc", "x", ("market_cap", "ni_ttm"), True, True, _compute_per, _fmt_ratio, "주가수익비율", "market_cap / ni_ttm"),
    "pbr": MetricDefinition("pbr", "PBR", "valuation", "asc", "x", ("market_cap", "equity"), True, True, _compute_pbr, _fmt_ratio, "주가순자산비율", "market_cap / equity"),
    "psr": MetricDefinition("psr", "PSR", "valuation", "asc", "x", ("market_cap", "revenue_ttm"), True, True, _compute_psr, _fmt_ratio, "주가매출비율", "market_cap / revenue_ttm"),
    "pcr": MetricDefinition("pcr", "PCR", "valuation", "asc", "x", ("market_cap", "ocf_ttm"), True, True, _compute_pcr, _fmt_ratio, "주가현금흐름비율", "market_cap / ocf_ttm"),
    "ev_ebitda": MetricDefinition("ev_ebitda", "EV/EBITDA", "valuation", "asc", "x", ("market_cap", "total_debt", "cash_and_equivalents", "ebitda_ttm"), True, True, _compute_ev_ebitda, _fmt_ratio, "기업가치 대비 EBITDA"),
    "fcf_yield": MetricDefinition("fcf_yield", "FCF Yield", "valuation", "desc", "pct", ("fcf_ttm", "market_cap"), True, True, _compute_fcf_yield, _fmt_pct, "잉여현금흐름 수익률", "fcf_ttm / market_cap"),
    "earnings_yield": MetricDefinition("earnings_yield", "Earnings Yield", "valuation", "desc", "pct", ("ni_ttm", "market_cap"), True, True, _compute_earnings_yield, _fmt_pct, "이익수익률", "ni_ttm / market_cap"),
    "dividend_yield": MetricDefinition("dividend_yield", "Dividend Yield", "valuation", "desc", "pct", ("dividends_ttm", "market_cap"), True, True, _compute_dividend_yield, _fmt_pct, "배당수익률"),

    "roe": MetricDefinition("roe", "ROE", "quality", "desc", "pct", ("ni_ttm", "avg_equity"), True, True, _compute_roe, _fmt_pct, "자기자본이익률"),
    "roa": MetricDefinition("roa", "ROA", "quality", "desc", "pct", ("ni_ttm", "avg_assets"), True, True, _compute_roa, _fmt_pct, "총자산이익률"),
    "roic": MetricDefinition("roic", "ROIC", "quality", "desc", "pct", ("op_income_ttm", "equity", "liabilities"), True, True, _compute_roic, _fmt_pct, "투하자본이익률(근사)"),
    "gpa": MetricDefinition("gpa", "GP/A", "quality", "desc", "pct", ("gross_profit_ttm", "avg_assets"), True, True, _compute_gpa, _fmt_pct, "Gross Profit / Assets"),
    "gross_margin": MetricDefinition("gross_margin", "Gross Margin", "quality", "desc", "pct", ("gross_profit_ttm", "revenue_ttm"), True, True, _compute_gross_margin, _fmt_pct, "매출총이익률"),
    "op_margin": MetricDefinition("op_margin", "Operating Margin", "quality", "desc", "pct", ("op_income_ttm", "revenue_ttm"), True, True, _compute_op_margin, _fmt_pct, "영업이익률"),
    "net_margin": MetricDefinition("net_margin", "Net Margin", "quality", "desc", "pct", ("ni_ttm", "revenue_ttm"), True, True, _compute_net_margin, _fmt_pct, "순이익률"),
    "accruals": MetricDefinition("accruals", "Accruals", "quality", "asc", "pct", ("ni_ttm", "ocf_ttm", "avg_assets"), True, True, _compute_accruals, _fmt_pct, "발생주의 프록시"),
    "piotroski_f": MetricDefinition("piotroski_f", "Piotroski F-Score", "quality", "desc", "score", ("ni_ttm", "ocf_ttm", "assets", "liabilities", "equity", "revenue_ttm", "gross_profit_ttm", "shares"), True, True, _compute_piotroski_f, _fmt_ratio, "가치 함정 방어 스코어(근사)"),

    "debt_to_equity": MetricDefinition("debt_to_equity", "Debt to Equity", "stability", "asc", "x", ("liabilities", "equity"), True, True, _compute_debt_to_equity, _fmt_ratio),
    "debt_to_assets": MetricDefinition("debt_to_assets", "Debt to Assets", "stability", "asc", "x", ("liabilities", "assets"), True, True, _compute_debt_to_assets, _fmt_ratio),
    "interest_coverage": MetricDefinition("interest_coverage", "Interest Coverage", "stability", "desc", "x", ("op_income_ttm", "interest_expense_ttm"), True, True, _compute_interest_coverage, _fmt_ratio),
    "current_ratio": MetricDefinition("current_ratio", "Current Ratio", "stability", "desc", "x", ("current_assets", "current_liabilities"), True, True, _compute_current_ratio, _fmt_ratio),

    "mom_12m": MetricDefinition("mom_12m", "12M Momentum", "momentum", "desc", "pct", ("close",), True, True, lambda panel: _compute_momentum(panel, 252, 0), _fmt_pct),
    "mom_12_1": MetricDefinition("mom_12_1", "12-1M Momentum", "momentum", "desc", "pct", ("close",), True, True, lambda panel: _compute_momentum(panel, 252, 21), _fmt_pct),
    "mom_6m": MetricDefinition("mom_6m", "6M Momentum", "momentum", "desc", "pct", ("close",), True, True, lambda panel: _compute_momentum(panel, 126, 0), _fmt_pct),
    "mom_3m": MetricDefinition("mom_3m", "3M Momentum", "momentum", "desc", "pct", ("close",), True, True, lambda panel: _compute_momentum(panel, 63, 0), _fmt_pct),
    "rev_1m": MetricDefinition("rev_1m", "1M Reversal", "momentum", "asc", "pct", ("close",), True, True, lambda panel: _compute_momentum(panel, 21, 0), _fmt_pct),
    "vol_20d": MetricDefinition("vol_20d", "Volatility 20D", "technical", "asc", "pct", ("close",), True, True, lambda panel: _compute_vol(panel, 20), _fmt_pct),
    "vol_60d": MetricDefinition("vol_60d", "Volatility 60D", "technical", "asc", "pct", ("close",), True, True, lambda panel: _compute_vol(panel, 60), _fmt_pct),
    "sma50": MetricDefinition("sma50", "SMA50", "technical", "desc", "usd", ("close",), True, False, lambda panel: _compute_sma(panel, 50), _fmt_usd),
    "sma200": MetricDefinition("sma200", "SMA200", "technical", "desc", "usd", ("close",), True, False, lambda panel: _compute_sma(panel, 200), _fmt_usd),
    "price_above_sma200": MetricDefinition("price_above_sma200", "Price > SMA200", "technical", "desc", "flag", ("close",), True, True, _compute_price_above_sma200, _fmt_ratio),
    "sma50_above_sma200": MetricDefinition("sma50_above_sma200", "SMA50 > SMA200", "technical", "desc", "flag", ("close",), True, True, _compute_sma50_above_sma200, _fmt_ratio),
    "ma_gap": MetricDefinition("ma_gap", "MA Gap(Price/SMA200-1)", "technical", "desc", "pct", ("close",), True, True, _compute_ma_gap, _fmt_pct),
    "rsi2": MetricDefinition("rsi2", "RSI(2)", "technical", "asc", "score", ("close",), True, True, lambda panel: _compute_rsi(panel, 2), _fmt_ratio),
    "rsi14": MetricDefinition("rsi14", "RSI(14)", "technical", "asc", "score", ("close",), True, True, lambda panel: _compute_rsi(panel, 14), _fmt_ratio),
    "boll_z": MetricDefinition("boll_z", "Bollinger Z", "technical", "asc", "z", ("close",), True, True, _compute_boll_z, _fmt_ratio),

    "market_cap": MetricDefinition("market_cap", "Market Cap", "liquidity_size", "desc", "usd", ("market_cap",), True, False, _compute_market_cap, _fmt_usd),
    "dollar_volume_20d": MetricDefinition("dollar_volume_20d", "Dollar Volume 20D", "liquidity_size", "desc", "usd", ("dollar_volume_20d",), True, False, _compute_dollar_volume_20d, _fmt_usd),
    "price": MetricDefinition("price", "Price", "liquidity_size", "desc", "usd", ("close",), True, False, _compute_price, _fmt_usd),
    "biz_segment_count": MetricDefinition("biz_segment_count", "Business Segment Count", "business", "desc", "count", ("biz_segment_count",), True, True, lambda panel: _compute_named(panel, "biz_segment_count"), _fmt_ratio),
    "biz_top1_revenue_share": MetricDefinition("biz_top1_revenue_share", "Top1 Segment Revenue Share", "business", "asc", "pct", ("biz_top1_revenue_share",), True, True, lambda panel: _compute_named(panel, "biz_top1_revenue_share"), _fmt_pct),
    "product_segment_count": MetricDefinition("product_segment_count", "Product Segment Count", "business", "desc", "count", ("product_segment_count",), True, True, lambda panel: _compute_named(panel, "product_segment_count"), _fmt_ratio),
    "geo_segment_count": MetricDefinition("geo_segment_count", "Geo Segment Count", "business", "desc", "count", ("geo_segment_count",), True, True, lambda panel: _compute_named(panel, "geo_segment_count"), _fmt_ratio),
    "geo_top1_revenue_share": MetricDefinition("geo_top1_revenue_share", "Top1 Geo Revenue Share", "business", "asc", "pct", ("geo_top1_revenue_share",), True, True, lambda panel: _compute_named(panel, "geo_top1_revenue_share"), _fmt_pct),
}


def list_metric_definitions() -> list[MetricDefinition]:
    return list(METRICS.values())


def metric_catalog_map() -> dict[str, dict[str, str | bool | tuple[str, ...]]]:
    out: dict[str, dict[str, str | bool | tuple[str, ...]]] = {}
    for mid, m in METRICS.items():
        out[mid] = {
            "label": m.label,
            "category": m.category,
            "direction_default": m.direction_default,
            "unit": m.unit,
            "required_columns": m.required_columns,
            "filterable": m.filterable,
            "rankable": m.rankable,
            "description": m.description,
            "formula": m.formula,
            "data_source": m.data_source,
            "missing_policy": m.missing_policy,
        }
    return out


def apply_metrics_catalog(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    out = panel.copy()
    availability: dict[str, bool] = {}

    for metric_id, metric in METRICS.items():
        has_columns = all(col in out.columns for col in metric.required_columns)
        if not has_columns:
            out[metric_id] = np.nan
            availability[metric_id] = False
            continue

        try:
            values = metric.compute_fn(out)
            values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        except Exception:
            values = pd.Series(np.nan, index=out.index, dtype=float)

        out[metric_id] = values
        availability[metric_id] = bool(values.notna().any())

    out.attrs["metric_availability_map"] = availability
    return out, availability
