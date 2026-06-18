from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


VALUATION_TTM_WINDOW_SIZES: dict[str, int] = {
    "5y": 20,
    "10y": 40,
}

VALUATION_TTM_QUANTILES: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
VALUATION_TTM_QUANTILE_KEYS: tuple[str, ...] = ("q10", "q25", "q50", "q75", "q90")
VALUATION_TTM_REFERENCE_LEVELS: dict[str, list[float]] = {
    "pbr": [0.5, 1.0, 2.0, 3.0, 4.0],
    "psr": [0.5, 1.0, 2.0, 3.0, 4.0],
    "per": [5.0, 10.0, 15.0, 20.0, 25.0],
    "por": [5.0, 10.0, 15.0, 20.0],
}

VALUATION_TTM_SERIES_KEYS: tuple[str, ...] = (
    "price",
    "market_cap",
    "shares",
    "revenue",
    "operating_income",
    "net_income",
    "equity",
    "cfo",
    "capex_outflow",
    "fcf",
    "eps",
    "bps",
    "sps",
    "ops",
    "oofps",
    "fcfps",
    "per",
    "pbr",
    "psr",
    "por",
    "pfcfr",
    "eps_growth",
    "peg",
)


def _safe_json_number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return number


def _coerce_list(values: list[Any] | tuple[Any, ...] | None, length: int) -> list[Any]:
    out = list(values or [])
    if len(out) < length:
        out.extend([None] * (length - len(out)))
    return out[:length]


def _series_from_values(values: list[Any] | tuple[Any, ...] | None, length: int) -> pd.Series:
    return pd.to_numeric(pd.Series(_coerce_list(values, length), dtype="object"), errors="coerce")


def _positive_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den_num = pd.to_numeric(den, errors="coerce")
    return pd.to_numeric(num, errors="coerce") / den_num.where(den_num > 0, np.nan)


def _pct_change(series: pd.Series, lag: int) -> pd.Series:
    current = pd.to_numeric(series, errors="coerce")
    prev = current.shift(lag)
    return (current / prev.where(prev > 0, np.nan) - 1.0) * 100.0


def _period_label(period: Any) -> str:
    ts = pd.Timestamp(period)
    return f"{ts.year}Q{ts.quarter}"


def _build_price_frame(prices_block: dict[str, Any] | None) -> pd.DataFrame:
    if not isinstance(prices_block, dict):
        return pd.DataFrame()
    dates = prices_block.get("dates") or []
    if not dates:
        return pd.DataFrame()
    frame = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce")})
    frame = frame.dropna(subset=["date"]).copy()
    if frame.empty:
        return pd.DataFrame()
    for key in ("close", "adj_close", "market_cap", "shares_outstanding"):
        if key in prices_block:
            frame[key] = pd.to_numeric(pd.Series(prices_block[key]), errors="coerce")
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last").set_index("date")
    return frame


def _align_price_metric(price_frame: pd.DataFrame, periods: list[Any], candidates: tuple[str, ...]) -> pd.Series:
    if price_frame.empty:
        return pd.Series(np.nan, index=range(len(periods)), dtype=float)
    out: list[float | None] = []
    for period in pd.to_datetime(periods, errors="coerce"):
        if pd.isna(period):
            out.append(None)
            continue
        value = None
        for column in candidates:
            if column not in price_frame.columns:
                continue
            sub = pd.to_numeric(price_frame.loc[price_frame.index <= period, column], errors="coerce").dropna()
            if not sub.empty:
                value = float(sub.iloc[-1])
                break
        out.append(value)
    return pd.Series(out, dtype=float)


def _block_top_series(block: dict[str, Any], key: str, length: int) -> pd.Series:
    if not isinstance(block, dict):
        return pd.Series(np.nan, index=range(length), dtype=float)
    return _series_from_values(block.get(key), length)


def _block_section_series(block: dict[str, Any], section: str, key: str, length: int) -> pd.Series:
    if not isinstance(block, dict):
        return pd.Series(np.nan, index=range(length), dtype=float)
    values = (block.get(section) or {}).get(key) if isinstance(block.get(section), dict) else None
    return _series_from_values(values, length)


def _block_labels(block: dict[str, Any] | None) -> list[str]:
    if not isinstance(block, dict):
        return []
    periods = list(block.get("periods") or [])
    labels = list(block.get("fiscal_label") or [])
    if len(labels) == len(periods) and labels:
        return [str(v) for v in labels]
    terms = list(block.get("term") or [])
    if len(terms) == len(periods) and terms:
        return [str(v) for v in terms]
    return [_period_label(period) for period in periods]


def _align_quarterly_block_series(
    block: dict[str, Any] | None,
    labels: list[str],
    *,
    section: str,
    key: str,
) -> pd.Series:
    if not isinstance(block, dict):
        return pd.Series(np.nan, index=range(len(labels)), dtype=float)
    block_labels = _block_labels(block)
    if not block_labels:
        return pd.Series(np.nan, index=range(len(labels)), dtype=float)
    section_payload = block.get(section)
    if not isinstance(section_payload, dict):
        return pd.Series(np.nan, index=range(len(labels)), dtype=float)
    raw_values = list(section_payload.get(key) or [])
    if not raw_values:
        return pd.Series(np.nan, index=range(len(labels)), dtype=float)
    label_to_value: dict[str, Any] = {}
    for label, value in zip(block_labels, raw_values, strict=False):
        label_to_value[str(label)] = value
    return pd.to_numeric(pd.Series([label_to_value.get(str(label)) for label in labels]), errors="coerce")


def _smooth_outlier_series(values: pd.Series, *, low: float = 0.5, high: float = 2.0) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce").copy()
    if out.notna().sum() < 3:
        return out
    for _ in range(2):
        arr = out.to_numpy(dtype=float)
        for i in range(1, len(arr) - 1):
            prev_v = arr[i - 1]
            curr_v = arr[i]
            next_v = arr[i + 1]
            if not np.isfinite(prev_v) or prev_v <= 0 or not np.isfinite(curr_v) or curr_v <= 0 or not np.isfinite(next_v) or next_v <= 0:
                continue
            median = float(np.median([prev_v, next_v]))
            if median <= 0 or not np.isfinite(median):
                continue
            ratio = curr_v / median
            if ratio < low or ratio > high:
                arr[i] = median
        out = pd.Series(arr, index=out.index, dtype=float)
    return out


def _normalize_split_like_share_series(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce").copy()
    if out.notna().sum() < 3:
        return out

    arr = out.to_numpy(dtype=float)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(arr)):
            prev_v = arr[i - 1]
            curr_v = arr[i]
            if not np.isfinite(prev_v) or prev_v <= 0 or not np.isfinite(curr_v) or curr_v <= 0:
                continue
            ratio = curr_v / prev_v
            if 5.0 <= ratio <= 200.0:
                arr[:i] = arr[:i] * ratio
                changed = True
                break
    return _smooth_outlier_series(pd.Series(arr, index=out.index, dtype=float))


def _share_series_quality(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.where(s > 0, np.nan)
    if s.notna().sum() < 2:
        return float("inf")
    logv = np.log(s.dropna())
    diff = logv.diff().abs().dropna()
    if diff.empty:
        return 0.0
    return float(diff.median())


def _resolve_share_series(
    *,
    market: str,
    labels: list[str],
    price: pd.Series,
    market_cap: pd.Series,
    financials_ttm_block: dict[str, Any] | None,
    quarterly_block: dict[str, Any] | None,
) -> tuple[pd.Series, pd.Series]:
    shares_top = _block_top_series(financials_ttm_block or {}, "shares_outstanding", len(labels))
    shares_top = shares_top.where(shares_top > 0, np.nan)
    shares_from_market_cap = _positive_div(market_cap, price)
    shares_from_quarterly_eps = pd.Series(np.nan, index=range(len(labels)), dtype=float)

    if quarterly_block:
        q_net_income = _align_quarterly_block_series(quarterly_block, labels, section="income", key="net_income")
        q_eps = _align_quarterly_block_series(quarterly_block, labels, section="income", key="eps")
        raw_quarterly_shares = (pd.to_numeric(q_net_income, errors="coerce") / pd.to_numeric(q_eps, errors="coerce").replace(0, np.nan)).abs()
        raw_quarterly_shares = raw_quarterly_shares.where(np.isfinite(raw_quarterly_shares) & (raw_quarterly_shares > 0), np.nan)
        if str(market).lower() == "kr":
            shares_from_quarterly_eps = _normalize_split_like_share_series(raw_quarterly_shares)
        else:
            shares_from_quarterly_eps = _smooth_outlier_series(raw_quarterly_shares)

    candidates: list[pd.Series] = []
    used_quarterly_share_reconstruction = False
    if shares_top.notna().any():
        candidates.append(shares_top)
    if shares_from_quarterly_eps.notna().any():
        candidates.append(shares_from_quarterly_eps)
    if shares_from_market_cap.notna().any():
        candidates.append(shares_from_market_cap)
    if not candidates:
        return shares_top, market_cap

    best = min(candidates, key=_share_series_quality)
    used_quarterly_share_reconstruction = best is shares_from_quarterly_eps
    shares = pd.to_numeric(best, errors="coerce").where(pd.to_numeric(best, errors="coerce") > 0, np.nan)
    if str(market).lower() == "kr":
        shares = _normalize_split_like_share_series(shares)

    if shares.notna().any() and price.notna().any():
        recomputed_market_cap = price * shares
        if str(market).lower() == "kr":
            market_cap = recomputed_market_cap
        elif market_cap.dropna().empty or used_quarterly_share_reconstruction:
            market_cap = recomputed_market_cap
        else:
            ratio = _positive_div(market_cap, recomputed_market_cap)
            ratio = ratio.where(np.isfinite(ratio) & (ratio > 0), np.nan)
            if ratio.dropna().empty:
                market_cap = recomputed_market_cap
            else:
                median_ratio = float(ratio.dropna().median())
                if median_ratio < 0.5 or median_ratio > 2.0:
                    market_cap = recomputed_market_cap

    return shares, market_cap


def _quantile_map(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    s = s[s > 0]
    if s.empty:
        return {}
    out: dict[str, float] = {}
    for key, q in zip(VALUATION_TTM_QUANTILE_KEYS, VALUATION_TTM_QUANTILES, strict=False):
        try:
            out[key] = float(s.quantile(q))
        except Exception:
            continue
    return out


def _table_rows_from_window(frame: pd.DataFrame, quantiles: dict[str, dict[str, float]]) -> dict[str, Any]:
    latest = frame.iloc[-1] if not frame.empty else pd.Series(dtype=float)

    band_rows: list[dict[str, Any]] = []
    for metric in ("pbr", "psr", "per", "por", "pfcfr"):
        row: dict[str, Any] = {"metric": metric.upper()}
        qmap = quantiles.get(metric, {})
        for key in VALUATION_TTM_QUANTILE_KEYS:
            row[key] = _safe_json_number(qmap.get(key))
        band_rows.append(row)

    value_rows = [
        {"metric": "PBR", "latest": _safe_json_number(latest.get("pbr")), "median_q50": _safe_json_number(quantiles.get("pbr", {}).get("q50"))},
        {"metric": "PSR", "latest": _safe_json_number(latest.get("psr")), "median_q50": _safe_json_number(quantiles.get("psr", {}).get("q50"))},
        {"metric": "PER", "latest": _safe_json_number(latest.get("per")), "median_q50": _safe_json_number(quantiles.get("per", {}).get("q50"))},
        {"metric": "POR", "latest": _safe_json_number(latest.get("por")), "median_q50": _safe_json_number(quantiles.get("por", {}).get("q50"))},
        {"metric": "PEG", "latest": _safe_json_number(latest.get("peg")), "median_q50": None},
    ]

    per_share_rows = [
        {"metric": "BPS", "latest": _safe_json_number(latest.get("bps"))},
        {"metric": "SPS", "latest": _safe_json_number(latest.get("sps"))},
        {"metric": "OPS", "latest": _safe_json_number(latest.get("ops"))},
        {"metric": "EPS", "latest": _safe_json_number(latest.get("eps"))},
        {"metric": "OOFPS", "latest": _safe_json_number(latest.get("oofps"))},
        {"metric": "FCFPS", "latest": _safe_json_number(latest.get("fcfps"))},
    ]

    return {
        "band": {"rows": band_rows},
        "value": {"rows": value_rows},
        "per_share": {"rows": per_share_rows},
    }


def _slice_window(frame: pd.DataFrame, labels: list[str], years: str) -> tuple[pd.DataFrame, list[str]]:
    size = VALUATION_TTM_WINDOW_SIZES[years]
    sliced = frame.tail(size).copy()
    return sliced, labels[-len(sliced):]


def build_valuation_ttm_payload(
    *,
    ticker: str,
    market: str,
    prices_block: dict[str, Any] | None,
    financials_ttm_block: dict[str, Any] | None,
    quarterly_block: dict[str, Any] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(financials_ttm_block, dict):
        return None

    periods_raw = list(financials_ttm_block.get("periods") or [])
    if not periods_raw:
        return None

    labels = _block_labels(financials_ttm_block)
    if len(labels) != len(periods_raw):
        labels = [_period_label(period) for period in periods_raw]
    length = len(periods_raw)
    price_frame = _build_price_frame(prices_block)

    price_field = "close" if "close" in price_frame.columns else "adj_close" if "adj_close" in price_frame.columns else ""
    price = _align_price_metric(price_frame, periods_raw, ("close", "adj_close"))
    market_cap = _block_top_series(financials_ttm_block, "market_cap", length)
    if market_cap.dropna().empty:
        market_cap = _align_price_metric(price_frame, periods_raw, ("market_cap",))

    shares = _block_top_series(financials_ttm_block, "shares_outstanding", length)
    revenue = _block_section_series(financials_ttm_block, "income", "revenue", length)
    operating_income = _block_section_series(financials_ttm_block, "income", "operating_income", length)
    net_income = _block_section_series(financials_ttm_block, "income", "net_income", length)
    equity = _block_section_series(financials_ttm_block, "balance", "shareholders_equity", length)
    cfo = _block_section_series(financials_ttm_block, "cashflow", "cfo", length)
    capex_raw = _block_section_series(financials_ttm_block, "cashflow", "capex", length)
    capex_outflow = capex_raw.abs()
    fcf = cfo - capex_outflow

    shares, market_cap = _resolve_share_series(
        market=str(market).lower(),
        labels=labels,
        price=price,
        market_cap=market_cap,
        financials_ttm_block=financials_ttm_block,
        quarterly_block=quarterly_block,
    )

    if price.dropna().empty:
        return None

    eps = _positive_div(net_income, shares)
    eps_for_per = eps.copy()
    bps = _positive_div(equity, shares)
    sps = _positive_div(revenue, shares)
    ops = _positive_div(operating_income, shares)
    oofps = _positive_div(cfo, shares)
    fcfps = _positive_div(fcf, shares)
    pbr = _positive_div(price, bps)
    psr = _positive_div(price, sps)
    per = _positive_div(price, eps_for_per)
    por = _positive_div(price, ops)
    pfcfr = _positive_div(price, fcfps)
    eps_growth = _pct_change(eps_for_per, 4)
    peg = pd.to_numeric(per, errors="coerce") / eps_growth.where(eps_growth > 0, np.nan)

    frame = pd.DataFrame(
        {
            "price": price,
            "market_cap": market_cap,
            "shares": shares,
            "revenue": revenue,
            "operating_income": operating_income,
            "net_income": net_income,
            "equity": equity,
            "cfo": cfo,
            "capex_outflow": capex_outflow,
            "fcf": fcf,
            "eps": eps,
            "bps": bps,
            "sps": sps,
            "ops": ops,
            "oofps": oofps,
            "fcfps": fcfps,
            "per": per,
            "pbr": pbr,
            "psr": psr,
            "por": por,
            "pfcfr": pfcfr,
            "eps_growth": eps_growth,
            "peg": peg,
        }
    ).replace([np.inf, -np.inf], np.nan)

    windows: dict[str, Any] = {}
    for window_name in VALUATION_TTM_WINDOW_SIZES:
        frame_window, labels_window = _slice_window(frame, labels, window_name)
        quantiles = {
            "pbr": _quantile_map(frame_window.get("pbr")),
            "psr": _quantile_map(frame_window.get("psr")),
            "per": _quantile_map(frame_window.get("per")),
            "por": _quantile_map(frame_window.get("por")),
            "pfcfr": _quantile_map(frame_window.get("pfcfr")),
        }
        series = {
            key: [_safe_json_number(v) for v in pd.to_numeric(frame_window.get(key), errors="coerce").tolist()]
            for key in VALUATION_TTM_SERIES_KEYS
        }
        windows[window_name] = {
            "periods": labels_window,
            "series": series,
            "quantiles": quantiles,
            "tables": _table_rows_from_window(frame_window, quantiles),
        }

    return {
        "schema_version": 1,
        "ticker": str(ticker),
        "market": str(market).lower(),
        "basis": "ttm",
        "updated_at": updated_at or datetime.now().isoformat(timespec="seconds"),
        "meta": {
            "currency": "KRW" if str(market).lower() == "kr" else "USD",
            "price_field": price_field or "close",
            "valuation_source": "export_json_ttm_v1",
            "reference_levels": VALUATION_TTM_REFERENCE_LEVELS,
        },
        "windows": windows,
    }


def build_valuation_frame_from_precomputed(
    precomputed_payload: dict[str, Any],
    window: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any], dict[str, list[float]]] | None:
    windows = precomputed_payload.get("windows") if isinstance(precomputed_payload, dict) else None
    if not isinstance(windows, dict):
        return None
    window_payload = windows.get(window)
    if not isinstance(window_payload, dict):
        return None

    labels = list(window_payload.get("periods") or [])
    if not labels:
        return None
    length = len(labels)
    raw_series = window_payload.get("series")
    if not isinstance(raw_series, dict):
        return None

    frame = pd.DataFrame(
        {
            key: _series_from_values(raw_series.get(key), length)
            for key in VALUATION_TTM_SERIES_KEYS
        }
    ).replace([np.inf, -np.inf], np.nan)

    quantiles = window_payload.get("quantiles") if isinstance(window_payload.get("quantiles"), dict) else {}
    for metric, base_key in (
        ("pbr", "bps"),
        ("psr", "sps"),
        ("per", "eps"),
        ("por", "ops"),
        ("pfcfr", "fcfps"),
    ):
        qmap = quantiles.get(metric) if isinstance(quantiles.get(metric), dict) else {}
        base = pd.to_numeric(frame.get(base_key), errors="coerce").where(pd.to_numeric(frame.get(base_key), errors="coerce") > 0, np.nan)
        for q_key in VALUATION_TTM_QUANTILE_KEYS:
            value = qmap.get(q_key)
            try:
                qv = float(value)
            except Exception:
                qv = np.nan
            frame[f"{metric}_band_{q_key}"] = base * qv if np.isfinite(qv) else np.nan

    reference_levels = (precomputed_payload.get("meta") or {}).get("reference_levels") if isinstance(precomputed_payload, dict) else {}
    if not isinstance(reference_levels, dict):
        reference_levels = {}
    for metric, refs in VALUATION_TTM_REFERENCE_LEVELS.items():
        levels = reference_levels.get(metric)
        if not isinstance(levels, list):
            levels = refs
        for ref in levels:
            frame[f"{metric}_ref_{str(ref).replace('.', '_')}"] = float(ref)

    tables = window_payload.get("tables") if isinstance(window_payload.get("tables"), dict) else {
        "band": {"rows": []},
        "value": {"rows": []},
        "per_share": {"rows": []},
    }
    return frame, labels, tables, reference_levels or VALUATION_TTM_REFERENCE_LEVELS
