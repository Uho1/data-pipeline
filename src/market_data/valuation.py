from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_data.sec_term_reader import load_ticker_quarterly_cache


@dataclass
class ValuationSeries:
    eps_daily: pd.Series | None
    bps_daily: pd.Series | None
    per_series: pd.Series | None
    pbr_series: pd.Series | None
    default_per_levels: np.ndarray
    default_pbr_levels: np.ndarray
    valuation_source: str
    per_levels: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    price_bands: pd.DataFrame | None = None
    per_quarterly: pd.Series | None = None
    eps_ttm_quarterly: pd.Series | None = None
    eps_source_quarterly: pd.Series | None = None
    price_field: str = "adjclose"


def _as_datetime_index(values: pd.Series | pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(values, errors="coerce")
    idx = pd.DatetimeIndex(idx)
    idx = idx[~idx.isna()]
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx


def _to_numeric_series(values: pd.Series | None, index: pd.DatetimeIndex | None = None) -> pd.Series:
    if values is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(index=index, dtype=float)
    out = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if index is not None:
        out = out.reindex(index)
    return out.astype(float)


def _first_valid_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) > 0:
            return s
    return pd.Series(index=df.index, dtype=float)


def _prepare_quarterly_frame(df_quarterly: pd.DataFrame) -> pd.DataFrame:
    if df_quarterly is None or df_quarterly.empty:
        return pd.DataFrame()

    out = df_quarterly.copy()
    if "StatementDate" in out.columns:
        idx = _as_datetime_index(out["StatementDate"])
        out = out.loc[out["StatementDate"].notna()].copy()
    elif "end_date" in out.columns:
        idx = _as_datetime_index(out["end_date"])
        out = out.loc[out["end_date"].notna()].copy()
    else:
        return pd.DataFrame()

    if len(idx) != len(out):
        # Normalized index after dropping invalid timestamps.
        if "StatementDate" in out.columns:
            out["StatementDate"] = pd.to_datetime(out["StatementDate"], errors="coerce")
            out = out.loc[out["StatementDate"].notna()].copy()
            idx = pd.DatetimeIndex(out["StatementDate"])
        else:
            out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
            out = out.loc[out["end_date"].notna()].copy()
            idx = pd.DatetimeIndex(out["end_date"])

    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    return out


def _split_factor_between(
    split_series: pd.Series | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    if split_series is None or split_series.empty:
        return 1.0
    s = pd.to_numeric(split_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 1.0
    s = s[(s > 0.0) & (~np.isclose(s, 1.0))]
    if s.empty:
        return 1.0
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert(None)
    window = s.loc[(s.index > start_date) & (s.index <= end_date)]
    if window.empty:
        return 1.0
    return float(np.prod(window.to_numpy(dtype=float)))


def _sanitize_share_jumps_with_splits(shares: pd.Series, split_series: pd.Series | None) -> pd.Series:
    out = pd.to_numeric(shares, errors="coerce").replace([np.inf, -np.inf], np.nan).copy()
    if out.notna().sum() < 3:
        return out

    factors = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 28.0]
    tol = 0.12
    idx = pd.DatetimeIndex(out.index)

    for i in range(1, len(out)):
        prev = float(out.iloc[i - 1]) if pd.notna(out.iloc[i - 1]) else np.nan
        cur = float(out.iloc[i]) if pd.notna(out.iloc[i]) else np.nan
        if not (np.isfinite(prev) and np.isfinite(cur) and prev > 0 and cur > 0):
            continue

        ratio = cur / prev
        if 0.5 <= ratio <= 2.0:
            continue

        expected_split = _split_factor_between(split_series, pd.Timestamp(idx[i - 1]), pd.Timestamp(idx[i]))
        if expected_split > 1.0:
            if abs(ratio - expected_split) / expected_split <= 0.25:
                continue
            if abs((1.0 / ratio) - expected_split) / expected_split <= 0.25:
                continue

        fixed = cur
        for f in factors:
            if abs(ratio - f) / f <= tol:
                fixed = cur / f
                break
            inv = 1.0 / f
            if abs(ratio - inv) / inv <= tol:
                fixed = cur * f
                break
        out.iloc[i] = fixed

    return out


def build_eps_quarterly(
    df_quarterly: pd.DataFrame,
    stock_splits_series: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return quarterly diluted EPS series with source labels.

    Source priority per quarter:
    1) SEC diluted EPS (per-share)
    2) Net income common / diluted weighted avg shares
    3) Net income common / basic shares (or generic shares fallback)
    """

    q = _prepare_quarterly_frame(df_quarterly)
    if q.empty:
        return pd.Series(dtype=float), pd.Series(dtype="object")

    diluted_eps = _first_valid_series(q, ["diluted_eps", "Diluted EPS"])
    net_income_common = _first_valid_series(q, ["net_income_common", "Net Income Common", "Net Income"])

    shares_outstanding = _first_valid_series(q, ["Shares"])
    diluted_shares = _first_valid_series(q, ["diluted_shares", "Diluted Shares"])
    basic_shares_raw = _first_valid_series(q, ["basic_shares", "Basic Shares"])

    diluted_shares = _sanitize_share_jumps_with_splits(diluted_shares, stock_splits_series)
    basic_shares_raw = _sanitize_share_jumps_with_splits(basic_shares_raw, stock_splits_series)
    shares_outstanding = _sanitize_share_jumps_with_splits(shares_outstanding, stock_splits_series)

    # Weighted-average shares must be reasonably close to period-end shares.
    # If not, SEC often provided YTD/cumulative averages that distort quarterly EPS.
    if shares_outstanding.notna().any():
        dil_ratio = (diluted_shares / shares_outstanding.replace(0.0, np.nan)).abs()
        diluted_shares = diluted_shares.mask((dil_ratio < 0.5) | (dil_ratio > 2.0))
        bas_ratio = (basic_shares_raw / shares_outstanding.replace(0.0, np.nan)).abs()
        basic_shares_raw = basic_shares_raw.mask((bas_ratio < 0.5) | (bas_ratio > 2.0))

    basic_shares = basic_shares_raw.where(basic_shares_raw.notna(), shares_outstanding)

    eps_from_diluted = net_income_common / diluted_shares.replace(0.0, np.nan)
    eps_from_basic = net_income_common / basic_shares.replace(0.0, np.nan)

    eps_q = diluted_eps.copy()
    source = pd.Series("none", index=q.index, dtype="object")

    has_sec_eps = diluted_eps.notna()
    source.loc[has_sec_eps] = "sec_eps"

    use_diluted = eps_q.isna() & eps_from_diluted.notna()
    eps_q.loc[use_diluted] = eps_from_diluted.loc[use_diluted]
    source.loc[use_diluted] = "ni_over_shares_diluted"

    use_basic = eps_q.isna() & eps_from_basic.notna()
    eps_q.loc[use_basic] = eps_from_basic.loc[use_basic]
    source.loc[use_basic] = "ni_over_shares_basic"

    eps_q = eps_q.replace([np.inf, -np.inf], np.nan)
    return eps_q, source


def build_eps_ttm(eps_q: pd.Series) -> pd.Series:
    s = pd.to_numeric(eps_q, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s.rolling(window=4, min_periods=1).sum()


def build_bps_quarterly(df_quarterly: pd.DataFrame) -> pd.Series:
    q = _prepare_quarterly_frame(df_quarterly)
    if q.empty:
        return pd.Series(dtype=float)
    equity = _first_valid_series(
        q,
        [
            "Shareholders Equity",
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        ],
    )
    # BPS should primarily use period-end outstanding shares.
    shares = _first_valid_series(q, ["Shares", "basic_shares", "diluted_shares", "Basic Shares", "Diluted Shares"])
    bps = equity / shares.replace(0.0, np.nan)
    return bps.replace([np.inf, -np.inf], np.nan)


def align_to_price(
    series_quarterly: pd.Series,
    price_daily: pd.Series,
    mode: str = "ffill_from_quarter_end",
) -> pd.Series:
    s = pd.to_numeric(series_quarterly, errors="coerce").replace([np.inf, -np.inf], np.nan)
    p = pd.to_numeric(price_daily, errors="coerce").replace([np.inf, -np.inf], np.nan)
    p = p.sort_index()
    if s.empty or p.empty:
        return pd.Series(index=p.index, dtype=float)

    if mode != "ffill_from_quarter_end":
        raise ValueError(f"unsupported align mode: {mode}")

    s = s.sort_index()
    s = s.loc[~s.index.duplicated(keep="last")]
    out = s.reindex(p.index.union(s.index)).sort_index().ffill().reindex(p.index)
    return pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan)


def compute_per(
    price_daily: pd.Series,
    eps_ttm_daily: pd.Series,
    per_negative: str = "nan",
) -> pd.Series:
    price = pd.to_numeric(price_daily, errors="coerce").replace([np.inf, -np.inf], np.nan)
    eps = pd.to_numeric(eps_ttm_daily, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = price / eps.replace(0.0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)

    policy = str(per_negative or "nan").strip().lower()
    if policy not in {"nan", "allow"}:
        policy = "nan"
    if policy == "nan":
        out = out.where(eps > 0.0)
    return out


def _compute_pbr(price_daily: pd.Series, bps_daily: pd.Series) -> pd.Series:
    p = pd.to_numeric(price_daily, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bps = pd.to_numeric(bps_daily, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = p / bps.replace(0.0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(bps > 0.0)
    return out


def _parse_band_quantiles(raw: str | list[float] | tuple[float, ...] | None) -> np.ndarray:
    default = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
    if raw is None:
        return default

    vals: list[float] = []
    if isinstance(raw, str):
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                vals.append(float(token))
            except Exception:
                continue
    else:
        for v in raw:
            try:
                vals.append(float(v))
            except Exception:
                continue

    arr = np.array(vals, dtype=float)
    if arr.size != 5:
        return default
    if np.any(~np.isfinite(arr)):
        return default
    if np.any(arr <= 0.0) or np.any(arr >= 1.0):
        return default
    arr = np.sort(arr)
    return arr


def _apply_window(series: pd.Series, window: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return s
    window_norm = str(window or "all").strip().lower()
    if window_norm == "all":
        return s

    end = pd.Timestamp(s.index.max())
    if window_norm == "10y":
        start = end - pd.DateOffset(years=10)
    elif window_norm == "5y":
        start = end - pd.DateOffset(years=5)
    else:
        return s
    return s.loc[s.index >= start]


def _apply_outlier_option(series: pd.Series, outlier: str | None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    mode = str(outlier or "none").strip().lower()
    if s.empty or mode in {"", "none", "off"}:
        return s
    if mode == "winsorize-1-99":
        lo = float(s.quantile(0.01))
        hi = float(s.quantile(0.99))
        return s.clip(lower=lo, upper=hi)
    return s


def _default_levels(series: pd.Series | None, fallback_min: float, fallback_max: float) -> np.ndarray:
    if series is None or series.empty:
        return np.linspace(fallback_min, fallback_max, 5)

    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0.0]
    if len(s) < 8:
        return np.linspace(fallback_min, fallback_max, 5)

    q20 = float(s.quantile(0.2))
    q80 = float(s.quantile(0.8))
    if not np.isfinite(q20) or not np.isfinite(q80) or q80 <= q20:
        return np.linspace(fallback_min, fallback_max, 5)
    return np.linspace(q20, q80, 5)


def _compute_per_levels(
    per_daily: pd.Series,
    quantiles: np.ndarray,
    band_window: str,
    outlier: str | None,
) -> np.ndarray:
    s = pd.to_numeric(per_daily, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s[s > 0.0].dropna()
    s = _apply_window(s, band_window)
    s = _apply_outlier_option(s, outlier)
    if len(s) < 10:
        return _default_levels(per_daily, fallback_min=8.0, fallback_max=24.0)

    vals = np.quantile(s.to_numpy(dtype=float), quantiles)
    vals = np.array(vals, dtype=float)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    if np.isnan(vals).any():
        return _default_levels(per_daily, fallback_min=8.0, fallback_max=24.0)

    vals = np.maximum.accumulate(vals)
    return vals


def _build_price_bands(eps_ttm_daily: pd.Series, per_levels: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=eps_ttm_daily.index)
    for i, level in enumerate(per_levels, start=1):
        out[f"per_band_{i}"] = pd.to_numeric(eps_ttm_daily, errors="coerce") * float(level)
    return out


def load_valuation_series(
    ticker: str,
    market: str,
    price_index: pd.DatetimeIndex,
    close_series: pd.Series,
    stock_splits_series: pd.Series | None = None,
    mask_unstable_history: bool = False,
    price_field: str = "adjclose",
    per_negative: str = "nan",
    band_window: str = "all",
    band_quantiles: str | list[float] | tuple[float, ...] | None = None,
    outlier: str | None = None,
    align_mode: str = "ffill_from_quarter_end",
    offline_mode: bool = False,
) -> ValuationSeries:
    del market  # currently valuation uses local SEC quarterly cache only.
    del mask_unstable_history  # retained for compatibility with existing callers.

    q_df = load_ticker_quarterly_cache(ticker=ticker, rebuild_if_stale=not offline_mode)
    q_df = _prepare_quarterly_frame(q_df)

    price = _to_numeric_series(close_series, index=pd.DatetimeIndex(price_index))

    eps_q, eps_source_q = build_eps_quarterly(q_df, stock_splits_series=stock_splits_series)
    eps_ttm_q = build_eps_ttm(eps_q)
    eps_daily = align_to_price(eps_ttm_q, price, mode=align_mode) if not eps_ttm_q.empty else None

    per_series = None
    per_q = None
    if eps_daily is not None:
        per_series = compute_per(price, eps_daily, per_negative=per_negative)
        price_q = (
            price.reindex(price.index.union(eps_ttm_q.index))
            .sort_index()
            .ffill()
            .reindex(eps_ttm_q.index)
        )
        per_q = compute_per(price_q, eps_ttm_q, per_negative=per_negative)

    bps_q = build_bps_quarterly(q_df)
    bps_daily = align_to_price(bps_q, price, mode=align_mode) if not bps_q.empty else None
    pbr_series = _compute_pbr(price, bps_daily) if bps_daily is not None else None

    quantiles = _parse_band_quantiles(band_quantiles)
    per_levels = _compute_per_levels(
        per_daily=per_series if per_series is not None else pd.Series(dtype=float),
        quantiles=quantiles,
        band_window=band_window,
        outlier=outlier,
    )

    price_bands = None
    if eps_daily is not None and not eps_daily.empty:
        price_bands = _build_price_bands(eps_daily, per_levels)

    default_pbr_levels = _default_levels(pbr_series, fallback_min=0.8, fallback_max=3.2)

    source_priority = "none"
    if not eps_source_q.empty:
        counts = eps_source_q.value_counts(dropna=True)
        if not counts.empty:
            source_priority = str(counts.index[0])

    valuation_source = (
        f"sec_term_cache|eps_source={source_priority}|price_field={str(price_field).lower()}|"
        f"per_negative={str(per_negative).lower()}|band_window={str(band_window).lower()}|"
        f"outlier={(str(outlier).lower() if outlier else 'none')}"
    )

    return ValuationSeries(
        eps_daily=eps_daily,
        bps_daily=bps_daily,
        per_series=per_series,
        pbr_series=pbr_series,
        default_per_levels=per_levels,
        default_pbr_levels=default_pbr_levels,
        valuation_source=valuation_source,
        per_levels=per_levels,
        price_bands=price_bands,
        per_quarterly=per_q,
        eps_ttm_quarterly=eps_ttm_q,
        eps_source_quarterly=eps_source_q,
        price_field=str(price_field).lower(),
    )
