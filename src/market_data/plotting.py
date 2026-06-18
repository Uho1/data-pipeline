from __future__ import annotations

from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.ticker import FuncFormatter, ScalarFormatter

UP_COLOR = "#1fa774"
DOWN_COLOR = "#d84a4a"


@dataclass
class OhlcvData:
    df: pd.DataFrame
    x_num: np.ndarray


def series_bounds(series_list: list[pd.Series], size: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    arrays: list[np.ndarray] = []
    for series in series_list:
        if series is None or series.empty:
            continue
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if arr.size != size:
            continue
        arrays.append(arr)

    if not arrays:
        return None, None

    stack = np.vstack(arrays)
    finite = np.isfinite(stack)
    any_finite = np.any(finite, axis=0)

    low = np.min(np.where(finite, stack, np.inf), axis=0)
    high = np.max(np.where(finite, stack, -np.inf), axis=0)
    low = np.where(any_finite, low, np.nan)
    high = np.where(any_finite, high, np.nan)
    return low, high


def draw_indicators(
    ax: plt.Axes,
    ohlc: OhlcvData,
    mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    df = ohlc.df
    if mode not in {"none", "ichimoku", "bollinger", "both"}:
        mode = "none"
    if mode == "none":
        return None, None

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    overlays: list[pd.Series] = []

    if mode in {"bollinger", "both"}:
        bb_mid = close.rolling(window=20, min_periods=20).mean()
        bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std

        ax.plot(df.index, bb_mid, color="#2f5fb3", linewidth=0.85, alpha=0.95)
        ax.plot(df.index, bb_upper, color="#2f5fb3", linewidth=0.75, alpha=0.75)
        ax.plot(df.index, bb_lower, color="#2f5fb3", linewidth=0.75, alpha=0.75)
        ax.fill_between(df.index, bb_lower, bb_upper, color="#2f5fb3", alpha=0.08)
        overlays.extend([bb_mid, bb_upper, bb_lower])

    if mode in {"ichimoku", "both"}:
        tenkan = (high.rolling(window=9, min_periods=9).max() + low.rolling(window=9, min_periods=9).min()) / 2.0
        kijun = (high.rolling(window=26, min_periods=26).max() + low.rolling(window=26, min_periods=26).min()) / 2.0
        span_a = ((tenkan + kijun) / 2.0).shift(26)
        span_b = (
            (high.rolling(window=52, min_periods=52).max() + low.rolling(window=52, min_periods=52).min()) / 2.0
        ).shift(26)
        chikou = close.shift(-26)

        ax.plot(df.index, tenkan, color="#ff8c00", linewidth=0.85, alpha=0.95)
        ax.plot(df.index, kijun, color="#8b4513", linewidth=0.85, alpha=0.95)
        ax.plot(df.index, chikou, color="#6f42c1", linewidth=0.75, alpha=0.70)
        ax.plot(df.index, span_a, color="#228b22", linewidth=0.75, alpha=0.80)
        ax.plot(df.index, span_b, color="#b22222", linewidth=0.75, alpha=0.80)

        bullish = (span_a >= span_b).fillna(False).to_numpy(dtype=bool)
        bearish = (span_a < span_b).fillna(False).to_numpy(dtype=bool)
        ax.fill_between(df.index, span_a, span_b, where=bullish, color="#228b22", alpha=0.08, interpolate=False)
        ax.fill_between(df.index, span_a, span_b, where=bearish, color="#b22222", alpha=0.08, interpolate=False)

        # Future cloud: extend span_a and span_b 26 business days beyond data
        CLOUD_FORWARD = 26
        if len(df) >= CLOUD_FORWARD:
            try:
                # Unshifted senkou values: future cloud comes from unshifted values of last 26 bars
                senkou_a_raw = (tenkan + kijun) / 2.0
                senkou_b_raw = (
                    (high.rolling(window=52, min_periods=1).max() + low.rolling(window=52, min_periods=1).min()) / 2.0
                )
                future_sa = senkou_a_raw.iloc[-CLOUD_FORWARD:].values
                future_sb = senkou_b_raw.iloc[-CLOUD_FORWARD:].values
                last_date = df.index[-1]
                future_dates = pd.bdate_range(last_date, periods=CLOUD_FORWARD + 1)[1:]
                if len(future_dates) > 0:
                    n = min(len(future_dates), len(future_sa), len(future_sb))
                    fd = future_dates[:n]
                    fsa = future_sa[:n]
                    fsb = future_sb[:n]
                    ax.plot(fd, fsa, color="#228b22", linewidth=0.75, alpha=0.80, linestyle="--")
                    ax.plot(fd, fsb, color="#b22222", linewidth=0.75, alpha=0.80, linestyle="--")
                    bullish_f = fsa >= fsb
                    bearish_f = fsa < fsb
                    ax.fill_between(fd, fsa, fsb, where=bullish_f, color="#228b22", alpha=0.06, interpolate=False)
                    ax.fill_between(fd, fsa, fsb, where=bearish_f, color="#b22222", alpha=0.06, interpolate=False)
                    # Include future values in bounds
                    fs_a_series = pd.Series(fsa)
                    fs_b_series = pd.Series(fsb)
                    overlays.extend([fs_a_series, fs_b_series])
            except Exception:
                pass

        overlays.extend([tenkan, kijun, span_a, span_b, chikou])

    return series_bounds(overlays, size=len(df))


def prepare_ohlcv(df: pd.DataFrame) -> OhlcvData:
    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    out = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=required).sort_index()
    if out.empty:
        raise ValueError("No valid OHLC rows after cleaning")

    x_num = mdates.date2num(out.index.to_pydatetime())
    return OhlcvData(df=out, x_num=x_num)


def nearest_index(x_values: np.ndarray, x: float) -> int:
    idx = int(np.searchsorted(x_values, x, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(x_values):
        return len(x_values) - 1
    left = x_values[idx - 1]
    right = x_values[idx]
    return idx - 1 if abs(x - left) <= abs(right - x) else idx


def _month_tick_label(x_value: float, _pos: int) -> str:
    dt = mdates.num2date(x_value)
    if dt.month == 1:
        return f"{dt.year}"
    return f"{dt.month:02d}"


def update_time_axis(ax_x: plt.Axes) -> None:
    span_days = abs(ax_x.get_xlim()[1] - ax_x.get_xlim()[0])
    target_ticks = 8

    # Keep year-only labels for very wide ranges.
    # Around ~2 years, switch to month-aware labels so months remain visible.
    if span_days >= 365 * 3:
        years = max(1, int(round((span_days / 365.0) / target_ticks)))
        ax_x.xaxis.set_major_locator(mdates.YearLocator(base=years))
        ax_x.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        return

    if span_days >= 120:
        months = max(1, int(round((span_days / 30.0) / target_ticks)))
        ax_x.xaxis.set_major_locator(mdates.MonthLocator(interval=months))
        ax_x.xaxis.set_major_formatter(FuncFormatter(_month_tick_label))
        return

    if span_days >= 20:
        weeks = max(1, int(round((span_days / 7.0) / target_ticks)))
        ax_x.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=weeks))
        ax_x.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        return

    days = max(1, int(round(span_days / target_ticks)))
    ax_x.xaxis.set_major_locator(mdates.DayLocator(interval=days))
    ax_x.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))


def style_axes(
    ax_price: plt.Axes,
    ax_vol: plt.Axes,
    title: str,
) -> None:
    # Price axis on the right, with plain numbers and smaller labels.
    for axis in (ax_price, ax_vol):
        axis.ticklabel_format(axis="y", style="plain")
        scalar = ScalarFormatter(useOffset=False)
        scalar.set_scientific(False)
        axis.yaxis.set_major_formatter(scalar)
        axis.tick_params(axis="both", labelsize=5)

    ax_price.yaxis.tick_right()
    ax_price.yaxis.set_label_position("right")
    ax_vol.yaxis.tick_right()
    ax_vol.yaxis.set_label_position("right")

    # Dynamic labels based on zoom/pan range.
    update_time_axis(ax_vol)
    ax_vol.tick_params(axis="x", labelsize=5)
    ax_price.tick_params(axis="x", labelsize=5)

    ax_price.set_title(title, fontsize=6)


def draw_candles(
    ax: plt.Axes,
    ohlc: OhlcvData,
    width_days: float = 0.6,
    up_color: str = UP_COLOR,
    down_color: str = DOWN_COLOR,
) -> None:
    df = ohlc.df
    x_num = ohlc.x_num

    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)

    y_min = float(np.nanmin(lows))
    y_max = float(np.nanmax(highs))
    min_body = max((y_max - y_min) * 0.0008, 1e-8)

    up_mask = closes >= opens
    colors = np.where(up_mask, up_color, down_color)

    # Draw wicks as one collection (much faster than per-candle artists).
    wick_segments = np.stack(
        [
            np.column_stack((x_num, lows)),
            np.column_stack((x_num, highs)),
        ],
        axis=1,
    )
    wick_collection = LineCollection(
        wick_segments,
        colors=colors,
        linewidths=1.0,
        alpha=0.95,
    )
    ax.add_collection(wick_collection)

    # Draw candle bodies as one polygon collection.
    body_low = np.minimum(opens, closes)
    body_high = np.maximum(opens, closes)
    body_height = body_high - body_low

    thin = body_height < min_body
    body_low = np.where(thin, body_low - min_body / 2.0, body_low)
    body_high = np.where(thin, body_low + min_body, body_high)

    left = x_num - width_days / 2.0
    right = x_num + width_days / 2.0
    polygons = np.stack(
        [
            np.column_stack((left, body_low)),
            np.column_stack((left, body_high)),
            np.column_stack((right, body_high)),
            np.column_stack((right, body_low)),
        ],
        axis=1,
    )
    body_collection = PolyCollection(
        polygons,
        facecolors=colors,
        edgecolors=colors,
        linewidths=0.8,
        alpha=0.95,
    )
    ax.add_collection(body_collection)

    ax.set_xlim(x_num[0] - 2.0, x_num[-1] + 2.0)
    ax.xaxis_date()
    ax.grid(alpha=0.20)


def draw_volume(
    ax: plt.Axes,
    ohlc: OhlcvData,
    up_color: str = UP_COLOR,
    down_color: str = DOWN_COLOR,
) -> None:
    df = ohlc.df
    if "Volume" not in df.columns:
        ax.set_visible(False)
        return

    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    opens = df["Open"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    colors = np.where(closes >= opens, up_color, down_color)
    ax.vlines(ohlc.x_num, 0.0, volume, colors=colors, linewidth=2.0, alpha=0.60)
    ax.set_xlim(ohlc.x_num[0] - 2.0, ohlc.x_num[-1] + 2.0)
    ax.grid(alpha=0.18)
    ax.set_ylabel("Volume")
