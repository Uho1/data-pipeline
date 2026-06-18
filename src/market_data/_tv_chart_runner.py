#!/usr/bin/env python3
"""TradingView Lightweight Charts runner — runs in subprocess.

Called by tv_chart.py. Reads chart config from a JSON temp file, opens a
TradingView-style interactive chart window with in-chart topbar controls,
and blocks until the window is closed.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


def _ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def _line_df(time_col: pd.Series, values: pd.Series, col_name: str) -> pd.DataFrame:
    """Build a {time, col_name} DataFrame suitable for Line.set().

    The column must be named exactly the same as the Line's name parameter.
    """
    return pd.DataFrame({"time": time_col, col_name: values}).dropna()


def _area_df(time_col: pd.Series, values: pd.Series) -> pd.DataFrame:
    """Build {time, value} records for custom area series."""
    out = pd.DataFrame({"time": time_col, "value": values}).dropna()
    if out.empty:
        return out
    out = out.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time", "value"])
    out["time"] = out["time"].astype("int64") // 10 ** 9
    return out


def _normalize_levels(raw: list, fallback: list[float]) -> list[float]:
    levels: list[float] = []
    for value in raw or []:
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if num > 0 and pd.notna(num):
            levels.append(num)

    if not levels:
        levels = [float(x) for x in fallback if float(x) > 0]
    levels = sorted(set(levels))
    return levels[:5]


def _parse_levels_text(text: str, fallback: list[float]) -> list[float]:
    tokens = [t for t in re.split(r"[,\s;]+", str(text).strip()) if t]
    values: list[float] = []
    for token in tokens:
        cleaned = token.lower().replace("x", "")
        try:
            num = float(cleaned)
        except ValueError:
            continue
        if num > 0 and pd.notna(num):
            values.append(float(num))

    if not values:
        return list(fallback)
    if len(values) == 2:
        lo = min(values)
        hi = max(values)
        if hi > lo:
            return [float(x) for x in np.linspace(lo, hi, 5)]
    return sorted(set(values))[:5]


def _levels_text(levels: list[float]) -> str:
    return ", ".join(f"{float(v):.2f}" for v in levels)


def _sanitize_input_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _set_textbox_value(widget, text: str) -> None:
    safe = _sanitize_input_text(text)
    widget.value = text
    widget.run_script(f'{widget.id}.value = "{safe}"')
    widget.run_script(f'{widget.id}.style.width = "{max(len(text) + 2, 4)}ch"')




def _build_band_defs(
    source_df: pd.DataFrame,
    levels: list[float],
    prefix: str,
    palette: list[str],
) -> list[tuple[str, str, pd.DataFrame]]:
    defs: list[tuple[str, str, pd.DataFrame]] = []
    if source_df.empty or not levels:
        return defs

    for idx, level in enumerate(levels[:5]):
        level_val = float(level)
        label = f"{prefix} {level_val:.1f}x"
        band = pd.DataFrame(
            {
                "time": source_df["time"],
                label: source_df["value"] * level_val,
            }
        ).dropna()
        if not band.empty:
            defs.append((label, palette[idx % len(palette)], band))
    return defs


def _prepare_ichimoku_frames(
    df: pd.DataFrame,
    forward_days: int = 26,
) -> tuple[dict[str, tuple[str, pd.DataFrame]], dict[str, pd.DataFrame]]:
    """Return Ichimoku line frames and cloud area frames with forward projection."""
    time_dt = pd.to_datetime(df["time"], errors="coerce")
    if time_dt.isna().any():
        time_dt = time_dt.ffill().bfill()

    future_days = pd.bdate_range(start=time_dt.iloc[-1] + pd.Timedelta(days=1), periods=max(int(forward_days), 0))
    time_ext = pd.Series(
        [*time_dt.dt.strftime("%Y-%m-%d").tolist(), *future_days.strftime("%Y-%m-%d").tolist()]
    )
    ext_len = len(time_ext)

    high_ext = pd.to_numeric(df["high"], errors="coerce").reindex(range(ext_len))
    low_ext = pd.to_numeric(df["low"], errors="coerce").reindex(range(ext_len))
    close_ext = pd.to_numeric(df["close"], errors="coerce").reindex(range(ext_len))

    tenkan = (high_ext.rolling(9, min_periods=9).max() + low_ext.rolling(9, min_periods=9).min()) / 2
    kijun = (high_ext.rolling(26, min_periods=26).max() + low_ext.rolling(26, min_periods=26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high_ext.rolling(52, min_periods=52).max() + low_ext.rolling(52, min_periods=52).min()) / 2).shift(26)
    chikou = close_ext.shift(-26)

    ichi_defs: dict[str, tuple[str, pd.DataFrame]] = {
        "Tenkan": ("#ff8c00", _line_df(time_ext, tenkan, "Tenkan")),
        "Kijun": ("#8b4513", _line_df(time_ext, kijun, "Kijun")),
        "Span A": ("#228b22", _line_df(time_ext, span_a, "Span A")),
        "Span B": ("#b22222", _line_df(time_ext, span_b, "Span B")),
        "Chikou": ("#9c27b0", _line_df(time_ext, chikou, "Chikou")),
    }

    cloud_base = pd.DataFrame({"time": time_ext, "span_a": span_a, "span_b": span_b})
    cloud_base = cloud_base.dropna(subset=["time", "span_a", "span_b"]).copy()
    if cloud_base.empty:
        empty = pd.DataFrame(columns=["time", "value"])
        return ichi_defs, {
            "bull_upper": empty.copy(),
            "bull_lower": empty.copy(),
            "bear_upper": empty.copy(),
            "bear_lower": empty.copy(),
        }

    is_bull = cloud_base["span_a"] >= cloud_base["span_b"]
    bull_upper = cloud_base["span_a"]
    bull_lower = pd.Series(
        np.where(is_bull.to_numpy(), cloud_base["span_b"].to_numpy(), cloud_base["span_a"].to_numpy()),
        index=cloud_base.index,
    )
    bear_upper = cloud_base["span_b"]
    bear_lower = pd.Series(
        np.where((~is_bull).to_numpy(), cloud_base["span_a"].to_numpy(), cloud_base["span_b"].to_numpy()),
        index=cloud_base.index,
    )

    cloud_frames = {
        "bull_upper": _area_df(cloud_base["time"], bull_upper),
        "bull_lower": _area_df(cloud_base["time"], bull_lower),
        "bear_upper": _area_df(cloud_base["time"], bear_upper),
        "bear_lower": _area_df(cloud_base["time"], bear_lower),
    }
    return ichi_defs, cloud_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: _tv_chart_runner.py <config_json_path>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)
    try:
        os.unlink(config_path)
    except OSError:
        pass

    ticker: str = config.get("ticker", "?")
    market: str = config.get("market", "")
    chart_type: str = config.get("chart_type", "candles")
    indicator_mode: str = config.get("indicator_mode", "none")
    data: list[dict] = config.get("data", [])
    valuation: dict = config.get("valuation", {})

    df = pd.DataFrame(data)
    if df.empty or "time" not in df.columns:
        print("No data", file=sys.stderr)
        sys.exit(1)
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).reset_index(drop=True)
    if df.empty:
        print("No valid OHLC rows", file=sys.stderr)
        sys.exit(1)

    try:
        from lightweight_charts import Chart  # type: ignore[import]
    except ImportError:
        print(
            "lightweight-charts not installed. Run: pip install lightweight-charts",
            file=sys.stderr,
        )
        sys.exit(1)

    close_s = df["close"]
    time_col = df["time"]
    high_s = df["high"]
    low_s = df["low"]

    # ---------------------------------------------------------------------------
    # Pre-compute all indicator data
    # (column name must match the Line name that will be created)
    # ---------------------------------------------------------------------------

    # Moving averages
    MA_META: dict[str, tuple[str, object, int, str]] = {
        "sma20":  ("SMA 20",  _sma, 20,  "#2962ff"),
        "sma50":  ("SMA 50",  _sma, 50,  "#ff6d00"),
        "sma200": ("SMA 200", _sma, 200, "#e040fb"),
        "ema20":  ("EMA 20",  _ema, 20,  "#00bcd4"),
    }
    ma_dfs: dict[str, pd.DataFrame] = {}
    for key, (label, fn, period, _color) in MA_META.items():
        ldf = _line_df(time_col, fn(close_s, period), label)
        if not ldf.empty:
            ma_dfs[key] = ldf

    # Bollinger Bands
    bb_mid_s = close_s.rolling(20).mean()
    bb_std_s = close_s.rolling(20).std()
    bb_upper_s = bb_mid_s + 2 * bb_std_s
    bb_lower_s = bb_mid_s - 2 * bb_std_s
    BB_DEFS: dict[str, tuple[str, pd.Series]] = {
        "BB Mid":   ("#2962ff",             bb_mid_s),
        "BB Upper": ("rgba(41,98,255,0.6)", bb_upper_s),
        "BB Lower": ("rgba(41,98,255,0.6)", bb_lower_s),
    }
    bb_dfs: dict[str, pd.DataFrame] = {
        lbl: _line_df(time_col, s, lbl)
        for lbl, (_c, s) in BB_DEFS.items()
    }
    # BB band fill records (open=lower, close=upper — always blue, no wicks)
    # time must be "YYYY-MM-DD" strings to match the main candlestick series
    _bb_fill_df = pd.DataFrame({"time": time_col, "u": bb_upper_s, "l": bb_lower_s}).dropna()
    bb_fill_records: list[dict] = [
        {
            "time": str(r["time"]),
            "open": float(r["l"]),
            "high": float(r["u"]),
            "low": float(r["l"]),
            "close": float(r["u"]),
        }
        for _, r in _bb_fill_df.iterrows()
        if np.isfinite(float(r["u"])) and np.isfinite(float(r["l"]))
    ]
    del _bb_fill_df

    # Ichimoku (forward projection + cloud fill data)
    ICHI_DEFS, ichi_cloud_frames = _prepare_ichimoku_frames(df, forward_days=26)

    # Build Kumo candlestick records (open=span_b, close=span_a →
    #   green when span_a≥span_b, red when span_b>span_a; no wicks)
    _ku_df = pd.merge(
        ichi_cloud_frames["bull_upper"][["time", "value"]].rename(columns={"value": "span_a"}),
        ichi_cloud_frames["bear_upper"][["time", "value"]].rename(columns={"value": "span_b"}),
        on="time",
        how="inner",
    )
    # time in _area_df is Unix timestamps; convert back to "YYYY-MM-DD" strings
    kumo_records: list[dict] = [
        {
            "time": pd.Timestamp(int(r["time"]), unit="s").strftime("%Y-%m-%d"),
            "open": float(r["span_b"]),
            "high": float(max(r["span_a"], r["span_b"])),
            "low": float(min(r["span_a"], r["span_b"])),
            "close": float(r["span_a"]),
        }
        for _, r in _ku_df.iterrows()
        if np.isfinite(float(r["span_a"])) and np.isfinite(float(r["span_b"]))
    ]
    del _ku_df

    # Valuation bands
    per_levels: list = valuation.get("per_levels", [])
    pbr_levels: list = valuation.get("pbr_levels", [])
    eps_data: list[dict] = valuation.get("eps", [])
    bps_data: list[dict] = valuation.get("bps", [])

    per_palette = ["#73a7ff", "#5c8ff2", "#4576de", "#315fc0", "#234b9f"]
    pbr_palette = ["#f0a36b", "#e68f54", "#d5793c", "#bf6528", "#9e4e1b"]

    eps_df = pd.DataFrame(eps_data) if eps_data else pd.DataFrame(columns=["time", "value"])
    bps_df = pd.DataFrame(bps_data) if bps_data else pd.DataFrame(columns=["time", "value"])
    if not eps_df.empty:
        eps_df["time"] = pd.to_datetime(eps_df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
        eps_df["value"] = pd.to_numeric(eps_df["value"], errors="coerce")
        eps_df = eps_df.dropna(subset=["time", "value"]).reset_index(drop=True)
    if not bps_df.empty:
        bps_df["time"] = pd.to_datetime(bps_df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
        bps_df["value"] = pd.to_numeric(bps_df["value"], errors="coerce")
        bps_df = bps_df.dropna(subset=["time", "value"]).reset_index(drop=True)

    per_levels_current = _normalize_levels(per_levels, [])
    pbr_levels_current = _normalize_levels(pbr_levels, [])

    # ---------------------------------------------------------------------------
    # Create chart
    # ---------------------------------------------------------------------------

    chart = Chart(
        width=1440,
        height=880,
        title=f"{ticker}  ·  {market.upper()}  ·  Market Data",
        toolbox=True,
    )
    chart.layout(
        background_color="#1e222d",
        text_color="#d1d4dc",
        font_size=12,
        font_family="Arial",
    )
    chart.candle_style(
        up_color="#26a69a",
        down_color="#ef5350",
        wick_up_color="#26a69a",
        wick_down_color="#ef5350",
        border_up_color="#26a69a",
        border_down_color="#ef5350",
    )
    chart.volume_config(
        up_color="rgba(38,166,154,0.5)",
        down_color="rgba(239,83,80,0.5)",
    )
    chart.crosshair(
        mode="normal",
        vert_color="#758696",
        vert_style="dotted",
        horz_color="#758696",
        horz_style="dotted",
    )
    chart.grid(vert_enabled=True, horz_enabled=True)
    chart.watermark(ticker, font_size=22, color="rgba(255,255,255,0.07)")

    # Set OHLCV data
    chart.set(df)

    # Line mode overlay (column must match line name)
    if chart_type == "line":
        close_col = "adj_close" if "adj_close" in df.columns else "close"
        line_name = ticker
        close_data = (
            df[["time", close_col]]
            .rename(columns={close_col: line_name})
            .dropna()
        )
        line_main = chart.create_line(line_name, color="#2962ff", width=2)
        line_main.set(close_data)

    # ---------------------------------------------------------------------------
    # Pre-create all indicator lines (hidden — will be activated via topbar)
    # ---------------------------------------------------------------------------

    _IND_LINE_OPTS = {"price_line": False, "price_label": False}

    ma_lines: dict[str, object] = {}
    for key, (label, _fn, _period, color) in MA_META.items():
        if key in ma_dfs:
            ma_lines[key] = chart.create_line(label, color=color, width=1, **_IND_LINE_OPTS)

    bb_lines: dict[str, object] = {}
    for lbl, (color, _s) in BB_DEFS.items():
        if not bb_dfs[lbl].empty:
            bb_lines[lbl] = chart.create_line(lbl, color=color, width=1, **_IND_LINE_OPTS)

    ichi_lines: dict[str, object] = {}
    for lbl, (color, line_df) in ICHI_DEFS.items():
        if not line_df.empty:
            ichi_lines[lbl] = chart.create_line(lbl, color=color, width=1, **_IND_LINE_OPTS)

    # Ichimoku cloud — native LWC candlestick series (no opacity-mask artifacts)
    # open=span_b / close=span_a → upColor (green) when span_a≥span_b, else downColor (red)
    if kumo_records:
        chart.run_script(
            f"""if (typeof {chart.id}._kumoSeries === 'undefined') {{
  {chart.id}._kumoSeries = {chart.id}.chart.addCandlestickSeries({{
    upColor: 'rgba(38, 166, 154, 0.28)',
    downColor: 'rgba(220, 76, 70, 0.28)',
    borderUpColor: 'rgba(0,0,0,0)',
    borderDownColor: 'rgba(0,0,0,0)',
    wickUpColor: 'rgba(0,0,0,0)',
    wickDownColor: 'rgba(0,0,0,0)',
    lastValueVisible: false,
    priceLineVisible: false,
  }});
  {chart.id}._kumoSeries.setData({json.dumps(kumo_records)});
  {chart.id}._kumoSeries.applyOptions({{visible: false}});
}}"""
        )
    ichi_cloud_visible = False

    # Bollinger Bands shaded fill (same candlestick trick, single blue color)
    if bb_fill_records:
        chart.run_script(
            f"""if (typeof {chart.id}._bbFillSeries === 'undefined') {{
  {chart.id}._bbFillSeries = {chart.id}.chart.addCandlestickSeries({{
    upColor: 'rgba(41, 98, 255, 0.08)',
    downColor: 'rgba(41, 98, 255, 0.08)',
    borderUpColor: 'rgba(0,0,0,0)',
    borderDownColor: 'rgba(0,0,0,0)',
    wickUpColor: 'rgba(0,0,0,0)',
    wickDownColor: 'rgba(0,0,0,0)',
    lastValueVisible: false,
    priceLineVisible: false,
  }});
  {chart.id}._bbFillSeries.setData({json.dumps(bb_fill_records)});
  {chart.id}._bbFillSeries.applyOptions({{visible: false}});
}}"""
        )

    per_line_pairs: list[tuple] = []
    pbr_line_pairs: list[tuple] = []

    def _clear_line_pairs(line_pairs: list[tuple]) -> None:
        for line, _ in line_pairs:
            try:
                line.delete()
            except Exception:
                pass

    def _rebuild_per_lines() -> None:
        nonlocal per_line_pairs
        _clear_line_pairs(per_line_pairs)
        per_line_pairs = []
        for lbl, color, band_df in _build_band_defs(eps_df, per_levels_current, "PER", per_palette):
            per_line_pairs.append((
                chart.create_line(lbl, color=color, width=1, price_line=False, price_label=False),
                band_df,
            ))

    def _rebuild_pbr_lines() -> None:
        nonlocal pbr_line_pairs
        _clear_line_pairs(pbr_line_pairs)
        pbr_line_pairs = []
        for lbl, color, band_df in _build_band_defs(bps_df, pbr_levels_current, "PBR", pbr_palette):
            pbr_line_pairs.append((
                chart.create_line(lbl, color=color, width=1, price_line=False, price_label=False),
                band_df,
            ))

    _rebuild_per_lines()
    _rebuild_pbr_lines()

    # Tool overlays (Fib / Long Position / Date-Price Range)
    drag_salt = chart.id.split(".")[-1]
    fib_svg = (
        # Diagonal guide line connecting the two anchor points
        '<path class="tv-icon-guide" d="M7.5 21L21.5 8"/>'
        # Four horizontal fib level lines (1.0, 0.618, 0.382, 0.0)
        '<path class="tv-icon-stroke" d="M7.5 8H21.5"/>'
        '<path class="tv-icon-stroke" d="M7.5 12H21.5"/>'
        '<path class="tv-icon-stroke" d="M7.5 16.5H21.5"/>'
        '<path class="tv-icon-stroke" d="M7.5 21H21.5"/>'
        # Anchor dots
        '<circle class="tv-icon-fill" cx="7.5" cy="21" r="1.3"/>'
        '<circle class="tv-icon-fill" cx="21.5" cy="8" r="1.3"/>'
    )
    long_svg = (
        # Green profit zone (above entry)
        '<rect class="tv-icon-accent-up-fill" x="8" y="8.5" width="13" height="5.5" rx="0.4"/>'
        # Red loss zone (below entry)
        '<rect class="tv-icon-accent-down-fill" x="8" y="15" width="13" height="5.5" rx="0.4"/>'
        # TP / Entry / SL boundary lines
        '<path class="tv-icon-stroke" d="M8 8.5H21"/>'
        '<path class="tv-icon-stroke" d="M8 15H21"/>'
        '<path class="tv-icon-stroke" d="M8 20.5H21"/>'
        # Entry anchor dot (left-middle)
        '<circle class="tv-icon-fill" cx="8" cy="15" r="1.3"/>'
    )
    range_svg = (
        # Outer rectangle
        '<rect class="tv-icon-stroke" x="7.5" y="7.5" width="14" height="14" rx="0.8"/>'
        # Horizontal bidirectional arrow (body + arrowheads)
        '<path class="tv-icon-guide" d="M11 14.5H18"/>'
        '<path class="tv-icon-guide" d="M11 14.5L13 12.7M11 14.5L13 16.3"/>'
        '<path class="tv-icon-guide" d="M18 14.5L16 12.7M18 14.5L16 16.3"/>'
        # Vertical bidirectional arrow (body + arrowheads)
        '<path class="tv-icon-guide" d="M14.5 11V18"/>'
        '<path class="tv-icon-guide" d="M14.5 11L12.7 13M14.5 11L16.3 13"/>'
        '<path class="tv-icon-guide" d="M14.5 18L12.7 16M14.5 18L16.3 16"/>'
        # Corner anchor dots
        '<circle class="tv-icon-fill" cx="7.5" cy="7.5" r="1.2"/>'
        '<circle class="tv-icon-fill" cx="21.5" cy="21.5" r="1.2"/>'
    )
    callout_svg = (
        # Tail line from anchor point to speech bubble corner
        '<path class="tv-icon-stroke" d="M7.5 20.5L12 15.5"/>'
        # Speech bubble rectangle
        '<rect class="tv-icon-stroke" x="11" y="7" width="10.5" height="8.5" rx="1.8"/>'
        # Text guide lines inside bubble
        '<path class="tv-icon-guide" d="M13.2 10H19.5"/>'
        '<path class="tv-icon-guide" d="M13.2 12.5H17.8"/>'
    )
    text_svg = (
        # Horizontal top bar
        '<path class="tv-icon-stroke" d="M8.5 8.5H20.5"/>'
        # Vertical stem
        '<path class="tv-icon-stroke" d="M14.5 8.5V21"/>'
    )
    chart.run_script(
        f"""
if (!window.__customToolBoxInit_{drag_salt}) {{
  window.__customToolBoxInit_{drag_salt} = true;
  if (!window.__customToolBoxStyleInit) {{
    window.__customToolBoxStyleInit = true;
    const __style = document.createElement('style');
    __style.textContent = `
      .custom-tv-tool-button {{
        --color: #d1d4dc;
        --active-color: #4299ff;
        margin-top: 2px;
        margin-bottom: 2px;
      }}
      .custom-tv-tool-button svg {{
        display: block;
      }}
      .custom-tv-tool-button .tv-icon-stroke {{
        fill: none;
        stroke: var(--color);
        stroke-width: 1.65;
        stroke-linecap: round;
        stroke-linejoin: round;
      }}
      .custom-tv-tool-button .tv-icon-guide {{
        fill: none;
        stroke: var(--color);
        stroke-opacity: 0.72;
        stroke-width: 1.35;
        stroke-linecap: round;
        stroke-linejoin: round;
      }}
      .custom-tv-tool-button .tv-icon-fill {{
        fill: var(--color);
      }}
      .custom-tv-tool-button .tv-icon-accent-up-fill {{
        fill: rgba(46, 189, 133, 0.42);
      }}
      .custom-tv-tool-button .tv-icon-accent-down-fill {{
        fill: rgba(220, 76, 70, 0.42);
      }}
      .custom-tv-tool-button.active-toolbox-button .tv-icon-stroke {{
        stroke: var(--active-color);
      }}
      .custom-tv-tool-button.active-toolbox-button .tv-icon-guide {{
        stroke: var(--active-color);
        stroke-opacity: 0.95;
      }}
      .custom-tv-tool-button.active-toolbox-button .tv-icon-fill {{
        fill: var(--active-color);
      }}
      .custom-tv-tool-button.active-toolbox-button .tv-icon-accent-up-fill,
      .custom-tv-tool-button.active-toolbox-button .tv-icon-accent-down-fill {{
        fill: var(--active-color);
        opacity: 0.32;
      }}
      .custom-tv-tool-separator {{
        width: 20px;
        height: 1px;
        margin: 7px 0 5px 0;
        background: rgba(209, 212, 220, 0.30);
        opacity: 1.0;
      }}
    `;
    document.head.appendChild(__style);
  }}
  const __handler = {chart.id};
  const __toolBoxObj = __handler.toolBox || null;
  if (__toolBoxObj) {{
    const __series = __handler.series;
    const __longSettings = {{
      accountSize: 100000.0,
      riskType: 'percent',   // 'percent' | 'absolute'
      riskValue: 1.0,
      lotSize: 1.0,
      leverage: 1.0,
      pointValue: 1.0,
      qtyPrecision: 2,
    }};
    const __lineStyle = (window.LightweightCharts && LightweightCharts.LineStyle)
      ? LightweightCharts.LineStyle
      : {{ Solid: 0, Dotted: 1, Dashed: 2 }};
    let __activeMode = null;
    let __startPoint = null;
    let __previewSet = null;
    const __customButtons = {{}};
    const __fibGroups = new Set();
    const __longGroups = new Set();
    const __calloutGroups = new Set();
    const __textGroups = new Set();
    const __rangeGroups = new Set();

    const __toNumber = (value) => {{
      if (value === null || value === undefined || value === '') return NaN;
      if (typeof value === 'number') return value;
      if (value && typeof value.valueOf === 'function') return Number(value.valueOf());
      return Number(value);
    }};

    const __safePositive = (value, fallback = 1.0) => {{
      const v = __toNumber(value);
      return Number.isFinite(v) && v > 0 ? v : fallback;
    }};

    const __floorTo = (value, precision) => {{
      const p = Math.max(0, Math.floor(__toNumber(precision)));
      const m = Math.pow(10, p);
      if (!Number.isFinite(value)) return 0;
      return Math.floor(Math.max(value, 0) * m + 1e-9) / m;
    }};

    const __fmt = (value, digits = 2) => {{
      if (!Number.isFinite(value)) return 'n/a';
      return value.toLocaleString(undefined, {{
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      }});
    }};

    const __fmtSigned = (value, digits = 2, prefix = '') => {{
      if (!Number.isFinite(value)) return 'n/a';
      const sign = value >= 0 ? '+' : '';
      return `${{sign}}${{prefix}}${{__fmt(value, digits)}}`;
    }};

    const __clamp = (value, min, max) => {{
      if (!Number.isFinite(value)) return min;
      return Math.max(min, Math.min(max, value));
    }};

    const __pointFromEvent = (evt) => {{
      if (!evt || !evt.point || evt.logical === undefined || evt.logical === null) return null;
      const logical = __toNumber(evt.logical);
      if (!Number.isFinite(logical)) return null;
      const priceRaw = __series.coordinateToPrice(evt.point.y);
      const price = __toNumber(priceRaw);
      if (!Number.isFinite(price)) return null;
      return {{ time: evt.time || null, logical, price }};
    }};

    const __point = (base, priceOverride = null) => {{
      const timeValue = (base && base.time && typeof base.time === 'object') ? {{ ...base.time }} : (base ? base.time : null);
      const price = Number.isFinite(priceOverride) ? priceOverride : (base ? base.price : NaN);
      return {{
        time: timeValue || null,
        logical: base ? __toNumber(base.logical) : NaN,
        price: __toNumber(price),
      }};
    }};

    const __pointByLogical = (logical, price) => {{
      return {{
        time: null,
        logical: __toNumber(logical),
        price: __toNumber(price),
      }};
    }};

    const __collectDrawings = (setObj) => {{
      if (!setObj || !setObj.group) return [];
      return [setObj.group];
    }};

    const __attachPreview = (setObj) => {{
      __collectDrawings(setObj).forEach((drawing) => {{
        __series.attachPrimitive(drawing);
      }});
    }};

    const __detachPreview = () => {{
      if (!__previewSet) return;
      __collectDrawings(__previewSet).forEach((drawing) => {{
        try {{
          drawing.detach();
        }} catch (_err) {{}}
      }});
      __previewSet = null;
    }};

    const __commitFinal = (setObj) => {{
      __collectDrawings(setObj).forEach((drawing) => {{
        __toolBoxObj.addNewDrawing(drawing);
      }});
    }};

    const __fibLevels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0];
    // TV-style palette: purple → blue → teal → green → gold(key) → orange → red
    const __fibColors = ["#9b59b6", "#3498db", "#1abc9c", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"];

    const __hexToRgba = (hex, alpha = 0.18) => {{
      const raw = String(hex || '').trim();
      const normal = raw.startsWith('#') ? raw.slice(1) : raw;
      if (!(normal.length === 3 || normal.length === 6)) {{
        return `rgba(125, 150, 180, ${{alpha}})`;
      }}
      const full = normal.length === 3
        ? normal.split('').map((c) => c + c).join('')
        : normal;
      const r = parseInt(full.slice(0, 2), 16);
      const g = parseInt(full.slice(2, 4), 16);
      const b = parseInt(full.slice(4, 6), 16);
      return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
    }};

    const __trimLabelText = (text, fallback) => {{
      const t = String(text ?? '').trim();
      return t ? t.slice(0, 120) : fallback;
    }};

    class __FibGroup extends Lib.TrendLine {{
      constructor(anchorPoint) {{
        super(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "rgba(0,0,0,0)",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._type = "FibRetracement";
        // Band fill: use the upper level's color at low opacity (highlights the zone)
        this._bandBoxes = __fibLevels.slice(1).map((_, idx) => {{
          return new Lib.Box(
            __point(anchorPoint),
            __point(anchorPoint),
            {{
              lineColor: "rgba(0,0,0,0)",
              fillColor: __hexToRgba(__fibColors[idx + 1], 0.11),
              lineStyle: __lineStyle.Solid,
              width: 1,
            }}
          );
        }});
        // Level lines: 0.0/0.618/1.0 = solid+thick, 0.5 = dashed, rest = dotted
        this._levelLines = __fibLevels.map((lv, idx) => {{
          const solid  = lv === 0.0 || lv === 0.618 || lv === 1.0;
          const dashed = lv === 0.5;
          return new Lib.TrendLine(
            __point(anchorPoint),
            __point(anchorPoint),
            {{
              lineColor: __fibColors[idx],
              lineStyle: solid ? __lineStyle.Solid : dashed ? __lineStyle.Dashed : __lineStyle.Dotted,
              width: solid ? 2 : 1,
            }}
          );
        }});
        this._labels = __fibLevels.map((_, idx) => this._createLevelLabel(__fibColors[idx]));
        this._children = [...this._bandBoxes, ...this._levelLines];
        this._children.forEach((drawing) => __series.attachPrimitive(drawing));
        __fibGroups.add(this);
        this.refreshVisuals();
      }}

      _createLevelLabel(color) {{
        const el = document.createElement('div');
        el.className = 'fib-level-label';
        el.style.position = 'absolute';
        el.style.pointerEvents = 'none';
        el.style.fontSize = '11px';
        el.style.lineHeight = '1.3';
        el.style.fontWeight = '500';
        el.style.fontVariantNumeric = 'tabular-nums';
        el.style.whiteSpace = 'pre';
        el.style.color = color;
        el.style.background = 'rgba(14, 20, 34, 0.84)';
        el.style.borderLeft = `2px solid ${{color}}`;
        el.style.borderRadius = '0 3px 3px 0';
        el.style.padding = '2px 7px 2px 5px';
        el.style.transform = 'translate(0, -50%)';
        el.style.zIndex = '2097';
        __handler.div.appendChild(el);
        return el;
      }}

      _removeLevelLabels() {{
        (this._labels || []).forEach((el) => {{
          try {{ el.remove(); }} catch (_err) {{}}
        }});
      }}

      _setLabelsVisible(visible) {{
        (this._labels || []).forEach((el) => {{
          if (!el) return;
          el.style.display = visible ? 'block' : 'none';
        }});
      }}

      _updateLevelLabel(el, text, x, y, containerW, containerH) {{
        if (!el || !Number.isFinite(x) || !Number.isFinite(y)) {{
          if (el) el.style.display = 'none';
          return;
        }}
        el.textContent = text;
        el.style.left = `${{Math.round(__clamp(x, 4, Math.max(4, containerW - 4)))}}px`;
        el.style.top = `${{Math.round(__clamp(y, 4, Math.max(4, containerH - 4)))}}px`;
        el.style.display = 'block';
      }}

      _priceToY(price) {{
        return __toNumber(__series.priceToCoordinate(price));
      }}

      _logicalToX(logical) {{
        return __toNumber(__handler.chart.timeScale().logicalToCoordinate(logical));
      }}

      _levelPrice(top, bottom, coeff) {{
        return top - (top - bottom) * coeff;
      }}

      refreshVisuals() {{
        const p1 = this.p1;
        const p2 = this.p2;
        if (!p1 || !p2) {{
          this._setLabelsVisible(false);
          return;
        }}
        let leftLogical = Math.min(__toNumber(p1.logical), __toNumber(p2.logical));
        let rightLogical = Math.max(__toNumber(p1.logical), __toNumber(p2.logical));
        const price1 = __toNumber(p1.price);
        const price2 = __toNumber(p2.price);
        if (!Number.isFinite(leftLogical) || !Number.isFinite(rightLogical) || !Number.isFinite(price1) || !Number.isFinite(price2)) {{
          this._setLabelsVisible(false);
          return;
        }}
        if (rightLogical < leftLogical + 0.02) {{
          rightLogical = leftLogical + 0.02;
          this.p2.logical = rightLogical;
          this.p2.time = __series.dataByIndex(rightLogical)?.time || this.p2.time || null;
        }}
        const top = Math.max(price1, price2);
        const bottom = Math.min(price1, price2);
        const levelPrices = __fibLevels.map((lv) => this._levelPrice(top, bottom, lv));

        __fibLevels.forEach((_, idx) => {{
          const y = levelPrices[idx];
          this._levelLines[idx].updatePoints(
            __pointByLogical(leftLogical, y),
            __pointByLogical(rightLogical, y),
          );
        }});
        for (let i = 1; i < levelPrices.length; i += 1) {{
          const upper = Math.max(levelPrices[i - 1], levelPrices[i]);
          const lower = Math.min(levelPrices[i - 1], levelPrices[i]);
          this._bandBoxes[i - 1].updatePoints(
            __pointByLogical(leftLogical, upper),
            __pointByLogical(rightLogical, lower),
          );
        }}

        const xLeft  = this._logicalToX(leftLogical);
        const xRight = this._logicalToX(rightLogical);
        const containerW = __handler.div.clientWidth || 0;
        const containerH = __handler.div.clientHeight || 0;
        if (!Number.isFinite(xLeft) || !Number.isFinite(xRight) || containerW < 20 || containerH < 20) {{
          this._setLabelsVisible(false);
          return;
        }}
        // Hide labels when the entire fib range is scrolled off-screen
        if (xRight <= 0 || xLeft >= containerW) {{
          this._setLabelsVisible(false);
          return;
        }}
        // Labels sit at the right edge of the range; if the range extends past the
        // viewport's right edge the labels pin to the chart boundary (TV behaviour).
        const labelX = xRight < containerW - 4 ? xRight + 6 : containerW - 2;
        __fibLevels.forEach((lv, idx) => {{
          const y = this._priceToY(levelPrices[idx]);
          const coeffText = lv.toFixed(3).replace(/0+$/, '').replace(/\\.$/, '');
          const text = `${{coeffText}}  ${{__fmt(levelPrices[idx], 2)}}`;
          this._updateLevelLabel(this._labels[idx], text, labelX, y, containerW, containerH);
        }});
        this._setLabelsVisible(true);
      }}

      updatePoints(...pts) {{
        super.updatePoints(...pts);
        this.refreshVisuals();
      }}

      _onDrag(diff) {{
        super._onDrag(diff);
        this.refreshVisuals();
      }}

      detach() {{
        this._children.forEach((drawing) => {{
          try {{
            drawing.detach();
          }} catch (_err) {{}}
        }});
        this._removeLevelLabels();
        __fibGroups.delete(this);
        super.detach();
      }}
    }}

    class __CalloutGroup extends Lib.TrendLine {{
      constructor(anchorPoint) {{
        super(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "rgba(134, 173, 255, 0.95)",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._type = "Callout";
        this._text = "콜아웃";
        this._label = this._createLabel();
        this._syncTextFromOptions();
        __calloutGroups.add(this);
        this.refreshVisuals();
      }}

      _createLabel() {{
        const el = document.createElement('div');
        el.className = 'callout-label';
        el.style.position = 'absolute';
        el.style.pointerEvents = 'auto';
        el.style.cursor = 'text';
        el.style.padding = '4px 8px';
        el.style.borderRadius = '5px';
        el.style.fontSize = '11px';
        el.style.lineHeight = '1.25';
        el.style.fontWeight = '600';
        el.style.color = '#f5f8ff';
        el.style.background = 'rgba(26, 35, 52, 0.92)';
        el.style.border = '1px solid rgba(134, 173, 255, 0.45)';
        el.style.boxShadow = '0 2px 8px rgba(15, 22, 36, 0.22)';
        el.style.whiteSpace = 'pre';
        el.style.transform = 'translate(0, -50%)';
        el.style.zIndex = '2099';
        el.addEventListener('dblclick', (e) => {{
          e.stopPropagation();
          __showInlineEditor(this, parseFloat(el.style.left) || 40, parseFloat(el.style.top) || 40, '콜아웃');
        }});
        __handler.div.appendChild(el);
        return el;
      }}

      _syncTextFromOptions() {{
        this._text = __trimLabelText(this._options?.text, "콜아웃");
        this._options.text = this._text;
      }}

      getText() {{
        return this._text;
      }}

      setText(text) {{
        this._text = __trimLabelText(text, "콜아웃");
        this._options.text = this._text;
        this.refreshVisuals();
      }}

      applyOptions(options) {{
        super.applyOptions(options);
        this._syncTextFromOptions();
        this.refreshVisuals();
      }}

      _setLabelVisible(visible) {{
        if (!this._label) return;
        this._label.style.display = visible ? 'block' : 'none';
      }}

      _pointToScreen(pt) {{
        if (!pt) return null;
        const logical = __toNumber(pt.logical);
        const x = Number.isFinite(logical)
          ? __toNumber(__handler.chart.timeScale().logicalToCoordinate(logical))
          : __toNumber(__handler.chart.timeScale().timeToCoordinate(pt.time));
        const y = __toNumber(__series.priceToCoordinate(pt.price));
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        return {{ x, y }};
      }}

      refreshVisuals() {{
        const p2xy = this._pointToScreen(this.p2);
        const containerW = __handler.div.clientWidth || 0;
        const containerH = __handler.div.clientHeight || 0;
        if (!p2xy || containerW < 20 || containerH < 20) {{
          this._setLabelVisible(false);
          return;
        }}
        this._label.textContent = this._text;
        const x = __clamp(p2xy.x + 4, 4, Math.max(4, containerW - 4));
        const y = __clamp(p2xy.y, 4, Math.max(4, containerH - 4));
        this._label.style.left = `${{Math.round(x)}}px`;
        this._label.style.top = `${{Math.round(y)}}px`;
        this._setLabelVisible(true);
      }}

      updatePoints(...pts) {{
        super.updatePoints(...pts);
        this.refreshVisuals();
      }}

      _onDrag(diff) {{
        super._onDrag(diff);
        this.refreshVisuals();
      }}

      detach() {{
        try {{ this._label.remove(); }} catch (_err) {{}}
        __calloutGroups.delete(this);
        super.detach();
      }}
    }}

    class __TextGroup extends Lib.TrendLine {{
      constructor(anchorPoint) {{
        super(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "rgba(0, 0, 0, 0.01)",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._type = "Text";
        this._text = "텍스트";
        this._label = this._createLabel();
        this._syncTextFromOptions();
        this._syncSecondPoint();
        __textGroups.add(this);
        this.refreshVisuals();
      }}

      _createLabel() {{
        const el = document.createElement('div');
        el.className = 'floating-text-label';
        el.style.position = 'absolute';
        el.style.pointerEvents = 'auto';
        el.style.cursor = 'text';
        el.style.fontSize = '13px';
        el.style.lineHeight = '1.2';
        el.style.fontWeight = '700';
        el.style.color = '#eef3ff';
        el.style.textShadow = '0 1px 2px rgba(0,0,0,0.55)';
        el.style.whiteSpace = 'pre';
        el.style.transform = 'translate(2px, -50%)';
        el.style.zIndex = '2099';
        el.addEventListener('dblclick', (e) => {{
          e.stopPropagation();
          __showInlineEditor(this, parseFloat(el.style.left) || 40, parseFloat(el.style.top) || 40, '텍스트');
        }});
        __handler.div.appendChild(el);
        return el;
      }}

      _syncTextFromOptions() {{
        this._text = __trimLabelText(this._options?.text, "텍스트");
        this._options.text = this._text;
      }}

      _syncSecondPoint() {{
        if (!this.p1 || !this.p2) return;
        const logical = __toNumber(this.p1.logical);
        if (!Number.isFinite(logical)) return;
        this.p2.logical = logical + 0.02;
        this.p2.time = __series.dataByIndex(this.p2.logical)?.time || this.p1.time || null;
        this.p2.price = __toNumber(this.p1.price);
      }}

      getText() {{
        return this._text;
      }}

      setText(text) {{
        this._text = __trimLabelText(text, "텍스트");
        this._options.text = this._text;
        this.refreshVisuals();
      }}

      applyOptions(options) {{
        super.applyOptions(options);
        this._syncTextFromOptions();
        this.refreshVisuals();
      }}

      _setLabelVisible(visible) {{
        if (!this._label) return;
        this._label.style.display = visible ? 'block' : 'none';
      }}

      _pointToScreen(pt) {{
        if (!pt) return null;
        const logical = __toNumber(pt.logical);
        const x = Number.isFinite(logical)
          ? __toNumber(__handler.chart.timeScale().logicalToCoordinate(logical))
          : __toNumber(__handler.chart.timeScale().timeToCoordinate(pt.time));
        const y = __toNumber(__series.priceToCoordinate(pt.price));
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        return {{ x, y }};
      }}

      refreshVisuals() {{
        const p1xy = this._pointToScreen(this.p1);
        const containerW = __handler.div.clientWidth || 0;
        const containerH = __handler.div.clientHeight || 0;
        if (!p1xy || containerW < 20 || containerH < 20) {{
          this._setLabelVisible(false);
          return;
        }}
        this._label.textContent = this._text;
        const x = __clamp(p1xy.x + 4, 4, Math.max(4, containerW - 4));
        const y = __clamp(p1xy.y, 4, Math.max(4, containerH - 4));
        this._label.style.left = `${{Math.round(x)}}px`;
        this._label.style.top = `${{Math.round(y)}}px`;
        this._setLabelVisible(true);
      }}

      updatePoints(...pts) {{
        super.updatePoints(...pts);
        this._syncSecondPoint();
        this.refreshVisuals();
      }}

      _onDrag(diff) {{
        super._onDrag(diff);
        this._syncSecondPoint();
        this.refreshVisuals();
      }}

      detach() {{
        try {{ this._label.remove(); }} catch (_err) {{}}
        __textGroups.delete(this);
        super.detach();
      }}
    }}

    const __createFibSet = (anchor) => {{
      const group = new __FibGroup(anchor);
      return {{
        kind: 'fib',
        group,
        update: (a, b) => {{
          group.updatePoints(__point(a), __point(b));
        }},
      }};
    }};

    const __createCalloutSet = (anchor) => {{
      const group = new __CalloutGroup(anchor);
      return {{
        kind: 'callout',
        group,
        update: (a, b) => {{
          group.updatePoints(__point(a), __point(b));
        }},
      }};
    }};

    const __createTextSet = (anchor) => {{
      const group = new __TextGroup(anchor);
      return {{
        kind: 'text',
        group,
        update: (a, b) => {{
          const target = b || a;
          group.updatePoints(__point(target), __point(target));
        }},
      }};
    }};

    class __LongPositionGroup extends Lib.Box {{
      constructor(anchorPoint) {{
        const entry = __toNumber(anchorPoint.price);
        const tp = Number.isFinite(entry) ? entry * 1.10 : __toNumber(anchorPoint.price);
        const sl = Number.isFinite(entry) ? entry * 0.95 : __toNumber(anchorPoint.price);
        super(
          __point(anchorPoint, tp),
          __point(anchorPoint, sl),
          {{
            lineColor: "rgba(146,160,184,0.40)",
            fillColor: "rgba(0,0,0,0.01)",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._type = "ProjectionLong";
        this._entryPrice = entry;
        this._riskBox = new Lib.Box(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "rgba(0,0,0,0)",
            fillColor: "rgba(220,76,70,0.26)",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._rewardBox = new Lib.Box(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "rgba(0,0,0,0)",
            fillColor: "rgba(46,189,133,0.26)",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._entryLine = new Lib.TrendLine(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "#f2f2f2",
            lineStyle: __lineStyle.Solid,
            width: 2,
          }}
        );
        this._slLine = new Lib.TrendLine(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "#dc4c46",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._tpLine = new Lib.TrendLine(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "#2ebd85",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._children = [this._riskBox, this._rewardBox, this._entryLine, this._slLine, this._tpLine];
        this._labels = this._createLabels();
        this._setLabelsVisible(false);
        this._children.forEach((drawing) => __series.attachPrimitive(drawing));
        __longGroups.add(this);
        this.refreshVisuals();
      }}

      _minGap() {{
        const base = Math.abs(__toNumber(this._entryPrice));
        return Math.max(base * 1e-6, 1e-6);
      }}

      _createLabel(className, bgColor) {{
        const el = document.createElement('div');
        el.className = className;
        el.style.position = 'absolute';
        el.style.pointerEvents = 'none';
        el.style.padding = '2px 6px';
        el.style.borderRadius = '4px';
        el.style.fontSize = '11px';
        el.style.lineHeight = '1.25';
        el.style.fontWeight = '600';
        el.style.whiteSpace = 'pre';
        el.style.color = '#f5f7fa';
        el.style.background = bgColor;
        el.style.zIndex = '2100';
        __handler.div.appendChild(el);
        return el;
      }}

      _createLabels() {{
        return {{
          target: this._createLabel('long-target-label', 'rgba(33, 170, 137, 0.95)'),
          center: this._createLabel('long-center-label', 'rgba(28, 38, 58, 0.92)'),
          stop: this._createLabel('long-stop-label', 'rgba(224, 77, 90, 0.96)'),
        }};
      }}

      _removeLabels() {{
        Object.values(this._labels).forEach((el) => {{
          try {{ el.remove(); }} catch (_err) {{}}
        }});
      }}

      _setLabelsVisible(visible) {{
        Object.values(this._labels).forEach((el) => {{
          if (!el) return;
          el.style.display = visible ? 'block' : 'none';
        }});
      }}

      _updateLabel(el, text, x, y, containerW, containerH) {{
        if (!el) return;
        el.textContent = text;
        const maxX = Math.max(0, containerW - (el.offsetWidth || 0) - 2);
        const maxY = Math.max(0, containerH - (el.offsetHeight || 0) - 2);
        const left = __clamp(x, 2, maxX);
        const top = __clamp(y, 2, maxY);
        el.style.left = `${{left}}px`;
        el.style.top = `${{top}}px`;
      }}

      _priceToY(price) {{
        const y = __series.priceToCoordinate(price);
        return __toNumber(y);
      }}

      _pointToX(pt) {{
        if (!pt) return NaN;
        if (pt.time) {{
          return __toNumber(__handler.chart.timeScale().timeToCoordinate(pt.time));
        }}
        return __toNumber(__handler.chart.timeScale().logicalToCoordinate(pt.logical));
      }}

      _syncEntryFromOptions() {{
        const optEntry = __toNumber(this._options?.entryPrice);
        if (Number.isFinite(optEntry)) {{
          this._entryPrice = optEntry;
        }}
        if (!Number.isFinite(this._entryPrice)) {{
          const p1Price = __toNumber(this.p1?.price);
          const p2Price = __toNumber(this.p2?.price);
          if (Number.isFinite(p1Price) && Number.isFinite(p2Price)) {{
            this._entryPrice = (p1Price + p2Price) / 2.0;
          }} else {{
            this._entryPrice = p1Price;
          }}
        }}
      }}

      setEntryPrice(price) {{
        const v = __toNumber(price);
        if (!Number.isFinite(v)) return;
        this._entryPrice = v;
        this._options.entryPrice = v;
        this.refreshVisuals();
      }}

      updateFromAnchor(a, b) {{
        const entry = __toNumber(a?.price);
        const bPrice = __toNumber(b?.price);
        this._entryPrice = Number.isFinite(entry) ? entry : __toNumber(this._entryPrice);

        const defaultTp = Number.isFinite(this._entryPrice) ? this._entryPrice * 1.10 : bPrice;
        const defaultSl = Number.isFinite(this._entryPrice) ? this._entryPrice * 0.95 : bPrice;
        const tp = Number.isFinite(bPrice) && bPrice >= this._entryPrice ? bPrice : defaultTp;
        const sl = Number.isFinite(bPrice) && bPrice < this._entryPrice ? bPrice : defaultSl;

        const left = __point(a, tp);
        const right = __point(b, sl);
        super.updatePoints(left, right);
        this.refreshVisuals();
      }}

      refreshVisuals() {{
        const p1 = this.p1;
        const p2 = this.p2;
        if (!p1 || !p2) {{
          this._setLabelsVisible(false);
          return;
        }}

        this._syncEntryFromOptions();

        let leftLogical = __toNumber(p1.logical);
        let rightLogical = __toNumber(p2.logical);
        if (!Number.isFinite(leftLogical) || !Number.isFinite(rightLogical)) {{
          this._setLabelsVisible(false);
          return;
        }}
        const gap = this._minGap();
        let entry = __toNumber(this._entryPrice);
        let tp = Math.max(__toNumber(p1.price), entry + gap);
        let sl = Math.min(__toNumber(p2.price), entry - gap);
        if (!Number.isFinite(tp)) tp = entry + gap;
        if (!Number.isFinite(sl)) sl = entry - gap;
        p1.price = tp;
        p2.price = sl;
        this._options.entryPrice = entry;

        const accountSize = __safePositive(__longSettings.accountSize, 100000.0);
        const lotSize = __safePositive(__longSettings.lotSize, 1.0);
        const leverage = __safePositive(__longSettings.leverage, 1.0);
        const pointValue = __safePositive(__longSettings.pointValue, 1.0);
        const qtyPrecision = Math.max(0, Math.floor(__toNumber(__longSettings.qtyPrecision)));
        const riskInput = __safePositive(__longSettings.riskValue, 1.0);
        const riskSize = (__longSettings.riskType === 'absolute')
          ? riskInput
          : (accountSize * riskInput / 100.0);

        const riskDist = entry - sl;
        const rewardDist = tp - entry;
        const qtyRisk = (riskDist > 0)
          ? (riskSize / (riskDist * pointValue)) / lotSize
          : 0;
        const qtyLvg = (entry > 0)
          ? (accountSize * leverage / entry) * pointValue / lotSize
          : 0;
        const qty = __floorTo(Math.min(qtyRisk, qtyLvg), qtyPrecision);

        const tpPct = entry !== 0 ? (rewardDist / entry) * 100.0 : 0.0;
        const slPct = entry !== 0 ? (riskDist / entry) * 100.0 : 0.0;
        const rr = riskDist > 0 ? rewardDist / riskDist : NaN;

        const profitPnl = rewardDist * qty * pointValue * lotSize;
        const lossPnl = (sl - entry) * qty * pointValue * lotSize;
        const amountTp = accountSize + profitPnl;
        const amountSl = accountSize + lossPnl;

        this._riskBox.updatePoints(__point(p1, Math.max(entry, sl)), __point(p2, Math.min(entry, sl)));
        this._rewardBox.updatePoints(__point(p1, Math.max(tp, entry)), __point(p2, Math.min(tp, entry)));
        this._entryLine.updatePoints(__point(p1, entry), __point(p2, entry));
        this._slLine.updatePoints(__point(p1, sl), __point(p2, sl));
        this._tpLine.updatePoints(__point(p1, tp), __point(p2, tp));

        const xLeft = this._pointToX(p1);
        const xRight = this._pointToX(p2);
        const yEntry = this._priceToY(entry);
        const ySl = this._priceToY(sl);
        const yTp = this._priceToY(tp);
        if (!Number.isFinite(xLeft) || !Number.isFinite(xRight) || !Number.isFinite(yEntry) || !Number.isFinite(ySl) || !Number.isFinite(yTp)) {{
          this._setLabelsVisible(false);
          return;
        }}

        const innerX = Math.min(xLeft, xRight) + 8;
        const greenCenterY = (Math.min(yTp, yEntry) + Math.max(yTp, yEntry)) / 2 - 10;
        const redCenterY = (Math.min(ySl, yEntry) + Math.max(ySl, yEntry)) / 2 - 10;
        const centerY = yEntry - 16;
        const containerW = __handler.div.clientWidth || 0;
        const containerH = __handler.div.clientHeight || 0;

        const rrText = Number.isFinite(rr) ? __fmt(rr, 2) : '\u2014';
        const targetText = `+${{__fmt(tpPct, 2)}}%  +${{__fmt(profitPnl, 0)}}`;
        const centerText = `진입 ${{__fmt(entry, 2)}}  |  R/R ${{rrText}}  |  수량 ${{__fmt(qty, qtyPrecision)}}`;
        const stopText = `-${{__fmt(slPct, 2)}}%  -${{__fmt(Math.abs(lossPnl), 0)}}`;

        this._updateLabel(this._labels.target, targetText, innerX, greenCenterY, containerW, containerH);
        this._updateLabel(this._labels.center, centerText, innerX, centerY, containerW, containerH);
        this._updateLabel(this._labels.stop, stopText, innerX, redCenterY, containerW, containerH);
        this._setLabelsVisible(true);
      }}

      updatePoints(...pts) {{
        super.updatePoints(...pts);
        this.refreshVisuals();
      }}

      _onDrag(diff) {{
        super._onDrag(diff);
        if (this._state === 2) {{
          this._entryPrice = __toNumber(this._entryPrice) + __toNumber(diff.price);
        }} else if (this._state === 3 || this._state === 6) {{
          this.p1.price = Math.max(__toNumber(this.p1.price), this._entryPrice + this._minGap());
        }} else if (this._state === 4 || this._state === 5) {{
          this.p2.price = Math.min(__toNumber(this.p2.price), this._entryPrice - this._minGap());
        }}
        this.refreshVisuals();
      }}

      detach() {{
        this._children.forEach((drawing) => {{
          try {{
            drawing.detach();
          }} catch (_err) {{}}
        }});
        this._removeLabels();
        __longGroups.delete(this);
        super.detach();
      }}
    }}

    const __refreshLongGroups = () => {{
      __longGroups.forEach((group) => {{
        try {{
          group.refreshVisuals();
        }} catch (_err) {{}}
      }});
    }};

    const __createLongSet = (anchor) => {{
      const group = new __LongPositionGroup(anchor);
      return {{
        kind: 'long',
        group,
        update: (a, b) => {{
          group.updateFromAnchor(a, b);
        }},
      }};
    }};

    class __RangeGroup extends Lib.Box {{
      constructor(anchorPoint) {{
        super(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "rgba(67,157,255,0.80)",
            fillColor: "rgba(67,157,255,0.15)",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._type = "DatePriceRange";
        this._crossH = new Lib.TrendLine(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "rgba(41,98,255,0.95)",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._crossV = new Lib.TrendLine(
          __point(anchorPoint),
          __point(anchorPoint),
          {{
            lineColor: "rgba(41,98,255,0.95)",
            lineStyle: __lineStyle.Solid,
            width: 1,
          }}
        );
        this._label = this._createInfoLabel();
        this._children = [this._crossH, this._crossV];
        this._children.forEach((drawing) => __series.attachPrimitive(drawing));
        __rangeGroups.add(this);
        this.refreshVisuals();
      }}

      _createInfoLabel() {{
        const el = document.createElement('div');
        el.className = 'range-info-label';
        el.style.position = 'absolute';
        el.style.pointerEvents = 'none';
        el.style.padding = '8px 12px';
        el.style.borderRadius = '6px';
        el.style.fontSize = '11px';
        el.style.lineHeight = '1.35';
        el.style.fontWeight = '600';
        el.style.fontVariantNumeric = 'tabular-nums';
        el.style.whiteSpace = 'pre';
        el.style.textAlign = 'center';
        el.style.color = '#d1d4dc';
        el.style.background = 'rgba(24, 32, 50, 0.95)';
        el.style.border = '1px solid rgba(67, 157, 255, 0.40)';
        el.style.boxShadow = '0 2px 8px rgba(10, 18, 32, 0.35)';
        el.style.zIndex = '2100';
        el.style.transform = 'translate(-50%, -100%)';
        __handler.div.appendChild(el);
        return el;
      }}

      _setLabelVisible(visible) {{
        if (!this._label) return;
        this._label.style.display = visible ? 'block' : 'none';
      }}

      _updateLabel(text, centerX, anchorTopY, containerW, containerH) {{
        if (!this._label) return;
        this._label.textContent = text;
        const left = __clamp(centerX, 24, Math.max(24, containerW - 24));
        const top = __clamp(anchorTopY - 12, 8, Math.max(8, containerH - 8));
        this._label.style.left = `${{Math.round(left)}}px`;
        this._label.style.top = `${{Math.round(top)}}px`;
      }}

      _timeToMs(time) {{
        if (time === null || time === undefined) return NaN;
        if (typeof time === 'number') return time * 1000;
        if (typeof time === 'string') {{
          const ms = Date.parse(time);
          return Number.isFinite(ms) ? ms : NaN;
        }}
        if (typeof time === 'object' && Number.isFinite(time.year) && Number.isFinite(time.month) && Number.isFinite(time.day)) {{
          return Date.UTC(time.year, time.month - 1, time.day);
        }}
        return NaN;
      }}

      _priceToY(price) {{
        return __toNumber(__series.priceToCoordinate(price));
      }}

      _logicalToX(logical) {{
        return __toNumber(__handler.chart.timeScale().logicalToCoordinate(logical));
      }}

      _renderOverlay(leftLogical, rightLogical, high, low, diff, pct, bars, days) {{
        const xLeft = this._logicalToX(leftLogical);
        const xRight = this._logicalToX(rightLogical);
        const yTop = this._priceToY(high);
        const containerW = __handler.div.clientWidth || 0;
        const containerH = __handler.div.clientHeight || 0;
        if (!Number.isFinite(xLeft) || !Number.isFinite(xRight) || !Number.isFinite(yTop) || containerW < 20 || containerH < 20) {{
          this._setLabelVisible(false);
          return;
        }}
        const rawLeft = Math.min(xLeft, xRight);
        const rawRight = Math.max(xLeft, xRight);
        const visibleLeft = Math.max(rawLeft, 0);
        const visibleRight = Math.min(rawRight, containerW);
        if (!Number.isFinite(visibleLeft) || !Number.isFinite(visibleRight) || (visibleRight - visibleLeft) < 2) {{
          this._setLabelVisible(false);
          return;
        }}
        const xCenter = (visibleLeft + visibleRight) / 2.0;

        const text = `${{__fmt(diff, 0)}} (${{__fmt(pct, 2)}}%)\n${{bars}} 봉, ${{days}}날`;
        this._updateLabel(text, xCenter, yTop, containerW, containerH);
        this._setLabelVisible(true);
      }}

      refreshVisuals() {{
        const p1 = this.p1;
        const p2 = this.p2;
        if (!p1 || !p2) {{
          this._setLabelVisible(false);
          return;
        }}

        const logical1 = __toNumber(p1.logical);
        const logical2 = __toNumber(p2.logical);
        if (!Number.isFinite(logical1) || !Number.isFinite(logical2)) {{
          this._setLabelVisible(false);
          return;
        }}

        const leftLogical = Math.min(logical1, logical2);
        const rightLogical = Math.max(logical1, logical2);
        const high = Math.max(__toNumber(p1.price), __toNumber(p2.price));
        const low = Math.min(__toNumber(p1.price), __toNumber(p2.price));
        const midLogical = (leftLogical + rightLogical) / 2.0;
        const midPrice = (high + low) / 2.0;

        this._crossH.updatePoints(__pointByLogical(leftLogical, midPrice), __pointByLogical(rightLogical, midPrice));
        this._crossV.updatePoints(__pointByLogical(midLogical, low), __pointByLogical(midLogical, high));

        const diff = high - low;
        const pct = low !== 0 ? (diff / Math.abs(low)) * 100.0 : NaN;
        const bars = Math.max(1, Math.floor(Math.abs(rightLogical - leftLogical)) + 1);
        const t1 = this._timeToMs(p1.time);
        const t2 = this._timeToMs(p2.time);
        const days = (Number.isFinite(t1) && Number.isFinite(t2))
          ? Math.max(0, Math.round(Math.abs(t2 - t1) / 86400000))
          : bars;

        this._renderOverlay(leftLogical, rightLogical, high, low, diff, pct, bars, days);
      }}

      updatePoints(...pts) {{
        super.updatePoints(...pts);
        this.refreshVisuals();
      }}

      _onDrag(diff) {{
        super._onDrag(diff);
        this.refreshVisuals();
      }}

      detach() {{
        this._children.forEach((drawing) => {{
          try {{
            drawing.detach();
          }} catch (_err) {{}}
        }});
        try {{ this._label.remove(); }} catch (_err) {{}}
        __rangeGroups.delete(this);
        super.detach();
      }}
    }}

    const __createRangeSet = (anchor) => {{
      const group = new __RangeGroup(anchor);
      return {{
        kind: 'range',
        group,
        update: (a, b) => {{
          const low = Math.min(__toNumber(a.price), __toNumber(b.price));
          const high = Math.max(__toNumber(a.price), __toNumber(b.price));
          group.updatePoints(__point(a, high), __point(b, low));
        }},
      }};
    }};

    const __refreshFibGroups = () => {{
      __fibGroups.forEach((group) => {{
        try {{
          group.refreshVisuals();
        }} catch (_err) {{}}
      }});
    }};

    const __refreshCalloutGroups = () => {{
      __calloutGroups.forEach((group) => {{
        try {{
          group.refreshVisuals();
        }} catch (_err) {{}}
      }});
    }};

    const __refreshTextGroups = () => {{
      __textGroups.forEach((group) => {{
        try {{
          group.refreshVisuals();
        }} catch (_err) {{}}
      }});
    }};

    const __refreshRangeGroups = () => {{
      __rangeGroups.forEach((group) => {{
        try {{
          group.refreshVisuals();
        }} catch (_err) {{}}
      }});
    }};

    const __refreshOverlayGroups = () => {{
      __refreshFibGroups();
      __refreshLongGroups();
      __refreshCalloutGroups();
      __refreshTextGroups();
      __refreshRangeGroups();
    }};

    __handler.chart.timeScale().subscribeVisibleLogicalRangeChange(() => __refreshOverlayGroups());
    window.addEventListener('resize', () => __refreshOverlayGroups());

    const __builders = {{
      fib: __createFibSet,
      long: __createLongSet,
      range: __createRangeSet,
      callout: __createCalloutSet,
      text: __createTextSet,
    }};

    const __restoreCustomDrawing = (entry) => {{
      if (!entry || !Array.isArray(entry.points) || entry.points.length < 1) return false;
      const p1 = entry.points[0];
      const p2 = entry.points[1] || entry.points[0];
      if (!p1 || !p2) return false;

      let group = null;
      if (entry.type === 'FibRetracement') {{
        group = new __FibGroup(p1);
      }} else if (entry.type === 'ProjectionLong' || entry.type === 'LongPosition') {{
        group = new __LongPositionGroup(p1);
      }} else if (entry.type === 'DatePriceRange') {{
        group = new __RangeGroup(p1);
      }} else if (entry.type === 'Callout') {{
        group = new __CalloutGroup(p1);
      }} else if (entry.type === 'Text') {{
        group = new __TextGroup(p1);
      }}
      if (!group) return false;

      try {{
        if (entry.options && typeof entry.options === 'object') {{
          group.applyOptions(entry.options);
          if (group.setEntryPrice && Number.isFinite(__toNumber(entry.options.entryPrice))) {{
            group.setEntryPrice(entry.options.entryPrice);
          }}
        }}
      }} catch (_err) {{}}

      try {{
        group.updatePoints(__point(p1), __point(p2));
      }} catch (_err) {{
        try {{ group.detach(); }} catch (_detachErr) {{}}
        return false;
      }}

      __toolBoxObj.addNewDrawing(group);
      return true;
    }};

    if (__toolBoxObj.loadDrawings && !__toolBoxObj.__customLoadPatched) {{
      __toolBoxObj.__customLoadPatched = true;
      const __origLoadDrawings = __toolBoxObj.loadDrawings.bind(__toolBoxObj);
      __toolBoxObj.loadDrawings = (drawings) => {{
        if (!Array.isArray(drawings)) {{
          __origLoadDrawings(drawings);
          return;
        }}
        const nativeDrawings = [];
        drawings.forEach((entry) => {{
          if (!__restoreCustomDrawing(entry)) {{
            nativeDrawings.push(entry);
          }}
        }});
        if (nativeDrawings.length > 0) {{
          __origLoadDrawings(nativeDrawings);
        }}
      }};
    }}

    const __setCustomActive = (mode) => {{
      Object.entries(__customButtons).forEach(([key, btn]) => {{
        if (mode && key === mode) btn.classList.add('active-toolbox-button');
        else btn.classList.remove('active-toolbox-button');
      }});
    }};

    const __deactivateBuiltIn = () => {{
      if (__toolBoxObj.activeIcon) {{
        __toolBoxObj.activeIcon.div.classList.remove('active-toolbox-button');
        __toolBoxObj.activeIcon = null;
      }}
      if (__toolBoxObj._drawingTool) {{
        __toolBoxObj._drawingTool.stopDrawing();
      }}
    }};

    const __resetCustom = () => {{
      __activeMode = null;
      __startPoint = null;
      __detachPreview();
      __setCustomActive(null);
      window.setCursor('default');
    }};

    const __activateCustom = (mode) => {{
      if (__activeMode === mode) {{
        __resetCustom();
        return;
      }}
      __deactivateBuiltIn();
      __resetCustom();
      __activeMode = mode;
      __setCustomActive(mode);
      window.setCursor('crosshair');
    }};

    const __makeToolButton = (mode, svgMarkup, title) => {{
      const button = document.createElement('div');
      button.classList.add('toolbox-button');
      button.classList.add('custom-tv-tool-button');
      button.title = title;
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('width', '29');
      svg.setAttribute('height', '29');
      svg.setAttribute('viewBox', '0 0 29 29');
      const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      group.innerHTML = svgMarkup;
      svg.appendChild(group);
      button.appendChild(svg);
      button.addEventListener('click', (evt) => {{
        evt.preventDefault();
        evt.stopPropagation();
        __activateCustom(mode);
      }});
      return button;
    }};

    const __showInlineEditor = (group, anchorX, anchorY, fallback) => {{
      const existing = document.getElementById('tv-inline-text-editor');
      if (existing) existing.remove();

      const containerW = __handler.div.clientWidth || 400;
      const containerH = __handler.div.clientHeight || 300;

      const textarea = document.createElement('textarea');
      textarea.id = 'tv-inline-text-editor';
      textarea.value = __trimLabelText(group.getText(), fallback);
      textarea.rows = Math.max(1, textarea.value.split(String.fromCharCode(10)).length);
      textarea.style.position = 'absolute';
      textarea.style.zIndex = '9999';
      textarea.style.background = 'rgba(18, 26, 44, 0.97)';
      textarea.style.color = '#d1d4dc';
      textarea.style.border = '1.5px solid rgba(67,157,255,0.75)';
      textarea.style.borderRadius = '6px';
      textarea.style.padding = '6px 10px';
      textarea.style.fontSize = '13px';
      textarea.style.fontFamily = 'inherit';
      textarea.style.lineHeight = '1.5';
      textarea.style.resize = 'both';
      textarea.style.minWidth = '140px';
      textarea.style.minHeight = '32px';
      textarea.style.outline = 'none';
      textarea.style.boxShadow = '0 4px 16px rgba(0,0,0,0.55)';
      textarea.style.left = `${{Math.max(4, Math.min(anchorX, containerW - 180))}}px`;
      textarea.style.top  = `${{Math.max(4, Math.min(anchorY, containerH - 64))}}px`;

      let done = false;
      const confirm = () => {{
        if (done) return;
        done = true;
        const text = textarea.value;
        textarea.remove();
        group.setText(text || fallback);
        if (__toolBoxObj.saveDrawings) __toolBoxObj.saveDrawings();
      }};
      const cancel = () => {{
        done = true;
        textarea.remove();
      }};

      textarea.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape') {{ e.preventDefault(); cancel(); return; }}
        if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); confirm(); }}
      }});
      textarea.addEventListener('blur', confirm);

      __handler.div.appendChild(textarea);
      textarea.focus();
      textarea.select();
    }};

    const __maybeEditToolText = (mode, group) => {{
      if (!group || (mode !== 'callout' && mode !== 'text')) return;
      if (typeof group.getText !== 'function' || typeof group.setText !== 'function') return;
      const fallback = mode === 'callout' ? '콜아웃' : '텍스트';
      const labelEl = group._label;
      const x = labelEl ? (parseFloat(labelEl.style.left) || 40) : 40;
      const y = labelEl ? (parseFloat(labelEl.style.top) || 40) : 40;
      __showInlineEditor(group, x, y, fallback);
    }};

    __customButtons.fib = __makeToolButton('fib', {json.dumps(fib_svg)}, 'Fibonacci Retracement');
    __customButtons.long = __makeToolButton('long', {json.dumps(long_svg)}, 'Projection Long');
    __customButtons.range = __makeToolButton('range', {json.dumps(range_svg)}, 'Date & Price Range');
    __customButtons.callout = __makeToolButton('callout', {json.dumps(callout_svg)}, 'Callout');
    __customButtons.text = __makeToolButton('text', {json.dumps(text_svg)}, 'Text');

    const __separator = document.createElement('div');
    __separator.classList.add('custom-tv-tool-separator');
    __toolBoxObj.div.appendChild(__separator);
    __toolBoxObj.div.appendChild(__customButtons.fib);
    __toolBoxObj.div.appendChild(__customButtons.long);
    __toolBoxObj.div.appendChild(__customButtons.range);
    __toolBoxObj.div.appendChild(__customButtons.callout);
    __toolBoxObj.div.appendChild(__customButtons.text);

    if (Array.isArray(__toolBoxObj.buttons)) {{
      __toolBoxObj.buttons.forEach((btnObj) => {{
        if (!btnObj || !btnObj.div) return;
        btnObj.div.addEventListener('click', () => {{
          __resetCustom();
        }});
      }});
    }}

    const __origToolboxClick = __toolBoxObj._onIconClick ? __toolBoxObj._onIconClick.bind(__toolBoxObj) : null;
    if (__origToolboxClick) {{
      __toolBoxObj._onIconClick = (icon) => {{
        __resetCustom();
        __origToolboxClick(icon);
      }};
    }}

    __handler.chart.subscribeCrosshairMove((evt) => {{
      if (__activeMode && __startPoint && __previewSet) {{
        const pt = __pointFromEvent(evt);
        if (!pt) return;
        __previewSet.update(__startPoint, pt);
        return;
      }}
      __refreshOverlayGroups();
    }});

    __handler.chart.subscribeClick((evt) => {{
      if (!__activeMode) return;
      const pt = __pointFromEvent(evt);
      if (!pt) return;

      if (!__startPoint) {{
        __startPoint = __point(pt);
        const builder = __builders[__activeMode];
        if (!builder) {{
          __resetCustom();
          return;
        }}
        __previewSet = builder(__startPoint);
        __attachPreview(__previewSet);
        __previewSet.update(__startPoint, pt);
        return;
      }}

      const builder = __builders[__activeMode];
      if (!builder) {{
        __resetCustom();
        return;
      }}

      __detachPreview();
      const finalSet = builder(__startPoint);
      finalSet.update(__startPoint, pt);
      __maybeEditToolText(__activeMode, finalSet.group);
      __commitFinal(finalSet);
      if (__toolBoxObj.saveDrawings) {{
        __toolBoxObj.saveDrawings();
      }}
      __resetCustom();
    }});

    document.body.addEventListener('keydown', (evt) => {{
      if (evt.key === 'Escape' && __activeMode) {{
        evt.preventDefault();
        __resetCustom();
      }}
    }});
  }}
}}
"""
    )

    # ---------------------------------------------------------------------------
    # Topbar controls + apply function
    # ---------------------------------------------------------------------------

    IND_OPTS = ("없음", "볼린저", "이치모쿠", "둘 다")
    IND_MAP  = {"없음": "none", "볼린저": "bollinger", "이치모쿠": "ichimoku", "둘 다": "both"}
    IND_DEFAULT = {v: k for k, v in IND_MAP.items()}.get(indicator_mode, "없음")

    BAND_OPTS = ("없음", "PER", "PBR", "둘 다")
    BAND_MAP  = {"없음": "none", "PER": "per", "PBR": "pbr", "둘 다": "both"}
    BAND_DEFAULT = "없음"

    has_bands = (not eps_df.empty) or (not bps_df.empty)

    def _set_ichimoku_cloud(enabled: bool) -> None:
        nonlocal ichi_cloud_visible
        if enabled == ichi_cloud_visible:
            return
        ichi_cloud_visible = enabled
        vis = "true" if enabled else "false"
        chart.run_script(
            f"if (typeof {chart.id}._kumoSeries !== 'undefined') "
            f"{chart.id}._kumoSeries.applyOptions({{visible: {vis}}})"
        )

    def _apply_all(chart) -> None:
        """Read topbar state and show/hide indicator lines."""
        # Moving averages
        for key, line in ma_lines.items():
            w = chart.topbar[key]
            line.set(ma_dfs[key] if w.value else None)

        # Bollinger / Ichimoku
        ind = IND_MAP.get(chart.topbar["indicator"].value, "none")
        show_bb = ind in ("bollinger", "both")
        show_ichi = ind in ("ichimoku", "both")
        for lbl, line in bb_lines.items():
            line.set(bb_dfs[lbl] if show_bb else None)
        bb_vis = "true" if show_bb else "false"
        chart.run_script(
            f"if (typeof {chart.id}._bbFillSeries !== 'undefined') "
            f"{chart.id}._bbFillSeries.applyOptions({{visible: {bb_vis}}})"
        )
        for lbl, line in ichi_lines.items():
            line.set(ICHI_DEFS[lbl][1] if show_ichi else None)
        _set_ichimoku_cloud(show_ichi)

        # Valuation bands
        if has_bands:
            band = BAND_MAP.get(chart.topbar["band"].value, "none")
            for line, df in per_line_pairs:
                line.set(df if band in ("per", "both") else None)
            for line, df in pbr_line_pairs:
                line.set(df if band in ("pbr", "both") else None)

    def _on_per_levels_change(chart) -> None:
        nonlocal per_levels_current
        if eps_df.empty:
            return
        per_levels_current = _parse_levels_text(chart.topbar["per_levels"].value, per_levels_current)
        _set_textbox_value(chart.topbar["per_levels"], _levels_text(per_levels_current))
        _rebuild_per_lines()
        _apply_all(chart)

    def _on_pbr_levels_change(chart) -> None:
        nonlocal pbr_levels_current
        if bps_df.empty:
            return
        pbr_levels_current = _parse_levels_text(chart.topbar["pbr_levels"].value, pbr_levels_current)
        _set_textbox_value(chart.topbar["pbr_levels"], _levels_text(pbr_levels_current))
        _rebuild_pbr_lines()
        _apply_all(chart)

    # Indicator switcher
    chart.topbar.switcher("indicator", IND_OPTS, default=IND_DEFAULT, func=_apply_all)

    # MA toggle buttons
    for key, (label, _fn, _period, _color) in MA_META.items():
        if key in ma_lines:
            chart.topbar.button(key, label, separator=False, toggle=True, func=_apply_all)

    # Band switcher (only if valuation data is available)
    if has_bands:
        chart.topbar.switcher("band", BAND_OPTS, default=BAND_DEFAULT, func=_apply_all)
        if not eps_df.empty:
            chart.topbar.textbox("per_label", "PER", align="right")
            chart.topbar.textbox(
                "per_levels",
                _levels_text(per_levels_current),
                align="right",
                func=_on_per_levels_change,
            )
        if not bps_df.empty:
            chart.topbar.textbox("pbr_label", "PBR", align="right")
            chart.topbar.textbox(
                "pbr_levels",
                _levels_text(pbr_levels_current),
                align="right",
                func=_on_pbr_levels_change,
            )

    # Apply initial indicator / band state (queued before show)
    _apply_all(chart)

    chart.show(block=True)


if __name__ == "__main__":
    main()
