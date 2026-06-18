"""TradingView chart subprocess manager for market_data_lake.

Opens a TradingView Lightweight Charts window in a subprocess so the
pywebview/WKWebView event loop does not conflict with tkinter's mainloop.

Usage:
    from market_data.tv_chart import open_tv_chart, close_tv_chart, is_tv_chart_running

    open_tv_chart(df, "AAPL", "us", chart_type="candles", indicator_mode="bollinger")
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_RUNNER = Path(__file__).parent / "_tv_chart_runner.py"

_lock = threading.Lock()
_proc: subprocess.Popen | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def open_tv_chart(
    df: pd.DataFrame,
    ticker: str,
    market: str,
    chart_type: str = "candles",
    indicator_mode: str = "none",
    valuation: Any = None,          # ValuationResult or None
    per_levels: list[float] | None = None,
    pbr_levels: list[float] | None = None,
) -> None:
    """Serialize chart data and open the TradingView chart in a subprocess."""
    records = _df_to_records(df)
    val_payload = _build_valuation_payload(
        valuation=valuation,
        per_levels=per_levels,
        pbr_levels=pbr_levels,
    )

    config: dict[str, Any] = {
        "ticker": str(ticker).strip().upper(),
        "market": str(market).strip().upper(),
        "chart_type": chart_type,
        "indicator_mode": indicator_mode,
        "data": records,
        "valuation": val_payload,
    }

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="tv_chart_"
    )
    json.dump(config, tmp, default=str)
    tmp.close()

    _launch(tmp.name)


def close_tv_chart() -> None:
    """Terminate the TV chart subprocess if running."""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
        _proc = None


def is_tv_chart_running() -> bool:
    """Return True if the TV chart subprocess is currently running."""
    with _lock:
        return _proc is not None and _proc.poll() is None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _launch(config_path: str) -> None:
    """Kill any existing chart subprocess and start a new one."""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
        _proc = subprocess.Popen(
            [sys.executable, str(_RUNNER), config_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert OHLCV DataFrame to list of dicts for JSON serialization."""
    out = df.copy()

    if isinstance(out.index, pd.DatetimeIndex) and "Date" not in out.columns:
        out = out.reset_index()

    rename = {
        "Date": "time",
        "index": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], errors="coerce").dt.strftime("%Y-%m-%d")

    required = ["time", "open", "high", "low", "close"]
    out = out.dropna(subset=[c for c in required if c in out.columns])
    out = out.sort_values("time")

    cols = [c for c in ["time", "open", "high", "low", "close", "adj_close", "volume"]
            if c in out.columns]
    return out[cols].to_dict("records")


def _series_to_records(series: pd.Series | None) -> list[dict]:
    """Convert a DatetimeIndex-indexed Series to [{time, value}, ...]."""
    if series is None or series.empty:
        return []
    s = series.dropna()
    out = []
    for ts, val in s.items():
        try:
            fval = float(val)
            if not np.isfinite(fval):
                continue
            out.append({"time": str(ts.date()), "value": fval})
        except (TypeError, ValueError):
            continue
    return out


def _build_valuation_payload(
    valuation: Any,
    per_levels: list[float] | None,
    pbr_levels: list[float] | None,
) -> dict:
    """Build the valuation section of the chart config."""
    eps_records: list[dict] = []
    bps_records: list[dict] = []
    per_lvls: list[float] = []
    pbr_lvls: list[float] = []

    if valuation is not None:
        eps_records = _series_to_records(getattr(valuation, "eps_daily", None))
        bps_records = _series_to_records(getattr(valuation, "bps_daily", None))
        if per_levels is not None:
            per_lvls = [float(x) for x in per_levels if np.isfinite(float(x))]
        elif hasattr(valuation, "default_per_levels"):
            per_lvls = [float(x) for x in valuation.default_per_levels
                        if np.isfinite(float(x))]
        if pbr_levels is not None:
            pbr_lvls = [float(x) for x in pbr_levels if np.isfinite(float(x))]
        elif hasattr(valuation, "default_pbr_levels"):
            pbr_lvls = [float(x) for x in valuation.default_pbr_levels
                        if np.isfinite(float(x))]

    return {
        "eps": eps_records,
        "bps": bps_records,
        "per_levels": per_lvls,
        "pbr_levels": pbr_lvls,
    }
