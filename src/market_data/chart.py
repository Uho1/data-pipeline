from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from market_data.plotting import draw_candles, draw_volume, prepare_ohlcv, style_axes
from market_data.reader import load_price_dataframe
from market_data.utils import ensure_dir, now_utc_iso


def _validate_price_column(df, price_col: str) -> str:
    if price_col in df.columns:
        return price_col
    if "Adj Close" in df.columns:
        return "Adj Close"
    if "Close" in df.columns:
        return "Close"
    raise ValueError(f"Requested column '{price_col}' not found and no fallback column is available")


def _autoscale_price(ax: plt.Axes, highs: np.ndarray, lows: np.ndarray) -> None:
    y_min = float(np.nanmin(lows))
    y_max = float(np.nanmax(highs))
    pad = max((y_max - y_min) * 0.04, 1e-8)
    ax.set_ylim(y_min - pad, y_max + pad)


def run_chart(
    ticker: str,
    market: str | None,
    price_col: str,
    chart_type: str,
    save_path: str | None,
    show: bool,
) -> int:
    df, source_path = load_price_dataframe(ticker=ticker, market=market)
    ohlc = prepare_ohlcv(df)
    col = _validate_price_column(ohlc.df, price_col)

    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    fig.subplots_adjust(hspace=0.04)

    if chart_type == "line":
        ax_price.plot(ohlc.df.index, ohlc.df[col], label=col, linewidth=1.3, color="#1f77b4")
        ax_price.grid(alpha=0.22)
        ax_price.legend(loc="upper left")
    else:
        draw_candles(ax_price, ohlc)
        _autoscale_price(
            ax_price,
            highs=ohlc.df["High"].to_numpy(dtype=float),
            lows=ohlc.df["Low"].to_numpy(dtype=float),
        )

    draw_volume(ax_vol, ohlc)
    style_axes(ax_price=ax_price, ax_vol=ax_vol, title=f"{ticker} | {chart_type.upper()}")
    ax_price.set_ylabel("Price", fontsize=6)
    ax_vol.set_ylabel("Volume", fontsize=6)
    ax_vol.set_xlabel("Date", fontsize=6)

    output_path: Path | None = None
    if save_path:
        output_path = Path(save_path)
    elif not show:
        output_path = Path("logs") / "charts" / f"{ticker}_{chart_type}_{col.replace(' ', '_')}.png"

    if output_path is not None:
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        print(f"[CHART] saved={output_path} source={source_path} rows={len(df)} at={now_utc_iso()}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return 0
