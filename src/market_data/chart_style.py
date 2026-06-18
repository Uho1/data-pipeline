from __future__ import annotations

from matplotlib.figure import Figure


CARD_GAP_PX = 16
CARD_MIN_HEIGHT_PX = 500
CARD_MIN_WIDTH_FOR_TWO_COL_PX = 620
CARD_CHART_MIN_HEIGHT_PX = 380
FIG_DPI = 100


def get_layout_mode(window_width: int) -> str:
    width = max(int(window_width), 0)
    # two-col requires each card width >= 620 with 16px gap
    card_width = (width - CARD_GAP_PX) / 2.0
    return "two_col" if card_width >= CARD_MIN_WIDTH_FOR_TWO_COL_PX else "one_col"


def get_figure_size_px(mode: str) -> tuple[int, int]:
    if str(mode).strip().lower() == "one_col":
        return (1160, 480)
    return (740, 408)


def make_figure_for_card(mode: str) -> tuple[Figure, object]:
    w_px, h_px = get_figure_size_px(mode)
    h_px = max(h_px, CARD_MIN_HEIGHT_PX)
    fig = Figure(figsize=(w_px / FIG_DPI, h_px / FIG_DPI), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.08, right=0.96)
    return fig, ax


def apply_matplotlib_theme(plt) -> None:  # type: ignore[no-untyped-def]
    try:
        from matplotlib import font_manager

        available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available_fonts = set()

    preferred_kr_fonts = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "Arial Unicode MS",
    ]
    selected_font = next((name for name in preferred_kr_fonts if name in available_fonts), "DejaVu Sans")

    plt.rcParams["font.family"] = selected_font
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["path.simplify"] = True
    plt.rcParams["path.simplify_threshold"] = 1.0
    plt.rcParams["agg.path.chunksize"] = 20000
    plt.rcParams["figure.facecolor"] = "#f7f9fd"
    plt.rcParams["axes.facecolor"] = "#ffffff"
    plt.rcParams["axes.edgecolor"] = "#dbe2ec"
    plt.rcParams["axes.labelcolor"] = "#667085"
    plt.rcParams["axes.titlecolor"] = "#111827"
    plt.rcParams["xtick.color"] = "#667085"
    plt.rcParams["ytick.color"] = "#667085"
    plt.rcParams["grid.color"] = "#e7edf5"
    plt.rcParams["grid.alpha"] = 1.0
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["legend.fontsize"] = 10


def apply_chart_style(ax) -> None:  # type: ignore[no-untyped-def]
    ax.set_facecolor("#ffffff")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#dbe2ec")
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color="#e7edf5", linewidth=0.6, alpha=1.0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", labelsize=9, colors="#667085")


def apply_secondary_axis_style(ax) -> None:  # type: ignore[no-untyped-def]
    ax.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_color("#dbe2ec")
    ax.spines["right"].set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=9, colors="#667085")


def add_axis_unit_label(ax, text: str, side: str = "left") -> None:  # type: ignore[no-untyped-def]
    if not text:
        return
    if side == "right":
        ax.text(
            0.99,
            0.985,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#8a94a8",
        )
    else:
        ax.text(
            0.01,
            0.985,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#8a94a8",
        )


def format_legend(legend) -> None:  # type: ignore[no-untyped-def]
    if legend is None:
        return
    frame = legend.get_frame()
    if frame is not None:
        frame.set_facecolor("none")
        frame.set_edgecolor("none")
        frame.set_alpha(0.0)
    for text in legend.get_texts():
        text.set_color("#4b5563")
