from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Backgrounds
UI_BG       = "#f0f4f8"   # main window background (cool light gray)
PANEL_BG    = "#f7fafd"   # panel / sidebar background
SURFACE_BG  = "#ffffff"   # card / content surface

# Text
TEXT_MAIN   = "#0f172a"   # slate-900  (high contrast)
TEXT_SUB    = "#475569"   # slate-600  (secondary)
TEXT_MUTED  = "#94a3b8"   # slate-400  (muted / placeholder)

# Borders
BORDER        = "#dde5f0"   # standard border
BORDER_STRONG = "#bcc9de"   # emphasis / focused border

# ---------------------------------------------------------------------------
# Theme palettes
# ---------------------------------------------------------------------------

PALETTE_LIGHT: dict[str, str] = {
    "UI_BG":        "#f0f4f8",
    "PANEL_BG":     "#f7fafd",
    "SURFACE_BG":   "#ffffff",
    "TEXT_MAIN":    "#0f172a",
    "TEXT_SUB":     "#475569",
    "TEXT_MUTED":   "#94a3b8",
    "BORDER":       "#dde5f0",
    "BORDER_STRONG":"#bcc9de",
    "FIELD_BG":     "#ffffff",
    "SEG_INACTIVE": "#dde7f5",
    "SEG_ACTIVE":   "#ffffff",
    "SEG_FRAME":    "#f0f4f8",
}

PALETTE_DARK: dict[str, str] = {
    "UI_BG":        "#0f172a",
    "PANEL_BG":     "#1e293b",
    "SURFACE_BG":   "#1e293b",
    "TEXT_MAIN":    "#f1f5f9",
    "TEXT_SUB":     "#94a3b8",
    "TEXT_MUTED":   "#64748b",
    "BORDER":       "#334155",
    "BORDER_STRONG":"#475569",
    "FIELD_BG":     "#0f172a",
    "SEG_INACTIVE": "#1e3a5f",
    "SEG_ACTIVE":   "#2563eb",
    "SEG_FRAME":    "#0f172a",
}


def get_palette(theme: str) -> dict[str, str]:
    return PALETTE_DARK if theme == "dark" else PALETTE_LIGHT

# Brand accent — teal
ACCENT        = "#0f766e"
ACCENT_DARK   = "#0a5f57"
ACCENT_LIGHT  = "#ccf9f4"

# Primary action — blue
PRIMARY       = "#2563eb"
PRIMARY_DARK  = "#1d4ed8"
PRIMARY_LIGHT = "#dbeafe"

# Status
DANGER        = "#dc2626"
DANGER_DARK   = "#b91c1c"
DANGER_LIGHT  = "#fee2e2"
SUCCESS       = "#16a34a"
SUCCESS_DARK  = "#15803d"
SUCCESS_LIGHT = "#dcfce7"
WARNING       = "#d97706"
WARNING_LIGHT = "#fef3c7"

# Input fields
FIELD_BG      = "#ffffff"

# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

def _pick_ui_font_family(root: tk.Tk) -> str:
    try:
        families = set(tkfont.families(root))
    except Exception:
        families = set()
    preferred = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
    ]
    for name in preferred:
        if name in families:
            return name
    return "Helvetica"


# ---------------------------------------------------------------------------
# Main style configuration
# ---------------------------------------------------------------------------

def configure_ttk_style(root: tk.Misc, ttk, theme: str = "light") -> str:  # type: ignore[no-untyped-def]
    """Apply comprehensive ttk styles.  Returns the resolved font family."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    ff = _pick_ui_font_family(root)  # font family
    _apply_ttk_palette(style, root, ff, get_palette(theme))

    fspec = f"{{{ff}}} 12" if " " in ff else f"{ff} 12"
    root.option_add("*Font", fspec)
    root.option_add("*TCombobox*Listbox.font", fspec)

    # ------------------------------------------------------------------
    _apply_ttk_palette(style, root, ff, get_palette(theme))
    return ff


def _apply_ttk_palette(style, root: tk.Misc, ff: str, p: dict[str, str]) -> None:  # type: ignore[no-untyped-def]
    """Reconfigure all ttk styles using the given palette dict ``p``."""
    ui_bg        = p["UI_BG"]
    panel_bg     = p["PANEL_BG"]
    surface_bg   = p["SURFACE_BG"]
    text_main    = p["TEXT_MAIN"]
    text_sub     = p["TEXT_SUB"]
    text_muted   = p["TEXT_MUTED"]
    border       = p["BORDER"]
    border_strong= p["BORDER_STRONG"]
    field_bg     = p["FIELD_BG"]

    root.configure(bg=ui_bg)

    # ------------------------------------------------------------------
    # Base / global
    # ------------------------------------------------------------------
    style.configure(".", background=ui_bg, foreground=text_main,
                    font=(ff, 12), relief="flat")

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    style.configure("TFrame",         background=ui_bg)
    style.configure("Panel.TFrame",   background=panel_bg)
    style.configure("Surface.TFrame", background=surface_bg)
    style.configure(
        "Card.TFrame",
        background=surface_bg,
        relief="solid",
        borderwidth=1,
        bordercolor=border,
    )

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    style.configure("TLabel",              background=ui_bg,      foreground=text_main, font=(ff, 12))
    style.configure("Dim.TLabel",          background=ui_bg,      foreground=text_sub,  font=(ff, 11))
    style.configure("Info.TLabel",         background=ui_bg,      foreground=text_muted,font=(ff, 11))
    style.configure("Muted.TLabel",        background=ui_bg,      foreground=text_muted,font=(ff, 10))
    style.configure("SectionTitle.TLabel", background=ui_bg,      foreground=text_main, font=(ff, 16, "bold"))
    style.configure("Header.TLabel",       background=ui_bg,      foreground=text_main, font=(ff, 13, "bold"))
    style.configure("SubHeader.TLabel",    background=ui_bg,      foreground=text_sub,  font=(ff, 11, "bold"))

    style.configure("Success.TLabel", background=SUCCESS_LIGHT, foreground=SUCCESS_DARK, font=(ff, 11, "bold"), padding=(6, 3))
    style.configure("Danger.TLabel",  background=DANGER_LIGHT,  foreground=DANGER_DARK,  font=(ff, 11, "bold"), padding=(6, 3))
    style.configure("Warning.TLabel", background=WARNING_LIGHT, foreground=WARNING,       font=(ff, 11, "bold"), padding=(6, 3))
    style.configure("Accent.TLabel",  background=ACCENT_LIGHT,  foreground=ACCENT_DARK,  font=(ff, 11, "bold"), padding=(6, 3))
    style.configure("Primary.TLabel", background=PRIMARY_LIGHT, foreground=PRIMARY_DARK, font=(ff, 11, "bold"), padding=(6, 3))

    style.configure("KPI.TLabel",     background=surface_bg, foreground=text_main,
                    font=(ff, 12, "bold"), relief="solid", borderwidth=1, bordercolor=border, padding=(10, 6))
    style.configure("KPI.Pos.TLabel", background=SUCCESS_LIGHT, foreground=SUCCESS_DARK,
                    font=(ff, 12, "bold"), relief="solid", borderwidth=1, bordercolor=SUCCESS, padding=(10, 6))
    style.configure("KPI.Neg.TLabel", background=DANGER_LIGHT,  foreground=DANGER_DARK,
                    font=(ff, 12, "bold"), relief="solid", borderwidth=1, bordercolor=DANGER,  padding=(10, 6))

    style.configure("AppHeader.TLabel",    background=PRIMARY,  foreground="#ffffff",  font=(ff, 15, "bold"), padding=(16, 10))
    style.configure("AppSubHeader.TLabel", background=PRIMARY,  foreground="#c7d9ff",  font=(ff, 10),         padding=(0, 0, 16, 10))
    style.configure("StatusBar.TLabel",    background=border,   foreground=text_sub,   font=(ff, 10),         padding=(6, 3))

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------
    _btn_bg      = "#1e3a5f" if ui_bg == PALETTE_DARK["UI_BG"] else "#e8eef8"
    _btn_bg_act  = "#2a4f7f" if ui_bg == PALETTE_DARK["UI_BG"] else "#d8e4f4"
    _btn_bg_prs  = "#163055" if ui_bg == PALETTE_DARK["UI_BG"] else "#cad9ef"
    _btn_bg_dis  = "#1a2f4f" if ui_bg == PALETTE_DARK["UI_BG"] else "#eaeff7"

    style.configure("Ghost.TButton",  background=ui_bg, foreground=text_sub,
                    bordercolor=ui_bg, padding=(8, 5), relief="flat", font=(ff, 11))
    style.map("Ghost.TButton",
              background=[("active", border), ("pressed", border_strong)],
              foreground=[("active", text_main)])

    style.configure("TButton", background=_btn_bg, foreground=text_main,
                    bordercolor=border, padding=(12, 7), relief="flat", font=(ff, 12))
    style.map("TButton",
              background=[("active", _btn_bg_act), ("pressed", _btn_bg_prs), ("disabled", _btn_bg_dis)],
              foreground=[("disabled", text_muted)])

    style.configure("Small.TButton", background=_btn_bg, foreground=text_main,
                    bordercolor=border, padding=(7, 4), relief="flat", font=(ff, 11))
    style.map("Small.TButton",
              background=[("active", _btn_bg_act), ("pressed", _btn_bg_prs)])

    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                    bordercolor=ACCENT, padding=(12, 7), relief="flat", font=(ff, 12, "bold"))
    style.map("Accent.TButton",
              background=[("active", ACCENT_DARK), ("pressed", ACCENT_DARK), ("disabled", "#5fbdb6")],
              foreground=[("disabled", "#d0efed")])

    style.configure("Primary.TButton", background=PRIMARY, foreground="#ffffff",
                    bordercolor=PRIMARY, padding=(12, 7), relief="flat", font=(ff, 12, "bold"))
    style.map("Primary.TButton",
              background=[("active", PRIMARY_DARK), ("pressed", PRIMARY_DARK), ("disabled", "#93b4f3")],
              foreground=[("disabled", "#dde9ff")])

    style.configure("Run.TButton", background=PRIMARY, foreground="#ffffff",
                    bordercolor=PRIMARY, padding=(18, 10), relief="flat", font=(ff, 13, "bold"))
    style.map("Run.TButton",
              background=[("active", PRIMARY_DARK), ("pressed", PRIMARY_DARK), ("disabled", "#93b4f3")],
              foreground=[("disabled", "#dde9ff")])

    style.configure("Danger.TButton", background=DANGER, foreground="#ffffff",
                    bordercolor=DANGER, padding=(12, 7), relief="flat", font=(ff, 12))
    style.map("Danger.TButton",
              background=[("active", DANGER_DARK), ("pressed", DANGER_DARK), ("disabled", "#f09090")],
              foreground=[("disabled", "#fdd")])

    # ------------------------------------------------------------------
    # Checkbutton & Radiobutton
    # ------------------------------------------------------------------
    style.configure("TCheckbutton", background=ui_bg, foreground=text_sub,  font=(ff, 12))
    style.configure("TRadiobutton", background=ui_bg, foreground=text_sub,  font=(ff, 12))
    style.map("TCheckbutton",
              background=[("active", ui_bg)],
              indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT_DARK)])
    style.map("TRadiobutton",
              background=[("active", ui_bg)],
              indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT_DARK)])

    # ------------------------------------------------------------------
    # Entry & Combobox
    # ------------------------------------------------------------------
    style.configure("TEntry",
                    fieldbackground=field_bg, bordercolor=border,
                    lightcolor=border, darkcolor=border,
                    selectbackground=PRIMARY_LIGHT, selectforeground=PRIMARY_DARK,
                    foreground=text_main, padding=(8, 5))
    style.map("TEntry",
              bordercolor=[("focus", ACCENT), ("invalid", DANGER)],
              lightcolor=[("focus", ACCENT), ("invalid", DANGER)],
              darkcolor=[("focus", ACCENT), ("invalid", DANGER)])
    style.configure("Invalid.TEntry",
                    fieldbackground="#fff0f0", bordercolor=DANGER,
                    lightcolor=DANGER, darkcolor=DANGER, padding=(8, 5))

    style.configure("TCombobox",
                    fieldbackground=field_bg, bordercolor=border,
                    lightcolor=border, darkcolor=border,
                    selectbackground=PRIMARY_LIGHT, selectforeground=PRIMARY_DARK,
                    foreground=text_main, arrowsize=14, padding=(8, 4))
    style.map("TCombobox",
              bordercolor=[("focus", ACCENT)],
              lightcolor=[("focus", ACCENT)],
              darkcolor=[("focus", ACCENT)])

    # ------------------------------------------------------------------
    # Notebook
    # ------------------------------------------------------------------
    _tab_bg     = "#1e3a5f" if ui_bg == PALETTE_DARK["UI_BG"] else "#dde7f5"
    _tab_bg_act = "#2a4f7f" if ui_bg == PALETTE_DARK["UI_BG"] else "#e8f0fa"
    style.configure("TNotebook", background=ui_bg, borderwidth=0, tabmargins=[0, 0, 0, 0])
    style.configure("TNotebook.Tab",
                    background=_tab_bg, foreground=text_sub,
                    padding=(20, 10), borderwidth=0, font=(ff, 12, "bold"))
    style.map("TNotebook.Tab",
              background=[("selected", surface_bg), ("active", _tab_bg_act)],
              foreground=[("selected", PRIMARY), ("active", text_main)],
              expand=[("selected", [0, 0, 0, 2])])

    # ------------------------------------------------------------------
    # LabelFrame
    # ------------------------------------------------------------------
    style.configure("TLabelframe",
                    background=surface_bg, bordercolor=border_strong,
                    borderwidth=1, relief="solid", padding=(10, 6))
    style.configure("TLabelframe.Label",
                    background=surface_bg, foreground=ACCENT,
                    font=(ff, 11, "bold"), padding=(4, 0))

    # ------------------------------------------------------------------
    # Treeview
    # ------------------------------------------------------------------
    _tv_head_bg = "#1e3a5f" if ui_bg == PALETTE_DARK["UI_BG"] else "#e8eef8"
    style.configure("Treeview",
                    background=surface_bg, foreground=text_main,
                    bordercolor=border, lightcolor=border, darkcolor=border,
                    rowheight=30, fieldbackground=surface_bg, font=(ff, 11))
    style.configure("Treeview.Heading",
                    background=_tv_head_bg, foreground=text_sub,
                    relief="flat", font=(ff, 11, "bold"), padding=(4, 4))
    style.map("Treeview",
              background=[("selected", PRIMARY_LIGHT)],
              foreground=[("selected", PRIMARY_DARK)])
    style.map("Treeview.Heading",
              background=[("active", _tab_bg_act)])

    # ------------------------------------------------------------------
    # Scrollbar / Separator / Progressbar / Scale / PanedWindow
    # ------------------------------------------------------------------
    style.configure("TScrollbar",
                    background=border, troughcolor=ui_bg, bordercolor=ui_bg,
                    arrowcolor=text_muted, relief="flat", width=10)
    style.map("TScrollbar",
              background=[("active", border_strong), ("pressed", text_muted)])
    style.configure("TSeparator", background=border)
    style.configure("TProgressbar",
                    troughcolor=border, background=ACCENT, bordercolor=border,
                    lightcolor=ACCENT, darkcolor=ACCENT_DARK, thickness=6)
    style.configure("Primary.TProgressbar",
                    background=PRIMARY, lightcolor=PRIMARY, darkcolor=PRIMARY_DARK)
    style.configure("TScale", background=ui_bg, troughcolor=border)
    style.configure("TPanedwindow", background=ui_bg)
    style.configure("Sash", sashthickness=4, sashpad=2)


def apply_theme(style, root: tk.Misc, theme: str, ff: str,
                tk_widgets: list[tuple[tk.Widget, str]] | None = None) -> dict[str, str]:
    """Switch the app to *theme* ('light' or 'dark').

    Reconfigures all ttk styles and optionally updates non-ttk widgets.
    ``tk_widgets`` is a list of ``(widget, palette_key)`` pairs, e.g.
    ``[(my_label, "UI_BG")]``.  The widget's ``bg`` is set to the palette
    colour for that key.

    Returns the palette dict for the requested theme.
    """
    p = get_palette(theme)
    _apply_ttk_palette(style, root, ff, p)
    if tk_widgets:
        for widget, key in tk_widgets:
            try:
                widget.configure(bg=p[key])
            except Exception:
                pass
    return p


# ---------------------------------------------------------------------------
# Segmented control (styled radio-button row)
# ---------------------------------------------------------------------------

def create_segmented_control(
    parent: tk.Widget,
    variable: tk.Variable,
    options: list[tuple[str, str]],
    on_change=None,
    font_family: str = "Helvetica",
    palette: dict[str, str] | None = None,
) -> tk.Frame:
    """Return a frame containing styled radio-buttons that look like a
    segmented control (pill / tab-bar style)."""
    p = palette or PALETTE_LIGHT
    frame_bg    = p.get("SEG_FRAME", p["UI_BG"])
    inactive_bg = p.get("SEG_INACTIVE", "#dde7f5")
    active_bg   = p.get("SEG_ACTIVE", p["SURFACE_BG"])
    inactive_fg = p["TEXT_SUB"]
    active_fg   = PRIMARY
    hover_bg    = p["BORDER"]

    frame = tk.Frame(parent, bg=frame_bg, highlightthickness=1,
                     highlightbackground=p["BORDER"], highlightcolor=ACCENT, bd=0)

    def _sync() -> None:
        selected = str(variable.get())
        for value, btn in buttons.items():
            is_sel = value == selected
            btn.configure(
                bg=active_bg   if is_sel else inactive_bg,
                fg=active_fg   if is_sel else inactive_fg,
                activebackground=active_bg if is_sel else hover_bg,
                activeforeground=active_fg if is_sel else p["TEXT_MAIN"],
                relief="solid" if is_sel else "flat",
                bd=1 if is_sel else 0,
            )

    def _on_press() -> None:
        _sync()
        if on_change is not None:
            on_change()

    buttons: dict[str, tk.Radiobutton] = {}
    for i, (label, value) in enumerate(options):
        btn = tk.Radiobutton(
            frame,
            text=label,
            value=value,
            variable=variable,
            indicatoron=0,
            relief="flat",
            borderwidth=0,
            padx=16,
            pady=7,
            font=(font_family, 11, "bold"),
            command=_on_press,
            cursor="hand2",
        )
        btn.pack(side="left", padx=0)
        buttons[value] = btn

    try:
        variable.trace_add("write", lambda *_: _sync())
    except Exception:
        pass
    _sync()
    return frame


def create_header_toggle(
    parent: tk.Widget,
    variable: tk.Variable,
    options: list[tuple[str, str]],
    font_family: str = "Helvetica",
    on_change=None,
) -> tk.Frame:
    """Segmented control styled for the app header bar (blue background)."""
    frame = tk.Frame(parent, bg=PRIMARY, highlightthickness=0, bd=0)

    active_bg   = "#ffffff"
    active_fg   = PRIMARY
    inactive_bg = "#3b74f0"
    inactive_fg = "#d0e4ff"
    hover_bg    = "#4d82f7"

    def _sync() -> None:
        selected = str(variable.get())
        for value, btn in buttons.items():
            is_sel = value == selected
            btn.configure(
                bg=active_bg   if is_sel else inactive_bg,
                fg=active_fg   if is_sel else inactive_fg,
                activebackground=active_bg if is_sel else hover_bg,
                activeforeground=active_fg if is_sel else "#ffffff",
                relief="solid" if is_sel else "flat",
                bd=1 if is_sel else 0,
            )

    def _on_press() -> None:
        _sync()
        if on_change is not None:
            on_change()

    buttons: dict[str, tk.Radiobutton] = {}
    for _i, (label, value) in enumerate(options):
        btn = tk.Radiobutton(
            frame,
            text=label,
            value=value,
            variable=variable,
            indicatoron=0,
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=6,
            font=(font_family, 11, "bold"),
            command=_on_press,
            cursor="hand2",
        )
        btn.pack(side="left", padx=0)
        buttons[value] = btn

    try:
        variable.trace_add("write", lambda *_: _sync())
    except Exception:
        pass
    _sync()
    return frame


# ---------------------------------------------------------------------------
# Convenience: apply Treeview tag colors (profit / loss rows)
# ---------------------------------------------------------------------------

def configure_treeview_tags(tree) -> None:  # type: ignore[no-untyped-def]
    """Apply standard row-tag colors to a Treeview widget."""
    tree.tag_configure("pos",   background=SUCCESS_LIGHT, foreground=SUCCESS_DARK)
    tree.tag_configure("neg",   background=DANGER_LIGHT,  foreground=DANGER_DARK)
    tree.tag_configure("even",  background=SURFACE_BG)
    tree.tag_configure("odd",   background="#f5f8fd")
    tree.tag_configure("bold",  font=("", 0, "bold"))
    tree.tag_configure("muted", foreground=TEXT_MUTED)
