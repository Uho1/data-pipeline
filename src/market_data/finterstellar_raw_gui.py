from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from market_data.ingest import DEFAULT_FINANCIAL_WORKERS, IngestOptions, ingest_data
from market_data.price_refresher import PriceUpdateOptions, run_price_update
from market_data.ui_style import (
    BORDER,
    PRIMARY,
    SURFACE_BG,
    TEXT_MUTED,
    TEXT_SUB,
    UI_BG,
    configure_ttk_style,
    configure_treeview_tags,
)

_DEFAULT_SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
_DEFAULT_UNIVERSE_CSV = str(Path("data") / "universe" / "symbols_nasdaq_stock_only.csv")


class RecentDataUpdaterGUI:
    def __init__(self, root: tk.Tk, font_family: str = "Helvetica") -> None:
        self.root = root
        self.ff = font_family
        self.root.title("데이터 업데이터 — Prices & SEC Financials")
        self.root.geometry("1020x720")
        self.root.minsize(880, 600)
        self.root.configure(bg=UI_BG)

        self.log_q: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        # ── Price section vars ──
        self.price_market_var = tk.StringVar(value="us")
        self.price_tickers_var = tk.StringVar(value="")
        self.price_start_var = tk.StringVar(value="2000-01-01")
        self.price_interval_var = tk.StringVar(value="1d")
        self.price_force_var = tk.BooleanVar(value=False)

        # ── SEC Financials section vars ──
        self.fin_universe_var = tk.StringVar(value="direct")
        self.fin_tickers_file_var = tk.StringVar(value=_DEFAULT_UNIVERSE_CSV)
        self.fin_tickers_direct_var = tk.StringVar(value="")
        self.fin_start_var = tk.StringVar(value="2000-01-01")
        self.fin_user_agent_var = tk.StringVar(value=_DEFAULT_SEC_USER_AGENT)
        self.fin_workers_var = tk.IntVar(value=2)
        self.fin_force_var = tk.BooleanVar(value=False)
        self.fin_next_trading_day_var = tk.BooleanVar(value=True)
        self.fin_skip_sector_var = tk.BooleanVar(value=False)

        self._build()
        self.root.after(100, self._poll_logs)

    # ──────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────
    def _build(self) -> None:
        # Header bar
        hdr = tk.Frame(self.root, bg=PRIMARY, height=48)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  데이터 업데이터", bg=PRIMARY, fg="#ffffff",
                 font=(self.ff, 14, "bold"), anchor="w").pack(side="left", fill="y", padx=(8, 0))
        tk.Label(hdr, text="가격 데이터 · SEC 재무제표 · 섹터 분류", bg=PRIMARY, fg="#c7d9ff",
                 font=(self.ff, 10), anchor="w").pack(side="left", fill="y", padx=(10, 0))

        # Scrollable content
        outer = ttk.Frame(self.root, style="TFrame")
        outer.pack(fill="both", expand=True)

        self._build_price_section(outer)
        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=14, pady=8)
        self._build_financials_section(outer)
        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=14, pady=8)
        self._build_log_section(outer)

    def _row_label(self, parent: ttk.Frame, text: str, row: int) -> ttk.Label:
        lbl = ttk.Label(parent, text=text, style="Dim.TLabel", width=18, anchor="w")
        lbl.grid(row=row, column=0, padx=(0, 8), pady=4, sticky="w")
        return lbl

    def _build_price_section(self, parent: ttk.Frame) -> None:
        frm = ttk.LabelFrame(parent, text="가격 데이터 업데이트", padding=(14, 10))
        frm.pack(fill="x", padx=14, pady=(10, 0))
        frm.columnconfigure(1, weight=1)

        # Market
        self._row_label(frm, "마켓", 0)
        mkt_combo = ttk.Combobox(frm, textvariable=self.price_market_var,
                                  values=["us", "kr"], state="readonly", width=8)
        mkt_combo.grid(row=0, column=1, padx=(0, 8), pady=4, sticky="w")

        # Tickers
        self._row_label(frm, "티커", 1)
        ttk.Entry(frm, textvariable=self.price_tickers_var, width=50).grid(
            row=1, column=1, columnspan=2, padx=(0, 8), pady=4, sticky="ew")
        ttk.Label(frm, text="쉼표 구분, 공백 = 전체", style="Muted.TLabel").grid(
            row=1, column=3, padx=4, sticky="w")

        # Start date / Interval
        self._row_label(frm, "시작일", 2)
        date_row = ttk.Frame(frm)
        date_row.grid(row=2, column=1, columnspan=3, sticky="w", pady=4)
        ttk.Entry(date_row, textvariable=self.price_start_var, width=14).pack(side="left")
        ttk.Label(date_row, text="  인터벌", style="Dim.TLabel").pack(side="left")
        ttk.Entry(date_row, textvariable=self.price_interval_var, width=6).pack(side="left", padx=(6, 0))

        # Button row
        btn_row = ttk.Frame(frm)
        btn_row.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 2))
        ttk.Checkbutton(btn_row, text="강제 전체 재빌드", variable=self.price_force_var).pack(side="left", padx=(0, 12))
        self.price_btn = ttk.Button(btn_row, text="가격 데이터 업데이트",
                                     command=self._start_price_update, style="Primary.TButton")
        self.price_btn.pack(side="left")

    def _build_financials_section(self, parent: ttk.Frame) -> None:
        frm = ttk.LabelFrame(parent, text="SEC 재무제표 + 섹터 분류", padding=(14, 10))
        frm.pack(fill="x", padx=14, pady=0)
        frm.columnconfigure(1, weight=1)

        # Universe
        self._row_label(frm, "유니버스", 0)
        radio_frm = ttk.Frame(frm)
        radio_frm.grid(row=0, column=1, columnspan=3, sticky="w", pady=4)
        ttk.Radiobutton(radio_frm, text="local (기존 parquet 기준)",
                         variable=self.fin_universe_var, value="local",
                         command=self._on_universe_change).pack(side="left")
        ttk.Radiobutton(radio_frm, text="custom (파일 지정)",
                         variable=self.fin_universe_var, value="custom",
                         command=self._on_universe_change).pack(side="left", padx=14)
        ttk.Radiobutton(radio_frm, text="직접 입력",
                         variable=self.fin_universe_var, value="direct",
                         command=self._on_universe_change).pack(side="left")

        # Tickers file (custom)
        self.fin_file_label = self._row_label(frm, "티커 파일", 1)
        self.fin_tickers_file_entry = ttk.Entry(frm, textvariable=self.fin_tickers_file_var, width=50)
        self.fin_tickers_file_entry.grid(row=1, column=1, columnspan=2, padx=(0, 8), pady=4, sticky="ew")
        self.fin_browse_btn = ttk.Button(frm, text="찾아보기",
                                          command=self._browse_tickers_file, style="Small.TButton")
        self.fin_browse_btn.grid(row=1, column=3, padx=4, pady=4, sticky="w")

        # Direct ticker input
        self.fin_direct_label = self._row_label(frm, "티커 직접 입력", 2)
        self.fin_tickers_direct_entry = ttk.Entry(frm, textvariable=self.fin_tickers_direct_var, width=50)
        self.fin_tickers_direct_entry.grid(row=2, column=1, columnspan=2, padx=(0, 8), pady=4, sticky="ew")
        self.fin_direct_hint = ttk.Label(frm, text="예: AAPL, MSFT, NVDA", style="Muted.TLabel")
        self.fin_direct_hint.grid(row=2, column=3, padx=4, sticky="w")

        # Start date
        self._row_label(frm, "시작일", 3)
        ttk.Entry(frm, textvariable=self.fin_start_var, width=14).grid(
            row=3, column=1, padx=(0, 8), pady=4, sticky="w")

        # SEC User-Agent
        self._row_label(frm, "SEC User-Agent", 4)
        ttk.Entry(frm, textvariable=self.fin_user_agent_var, width=50).grid(
            row=4, column=1, columnspan=2, padx=(0, 8), pady=4, sticky="ew")
        ttk.Label(frm, text="예: 이름 이메일@example.com", style="Muted.TLabel").grid(
            row=4, column=3, padx=4, sticky="w")

        # Workers
        self._row_label(frm, "병렬 워커 수", 5)
        spn_frm = ttk.Frame(frm)
        spn_frm.grid(row=5, column=1, columnspan=3, sticky="w", pady=4)
        ttk.Spinbox(spn_frm, from_=1, to=8, textvariable=self.fin_workers_var,
                     width=5).pack(side="left")
        ttk.Label(spn_frm, text="  (SEC rate limit 고려 권장: 2~4)",
                   style="Muted.TLabel").pack(side="left", padx=(8, 0))

        # Checkboxes
        chk_frm = ttk.Frame(frm)
        chk_frm.grid(row=6, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Checkbutton(chk_frm, text="강제 재다운로드", variable=self.fin_force_var).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(chk_frm, text="다음 거래일 기준 AvailableDate",
                         variable=self.fin_next_trading_day_var).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(chk_frm, text="섹터 분류 스킵", variable=self.fin_skip_sector_var).pack(side="left")

        # Button
        btn_row = ttk.Frame(frm)
        btn_row.grid(row=7, column=0, columnspan=4, sticky="w", pady=(10, 4))
        self.fin_btn = ttk.Button(btn_row, text="재무제표 + 섹터 업데이트",
                                   command=self._start_financial_update, style="Accent.TButton")
        self.fin_btn.pack(side="left")

        frm.columnconfigure(1, weight=1)
        self._on_universe_change()

    def _build_log_section(self, parent: ttk.Frame) -> None:
        log_frm = ttk.LabelFrame(parent, text="실행 로그", padding=(10, 6))
        log_frm.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        ctrl = ttk.Frame(log_frm)
        ctrl.pack(fill="x", pady=(0, 6))
        ttk.Button(ctrl, text="로그 지우기", command=self._clear_log,
                    style="Ghost.TButton").pack(side="left")

        self.term = ScrolledText(
            log_frm, wrap="word",
            font=(self.ff, 11),
            bg="#1e2837", fg="#d4e0f0",
            insertbackground="#d4e0f0",
            selectbackground="#2563eb",
            relief="flat", borderwidth=0,
        )
        self.term.pack(fill="both", expand=True)
        self.term.configure(state="disabled")
        self._append("[READY] 버튼을 눌러 업데이트를 시작하세요.")

    # ──────────────────────────────────────────
    # UI helpers
    # ──────────────────────────────────────────
    def _on_universe_change(self) -> None:
        mode = self.fin_universe_var.get()
        file_state   = "normal" if mode == "custom" else "disabled"
        direct_state = "normal" if mode == "direct" else "disabled"
        self.fin_tickers_file_entry.configure(state=file_state)
        self.fin_browse_btn.configure(state=file_state)
        self.fin_tickers_direct_entry.configure(state=direct_state)
        self.fin_file_label.configure(
            foreground=TEXT_SUB if mode == "custom" else TEXT_MUTED)
        self.fin_direct_label.configure(
            foreground=TEXT_SUB if mode == "direct" else TEXT_MUTED)

    def _browse_tickers_file(self) -> None:
        path = filedialog.askopenfilename(
            title="티커 CSV 파일 선택",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(Path("data/universe").resolve()),
        )
        if path:
            self.fin_tickers_file_var.set(path)

    def _append(self, text: str) -> None:
        self.term.configure(state="normal")
        self.term.insert("end", text.rstrip() + "\n")
        self.term.see("end")
        self.term.configure(state="disabled")

    def _clear_log(self) -> None:
        self.term.configure(state="normal")
        self.term.delete("1.0", "end")
        self.term.configure(state="disabled")

    def _poll_logs(self) -> None:
        try:
            while True:
                msg = self.log_q.get_nowait()
                if msg == "__DONE__":
                    self.price_btn.configure(state="normal")
                    self.fin_btn.configure(state="normal")
                    continue
                self._append(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_logs)

    def _log(self, msg: str) -> None:
        self.log_q.put(msg)

    def _disable_buttons(self) -> None:
        self.price_btn.configure(state="disabled")
        self.fin_btn.configure(state="disabled")

    # ──────────────────────────────────────────
    # Price update
    # ──────────────────────────────────────────
    def _start_price_update(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("실행 중", "이미 작업이 진행 중입니다.")
            return

        raw_tickers = self.price_tickers_var.get().strip()
        tickers = [t.strip().upper() for t in raw_tickers.replace(";", ",").split(",") if t.strip()]
        opts = PriceUpdateOptions(
            market=self.price_market_var.get().strip().lower() or "us",
            tickers=tickers or None,
            start_default=self.price_start_var.get().strip() or "2000-01-01",
            interval=self.price_interval_var.get().strip() or "1d",
            force_full=bool(self.price_force_var.get()),
        )
        self._disable_buttons()
        self._append(f"[START] 가격 업데이트  market={opts.market}  tickers={'ALL' if not tickers else len(tickers)}")

        def worker() -> None:
            try:
                import sys
                import io

                class _Redirector(io.TextIOBase):
                    def __init__(self, cb):
                        self._cb = cb
                    def write(self, s):
                        if s.strip():
                            self._cb(s.rstrip())
                        return len(s)

                old_stdout = sys.stdout
                sys.stdout = _Redirector(self._log)
                try:
                    code = run_price_update(opts, log_cb=self._log)
                finally:
                    sys.stdout = old_stdout
                self._log(f"[DONE] 가격 업데이트 완료 exit={code}")
            except Exception as exc:  # noqa: BLE001
                self._log(f"[ERROR] {exc}")
            finally:
                self.log_q.put("__DONE__")

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    # ──────────────────────────────────────────
    # SEC Financial + Sector update
    # ──────────────────────────────────────────
    def _start_financial_update(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("실행 중", "이미 작업이 진행 중입니다.")
            return

        user_agent = self.fin_user_agent_var.get().strip()
        if not user_agent:
            messagebox.showerror("User-Agent 필요", "SEC User-Agent를 입력하세요.\n예: YourName your@email.com")
            return

        universe = self.fin_universe_var.get()
        tickers_file: str | None = None

        if universe == "custom":
            tickers_file = self.fin_tickers_file_var.get().strip()
            if not tickers_file or not Path(tickers_file).exists():
                messagebox.showerror("파일 없음", f"티커 파일을 찾을 수 없습니다:\n{tickers_file}")
                return
        elif universe == "direct":
            raw = self.fin_tickers_direct_var.get().strip()
            tickers = [t.strip().upper() for t in raw.replace(";", ",").split(",") if t.strip()]
            if not tickers:
                messagebox.showerror("티커 없음", "티커를 입력하세요. 예: AAPL, MSFT, NVDA")
                return
            import tempfile
            import csv as _csv
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False,
                dir="data/universe", prefix="_direct_input_",
            )
            writer = _csv.writer(tmp)
            writer.writerow(["Symbol"])
            for t in tickers:
                writer.writerow([t])
            tmp.close()
            tickers_file = tmp.name
            universe = "custom"

        workers = max(1, int(self.fin_workers_var.get()))
        opts = IngestOptions(
            universe=universe,
            start=self.fin_start_var.get().strip() or "2000-01-01",
            end=None,
            interval="1d",
            tickers_file=tickers_file,
            kospi_external_url="",
            kospi_top_n=None,
            fresh_days=7,
            retries=3,
            backoff_base=1.0,
            financial_workers=DEFAULT_FINANCIAL_WORKERS,
            workers=workers,
            force=bool(self.fin_force_var.get()),
            price_batch_size=50,
            disable_price_batch=True,
            financial_source="sec",
            sec_user_agent=user_agent,
            use_next_trading_day_availability=bool(self.fin_next_trading_day_var.get()),
            fundamentals_availability_fallback=True,
            fundamentals_fallback_q_days=45,
            fundamentals_fallback_k_days=90,
            reparse_sec_from_cache=False,
            include_sector_cache=not bool(self.fin_skip_sector_var.get()),
            skip_price=True,
        )

        _tmp_csv = tickers_file if (self.fin_universe_var.get() == "direct") else None

        self._disable_buttons()
        self._append(
            f"[START] SEC 재무제표 + 섹터  universe={universe}  workers={workers}  "
            f"force={opts.force}  sector={'skip' if not opts.include_sector_cache else 'include'}"
        )

        def worker() -> None:
            try:
                import sys
                import io

                class _Redirector(io.TextIOBase):
                    def __init__(self, cb):
                        self._cb = cb
                    def write(self, s):
                        if s.strip():
                            self._cb(s.rstrip())
                        return len(s)

                old_stdout = sys.stdout
                sys.stdout = _Redirector(self._log)
                try:
                    code = ingest_data(opts)
                finally:
                    sys.stdout = old_stdout
                self._log(f"[DONE] 재무제표 + 섹터 완료  exit={code}")
            except Exception as exc:  # noqa: BLE001
                self._log(f"[ERROR] {exc}")
            finally:
                if _tmp_csv and Path(_tmp_csv).exists():
                    Path(_tmp_csv).unlink(missing_ok=True)
                self.log_q.put("__DONE__")

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()


def run_finterstellar_raw_gui() -> int:
    root = tk.Tk()
    ff = configure_ttk_style(root, ttk)
    RecentDataUpdaterGUI(root, font_family=ff)
    root.mainloop()
    return 0


def run_recent_data_gui() -> int:
    return run_finterstellar_raw_gui()
