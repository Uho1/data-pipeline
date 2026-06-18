from __future__ import annotations

import logging
from pathlib import Path
import json
from copy import deepcopy
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from market_data.backtest.models import (
    ConditionClause,
    ConditionSet,
    CostModel,
    ExecutionConfig,
    FundingConfig,
    PositionSizingConfig,
    RiskLimitsConfig,
    StrategyConfig,
    UniverseFilterConfig,
    load_strategy_config,
    strategy_config_from_dict,
)
from market_data.backtest.condition_engine import (
    dump_condition_set_dict,
    load_condition_set_dict,
    validate_condition_set,
)
from market_data import __version__ as APP_VERSION
from market_data.backtest.report import compare_results, export_ai_review_bundle
from market_data.backtest.simulator import run_backtest, run_funding_comparison, run_strategy_backtest_from_config
from market_data.metrics_catalog import MetricDefinition, list_metric_definitions
from market_data.ui_style import configure_treeview_tags

LOGGER = logging.getLogger(__name__)


TXT: dict[str, str] = {
    "strategy_config": "전략 설정 파일",
    "strategy_source": "전략 소스",
    "strategy_source_file": "파일",
    "strategy_source_builder": "빌더",
    "strategy_builder": "전략 빌더...",
    "strategy_builder_save": "전략 저장",
    "strategy_builder_preview": "전략 설정 미리보기",
    "strategy_builder_summary_default": "전략(빌더): 미설정",
    "browse": "찾아보기",
    "screen_rule": "스크리너 조건",
    "rule_builder": "조건 빌더...",
    "validate_conditions": "조건 검증",
    "save_conditions": "조건 저장",
    "load_conditions": "조건 불러오기",
    "conditions_valid": "조건식 검증 통과",
    "conditions_invalid": "조건식 검증 실패",
    "condition_file_title": "조건식 파일",
    "condition_file_error": "조건식 파일 처리 오류",
    "builder_applied": "빌더 적용됨",
    "builder_manual": "수동 입력",
    "market": "시장",
    "start": "시작",
    "end": "종료",
    "freq": "리밸런싱 주기",
    "holdings": "보유 종목 수",
    "benchmark": "벤치마크",
    "exec": "체결 시점",
    "initial_cash": "초기 자본",
    "commission_bps": "수수료(bps)",
    "slippage_bps": "슬리피지(bps)",
    "share_mode": "매수 단위",
    "cash_buffer_pct": "현금 버퍼(%)",
    "position_weight_pct": "종목당 비중(%)",
    "max_holdings_cfg": "최대 보유 수",
    "max_new_buys": "1일 최대 신규매수",
    "max_buy_amount": "종목당 최대 매수금액",
    "min_cash_reserve_pct": "최소 현금보유(%)",
    "exec_price_basis": "가격 기준",
    "price_offset_pct": "가격 오프셋(%)",
    "price_basis_auto": "auto",
    "price_basis_open": "open",
    "price_basis_close": "close",
    "price_basis_prev_close": "prev_close",
    "universe_filter_summary": "유니버스 필터",
    "universe_exchanges": "거래소(쉼표)",
    "universe_size_buckets": "규모버킷(쉼표)",
    "universe_include_sectors": "업종 포함(쉼표)",
    "universe_exclude_sectors": "업종 제외(쉼표)",
    "universe_include_subsectors": "세부업종 포함",
    "universe_exclude_subsectors": "세부업종 제외",
    "universe_exclude_tickers": "제외 티커(쉼표)",
    "funding_mode": "자금투입 방식",
    "funding_mode_lump": "일시불",
    "funding_mode_dca": "정액적립(DCA)",
    "funding_mode_va": "가치평균(VA)",
    "funding_mode_compare": "비교(Lump/DCA/VA)",
    "funding_contribution_freq": "자금투입 주기",
    "funding_freq_monthly": "월간(M)",
    "funding_freq_quarterly": "분기(Q)",
    "funding_freq_yearly": "연간(Y)",
    "funding_freq_hint": "자금투입 주기는 리밸런싱 주기와 독립적으로 설정됩니다.",
    "funding_event_order_hint": "같은 날짜에는 자금투입 후 리밸런싱 순서로 처리됩니다.",
    "funding_dca_amount": "DCA 납입액",
    "funding_va_step": "VA 목표증가액",
    "funding_va_min": "VA 최소 납입(정책에 따라 적용)",
    "funding_va_max": "VA 최대 납입",
    "funding_va_min_policy": "VA 최소납입 정책",
    "va_min_policy_positive_raw_only": "양수 raw일 때만 적용(기존)",
    "va_min_policy_every_rebalance": "매 리밸런싱 최소납입 강제(권장)",
    "va_min_policy_hint": "강제 정책 선택 시, VA 최소납입이 DCA 납입액과 같으면 VA 누적납입원금은 DCA 이상이 될 수 있습니다.",
    "funding_va_withdraw": "VA 음수납입 허용",
    "funding_help": "VA = 목표 포트폴리오 가치 경로 기준 가변 납입",
    "funding_preview": "자금투입 미리보기",
    "funding_view": "자금 방식 표시",
    "funding_compare_basis": "비교 기준",
    "funding_compare_default": "기본(현재 방식)",
    "funding_compare_align_dca": "DCA 총납입원금 기준으로 Lump 맞춤",
    "funding_compare_custom": "사용자 지정 총원금 기준",
    "funding_custom_total": "사용자 지정 총원금",
    "funding_lump_aligned": "일시불(총원금맞춤)",
    "funding_parse_warning": "입력 경고",
    "preflight_failed": "실행 전 검증 실패",
    "funding_flows_tab": "자금흐름",
    "funding_flows_all_tab": "전체(통합)",
    "funding_flows_lump_tab": "일시불",
    "funding_flows_dca_tab": "정액적립(DCA)",
    "funding_flows_va_tab": "가치평균(VA)",
    "funding_flow_view_mode": "표시 모드",
    "funding_flow_view_summary": "요약",
    "funding_flow_view_detail": "상세",
    "funding_flow_explain": "계좌평가금액 = 현금 + 포지션평가금액 / 포지션평가금액은 실제 시장에 투자된 금액입니다.",
    "funding_compare_tab": "자금투입 비교",
    "funding_compare_explain": "TWR=전략 성과(외부 납입 제외), 계좌 기준=현금유입 포함 실제 잔고 리스크, IRR=투자자 체감 성과",
    "context_strategy_compare": "전략 비교: 스크리너 vs 전략",
    "context_funding_compare": "자금방식 비교: 일시불 / DCA / VA",
    "section_strategy": "전략 설정",
    "section_backtest": "백테스트 설정",
    "section_funding": "자금투입 설정",
    "section_actions": "실행",
    "section_summary": "요약",
    "section_buy": "매수 조건",
    "section_sell": "매도 조건",
    "section_universe": "매매 대상 설정",
    "buy_condition_help": "슬롯 A/B/C를 채우고 논리식에 조합하세요. 예: (A and B) or not C",
    "sell_condition_help": "조건 매도 슬롯. 비우면 기존 sell_mode 규칙만 사용합니다.",
    "condition_slot": "조건 슬롯",
    "logical_expression": "논리식",
    "condition_left_expr": "좌변식",
    "condition_operator": "연산",
    "condition_right_value": "우변값",
    "condition_reset": "초기화",
    "save_strategy": "전략 저장",
    "load_strategy": "전략 불러오기",
    "preflight_check": "실행 전 검증",
    "preflight_ok": "실행 전 검증 통과",
    "strategy_saved": "전략 저장 완료",
    "strategy_loaded": "전략 불러오기 완료",
    "strategy_file_error": "전략 파일 처리 오류",
    "audit_prompt_copy": "감사 프롬프트 복사",
    "audit_prompt_copied": "감사 프롬프트를 클립보드에 복사했습니다.",
    "sell_mode_ui": "매도 모드",
    "sell_mode_a": "A(전량교체)",
    "sell_mode_b": "B(매수조건이탈)",
    "sell_mode_c": "C(랭크하락)",
    "rank_drop_mult": "랭크 하락 배수",
    "hold_days": "최대 보유일(예정)",
    "todo_not_connected": "TODO: 엔진 미연동(표시 전용)",
    "universe_count_hint": "거래소/규모/업종/세부업종/제외종목을 조합해 유니버스를 필터링합니다.",
    "pit_guard_hint": "업종은 신호일(as-of) 기준 PIT 데이터만 사용합니다. 미래 filing/acceptance는 제외됩니다.",
    "advanced_toggle": "고급 설정",
    "advanced_on": "고급 설정 접기",
    "advanced_off": "고급 설정 펼치기",
    "override": "UI 설정으로 덮어쓰기",
    "run_screen": "스크리너 실행",
    "run_strategy": "전략 실행",
    "compare": "비교",
    "ai_export": "AI 검증용 저장",
    "ai_export_error": "AI 검증용 저장 오류",
    "ai_export_done": "AI 검증용 저장 완료",
    "ai_export_no_result": "저장할 백테스트 결과가 없습니다. 먼저 실행해 주세요.",
    "lower_chart": "하단 그래프",
    "table_source": "표 데이터",
    "idle": "대기",
    "done": "완료",
    "running_screen": "스크리너 실행 중...",
    "running_strategy": "전략 실행 중...",
    "running_compare": "비교 실행 중...",
    "filters": "필터",
    "ranking": "랭킹",
    "builder_summary_default": "필터: - | 랭킹: -",
    "equity_curve": "누적 자산",
    "drawdown": "드로다운",
    "relative_curve": "벤치 대비 누적 초과수익",
    "kpi_cagr": "연복리",
    "kpi_mdd": "최대낙폭",
    "kpi_sharpe": "샤프",
    "kpi_turnover": "회전율",
    "kpi_trades": "거래횟수",
    "monthly_returns_tab": "월별 수익률",
    "yearly_returns_tab": "연도별 수익률",
    "trades_tab": "거래내역",
    "rebalance_log_tab": "리밸런싱 로그",
    "snapshots_tab": "스냅샷",
    "metrics_tab": "성과지표",
    "preview_snapshot": "선택 스냅샷 미리보기",
    "snapshot_preview": "스냅샷 미리보기",
    "all": "전체",
    "source_auto": "자동",
    "source_screen": "스크리너",
    "source_strategy": "전략",
    "chart_mode_relative": "초과수익",
    "chart_mode_drawdown": "드로다운",
    "share_mode_fractional": "소수주",
    "share_mode_integer": "정수주",
    "trade_analysis": "거래 분석",
    "top_n": "상위 티커 수",
    "ticker_scope": "표시 범위",
    "scope_all": "전체",
    "scope_top_n": "상위 N",
    "heat_metric": "히트맵 지표",
    "metric_net_notional": "순매수금액",
    "metric_net_count": "순매수건수",
    "sort_by": "정렬 기준",
    "sort_notional": "거래금액합",
    "sort_count": "거래횟수",
    "month_filter": "월 선택",
    "ticker_filter": "티커 선택",
    "heatmap_title_notional": "월별 티커 순매수금액 히트맵",
    "heatmap_title_count": "월별 티커 순매수건수 히트맵",
    "monthly_trade_summary": "월별 매수/매도 요약(건수)",
    "monthly_return_chart": "월별 수익률(막대) + 누적선",
    "return_plot_mode": "수익 그래프 표시",
    "ret_mode_strat_bench": "전략 vs 벤치",
    "ret_mode_strat_excess": "전략 vs 초과",
    "ret_mode_excess_only": "초과만",
    "cum_line_mode": "누적선",
    "cum_mode_strategy": "누적 전략",
    "cum_mode_excess": "누적 초과",
    "cum_mode_none": "없음",
    "hover_info": "Hover 정보",
    "hover_default": "히트맵 셀 위에 마우스를 올리면 상세 정보가 표시됩니다. 클릭하면 거래내역이 해당 티커/월로 필터링됩니다.",
    "hover_no_trade": "해당 월/티커 거래가 없습니다.",
    "tooltip_on": "툴팁 표시",
    "legend_strategy": "전략",
    "legend_benchmark": "벤치",
    "legend_excess": "초과",
    "buy_count": "매수 건수",
    "sell_count": "매도 건수",
    "trade_reason_detail": "상세 사유",
    "no_trade_data": "거래 데이터가 없습니다.",
    "metric_total_contributed": "누적납입원금",
    "metric_ending_value": "최종평가금액",
    "metric_pnl": "손익",
    "metric_twr_cagr": "TWR 연복리",
    "metric_mwr_irr": "IRR(XIRR)",
    "legend_contributed": "누적 납입원금",
    "rule_builder_title": "조건 빌더",
    "search": "검색",
    "preset": "전략 프리셋",
    "preset_custom": "사용자 정의",
    "preset_hint": "프리셋 안내",
    "category": "카테고리",
    "category_all": "전체",
    "view_mode": "보기",
    "view_both": "필터+랭킹",
    "view_filter": "필터 전용",
    "view_rank": "랭킹 전용",
    "selected_only": "선택만",
    "favorites_only": "즐겨찾기만",
    "favorite": "즐겨찾기",
    "metric_list": "지표 목록",
    "metric_detail": "지표 상세",
    "metric_availability": "계산 가능",
    "metric_description": "설명",
    "metric_formula": "계산식",
    "metric_source": "데이터 소스",
    "metric_required": "필요 컬럼",
    "filter_group": "필터 그룹",
    "normalize_weights": "가중치 자동정규화(합=100)",
    "builder_warning": "경고",
    "rule_preview": "조건 미리보기",
    "strategy_sell_section": "매도 조건",
    "sell_mode": "매도 모드",
    "rank_drop_mult": "랭크 컷오프 배수",
    "na_policy": "NA 정책",
    "sell_mode_hint": "A=전량교체, B=조건미충족 매도, C=랭크하락 매도",
    "strategy_builder_name": "전략(빌더)",
    "strategy_builder_saved": "전략 저장 완료",
    "apply": "적용",
    "apply_close": "적용 후 닫기",
    "cancel": "취소",
    "reset": "초기화",
    "load_preset": "불러오기",
    "save_preset": "저장",
    "saved": "저장 완료",
    "input_error": "입력 오류",
    "run_screen_error": "스크리너 실행 오류",
    "run_strategy_error": "전략 실행 오류",
    "compare_error": "비교 오류",
    "snapshot_preview_error": "스냅샷 미리보기 오류",
    "load_preset_error": "프리셋 불러오기 오류",
    "save_preset_error": "프리셋 저장 오류",
    "rule_builder_error": "조건 빌더 오류",
    "strategy_path_required": "전략 설정 파일 경로가 필요합니다.",
    "screen_or_strategy_hint": "스크리너/전략 실행 후 결과가 표시됩니다.",
    "tab_settings": "포트 설정",
    "tab_results": "백테스터 결과",
    "stream_log_title": "실시간 거래 로그",
    "stream_col_date": "날짜",
    "stream_col_ticker": "종목",
    "stream_col_side": "매매",
    "stream_col_price": "가격",
    "stream_col_shares": "수량",
    "stream_buy": "매수",
    "stream_sell": "매도",
    "progress_running_pct": "실행 중",
    "progress_idle": "대기",
    "heatmap_tab": "히트맵",
    "ticker_perf_tab": "종목별 수익률",
    "ticker_top_label": "수익률 상위",
    "ticker_bottom_label": "수익률 하위",
    "ticker_no_data": "거래 데이터가 없습니다.",
    "ticker_pnl_col": "손익",
    "ticker_ret_col": "수익률",
    "ticker_cost_col": "매수금액",
}


COL_TXT: dict[str, str] = {
    "source": "구분",
    "month": "월",
    "strategy_ret": "전략 수익률",
    "bench_ret": "벤치 수익률",
    "excess_ret": "초과수익",
    "year": "연도",
    "screen": "스크리너",
    "strategy": "전략",
    "exec_date": "체결일",
    "ticker": "티커",
    "side": "매매",
    "shares": "수량",
    "exec_price": "체결가",
    "notional": "거래금액",
    "commission": "수수료",
    "slippage": "슬리피지",
    "reason": "사유",
    "signal_date": "신호일",
    "mode": "모드",
    "universe_count": "유니버스 수",
    "candidate_count": "후보 수",
    "selected_count": "선정 수",
    "lookahead_ok": "룩어헤드 검증",
    "benchmark": "벤치마크",
    "snapshot_path": "스냅샷 파일",
    "filter_pass": "필터 통과",
    "rank_score": "랭킹 점수",
    "rank": "순위",
    "selected": "선정",
    "target_weight": "목표 비중",
    "metric": "지표",
    "mode_name": "방식",
    "mode_label": "방식",
    "twr_total_return": "TWR 총수익률",
    "total_contributed": "누적납입원금",
    "ending_value": "최종평가금액",
    "pnl": "손익",
    "twr_cagr": "TWR 연복리",
    "twr_mdd": "TWR MDD",
    "twr_sharpe": "TWR 샤프",
    "twr_vol": "TWR 변동성",
    "mdd": "MDD",
    "sharpe": "샤프",
    "vol": "변동성",
    "account_cagr": "계좌 연복리",
    "account_mdd": "계좌 MDD",
    "account_sharpe": "계좌 샤프",
    "account_vol": "계좌 변동성",
    "mwr_irr": "IRR(XIRR)",
    "trades_count": "거래횟수",
    "turnover": "회전율",
    "date": "일자",
    "step_index": "회차",
    "funding_mode": "자금방식",
    "contribution_freq": "투입주기",
    "contribution": "납입금",
    "raw_contribution": "원시납입",
    "cumulative_contributed_before": "누적납입원금(전)",
    "cumulative_contributed_after": "누적납입원금(후)",
    "target_value": "목표가치",
    "cash_before_funding": "투입전 현금",
    "positions_value_before_funding": "자금투입전 포지션평가",
    "account_equity_before_funding": "자금투입전 계좌평가",
    "cash_after_funding": "투입후 현금",
    "positions_value_after_funding": "포지션평가금액(후)",
    "account_equity_after_funding": "계좌평가금액(후)",
    "cash_after_rebalance": "리밸런싱후 현금",
    "positions_value_after_rebalance": "리밸런싱후 포지션평가",
    "account_equity_after_rebalance": "리밸런싱후 계좌평가",
    "had_rebalance_same_day": "당일 리밸런싱",
    "rebalance_exec_timing": "리밸런싱 체결시점",
    "invested_ratio_after_funding": "투자비중(후)",
    "invested_ratio_after_rebalance": "투자비중(리밸런싱후)",
    "is_rebalance_same_day": "당일리밸런싱",
    "event_order_tag": "이벤트순서",
    "cash_before_rebalance": "리밸런싱전 현금",
    "cash_after_rebalance": "리밸런싱후 현금",
    "had_funding_same_day": "당일투입여부",
    "portfolio_value_before": "납입전 가치",
    "portfolio_value_after": "납입후 가치",
    "va_min_contribution": "VA 최소납입",
    "va_max_contribution": "VA 최대납입",
    "va_min_policy": "VA 최소정책",
    "floor_applied": "최소납입적용",
    "max_cap_applied": "최대캡적용",
    "cap_applied_min": "최소캡적용",
    "cap_applied_max": "최대캡적용",
    "compare_basis": "비교기준",
    "ticker_return": "수익률",
    "ticker_pnl": "손익",
    "ticker_cost": "매수금액",
}


def tr(key: str) -> str:
    return TXT.get(key, key)


def _install_text_context_menus(root: tk.Misc) -> None:
    """Entry / Text 위젯 전체에 우클릭 컨텍스트 메뉴를 등록한다.

    bind_class 를 사용하므로 같은 Tk 인스턴스 안의 모든 Entry / Text 에
    일괄 적용된다.  두 번 호출해도 중복 등록되지 않도록 플래그를 사용한다.
    """
    top: tk.Tk | tk.Toplevel = root.winfo_toplevel()  # type: ignore[assignment]
    flag = "_ctx_menu_installed"
    if getattr(top, flag, False):
        return
    setattr(top, flag, True)

    def _show(event: tk.Event) -> None:  # noqa: ANN001
        w = event.widget
        try:
            state = str(w.cget("state"))
        except Exception:
            state = "normal"
        editable = state not in ("disabled", "readonly")

        menu = tk.Menu(w, tearoff=0)

        def _cmd(virtual: str) -> Callable[[], None]:
            return lambda: w.event_generate(virtual)

        if editable:
            menu.add_command(label="잘라내기", command=_cmd("<<Cut>>"),       accelerator="⌘X")
        menu.add_command(    label="복사",     command=_cmd("<<Copy>>"),      accelerator="⌘C")
        if editable:
            menu.add_command(label="붙여넣기", command=_cmd("<<Paste>>"),     accelerator="⌘V")
        menu.add_separator()
        menu.add_command(    label="전체 선택", command=_cmd("<<SelectAll>>"), accelerator="⌘A")

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    for cls in ("Entry", "Text"):
        top.bind_class(cls, "<Button-3>", _show, add="+")


def _parse_optional_float(value: Any) -> tuple[float | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, (int, float)):
        return float(value), False
    text = str(value).strip()
    if not text:
        return None, False
    try:
        return float(text), False
    except Exception:
        return None, True


def _format_number(value: float) -> str:
    if abs(value - int(value)) < 1e-12:
        return str(int(value))
    return f"{value:g}"


def parse_csv_values(
    text: str,
    *,
    upper: bool = False,
    lower: bool = False,
) -> list[str]:
    raw = str(text or "").replace("\n", ",").replace(";", ",")
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if upper:
        return [v.upper() for v in vals]
    if lower:
        return [v.lower() for v in vals]
    return vals


UNIVERSE_BUCKETS: list[str] = ["mega", "large", "mid", "small", "micro"]
UNIVERSE_PRESET_TO_EXCHANGE: dict[str, str] = {
    "nasdaq_all": "NASDAQ",
    "nyse_all": "NYSE",
}


def normalize_universe_bucket_selections(raw: dict[str, Any] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not isinstance(raw, dict):
        return out
    for key, vals in raw.items():
        k = str(key).strip().upper()
        if k not in {"NASDAQ", "NYSE"}:
            continue
        bucket_vals = [
            str(v).strip().lower()
            for v in (vals or [])
            if str(v).strip() and str(v).strip().lower() in set(UNIVERSE_BUCKETS)
        ]
        out[k] = list(dict.fromkeys(bucket_vals))
    return out


def summarize_universe_parent_state(selected_buckets: list[str]) -> tuple[bool, int, int]:
    uniq = list(dict.fromkeys([str(x).strip().lower() for x in (selected_buckets or []) if str(x).strip()]))
    total = len(UNIVERSE_BUCKETS)
    count = len([x for x in uniq if x in set(UNIVERSE_BUCKETS)])
    return count == total, count, total


def map_universe_presets_to_filters(
    *,
    presets: list[str],
    bucket_selections: dict[str, list[str]],
    manual_exchanges: list[str] | None = None,
    manual_size_buckets: list[str] | None = None,
) -> dict[str, Any]:
    allowed_presets = {"sp500", "sp500_pit", "nasdaq_all", "nyse_all", "nasdaq_stock_only_financial"}
    normalized_presets = [str(p).strip().lower() for p in (presets or []) if str(p).strip().lower() in allowed_presets]
    normalized_presets = list(dict.fromkeys(normalized_presets))
    normalized_buckets = normalize_universe_bucket_selections(bucket_selections)
    exchanges = [str(x).strip().upper() for x in (manual_exchanges or []) if str(x).strip()]
    size_buckets = [str(x).strip().lower() for x in (manual_size_buckets or []) if str(x).strip()]
    errors: list[str] = []
    warnings: list[str] = []

    has_sp500 = "sp500" in normalized_presets
    has_sp500_pit = "sp500_pit" in normalized_presets
    has_nasdaq_fin = "nasdaq_stock_only_financial" in normalized_presets
    
    if (has_sp500 or has_sp500_pit or has_nasdaq_fin) and any(p in normalized_presets for p in {"nasdaq_all", "nyse_all"}):
        errors.append("S&P500/Nasdaq-Fin preset은 NASDAQ/NYSE preset과 동시에 사용할 수 없습니다.")

    if (has_sp500 or has_sp500_pit or has_nasdaq_fin) and exchanges:
        errors.append("S&P500/Nasdaq-Fin preset 사용 시 거래소 필터를 동시에 지정할 수 없습니다.")

    if (has_sp500 or has_sp500_pit or has_nasdaq_fin) and any(v for v in normalized_buckets.values()):
        errors.append("S&P500/Nasdaq-Fin preset 사용 시 거래소 버킷 선택을 동시에 지정할 수 없습니다.")

    if has_sp500 or has_sp500_pit or has_nasdaq_fin:
        source = "sp500"
        if has_sp500_pit:
            source = "sp500_pit"
        elif has_nasdaq_fin:
            source = "nasdaq_stock_only_financial"
            
        return {
            "universe_source": source,
            "exchanges": [],
            "size_buckets": [],
            "presets": normalized_presets,
            "bucket_selections": normalized_buckets,
            "errors": errors,
            "warnings": warnings,
        }

    out_exchanges = list(dict.fromkeys(exchanges))
    out_sizes = list(dict.fromkeys([b for b in size_buckets if b in set(UNIVERSE_BUCKETS)]))

    for preset, exchange in UNIVERSE_PRESET_TO_EXCHANGE.items():
        key = exchange
        selected = preset in normalized_presets
        chosen = normalized_buckets.get(key, [])

        if chosen and not selected:
            warnings.append(f"{exchange} 버킷만 선택되어 {exchange} 전체 선택으로 자동 보정합니다.")
            selected = True
            normalized_presets.append(preset)

        if selected:
            out_exchanges.append(exchange)
            if chosen:
                out_sizes.extend(chosen)
            else:
                out_sizes.extend(UNIVERSE_BUCKETS)

    out_exchanges = list(dict.fromkeys(out_exchanges))
    out_sizes = list(dict.fromkeys([b for b in out_sizes if b in set(UNIVERSE_BUCKETS)]))
    return {
        "universe_source": "symbols",
        "exchanges": out_exchanges,
        "size_buckets": out_sizes,
        "presets": list(dict.fromkeys(normalized_presets)),
        "bucket_selections": normalized_buckets,
        "errors": errors,
        "warnings": list(dict.fromkeys(warnings)),
    }


def preflight_validate_strategy_config(
    cfg: StrategyConfig,
    *,
    available_columns: list[str] | None = None,
    universe_selection: dict[str, Any] | None = None,
) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    cols = list(available_columns or [])

    if cfg.buy_condition_set and cfg.buy_condition_set.conditions:
        vr = validate_condition_set(cfg.buy_condition_set, available_columns=cols)
        if not vr.ok:
            errors.extend(vr.errors)
        warnings.extend(vr.warnings)
    if cfg.sell_condition_set and cfg.sell_condition_set.conditions:
        vr = validate_condition_set(cfg.sell_condition_set, available_columns=cols)
        if not vr.ok:
            errors.extend(vr.errors)
        warnings.extend(vr.warnings)

    if cfg.position_sizing.max_holdings is not None and cfg.position_sizing.max_holdings <= 0:
        errors.append("position_sizing.max_holdings must be > 0")
    if cfg.holdings <= 0:
        errors.append("holdings must be > 0")
    if not str(cfg.buy_rules or "").strip() and not cfg.buy_condition_set.conditions and not cfg.indicators:
        errors.append("At least one buy rule / condition / indicator is required")
    if cfg.position_sizing.max_holdings is not None and cfg.position_sizing.max_holdings < cfg.holdings:
        warnings.append("position_sizing.max_holdings is lower than holdings; effective holdings will be reduced")
    if cfg.risk_limits.min_cash_reserve_pct < 0 or cfg.risk_limits.min_cash_reserve_pct >= 100:
        errors.append("risk_limits.min_cash_reserve_pct must be in [0, 100)")
    if cfg.execution.price_basis not in {"auto", "open", "close", "prev_close"}:
        errors.append("execution.price_basis must be one of auto/open/close/prev_close")
    if abs(float(cfg.execution.price_offset_pct or 0.0)) > 50.0:
        warnings.append("execution.price_offset_pct is very large (>50%)")
    if cfg.position_sizing.position_weight_pct is not None and cfg.position_sizing.position_weight_pct <= 0:
        errors.append("position_sizing.position_weight_pct must be > 0")
    if cfg.risk_limits.min_cash_reserve_pct > 20 and (cfg.position_sizing.position_weight_pct or 0) > 20:
        warnings.append("High cash reserve + high position weight may reduce fills significantly")
    if getattr(cfg, "strict_pit", False):
        warnings.append("Strict PIT mode is enabled: Backtest will fail immediately if look-ahead is detected or data is missing.")
    if cfg.universe.source == "sp500_pit" and not cfg.use_fundamentals_pit:
        warnings.append("sp500_pit universe is selected but fundamentals_pit is disabled. This mixes PIT and non-PIT data.")
        
    if isinstance(universe_selection, dict):
        errors.extend(list(universe_selection.get("errors", []) or []))
        warnings.extend(list(universe_selection.get("warnings", []) or []))
    return (len(errors) == 0), errors, warnings


def compile_indicator_payload(
    indicators_payload: list[dict[str, Any]],
    strict: bool = False,
) -> tuple[str, list[dict[str, Any]], str, list[str]]:
    filters_g1: list[str] = []
    filters_g2: list[str] = []
    ranking: list[str] = []
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []

    for item in indicators_payload:
        if not isinstance(item, dict):
            continue

        metric_id = str(item.get("id", "")).strip()
        if not metric_id:
            continue

        enabled = bool(item.get("enabled", False))
        if not enabled:
            continue

        as_filter = bool(item.get("as_filter", False))
        as_rank = bool(item.get("as_rank", False))
        op = str(item.get("op", "<=")).strip()
        if op not in {"<=", "<", ">=", ">", "==", "!="}:
            op = "<="
        filter_group = 1
        try:
            filter_group = int(item.get("filter_group", 1))
        except Exception:
            filter_group = 1
        if filter_group not in {1, 2}:
            filter_group = 1

        direction = str(item.get("direction_override") or item.get("direction") or "asc").strip().lower()
        if direction not in {"asc", "desc"}:
            direction = "asc"

        threshold, threshold_invalid = _parse_optional_float(item.get("threshold"))
        weight, weight_invalid = _parse_optional_float(item.get("weight"))
        winsor_low, winsor_low_invalid = _parse_optional_float(item.get("winsorize_low"))
        winsor_high, winsor_high_invalid = _parse_optional_float(item.get("winsorize_high"))

        if as_filter:
            if threshold_invalid:
                errors.append(f"{metric_id}: threshold")
            elif threshold is not None:
                token = f"{metric_id} {op} {_format_number(float(threshold))}"
                if filter_group == 2:
                    filters_g2.append(token)
                else:
                    filters_g1.append(token)

        if as_rank:
            if weight_invalid:
                errors.append(f"{metric_id}: weight")
            weight_val = 1.0 if weight is None else float(weight)
            ranking.append(f"{metric_id}({weight_val:.2f} {direction})")
        else:
            weight_val = 1.0 if weight is None else float(weight)

        if winsor_low_invalid:
            errors.append(f"{metric_id}: winsorize_low")
        if winsor_high_invalid:
            errors.append(f"{metric_id}: winsorize_high")

        normalized.append(
            {
                "id": metric_id,
                "enabled": True,
                "as_filter": as_filter,
                "op": op,
                "threshold": threshold,
                "filter_group": int(filter_group),
                "as_rank": as_rank,
                "weight": weight_val,
                "direction_override": direction,
                "winsorize_low": winsor_low,
                "winsorize_high": winsor_high,
            }
        )

    if strict and errors:
        return "", normalized, "", errors

    g1_expr = " & ".join(filters_g1)
    g2_expr = " & ".join(filters_g2)
    if g1_expr and g2_expr:
        rule_expr = f"({g1_expr}) | ({g2_expr})"
    else:
        rule_expr = g1_expr or g2_expr

    if filters_g1 and filters_g2:
        filter_txt = f"G1[{', '.join(filters_g1)}] OR G2[{', '.join(filters_g2)}]"
    else:
        filter_txt = ", ".join(filters_g1 or filters_g2) if (filters_g1 or filters_g2) else "-"

    summary = f"{tr('filters')}: {filter_txt} | {tr('ranking')}: {', '.join(ranking) if ranking else '-'}"
    return rule_expr, normalized, summary, errors


def build_trade_reason_summary_lines(
    trades_sub: pd.DataFrame,
    max_lines: int = 5,
) -> list[str]:
    if trades_sub is None or trades_sub.empty:
        return []

    reason_col = "reason_detail" if "reason_detail" in trades_sub.columns else "reason"
    reasons = trades_sub.get(reason_col)
    if reasons is None:
        return []

    text = reasons.fillna("").astype(str).str.strip()
    text = text.loc[text != ""]
    if text.empty:
        return []

    counts = text.value_counts()
    lines: list[str] = []
    for idx, (reason, cnt) in enumerate(counts.items()):
        if idx >= max_lines:
            break
        short = reason if len(reason) <= 120 else f"{reason[:117]}..."
        lines.append(f"- {short} ({int(cnt)}건)")

    extra = int(len(counts) - len(lines))
    if extra > 0:
        lines.append(f"...({extra}건 더)")
    return lines


def _default_funding_mode_labels() -> dict[str, str]:
    return {
        "lump_sum": tr("funding_mode_lump"),
        "dca": tr("funding_mode_dca"),
        "va": tr("funding_mode_va"),
    }


FUNDING_FLOW_COLUMNS: list[str] = [
    "mode_label",
    "date",
    "step_index",
    "funding_mode",
    "contribution_freq",
    "contribution",
    "raw_contribution",
    "reason",
    "cumulative_contributed_before",
    "cumulative_contributed_after",
    "cash_before_funding",
    "positions_value_before_funding",
    "account_equity_before_funding",
    "cash_after_funding",
    "positions_value_after_funding",
    "account_equity_after_funding",
    "cash_after_rebalance",
    "positions_value_after_rebalance",
    "account_equity_after_rebalance",
    "had_rebalance_same_day",
    "is_rebalance_same_day",
    "event_order_tag",
    "rebalance_exec_timing",
    "invested_ratio_after_funding",
    "invested_ratio_after_rebalance",
    "va_min_policy",
    "va_min_contribution",
    "va_max_contribution",
    "floor_applied",
    "max_cap_applied",
]

FUNDING_FLOW_SUMMARY_COLUMNS: list[str] = [
    "mode_label",
    "date",
    "contribution",
    "cumulative_contributed_after",
    "account_equity_after_funding",
    "positions_value_after_funding",
    "cash_after_funding",
    "reason",
]


def build_combined_funding_flows(
    results_by_mode: dict[str, Any] | None,
    mode_label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    base_cols = [
        "mode_key",
        "mode_label",
        "date",
        "step_index",
        "contribution",
        "raw_contribution",
        "reason",
        "contribution_freq",
        "is_rebalance_same_day",
        "event_order_tag",
    ]
    if not isinstance(results_by_mode, dict) or not results_by_mode:
        return pd.DataFrame(columns=base_cols)

    labels = mode_label_map or _default_funding_mode_labels()
    frames: list[pd.DataFrame] = []
    mode_order = ["lump_sum", "dca", "va"]
    mode_keys = mode_order + [m for m in results_by_mode.keys() if m not in mode_order]
    for mode in mode_keys:
        result = results_by_mode.get(mode)
        flows = getattr(result, "funding_flows", None)
        if flows is None or not isinstance(flows, pd.DataFrame) or flows.empty:
            continue
        ff = flows.copy()
        ff["mode_key"] = str(mode)
        ff["mode_label"] = labels.get(str(mode), str(mode))
        ff["date"] = pd.to_datetime(ff.get("date"), errors="coerce")
        frames.append(ff)

    if not frames:
        return pd.DataFrame(columns=base_cols)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    for col in base_cols:
        if col not in combined.columns:
            combined[col] = np.nan
    step = pd.to_numeric(combined.get("step_index"), errors="coerce").fillna(0).astype(int)
    combined["step_index"] = step
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values(["date", "mode_label", "step_index"]).reset_index(drop=True)
    return combined


DEFAULT_FACTOR_COLUMNS: set[str] = {
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "price",
    "adv20",
    "dollar_volume_20d",
    "market_cap",
    "eps_ttm",
    "bps",
    "revenue_ttm",
    "op_income_ttm",
    "gross_profit_ttm",
    "ocf_ttm",
    "fcf_ttm",
    "ni_ttm",
    "avg_equity",
    "equity",
    "avg_assets",
    "assets",
    "liabilities",
    "shares",
}


BUILDER_PRESETS: list[dict[str, Any]] = [
    {
        "name": "저PER+저PSR 가치",
        "frequency": "Q",
        "holdings": "20~30",
        "indicators": [
            {"id": "per", "enabled": True, "as_filter": True, "op": "<=", "threshold": 10, "filter_group": 1, "as_rank": True, "weight": 35, "direction_override": "asc"},
            {"id": "psr", "enabled": True, "as_filter": True, "op": "<=", "threshold": 3, "filter_group": 1, "as_rank": True, "weight": 25, "direction_override": "asc"},
            {"id": "pbr", "enabled": True, "as_filter": True, "op": "<=", "threshold": 2.5, "filter_group": 1, "as_rank": True, "weight": 15, "direction_override": "asc"},
            {"id": "op_margin", "enabled": True, "as_filter": True, "op": ">", "threshold": 0, "filter_group": 1, "as_rank": False},
            {"id": "fcf_yield", "enabled": True, "as_filter": True, "op": ">", "threshold": 0, "filter_group": 1, "as_rank": True, "weight": 15, "direction_override": "desc"},
            {"id": "debt_to_equity", "enabled": True, "as_filter": False, "as_rank": True, "weight": 10, "direction_override": "asc"},
        ],
    },
    {
        "name": "가치+퀄리티 결합",
        "frequency": "Q",
        "holdings": "20~30",
        "indicators": [
            {"id": "per", "enabled": True, "as_filter": True, "op": "<=", "threshold": 15, "filter_group": 1, "as_rank": True, "weight": 25, "direction_override": "asc"},
            {"id": "ev_ebitda", "enabled": True, "as_filter": True, "op": "<=", "threshold": 10, "filter_group": 2, "as_rank": True, "weight": 25, "direction_override": "asc"},
            {"id": "psr", "enabled": True, "as_filter": True, "op": "<=", "threshold": 4, "filter_group": 1, "as_rank": True, "weight": 10, "direction_override": "asc"},
            {"id": "roe", "enabled": True, "as_filter": True, "op": ">=", "threshold": 0.10, "filter_group": 1, "as_rank": True, "weight": 15, "direction_override": "desc"},
            {"id": "op_margin", "enabled": True, "as_filter": True, "op": ">", "threshold": 0, "filter_group": 1, "as_rank": True, "weight": 10, "direction_override": "desc"},
            {"id": "debt_to_equity", "enabled": True, "as_filter": True, "op": "<=", "threshold": 1.5, "filter_group": 1, "as_rank": True, "weight": 10, "direction_override": "asc"},
            {"id": "dollar_volume_20d", "enabled": True, "as_filter": False, "as_rank": True, "weight": 5, "direction_override": "desc"},
        ],
    },
    {
        "name": "크로스섹션 모멘텀(12-1)",
        "frequency": "M",
        "holdings": "20~50",
        "indicators": [
            {"id": "price", "enabled": True, "as_filter": True, "op": ">=", "threshold": 5, "filter_group": 1, "as_rank": False},
            {"id": "dollar_volume_20d", "enabled": True, "as_filter": True, "op": ">=", "threshold": 5_000_000, "filter_group": 1, "as_rank": False},
            {"id": "mom_12_1", "enabled": True, "as_filter": False, "as_rank": True, "weight": 50, "direction_override": "desc"},
            {"id": "mom_6m", "enabled": True, "as_filter": False, "as_rank": True, "weight": 25, "direction_override": "desc"},
            {"id": "mom_3m", "enabled": True, "as_filter": False, "as_rank": True, "weight": 15, "direction_override": "desc"},
            {"id": "vol_60d", "enabled": True, "as_filter": False, "as_rank": True, "weight": 10, "direction_override": "asc"},
        ],
    },
    {
        "name": "듀얼 모멘텀(ETF)",
        "frequency": "M",
        "holdings": "3~10",
        "indicators": [
            {"id": "mom_12m", "enabled": True, "as_filter": True, "op": ">", "threshold": 0, "filter_group": 1, "as_rank": True, "weight": 70, "direction_override": "desc"},
            {"id": "mom_6m", "enabled": True, "as_filter": False, "as_rank": True, "weight": 30, "direction_override": "desc"},
        ],
    },
    {
        "name": "추세추종(TSMOM)",
        "frequency": "M",
        "holdings": "20~40",
        "indicators": [
            {"id": "price_above_sma200", "enabled": True, "as_filter": True, "op": ">=", "threshold": 0.5, "filter_group": 1, "as_rank": False},
            {"id": "mom_12m", "enabled": True, "as_filter": True, "op": ">", "threshold": 0, "filter_group": 1, "as_rank": True, "weight": 40, "direction_override": "desc"},
            {"id": "mom_6m", "enabled": True, "as_filter": False, "as_rank": True, "weight": 25, "direction_override": "desc"},
            {"id": "mom_3m", "enabled": True, "as_filter": False, "as_rank": True, "weight": 15, "direction_override": "desc"},
            {"id": "ma_gap", "enabled": True, "as_filter": False, "as_rank": True, "weight": 10, "direction_override": "desc"},
            {"id": "vol_60d", "enabled": True, "as_filter": False, "as_rank": True, "weight": 10, "direction_override": "asc"},
        ],
    },
    {
        "name": "평균회귀(단기)",
        "frequency": "W",
        "holdings": "10~20",
        "indicators": [
            {"id": "rsi2", "enabled": True, "as_filter": True, "op": "<", "threshold": 10, "filter_group": 1, "as_rank": True, "weight": 50, "direction_override": "asc"},
            {"id": "boll_z", "enabled": True, "as_filter": True, "op": "<", "threshold": -1.0, "filter_group": 1, "as_rank": True, "weight": 30, "direction_override": "asc"},
            {"id": "price_above_sma200", "enabled": True, "as_filter": False, "as_rank": True, "weight": 20, "direction_override": "desc"},
        ],
    },
    {
        "name": "퀄리티+모멘텀",
        "frequency": "M",
        "holdings": "20~40",
        "indicators": [
            {"id": "roe", "enabled": True, "as_filter": True, "op": ">=", "threshold": 0.12, "filter_group": 1, "as_rank": True, "weight": 25, "direction_override": "desc"},
            {"id": "op_margin", "enabled": True, "as_filter": True, "op": ">", "threshold": 0, "filter_group": 1, "as_rank": True, "weight": 20, "direction_override": "desc"},
            {"id": "debt_to_equity", "enabled": True, "as_filter": True, "op": "<=", "threshold": 1.5, "filter_group": 1, "as_rank": True, "weight": 15, "direction_override": "asc"},
            {"id": "mom_12_1", "enabled": True, "as_filter": True, "op": ">", "threshold": 0, "filter_group": 1, "as_rank": True, "weight": 25, "direction_override": "desc"},
            {"id": "price_above_sma200", "enabled": True, "as_filter": True, "op": ">=", "threshold": 0.5, "filter_group": 1, "as_rank": False},
            {"id": "mom_6m", "enabled": True, "as_filter": False, "as_rank": True, "weight": 15, "direction_override": "desc"},
        ],
    },
]


class RuleBuilderDialog:
    def __init__(
        self,
        parent: tk.Widget,
        initial_indicators: list[dict[str, Any]],
        metrics: list[MetricDefinition],
        mode: str = "screen",
        initial_sell_mode: str = "A",
        initial_rank_drop_multiplier: float = 2.0,
        initial_na_policy: str = "fail",
        on_apply: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.parent = parent.winfo_toplevel()
        self.metrics = sorted(metrics, key=lambda m: (m.category, m.label))
        self.mode = str(mode or "screen").strip().lower()
        self.on_apply = on_apply
        self.result: dict[str, Any] | None = None
        self._detail_updating = False
        self._metric_defs = {m.id: m for m in self.metrics}
        self._indicator_map = {
            str(item.get("id", "")).strip(): item
            for item in initial_indicators
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
        self._metric_state: dict[str, dict[str, Any]] = {}
        self._selected_metric_id: str | None = None
        self._preset_lookup = {str(p["name"]): p for p in BUILDER_PRESETS}
        self._build_state()

        self.window = tk.Toplevel(self.parent)
        self.window.title(tr("rule_builder_title"))
        self.window.geometry("1040x700")
        self.window.minsize(920, 620)

        self.search_var = tk.StringVar(value="")
        self.category_var = tk.StringVar(value=tr("category_all"))
        self.view_mode_var = tk.StringVar(value=tr("view_both"))
        self.selected_only_var = tk.BooleanVar(value=False)
        self.favorites_only_var = tk.BooleanVar(value=False)
        self.preset_var = tk.StringVar(value=tr("preset_custom"))
        self.preset_hint_var = tk.StringVar(value="")

        self.preview_rule_var = tk.StringVar(value="")
        self.preview_summary_var = tk.StringVar(value=tr("builder_summary_default"))
        self.status_var = tk.StringVar(value="")

        self.sell_mode_var = tk.StringVar(value=str(initial_sell_mode or "A").strip().upper())
        self.rank_multiplier_var = tk.StringVar(value=f"{float(initial_rank_drop_multiplier):.2f}")
        self.na_policy_var = tk.StringVar(value=str(initial_na_policy or "fail").strip().lower())

        self.detail_enabled_var = tk.BooleanVar(value=False)
        self.detail_favorite_var = tk.BooleanVar(value=False)
        self.detail_filter_var = tk.BooleanVar(value=False)
        self.detail_group_var = tk.StringVar(value="1")
        self.detail_op_var = tk.StringVar(value="<=")
        self.detail_threshold_var = tk.StringVar(value="")
        self.detail_rank_var = tk.BooleanVar(value=False)
        self.detail_weight_var = tk.StringVar(value="1.0")
        self.detail_direction_var = tk.StringVar(value="asc")
        self.detail_winsor_low_var = tk.StringVar(value="")
        self.detail_winsor_high_var = tk.StringVar(value="")

        self.metric_title_var = tk.StringVar(value="-")
        self.metric_meta_var = tk.StringVar(value="-")
        self.metric_desc_var = tk.StringVar(value="-")
        self.metric_formula_var = tk.StringVar(value="-")
        self.metric_required_var = tk.StringVar(value="-")
        self.metric_source_var = tk.StringVar(value="-")
        self.metric_available_var = tk.StringVar(value="-")

        self._build_ui()
        self._bind_traces()
        self._refresh_metric_list()
        self._select_first_metric()
        self._update_preview()

    def _build_state(self) -> None:
        for metric in self.metrics:
            initial = self._indicator_map.get(metric.id, {})
            direction = str(initial.get("direction_override") or metric.direction_default).strip().lower()
            if direction not in {"asc", "desc"}:
                direction = metric.direction_default
            try:
                group = int(initial.get("filter_group", 1))
            except Exception:
                group = 1
            if group not in {1, 2}:
                group = 1

            self._metric_state[metric.id] = {
                "enabled": bool(initial.get("enabled", False)),
                "favorite": bool(initial.get("favorite", False)),
                "as_filter": bool(initial.get("as_filter", False)),
                "filter_group": str(group),
                "op": str(initial.get("op", "<=")).strip() or "<=",
                "threshold": "" if initial.get("threshold") is None else str(initial.get("threshold")),
                "as_rank": bool(initial.get("as_rank", False)),
                "weight": str(initial.get("weight", "1.0")),
                "direction_override": direction,
                "winsorize_low": "" if initial.get("winsorize_low") is None else str(initial.get("winsorize_low")),
                "winsorize_high": "" if initial.get("winsorize_high") is None else str(initial.get("winsorize_high")),
            }

    def _build_ui(self) -> None:
        style = ttk.Style(self.window)
        style.configure("Invalid.TEntry", fieldbackground="#ffe5e5")

        body = ttk.Frame(self.window, padding=(10, 8, 10, 8))
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(2, weight=1)

        top1 = ttk.Frame(body)
        top1.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(top1, text=tr("preset")).pack(side="left")
        ttk.Combobox(
            top1,
            textvariable=self.preset_var,
            values=[tr("preset_custom")] + [p["name"] for p in BUILDER_PRESETS],
            width=22,
            state="readonly",
        ).pack(side="left", padx=(6, 10))
        ttk.Label(top1, text=tr("search")).pack(side="left")
        ttk.Entry(top1, textvariable=self.search_var, width=24).pack(side="left", padx=(6, 10))
        ttk.Label(top1, text=tr("category")).pack(side="left")
        categories = sorted({m.category for m in self.metrics})
        ttk.Combobox(
            top1,
            textvariable=self.category_var,
            values=[tr("category_all")] + categories,
            width=14,
            state="readonly",
        ).pack(side="left", padx=(6, 10))
        ttk.Label(top1, text=tr("view_mode")).pack(side="left")
        ttk.Combobox(
            top1,
            textvariable=self.view_mode_var,
            values=[tr("view_both"), tr("view_filter"), tr("view_rank")],
            width=10,
            state="readonly",
        ).pack(side="left", padx=(6, 10))
        ttk.Checkbutton(top1, text=tr("selected_only"), variable=self.selected_only_var).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(top1, text=tr("favorites_only"), variable=self.favorites_only_var).pack(side="left")

        ttk.Label(body, textvariable=self.preset_hint_var).grid(row=1, column=0, sticky="w", pady=(0, 6))

        paned = ttk.Panedwindow(body, orient="horizontal")
        paned.grid(row=2, column=0, sticky="nsew")

        left = ttk.Frame(paned, padding=(0, 0, 6, 0))
        right = ttk.Frame(paned, padding=(6, 0, 0, 0))
        paned.add(left, weight=2)
        paned.add(right, weight=3)

        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        ttk.Label(left, text=tr("metric_list")).grid(row=0, column=0, sticky="w", pady=(0, 4))
        tree_frame = ttk.Frame(left)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self.metric_tree = ttk.Treeview(
            tree_frame,
            columns=("id", "category", "flags", "fav", "avail"),
            show="headings",
            selectmode="browse",
            height=16,
        )
        self.metric_tree.heading("id", text="ID")
        self.metric_tree.heading("category", text=tr("category"))
        self.metric_tree.heading("flags", text="F/R")
        self.metric_tree.heading("fav", text="★")
        self.metric_tree.heading("avail", text=tr("metric_availability"))
        self.metric_tree.column("id", width=190, anchor="w")
        self.metric_tree.column("category", width=105, anchor="w")
        self.metric_tree.column("flags", width=45, anchor="center")
        self.metric_tree.column("fav", width=40, anchor="center")
        self.metric_tree.column("avail", width=70, anchor="center")
        self.metric_tree.grid(row=0, column=0, sticky="nsew")
        ysb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.metric_tree.yview)
        self.metric_tree.configure(yscrollcommand=ysb.set)
        ysb.grid(row=0, column=1, sticky="ns")
        self.metric_tree.bind("<<TreeviewSelect>>", self._on_metric_select)

        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)
        ttk.Label(right, text=tr("metric_detail")).grid(row=0, column=0, sticky="w")
        ttk.Label(right, textvariable=self.metric_title_var, font=("TkDefaultFont", 11, "bold")).grid(row=1, column=0, sticky="w", pady=(2, 4))

        meta = ttk.Frame(right)
        meta.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(meta, textvariable=self.metric_meta_var).grid(row=0, column=0, sticky="w")
        ttk.Label(meta, text=f"{tr('metric_availability')}:").grid(row=0, column=1, sticky="e", padx=(12, 2))
        ttk.Label(meta, textvariable=self.metric_available_var).grid(row=0, column=2, sticky="w")

        cfg = ttk.LabelFrame(right, text=tr("metric_detail"), padding=(8, 6, 8, 6))
        cfg.grid(row=3, column=0, sticky="nsew")
        cfg.grid_columnconfigure(1, weight=1)

        ttk.Checkbutton(cfg, text="Enabled", variable=self.detail_enabled_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(cfg, text=tr("favorite"), variable=self.detail_favorite_var).grid(row=0, column=1, sticky="w")

        ttk.Checkbutton(cfg, text="Filter(F)", variable=self.detail_filter_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        row_f = ttk.Frame(cfg)
        row_f.grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(row_f, text=tr("filter_group")).pack(side="left")
        self.filter_group_combo = ttk.Combobox(row_f, textvariable=self.detail_group_var, values=["1", "2"], width=3, state="readonly")
        self.filter_group_combo.pack(side="left", padx=(4, 8))
        self.filter_op_combo = ttk.Combobox(row_f, textvariable=self.detail_op_var, values=["<=", "<", ">=", ">", "==", "!="], width=4, state="readonly")
        self.filter_op_combo.pack(side="left", padx=(0, 4))
        self.threshold_entry = ttk.Entry(row_f, textvariable=self.detail_threshold_var, width=12)
        self.threshold_entry.pack(side="left")

        ttk.Checkbutton(cfg, text="Rank(R)", variable=self.detail_rank_var).grid(row=2, column=0, sticky="w", pady=(6, 0))
        row_r = ttk.Frame(cfg)
        row_r.grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Label(row_r, text="W").pack(side="left")
        self.weight_entry = ttk.Entry(row_r, textvariable=self.detail_weight_var, width=10)
        self.weight_entry.pack(side="left", padx=(4, 8))
        self.direction_combo = ttk.Combobox(row_r, textvariable=self.detail_direction_var, values=["asc", "desc"], width=6, state="readonly")
        self.direction_combo.pack(side="left")
        ttk.Button(row_r, text=tr("normalize_weights"), command=self._on_normalize_weights).pack(side="left", padx=(10, 0))

        row_w = ttk.Frame(cfg)
        row_w.grid(row=3, column=1, sticky="w", pady=(6, 0))
        ttk.Label(row_w, text="wL").pack(side="left")
        self.wl_entry = ttk.Entry(row_w, textvariable=self.detail_winsor_low_var, width=8)
        self.wl_entry.pack(side="left", padx=(4, 8))
        ttk.Label(row_w, text="wH").pack(side="left")
        self.wh_entry = ttk.Entry(row_w, textvariable=self.detail_winsor_high_var, width=8)
        self.wh_entry.pack(side="left", padx=(4, 0))

        info = ttk.LabelFrame(cfg, text=tr("rule_preview"), padding=(6, 4, 6, 4))
        info.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(info, text=f"{tr('metric_description')}:").grid(row=0, column=0, sticky="nw")
        ttk.Label(info, textvariable=self.metric_desc_var, wraplength=560, justify="left").grid(row=0, column=1, sticky="w")
        ttk.Label(info, text=f"{tr('metric_formula')}:").grid(row=1, column=0, sticky="nw")
        ttk.Label(info, textvariable=self.metric_formula_var, wraplength=560, justify="left").grid(row=1, column=1, sticky="w")
        ttk.Label(info, text=f"{tr('metric_required')}:").grid(row=2, column=0, sticky="nw")
        ttk.Label(info, textvariable=self.metric_required_var, wraplength=560, justify="left").grid(row=2, column=1, sticky="w")
        ttk.Label(info, text=f"{tr('metric_source')}:").grid(row=3, column=0, sticky="nw")
        ttk.Label(info, textvariable=self.metric_source_var, wraplength=560, justify="left").grid(row=3, column=1, sticky="w")

        preview_box = ttk.LabelFrame(right, text=tr("rule_preview"), padding=(8, 6, 8, 6))
        preview_box.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(preview_box, textvariable=self.preview_rule_var, justify="left", wraplength=620).pack(anchor="w")
        ttk.Label(preview_box, textvariable=self.preview_summary_var, justify="left", wraplength=620).pack(anchor="w", pady=(4, 0))

        if self.mode == "strategy":
            sell_frame = ttk.LabelFrame(right, text=tr("strategy_sell_section"), padding=(8, 6, 8, 6))
            sell_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
            ttk.Label(sell_frame, text=tr("sell_mode")).grid(row=0, column=0, sticky="w")
            ttk.Combobox(
                sell_frame,
                textvariable=self.sell_mode_var,
                values=["A", "B", "C"],
                width=6,
                state="readonly",
            ).grid(row=0, column=1, sticky="w", padx=(4, 10))
            ttk.Label(sell_frame, text=tr("rank_drop_mult")).grid(row=0, column=2, sticky="w")
            ttk.Entry(sell_frame, textvariable=self.rank_multiplier_var, width=8).grid(row=0, column=3, sticky="w", padx=(4, 10))
            ttk.Label(sell_frame, text=tr("na_policy")).grid(row=0, column=4, sticky="w")
            ttk.Combobox(
                sell_frame,
                textvariable=self.na_policy_var,
                values=["fail", "pass"],
                width=8,
                state="readonly",
            ).grid(row=0, column=5, sticky="w")
            ttk.Label(sell_frame, text=tr("sell_mode_hint")).grid(row=1, column=0, columnspan=6, sticky="w", pady=(4, 0))

        bottom = ttk.Frame(body)
        bottom.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left")
        ttk.Button(bottom, text=tr("reset"), command=self._on_reset).pack(side="right", padx=(4, 0))
        ttk.Button(bottom, text=tr("cancel"), command=self._on_cancel).pack(side="right", padx=(4, 0))
        ttk.Button(bottom, text=tr("apply_close"), command=lambda: self._on_apply(close_after=True)).pack(side="right", padx=(4, 0))
        ttk.Button(bottom, text=tr("apply"), command=lambda: self._on_apply(close_after=False)).pack(side="right", padx=(4, 0))
        ttk.Button(bottom, text=tr("save_preset"), command=self._on_save_preset).pack(side="right", padx=(12, 0))
        ttk.Button(bottom, text=tr("load_preset"), command=self._on_load_preset).pack(side="right", padx=(4, 0))

        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _bind_traces(self) -> None:
        self.search_var.trace_add("write", lambda *_: self._refresh_metric_list())
        self.category_var.trace_add("write", lambda *_: self._refresh_metric_list())
        self.view_mode_var.trace_add("write", lambda *_: self._refresh_metric_list())
        self.selected_only_var.trace_add("write", lambda *_: self._refresh_metric_list())
        self.favorites_only_var.trace_add("write", lambda *_: self._refresh_metric_list())
        self.preset_var.trace_add("write", lambda *_: self._on_preset_selected())

        for var in [
            self.detail_enabled_var,
            self.detail_favorite_var,
            self.detail_filter_var,
            self.detail_group_var,
            self.detail_op_var,
            self.detail_threshold_var,
            self.detail_rank_var,
            self.detail_weight_var,
            self.detail_direction_var,
            self.detail_winsor_low_var,
            self.detail_winsor_high_var,
        ]:
            var.trace_add("write", lambda *_: self._on_detail_changed())

        if self.mode == "strategy":
            self.sell_mode_var.trace_add("write", lambda *_: self._update_preview())
            self.rank_multiplier_var.trace_add("write", lambda *_: self._update_preview())
            self.na_policy_var.trace_add("write", lambda *_: self._update_preview())

    def _on_preset_selected(self) -> None:
        name = str(self.preset_var.get() or tr("preset_custom")).strip()
        if not name or name == tr("preset_custom"):
            return
        preset = self._preset_lookup.get(name)
        if not preset:
            return
        self._apply_preset(preset)

    def _apply_preset(self, preset: dict[str, Any]) -> None:
        self._on_reset(silent=True)
        indicators = preset.get("indicators", [])
        if isinstance(indicators, list):
            for item in indicators:
                if not isinstance(item, dict):
                    continue
                metric_id = str(item.get("id", "")).strip()
                if metric_id not in self._metric_state:
                    continue
                st = self._metric_state[metric_id]
                st["enabled"] = bool(item.get("enabled", True))
                st["as_filter"] = bool(item.get("as_filter", False))
                st["filter_group"] = str(int(item.get("filter_group", 1))) if str(item.get("filter_group", "1")).isdigit() else "1"
                st["op"] = str(item.get("op", st["op"])).strip() or "<="
                st["threshold"] = "" if item.get("threshold") is None else str(item.get("threshold"))
                st["as_rank"] = bool(item.get("as_rank", False))
                st["weight"] = str(item.get("weight", st["weight"]))
                direction = str(item.get("direction_override", st["direction_override"])).strip().lower()
                st["direction_override"] = direction if direction in {"asc", "desc"} else st["direction_override"]
                st["winsorize_low"] = "" if item.get("winsorize_low") is None else str(item.get("winsorize_low"))
                st["winsorize_high"] = "" if item.get("winsorize_high") is None else str(item.get("winsorize_high"))
        self.preset_hint_var.set(f"freq={preset.get('frequency', '-')}, holdings={preset.get('holdings', '-')}")
        self._refresh_metric_list()
        self._select_first_metric(prefer_selected=True)
        self._update_preview()

    def _is_metric_available(self, metric: MetricDefinition) -> bool:
        return all(col in DEFAULT_FACTOR_COLUMNS for col in metric.required_columns)

    def _matches_view_mode(self, metric: MetricDefinition, state: dict[str, Any]) -> bool:
        mode = str(self.view_mode_var.get() or tr("view_both")).strip()
        if mode == tr("view_filter"):
            return bool(metric.filterable)
        if mode == tr("view_rank"):
            return bool(metric.rankable)
        return True

    def _refresh_metric_list(self) -> None:
        current = self._selected_metric_id
        for iid in self.metric_tree.get_children():
            self.metric_tree.delete(iid)

        query = str(self.search_var.get() or "").strip().lower()
        category = str(self.category_var.get() or tr("category_all")).strip()
        selected_only = bool(self.selected_only_var.get())
        favorites_only = bool(self.favorites_only_var.get())

        metric_ids: list[str] = []
        for metric in self.metrics:
            st = self._metric_state.get(metric.id, {})
            if category != tr("category_all") and metric.category != category:
                continue
            if not self._matches_view_mode(metric, st):
                continue
            if selected_only and not bool(st.get("enabled") or st.get("as_filter") or st.get("as_rank")):
                continue
            if favorites_only and not bool(st.get("favorite")):
                continue
            text = f"{metric.id} {metric.label} {metric.category} {metric.description}".lower()
            if query and query not in text:
                continue
            metric_ids.append(metric.id)

        for mid in metric_ids:
            metric = self._metric_defs[mid]
            st = self._metric_state[mid]
            flags = "".join(["F" if bool(st.get("as_filter")) else "", "R" if bool(st.get("as_rank")) else ""]) or "-"
            fav = "★" if bool(st.get("favorite")) else ""
            avail = "Y" if self._is_metric_available(metric) else "N"
            self.metric_tree.insert("", "end", iid=mid, values=(f"{mid} ({metric.label})", metric.category, flags, fav, avail))

        if current and current in self.metric_tree.get_children():
            self.metric_tree.selection_set(current)
            self.metric_tree.focus(current)
        elif self.metric_tree.get_children():
            first = self.metric_tree.get_children()[0]
            self.metric_tree.selection_set(first)
            self.metric_tree.focus(first)
            self._on_metric_select()
        else:
            self._selected_metric_id = None

    def _select_first_metric(self, prefer_selected: bool = False) -> None:
        if not self.metric_tree.get_children():
            return
        if prefer_selected:
            for mid, st in self._metric_state.items():
                if mid in self.metric_tree.get_children() and bool(st.get("enabled") or st.get("as_filter") or st.get("as_rank")):
                    self.metric_tree.selection_set(mid)
                    self.metric_tree.focus(mid)
                    self._on_metric_select()
                    return
        first = self.metric_tree.get_children()[0]
        self.metric_tree.selection_set(first)
        self.metric_tree.focus(first)
        self._on_metric_select()

    def _on_metric_select(self, _event=None) -> None:
        selected = self.metric_tree.selection()
        if not selected:
            return
        metric_id = str(selected[0])
        if metric_id not in self._metric_state:
            return
        self._selected_metric_id = metric_id
        self._load_metric_detail(metric_id)

    def _load_metric_detail(self, metric_id: str) -> None:
        metric = self._metric_defs[metric_id]
        st = self._metric_state[metric_id]
        available = self._is_metric_available(metric)

        self._detail_updating = True
        try:
            self.metric_title_var.set(f"{metric.id} ({metric.label})")
            self.metric_meta_var.set(f"{metric.category} | unit={metric.unit} | direction={metric.direction_default}")
            self.metric_desc_var.set(metric.description or "-")
            self.metric_formula_var.set(metric.formula or "-")
            self.metric_required_var.set(", ".join(metric.required_columns) if metric.required_columns else "-")
            self.metric_source_var.set(metric.data_source or "-")
            self.metric_available_var.set("가능" if available else "불가")

            self.detail_enabled_var.set(bool(st.get("enabled", False)))
            self.detail_favorite_var.set(bool(st.get("favorite", False)))
            self.detail_filter_var.set(bool(st.get("as_filter", False)))
            self.detail_group_var.set(str(st.get("filter_group", "1")))
            self.detail_op_var.set(str(st.get("op", "<=")))
            self.detail_threshold_var.set(str(st.get("threshold", "")))
            self.detail_rank_var.set(bool(st.get("as_rank", False)))
            self.detail_weight_var.set(str(st.get("weight", "1.0")))
            self.detail_direction_var.set(str(st.get("direction_override", metric.direction_default)))
            self.detail_winsor_low_var.set(str(st.get("winsorize_low", "")))
            self.detail_winsor_high_var.set(str(st.get("winsorize_high", "")))
        finally:
            self._detail_updating = False

        state = "normal" if available else "disabled"
        for w in [
            self.filter_group_combo,
            self.filter_op_combo,
            self.threshold_entry,
            self.weight_entry,
            self.direction_combo,
            self.wl_entry,
            self.wh_entry,
        ]:
            w.configure(state=state if w not in {self.filter_group_combo, self.filter_op_combo, self.direction_combo} else ("readonly" if available else "disabled"))

    def _on_detail_changed(self) -> None:
        if self._detail_updating or self._selected_metric_id is None:
            return
        metric = self._metric_defs.get(self._selected_metric_id)
        if metric is None:
            return

        st = self._metric_state[self._selected_metric_id]
        st["enabled"] = bool(self.detail_enabled_var.get())
        st["favorite"] = bool(self.detail_favorite_var.get())
        st["as_filter"] = bool(self.detail_filter_var.get())
        st["filter_group"] = str(self.detail_group_var.get() or "1")
        st["op"] = str(self.detail_op_var.get() or "<=").strip()
        st["threshold"] = str(self.detail_threshold_var.get() or "").strip()
        st["as_rank"] = bool(self.detail_rank_var.get())
        st["weight"] = str(self.detail_weight_var.get() or "").strip()
        d = str(self.detail_direction_var.get() or metric.direction_default).strip().lower()
        st["direction_override"] = d if d in {"asc", "desc"} else metric.direction_default
        st["winsorize_low"] = str(self.detail_winsor_low_var.get() or "").strip()
        st["winsorize_high"] = str(self.detail_winsor_high_var.get() or "").strip()
        self._refresh_metric_list()
        self._update_preview()

    def _collect_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for metric in self.metrics:
            st = self._metric_state[metric.id]
            enabled = bool(st.get("enabled", False)) and self._is_metric_available(metric)
            payload.append(
                {
                    "id": metric.id,
                    "enabled": enabled,
                    "favorite": bool(st.get("favorite", False)),
                    "as_filter": bool(st.get("as_filter", False)),
                    "filter_group": int(st.get("filter_group", "1")) if str(st.get("filter_group", "1")).isdigit() else 1,
                    "op": str(st.get("op", "<=")),
                    "threshold": str(st.get("threshold", "")),
                    "as_rank": bool(st.get("as_rank", False)),
                    "weight": str(st.get("weight", "1.0")),
                    "direction_override": str(st.get("direction_override", metric.direction_default)),
                    "winsorize_low": str(st.get("winsorize_low", "")),
                    "winsorize_high": str(st.get("winsorize_high", "")),
                }
            )
        return payload

    def _apply_search_filter(self) -> None:
        self._refresh_metric_list()

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if pd.isna(out):
            return float(default)
        return float(out)

    def _apply_entry_error_style(self, fields: list[str]) -> None:
        self.threshold_entry.configure(style="Invalid.TEntry" if "threshold" in fields else "TEntry")
        self.weight_entry.configure(style="Invalid.TEntry" if "weight" in fields else "TEntry")
        self.wl_entry.configure(style="Invalid.TEntry" if "winsorize_low" in fields else "TEntry")
        self.wh_entry.configure(style="Invalid.TEntry" if "winsorize_high" in fields else "TEntry")

    def _update_preview(self) -> None:
        payload = self._collect_payload()
        rule_expr, _indicators, summary, errors = compile_indicator_payload(payload, strict=False)
        self.preview_rule_var.set(rule_expr or "(no filter rule)")
        if self.mode == "strategy":
            summary = (
                f"{summary} | {tr('sell_mode')}: {self.sell_mode_var.get().strip().upper()} "
                f"| {tr('rank_drop_mult')}: {self.rank_multiplier_var.get().strip() or '2.0'} "
                f"| {tr('na_policy')}: {self.na_policy_var.get().strip().lower() or 'fail'}"
            )
        self.preview_summary_var.set(summary)
        self.status_var.set("; ".join(errors) if errors else "")

        selected_errors: list[str] = []
        if self._selected_metric_id is not None:
            prefix = f"{self._selected_metric_id}:"
            for err in errors:
                if str(err).startswith(prefix):
                    selected_errors.append(str(err).split(":", 1)[1].strip())
        self._apply_entry_error_style(selected_errors)

    def _on_normalize_weights(self) -> None:
        ranked_ids = [mid for mid, st in self._metric_state.items() if bool(st.get("enabled")) and bool(st.get("as_rank"))]
        if not ranked_ids:
            return
        vals: dict[str, float] = {}
        total = 0.0
        for mid in ranked_ids:
            v = self._safe_float(self._metric_state[mid].get("weight"), 0.0)
            if v < 0:
                v = 0.0
            vals[mid] = v
            total += v
        if total <= 0:
            return
        for mid in ranked_ids:
            self._metric_state[mid]["weight"] = f"{(vals[mid] / total) * 100.0:.2f}"
        if self._selected_metric_id in ranked_ids:
            self._load_metric_detail(self._selected_metric_id)
        self._update_preview()

    def _apply_result(self, close_after: bool) -> None:
        payload = self._collect_payload()
        rule_expr, indicators, summary, errors = compile_indicator_payload(payload, strict=True)
        sell_mode = self.sell_mode_var.get().strip().upper() if self.mode == "strategy" else "A"
        rank_drop_multiplier = self._safe_float(self.rank_multiplier_var.get(), 2.0) if self.mode == "strategy" else 2.0
        na_policy = self.na_policy_var.get().strip().lower() if self.mode == "strategy" else "fail"
        if na_policy not in {"fail", "pass"}:
            na_policy = "fail"

        if self.mode == "strategy" and sell_mode not in {"A", "B", "C"}:
            errors.append("sell_mode")
        if self.mode == "strategy" and rank_drop_multiplier <= 0:
            errors.append("rank_drop_multiplier")

        if errors:
            messagebox.showerror(tr("rule_builder_error"), "\n".join(errors), parent=self.window)
            return

        self.result = {
            "rule_expr": rule_expr,
            "indicators": indicators,
            "summary": summary,
            "sell_mode": sell_mode,
            "rank_drop_multiplier": float(rank_drop_multiplier),
            "na_policy": na_policy,
            "preset": str(self.preset_var.get() or tr("preset_custom")),
            "preset_hint": str(self.preset_hint_var.get() or ""),
        }
        if self.on_apply is not None:
            self.on_apply(self.result)

        if close_after:
            self.window.destroy()

    def _on_apply(self, close_after: bool) -> None:
        self._apply_result(close_after=close_after)

    def _on_cancel(self) -> None:
        self.result = None
        self.window.destroy()

    def _on_reset(self, silent: bool = False) -> None:
        for metric in self.metrics:
            self._metric_state[metric.id] = {
                "enabled": False,
                "favorite": False,
                "as_filter": False,
                "filter_group": "1",
                "op": "<=",
                "threshold": "",
                "as_rank": False,
                "weight": "1.0",
                "direction_override": metric.direction_default,
                "winsorize_low": "",
                "winsorize_high": "",
            }
        if self.mode == "strategy":
            self.sell_mode_var.set("B")
            self.rank_multiplier_var.set("2.00")
            self.na_policy_var.set("fail")
        if not silent:
            self.preset_var.set(tr("preset_custom"))
            self.preset_hint_var.set("")
        self._refresh_metric_list()
        self._select_first_metric()
        self._update_preview()

    def _on_load_preset(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window,
            title=tr("load_preset"),
            filetypes=[("JSON/YAML", "*.json *.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            text = Path(path).read_text(encoding="utf-8")
            suffix = Path(path).suffix.lower()
            if suffix == ".json":
                data = json.loads(text)
            else:
                import yaml

                data = yaml.safe_load(text)
        except Exception as exc:
            messagebox.showerror(tr("load_preset_error"), str(exc), parent=self.window)
            return

        if not isinstance(data, dict):
            messagebox.showerror(tr("load_preset_error"), "invalid preset", parent=self.window)
            return

        self._on_reset(silent=True)
        indicators = data.get("indicators", [])
        if isinstance(indicators, list):
            for item in indicators:
                if not isinstance(item, dict):
                    continue
                mid = str(item.get("id", "")).strip()
                if mid not in self._metric_state:
                    continue
                st = self._metric_state[mid]
                st["enabled"] = bool(item.get("enabled", False))
                st["favorite"] = bool(item.get("favorite", False))
                st["as_filter"] = bool(item.get("as_filter", False))
                st["filter_group"] = str(item.get("filter_group", "1"))
                st["op"] = str(item.get("op", "<=")).strip() or "<="
                st["threshold"] = "" if item.get("threshold") is None else str(item.get("threshold"))
                st["as_rank"] = bool(item.get("as_rank", False))
                st["weight"] = str(item.get("weight", "1.0"))
                direction = str(item.get("direction_override", self._metric_defs[mid].direction_default)).strip().lower()
                st["direction_override"] = direction if direction in {"asc", "desc"} else self._metric_defs[mid].direction_default
                st["winsorize_low"] = "" if item.get("winsorize_low") is None else str(item.get("winsorize_low"))
                st["winsorize_high"] = "" if item.get("winsorize_high") is None else str(item.get("winsorize_high"))

        if self.mode == "strategy":
            self.sell_mode_var.set(str(data.get("sell_mode", self.sell_mode_var.get())).strip().upper() or self.sell_mode_var.get())
            self.rank_multiplier_var.set(str(data.get("rank_drop_multiplier", self.rank_multiplier_var.get())))
            self.na_policy_var.set(str(data.get("rule_na_policy", self.na_policy_var.get())).strip().lower() or self.na_policy_var.get())

        self.preset_var.set(str(data.get("preset", tr("preset_custom"))))
        self.preset_hint_var.set(str(data.get("preset_hint", "")))
        self._refresh_metric_list()
        self._select_first_metric(prefer_selected=True)
        self._update_preview()

    def _on_save_preset(self) -> None:
        default_dir = Path("strategies") / "presets"
        default_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.window,
            title=tr("save_preset"),
            initialdir=str(default_dir),
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        payload = self._collect_payload()
        _, indicators, _, _ = compile_indicator_payload(payload, strict=False)

        try:
            to_save: dict[str, Any] = {
                "preset": str(self.preset_var.get() or tr("preset_custom")),
                "preset_hint": str(self.preset_hint_var.get() or ""),
                "indicators": indicators,
            }
            if self.mode == "strategy":
                to_save["sell_mode"] = self.sell_mode_var.get().strip().upper() or "B"
                to_save["rank_drop_multiplier"] = self._safe_float(self.rank_multiplier_var.get(), 2.0)
                to_save["rule_na_policy"] = self.na_policy_var.get().strip().lower() or "fail"
            Path(path).write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror(tr("save_preset_error"), str(exc), parent=self.window)
            return

        self.status_var.set(f"{tr('saved')}: {path}")

    def show(self) -> dict[str, Any] | None:
        self.window.transient(self.parent)
        self.window.grab_set()
        self.window.focus_set()
        self.parent.wait_window(self.window)
        return self.result


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: ttk.Widget) -> None:
        super().__init__(parent)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._scroll_job: str | None = None

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.inner = ttk.Frame(self.canvas)
        self.inner_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.schedule_scrollregion_update()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        width = max(1, int(event.width))
        self.canvas.itemconfigure(self.inner_window, width=width)
        self.schedule_scrollregion_update()

    def sync_inner_width(self) -> None:
        canvas_w = max(1, int(self.canvas.winfo_width()))
        self.canvas.itemconfigure(self.inner_window, width=canvas_w)

    def schedule_scrollregion_update(self) -> None:
        if self._scroll_job is not None:
            self.after_cancel(self._scroll_job)
        self._scroll_job = self.after(80, self._update_scrollregion)

    def _update_scrollregion(self) -> None:
        self._scroll_job = None
        canvas_w = max(1, int(self.canvas.winfo_width()))
        req_w = max(1, int(self.inner.winfo_reqwidth()))
        req_h = max(1, int(self.inner.winfo_reqheight()))
        scroll_w = max(canvas_w, req_w)
        self.canvas.configure(scrollregion=(0, 0, scroll_w, req_h))
        if req_w <= canvas_w:
            self.canvas.xview_moveto(0.0)

    @staticmethod
    def _is_text_like_widget(widget: tk.Misc | None) -> bool:
        if widget is None:
            return False
        widget_class = str(widget.winfo_class())
        return widget_class in {"Entry", "TEntry", "Text", "Spinbox", "Combobox", "TCombobox"}

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        widget = event.widget if isinstance(event.widget, tk.Misc) else None
        if self._is_text_like_widget(widget):
            return None

        delta = 0
        if hasattr(event, "num") and event.num == 4:
            delta = 1
        elif hasattr(event, "num") and event.num == 5:
            delta = -1
        elif hasattr(event, "delta"):
            raw = int(event.delta)
            if raw != 0:
                delta = 1 if raw > 0 else -1

        if delta == 0:
            return None

        self.canvas.yview_scroll(-delta, "units")
        return "break"

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")


class HoverTooltip:
    def __init__(self, parent: tk.Widget) -> None:
        self.parent = parent.winfo_toplevel()
        self.win = tk.Toplevel(self.parent)
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        try:
            self.win.attributes("-alpha", 0.92)
        except tk.TclError:
            pass

        frame = tk.Frame(self.win, bg="#fffbe6", bd=1, relief="solid")
        frame.pack(fill="both", expand=True)
        self.label = tk.Label(
            frame,
            text="",
            justify="left",
            wraplength=420,
            bg="#fffbe6",
            anchor="w",
            padx=8,
            pady=6,
        )
        self.label.pack(fill="both", expand=True)
        self._visible = False
        self._text = ""

    def _calc_position(self, x_root: int, y_root: int) -> tuple[int, int]:
        self.win.update_idletasks()
        req_w = max(1, int(self.win.winfo_reqwidth()))
        req_h = max(1, int(self.win.winfo_reqheight()))

        screen_w = max(1, int(self.parent.winfo_screenwidth()))
        screen_h = max(1, int(self.parent.winfo_screenheight()))
        offset_x = 16
        offset_y = 18

        x = int(x_root) + offset_x
        y = int(y_root) + offset_y

        if x + req_w > screen_w - 8:
            x = int(x_root) - req_w - 12
        if y + req_h > screen_h - 8:
            y = int(y_root) - req_h - 12

        x = max(4, x)
        y = max(4, y)
        return x, y

    def show(self, text: str, x_root: int, y_root: int) -> None:
        if text != self._text:
            self.label.configure(text=text)
            self._text = text
            self.win.update_idletasks()
        x, y = self._calc_position(x_root=x_root, y_root=y_root)
        self.win.geometry(f"+{x}+{y}")
        if not self._visible:
            self.win.deiconify()
            self._visible = True

    def move(self, x_root: int, y_root: int) -> None:
        if not self._visible:
            return
        x, y = self._calc_position(x_root=x_root, y_root=y_root)
        self.win.geometry(f"+{x}+{y}")

    def hide(self) -> None:
        if self._visible:
            self.win.withdraw()
            self._visible = False


class ConditionSetEditor:
    def __init__(
        self,
        parent: ttk.Widget,
        *,
        title: str,
        help_text: str,
        translator: Callable[[str], str],
    ) -> None:
        self._tr = translator
        self.frame = ttk.LabelFrame(parent, text=title, padding=(8, 6, 8, 6))
        self.frame.grid_columnconfigure(0, weight=1)
        ttk.Label(self.frame, text=help_text, style="Dim.TLabel", wraplength=980, justify="left").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        self.rows_wrap = ttk.Frame(self.frame)
        self.rows_wrap.grid(row=1, column=0, sticky="ew")
        self.rows_wrap.grid_columnconfigure(0, weight=1)

        self._row_vars: dict[str, dict[str, tk.StringVar]] = {}
        self._row_enabled: dict[str, tk.BooleanVar] = {}
        self._labels = ["A", "B", "C"]
        for i, label in enumerate(self._labels):
            row = ttk.Frame(self.rows_wrap)
            row.grid(row=i, column=0, sticky="ew", pady=(0, 3))
            row.grid_columnconfigure(2, weight=1)
            en_var = tk.BooleanVar(value=False)
            self._row_enabled[label] = en_var
            ttk.Checkbutton(row, variable=en_var).grid(row=0, column=0, sticky="w", padx=(0, 4))
            ttk.Label(row, text=f"{self._tr('condition_slot')} {label}").grid(row=0, column=1, sticky="w", padx=(0, 6))
            left = tk.StringVar(value="")
            op = tk.StringVar(value="<=")
            right = tk.StringVar(value="")
            self._row_vars[label] = {"left": left, "op": op, "right": right}
            ttk.Entry(row, textvariable=left, width=38).grid(row=0, column=2, sticky="ew", padx=(0, 4))
            ttk.Combobox(
                row,
                textvariable=op,
                values=["<=", "<", ">=", ">", "==", "!="],
                width=4,
                state="readonly",
            ).grid(row=0, column=3, sticky="w", padx=(0, 4))
            ttk.Entry(row, textvariable=right, width=10).grid(row=0, column=4, sticky="w")

        logic_row = ttk.Frame(self.frame)
        logic_row.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        logic_row.grid_columnconfigure(1, weight=1)
        ttk.Label(logic_row, text=self._tr("logical_expression")).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.logical_var = tk.StringVar(value="")
        ttk.Entry(logic_row, textvariable=self.logical_var).grid(row=0, column=1, sticky="ew")
        self.message_var = tk.StringVar(value="")
        ttk.Label(self.frame, textvariable=self.message_var, style="Dim.TLabel", wraplength=960, justify="left").grid(
            row=3, column=0, sticky="ew", pady=(4, 0)
        )

    def get_condition_set(self, *, na_policy: str = "fail") -> ConditionSet:
        clauses: list[ConditionClause] = []
        for label in self._labels:
            if not bool(self._row_enabled[label].get()):
                continue
            left = str(self._row_vars[label]["left"].get() or "").strip()
            op = str(self._row_vars[label]["op"].get() or "").strip()
            right_txt = str(self._row_vars[label]["right"].get() or "").strip()
            if not left or op not in {"<", "<=", ">", ">=", "==", "!="}:
                continue
            try:
                right = float(right_txt.replace(",", ""))
            except Exception:
                continue
            clauses.append(ConditionClause(label=label, left=left, op=op, right=float(right)))
        logical = str(self.logical_var.get() or "").strip()
        if not logical and clauses:
            logical = " and ".join([c.label for c in clauses])
        return ConditionSet(conditions=clauses, logical_expression=logical, na_policy=str(na_policy or "fail"))

    def set_condition_set(self, condition_set: ConditionSet | None) -> None:
        for label in self._labels:
            self._row_enabled[label].set(False)
            self._row_vars[label]["left"].set("")
            self._row_vars[label]["op"].set("<=")
            self._row_vars[label]["right"].set("")
        self.logical_var.set("")
        if condition_set is None:
            return
        label_map = {c.label.upper(): c for c in condition_set.conditions}
        for label in self._labels:
            c = label_map.get(label)
            if c is None:
                continue
            self._row_enabled[label].set(True)
            self._row_vars[label]["left"].set(str(c.left))
            self._row_vars[label]["op"].set(str(c.op))
            self._row_vars[label]["right"].set(str(c.right))
        logic = str(condition_set.logical_expression or "").strip()
        if logic:
            # Rebuild the logical expression to only reference labels that
            # are actually available in this editor (self._labels).  If the
            # loaded strategy had more conditions (e.g. D, E) than this
            # editor supports, keeping those references intact would cause
            # validate_condition_set to raise "[logical] unknown label
            # reference: D/E" at run time.
            available = set(self._labels)
            try:
                import ast as _ast
                tree = _ast.parse(logic, mode="eval")
                referenced = {n.id.upper() for n in _ast.walk(tree) if isinstance(n, _ast.Name)}
                if referenced - available:
                    # Some labels are out of range — fall back to "A and B …"
                    # for every label that was actually loaded with a condition.
                    loaded = [lbl for lbl in self._labels if bool(self._row_enabled[lbl].get())]
                    logic = " and ".join(loaded) if loaded else ""
            except Exception:
                loaded = [lbl for lbl in self._labels if bool(self._row_enabled[lbl].get())]
                logic = " and ".join(loaded) if loaded else ""
        self.logical_var.set(logic)

    def show_message(self, text: str) -> None:
        self.message_var.set(text)


class UniverseSectorDialog:
    """업종/세부업종 선택 다이얼로그.

    - 대분류(L1)를 체크하면 중분류(L2)가 해당 대분류 항목만 필터링됩니다.
    - 포함/제외 탭으로 구분합니다.
    - 각 항목 옆에 보유 기업 수를 표시합니다 (데이터 있는 경우).
    """

    def __init__(
        self,
        parent: ttk.Widget,
        *,
        sectors: list[str],
        subsectors: list[str],
        l1_to_l2: dict[str, list[str]] | None = None,
        sector_counts: dict[str, int] | None = None,
        subsector_counts: dict[str, int] | None = None,
        include_sectors: list[str],
        exclude_sectors: list[str],
        include_subsectors: list[str],
        exclude_subsectors: list[str],
    ) -> None:
        self.parent = parent
        self.result: dict[str, list[str]] | None = None
        self._sectors = list(sectors)
        self._subsectors = list(subsectors)
        self._l1_to_l2: dict[str, list[str]] = l1_to_l2 or {}
        self._sector_counts: dict[str, int] = sector_counts or {}
        self._subsector_counts: dict[str, int] = subsector_counts or {}

        self._include_sector_vars: dict[str, tk.BooleanVar] = {
            s: tk.BooleanVar(value=s in set(include_sectors)) for s in sectors
        }
        self._exclude_sector_vars: dict[str, tk.BooleanVar] = {
            s: tk.BooleanVar(value=s in set(exclude_sectors)) for s in sectors
        }
        self._include_subsector_vars: dict[str, tk.BooleanVar] = {
            s: tk.BooleanVar(value=s in set(include_subsectors)) for s in subsectors
        }
        self._exclude_subsector_vars: dict[str, tk.BooleanVar] = {
            s: tk.BooleanVar(value=s in set(exclude_subsectors)) for s in subsectors
        }

        # Track selected L1 sectors for cascade filtering
        self._selected_l1_for_filter: set[str] = set()

        self.top = tk.Toplevel(parent.winfo_toplevel())
        self.top.title("업종/세부업종 선택")
        self.top.geometry("1000x680")
        self.top.transient(parent.winfo_toplevel())
        self.top.grab_set()
        self.top.minsize(700, 480)

        self._build_ui()
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)
        self._refresh_subsector_list()
        self._refresh_count_label()

    def _build_ui(self) -> None:
        wrap = ttk.Frame(self.top, padding=(10, 8, 10, 8))
        wrap.pack(fill="both", expand=True)
        wrap.grid_columnconfigure(0, weight=2)
        wrap.grid_columnconfigure(1, weight=3)
        wrap.grid_rowconfigure(1, weight=1)

        # Header hint
        ttk.Label(
            wrap,
            text="대분류를 선택하면 중분류가 자동으로 필터링됩니다. 포함(Include)/제외(Exclude) 탭에서 설정하세요.",
            style="Dim.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        # Left: L1 sectors
        left = ttk.LabelFrame(wrap, text="대분류 (L1)", padding=(8, 6, 8, 6))
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        # Tab notebook for include/exclude on left
        self._left_tabs = ttk.Notebook(left)
        self._left_tabs.grid(row=1, column=0, sticky="nsew")
        left_inc_tab = ttk.Frame(self._left_tabs)
        left_exc_tab = ttk.Frame(self._left_tabs)
        self._left_tabs.add(left_inc_tab, text="포함 (Include)")
        self._left_tabs.add(left_exc_tab, text="제외 (Exclude)")

        self._sector_inc_canvas, self._sector_inc_inner = self._make_scrollable(left_inc_tab)
        self._sector_exc_canvas, self._sector_exc_inner = self._make_scrollable(left_exc_tab)
        self._sector_inc_rows: dict[str, ttk.Frame] = {}
        self._sector_exc_rows: dict[str, ttk.Frame] = {}

        for s in self._sectors:
            self._sector_inc_rows[s] = self._add_sector_row(
                self._sector_inc_inner, s,
                check_var=self._include_sector_vars[s],
                count=self._sector_counts.get(s),
                on_change=self._refresh_subsector_list,
            )
            self._sector_exc_rows[s] = self._add_sector_row(
                self._sector_exc_inner, s,
                check_var=self._exclude_sector_vars[s],
                count=self._sector_counts.get(s),
            )

        # Right: L2 subsectors (cascaded)
        right = ttk.LabelFrame(wrap, text="중분류 (L2)", padding=(8, 6, 8, 6))
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=0)
        right.grid_rowconfigure(1, weight=1)

        # Filter hint for right pane
        self._cascade_hint_var = tk.StringVar(value="대분류 전체 표시")
        ttk.Label(right, textvariable=self._cascade_hint_var, style="Dim.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )

        self._right_tabs = ttk.Notebook(right)
        self._right_tabs.grid(row=1, column=0, sticky="nsew")
        right_inc_tab = ttk.Frame(self._right_tabs)
        right_exc_tab = ttk.Frame(self._right_tabs)
        self._right_tabs.add(right_inc_tab, text="포함 (Include)")
        self._right_tabs.add(right_exc_tab, text="제외 (Exclude)")

        self._subsector_inc_canvas, self._subsector_inc_inner = self._make_scrollable(right_inc_tab)
        self._subsector_exc_canvas, self._subsector_exc_inner = self._make_scrollable(right_exc_tab)
        self._subsector_inc_rows: dict[str, ttk.Frame] = {}
        self._subsector_exc_rows: dict[str, ttk.Frame] = {}

        for s in self._subsectors:
            self._subsector_inc_rows[s] = self._add_sector_row(
                self._subsector_inc_inner, s,
                check_var=self._include_subsector_vars[s],
                count=self._subsector_counts.get(s),
            )
            self._subsector_exc_rows[s] = self._add_sector_row(
                self._subsector_exc_inner, s,
                check_var=self._exclude_subsector_vars[s],
                count=self._subsector_counts.get(s),
            )

        # Bottom bar
        bot = ttk.Frame(wrap)
        bot.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bot.grid_columnconfigure(2, weight=1)

        ttk.Button(bot, text="대분류 전체선택", command=self._select_all_sectors).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(bot, text="전체해제", command=self._clear_all).grid(row=0, column=1, padx=(0, 12))
        self._count_var = tk.StringVar(value="")
        ttk.Label(bot, textvariable=self._count_var, style="Dim.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Button(bot, text="적용", command=self._apply).grid(row=0, column=3, padx=(4, 0))
        ttk.Button(bot, text="취소", command=self._cancel).grid(row=0, column=4, padx=(4, 0))

    @staticmethod
    def _make_scrollable(parent: ttk.Frame) -> tuple[tk.Canvas, ttk.Frame]:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        vbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        inner = ttk.Frame(canvas)
        wid = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _cfg_inner(_e: tk.Event) -> None:
            canvas.configure(scrollregion=(0, 0, max(inner.winfo_reqwidth(), canvas.winfo_width()), inner.winfo_reqheight()))

        def _cfg_canvas(e: tk.Event) -> None:
            canvas.itemconfigure(wid, width=e.width)

        inner.bind("<Configure>", _cfg_inner)
        canvas.bind("<Configure>", _cfg_canvas)
        return canvas, inner

    def _add_sector_row(
        self,
        parent: ttk.Frame,
        label: str,
        *,
        check_var: tk.BooleanVar,
        count: int | None = None,
        on_change: Any = None,
    ) -> ttk.Frame:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=2, pady=1)
        cmd = on_change if on_change else lambda: None
        cb = ttk.Checkbutton(row, variable=check_var, command=cmd)
        cb.pack(side="left")
        lbl_text = label
        if count is not None:
            lbl_text = f"{label}  ({count}개)"
        lbl = ttk.Label(row, text=lbl_text, cursor="hand2")
        lbl.pack(side="left", padx=(4, 0))
        lbl.bind("<Button-1>", lambda _e: (check_var.set(not bool(check_var.get())), cmd()))
        return row

    def _refresh_subsector_list(self) -> None:
        """Show only subsectors matching selected L1 sectors (cascade)."""
        selected_l1 = {s for s, v in self._include_sector_vars.items() if bool(v.get())}
        selected_l1 |= {s for s, v in self._exclude_sector_vars.items() if bool(v.get())}

        if selected_l1 and self._l1_to_l2:
            allowed_l2: set[str] = set()
            for l1 in selected_l1:
                allowed_l2.update(self._l1_to_l2.get(l1, []))
            hint = f"대분류 선택됨 ({len(selected_l1)}개) → 중분류 {len(allowed_l2)}개 표시"
        else:
            allowed_l2 = set(self._subsectors)
            hint = "대분류 전체 표시"

        self._cascade_hint_var.set(hint)

        for s, row in self._subsector_inc_rows.items():
            if not allowed_l2 or s in allowed_l2:
                row.pack(fill="x", padx=2, pady=1)
            else:
                row.pack_forget()
        for s, row in self._subsector_exc_rows.items():
            if not allowed_l2 or s in allowed_l2:
                row.pack(fill="x", padx=2, pady=1)
            else:
                row.pack_forget()

        self._refresh_count_label()

    def _refresh_count_label(self) -> None:
        inc_s = sum(1 for v in self._include_sector_vars.values() if bool(v.get()))
        exc_s = sum(1 for v in self._exclude_sector_vars.values() if bool(v.get()))
        inc_ss = sum(1 for v in self._include_subsector_vars.values() if bool(v.get()))
        exc_ss = sum(1 for v in self._exclude_subsector_vars.values() if bool(v.get()))

        # Estimate total companies included
        total = 0
        if self._sector_counts and inc_s:
            inc_names = {s for s, v in self._include_sector_vars.items() if bool(v.get())}
            total = sum(self._sector_counts.get(s, 0) for s in inc_names)

        parts = []
        if inc_s:
            parts.append(f"대분류 포함 {inc_s}개")
        if exc_s:
            parts.append(f"제외 {exc_s}개")
        if inc_ss:
            parts.append(f"중분류 포함 {inc_ss}개")
        if exc_ss:
            parts.append(f"제외 {exc_ss}개")
        if total:
            parts.append(f"약 {total}개 기업")
        self._count_var.set("  |  ".join(parts) if parts else "미선택")

    def _select_all_sectors(self) -> None:
        for v in self._include_sector_vars.values():
            v.set(True)
        self._refresh_subsector_list()

    def _clear_all(self) -> None:
        for d in [
            self._include_sector_vars,
            self._exclude_sector_vars,
            self._include_subsector_vars,
            self._exclude_subsector_vars,
        ]:
            for v in d.values():
                v.set(False)
        self._refresh_subsector_list()

    @staticmethod
    def _collect(vars_map: dict[str, tk.BooleanVar]) -> list[str]:
        return [k for k, v in vars_map.items() if bool(v.get())]

    def _apply(self) -> None:
        self.result = {
            "include_sectors": self._collect(self._include_sector_vars),
            "exclude_sectors": self._collect(self._exclude_sector_vars),
            "include_subsectors": self._collect(self._include_subsector_vars),
            "exclude_subsectors": self._collect(self._exclude_subsector_vars),
        }
        self.top.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.top.destroy()

    def show(self) -> dict[str, list[str]] | None:
        self.top.wait_window()
        return self.result


class BacktestTab:
    def __init__(self, parent: ttk.Frame) -> None:
        self.parent = parent
        self.screen_result = None
        self.strategy_result = None
        self.compare_result: dict[str, pd.DataFrame] | None = None
        self.funding_compare_results: dict[str, Any] | None = None
        self._funding_compare_results_by_mode: dict[str, Any] | None = None
        self._last_screen_cfg: dict[str, Any] | None = None
        self._last_strategy_cfg: dict[str, Any] | None = None
        self._last_funding_variants: list[dict[str, Any]] | None = None

        self._is_syncing_screen_rule = False
        self._running = False
        self._current_indicators: list[dict[str, Any]] = []
        self._buy_condition_set: ConditionSet = ConditionSet()
        self._sell_condition_set: ConditionSet = ConditionSet()
        self._strategy_builder_payload: dict[str, Any] = {
            "rule_expr": "",
            "indicators": [],
            "summary": tr("builder_summary_default"),
            "sell_mode": "B",
            "rank_drop_multiplier": 2.0,
            "na_policy": "fail",
        }
        self._suspend_trade_filter_trace = False
        self._responsive_job: str | None = None
        self._heatmap_resize_job: str | None = None
        self._heatmap_scroll_job: str | None = None
        self._heatmap_hover_job: str | None = None
        self._heatmap_hover_pending_key: tuple[int, int] | None = None
        self._heatmap_hover_pending_pointer: tuple[int, int] | None = None
        self._heatmap_hover_last_key: tuple[int, int] | None = None
        self._last_header_cols: int | None = None
        self._last_chart_size: tuple[int, ...] | None = None
        self._funding_parse_warnings: list[str] = []
        self._heatmap_ax = None
        self._heatmap_rows: list[str] = []
        self._heatmap_cols: list[str] = []
        self._heatmap_values = np.empty((0, 0), dtype=float)
        self._heatmap_trades = pd.DataFrame()

        self._build()

    def _build(self) -> None:
        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_columnconfigure(0, weight=1)

        # Sub-notebook: 설정 / 결과 tabs
        self.sub_nb = ttk.Notebook(self.parent)
        self.sub_nb.grid(row=0, column=0, sticky="nsew")

        # --- Settings tab ---
        _settings_tab = ttk.Frame(self.sub_nb)
        _settings_tab.grid_rowconfigure(0, weight=1)
        _settings_tab.grid_columnconfigure(0, weight=1)
        self.sub_nb.add(_settings_tab, text=tr("tab_settings"))

        self.scrollable = ScrollableFrame(_settings_tab)
        self.scrollable.grid(row=0, column=0, sticky="nsew")

        self.root_frame = self.scrollable.inner
        self.root_frame.grid_columnconfigure(0, weight=1)
        self.root_frame.grid_rowconfigure(0, weight=0)

        # --- Results tab ---
        _results_tab = ttk.Frame(self.sub_nb)
        _results_tab.grid_rowconfigure(1, weight=1)
        _results_tab.grid_columnconfigure(0, weight=1)
        self.sub_nb.add(_results_tab, text=tr("tab_results"))

        # Progress bar (row 0 of results tab)
        self._run_progress_frame = ttk.Frame(_results_tab, padding=(8, 4, 8, 2))
        self._run_progress_frame.grid(row=0, column=0, sticky="ew")
        self._run_progress_frame.grid_columnconfigure(1, weight=1)
        self._run_progress_pct = tk.DoubleVar(value=0.0)
        self._run_progress_status = tk.StringVar(value=tr("progress_idle"))
        ttk.Label(self._run_progress_frame, textvariable=self._run_progress_status, width=22).grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        self._run_progress_bar = ttk.Progressbar(
            self._run_progress_frame,
            variable=self._run_progress_pct,
            maximum=100.0,
            mode="determinate",
            length=300,
        )
        self._run_progress_bar.grid(row=0, column=1, sticky="ew")

        # Results scrollable (row 1 of results tab)
        self._results_scrollable = ScrollableFrame(_results_tab)
        self._results_scrollable.grid(row=1, column=0, sticky="nsew")
        self._results_inner = self._results_scrollable.inner
        self._results_inner.grid_columnconfigure(0, weight=1)
        self._results_inner.grid_rowconfigure(0, weight=0)   # strategy preview
        self._results_inner.grid_rowconfigure(1, weight=0)   # kpi frame
        self._results_inner.grid_rowconfigure(2, weight=3, minsize=560)  # charts
        self._results_inner.grid_rowconfigure(3, weight=2, minsize=420)  # trade analytics
        self._results_inner.grid_rowconfigure(4, weight=2, minsize=340)  # tables

        self.header_frame = ttk.Frame(self.root_frame, padding=(10, 8, 10, 8))
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.strategy_path_var = tk.StringVar(value="")
        self.strategy_source_var = tk.StringVar(value=tr("strategy_source_file"))
        self.screen_rule_var = tk.StringVar(value="")

        self.data_source_var = tk.StringVar(value="offline")
        self.universe_input_mode_var = tk.StringVar(value="preset")
        self.universe_direct_tickers_var = tk.StringVar(value="")
        self.universe_financial_only_var = tk.BooleanVar(value=False)

        self.market_var = tk.StringVar(value="us")
        self.start_var = tk.StringVar(value="2000-01-01")
        self.end_var = tk.StringVar(value="")
        self.freq_var = tk.StringVar(value="Q")
        self.holdings_var = tk.StringVar(value="3")
        self.benchmark_var = tk.StringVar(value="SPY")
        self.exec_timing_var = tk.StringVar(value="next_open")

        self.initial_cash_var = tk.StringVar(value="1000000")
        self.commission_bps_var = tk.StringVar(value="0")
        self.slippage_bps_var = tk.StringVar(value="0")
        self.share_mode_var = tk.StringVar(value=tr("share_mode_fractional"))
        self.cash_buffer_pct_var = tk.StringVar(value="0")
        self.position_weight_pct_var = tk.StringVar(value="")
        self.max_holdings_cfg_var = tk.StringVar(value="")
        self.max_new_buys_var = tk.StringVar(value="")
        self.max_buy_amount_var = tk.StringVar(value="")
        self.min_cash_reserve_pct_var = tk.StringVar(value="0")
        self.exec_price_basis_var = tk.StringVar(value=tr("price_basis_auto"))
        self.price_offset_pct_var = tk.StringVar(value="0")
        self.universe_exchanges_var = tk.StringVar(value="")
        self.universe_size_buckets_var = tk.StringVar(value="")
        self.universe_include_sectors_var = tk.StringVar(value="")
        self.universe_exclude_sectors_var = tk.StringVar(value="")
        self.universe_include_subsectors_var = tk.StringVar(value="")
        self.universe_exclude_subsectors_var = tk.StringVar(value="")
        self.universe_exclude_tickers_var = tk.StringVar(value="")
        self.universe_watchlist_name_var = tk.StringVar(value="")
        self.stock_type_included_var = tk.BooleanVar(value=True)
        self.ptp_included_var = tk.BooleanVar(value=False)
        self.universe_preset_vars: dict[str, tk.BooleanVar] = {
            "sp500": tk.BooleanVar(value=False),
            "sp500_pit": tk.BooleanVar(value=False),
            "nasdaq_stock_only_financial": tk.BooleanVar(value=False),
            "nasdaq_all": tk.BooleanVar(value=False),
            "nyse_all": tk.BooleanVar(value=False),
        }
        self.universe_bucket_vars: dict[str, dict[str, tk.BooleanVar]] = {
            "NASDAQ": {b: tk.BooleanVar(value=False) for b in UNIVERSE_BUCKETS},
            "NYSE": {b: tk.BooleanVar(value=False) for b in UNIVERSE_BUCKETS},
        }
        self.universe_parent_status_vars: dict[str, tk.StringVar] = {
            "NASDAQ": tk.StringVar(value=""),
            "NYSE": tk.StringVar(value=""),
        }
        self.use_fundamentals_pit_var = tk.BooleanVar(value=True)
        self.strict_pit_var = tk.BooleanVar(value=False)
        self.funding_mode_var = tk.StringVar(value=tr("funding_mode_lump"))
        self.funding_contrib_freq_var = tk.StringVar(value=tr("funding_freq_monthly"))
        self.funding_dca_var = tk.StringVar(value="1000000")
        self.funding_va_step_var = tk.StringVar(value="1000000")
        self.funding_va_min_var = tk.StringVar(value="500000")
        self.funding_va_max_var = tk.StringVar(value="2000000")
        self.funding_va_min_policy_var = tk.StringVar(value=tr("va_min_policy_every_rebalance"))
        self.funding_va_withdraw_var = tk.BooleanVar(value=False)
        self.funding_compare_basis_var = tk.StringVar(value=tr("funding_compare_align_dca"))
        self.funding_custom_total_var = tk.StringVar(value="")
        self.funding_preview_var = tk.StringVar(value="")
        self.funding_view_var = tk.StringVar(value=tr("funding_mode_lump"))
        self.funding_flow_view_var = tk.StringVar(value=tr("funding_flow_view_summary"))
        self.funding_compare_meta: dict[str, Any] = {}

        self.override_strategy_var = tk.BooleanVar(value=False)
        self.lower_chart_mode_var = tk.StringVar(value=tr("chart_mode_drawdown"))
        self.table_source_var = tk.StringVar(value=tr("source_auto"))
        self.show_contributed_var = tk.BooleanVar(value=False)
        self.show_advanced_var = tk.BooleanVar(value=False)

        self.trade_top_n_var = tk.StringVar(value="20")
        self.trade_scope_var = tk.StringVar(value=tr("scope_all"))
        self.trade_metric_var = tk.StringVar(value=tr("metric_net_notional"))
        self.trade_sort_var = tk.StringVar(value=tr("sort_notional"))
        self.trade_filter_month_var = tk.StringVar(value=tr("all"))
        self.trade_filter_ticker_var = tk.StringVar(value=tr("all"))
        self.return_plot_mode_var = tk.StringVar(value=tr("ret_mode_strat_bench"))
        self.cum_line_mode_var = tk.StringVar(value=tr("cum_mode_strategy"))
        self.show_tooltip_var = tk.BooleanVar(value=True)
        self.sell_mode_ui_var = tk.StringVar(value=tr("sell_mode_b"))
        self.rank_drop_mult_ui_var = tk.StringVar(value="2.0")
        self.hold_days_var = tk.StringVar(value="")
        self.buy_section_open_var = tk.BooleanVar(value=False)
        self.sell_section_open_var = tk.BooleanVar(value=False)
        self.universe_section_open_var = tk.BooleanVar(value=False)

        self.strategy_section = ttk.LabelFrame(self.header_frame, text=tr("section_strategy"), padding=(8, 6, 8, 6))
        self.strategy_section.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.strategy_section.grid_columnconfigure(0, weight=1)

        self.backtest_section = ttk.LabelFrame(self.header_frame, text=tr("section_backtest"), padding=(8, 6, 8, 6))
        self.backtest_section.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.backtest_section.grid_columnconfigure(0, weight=1)

        self.buy_section = ttk.LabelFrame(self.header_frame, text=tr("section_buy"), padding=(8, 6, 8, 6))
        self.buy_section.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.buy_section.grid_columnconfigure(0, weight=1)

        self.sell_section = ttk.LabelFrame(self.header_frame, text=tr("section_sell"), padding=(8, 6, 8, 6))
        self.sell_section.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.sell_section.grid_columnconfigure(0, weight=1)

        self.universe_section = ttk.LabelFrame(self.header_frame, text=tr("section_universe"), padding=(8, 6, 8, 6))
        self.universe_section.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        self.universe_section.grid_columnconfigure(0, weight=1)

        self.funding_section = ttk.LabelFrame(self.header_frame, text=tr("section_funding"), padding=(8, 6, 8, 6))
        self.funding_section.grid(row=5, column=0, sticky="ew", pady=(0, 6))
        self.funding_section.grid_columnconfigure(0, weight=1)

        self.action_section = ttk.LabelFrame(self.header_frame, text=tr("section_actions"), padding=(8, 6, 8, 6))
        self.action_section.grid(row=6, column=0, sticky="ew", pady=(0, 6))
        self.action_section.grid_columnconfigure(0, weight=1)

        self.summary_section = ttk.LabelFrame(self.header_frame, text=tr("section_summary"), padding=(8, 6, 8, 6))
        self.summary_section.grid(row=7, column=0, sticky="ew", pady=(0, 6))
        self.summary_section.grid_columnconfigure(0, weight=1)

        # Strategy section
        self.strategy_file_item = ttk.Frame(self.strategy_section)
        ttk.Label(self.strategy_file_item, text=tr("strategy_config")).pack(side="left")
        self.strategy_path_entry = ttk.Entry(self.strategy_file_item, textvariable=self.strategy_path_var, width=24)
        self.strategy_path_entry.pack(side="left", padx=(4, 4))
        self.strategy_browse_btn = ttk.Button(self.strategy_file_item, text=tr("browse"), command=self._browse_strategy)
        self.strategy_browse_btn.pack(side="left")

        self.strategy_source_item = ttk.Frame(self.strategy_section)
        ttk.Label(self.strategy_source_item, text=tr("strategy_source")).pack(side="left")
        self.strategy_source_combo = ttk.Combobox(
            self.strategy_source_item,
            textvariable=self.strategy_source_var,
            values=[tr("strategy_source_file"), tr("strategy_source_builder")],
            width=6,
            state="readonly",
        )
        self.strategy_source_combo.pack(side="left", padx=(4, 6))
        self.strategy_builder_btn = ttk.Button(self.strategy_source_item, text=tr("strategy_builder"), command=self._open_strategy_builder)
        self.strategy_builder_btn.pack(side="left", padx=(0, 4))
        self.strategy_save_btn = ttk.Button(self.strategy_source_item, text=tr("strategy_builder_save"), command=self._save_strategy_builder_to_file)
        self.strategy_save_btn.pack(side="left")

        self.screen_rule_item = ttk.Frame(self.strategy_section)
        ttk.Label(self.screen_rule_item, text=tr("screen_rule")).pack(side="left")
        self.screen_rule_entry = ttk.Entry(self.screen_rule_item, textvariable=self.screen_rule_var, width=42)
        self.screen_rule_entry.pack(side="left", padx=(4, 4), fill="x", expand=True)
        self.rule_builder_btn = ttk.Button(self.screen_rule_item, text=tr("rule_builder"), command=self._open_rule_builder)
        self.rule_builder_btn.pack(side="left", padx=(0, 4))
        self.validate_conditions_btn = ttk.Button(
            self.screen_rule_item,
            text=tr("validate_conditions"),
            command=self._validate_conditions,
        )
        self.validate_conditions_btn.pack(side="left", padx=(0, 4))
        self.save_conditions_btn = ttk.Button(
            self.screen_rule_item,
            text=tr("save_conditions"),
            command=self._save_conditions_to_file,
        )
        self.save_conditions_btn.pack(side="left", padx=(0, 4))
        self.load_conditions_btn = ttk.Button(
            self.screen_rule_item,
            text=tr("load_conditions"),
            command=self._load_conditions_from_file,
        )
        self.load_conditions_btn.pack(side="left", padx=(0, 4))
        self.builder_state_var = tk.StringVar(value="")
        self.builder_state_label = ttk.Label(self.screen_rule_item, textvariable=self.builder_state_var)
        self.builder_state_label.pack(side="left")

        self.data_source_item = ttk.Frame(self.strategy_section)
        ttk.Label(self.data_source_item, text="데이터 소스").pack(side="left")
        ttk.Radiobutton(
            self.data_source_item, text="오프라인 (로컬 저장)", variable=self.data_source_var, value="offline"
        ).pack(side="left", padx=(4, 0))
        ttk.Radiobutton(
            self.data_source_item, text="온라인 (최신 다운로드)", variable=self.data_source_var, value="online"
        ).pack(side="left", padx=(4, 0))

        self._strategy_items: list[tuple[ttk.Frame, bool]] = [
            (self.data_source_item, False),
            (self.strategy_file_item, False),
            (self.strategy_source_item, False),
            (self.screen_rule_item, True),
        ]

        # Buy section
        self.buy_header_row = ttk.Frame(self.buy_section)
        self.buy_header_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.buy_header_row.grid_columnconfigure(0, weight=1)
        ttk.Label(self.buy_header_row, text=tr("buy_condition_help"), style="Dim.TLabel", wraplength=980, justify="left").grid(
            row=0, column=0, sticky="w"
        )
        self.buy_toggle_btn = ttk.Checkbutton(
            self.buy_header_row,
            text=tr("advanced_on"),
            variable=self.buy_section_open_var,
            command=self._update_buy_section_visibility,
        )
        self.buy_toggle_btn.grid(row=0, column=1, sticky="e")

        self.buy_body_frame = ttk.Frame(self.buy_section)
        self.buy_body_frame.grid(row=1, column=0, sticky="ew")
        self.buy_body_frame.grid_columnconfigure(0, weight=3)
        self.buy_body_frame.grid_columnconfigure(1, weight=2)

        self.buy_condition_editor = ConditionSetEditor(
            self.buy_body_frame,
            title=tr("filters"),
            help_text=tr("buy_condition_help"),
            translator=tr,
        )
        self.buy_condition_editor.frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.buy_condition_btn_row = ttk.Frame(self.buy_body_frame)
        self.buy_condition_btn_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(self.buy_condition_btn_row, text=tr("validate_conditions"), command=self._validate_conditions).pack(side="left", padx=(0, 4))
        ttk.Button(self.buy_condition_btn_row, text=tr("save_conditions"), command=self._save_conditions_to_file).pack(side="left", padx=(0, 4))
        ttk.Button(self.buy_condition_btn_row, text=tr("load_conditions"), command=self._load_conditions_from_file).pack(side="left", padx=(0, 4))
        ttk.Button(self.buy_condition_btn_row, text=tr("condition_reset"), command=self._reset_buy_conditions).pack(side="left")

        self.buy_hint_label = ttk.Label(
            self.buy_body_frame,
            text="함수 예시: sma({close},20), percentile({roe}), zscore({fcf_yield})",
            style="Dim.TLabel",
            wraplength=560,
            justify="left",
        )
        self.buy_hint_label.grid(row=2, column=0, sticky="w", pady=(4, 0))

        self.buy_right_frame = ttk.Frame(self.buy_body_frame)
        self.buy_right_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")
        self.buy_right_frame.grid_columnconfigure(0, weight=1)

        self.position_sizing_panel = ttk.LabelFrame(self.buy_right_frame, text="매수 비중/리스크", padding=(6, 4, 6, 4))
        self.position_sizing_panel.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.position_sizing_panel.grid_columnconfigure(1, weight=1)
        self._render_position_sizing_panel(self.position_sizing_panel)

        self.execution_panel = ttk.LabelFrame(self.buy_right_frame, text="매수/매도 체결 규칙", padding=(6, 4, 6, 4))
        self.execution_panel.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.execution_panel.grid_columnconfigure(1, weight=1)
        self._render_execution_rules_panel(self.execution_panel)

        self.asset_alloc_panel = ttk.LabelFrame(self.buy_right_frame, text="자산배분 옵션", padding=(6, 4, 6, 4))
        self.asset_alloc_panel.grid(row=2, column=0, sticky="ew")
        ttk.Label(self.asset_alloc_panel, text=tr("todo_not_connected"), style="Dim.TLabel").pack(anchor="w")

        # Sell section
        self.sell_header_row = ttk.Frame(self.sell_section)
        self.sell_header_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.sell_header_row.grid_columnconfigure(0, weight=1)
        ttk.Label(self.sell_header_row, text=tr("sell_condition_help"), style="Dim.TLabel", wraplength=980, justify="left").grid(
            row=0, column=0, sticky="w"
        )
        self.sell_toggle_btn = ttk.Checkbutton(
            self.sell_header_row,
            text=tr("advanced_on"),
            variable=self.sell_section_open_var,
            command=self._update_sell_section_visibility,
        )
        self.sell_toggle_btn.grid(row=0, column=1, sticky="e")

        self.sell_body_frame = ttk.Frame(self.sell_section)
        self.sell_body_frame.grid(row=1, column=0, sticky="ew")
        self.sell_body_frame.grid_columnconfigure(0, weight=3)
        self.sell_body_frame.grid_columnconfigure(1, weight=2)

        self.sell_condition_editor = ConditionSetEditor(
            self.sell_body_frame,
            title=tr("filters"),
            help_text=tr("sell_condition_help"),
            translator=tr,
        )
        self.sell_condition_editor.frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.sell_condition_btn_row = ttk.Frame(self.sell_body_frame)
        self.sell_condition_btn_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(self.sell_condition_btn_row, text=tr("validate_conditions"), command=self._validate_conditions).pack(side="left", padx=(0, 4))
        ttk.Button(self.sell_condition_btn_row, text=tr("condition_reset"), command=self._reset_sell_conditions).pack(side="left")

        self.sell_right_panel = ttk.LabelFrame(self.sell_body_frame, text="매도 규칙 보조 설정", padding=(6, 4, 6, 4))
        self.sell_right_panel.grid(row=0, column=1, rowspan=2, sticky="nsew")
        self.sell_right_panel.grid_columnconfigure(1, weight=1)
        ttk.Label(self.sell_right_panel, text=tr("sell_mode_ui")).grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Combobox(
            self.sell_right_panel,
            textvariable=self.sell_mode_ui_var,
            values=[tr("sell_mode_a"), tr("sell_mode_b"), tr("sell_mode_c")],
            width=16,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", pady=2)
        ttk.Label(self.sell_right_panel, text=tr("rank_drop_mult")).grid(row=1, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(self.sell_right_panel, textvariable=self.rank_drop_mult_ui_var, width=10).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(self.sell_right_panel, text=tr("hold_days")).grid(row=2, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(self.sell_right_panel, textvariable=self.hold_days_var, width=10).grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(self.sell_right_panel, text=tr("todo_not_connected"), style="Dim.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Universe section
        self.universe_header_row = ttk.Frame(self.universe_section)
        self.universe_header_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.universe_header_row.grid_columnconfigure(0, weight=1)
        ttk.Label(self.universe_header_row, text=tr("universe_count_hint"), style="Dim.TLabel", wraplength=980, justify="left").grid(
            row=0, column=0, sticky="w"
        )
        self.universe_toggle_btn = ttk.Checkbutton(
            self.universe_header_row,
            text=tr("advanced_on"),
            variable=self.universe_section_open_var,
            command=self._update_universe_section_visibility,
        )
        self.universe_toggle_btn.grid(row=0, column=1, sticky="e")

        self.universe_body_frame = ttk.Frame(self.universe_section)
        self.universe_body_frame.grid(row=1, column=0, sticky="ew")
        self.universe_body_frame.grid_columnconfigure(0, weight=1)
        self.universe_body_frame.grid_columnconfigure(1, weight=1)

        self._render_universe_filter_panel(self.universe_body_frame)

        # Backtest section
        self.backtest_toggle_row = ttk.Frame(self.backtest_section)
        self.backtest_toggle_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.backtest_toggle_row.grid_columnconfigure(1, weight=1)
        self.advanced_toggle_btn = ttk.Checkbutton(
            self.backtest_toggle_row,
            text=tr("advanced_off"),
            variable=self.show_advanced_var,
            command=self._update_advanced_visibility,
        )
        self.advanced_toggle_btn.grid(row=0, column=0, sticky="w")

        self.backtest_core_frame = ttk.Frame(self.backtest_section)
        self.backtest_core_frame.grid(row=1, column=0, sticky="ew")
        self.backtest_core_frame.grid_columnconfigure(0, weight=1)

        self.backtest_market_item = ttk.Frame(self.backtest_core_frame)
        ttk.Label(self.backtest_market_item, text=tr("market")).pack(side="left")
        self.market_combo = ttk.Combobox(self.backtest_market_item, textvariable=self.market_var, values=["us", "kr", "auto"], width=6, state="readonly")
        self.market_combo.pack(side="left", padx=(4, 0))

        self.backtest_start_item = ttk.Frame(self.backtest_core_frame)
        ttk.Label(self.backtest_start_item, text=tr("start")).pack(side="left")
        ttk.Entry(self.backtest_start_item, textvariable=self.start_var, width=11).pack(side="left", padx=(4, 0))

        self.backtest_end_item = ttk.Frame(self.backtest_core_frame)
        ttk.Label(self.backtest_end_item, text=tr("end")).pack(side="left")
        ttk.Entry(self.backtest_end_item, textvariable=self.end_var, width=11).pack(side="left", padx=(4, 0))

        self.backtest_freq_item = ttk.Frame(self.backtest_core_frame)
        ttk.Label(self.backtest_freq_item, text=tr("freq")).pack(side="left")
        self.freq_combo = ttk.Combobox(self.backtest_freq_item, textvariable=self.freq_var, values=["M", "Q", "Y"], width=4, state="readonly")
        self.freq_combo.pack(side="left", padx=(4, 0))

        self.backtest_holdings_item = ttk.Frame(self.backtest_core_frame)
        ttk.Label(self.backtest_holdings_item, text=tr("holdings")).pack(side="left")
        ttk.Entry(self.backtest_holdings_item, textvariable=self.holdings_var, width=6).pack(side="left", padx=(4, 0))

        self.backtest_bench_item = ttk.Frame(self.backtest_core_frame)
        ttk.Label(self.backtest_bench_item, text=tr("benchmark")).pack(side="left")
        ttk.Entry(self.backtest_bench_item, textvariable=self.benchmark_var, width=8).pack(side="left", padx=(4, 0))

        self._backtest_core_items: list[tuple[ttk.Frame, bool]] = [
            (self.backtest_market_item, False),
            (self.backtest_start_item, False),
            (self.backtest_end_item, False),
            (self.backtest_freq_item, False),
            (self.backtest_holdings_item, False),
            (self.backtest_bench_item, False),
        ]

        self.backtest_advanced_frame = ttk.Frame(self.backtest_section)
        self.backtest_advanced_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.backtest_advanced_frame.grid_columnconfigure(0, weight=1)

        self.backtest_exec_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_exec_item, text=tr("exec")).pack(side="left")
        self.exec_combo = ttk.Combobox(self.backtest_exec_item, textvariable=self.exec_timing_var, values=["next_open", "same_close"], width=11, state="readonly")
        self.exec_combo.pack(side="left", padx=(4, 0))

        self.backtest_comm_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_comm_item, text=tr("commission_bps")).pack(side="left")
        ttk.Entry(self.backtest_comm_item, textvariable=self.commission_bps_var, width=7).pack(side="left", padx=(4, 0))

        self.backtest_slip_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_slip_item, text=tr("slippage_bps")).pack(side="left")
        ttk.Entry(self.backtest_slip_item, textvariable=self.slippage_bps_var, width=7).pack(side="left", padx=(4, 0))

        self.backtest_share_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_share_item, text=tr("share_mode")).pack(side="left")
        self.share_mode_combo = ttk.Combobox(
            self.backtest_share_item,
            textvariable=self.share_mode_var,
            values=[tr("share_mode_fractional"), tr("share_mode_integer")],
            width=10,
            state="readonly",
        )
        self.share_mode_combo.pack(side="left", padx=(4, 0))

        self.backtest_cashbuf_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_cashbuf_item, text=tr("cash_buffer_pct")).pack(side="left")
        ttk.Entry(self.backtest_cashbuf_item, textvariable=self.cash_buffer_pct_var, width=7).pack(side="left", padx=(4, 0))

        self.backtest_override_item = ttk.Frame(self.backtest_advanced_frame)
        self.override_check = ttk.Checkbutton(self.backtest_override_item, text=tr("override"), variable=self.override_strategy_var)
        self.override_check.pack(side="left")

        self.backtest_lower_chart_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_lower_chart_item, text=tr("lower_chart")).pack(side="left")
        self.lower_chart_combo = ttk.Combobox(
            self.backtest_lower_chart_item,
            textvariable=self.lower_chart_mode_var,
            values=[tr("chart_mode_drawdown")],
            width=10,
            state="readonly",
        )
        self.lower_chart_combo.pack(side="left", padx=(4, 0))

        self.backtest_table_source_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_table_source_item, text=tr("table_source")).pack(side="left")
        self.table_source_combo = ttk.Combobox(
            self.backtest_table_source_item,
            textvariable=self.table_source_var,
            values=[tr("source_auto"), tr("source_screen"), tr("source_strategy")],
            width=10,
            state="readonly",
        )
        self.table_source_combo.pack(side="left", padx=(4, 0))

        self.backtest_funding_view_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_funding_view_item, text=tr("funding_view")).pack(side="left")
        self.funding_view_combo = ttk.Combobox(
            self.backtest_funding_view_item,
            textvariable=self.funding_view_var,
            values=[tr("funding_mode_lump"), tr("funding_mode_dca"), tr("funding_mode_va")],
            width=12,
            state="readonly",
        )
        self.funding_view_combo.pack(side="left", padx=(4, 0))

        self.backtest_contrib_item = ttk.Frame(self.backtest_advanced_frame)
        self.show_contributed_check = ttk.Checkbutton(self.backtest_contrib_item, text=tr("legend_contributed"), variable=self.show_contributed_var)
        self.show_contributed_check.pack(side="left")

        self.backtest_pos_weight_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_pos_weight_item, text=tr("position_weight_pct")).pack(side="left")
        ttk.Entry(self.backtest_pos_weight_item, textvariable=self.position_weight_pct_var, width=7).pack(side="left", padx=(4, 0))

        self.backtest_max_holdings_cfg_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_max_holdings_cfg_item, text=tr("max_holdings_cfg")).pack(side="left")
        ttk.Entry(self.backtest_max_holdings_cfg_item, textvariable=self.max_holdings_cfg_var, width=7).pack(side="left", padx=(4, 0))

        self.backtest_max_new_buys_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_max_new_buys_item, text=tr("max_new_buys")).pack(side="left")
        ttk.Entry(self.backtest_max_new_buys_item, textvariable=self.max_new_buys_var, width=7).pack(side="left", padx=(4, 0))

        self.backtest_max_buy_amount_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_max_buy_amount_item, text=tr("max_buy_amount")).pack(side="left")
        ttk.Entry(self.backtest_max_buy_amount_item, textvariable=self.max_buy_amount_var, width=10).pack(side="left", padx=(4, 0))

        self.backtest_min_cash_reserve_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_min_cash_reserve_item, text=tr("min_cash_reserve_pct")).pack(side="left")
        ttk.Entry(self.backtest_min_cash_reserve_item, textvariable=self.min_cash_reserve_pct_var, width=7).pack(side="left", padx=(4, 0))

        self.backtest_exec_basis_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_exec_basis_item, text=tr("exec_price_basis")).pack(side="left")
        self.exec_basis_combo = ttk.Combobox(
            self.backtest_exec_basis_item,
            textvariable=self.exec_price_basis_var,
            values=[
                tr("price_basis_auto"),
                tr("price_basis_open"),
                tr("price_basis_close"),
                tr("price_basis_prev_close"),
            ],
            width=10,
            state="readonly",
        )
        self.exec_basis_combo.pack(side="left", padx=(4, 0))

        self.backtest_exec_offset_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_exec_offset_item, text=tr("price_offset_pct")).pack(side="left")
        ttk.Entry(self.backtest_exec_offset_item, textvariable=self.price_offset_pct_var, width=7).pack(side="left", padx=(4, 0))

        self.backtest_universe_filter_item = ttk.Frame(self.backtest_advanced_frame)
        ttk.Label(self.backtest_universe_filter_item, text=tr("universe_filter_summary")).pack(side="left")
        ttk.Entry(self.backtest_universe_filter_item, textvariable=self.universe_exchanges_var, width=10).pack(side="left", padx=(4, 2))
        ttk.Entry(self.backtest_universe_filter_item, textvariable=self.universe_size_buckets_var, width=10).pack(side="left", padx=(2, 2))
        ttk.Entry(self.backtest_universe_filter_item, textvariable=self.universe_include_sectors_var, width=12).pack(side="left", padx=(2, 2))
        ttk.Entry(self.backtest_universe_filter_item, textvariable=self.universe_exclude_tickers_var, width=12).pack(side="left", padx=(2, 0))

        self._backtest_advanced_items: list[tuple[ttk.Frame, bool]] = [
            (self.backtest_exec_item, False),
            (self.backtest_comm_item, False),
            (self.backtest_slip_item, False),
            (self.backtest_share_item, False),
            (self.backtest_cashbuf_item, False),
            (self.backtest_override_item, False),
            (self.backtest_lower_chart_item, False),
            (self.backtest_table_source_item, False),
            (self.backtest_funding_view_item, False),
            (self.backtest_contrib_item, False),
            (self.backtest_pos_weight_item, False),
            (self.backtest_max_holdings_cfg_item, False),
            (self.backtest_max_new_buys_item, False),
            (self.backtest_max_buy_amount_item, False),
            (self.backtest_min_cash_reserve_item, False),
            (self.backtest_exec_basis_item, False),
            (self.backtest_exec_offset_item, False),
            (self.backtest_universe_filter_item, False),
        ]

        # Funding section
        self.funding_items_container = ttk.Frame(self.funding_section)
        self.funding_items_container.grid(row=0, column=0, sticky="ew")
        self.funding_items_container.grid_columnconfigure(0, weight=1)

        self.funding_mode_item = ttk.Frame(self.funding_items_container)
        ttk.Label(self.funding_mode_item, text=tr("funding_mode")).pack(side="left")
        self.funding_mode_combo = ttk.Combobox(
            self.funding_mode_item,
            textvariable=self.funding_mode_var,
            values=[
                tr("funding_mode_lump"),
                tr("funding_mode_dca"),
                tr("funding_mode_va"),
                tr("funding_mode_compare"),
            ],
            width=16,
            state="readonly",
        )
        self.funding_mode_combo.pack(side="left", padx=(4, 0))

        self.funding_contrib_freq_item = ttk.Frame(self.funding_items_container)
        ttk.Label(self.funding_contrib_freq_item, text=tr("funding_contribution_freq")).pack(side="left")
        self.funding_contrib_freq_combo = ttk.Combobox(
            self.funding_contrib_freq_item,
            textvariable=self.funding_contrib_freq_var,
            values=[
                tr("funding_freq_monthly"),
                tr("funding_freq_quarterly"),
                tr("funding_freq_yearly"),
            ],
            width=12,
            state="readonly",
        )
        self.funding_contrib_freq_combo.pack(side="left", padx=(4, 0))

        self.funding_initial_item = ttk.Frame(self.funding_items_container)
        ttk.Label(self.funding_initial_item, text=tr("initial_cash")).pack(side="left")
        self.funding_initial_entry = ttk.Entry(self.funding_initial_item, textvariable=self.initial_cash_var, width=12)
        self.funding_initial_entry.pack(side="left", padx=(4, 0))

        self.funding_dca_item = ttk.Frame(self.funding_items_container)
        ttk.Label(self.funding_dca_item, text=tr("funding_dca_amount")).pack(side="left")
        self.funding_dca_entry = ttk.Entry(self.funding_dca_item, textvariable=self.funding_dca_var, width=12)
        self.funding_dca_entry.pack(side="left", padx=(4, 0))

        self.funding_va_step_item = ttk.Frame(self.funding_items_container)
        ttk.Label(self.funding_va_step_item, text=tr("funding_va_step")).pack(side="left")
        self.funding_va_step_entry = ttk.Entry(self.funding_va_step_item, textvariable=self.funding_va_step_var, width=11)
        self.funding_va_step_entry.pack(side="left", padx=(4, 0))

        self.funding_va_min_item = ttk.Frame(self.funding_items_container)
        ttk.Label(self.funding_va_min_item, text=tr("funding_va_min")).pack(side="left")
        self.funding_va_min_entry = ttk.Entry(self.funding_va_min_item, textvariable=self.funding_va_min_var, width=11)
        self.funding_va_min_entry.pack(side="left", padx=(4, 0))

        self.funding_va_min_policy_item = ttk.Frame(self.funding_items_container)
        ttk.Label(self.funding_va_min_policy_item, text=tr("funding_va_min_policy")).pack(side="left")
        self.funding_va_min_policy_combo = ttk.Combobox(
            self.funding_va_min_policy_item,
            textvariable=self.funding_va_min_policy_var,
            values=[
                tr("va_min_policy_positive_raw_only"),
                tr("va_min_policy_every_rebalance"),
            ],
            width=28,
            state="readonly",
        )
        self.funding_va_min_policy_combo.pack(side="left", padx=(4, 0))

        self.funding_va_max_item = ttk.Frame(self.funding_items_container)
        ttk.Label(self.funding_va_max_item, text=tr("funding_va_max")).pack(side="left")
        self.funding_va_max_entry = ttk.Entry(self.funding_va_max_item, textvariable=self.funding_va_max_var, width=11)
        self.funding_va_max_entry.pack(side="left", padx=(4, 0))

        self.funding_va_withdraw_item = ttk.Frame(self.funding_items_container)
        self.funding_va_withdraw_chk = ttk.Checkbutton(
            self.funding_va_withdraw_item,
            text=tr("funding_va_withdraw"),
            variable=self.funding_va_withdraw_var,
        )
        self.funding_va_withdraw_chk.pack(side="left")

        self.funding_compare_basis_item = ttk.Frame(self.funding_items_container)
        ttk.Label(self.funding_compare_basis_item, text=tr("funding_compare_basis")).pack(side="left")
        self.funding_compare_basis_combo = ttk.Combobox(
            self.funding_compare_basis_item,
            textvariable=self.funding_compare_basis_var,
            values=[
                tr("funding_compare_default"),
                tr("funding_compare_align_dca"),
                tr("funding_compare_custom"),
            ],
            width=28,
            state="readonly",
        )
        self.funding_compare_basis_combo.pack(side="left", padx=(4, 0))

        self.funding_custom_total_item = ttk.Frame(self.funding_items_container)
        ttk.Label(self.funding_custom_total_item, text=tr("funding_custom_total")).pack(side="left")
        self.funding_custom_total_entry = ttk.Entry(self.funding_custom_total_item, textvariable=self.funding_custom_total_var, width=12)
        self.funding_custom_total_entry.pack(side="left", padx=(4, 0))

        self._funding_item_frames: dict[str, ttk.Frame] = {
            "mode": self.funding_mode_item,
            "contribution_freq": self.funding_contrib_freq_item,
            "initial_cash": self.funding_initial_item,
            "dca": self.funding_dca_item,
            "va_step": self.funding_va_step_item,
            "va_min": self.funding_va_min_item,
            "va_min_policy": self.funding_va_min_policy_item,
            "va_max": self.funding_va_max_item,
            "va_withdraw": self.funding_va_withdraw_item,
            "compare_basis": self.funding_compare_basis_item,
            "custom_total": self.funding_custom_total_item,
        }
        self._funding_visible_keys: list[str] = ["mode", "initial_cash"]

        # Action section
        self.action_primary_item = ttk.Frame(self.action_section)
        self.run_screen_btn = ttk.Button(self.action_primary_item, text=tr("run_screen"), command=self._run_screen, style="Primary.TButton")
        self.run_screen_btn.pack(side="left", padx=(0, 4))
        self.run_strategy_btn = ttk.Button(self.action_primary_item, text=tr("run_strategy"), command=self._run_strategy, style="Run.TButton")
        self.run_strategy_btn.pack(side="left", padx=(0, 4))
        self.compare_btn = ttk.Button(self.action_primary_item, text=tr("compare"), command=self._run_compare, style="Accent.TButton")
        self.compare_btn.pack(side="left", padx=(0, 8))
        self.preflight_btn = ttk.Button(self.action_primary_item, text=tr("preflight_check"), command=self._run_preflight_check, style="Small.TButton")
        self.preflight_btn.pack(side="left", padx=(0, 4))
        self.save_strategy_btn = ttk.Button(self.action_primary_item, text=tr("save_strategy"), command=self._save_strategy_state, style="Small.TButton")
        self.save_strategy_btn.pack(side="left", padx=(0, 4))
        self.load_strategy_btn = ttk.Button(self.action_primary_item, text=tr("load_strategy"), command=self._load_strategy_state, style="Small.TButton")
        self.load_strategy_btn.pack(side="left", padx=(0, 4))
        self.audit_prompt_btn = ttk.Button(self.action_primary_item, text=tr("audit_prompt_copy"), command=self._copy_audit_prompt_template, style="Ghost.TButton")
        self.audit_prompt_btn.pack(side="left", padx=(0, 8))
        self.ai_export_btn = ttk.Button(self.action_primary_item, text=tr("ai_export"), command=self._save_ai_review_bundle, style="Ghost.TButton")
        self.ai_export_btn.pack(side="left", padx=(0, 8))
        self.progress_var = tk.StringVar(value=tr("idle"))
        ttk.Label(self.action_primary_item, textvariable=self.progress_var, style="Dim.TLabel").pack(side="left", padx=(8, 0))
        self.compare_context_var = tk.StringVar(value="")
        self.action_context_item = ttk.Frame(self.action_section)
        ttk.Label(self.action_context_item, textvariable=self.compare_context_var).pack(side="left")
        self._action_items: list[tuple[ttk.Frame, bool]] = [
            (self.action_primary_item, True),
            (self.action_context_item, True),
        ]

        # Summary section
        self.builder_summary_var = tk.StringVar(value=tr("builder_summary_default"))
        self.strategy_builder_summary_var = tk.StringVar(value=tr("strategy_builder_summary_default"))
        self.funding_info_var = tk.StringVar(value="")

        self.builder_summary_label = ttk.Label(self.summary_section, textvariable=self.builder_summary_var, justify="left", anchor="w")
        self.builder_summary_label.grid(row=0, column=0, sticky="ew")
        self.strategy_summary_label = ttk.Label(self.summary_section, textvariable=self.strategy_builder_summary_var, justify="left", anchor="w")
        self.strategy_summary_label.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.funding_summary_label = ttk.Label(self.summary_section, textvariable=self.funding_info_var, justify="left", anchor="w")
        self.funding_summary_label.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self._summary_labels: list[ttk.Label] = [
            self.builder_summary_label,
            self.strategy_summary_label,
            self.funding_summary_label,
        ]

        self.strategy_preview_frame = ttk.LabelFrame(self._results_inner, text=tr("strategy_builder_preview"), padding=(6, 4, 6, 4))
        self.strategy_preview_frame.grid(row=0, column=0, sticky="ew", padx=(10, 10), pady=(8, 4))
        self.strategy_preview_frame.grid_rowconfigure(0, weight=1)
        self.strategy_preview_frame.grid_columnconfigure(0, weight=1)
        self.strategy_preview_text = tk.Text(self.strategy_preview_frame, height=5, wrap="none")
        self.strategy_preview_text.grid(row=0, column=0, sticky="ew")
        self.strategy_preview_text.configure(state="disabled")

        self.kpi_frame = ttk.Frame(self._results_inner)
        self.kpi_frame.grid(row=1, column=0, sticky="ew", padx=(10, 10), pady=(0, 4))
        self.kpi_vars = {
            "cagr": tk.StringVar(value=f"{tr('kpi_cagr')} -"),
            "mdd": tk.StringVar(value=f"{tr('kpi_mdd')} -"),
            "sharpe": tk.StringVar(value=f"{tr('kpi_sharpe')} -"),
            "turnover": tk.StringVar(value=f"{tr('kpi_turnover')} -"),
            "trades": tk.StringVar(value=f"{tr('kpi_trades')} -"),
        }
        self.kpi_labels: dict[str, ttk.Label] = {}
        for idx, key in enumerate(["cagr", "mdd", "sharpe", "turnover", "trades"]):
            self.kpi_frame.grid_columnconfigure(idx, weight=1)
            lbl = ttk.Label(self.kpi_frame, textvariable=self.kpi_vars[key], style="KPI.TLabel")
            lbl.grid(row=0, column=idx, sticky="ew", padx=(0, 6))
            self.kpi_labels[key] = lbl

        self.charts_frame = ttk.Frame(self._results_inner, padding=(10, 0, 10, 0))
        self.charts_frame.grid(row=2, column=0, sticky="nsew")
        self.charts_frame.grid_columnconfigure(0, weight=1)
        self.charts_frame.grid_rowconfigure(0, weight=1, minsize=320)
        self.charts_frame.grid_rowconfigure(1, weight=1, minsize=320)

        self.fig = Figure(figsize=(10, 7.5), dpi=100)
        _gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.35)
        self.ax_top = self.fig.add_subplot(_gs[0])
        self._ax_top_twin = self.ax_top.twinx()
        self.ax_bottom = self.fig.add_subplot(_gs[1], sharex=self.ax_top)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.charts_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, rowspan=2, sticky="nsew")

        self.trade_analytics_frame = ttk.LabelFrame(self._results_inner, text=tr("trade_analysis"), padding=(10, 6, 10, 6))
        self.trade_analytics_frame.grid(row=3, column=0, sticky="nsew")
        self.trade_analytics_frame.grid_columnconfigure(0, weight=1)
        self.trade_analytics_frame.grid_rowconfigure(0, weight=0, minsize=0)  # opt row
        self.trade_analytics_frame.grid_rowconfigure(1, weight=1, minsize=300)  # summary charts

        opt_row = ttk.Frame(self.trade_analytics_frame)
        opt_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(opt_row, text=tr("return_plot_mode")).grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            opt_row,
            textvariable=self.return_plot_mode_var,
            values=[tr("ret_mode_strat_bench"), tr("ret_mode_strat_excess"), tr("ret_mode_excess_only")],
            width=13,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=(4, 10))
        ttk.Label(opt_row, text=tr("cum_line_mode")).grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            opt_row,
            textvariable=self.cum_line_mode_var,
            values=[tr("cum_mode_strategy"), tr("cum_mode_excess"), tr("cum_mode_none")],
            width=10,
            state="readonly",
        ).grid(row=0, column=3, sticky="w", padx=(4, 0))
        ttk.Checkbutton(opt_row, text=tr("tooltip_on"), variable=self.show_tooltip_var).grid(row=0, column=4, sticky="w", padx=(12, 0))

        self.trade_summary_frame = ttk.Frame(self.trade_analytics_frame)
        self.trade_summary_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.trade_summary_frame.grid_rowconfigure(0, weight=1)
        self.trade_summary_frame.grid_columnconfigure(0, weight=1)

        self.trade_summary_fig = Figure(figsize=(10, 4.4), dpi=100)
        self.trade_summary_canvas = FigureCanvasTkAgg(self.trade_summary_fig, master=self.trade_summary_frame)
        self.trade_summary_canvas_widget = self.trade_summary_canvas.get_tk_widget()
        self.trade_summary_canvas_widget.grid(row=0, column=0, sticky="nsew")

        self.tables_frame = ttk.Frame(self._results_inner, padding=(10, 8, 10, 10))
        self.tables_frame.grid(row=4, column=0, sticky="nsew")
        self.tables_frame.grid_rowconfigure(0, weight=1)
        self.tables_frame.grid_columnconfigure(0, weight=1)

        tables = ttk.Notebook(self.tables_frame)
        tables.grid(row=0, column=0, sticky="nsew")
        self.tables_notebook = tables
        self.tables_notebook.bind("<<NotebookTabChanged>>", self._on_tables_tab_change)

        monthly_tab = ttk.Frame(tables)
        monthly_tab.grid_rowconfigure(0, weight=0, minsize=180)  # chart
        monthly_tab.grid_rowconfigure(1, weight=1)  # tree
        monthly_tab.grid_columnconfigure(0, weight=1)
        yearly_tab = ttk.Frame(tables)
        yearly_tab.grid_rowconfigure(0, weight=0, minsize=180)  # chart
        yearly_tab.grid_rowconfigure(1, weight=1)  # tree
        yearly_tab.grid_columnconfigure(0, weight=1)
        trades_tab = ttk.Frame(tables)
        rebalance_tab = ttk.Frame(tables)
        snapshots_tab = ttk.Frame(tables)
        metrics_tab = ttk.Frame(tables)
        funding_compare_tab = ttk.Frame(tables)
        funding_flows_tab = ttk.Frame(tables)

        tables.add(monthly_tab, text=tr("monthly_returns_tab"))
        tables.add(yearly_tab, text=tr("yearly_returns_tab"))
        tables.add(trades_tab, text=tr("trades_tab"))
        tables.add(rebalance_tab, text=tr("rebalance_log_tab"))
        tables.add(snapshots_tab, text=tr("snapshots_tab"))
        tables.add(metrics_tab, text=tr("metrics_tab"))
        tables.add(funding_compare_tab, text=tr("funding_compare_tab"))
        tables.add(funding_flows_tab, text=tr("funding_flows_tab"))
        self.trades_tab = trades_tab

        # Heatmap tab
        heatmap_tab = ttk.Frame(tables)
        tables.add(heatmap_tab, text=tr("heatmap_tab"))
        heatmap_tab.grid_rowconfigure(1, weight=1)
        heatmap_tab.grid_rowconfigure(2, weight=0)
        heatmap_tab.grid_columnconfigure(0, weight=1)

        heatmap_ctrl_row = ttk.Frame(heatmap_tab, padding=(4, 4, 4, 2))
        heatmap_ctrl_row.grid(row=0, column=0, sticky="ew")
        heatmap_ctrl_row.grid_columnconfigure(12, weight=1)
        ttk.Label(heatmap_ctrl_row, text=tr("ticker_scope")).grid(row=0, column=0, sticky="w")
        ttk.Combobox(heatmap_ctrl_row, textvariable=self.trade_scope_var, values=[tr("scope_all"), tr("scope_top_n")], width=9, state="readonly").grid(row=0, column=1, sticky="w", padx=(4, 10))
        ttk.Label(heatmap_ctrl_row, text=tr("top_n")).grid(row=0, column=2, sticky="w")
        ttk.Combobox(heatmap_ctrl_row, textvariable=self.trade_top_n_var, values=["10", "20", "30", "50"], width=6, state="readonly").grid(row=0, column=3, sticky="w", padx=(4, 10))
        ttk.Label(heatmap_ctrl_row, text=tr("heat_metric")).grid(row=0, column=4, sticky="w")
        ttk.Combobox(heatmap_ctrl_row, textvariable=self.trade_metric_var, values=[tr("metric_net_notional"), tr("metric_net_count")], width=14, state="readonly").grid(row=0, column=5, sticky="w", padx=(4, 10))
        ttk.Label(heatmap_ctrl_row, text=tr("sort_by")).grid(row=0, column=6, sticky="w")
        ttk.Combobox(heatmap_ctrl_row, textvariable=self.trade_sort_var, values=[tr("sort_notional"), tr("sort_count")], width=12, state="readonly").grid(row=0, column=7, sticky="w", padx=(4, 10))
        ttk.Label(heatmap_ctrl_row, text=tr("month_filter")).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.trade_month_combo = ttk.Combobox(heatmap_ctrl_row, textvariable=self.trade_filter_month_var, values=[tr("all")], width=10, state="readonly")
        self.trade_month_combo.grid(row=1, column=1, sticky="w", padx=(4, 10), pady=(4, 0))
        ttk.Label(heatmap_ctrl_row, text=tr("ticker_filter")).grid(row=1, column=2, sticky="w", pady=(4, 0))
        self.trade_ticker_combo = ttk.Combobox(heatmap_ctrl_row, textvariable=self.trade_filter_ticker_var, values=[tr("all")], width=10, state="readonly")
        self.trade_ticker_combo.grid(row=1, column=3, sticky="w", padx=(4, 0), pady=(4, 0))

        self.heatmap_scroll_frame = ttk.Frame(heatmap_tab)
        self.heatmap_scroll_frame.grid(row=1, column=0, sticky="nsew")
        self.heatmap_scroll_frame.grid_rowconfigure(0, weight=1)
        self.heatmap_scroll_frame.grid_columnconfigure(0, weight=1)
        self.heatmap_scroll_canvas = tk.Canvas(self.heatmap_scroll_frame, highlightthickness=0, borderwidth=0)
        self.heatmap_vscroll = ttk.Scrollbar(self.heatmap_scroll_frame, orient="vertical", command=self.heatmap_scroll_canvas.yview)
        self.heatmap_scroll_canvas.configure(yscrollcommand=self.heatmap_vscroll.set)
        self.heatmap_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self.heatmap_vscroll.grid(row=0, column=1, sticky="ns")
        self.heatmap_inner = ttk.Frame(self.heatmap_scroll_canvas)
        self.heatmap_window = self.heatmap_scroll_canvas.create_window((0, 0), window=self.heatmap_inner, anchor="nw")
        self.heatmap_inner.bind("<Configure>", self._on_heatmap_inner_configure)
        self.heatmap_scroll_canvas.bind("<Configure>", self._on_heatmap_canvas_configure)
        self.heatmap_scroll_canvas.bind("<Enter>", self._bind_heatmap_mousewheel)
        self.heatmap_scroll_canvas.bind("<Leave>", self._unbind_heatmap_mousewheel)
        self.heatmap_fig = Figure(figsize=(10, 8.0), dpi=90)
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, master=self.heatmap_inner)
        self.heatmap_canvas_widget = self.heatmap_canvas.get_tk_widget()
        self.heatmap_canvas_widget.grid(row=0, column=0, sticky="nsew")
        self.heatmap_canvas_widget.bind("<Enter>", self._bind_heatmap_mousewheel)
        self.heatmap_canvas_widget.bind("<Leave>", self._unbind_heatmap_mousewheel)
        self.heatmap_inner.grid_rowconfigure(0, weight=1)
        self.heatmap_inner.grid_columnconfigure(0, weight=1)
        self.heatmap_fig.canvas.mpl_connect("motion_notify_event", self._on_heatmap_motion)
        self.heatmap_fig.canvas.mpl_connect("figure_leave_event", self._on_heatmap_leave)
        self.heatmap_fig.canvas.mpl_connect("button_press_event", self._on_heatmap_click)
        self.heatmap_tooltip = HoverTooltip(self.parent)

        self.hover_info_frame = ttk.LabelFrame(heatmap_tab, text=tr("hover_info"), padding=(6, 4, 6, 4))
        self.hover_info_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.hover_info_frame.grid_rowconfigure(0, weight=1)
        self.hover_info_frame.grid_columnconfigure(0, weight=1)
        self.hover_info_text = tk.Text(self.hover_info_frame, height=3, wrap="word")
        self.hover_info_text.grid(row=0, column=0, sticky="ew")
        self.hover_info_text.configure(state="disabled")
        self._set_hover_info(tr("hover_default"))

        # Ticker performance tab (종목별 수익률)
        ticker_perf_tab = ttk.Frame(tables)
        tables.add(ticker_perf_tab, text=tr("ticker_perf_tab"))
        ticker_perf_tab.grid_rowconfigure(0, weight=0, minsize=280)
        ticker_perf_tab.grid_rowconfigure(1, weight=1)
        ticker_perf_tab.grid_columnconfigure(0, weight=1)

        self.ticker_perf_fig = Figure(figsize=(10, 3.5), dpi=90)
        self.ticker_perf_ax = self.ticker_perf_fig.add_subplot(111)
        self.ticker_perf_canvas = FigureCanvasTkAgg(self.ticker_perf_fig, master=ticker_perf_tab)
        self.ticker_perf_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        ticker_perf_tree_frame = ttk.Frame(ticker_perf_tab)
        ticker_perf_tree_frame.grid(row=1, column=0, sticky="nsew")
        ticker_perf_tree_frame.grid_rowconfigure(0, weight=1)
        ticker_perf_tree_frame.grid_columnconfigure(0, weight=1)
        self.ticker_perf_tree = self._make_tree(ticker_perf_tree_frame, ["ticker", "ticker_cost", "ticker_pnl", "ticker_return"], [80, 120, 120, 100], pack=False)
        self.ticker_perf_tree.tag_configure("pos", foreground="#0b7a3a")
        self.ticker_perf_tree.tag_configure("neg", foreground="#b42318")

        # Monthly chart (bar chart preview at top)
        self.monthly_chart_fig = Figure(figsize=(10, 2.2), dpi=90)
        self.monthly_chart_ax = self.monthly_chart_fig.add_subplot(111)
        self.monthly_chart_canvas = FigureCanvasTkAgg(self.monthly_chart_fig, master=monthly_tab)
        self.monthly_chart_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        monthly_tree_frame = ttk.Frame(monthly_tab)
        monthly_tree_frame.grid(row=1, column=0, sticky="nsew")
        monthly_tree_frame.grid_rowconfigure(0, weight=1)
        monthly_tree_frame.grid_columnconfigure(0, weight=1)
        self.monthly_tree = self._make_tree(monthly_tree_frame, ["source", "month", "strategy_ret", "bench_ret", "excess_ret"], [90, 90, 110, 110, 110], pack=False)
        self.monthly_tree.tag_configure("pos", foreground="#0b7a3a")
        self.monthly_tree.tag_configure("neg", foreground="#b42318")

        # Yearly chart (bar chart preview at top)
        self.yearly_chart_fig = Figure(figsize=(10, 2.2), dpi=90)
        self.yearly_chart_ax = self.yearly_chart_fig.add_subplot(111)
        self.yearly_chart_canvas = FigureCanvasTkAgg(self.yearly_chart_fig, master=yearly_tab)
        self.yearly_chart_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        yearly_tree_frame = ttk.Frame(yearly_tab)
        yearly_tree_frame.grid(row=1, column=0, sticky="nsew")
        yearly_tree_frame.grid_rowconfigure(0, weight=1)
        yearly_tree_frame.grid_columnconfigure(0, weight=1)
        self.yearly_tree = self._make_tree(yearly_tree_frame, ["year", "screen", "strategy"], [120, 130, 130], pack=False)
        self.yearly_tree.tag_configure("pos", foreground="#0b7a3a")
        self.yearly_tree.tag_configure("neg", foreground="#b42318")

        trades_tab.grid_rowconfigure(0, weight=1)
        trades_tab.grid_rowconfigure(1, weight=0)
        trades_tab.grid_columnconfigure(0, weight=1)
        trades_top = ttk.Frame(trades_tab)
        trades_top.grid(row=0, column=0, sticky="nsew")
        trades_top.grid_rowconfigure(0, weight=1)
        trades_top.grid_columnconfigure(0, weight=1)
        self.trades_tree = self._make_tree(
            trades_top,
            ["exec_date", "ticker", "side", "shares", "exec_price", "notional", "commission", "slippage", "reason"],
            [88, 72, 60, 80, 90, 100, 85, 80, 180],
            pack=False,
        )
        self.trades_tree.tag_configure("buy", foreground="#0b7a3a")
        self.trades_tree.tag_configure("sell", foreground="#b42318")
        self.trades_tree.column("exec_date", anchor="center")
        self.trades_tree.column("ticker", anchor="center")
        self.trades_tree.column("side", anchor="center")
        self.trades_tree.bind("<<TreeviewSelect>>", self._on_trade_select)

        reason_frame = ttk.LabelFrame(trades_tab, text=tr("trade_reason_detail"), padding=(6, 4, 6, 4))
        reason_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        reason_frame.grid_rowconfigure(0, weight=1)
        reason_frame.grid_columnconfigure(0, weight=1)
        self.trade_reason_text = tk.Text(reason_frame, height=4, wrap="word")
        self.trade_reason_text.grid(row=0, column=0, sticky="ew")
        self.trade_reason_text.configure(state="disabled")

        self.rebalance_tree = self._make_tree(
            rebalance_tab,
            ["signal_date", "exec_date", "mode", "universe_count", "candidate_count", "selected_count", "lookahead_ok", "benchmark"],
            [95, 95, 70, 95, 100, 95, 95, 95],
        )

        snapshots_tab.grid_rowconfigure(0, weight=2)
        snapshots_tab.grid_rowconfigure(1, weight=0)
        snapshots_tab.grid_rowconfigure(2, weight=1)
        snapshots_tab.grid_columnconfigure(0, weight=1)
        snap_top = ttk.Frame(snapshots_tab)
        snap_top.grid(row=0, column=0, sticky="nsew")
        snap_top.grid_rowconfigure(0, weight=1)
        snap_top.grid_columnconfigure(0, weight=1)
        self.snapshot_tree = self._make_tree(snap_top, ["source", "signal_date", "snapshot_path"], [90, 100, 620], pack=False)
        self.snapshot_tree.bind("<<TreeviewSelect>>", self._on_snapshot_select)

        snap_btn_row = ttk.Frame(snapshots_tab)
        snap_btn_row.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(snap_btn_row, text=tr("preview_snapshot"), command=self._on_snapshot_select).pack(side="left")

        snap_bottom = ttk.LabelFrame(snapshots_tab, text=tr("snapshot_preview"))
        snap_bottom.grid(row=2, column=0, sticky="nsew", pady=(6, 0))
        snap_bottom.grid_rowconfigure(0, weight=1)
        snap_bottom.grid_columnconfigure(0, weight=1)
        self.snapshot_preview_tree = self._make_tree(
            snap_bottom,
            ["ticker", "filter_pass", "rank_score", "rank", "selected", "target_weight", "lookahead_ok"],
            [100, 80, 90, 70, 70, 95, 90],
        )

        self.metrics_tree = self._make_tree(metrics_tab, ["metric", "screen", "strategy"], [180, 120, 120])
        ttk.Label(
            funding_compare_tab,
            text=tr("funding_compare_explain"),
            justify="left",
            wraplength=1200,
        ).pack(anchor="w", padx=6, pady=(6, 2))
        self.funding_compare_tree = self._make_tree(
            funding_compare_tab,
            [
                "mode_name",
                "twr_total_return",
                "twr_cagr",
                "twr_mdd",
                "twr_sharpe",
                "twr_vol",
                "account_cagr",
                "account_mdd",
                "account_sharpe",
                "account_vol",
                "turnover",
                "trades_count",
                "total_contributed",
                "ending_value",
                "pnl",
                "mwr_irr",
            ],
            [96, 98, 95, 88, 88, 90, 96, 90, 90, 92, 84, 80, 118, 118, 100, 96],
        )
        funding_flows_top = ttk.Frame(funding_flows_tab)
        funding_flows_top.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(funding_flows_top, text=tr("funding_flow_view_mode")).pack(side="left")
        self.funding_flow_view_combo = ttk.Combobox(
            funding_flows_top,
            textvariable=self.funding_flow_view_var,
            values=[tr("funding_flow_view_summary"), tr("funding_flow_view_detail")],
            width=8,
            state="readonly",
        )
        self.funding_flow_view_combo.pack(side="left", padx=(4, 8))
        ttk.Label(
            funding_flows_top,
            text=tr("funding_flow_explain"),
            justify="left",
            wraplength=980,
        ).pack(side="left", fill="x", expand=True)

        self.funding_flows_tabs = ttk.Notebook(funding_flows_tab)
        self.funding_flows_tabs.pack(fill="both", expand=True, padx=2, pady=(2, 0))

        ff_all_tab = ttk.Frame(self.funding_flows_tabs)
        ff_lump_tab = ttk.Frame(self.funding_flows_tabs)
        ff_dca_tab = ttk.Frame(self.funding_flows_tabs)
        ff_va_tab = ttk.Frame(self.funding_flows_tabs)
        self.funding_flows_tabs.add(ff_all_tab, text=tr("funding_flows_all_tab"))
        self.funding_flows_tabs.add(ff_lump_tab, text=tr("funding_flows_lump_tab"))
        self.funding_flows_tabs.add(ff_dca_tab, text=tr("funding_flows_dca_tab"))
        self.funding_flows_tabs.add(ff_va_tab, text=tr("funding_flows_va_tab"))

        widths = [
            100, 90, 62, 88, 78, 96, 96, 300, 112, 112, 100, 116, 116, 100, 116, 116, 108, 120, 120, 92, 92, 132, 108, 96, 120, 130, 95, 95, 90, 90
        ]
        self.funding_flows_all_tree = self._make_tree(ff_all_tab, FUNDING_FLOW_COLUMNS, widths)
        self.funding_flows_lump_tree = self._make_tree(ff_lump_tab, FUNDING_FLOW_COLUMNS, widths)
        self.funding_flows_dca_tree = self._make_tree(ff_dca_tab, FUNDING_FLOW_COLUMNS, widths)
        self.funding_flows_va_tree = self._make_tree(ff_va_tab, FUNDING_FLOW_COLUMNS, widths)
        self.funding_flows_tree = self.funding_flows_all_tree
        self._funding_flow_trees: dict[str, ttk.Treeview] = {
            "all": self.funding_flows_all_tree,
            "lump_sum": self.funding_flows_lump_tree,
            "dca": self.funding_flows_dca_tree,
            "va": self.funding_flows_va_tree,
        }
        for tag, color in {
            "lump_sum": "#f2f7ff",
            "dca": "#fff7ed",
            "va": "#eefdf4",
        }.items():
            self.funding_flows_all_tree.tag_configure(tag, background=color)
        self._apply_funding_flow_display_mode()

        self.lower_chart_mode_var.trace_add("write", lambda *_: self._render_charts())
        self.table_source_var.trace_add("write", lambda *_: self._render_all())
        self.screen_rule_var.trace_add("write", lambda *_: self._on_screen_rule_changed())
        self.strategy_source_var.trace_add("write", lambda *_: self._on_strategy_source_changed())
        self.trade_scope_var.trace_add("write", lambda *_: self._render_trade_analytics())
        self.trade_top_n_var.trace_add("write", lambda *_: self._render_trade_analytics())
        self.trade_metric_var.trace_add("write", lambda *_: self._render_trade_analytics())
        self.trade_sort_var.trace_add("write", lambda *_: self._render_trade_analytics())
        self.return_plot_mode_var.trace_add("write", lambda *_: self._render_trade_analytics())
        self.cum_line_mode_var.trace_add("write", lambda *_: self._render_trade_analytics())
        self.show_tooltip_var.trace_add("write", lambda *_: self._on_tooltip_toggle())
        self.trade_filter_month_var.trace_add("write", lambda *_: self._on_trade_filter_change())
        self.trade_filter_ticker_var.trace_add("write", lambda *_: self._on_trade_filter_change())
        self.show_contributed_var.trace_add("write", lambda *_: self._render_charts())
        self.funding_view_var.trace_add("write", lambda *_: self._render_all())
        self.funding_flow_view_var.trace_add("write", lambda *_: self._render_tables())
        self.funding_mode_var.trace_add("write", lambda *_: self._on_funding_mode_changed())
        self.funding_compare_basis_var.trace_add("write", lambda *_: self._on_funding_compare_basis_changed())
        for var in [
            self.funding_contrib_freq_var,
            self.funding_dca_var,
            self.funding_va_step_var,
            self.funding_va_min_var,
            self.funding_va_max_var,
            self.funding_va_min_policy_var,
            self.funding_va_withdraw_var,
            self.funding_custom_total_var,
        ]:
            var.trace_add("write", lambda *_: self._update_funding_preview())
        for var in [
            self.market_var,
            self.start_var,
            self.end_var,
            self.freq_var,
            self.holdings_var,
            self.benchmark_var,
            self.exec_timing_var,
            self.initial_cash_var,
            self.commission_bps_var,
            self.slippage_bps_var,
            self.share_mode_var,
            self.cash_buffer_pct_var,
        ]:
            var.trace_add("write", lambda *_: self._refresh_strategy_builder_preview() if self._is_strategy_builder_mode() else None)
        for var in [
            self.universe_exchanges_var,
            self.universe_size_buckets_var,
            self.universe_include_sectors_var,
            self.universe_exclude_sectors_var,
            self.universe_include_subsectors_var,
            self.universe_exclude_subsectors_var,
            self.universe_exclude_tickers_var,
            self.universe_watchlist_name_var,
        ]:
            var.trace_add("write", lambda *_: self._update_universe_filter_summary())
        self.stock_type_included_var.trace_add("write", lambda *_: self._update_universe_filter_summary())
        self.ptp_included_var.trace_add("write", lambda *_: self._update_universe_filter_summary())
        self.universe_financial_only_var.trace_add("write", lambda *_: self._update_universe_filter_summary())
        self.universe_direct_tickers_var.trace_add("write", lambda *_: self._update_universe_filter_summary())
        for key, var in self.universe_preset_vars.items():
            var.trace_add("write", lambda *_: self._update_universe_filter_summary())
        for exch in ["NASDAQ", "NYSE"]:
            for b in UNIVERSE_BUCKETS:
                self.universe_bucket_vars[exch][b].trace_add("write", lambda *_: self._update_universe_filter_summary())

        self.scrollable.canvas.bind("<Configure>", self._on_root_configure, add="+")
        self.root_frame.bind("<Configure>", self._on_root_configure, add="+")
        self.root_frame.after(120, self._apply_responsive_layout)
        self._on_funding_mode_changed()
        self._update_advanced_visibility()
        self._update_buy_section_visibility()
        self._update_sell_section_visibility()
        self._update_universe_section_visibility()
        self._sync_condition_editors_from_state()
        self._update_universe_filter_summary()
        self._on_strategy_source_changed()
        _install_text_context_menus(self.parent)

    def _render_position_sizing_panel(self, parent: ttk.LabelFrame) -> None:
        ttk.Label(parent, text=tr("position_weight_pct")).grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(parent, textvariable=self.position_weight_pct_var, width=9).grid(row=0, column=1, sticky="w", pady=2)
        ttk.Button(parent, text="+1", width=3, command=lambda: self._bump_numeric(self.position_weight_pct_var, 1.0)).grid(
            row=0, column=2, sticky="w", padx=(4, 0), pady=2
        )
        ttk.Button(parent, text="+5", width=3, command=lambda: self._bump_numeric(self.position_weight_pct_var, 5.0)).grid(
            row=0, column=3, sticky="w", padx=(2, 0), pady=2
        )

        ttk.Label(parent, text=tr("max_holdings_cfg")).grid(row=1, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(parent, textvariable=self.max_holdings_cfg_var, width=9).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Button(parent, text="+1", width=3, command=lambda: self._bump_numeric(self.max_holdings_cfg_var, 1.0)).grid(
            row=1, column=2, sticky="w", padx=(4, 0), pady=2
        )
        ttk.Button(parent, text="+5", width=3, command=lambda: self._bump_numeric(self.max_holdings_cfg_var, 5.0)).grid(
            row=1, column=3, sticky="w", padx=(2, 0), pady=2
        )

        ttk.Label(parent, text=tr("max_new_buys")).grid(row=2, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(parent, textvariable=self.max_new_buys_var, width=9).grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(parent, text=tr("max_buy_amount")).grid(row=3, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(parent, textvariable=self.max_buy_amount_var, width=12).grid(row=3, column=1, sticky="w", pady=2)
        ttk.Label(parent, text=tr("min_cash_reserve_pct")).grid(row=4, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(parent, textvariable=self.min_cash_reserve_pct_var, width=9).grid(row=4, column=1, sticky="w", pady=2)
        ttk.Label(
            parent,
            text="제한 적용 시 reason_detail: capped_by_max_holdings/max_buy_amount/daily_new_buy_limit/cash_reserve",
            style="Dim.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))

    def _render_execution_rules_panel(self, parent: ttk.LabelFrame) -> None:
        ttk.Label(parent, text=tr("exec")).grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Combobox(
            parent,
            textvariable=self.exec_timing_var,
            values=["next_open", "same_close"],
            width=12,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(parent, text=tr("exec_price_basis")).grid(row=1, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Combobox(
            parent,
            textvariable=self.exec_price_basis_var,
            values=[
                tr("price_basis_auto"),
                tr("price_basis_open"),
                tr("price_basis_close"),
                tr("price_basis_prev_close"),
            ],
            width=12,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(parent, text=tr("price_offset_pct")).grid(row=2, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(parent, textvariable=self.price_offset_pct_var, width=9).grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(parent, text=tr("commission_bps")).grid(row=3, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(parent, textvariable=self.commission_bps_var, width=9).grid(row=3, column=1, sticky="w", pady=2)
        ttk.Label(parent, text=tr("slippage_bps")).grid(row=4, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(parent, textvariable=self.slippage_bps_var, width=9).grid(row=4, column=1, sticky="w", pady=2)
        ttk.Label(
            parent,
            text="예: next_open + prev_close + 0.5% 오프셋. 현재는 매수/매도 공통 execution 설정이 적용됩니다.",
            style="Dim.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _render_universe_filter_panel(self, parent: ttk.Frame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        self.universe_summary_var = tk.StringVar(value="")

        type_row = ttk.LabelFrame(parent, text="종목 종류", padding=(6, 4, 6, 4))
        type_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Checkbutton(type_row, text="종류 주식", variable=self.stock_type_included_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(type_row, text="PTP 주식", variable=self.ptp_included_var).pack(side="left", padx=(0, 10))

        uni_row = ttk.LabelFrame(parent, text="주식 유니버스", padding=(6, 4, 6, 4))
        uni_row.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        uni_row.grid_columnconfigure(0, weight=1)

        # Universe input mode toggle
        mode_row = ttk.Frame(uni_row)
        mode_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(mode_row, text="입력 방식:").pack(side="left")
        ttk.Radiobutton(
            mode_row, text="프리셋 선택", variable=self.universe_input_mode_var, value="preset",
            command=self._on_universe_input_mode_changed,
        ).pack(side="left", padx=(6, 0))
        ttk.Radiobutton(
            mode_row, text="티커 직접 입력", variable=self.universe_input_mode_var, value="direct",
            command=self._on_universe_input_mode_changed,
        ).pack(side="left", padx=(8, 0))

        # Preset frame
        self._universe_preset_frame = ttk.Frame(uni_row)
        self._universe_preset_frame.grid(row=1, column=0, sticky="ew")
        self._universe_preset_frame.grid_columnconfigure(0, weight=1)

        sp500_row = ttk.Frame(self._universe_preset_frame)
        sp500_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(
            sp500_row,
            text="S&P 500 (현재 시점)",
            variable=self.universe_preset_vars["sp500"],
            command=lambda: self._on_universe_parent_toggled("sp500"),
        ).pack(side="left")
        ttk.Checkbutton(
            sp500_row,
            text="S&P 500 (과거 시점/PIT)",
            variable=self.universe_preset_vars["sp500_pit"],
            command=lambda: self._on_universe_parent_toggled("sp500_pit"),
        ).pack(side="left", padx=(15, 0))

        nasdaq_fin_row = ttk.Frame(self._universe_preset_frame)
        nasdaq_fin_row.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(
            nasdaq_fin_row,
            text="나스닥 보통주 (재무 데이터 보유)",
            variable=self.universe_preset_vars["nasdaq_stock_only_financial"],
            command=lambda: self._on_universe_parent_toggled("nasdaq_stock_only_financial"),
        ).pack(side="left")

        # Market cap bucket criteria hint
        _BUCKET_HINT = "mega ≥$200B  |  large ≥$10B  |  mid ≥$2B  |  small ≥$300M  |  micro <$300M"

        def _exchange_block(row_idx: int, preset_key: str, label: str, exch: str) -> None:
            outer = ttk.Frame(self._universe_preset_frame)
            outer.grid(row=row_idx, column=0, sticky="ew", pady=(0, 2))
            row = ttk.Frame(outer)
            row.pack(fill="x")
            ttk.Checkbutton(
                row,
                text=f"{label} 전체",
                variable=self.universe_preset_vars[preset_key],
                command=lambda k=preset_key, e=exch: self._on_universe_parent_toggled(k, e),
            ).pack(side="left")
            ttk.Label(row, textvariable=self.universe_parent_status_vars[exch], style="Dim.TLabel").pack(side="left", padx=(6, 0))
            for bucket in UNIVERSE_BUCKETS:
                ttk.Checkbutton(
                    row,
                    text=bucket,
                    variable=self.universe_bucket_vars[exch][bucket],
                    command=lambda e=exch: self._on_universe_bucket_toggled(e),
                ).pack(side="left", padx=(6, 0))
            ttk.Label(outer, text=_BUCKET_HINT, style="Dim.TLabel").pack(anchor="w", padx=(4, 0))

        _exchange_block(2, "nasdaq_all", "NASDAQ", "NASDAQ")
        _exchange_block(3, "nyse_all", "NYSE", "NYSE")

        # Financial data filter
        fin_row = ttk.Frame(self._universe_preset_frame)
        fin_row.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        ttk.Checkbutton(
            fin_row,
            text="재무제표 데이터 있는 종목만 (local 파일 기준)",
            variable=self.universe_financial_only_var,
        ).pack(side="left")

        # Direct ticker input frame
        self._universe_direct_frame = ttk.Frame(uni_row)
        self._universe_direct_frame.grid(row=1, column=0, sticky="ew")
        self._universe_direct_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(self._universe_direct_frame, text="티커 (쉼표 구분):").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(
            self._universe_direct_frame, textvariable=self.universe_direct_tickers_var, width=50,
        ).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(
            self._universe_direct_frame,
            text="예: AAPL, MSFT, GOOGL, AMZN",
            style="Dim.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w")

        # Initialize visibility
        self._on_universe_input_mode_changed()

        detail_row = ttk.Frame(parent)
        detail_row.grid(row=5, column=0, sticky="ew")
        detail_row.grid_columnconfigure(1, weight=1)
        detail_row.grid_columnconfigure(3, weight=1)
        ttk.Label(detail_row, text=tr("universe_exchanges")).grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(detail_row, textvariable=self.universe_exchanges_var).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(detail_row, text=tr("universe_size_buckets")).grid(row=0, column=2, sticky="w", padx=(8, 4), pady=2)
        ttk.Entry(detail_row, textvariable=self.universe_size_buckets_var).grid(row=0, column=3, sticky="ew", pady=2)
        ttk.Label(detail_row, text=tr("universe_exclude_tickers")).grid(row=1, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(detail_row, textvariable=self.universe_exclude_tickers_var).grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Label(detail_row, text="관심그룹 이름").grid(row=2, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(detail_row, textvariable=self.universe_watchlist_name_var).grid(row=2, column=1, sticky="ew", pady=2)

        btn_row = ttk.Frame(parent)
        btn_row.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(btn_row, text="업종/세부업종 선택...", command=self._open_universe_sector_dialog).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="전체해제", command=self._clear_universe_filters).pack(side="left")
        ttk.Label(btn_row, textvariable=self.universe_summary_var, style="Dim.TLabel").pack(side="left", padx=(12, 0))

        pit_row = ttk.Frame(parent)
        pit_row.grid(row=7, column=0, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(pit_row, text="재무 PIT 적용 (Fundamentals PIT)", variable=self.use_fundamentals_pit_var).pack(side="left")
        ttk.Checkbutton(pit_row, text="Strict 정책 (데이터 부족/미래 참조 시 실행 차단)", variable=self.strict_pit_var).pack(side="left", padx=(15, 0))

        ttk.Label(
            parent,
            text=tr("pit_guard_hint"),
            style="Dim.TLabel",
            wraplength=980,
            justify="left",
        ).grid(row=8, column=0, sticky="w", pady=(4, 0))

    def _toggle_section_body(self, *, flag_var: tk.BooleanVar, body: ttk.Frame, btn: ttk.Checkbutton) -> None:
        if bool(flag_var.get()):
            body.grid()
            btn.configure(text=tr("advanced_on"))
        else:
            body.grid_remove()
            btn.configure(text=tr("advanced_off"))
        self._apply_responsive_layout()

    def _update_buy_section_visibility(self) -> None:
        self._toggle_section_body(flag_var=self.buy_section_open_var, body=self.buy_body_frame, btn=self.buy_toggle_btn)

    def _update_sell_section_visibility(self) -> None:
        self._toggle_section_body(flag_var=self.sell_section_open_var, body=self.sell_body_frame, btn=self.sell_toggle_btn)

    def _update_universe_section_visibility(self) -> None:
        self._toggle_section_body(flag_var=self.universe_section_open_var, body=self.universe_body_frame, btn=self.universe_toggle_btn)

    def _bump_numeric(self, var: tk.StringVar, delta: float) -> None:
        current = self._parse_float(var.get(), 0.0) or 0.0
        out = current + float(delta)
        if abs(out - round(out)) < 1e-9:
            var.set(str(int(round(out))))
        else:
            var.set(f"{out:.4g}")

    def _reset_buy_conditions(self) -> None:
        self.buy_condition_editor.set_condition_set(ConditionSet())
        self._buy_condition_set = ConditionSet()
        self.buy_condition_editor.show_message("")

    def _reset_sell_conditions(self) -> None:
        self.sell_condition_editor.set_condition_set(ConditionSet())
        self._sell_condition_set = ConditionSet()
        self.sell_condition_editor.show_message("")

    def _on_universe_input_mode_changed(self) -> None:
        mode = str(self.universe_input_mode_var.get())
        if mode == "direct":
            self._universe_preset_frame.grid_remove()
            self._universe_direct_frame.grid()
        else:
            self._universe_direct_frame.grid_remove()
            self._universe_preset_frame.grid()
        self._update_universe_filter_summary()

    def _set_universe_parent_children(self, exchange: str, enabled: bool) -> None:
        for bucket in UNIVERSE_BUCKETS:
            self.universe_bucket_vars[exchange][bucket].set(bool(enabled))

    def _refresh_universe_parent_status(self, exchange: str) -> None:
        selected = [b for b in UNIVERSE_BUCKETS if bool(self.universe_bucket_vars[exchange][b].get())]
        all_on, count, total = summarize_universe_parent_state(selected)
        preset_key = f"{exchange.lower()}_all"
        if all_on:
            self.universe_preset_vars[preset_key].set(True)
            self.universe_parent_status_vars[exchange].set("")
        else:
            self.universe_preset_vars[preset_key].set(False)
            self.universe_parent_status_vars[exchange].set(f"({count}/{total})" if count > 0 else "")

    def _on_universe_parent_toggled(self, preset_key: str, exchange: str | None = None) -> None:
        if preset_key == "sp500":
            if bool(self.universe_preset_vars["sp500"].get()):
                self.universe_preset_vars["sp500_pit"].set(False)
            self._update_universe_filter_summary()
            return
        if preset_key == "sp500_pit":
            if bool(self.universe_preset_vars["sp500_pit"].get()):
                self.universe_preset_vars["sp500"].set(False)
            self._update_universe_filter_summary()
            return
        if preset_key == "nasdaq_stock_only_financial":
            self._update_universe_filter_summary()
            return
        if exchange is None:
            return
        enabled = bool(self.universe_preset_vars[preset_key].get())
        self._set_universe_parent_children(exchange, enabled)
        self.universe_parent_status_vars[exchange].set("" if enabled else "")
        self._sync_universe_text_filters_from_presets()
        self._update_universe_filter_summary()

    def _on_universe_bucket_toggled(self, exchange: str) -> None:
        self._refresh_universe_parent_status(exchange)
        self._sync_universe_text_filters_from_presets()
        self._update_universe_filter_summary()

    def _collect_universe_preset_state(self) -> tuple[list[str], dict[str, list[str]]]:
        presets: list[str] = []
        for key in ["sp500", "sp500_pit", "nasdaq_stock_only_financial", "nasdaq_all", "nyse_all"]:
            if bool(self.universe_preset_vars[key].get()):
                presets.append(key)
        bucket_sel: dict[str, list[str]] = {}
        for exch in ["NASDAQ", "NYSE"]:
            bucket_sel[exch] = [b for b in UNIVERSE_BUCKETS if bool(self.universe_bucket_vars[exch][b].get())]
        return presets, bucket_sel

    def _sync_universe_text_filters_from_presets(self) -> None:
        presets, buckets = self._collect_universe_preset_state()
        mapped = map_universe_presets_to_filters(
            presets=presets,
            bucket_selections=buckets,
            manual_exchanges=[],
            manual_size_buckets=[],
        )
        exchanges = mapped.get("exchanges", [])
        sizes = mapped.get("size_buckets", [])
        self.universe_exchanges_var.set(", ".join(exchanges))
        self.universe_size_buckets_var.set(", ".join(sizes))

    def _discover_sector_options(self) -> tuple[list[str], list[str], dict[str, list[str]], dict[str, int], dict[str, int]]:
        """Returns (sectors, subsectors, l1_to_l2, sector_counts, subsector_counts)."""
        sectors: set[str] = set()
        subsectors: set[str] = set()
        l1_to_l2: dict[str, list[str]] = {}
        sector_counts: dict[str, int] = {}
        subsector_counts: dict[str, int] = {}

        # Build L1→L2 mapping from SIC rules
        rules_path = Path("data/reference/sic_to_sector_proxy_rules.csv")
        if rules_path.exists():
            try:
                rules = pd.read_csv(rules_path)
                if "sector_l1_kr" in rules.columns:
                    sectors.update([str(v).strip() for v in rules["sector_l1_kr"].dropna() if str(v).strip()])
                if "sector_l2_kr" in rules.columns:
                    subsectors.update([str(v).strip() for v in rules["sector_l2_kr"].dropna() if str(v).strip()])
                if "sector_l1_kr" in rules.columns and "sector_l2_kr" in rules.columns:
                    for _, row in rules.dropna(subset=["sector_l1_kr", "sector_l2_kr"]).iterrows():
                        l1 = str(row["sector_l1_kr"]).strip()
                        l2 = str(row["sector_l2_kr"]).strip()
                        if l1 and l2:
                            l1_to_l2.setdefault(l1, [])
                            if l2 not in l1_to_l2[l1]:
                                l1_to_l2[l1].append(l2)
            except Exception:
                pass

        # Count companies from cached sector proxy files (best-effort, fast-fail)
        proxy_dir = Path("data/sec_identity_cache/ticker_sector_proxy")
        if proxy_dir.exists():
            try:
                dfs = [
                    pd.read_parquet(f, columns=["sector_l1_kr", "sector_l2_kr"])
                    for f in proxy_dir.glob("*.parquet")
                ]
                if dfs:
                    df_all = pd.concat(dfs, ignore_index=True)
                    sector_counts = df_all["sector_l1_kr"].astype(str).value_counts().to_dict()
                    subsector_counts = df_all["sector_l2_kr"].astype(str).value_counts().to_dict()
                    # Add any sectors from counts not already in set
                    sectors.update(k for k in sector_counts if k and k != "nan")
                    subsectors.update(k for k in subsector_counts if k and k != "nan")
            except Exception:
                pass

        # Ensure 미분류 is always present
        sectors.add("미분류")
        subsectors.add("미분류")

        if not sectors:
            sectors.update(["에너지", "소재", "산업재", "IT", "필수 소비재", "경기 소비재", "헬스케어", "금융", "통신 및 유틸리티", "부동산", "기타", "미분류"])
        if not subsectors:
            subsectors.update(["에너지", "화학 소재", "기초 소재", "자본재", "소프트웨어", "IT 서비스", "제약 및 생명공학", "은행", "통신 서비스", "유틸리티", "리츠", "미분류"])

        # Sort: 미분류 last
        def _sort_key(x: str) -> tuple:
            return (1 if x == "미분류" else 0, x)

        sorted_sectors = sorted(sectors, key=_sort_key)
        sorted_subsectors = sorted(subsectors, key=_sort_key)
        return sorted_sectors, sorted_subsectors, l1_to_l2, sector_counts, subsector_counts

    def _open_universe_sector_dialog(self) -> None:
        sectors, subsectors, l1_to_l2, sector_counts, subsector_counts = self._discover_sector_options()
        dlg = UniverseSectorDialog(
            self.parent,
            sectors=sectors,
            subsectors=subsectors,
            l1_to_l2=l1_to_l2,
            sector_counts=sector_counts,
            subsector_counts=subsector_counts,
            include_sectors=self._csv_values(self.universe_include_sectors_var.get()),
            exclude_sectors=self._csv_values(self.universe_exclude_sectors_var.get()),
            include_subsectors=self._csv_values(self.universe_include_subsectors_var.get()),
            exclude_subsectors=self._csv_values(self.universe_exclude_subsectors_var.get()),
        )
        result = dlg.show()
        if not result:
            return
        self.universe_include_sectors_var.set(", ".join(result.get("include_sectors", [])))
        self.universe_exclude_sectors_var.set(", ".join(result.get("exclude_sectors", [])))
        self.universe_include_subsectors_var.set(", ".join(result.get("include_subsectors", [])))
        self.universe_exclude_subsectors_var.set(", ".join(result.get("exclude_subsectors", [])))
        self._update_universe_filter_summary()

    def _clear_universe_filters(self) -> None:
        self.stock_type_included_var.set(True)
        self.ptp_included_var.set(False)
        self.universe_financial_only_var.set(False)
        self.universe_input_mode_var.set("preset")
        self.universe_direct_tickers_var.set("")
        for key in self.universe_preset_vars:
            self.universe_preset_vars[key].set(False)
        for exch in ["NASDAQ", "NYSE"]:
            self.universe_parent_status_vars[exch].set("")
            for b in UNIVERSE_BUCKETS:
                self.universe_bucket_vars[exch][b].set(False)
        self.universe_exchanges_var.set("")
        self.universe_size_buckets_var.set("")
        self.universe_include_sectors_var.set("")
        self.universe_exclude_sectors_var.set("")
        self.universe_include_subsectors_var.set("")
        self.universe_exclude_subsectors_var.set("")
        self.universe_exclude_tickers_var.set("")
        self.universe_watchlist_name_var.set("")
        self._on_universe_input_mode_changed()
        self._update_universe_filter_summary()

    def _update_universe_filter_summary(self) -> None:
        sectors = self._csv_values(self.universe_include_sectors_var.get())
        subsectors = self._csv_values(self.universe_include_subsectors_var.get())
        excludes = self._csv_values(self.universe_exclude_tickers_var.get())

        mode = str(getattr(self, "universe_input_mode_var", None) and self.universe_input_mode_var.get() or "preset")
        if mode == "direct":
            direct_tickers = self._csv_values(self.universe_direct_tickers_var.get(), upper=True)
            universe_txt = f"직접입력({len(direct_tickers)}개)" if direct_tickers else "직접입력(없음)"
        else:
            presets, buckets = self._collect_universe_preset_state()
            parts: list[str] = []
            if "sp500" in presets:
                parts.append("SP500")
            if "sp500_pit" in presets:
                parts.append("SP500-PIT")
            if "nasdaq_stock_only_financial" in presets:
                parts.append("NASDAQ-재무")
            for preset, exch in [("nasdaq_all", "NASDAQ"), ("nyse_all", "NYSE")]:
                if preset in presets:
                    chosen = buckets.get(exch, [])
                    if len(chosen) == len(UNIVERSE_BUCKETS):
                        parts.append(f"{exch}(전체)")
                    elif chosen:
                        parts.append(f"{exch}({'/'.join(chosen)})")
            universe_txt = " + ".join(parts) if parts else "-"

        self.universe_summary_var.set(
            f"유니버스: {universe_txt}, 업종 {len(sectors)}개, 세부업종 {len(subsectors)}개, 제외종목 {len(excludes)}개"
        )

    def _sync_condition_editors_from_state(self) -> None:
        self.buy_condition_editor.set_condition_set(self._buy_condition_set)
        self.sell_condition_editor.set_condition_set(self._sell_condition_set)

    def _sync_condition_sets_from_editors(self) -> None:
        try:
            self._buy_condition_set = self.buy_condition_editor.get_condition_set(na_policy="fail")
        except Exception:
            pass
        try:
            self._sell_condition_set = self.sell_condition_editor.get_condition_set(na_policy="fail")
        except Exception:
            pass

    def _on_root_configure(self, event: tk.Event) -> None:
        if event.widget not in {self.root_frame, self.scrollable.canvas}:
            return
        if self._responsive_job is not None:
            self.parent.after_cancel(self._responsive_job)
        self._responsive_job = self.parent.after(150, self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        self._responsive_job = None
        width_px = max(1, int(self.scrollable.canvas.winfo_width()))
        self.scrollable.sync_inner_width()
        self._relayout_header_sections(width_px)
        self._resize_plot_canvases()
        self.scrollable.schedule_scrollregion_update()

    def _layout_section_items(
        self,
        container: ttk.Frame,
        items: list[tuple[ttk.Frame, bool]],
        cols: int,
    ) -> None:
        cols = max(1, int(cols))
        for col in range(8):
            container.grid_columnconfigure(col, weight=0, uniform="")
        for col in range(cols):
            container.grid_columnconfigure(col, weight=1, uniform=f"sec_{id(container)}")

        for frame, _full in items:
            frame.grid_forget()

        row = 0
        col = 0
        for frame, full in items:
            if full:
                if col != 0:
                    row += 1
                    col = 0
                frame.grid(row=row, column=0, columnspan=cols, sticky="ew", padx=(0, 8), pady=(0, 4))
                row += 1
                continue

            frame.grid(row=row, column=col, sticky="ew", padx=(0, 8), pady=(0, 4))
            col += 1
            if col >= cols:
                row += 1
                col = 0

    def _update_summary_wraplength(self, width_px: int) -> None:
        wrap = max(420, int(width_px) - 80)
        for label in getattr(self, "_summary_labels", []):
            try:
                label.configure(wraplength=wrap)
            except Exception:
                pass

    def _relayout_header_sections(self, width_px: int) -> None:
        strategy_cols = 2 if width_px >= 1500 else 1
        core_cols = 6 if width_px >= 1800 else (4 if width_px >= 1400 else (3 if width_px >= 1200 else 2))
        advanced_cols = 5 if width_px >= 1800 else (4 if width_px >= 1500 else (3 if width_px >= 1200 else 2))
        funding_cols = 5 if width_px >= 1800 else (4 if width_px >= 1500 else (3 if width_px >= 1200 else 2))
        action_cols = 2 if width_px >= 1300 else 1

        self._layout_section_items(self.strategy_section, self._strategy_items, strategy_cols)
        self._layout_section_items(self.backtest_core_frame, self._backtest_core_items, core_cols)
        self._layout_section_items(self.backtest_advanced_frame, self._backtest_advanced_items, advanced_cols)

        funding_items = [(self._funding_item_frames[key], False) for key in self._funding_visible_keys if key in self._funding_item_frames]
        self._layout_section_items(self.funding_items_container, funding_items, funding_cols)
        self._layout_section_items(self.action_section, self._action_items, action_cols)
        self._update_summary_wraplength(width_px)

    def _resize_plot_canvases(self) -> None:
        chart_width = max(420, int(self.charts_frame.winfo_width()))
        chart_height = max(640, int(self.charts_frame.winfo_height()))
        summary_width = max(420, int(self.trade_summary_frame.winfo_width()))
        summary_height = max(460, int(self.trade_summary_frame.winfo_height()))
        size_key = (chart_width, chart_height, summary_width, summary_height)

        if self._last_chart_size == size_key:
            return
        self._last_chart_size = size_key

        dpi_main = float(self.fig.get_dpi() or 100.0)
        dpi_summary = float(self.trade_summary_fig.get_dpi() or 100.0)
        self.fig.set_size_inches(chart_width / dpi_main, chart_height / dpi_main, forward=True)
        self.trade_summary_fig.set_size_inches(summary_width / dpi_summary, summary_height / dpi_summary, forward=True)
        self.canvas.draw_idle()
        self.trade_summary_canvas.draw_idle()
        self.scrollable.schedule_scrollregion_update()

    def _make_tree(self, parent: ttk.Widget, columns: list[str], widths: list[int], pack: bool = True) -> ttk.Treeview:
        frame = ttk.Frame(parent) if pack else parent
        if pack:
            frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        for col, width in zip(columns, widths):
            tree.heading(col, text=COL_TXT.get(col, col))
            anchor = "w" if col in {"ticker", "reason", "snapshot_path", "metric", "source", "mode", "mode_name", "mode_label", "funding_mode", "contribution_freq", "va_min_policy", "event_order_tag", "rebalance_exec_timing"} else "e"
            tree.column(col, width=width, anchor=anchor)
        ysb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        if pack:
            tree.pack(side="left", fill="both", expand=True)
            ysb.pack(side="right", fill="y")
            xsb.pack(side="bottom", fill="x")
        else:
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            tree.grid(row=0, column=0, sticky="nsew")
            ysb.grid(row=0, column=1, sticky="ns")
            xsb.grid(row=1, column=0, sticky="ew")
        configure_treeview_tags(tree)
        return tree

    def _parse_float(self, raw: str, default: float | None = None) -> float | None:
        text = str(raw or "").strip()
        if not text:
            return default
        text = text.replace(",", "").replace("_", "")
        try:
            return float(text)
        except Exception:
            return default

    def _parse_int(self, raw: str, default: int) -> int:
        text = str(raw or "").strip()
        if not text:
            return default
        text = text.replace(",", "").replace("_", "")
        try:
            return int(text)
        except Exception:
            return default

    def _warn_funding_parse(self, field_label: str, raw: str, fallback: str) -> None:
        warning = f"{field_label}: '{raw}' 입력값을 해석하지 못해 {fallback} 사용"
        if warning not in self._funding_parse_warnings:
            self._funding_parse_warnings.append(warning)
        LOGGER.warning("Funding parse fallback: %s", warning)

    def _parse_funding_float(
        self,
        raw: str,
        default: float,
        *,
        field_label: str,
    ) -> float:
        text = str(raw or "").strip()
        if not text:
            return float(default)
        norm = text.replace(",", "").replace("_", "")
        try:
            return float(norm)
        except Exception:
            self._warn_funding_parse(field_label, text, f"{float(default):,.0f}")
            return float(default)

    def _parse_funding_optional_float(
        self,
        raw: str,
        *,
        field_label: str,
    ) -> float | None:
        text = str(raw or "").strip()
        if not text:
            return None
        norm = text.replace(",", "").replace("_", "")
        try:
            return float(norm)
        except Exception:
            self._warn_funding_parse(field_label, text, "빈값(None)")
            return None

    def _publish_funding_parse_warnings(self) -> None:
        if not self._funding_parse_warnings:
            return
        shown = self._funding_parse_warnings[:2]
        extra = len(self._funding_parse_warnings) - len(shown)
        detail = "; ".join(shown)
        if extra > 0:
            detail = f"{detail} ...(+{extra})"
        prefix = tr("funding_parse_warning")
        current = str(self.funding_info_var.get() or "").strip()
        text = f"{prefix}: {detail}"
        self.funding_info_var.set(f"{current} | {text}" if current else text)
        self._funding_parse_warnings = []

    def _effective_market(self) -> str:
        market = str(self.market_var.get() or "us").strip().lower()
        return "us" if market == "auto" else (market or "us")

    def _table_source_internal(self) -> str:
        val = str(self.table_source_var.get() or tr("source_auto")).strip()
        if val == tr("source_screen"):
            return "screen"
        if val == tr("source_strategy"):
            return "strategy"
        return "auto"

    def _chart_mode_internal(self) -> str:
        val = str(self.lower_chart_mode_var.get() or tr("chart_mode_relative")).strip()
        if val == tr("chart_mode_drawdown"):
            return "drawdown"
        return "relative"

    def _share_mode_internal(self) -> str:
        val = str(self.share_mode_var.get() or tr("share_mode_fractional")).strip()
        if val == tr("share_mode_integer"):
            return "integer"
        return "fractional"

    def _sell_mode_internal(self) -> str:
        val = str(self.sell_mode_ui_var.get() or tr("sell_mode_b")).strip()
        if val == tr("sell_mode_a"):
            return "A"
        if val == tr("sell_mode_c"):
            return "C"
        return "B"

    def _exec_price_basis_internal(self) -> str:
        val = str(self.exec_price_basis_var.get() or tr("price_basis_auto")).strip()
        if val == tr("price_basis_open"):
            return "open"
        if val == tr("price_basis_close"):
            return "close"
        if val == tr("price_basis_prev_close"):
            return "prev_close"
        return "auto"

    def _build_position_sizing(self) -> PositionSizingConfig:
        return PositionSizingConfig(
            position_weight_pct=self._parse_float(self.position_weight_pct_var.get(), None),
            max_holdings=self._parse_int(self.max_holdings_cfg_var.get(), 0) or None,
            max_new_buys_per_day=self._parse_int(self.max_new_buys_var.get(), 0) or None,
            max_buy_amount_per_position=self._parse_float(self.max_buy_amount_var.get(), None),
        )

    def _build_risk_limits(self) -> RiskLimitsConfig:
        return RiskLimitsConfig(
            min_cash_reserve_pct=float(self._parse_float(self.min_cash_reserve_pct_var.get(), 0.0) or 0.0),
        )

    def _build_execution_config(self) -> ExecutionConfig:
        return ExecutionConfig(
            timing=str(self.exec_timing_var.get() or "next_open").strip().lower(),
            price_basis=self._exec_price_basis_internal(),
            price_offset_pct=float(self._parse_float(self.price_offset_pct_var.get(), 0.0) or 0.0),
            commission_pct=None,
            slippage_pct=None,
        )

    def _build_universe_selection_mapping(self) -> dict[str, Any]:
        presets, buckets = self._collect_universe_preset_state()
        return map_universe_presets_to_filters(
            presets=presets,
            bucket_selections=buckets,
            manual_exchanges=self._csv_values(self.universe_exchanges_var.get(), upper=True),
            manual_size_buckets=self._csv_values(self.universe_size_buckets_var.get(), lower=True),
        )

    def _build_universe_filter_config(self) -> UniverseFilterConfig:
        mapped = self._build_universe_selection_mapping()
        return UniverseFilterConfig(
            exchanges=list(mapped.get("exchanges", [])),
            size_buckets=list(mapped.get("size_buckets", [])),
            include_sectors=self._csv_values(self.universe_include_sectors_var.get()),
            exclude_sectors=self._csv_values(self.universe_exclude_sectors_var.get()),
            include_subsectors=self._csv_values(self.universe_include_subsectors_var.get()),
            exclude_subsectors=self._csv_values(self.universe_exclude_subsectors_var.get()),
            exclude_tickers=self._csv_values(self.universe_exclude_tickers_var.get(), upper=True),
            watchlist_name=str(self.universe_watchlist_name_var.get() or "").strip() or None,
            universe_presets=list(mapped.get("presets", [])),
            universe_bucket_selections=normalize_universe_bucket_selections(mapped.get("bucket_selections")),
            stock_type_included=bool(self.stock_type_included_var.get()),
            ptp_included=bool(self.ptp_included_var.get()),
            financial_data_only=bool(self.universe_financial_only_var.get()),
        )

    def _apply_universe_selection_to_strategy_config(self, cfg: StrategyConfig) -> None:
        # Direct ticker input mode overrides all preset selection
        if str(self.universe_input_mode_var.get()) == "direct":
            direct_tickers = self._csv_values(self.universe_direct_tickers_var.get(), upper=True)
            if direct_tickers:
                cfg.universe.symbols = direct_tickers
                cfg.universe.source = "symbols"
                cfg.universe.symbols_file = None
            cfg.universe_filter.financial_data_only = bool(self.universe_financial_only_var.get())
            is_strict = bool(self.strict_pit_var.get())
            cfg.use_fundamentals_pit = bool(self.use_fundamentals_pit_var.get())
            cfg.strict_pit = is_strict
            cfg.universe.sp500_pit_strict = is_strict
            cfg.universe.sp500_pit_fail_closed = is_strict
            cfg.fundamentals_missing_policy = "error" if is_strict else "exclude"
            return

        mapped = self._build_universe_selection_mapping()
        cfg.universe_filter.exchanges = list(mapped.get("exchanges", []))
        cfg.universe_filter.size_buckets = list(mapped.get("size_buckets", []))
        cfg.universe_filter.universe_presets = list(mapped.get("presets", []))
        cfg.universe_filter.universe_bucket_selections = normalize_universe_bucket_selections(
            mapped.get("bucket_selections")
        )
        cfg.universe_filter.stock_type_included = bool(self.stock_type_included_var.get())
        cfg.universe_filter.ptp_included = bool(self.ptp_included_var.get())
        cfg.universe_filter.financial_data_only = bool(self.universe_financial_only_var.get())
        source = str(mapped.get("universe_source", "symbols"))
        if source in {"sp500", "sp500_pit", "nasdaq_stock_only_financial"}:
            cfg.universe.source = source
            cfg.universe.symbols = []
            cfg.universe.symbols_file = None
        elif str(cfg.universe.source).strip().lower() in {"sp500", "sp500_pit", "nasdaq_stock_only_financial"}:
            cfg.universe.source = "symbols"

        is_strict = bool(self.strict_pit_var.get())
        cfg.use_fundamentals_pit = bool(self.use_fundamentals_pit_var.get())
        cfg.strict_pit = is_strict
        cfg.universe.sp500_pit_strict = is_strict
        cfg.universe.sp500_pit_fail_closed = is_strict
        cfg.fundamentals_missing_policy = "error" if is_strict else "exclude"

    def _funding_mode_internal(self) -> str:
        val = str(self.funding_mode_var.get() or tr("funding_mode_lump")).strip()
        if val == tr("funding_mode_dca"):
            return "dca"
        if val == tr("funding_mode_va"):
            return "va"
        if val == tr("funding_mode_compare"):
            return "compare"
        return "lump_sum"

    def _funding_view_internal(self) -> str:
        val = str(self.funding_view_var.get() or tr("funding_mode_lump")).strip()
        if val == tr("funding_mode_dca"):
            return "dca"
        if val == tr("funding_mode_va"):
            return "va"
        return "lump_sum"

    def _funding_compare_basis_internal(self) -> str:
        val = str(self.funding_compare_basis_var.get() or tr("funding_compare_align_dca")).strip()
        if val == tr("funding_compare_custom"):
            return "custom_total"
        if val == tr("funding_compare_align_dca"):
            return "align_lump_to_dca_total"
        return "default"

    def _va_min_policy_internal(self) -> str:
        val = str(self.funding_va_min_policy_var.get() or tr("va_min_policy_every_rebalance")).strip()
        if val == tr("va_min_policy_positive_raw_only"):
            return "positive_raw_only"
        return "every_rebalance"

    def _funding_contribution_freq_internal(self) -> str:
        val = str(self.funding_contrib_freq_var.get() or tr("funding_freq_monthly")).strip()
        if val == tr("funding_freq_yearly"):
            return "Y"
        if val == tr("funding_freq_quarterly"):
            return "Q"
        return "M"

    def _funding_lump_mode_label(self) -> str:
        meta = self.funding_compare_meta if isinstance(self.funding_compare_meta, dict) else {}
        aligned = bool(meta.get("alignment_applied", False))
        align_mode = str(meta.get("alignment_mode", "default"))
        if aligned and align_mode in {"align_lump_to_dca_total", "custom_total"}:
            return tr("funding_lump_aligned")
        return tr("funding_mode_lump")

    def _update_advanced_visibility(self) -> None:
        if bool(self.show_advanced_var.get()):
            self.backtest_advanced_frame.grid()
            self.advanced_toggle_btn.configure(text=tr("advanced_on"))
        else:
            self.backtest_advanced_frame.grid_remove()
            self.advanced_toggle_btn.configure(text=tr("advanced_off"))
        self._apply_responsive_layout()

    def _update_funding_field_visibility(self) -> None:
        mode = self._funding_mode_internal()
        if mode == "lump_sum":
            keys = ["mode", "contribution_freq", "initial_cash"]
        elif mode == "dca":
            keys = ["mode", "contribution_freq", "initial_cash", "dca"]
        elif mode == "va":
            keys = ["mode", "contribution_freq", "initial_cash", "va_step", "va_min", "va_min_policy", "va_max", "va_withdraw"]
        else:
            keys = ["mode", "contribution_freq", "initial_cash", "dca", "va_step", "va_min", "va_min_policy", "va_max", "va_withdraw", "compare_basis"]
            if self._funding_compare_basis_internal() == "custom_total":
                keys.append("custom_total")
        self._funding_visible_keys = keys

    def _on_funding_compare_basis_changed(self) -> None:
        self._update_funding_field_visibility()
        self._update_funding_preview()
        self._apply_responsive_layout()

    def _on_funding_mode_changed(self) -> None:
        self._update_funding_field_visibility()
        mode = self._funding_mode_internal()
        try:
            self.funding_contrib_freq_combo.configure(state="disabled" if mode == "lump_sum" else "readonly")
        except Exception:
            pass
        self._update_funding_preview()
        self._apply_responsive_layout()

    def _update_funding_preview(self) -> None:
        mode = self._funding_mode_internal()
        dca = float(self._parse_float(self.funding_dca_var.get(), 0.0) or 0.0)
        va_step = float(self._parse_float(self.funding_va_step_var.get(), 0.0) or 0.0)
        va_min = float(self._parse_float(self.funding_va_min_var.get(), 0.0) or 0.0)
        va_max = self._parse_float(self.funding_va_max_var.get(), None)
        va_min_policy = self._va_min_policy_internal()
        va_min_policy_label = (
            tr("va_min_policy_every_rebalance")
            if va_min_policy == "every_rebalance"
            else tr("va_min_policy_positive_raw_only")
        )
        contribution_freq = self._funding_contribution_freq_internal()
        compare_basis = self._funding_compare_basis_internal()
        custom_total = self._parse_float(self.funding_custom_total_var.get(), None)
        mode_txt = tr("funding_mode_lump")
        if mode == "dca":
            mode_txt = tr("funding_mode_dca")
        elif mode == "va":
            mode_txt = tr("funding_mode_va")
        elif mode == "compare":
            mode_txt = tr("funding_mode_compare")

        if mode in {"va", "compare"}:
            v1 = va_step
            v2 = va_step * 2.0
            v3 = va_step * 3.0
            va_max_txt = "-" if va_max is None else f"{float(va_max):,.0f}"
            preview = (
                f"{tr('funding_help')} | {mode_txt} | "
                f"target(1~3)={v1:,.0f}/{v2:,.0f}/{v3:,.0f}, "
                f"freq={contribution_freq}, min={va_min:,.0f}, max={va_max_txt}, policy={va_min_policy_label}"
            )
        elif mode == "dca":
            preview = f"{tr('funding_help')} | {mode_txt} | freq={contribution_freq}, fixed={dca:,.0f}"
        else:
            preview = f"{tr('funding_help')} | {mode_txt}"
        if mode == "compare":
            if compare_basis == "align_lump_to_dca_total":
                preview += f" | {tr('funding_compare_align_dca')}"
            elif compare_basis == "custom_total":
                custom_txt = "-" if custom_total is None else f"{float(custom_total):,.0f}"
                preview += f" | {tr('funding_compare_custom')}={custom_txt}"
            else:
                preview += f" | {tr('funding_compare_default')}"
        if mode in {"va", "compare"}:
            preview += f" | {tr('va_min_policy_hint')}"
        if mode in {"dca", "va", "compare"}:
            preview += f" | {tr('funding_freq_hint')} | {tr('funding_event_order_hint')}"
        self.funding_info_var.set(preview)

    def _build_funding_config(self, mode: str | None = None) -> FundingConfig:
        self._funding_parse_warnings = []
        funding_mode = mode or self._funding_mode_internal()
        if funding_mode == "compare":
            funding_mode = "lump_sum"
        initial_cash = self._parse_funding_float(self.initial_cash_var.get(), 1_000_000.0, field_label=tr("initial_cash"))
        fixed_contribution = self._parse_funding_float(self.funding_dca_var.get(), 0.0, field_label=tr("funding_dca_amount"))
        va_target_step = self._parse_funding_float(self.funding_va_step_var.get(), 0.0, field_label=tr("funding_va_step"))
        va_min = self._parse_funding_float(self.funding_va_min_var.get(), 0.0, field_label=tr("funding_va_min"))
        va_max = self._parse_funding_optional_float(self.funding_va_max_var.get(), field_label=tr("funding_va_max"))
        va_min_policy = self._va_min_policy_internal()
        cfg = FundingConfig(
            mode=funding_mode,
            initial_cash=float(initial_cash),
            contribution_freq=self._funding_contribution_freq_internal(),
            fixed_contribution=float(fixed_contribution),
            va_target_step=float(va_target_step),
            va_target_base=0.0,
            va_min_contribution=float(va_min),
            va_min_policy=va_min_policy,
            va_max_contribution=(float(va_max) if va_max is not None else None),
            va_allow_withdrawal=bool(self.funding_va_withdraw_var.get()),
        )
        self._publish_funding_parse_warnings()
        return cfg

    def _build_funding_variants(self) -> list[FundingConfig]:
        self._funding_parse_warnings = []
        base_initial = self._parse_funding_float(self.initial_cash_var.get(), 1_000_000.0, field_label=tr("initial_cash"))
        freq = self._funding_contribution_freq_internal()
        dca = self._parse_funding_float(self.funding_dca_var.get(), 0.0, field_label=tr("funding_dca_amount"))
        va_step = self._parse_funding_float(self.funding_va_step_var.get(), 0.0, field_label=tr("funding_va_step"))
        va_min = self._parse_funding_float(self.funding_va_min_var.get(), 0.0, field_label=tr("funding_va_min"))
        va_max = self._parse_funding_optional_float(self.funding_va_max_var.get(), field_label=tr("funding_va_max"))
        va_min_policy = self._va_min_policy_internal()
        allow_withdraw = bool(self.funding_va_withdraw_var.get())
        out = [
            FundingConfig(mode="lump_sum", initial_cash=base_initial, contribution_freq=freq),
            FundingConfig(
                mode="dca",
                initial_cash=base_initial,
                contribution_freq=freq,
                fixed_contribution=dca,
            ),
            FundingConfig(
                mode="va",
                initial_cash=base_initial,
                contribution_freq=freq,
                va_target_step=va_step,
                va_target_base=0.0,
                va_min_contribution=va_min,
                va_min_policy=va_min_policy,
                va_max_contribution=(float(va_max) if va_max is not None else None),
                va_allow_withdrawal=allow_withdraw,
            ),
        ]
        self._publish_funding_parse_warnings()
        return out

    def _trade_metric_internal(self) -> str:
        val = str(self.trade_metric_var.get() or tr("metric_net_notional")).strip()
        return "net_count" if val == tr("metric_net_count") else "net_notional"

    def _trade_scope_internal(self) -> str:
        val = str(self.trade_scope_var.get() or tr("scope_all")).strip()
        return "top_n" if val == tr("scope_top_n") else "all"

    def _trade_sort_internal(self) -> str:
        val = str(self.trade_sort_var.get() or tr("sort_notional")).strip()
        return "count" if val == tr("sort_count") else "notional"

    def _return_plot_mode_internal(self) -> str:
        val = str(self.return_plot_mode_var.get() or tr("ret_mode_strat_bench")).strip()
        if val == tr("ret_mode_strat_excess"):
            return "strategy_excess"
        if val == tr("ret_mode_excess_only"):
            return "excess_only"
        return "strategy_bench"

    def _cum_line_mode_internal(self) -> str:
        val = str(self.cum_line_mode_var.get() or tr("cum_mode_strategy")).strip()
        if val == tr("cum_mode_excess"):
            return "cum_excess"
        if val == tr("cum_mode_none"):
            return "none"
        return "cum_strategy"

    @staticmethod
    def _apply_year_month_ticks(ax: Any, show_major_labels: bool) -> None:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.tick_params(axis="x", which="minor", labelbottom=False, length=2)
        ax.tick_params(axis="x", which="major", labelsize=8, labelrotation=0)
        if not show_major_labels:
            ax.tick_params(axis="x", which="major", labelbottom=False)

    def _on_heatmap_inner_configure(self, _event: tk.Event) -> None:
        self._schedule_heatmap_scrollregion()

    def _on_heatmap_canvas_configure(self, event: tk.Event) -> None:
        width = max(1, int(event.width))
        self.heatmap_scroll_canvas.itemconfigure(self.heatmap_window, width=width)
        self._schedule_heatmap_scrollregion()
        if self._heatmap_resize_job is not None:
            try:
                self.parent.after_cancel(self._heatmap_resize_job)
            except tk.TclError:
                pass
        self._heatmap_resize_job = self.parent.after(180, self._on_heatmap_resize_tick)

    def _on_heatmap_resize_tick(self) -> None:
        self._heatmap_resize_job = None
        self._render_trade_analytics()

    def _schedule_heatmap_scrollregion(self) -> None:
        if self._heatmap_scroll_job is not None:
            self.parent.after_cancel(self._heatmap_scroll_job)
        self._heatmap_scroll_job = self.parent.after(80, self._update_heatmap_scrollregion)

    def _update_heatmap_scrollregion(self) -> None:
        self._heatmap_scroll_job = None
        canvas_w = max(1, int(self.heatmap_scroll_canvas.winfo_width()))
        canvas_h = max(1, int(self.heatmap_scroll_canvas.winfo_height()))
        req_w = max(1, int(self.heatmap_inner.winfo_reqwidth()))
        req_h = max(1, int(self.heatmap_inner.winfo_reqheight()))
        self.heatmap_scroll_canvas.configure(scrollregion=(0, 0, max(canvas_w, req_w), max(canvas_h, req_h)))
        self.heatmap_scroll_canvas.xview_moveto(0.0)

    def _on_heatmap_mousewheel(self, event: tk.Event) -> str | None:
        delta = 0
        if hasattr(event, "num") and event.num == 4:
            delta = 1
        elif hasattr(event, "num") and event.num == 5:
            delta = -1
        elif hasattr(event, "delta"):
            raw = int(event.delta)
            if raw != 0:
                delta = 1 if raw > 0 else -1
        if delta == 0:
            return None
        self.heatmap_scroll_canvas.yview_scroll(-delta, "units")
        return "break"

    def _bind_heatmap_mousewheel(self, _event: tk.Event) -> None:
        self.heatmap_scroll_canvas.bind_all("<MouseWheel>", self._on_heatmap_mousewheel)
        self.heatmap_scroll_canvas.bind_all("<Button-4>", self._on_heatmap_mousewheel)
        self.heatmap_scroll_canvas.bind_all("<Button-5>", self._on_heatmap_mousewheel)

    def _unbind_heatmap_mousewheel(self, _event: tk.Event) -> None:
        self.heatmap_scroll_canvas.unbind_all("<MouseWheel>")
        self.heatmap_scroll_canvas.unbind_all("<Button-4>")
        self.heatmap_scroll_canvas.unbind_all("<Button-5>")
        # Restore outer tab scroll binding when pointer leaves the heatmap.
        self.scrollable._bind_mousewheel(_event)  # noqa: SLF001

    def _set_hover_info(self, text: str) -> None:
        self.hover_info_text.configure(state="normal")
        self.hover_info_text.delete("1.0", tk.END)
        self.hover_info_text.insert("1.0", text or tr("hover_default"))
        self.hover_info_text.configure(state="disabled")

    def _on_tooltip_toggle(self) -> None:
        if not bool(self.show_tooltip_var.get()):
            self.heatmap_tooltip.hide()

    @staticmethod
    def _format_signed_notional(value: float) -> str:
        sign = "+" if value >= 0 else "-"
        return f"{sign}${abs(float(value)):,.0f}"

    @staticmethod
    def _format_signed_count(value: float) -> str:
        sign = "+" if value >= 0 else "-"
        return f"{sign}{int(round(abs(float(value))))}"

    def _heatmap_index_from_event(self, event: Any) -> tuple[int, int] | None:
        if self._heatmap_ax is None:
            return None
        if event is None or event.inaxes != self._heatmap_ax:
            return None
        if event.xdata is None or event.ydata is None:
            return None
        if not self._heatmap_cols or not self._heatmap_rows:
            return None

        x_val = float(event.xdata)
        y_val = float(event.ydata)
        if x_val < -0.5 or x_val > (len(self._heatmap_cols) - 0.5):
            return None
        if y_val < -0.5 or y_val > (len(self._heatmap_rows) - 0.5):
            return None

        x_idx = int(round(x_val))
        y_idx = int(round(y_val))
        if x_idx < 0 or y_idx < 0:
            return None
        if x_idx >= len(self._heatmap_cols) or y_idx >= len(self._heatmap_rows):
            return None
        return x_idx, y_idx

    def _build_hover_text(self, ticker: str, month: str, metric_value: float, trades_sub: pd.DataFrame) -> str:
        metric_mode = self._trade_metric_internal()
        metric_label = tr("metric_net_count") if metric_mode == "net_count" else tr("metric_net_notional")
        if metric_mode == "net_count":
            metric_value_text = self._format_signed_count(metric_value)
        else:
            metric_value_text = self._format_signed_notional(metric_value)

        lines = [f"{month} | {ticker} | {metric_label}: {metric_value_text}"]

        if trades_sub is None or trades_sub.empty:
            lines.append(tr("hover_no_trade"))
            return "\n".join(lines)

        buy_count = int((trades_sub["side"] == "buy").sum()) if "side" in trades_sub.columns else 0
        sell_count = int((trades_sub["side"] == "sell").sum()) if "side" in trades_sub.columns else 0
        net_notional = float(pd.to_numeric(trades_sub.get("signed_notional"), errors="coerce").fillna(0.0).sum())
        lines.append(
            f"BUY {buy_count}건 / SELL {sell_count}건, net_notional={self._format_signed_notional(net_notional)}"
        )

        reason_lines = build_trade_reason_summary_lines(trades_sub, max_lines=5)
        lines.extend(reason_lines[:5])
        return "\n".join(lines)

    def _extract_root_pointer(self, event: Any) -> tuple[int, int] | None:
        gui_event = getattr(event, "guiEvent", None)
        if gui_event is not None and hasattr(gui_event, "x_root") and hasattr(gui_event, "y_root"):
            try:
                return int(gui_event.x_root), int(gui_event.y_root)
            except Exception:
                pass
        try:
            return int(self.heatmap_canvas_widget.winfo_pointerx()), int(self.heatmap_canvas_widget.winfo_pointery())
        except Exception:
            return None

    def _apply_heatmap_hover(self) -> None:
        self._heatmap_hover_job = None
        key = self._heatmap_hover_pending_key
        pointer = self._heatmap_hover_pending_pointer
        if key is None:
            self._heatmap_hover_last_key = None
            self._set_hover_info(tr("hover_default"))
            self.heatmap_tooltip.hide()
            return

        if key == self._heatmap_hover_last_key:
            if bool(self.show_tooltip_var.get()) and pointer is not None:
                self.heatmap_tooltip.move(x_root=pointer[0], y_root=pointer[1])
            return

        x_idx, y_idx = key
        if y_idx >= len(self._heatmap_rows) or x_idx >= len(self._heatmap_cols):
            self._heatmap_hover_last_key = None
            self._set_hover_info(tr("hover_default"))
            self.heatmap_tooltip.hide()
            return

        ticker = str(self._heatmap_rows[y_idx])
        month = str(self._heatmap_cols[x_idx])
        metric_value = float(self._heatmap_values[y_idx, x_idx]) if self._heatmap_values.size else 0.0

        sub = pd.DataFrame()
        if not self._heatmap_trades.empty:
            sub = self._heatmap_trades.loc[
                (self._heatmap_trades["ticker"] == ticker) & (self._heatmap_trades["month"] == month)
            ].copy()

        info_text = self._build_hover_text(ticker=ticker, month=month, metric_value=metric_value, trades_sub=sub)
        self._set_hover_info(info_text)
        if bool(self.show_tooltip_var.get()) and pointer is not None:
            self.heatmap_tooltip.show(info_text, x_root=pointer[0], y_root=pointer[1])
        else:
            self.heatmap_tooltip.hide()
        self._heatmap_hover_last_key = key

    def _schedule_heatmap_hover(
        self,
        key: tuple[int, int] | None,
        pointer: tuple[int, int] | None = None,
    ) -> None:
        if (
            key == self._heatmap_hover_pending_key
            and pointer == self._heatmap_hover_pending_pointer
            and key == self._heatmap_hover_last_key
        ):
            return
        self._heatmap_hover_pending_key = key
        self._heatmap_hover_pending_pointer = pointer
        if self._heatmap_hover_job is not None:
            self.parent.after_cancel(self._heatmap_hover_job)
        self._heatmap_hover_job = self.parent.after(40, self._apply_heatmap_hover)

    def _on_heatmap_motion(self, event: Any) -> None:
        key = self._heatmap_index_from_event(event)
        pointer = self._extract_root_pointer(event)
        self._schedule_heatmap_hover(key=key, pointer=pointer)

    def _on_heatmap_leave(self, _event: Any) -> None:
        self._schedule_heatmap_hover(None, None)

    def _apply_trade_filters_from_heatmap(self, ticker: str, month: str) -> None:
        month_values = set(map(str, self.trade_month_combo.cget("values")))
        ticker_values = set(map(str, self.trade_ticker_combo.cget("values")))

        target_month = month if month in month_values else tr("all")
        target_ticker = ticker if ticker in ticker_values else tr("all")

        self._suspend_trade_filter_trace = True
        try:
            self.trade_filter_month_var.set(target_month)
            self.trade_filter_ticker_var.set(target_ticker)
        finally:
            self._suspend_trade_filter_trace = False
        self._render_tables()
        try:
            self.tables_notebook.select(self.trades_tab)
        except Exception:
            pass

    def _on_heatmap_click(self, event: Any) -> None:
        key = self._heatmap_index_from_event(event)
        if key is None:
            return
        x_idx, y_idx = key
        ticker = str(self._heatmap_rows[y_idx])
        month = str(self._heatmap_cols[x_idx])
        self._apply_trade_filters_from_heatmap(ticker=ticker, month=month)

    def _browse_strategy(self) -> None:
        path = filedialog.askopenfilename(
            title=tr("strategy_config"),
            filetypes=[("YAML/JSON", "*.yaml *.yml *.json"), ("All files", "*.*")],
        )
        if path:
            self.strategy_path_var.set(path)

    def _set_screen_rule_text(self, value: str) -> None:
        self._is_syncing_screen_rule = True
        try:
            self.screen_rule_var.set(value)
        finally:
            self._is_syncing_screen_rule = False

    def _on_screen_rule_changed(self) -> None:
        if self._is_syncing_screen_rule:
            return
        if self._current_indicators:
            self.builder_state_var.set(tr("builder_manual"))

    def _strategy_source_internal(self) -> str:
        value = str(self.strategy_source_var.get() or tr("strategy_source_file")).strip()
        return "builder" if value == tr("strategy_source_builder") else "file"

    def _is_strategy_builder_mode(self) -> bool:
        return self._strategy_source_internal() == "builder"

    def _apply_builder_result(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        rule_expr = str(payload.get("rule_expr", "") or "").strip()
        indicators = payload.get("indicators", [])
        summary = str(payload.get("summary", tr("builder_summary_default")) or tr("builder_summary_default"))
        self._current_indicators = indicators if isinstance(indicators, list) else []
        self._buy_condition_set = self._build_condition_set_from_indicators()
        self.buy_condition_editor.set_condition_set(self._buy_condition_set)
        self._set_screen_rule_text(rule_expr)
        self.builder_summary_var.set(summary)
        self.builder_state_var.set(tr("builder_applied"))

    def _normalized_strategy_builder_payload(self) -> dict[str, Any]:
        raw = self._strategy_builder_payload if isinstance(self._strategy_builder_payload, dict) else {}
        indicators_raw = raw.get("indicators", [])
        indicators_input = indicators_raw if isinstance(indicators_raw, list) else []
        rule_expr, indicators, summary, _errors = compile_indicator_payload(indicators_input, strict=False)
        fallback_rule = str(raw.get("rule_expr", "") or "").strip()
        if not rule_expr:
            rule_expr = fallback_rule

        sell_mode = str(raw.get("sell_mode", "B") or "B").strip().upper()
        if sell_mode not in {"A", "B", "C"}:
            sell_mode = "B"
        rank_drop_multiplier = float(self._parse_float(str(raw.get("rank_drop_multiplier", "2.0")), 2.0) or 2.0)
        if rank_drop_multiplier <= 0:
            rank_drop_multiplier = 2.0
        na_policy = str(raw.get("na_policy", "fail") or "fail").strip().lower()
        if na_policy not in {"fail", "pass"}:
            na_policy = "fail"

        return {
            "rule_expr": rule_expr,
            "indicators": indicators,
            "summary": summary,
            "sell_mode": sell_mode,
            "rank_drop_multiplier": rank_drop_multiplier,
            "na_policy": na_policy,
        }

    def _strategy_summary_text(self, payload: dict[str, Any]) -> str:
        summary = str(payload.get("summary", tr("builder_summary_default")) or tr("builder_summary_default"))
        sell_mode = str(payload.get("sell_mode", "B"))
        rank_mult = float(payload.get("rank_drop_multiplier", 2.0))
        na_policy = str(payload.get("na_policy", "fail"))
        return (
            f"{tr('strategy_builder_name')} {summary} | {tr('sell_mode')}: {sell_mode} "
            f"| {tr('rank_drop_mult')}: {rank_mult:.2f} | {tr('na_policy')}: {na_policy}"
        )

    def _build_strategy_builder_config(self) -> StrategyConfig:
        payload = self._normalized_strategy_builder_payload()
        rule_expr = str(payload.get("rule_expr", "") or "").strip()
        indicators = payload.get("indicators", [])
        funding_cfg = self._build_funding_config()
        self._sync_condition_sets_from_editors()
        buy_condition_set = self._buy_condition_set
        if not buy_condition_set.conditions and indicators:
            buy_condition_set = self._build_condition_set_from_indicators()
        self._buy_condition_set = buy_condition_set
        self.buy_condition_editor.set_condition_set(self._buy_condition_set)
        sell_mode_ui = self._sell_mode_internal()
        rank_drop_ui = float(self._parse_float(self.rank_drop_mult_ui_var.get(), 2.0) or 2.0)

        cfg = StrategyConfig(
            name=tr("strategy_builder_name"),
            mode="strategy",
            market=self._effective_market(),
            start=str(self.start_var.get() or "2000-01-01").strip(),
            end=(str(self.end_var.get() or "").strip() or None),
            frequency=str(self.freq_var.get() or "Q").strip().upper(),
            holdings=max(1, self._parse_int(self.holdings_var.get(), 3)),
            initial_cash=float(funding_cfg.initial_cash),
            sizing="equal",
            buy_rules=rule_expr,
            sell_mode=sell_mode_ui or str(payload.get("sell_mode", "B")),
            rank_drop_multiplier=rank_drop_ui if rank_drop_ui > 0 else float(payload.get("rank_drop_multiplier", 2.0)),
            rule_na_policy=str(payload.get("na_policy", "fail")),
            benchmark=str(self.benchmark_var.get() or "SPY").strip() or "SPY",
            execution_timing=str(self.exec_timing_var.get() or "next_open").strip(),
            indicators=indicators if isinstance(indicators, list) else [],
            costs=CostModel(
                commission_bps=float(self._parse_float(self.commission_bps_var.get(), 0.0) or 0.0),
                slippage_bps=float(self._parse_float(self.slippage_bps_var.get(), 0.0) or 0.0),
            ),
            share_mode=self._share_mode_internal(),
            cash_buffer_pct=float(self._parse_float(self.cash_buffer_pct_var.get(), 0.0) or 0.0) / 100.0,
            buy_condition_set=buy_condition_set,
            sell_condition_set=self._sell_condition_set,
            position_sizing=self._build_position_sizing(),
            risk_limits=self._build_risk_limits(),
            execution=self._build_execution_config(),
            universe_filter=self._build_universe_filter_config(),
            funding=funding_cfg,
        )
        self._apply_universe_selection_to_strategy_config(cfg)
        return cfg

    @staticmethod
    def _strategy_config_to_dict(cfg: StrategyConfig) -> dict[str, Any]:
        return {
            "name": cfg.name,
            "mode": cfg.mode,
            "market": cfg.market,
            "start": cfg.start,
            "end": cfg.end,
            "frequency": cfg.frequency,
            "holdings": int(cfg.holdings),
            "initial_cash": float(cfg.initial_cash),
            "sizing": cfg.sizing,
            "buy_rules": cfg.buy_rules,
            "sell_rules": cfg.sell_rules,
            "sell_mode": cfg.sell_mode,
            "rank_drop_multiplier": float(cfg.rank_drop_multiplier),
            "rule_na_policy": cfg.rule_na_policy,
            "execution_timing": cfg.execution_timing,
            "benchmark": cfg.benchmark,
            "asof_mode": cfg.asof_mode,
            "use_fundamentals_pit": bool(cfg.use_fundamentals_pit),
            "strict_pit": getattr(cfg, "strict_pit", False),
            "asof_lag_trading_days": int(cfg.asof_lag_trading_days),
            "use_next_trading_day_availability": bool(cfg.use_next_trading_day_availability),
            "fundamentals_availability_fallback": bool(cfg.fundamentals_availability_fallback),
            "fundamentals_missing_policy": str(cfg.fundamentals_missing_policy),
            "fundamentals_fallback_q_days": int(cfg.fundamentals_fallback_q_days),
            "fundamentals_fallback_k_days": int(cfg.fundamentals_fallback_k_days),
            "missing_policy": cfg.missing_policy,
            "share_mode": cfg.share_mode,
            "cash_buffer_pct": float(cfg.cash_buffer_pct),
            "min_trade_notional": float(cfg.min_trade_notional),
            "out_dir": cfg.out_dir,
            "buy_condition_set": dump_condition_set_dict(cfg.buy_condition_set),
            "sell_condition_set": dump_condition_set_dict(cfg.sell_condition_set),
            "position_sizing": {
                "position_weight_pct": cfg.position_sizing.position_weight_pct,
                "max_holdings": cfg.position_sizing.max_holdings,
                "max_new_buys_per_day": cfg.position_sizing.max_new_buys_per_day,
                "max_buy_amount_per_position": cfg.position_sizing.max_buy_amount_per_position,
            },
            "risk_limits": {
                "min_cash_reserve_pct": float(cfg.risk_limits.min_cash_reserve_pct),
            },
            "execution": {
                "timing": cfg.execution.timing,
                "price_basis": cfg.execution.price_basis,
                "price_offset_pct": float(cfg.execution.price_offset_pct),
                "commission_pct": cfg.execution.commission_pct,
                "slippage_pct": cfg.execution.slippage_pct,
            },
            "universe": {
                "source": cfg.universe.source,
                "symbols": cfg.universe.symbols,
                "symbols_file": cfg.universe.symbols_file,
                "screen": cfg.universe.screen,
                "market": cfg.universe.market,
                "min_market_cap": cfg.universe.min_market_cap,
                "min_dollar_volume_20d": cfg.universe.min_dollar_volume_20d,
                "exclude_adr": bool(cfg.universe.exclude_adr),
                "exclude_preferred": bool(cfg.universe.exclude_preferred),
                "exclude_distress": bool(cfg.universe.exclude_distress),
                "include_sectors": cfg.universe.include_sectors,
                "exclude_sectors": cfg.universe.exclude_sectors,
                "include_subsectors": cfg.universe.include_subsectors,
                "exclude_subsectors": cfg.universe.exclude_subsectors,
                "exchanges": cfg.universe.exchanges,
                "size_buckets": cfg.universe.size_buckets,
                "exclude_tickers": cfg.universe.exclude_tickers,
                "watchlist_name": cfg.universe.watchlist_name,
            },
            "universe_filter": {
                "exchanges": cfg.universe_filter.exchanges,
                "size_buckets": cfg.universe_filter.size_buckets,
                "include_sectors": cfg.universe_filter.include_sectors,
                "exclude_sectors": cfg.universe_filter.exclude_sectors,
                "include_subsectors": cfg.universe_filter.include_subsectors,
                "exclude_subsectors": cfg.universe_filter.exclude_subsectors,
                "exclude_tickers": cfg.universe_filter.exclude_tickers,
                "watchlist_name": cfg.universe_filter.watchlist_name,
                "universe_presets": cfg.universe_filter.universe_presets,
                "universe_bucket_selections": cfg.universe_filter.universe_bucket_selections,
                "stock_type_included": cfg.universe_filter.stock_type_included,
                "ptp_included": cfg.universe_filter.ptp_included,
            },
            "costs": {
                "commission_bps": float(cfg.costs.commission_bps),
                "slippage_bps": float(cfg.costs.slippage_bps),
            },
            "funding": {
                "mode": cfg.funding.mode,
                "initial_cash": float(cfg.funding.initial_cash),
                "contribution_freq": cfg.funding.contribution_freq,
                "fixed_contribution": cfg.funding.fixed_contribution,
                "va_target_step": cfg.funding.va_target_step,
                "va_target_base": float(cfg.funding.va_target_base),
                "va_min_contribution": float(cfg.funding.va_min_contribution),
                "va_min_policy": str(cfg.funding.va_min_policy),
                "va_max_contribution": cfg.funding.va_max_contribution,
                "va_allow_withdrawal": bool(cfg.funding.va_allow_withdrawal),
            },
            "ranking": {
                "normalization": cfg.ranking.normalization,
                "factors": [
                    {
                        "name": f.name,
                        "direction": f.direction,
                        "weight": float(f.weight),
                        "winsorize_low": f.winsorize_low,
                        "winsorize_high": f.winsorize_high,
                    }
                    for f in cfg.ranking.factors
                ],
            },
            "indicators": cfg.indicators,
        }

    @staticmethod
    def _funding_config_to_dict(cfg: FundingConfig) -> dict[str, Any]:
        return {
            "mode": cfg.mode,
            "initial_cash": float(cfg.initial_cash),
            "contribution_freq": cfg.contribution_freq,
            "fixed_contribution": cfg.fixed_contribution,
            "va_target_step": cfg.va_target_step,
            "va_target_base": float(cfg.va_target_base),
            "va_min_contribution": float(cfg.va_min_contribution),
            "va_min_policy": str(cfg.va_min_policy),
            "va_max_contribution": cfg.va_max_contribution,
            "va_allow_withdrawal": bool(cfg.va_allow_withdrawal),
        }

    def _set_strategy_preview_text(self, text: str) -> None:
        self.strategy_preview_text.configure(state="normal")
        self.strategy_preview_text.delete("1.0", tk.END)
        if text:
            self.strategy_preview_text.insert("1.0", text)
        self.strategy_preview_text.configure(state="disabled")

    def _refresh_strategy_builder_preview(self) -> None:
        payload = self._normalized_strategy_builder_payload()
        self._strategy_builder_payload = payload
        self.strategy_builder_summary_var.set(self._strategy_summary_text(payload))
        cfg = self._build_strategy_builder_config()
        preview_text = json.dumps(self._strategy_config_to_dict(cfg), ensure_ascii=False, indent=2)
        self._set_strategy_preview_text(preview_text)

    def _on_strategy_source_changed(self) -> None:
        builder_mode = self._is_strategy_builder_mode()
        path_state = "disabled" if builder_mode else "normal"
        self.strategy_path_entry.configure(state=path_state)
        self.strategy_browse_btn.configure(state=path_state)

        if builder_mode:
            self.strategy_preview_frame.grid()
            self._refresh_strategy_builder_preview()
        else:
            self.strategy_preview_frame.grid_remove()

        if self._running:
            self.strategy_builder_btn.configure(state="disabled")
            self.strategy_save_btn.configure(state="disabled")
            return

        self.strategy_builder_btn.configure(state="normal")
        save_state = "normal" if builder_mode else "disabled"
        self.strategy_save_btn.configure(state=save_state)

    def _apply_strategy_builder_result(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self._strategy_builder_payload = {
            "rule_expr": str(payload.get("rule_expr", "") or "").strip(),
            "indicators": payload.get("indicators", []) if isinstance(payload.get("indicators"), list) else [],
            "summary": str(payload.get("summary", tr("builder_summary_default")) or tr("builder_summary_default")),
            "sell_mode": str(payload.get("sell_mode", "B") or "B").strip().upper(),
            "rank_drop_multiplier": float(self._parse_float(str(payload.get("rank_drop_multiplier", "2.0")), 2.0) or 2.0),
            "na_policy": str(payload.get("na_policy", "fail") or "fail").strip().lower(),
        }
        self.sell_mode_ui_var.set(
            tr("sell_mode_a")
            if self._strategy_builder_payload["sell_mode"] == "A"
            else (tr("sell_mode_c") if self._strategy_builder_payload["sell_mode"] == "C" else tr("sell_mode_b"))
        )
        self.rank_drop_mult_ui_var.set(str(self._strategy_builder_payload["rank_drop_multiplier"]))
        if self._is_strategy_builder_mode():
            self._refresh_strategy_builder_preview()

    def _open_strategy_builder(self) -> None:
        payload = self._normalized_strategy_builder_payload()
        dlg = RuleBuilderDialog(
            parent=self.parent,
            initial_indicators=payload.get("indicators", []),
            metrics=list_metric_definitions(),
            mode="strategy",
            initial_sell_mode=str(payload.get("sell_mode", "B")),
            initial_rank_drop_multiplier=float(payload.get("rank_drop_multiplier", 2.0)),
            initial_na_policy=str(payload.get("na_policy", "fail")),
            on_apply=self._apply_strategy_builder_result,
        )
        result = dlg.show()
        if result is not None:
            self._apply_strategy_builder_result(result)

    def _save_strategy_builder_to_file(self) -> None:
        try:
            cfg = self._build_strategy_builder_config()
            out_dir = Path("strategies")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"generated_{ts}.json"
            out_path.write_text(json.dumps(self._strategy_config_to_dict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
            self.strategy_path_var.set(str(out_path))
            messagebox.showinfo(tr("strategy_builder_saved"), str(out_path), parent=self.parent.winfo_toplevel())
        except Exception as exc:
            messagebox.showerror(tr("save_preset_error"), str(exc), parent=self.parent.winfo_toplevel())

    def _open_rule_builder(self) -> None:
        dlg = RuleBuilderDialog(
            parent=self.parent,
            initial_indicators=self._current_indicators,
            metrics=list_metric_definitions(),
            mode="screen",
            on_apply=self._apply_builder_result,
        )
        result = dlg.show()
        if result is not None:
            self._apply_builder_result(result)

    @staticmethod
    def _csv_values(text: str, *, upper: bool = False, lower: bool = False) -> list[str]:
        return parse_csv_values(text, upper=upper, lower=lower)

    def _build_condition_set_from_indicators(self) -> ConditionSet:
        clauses: list[ConditionClause] = []
        group1: list[str] = []
        group2: list[str] = []
        label_ord = ord("A")
        for item in self._current_indicators:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("enabled", True)) or not bool(item.get("as_filter", False)):
                continue
            metric_id = str(item.get("id", "")).strip()
            op = str(item.get("op", "<=")).strip()
            threshold = _parse_optional_float(item.get("threshold"))[0]
            if not metric_id or op not in {"<", "<=", ">", ">=", "==", "!="} or threshold is None:
                continue
            label = chr(label_ord)
            label_ord += 1
            if label_ord > ord("Z"):
                break
            clauses.append(ConditionClause(label=label, left=f"{{{metric_id}}}", op=op, right=float(threshold)))
            group = int(item.get("filter_group", 1)) if str(item.get("filter_group", "1")).isdigit() else 1
            if group == 2:
                group2.append(label)
            else:
                group1.append(label)

        expr_g1 = " and ".join(group1)
        expr_g2 = " and ".join(group2)
        if expr_g1 and expr_g2:
            logic = f"({expr_g1}) or ({expr_g2})"
        else:
            logic = expr_g1 or expr_g2
        return ConditionSet(conditions=clauses, logical_expression=logic, na_policy="fail")

    def _validate_conditions(self) -> bool:
        self._sync_condition_sets_from_editors()
        buy_set = self._buy_condition_set
        if not buy_set.conditions and self._current_indicators:
            buy_set = self._build_condition_set_from_indicators()
            self._buy_condition_set = buy_set
            self.buy_condition_editor.set_condition_set(buy_set)
        sell_set = self._sell_condition_set
        cols = sorted({m.id for m in list_metric_definitions()})
        buy_res = validate_condition_set(buy_set, available_columns=cols)
        sell_res = validate_condition_set(sell_set, available_columns=cols)
        buy_msg = tr("conditions_valid") if buy_res.ok else tr("conditions_invalid")
        sell_msg = tr("conditions_valid") if sell_res.ok else tr("conditions_invalid")
        self.buy_condition_editor.show_message(f"[BUY] {buy_msg}")
        self.sell_condition_editor.show_message(f"[SELL] {sell_msg}")

        all_errors = list(buy_res.errors) + list(sell_res.errors)
        all_warnings = list(buy_res.warnings) + list(sell_res.warnings)
        if not all_errors:
            msg = tr("conditions_valid")
            if all_warnings:
                msg += "\n\n" + "\n".join(all_warnings[:10])
            messagebox.showinfo(tr("validate_conditions"), msg, parent=self.parent.winfo_toplevel())
            return True

        msg = tr("conditions_invalid") + "\n\n" + "\n".join(all_errors[:20])
        if all_warnings:
            msg += "\n\nWARN:\n" + "\n".join(all_warnings[:10])
        messagebox.showerror(tr("validate_conditions"), msg, parent=self.parent.winfo_toplevel())
        return False

    def _save_conditions_to_file(self) -> None:
        self._sync_condition_sets_from_editors()
        payload = {
            "schema_version": "1.0",
            "buy_condition_set": dump_condition_set_dict(self._buy_condition_set),
            "sell_condition_set": dump_condition_set_dict(self._sell_condition_set),
        }
        path = filedialog.asksaveasfilename(
            parent=self.parent.winfo_toplevel(),
            title=tr("condition_file_title"),
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror(tr("condition_file_error"), str(exc), parent=self.parent.winfo_toplevel())

    def _load_conditions_from_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.parent.winfo_toplevel(),
            title=tr("condition_file_title"),
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self._buy_condition_set = load_condition_set_dict(payload.get("buy_condition_set"))
            self._sell_condition_set = load_condition_set_dict(payload.get("sell_condition_set"))
            self._sync_condition_editors_from_state()
            self.builder_state_var.set(tr("builder_applied"))
            self._validate_conditions()
        except Exception as exc:
            messagebox.showerror(tr("condition_file_error"), str(exc), parent=self.parent.winfo_toplevel())

    def _build_screen_config(self) -> StrategyConfig:
        buy_rules = str(self.screen_rule_var.get() or "").strip()
        use_indicators: list[dict[str, Any]] = []
        funding_cfg = self._build_funding_config()
        self._sync_condition_sets_from_editors()
        buy_condition_set = self._buy_condition_set
        if not buy_condition_set.conditions:
            buy_condition_set = self._build_condition_set_from_indicators()
        self._buy_condition_set = buy_condition_set
        self.buy_condition_editor.set_condition_set(self._buy_condition_set)

        if self._current_indicators:
            rule_expr, compiled, summary, _ = compile_indicator_payload(self._current_indicators, strict=False)
            use_indicators = compiled
            if rule_expr:
                buy_rules = rule_expr
                self._set_screen_rule_text(rule_expr)
            self.builder_summary_var.set(summary)

        cfg = StrategyConfig(
            name="Screen",
            mode="screen",
            market=self._effective_market(),
            start=str(self.start_var.get() or "2000-01-01").strip(),
            end=(str(self.end_var.get() or "").strip() or None),
            frequency=str(self.freq_var.get() or "Q").strip().upper(),
            holdings=max(1, self._parse_int(self.holdings_var.get(), 3)),
            initial_cash=float(funding_cfg.initial_cash),
            sizing="equal",
            buy_rules=buy_rules,
            sell_mode="A",
            benchmark=str(self.benchmark_var.get() or "SPY").strip() or "SPY",
            execution_timing=str(self.exec_timing_var.get() or "next_open").strip(),
            indicators=use_indicators,
            costs=CostModel(
                commission_bps=float(self._parse_float(self.commission_bps_var.get(), 0.0) or 0.0),
                slippage_bps=float(self._parse_float(self.slippage_bps_var.get(), 0.0) or 0.0),
            ),
            share_mode=self._share_mode_internal(),
            cash_buffer_pct=float(self._parse_float(self.cash_buffer_pct_var.get(), 0.0) or 0.0) / 100.0,
            buy_condition_set=buy_condition_set,
            sell_condition_set=self._sell_condition_set,
            position_sizing=self._build_position_sizing(),
            risk_limits=self._build_risk_limits(),
            execution=self._build_execution_config(),
            universe_filter=self._build_universe_filter_config(),
            funding=funding_cfg,
        )
        self._apply_universe_selection_to_strategy_config(cfg)
        return cfg

    def _apply_strategy_overrides(self, cfg: StrategyConfig) -> StrategyConfig:
        if not bool(self.override_strategy_var.get()):
            return cfg

        start_txt = str(self.start_var.get() or "").strip()
        end_txt = str(self.end_var.get() or "").strip()
        freq_txt = str(self.freq_var.get() or "").strip().upper()
        holdings_txt = str(self.holdings_var.get() or "").strip()

        cfg.market = self._effective_market()
        if start_txt:
            cfg.start = start_txt
        cfg.end = end_txt or None
        if freq_txt:
            cfg.frequency = freq_txt
        if holdings_txt:
            cfg.holdings = max(1, self._parse_int(holdings_txt, cfg.holdings))

        cfg.benchmark = str(self.benchmark_var.get() or cfg.benchmark).strip() or cfg.benchmark
        cfg.execution_timing = str(self.exec_timing_var.get() or cfg.execution_timing).strip() or cfg.execution_timing

        cfg.initial_cash = float(self._parse_float(self.initial_cash_var.get(), cfg.initial_cash) or cfg.initial_cash)
        cfg.costs.commission_bps = float(self._parse_float(self.commission_bps_var.get(), cfg.costs.commission_bps) or cfg.costs.commission_bps)
        cfg.costs.slippage_bps = float(self._parse_float(self.slippage_bps_var.get(), cfg.costs.slippage_bps) or cfg.costs.slippage_bps)
        cfg.share_mode = self._share_mode_internal()
        cfg.cash_buffer_pct = float(self._parse_float(self.cash_buffer_pct_var.get(), cfg.cash_buffer_pct * 100.0) or (cfg.cash_buffer_pct * 100.0)) / 100.0
        cfg.position_sizing = self._build_position_sizing()
        cfg.risk_limits = self._build_risk_limits()
        cfg.execution = self._build_execution_config()
        cfg.universe_filter = self._build_universe_filter_config()
        self._apply_universe_selection_to_strategy_config(cfg)
        cfg.funding = self._build_funding_config()
        cfg.initial_cash = float(cfg.funding.initial_cash)
        return cfg

    # ------------------------------------------------------------------
    # Progress / streaming helpers
    # ------------------------------------------------------------------
    def _reset_progress(self) -> None:
        self._run_progress_pct.set(0.0)
        self._run_progress_status.set("데이터 로딩 중...")
        self._run_progress_bar.configure(mode="indeterminate")
        self._run_progress_bar.start(12)
        self._loading_phase = True

    def _on_backtest_progress(self, pct: float, date_str: str, trade_list: list) -> None:
        """Called from background thread via root.after(); updates progress bar and trade log."""
        if getattr(self, "_loading_phase", False):
            self._loading_phase = False
            self._run_progress_bar.stop()
            self._run_progress_bar.configure(mode="determinate")
        self._run_progress_pct.set(pct * 100.0)
        self._run_progress_status.set(f"{tr('progress_running_pct')} {pct * 100.0:.0f}%")
        if trade_list:
            for t in trade_list:
                side = t.get("side", "")
                side_label = tr("stream_buy") if side == "buy" else tr("stream_sell")
                tag = "buy" if side == "buy" else "sell"
                self.trades_tree.insert(
                    "",
                    "end",
                    values=(
                        date_str,
                        t.get("ticker", ""),
                        side_label,
                        f"{float(t.get('shares', 0.0)):.4f}",
                        f"{float(t.get('price', 0.0)):.4f}",
                        "", "", "", "",
                    ),
                    tags=(tag,),
                )
            children = self.trades_tree.get_children()
            if len(children) > 500:
                for iid in children[:-500]:
                    self.trades_tree.delete(iid)
            last = self.trades_tree.get_children()
            if last:
                self.trades_tree.see(last[-1])

    def _set_running(self, running: bool, text: str) -> None:
        self._running = running
        state = "disabled" if running else "normal"
        self.run_screen_btn.configure(state=state)
        self.run_strategy_btn.configure(state=state)
        self.compare_btn.configure(state=state)
        self.preflight_btn.configure(state=state)
        self.save_strategy_btn.configure(state=state)
        self.load_strategy_btn.configure(state=state)
        self.audit_prompt_btn.configure(state=state)
        self.ai_export_btn.configure(state=state)
        self.rule_builder_btn.configure(state=state)
        self.validate_conditions_btn.configure(state=state)
        self.save_conditions_btn.configure(state=state)
        self.load_conditions_btn.configure(state=state)
        self.strategy_builder_btn.configure(state=state)
        self.strategy_save_btn.configure(state=state)
        self.advanced_toggle_btn.configure(state=state)
        self.buy_toggle_btn.configure(state=state)
        self.sell_toggle_btn.configure(state=state)
        self.universe_toggle_btn.configure(state=state)
        self.funding_mode_combo.configure(state="disabled" if running else "readonly")
        if running:
            self.funding_contrib_freq_combo.configure(state="disabled")
        else:
            mode = self._funding_mode_internal()
            self.funding_contrib_freq_combo.configure(state="disabled" if mode == "lump_sum" else "readonly")
        self.funding_view_combo.configure(state="disabled" if running else "readonly")
        self.exec_basis_combo.configure(state="disabled" if running else "readonly")
        self.progress_var.set(text)
        if not running:
            self._on_strategy_source_changed()
            self._on_funding_mode_changed()
        self.parent.update_idletasks()

    def _preflight_config(self, cfg: StrategyConfig) -> tuple[bool, list[str], list[str]]:
        cols = sorted({m.id for m in list_metric_definitions()})
        presets = list(cfg.universe_filter.universe_presets or [])
        if str(cfg.universe.source or "").strip().lower() == "sp500" and "sp500" not in presets:
            presets.append("sp500")
        mapped = map_universe_presets_to_filters(
            presets=presets,
            bucket_selections=cfg.universe_filter.universe_bucket_selections,
            manual_exchanges=list(cfg.universe_filter.exchanges or []),
            manual_size_buckets=list(cfg.universe_filter.size_buckets or []),
        )
        return preflight_validate_strategy_config(cfg, available_columns=cols, universe_selection=mapped)

    def _collect_strategy_config_from_ui(self) -> StrategyConfig:
        if self._is_strategy_builder_mode():
            cfg = self._build_strategy_builder_config()
        else:
            cfg = self._build_screen_config()
            cfg.mode = "strategy"
            cfg.name = "strategy_ui"
            cfg.sell_mode = self._sell_mode_internal()
            cfg.rank_drop_multiplier = float(self._parse_float(self.rank_drop_mult_ui_var.get(), cfg.rank_drop_multiplier) or cfg.rank_drop_multiplier)
        cfg.universe_filter = self._build_universe_filter_config()
        cfg.position_sizing = self._build_position_sizing()
        cfg.risk_limits = self._build_risk_limits()
        cfg.execution = self._build_execution_config()
        self._apply_universe_selection_to_strategy_config(cfg)
        self._sync_condition_sets_from_editors()
        if self._buy_condition_set.conditions:
            cfg.buy_condition_set = self._buy_condition_set
        if self._sell_condition_set.conditions:
            cfg.sell_condition_set = self._sell_condition_set
        return cfg

    def _apply_strategy_config_to_ui(self, cfg: StrategyConfig) -> None:
        self.market_var.set(str(cfg.market or "us"))
        self.start_var.set(str(cfg.start or ""))
        self.end_var.set(str(cfg.end or ""))
        self.freq_var.set(str(cfg.frequency or "Q"))
        self.holdings_var.set(str(int(cfg.holdings)))
        self.benchmark_var.set(str(cfg.benchmark or "SPY"))
        self.exec_timing_var.set(str(cfg.execution_timing or "next_open"))
        self.screen_rule_var.set(str(cfg.buy_rules or ""))
        self.share_mode_var.set(tr("share_mode_integer") if str(cfg.share_mode) == "integer" else tr("share_mode_fractional"))
        self.cash_buffer_pct_var.set(str(float(cfg.cash_buffer_pct) * 100.0))
        self.initial_cash_var.set(str(float(cfg.funding.initial_cash if cfg.funding else cfg.initial_cash)))
        self.commission_bps_var.set(str(float(cfg.costs.commission_bps)))
        self.slippage_bps_var.set(str(float(cfg.costs.slippage_bps)))
        self.position_weight_pct_var.set("" if cfg.position_sizing.position_weight_pct is None else str(cfg.position_sizing.position_weight_pct))
        self.max_holdings_cfg_var.set("" if cfg.position_sizing.max_holdings is None else str(cfg.position_sizing.max_holdings))
        self.max_new_buys_var.set("" if cfg.position_sizing.max_new_buys_per_day is None else str(cfg.position_sizing.max_new_buys_per_day))
        self.max_buy_amount_var.set("" if cfg.position_sizing.max_buy_amount_per_position is None else str(cfg.position_sizing.max_buy_amount_per_position))
        self.min_cash_reserve_pct_var.set(str(float(cfg.risk_limits.min_cash_reserve_pct)))
        basis_map = {
            "auto": tr("price_basis_auto"),
            "open": tr("price_basis_open"),
            "close": tr("price_basis_close"),
            "prev_close": tr("price_basis_prev_close"),
        }
        self.exec_price_basis_var.set(basis_map.get(str(cfg.execution.price_basis or "auto"), tr("price_basis_auto")))
        self.price_offset_pct_var.set(str(float(cfg.execution.price_offset_pct)))
        self.universe_exchanges_var.set(", ".join(cfg.universe_filter.exchanges))
        self.universe_size_buckets_var.set(", ".join(cfg.universe_filter.size_buckets))
        self.universe_include_sectors_var.set(", ".join(cfg.universe_filter.include_sectors))
        self.universe_exclude_sectors_var.set(", ".join(cfg.universe_filter.exclude_sectors))
        self.universe_include_subsectors_var.set(", ".join(cfg.universe_filter.include_subsectors))
        self.universe_exclude_subsectors_var.set(", ".join(cfg.universe_filter.exclude_subsectors))
        self.universe_exclude_tickers_var.set(", ".join(cfg.universe_filter.exclude_tickers))
        self.universe_watchlist_name_var.set(str(cfg.universe_filter.watchlist_name or ""))
        self.stock_type_included_var.set(
            True if cfg.universe_filter.stock_type_included is None else bool(cfg.universe_filter.stock_type_included)
        )
        self.ptp_included_var.set(bool(cfg.universe_filter.ptp_included))
        for key in self.universe_preset_vars:
            self.universe_preset_vars[key].set(False)
        for exch in ["NASDAQ", "NYSE"]:
            for b in UNIVERSE_BUCKETS:
                self.universe_bucket_vars[exch][b].set(False)
            self.universe_parent_status_vars[exch].set("")

        # Handle direct ticker source
        src = str(cfg.universe.source or "").strip().lower()
        if src == "symbols" and cfg.universe.symbols:
            self.universe_input_mode_var.set("direct")
            self.universe_direct_tickers_var.set(", ".join(cfg.universe.symbols))
        else:
            self.universe_input_mode_var.set("preset")
            self.universe_direct_tickers_var.set("")

        presets = [str(x).strip().lower() for x in (cfg.universe_filter.universe_presets or []) if str(x).strip()]
        if src == "sp500" and "sp500" not in presets:
            presets.append("sp500")
        if src in {"sp500_pit", "nasdaq_stock_only_financial"}:
            presets.append(src)
        for key in ["sp500", "sp500_pit", "nasdaq_stock_only_financial", "nasdaq_all", "nyse_all"]:
            if key in self.universe_preset_vars:
                self.universe_preset_vars[key].set(key in presets)

        buckets = normalize_universe_bucket_selections(cfg.universe_filter.universe_bucket_selections)
        for exch in ["NASDAQ", "NYSE"]:
            chosen = buckets.get(exch, [])
            for b in UNIVERSE_BUCKETS:
                self.universe_bucket_vars[exch][b].set(b in set(chosen))
            self._refresh_universe_parent_status(exch)

        self.universe_financial_only_var.set(bool(getattr(cfg.universe_filter, "financial_data_only", False)))
        self._on_universe_input_mode_changed()

        self._buy_condition_set = cfg.buy_condition_set or ConditionSet()
        self._sell_condition_set = cfg.sell_condition_set or ConditionSet()
        self._sync_condition_editors_from_state()
        self._current_indicators = list(cfg.indicators or [])
        if cfg.indicators:
            self.builder_summary_var.set(compile_indicator_payload(cfg.indicators, strict=False)[2])
        self.sell_mode_ui_var.set(
            tr("sell_mode_a")
            if str(cfg.sell_mode) == "A"
            else (tr("sell_mode_c") if str(cfg.sell_mode) == "C" else tr("sell_mode_b"))
        )
        self.rank_drop_mult_ui_var.set(str(float(cfg.rank_drop_multiplier)))

        if cfg.funding:
            mode_map = {
                "lump_sum": tr("funding_mode_lump"),
                "dca": tr("funding_mode_dca"),
                "va": tr("funding_mode_va"),
            }
            self.funding_mode_var.set(mode_map.get(str(cfg.funding.mode), tr("funding_mode_lump")))
            freq_map = {"M": tr("funding_freq_monthly"), "Q": tr("funding_freq_quarterly"), "Y": tr("funding_freq_yearly")}
            self.funding_contrib_freq_var.set(freq_map.get(str(cfg.funding.contribution_freq), tr("funding_freq_monthly")))
            self.funding_dca_var.set("" if cfg.funding.fixed_contribution is None else str(cfg.funding.fixed_contribution))
            self.funding_va_step_var.set("" if cfg.funding.va_target_step is None else str(cfg.funding.va_target_step))
            self.funding_va_min_var.set(str(float(cfg.funding.va_min_contribution)))
            self.funding_va_max_var.set("" if cfg.funding.va_max_contribution is None else str(cfg.funding.va_max_contribution))
            self.funding_va_withdraw_var.set(bool(cfg.funding.va_allow_withdrawal))
            self.funding_va_min_policy_var.set(
                tr("va_min_policy_positive_raw_only")
                if str(cfg.funding.va_min_policy) == "positive_raw_only"
                else tr("va_min_policy_every_rebalance")
            )
        self._update_universe_filter_summary()
        self._update_funding_preview()
        self._on_funding_mode_changed()
        self._on_strategy_source_changed()
        self._refresh_strategy_builder_preview()

    def _run_preflight_check(self) -> None:
        try:
            cfg = self._collect_strategy_config_from_ui()
            ok, errs, warns = self._preflight_config(cfg)
            if not ok:
                raise ValueError("\n".join(errs[:30]))
            msg = tr("preflight_ok")
            if warns:
                msg += "\n\nWARN:\n" + "\n".join(warns[:10])
            messagebox.showinfo(tr("preflight_check"), msg, parent=self.parent.winfo_toplevel())
        except Exception as exc:
            messagebox.showerror(tr("preflight_failed"), str(exc), parent=self.parent.winfo_toplevel())

    def _save_strategy_state(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.parent.winfo_toplevel(),
            title=tr("save_strategy"),
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            cfg = self._collect_strategy_config_from_ui()
            payload = {
                "schema_version": "2.0",
                "saved_at": pd.Timestamp.now().isoformat(),
                "app_version": APP_VERSION,
                "ui_layout_version": "genport_sections_v1",
                "strategy_source": self._strategy_source_internal(),
                "strategy_builder_payload": deepcopy(self._strategy_builder_payload),
                "strategy_config": self._strategy_config_to_dict(cfg),
                "ui_context": self._build_ui_context_snapshot(),
            }
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            messagebox.showinfo(tr("strategy_saved"), str(path), parent=self.parent.winfo_toplevel())
        except Exception as exc:
            messagebox.showerror(tr("strategy_file_error"), str(exc), parent=self.parent.winfo_toplevel())

    def _load_strategy_state(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.parent.winfo_toplevel(),
            title=tr("load_strategy"),
            filetypes=[("JSON/YAML", "*.json *.yaml *.yml"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            p = Path(path)
            if p.suffix.lower() in {".yaml", ".yml"}:
                cfg = load_strategy_config(str(p))
                self._apply_strategy_config_to_ui(cfg)
            else:
                payload = json.loads(p.read_text(encoding="utf-8"))
                cfg_raw = payload.get("strategy_config", payload)
                cfg = strategy_config_from_dict(cfg_raw)
                self._apply_strategy_config_to_ui(cfg)
                src = str(payload.get("strategy_source", "")).strip().lower()
                if src == "builder":
                    self.strategy_source_var.set(tr("strategy_source_builder"))
                else:
                    self.strategy_source_var.set(tr("strategy_source_file"))
                if isinstance(payload.get("strategy_builder_payload"), dict):
                    self._strategy_builder_payload = payload["strategy_builder_payload"]
            self.strategy_path_var.set(str(path))
            messagebox.showinfo(tr("strategy_loaded"), str(path), parent=self.parent.winfo_toplevel())
        except Exception as exc:
            messagebox.showerror(tr("strategy_file_error"), str(exc), parent=self.parent.winfo_toplevel())

    def _copy_audit_prompt_template(self) -> None:
        text = (
            "이 전략 결과 번들을 검증해주세요.\n"
            "1) ai_review_summary.json을 읽고 모드/설정을 확인\n"
            "2) trades/funding_flows/equity_curve/metrics 정합성 재계산\n"
            "3) snapshot 기반 조건식 재검산 가능 여부 보고\n"
            "4) PASS/WARN/FAIL 및 근거 행/날짜/티커를 표기\n"
        )
        try:
            self.parent.clipboard_clear()
            self.parent.clipboard_append(text)
            messagebox.showinfo(tr("audit_prompt_copy"), tr("audit_prompt_copied"), parent=self.parent.winfo_toplevel())
        except Exception as exc:
            messagebox.showerror(tr("audit_prompt_copy"), str(exc), parent=self.parent.winfo_toplevel())

    def _run_screen(self) -> None:
        if self._running:
            return
        # Build config and run preflight on the main thread (accesses tkinter vars).
        try:
            cfg = self._build_screen_config()
            ok, errs, warns = self._preflight_config(cfg)
            if not ok:
                raise ValueError(tr("preflight_failed") + "\n" + "\n".join(errs[:20]))
            if warns:
                self.builder_state_var.set("WARN: " + " | ".join(warns[:2]))
            self._last_screen_cfg = self._strategy_config_to_dict(cfg)
            self._last_strategy_cfg = None
            funding_mode = self._funding_mode_internal()
            if funding_mode == "compare":
                variants = self._build_funding_variants()
                self._last_funding_variants = [self._funding_config_to_dict(v) for v in variants]
                compare_basis = self._funding_compare_basis_internal()
                custom_total = self._parse_funding_optional_float(
                    self.funding_custom_total_var.get(),
                    field_label=tr("funding_custom_total"),
                )
                self._publish_funding_parse_warnings()
                target_mode = self._funding_view_internal()
            else:
                cfg.funding = self._build_funding_config(mode=funding_mode)
                cfg.initial_cash = float(cfg.funding.initial_cash)
                self._last_screen_cfg = self._strategy_config_to_dict(cfg)
                self._last_funding_variants = None
                variants = compare_basis = custom_total = target_mode = None
        except Exception as exc:
            messagebox.showerror(tr("run_screen_error"), str(exc))
            return

        # Switch to results tab and prepare UI.
        self._set_running(True, tr("running_screen"))
        self._reset_progress()
        self.sub_nb.select(1)
        root = self.parent.winfo_toplevel()

        def _bg() -> None:
            try:
                if funding_mode == "compare":
                    res_compare, meta = run_funding_comparison(
                        cfg, variants, alignment_mode=compare_basis,
                        custom_total=custom_total, return_meta=True,
                    )
                    screen_result = res_compare.get(target_mode) or res_compare.get("lump_sum")
                    if screen_result is None and res_compare:
                        screen_result = next(iter(res_compare.values()))
                    if screen_result is not None:
                        screen_result.name = tr("source_screen")
                    def _done_compare() -> None:
                        try:
                            self.screen_result = screen_result
                            self.funding_compare_results = res_compare
                            self._funding_compare_results_by_mode = res_compare
                            self.funding_compare_meta = meta
                            self.compare_result = None
                            self._render_all()
                        except Exception as render_exc:
                            LOGGER.exception("render error after screen compare run: %s", render_exc)
                        finally:
                            if getattr(self, "_loading_phase", False):
                                self._loading_phase = False
                                self._run_progress_bar.stop()
                                self._run_progress_bar.configure(mode="determinate")
                            self._run_progress_pct.set(100.0)
                            self._run_progress_status.set(tr("done"))
                            self._set_running(False, tr("done"))
                    root.after(0, _done_compare)
                else:
                    def _loading_cb(i: int, total: int, symbol: str) -> None:
                        status_text = f"데이터 로딩 중... ({i}/{total})"
                        root.after(0, lambda t=status_text: self._run_progress_status.set(t))
                    def _prog(pct: float, date_str: str, trades: list) -> None:
                        root.after(0, lambda: self._on_backtest_progress(pct, date_str, trades))
                    screen_result = run_backtest(cfg, progress_callback=_prog, loading_callback=_loading_cb)
                    if screen_result is not None:
                        screen_result.name = tr("source_screen")
                    def _done_simple() -> None:
                        try:
                            self.screen_result = screen_result
                            self.funding_compare_results = None
                            self._funding_compare_results_by_mode = None
                            self.funding_compare_meta = {}
                            self.compare_result = None
                            self._render_all()
                        except Exception as render_exc:
                            LOGGER.exception("render error after screen run: %s", render_exc)
                        finally:
                            if getattr(self, "_loading_phase", False):
                                self._loading_phase = False
                                self._run_progress_bar.stop()
                                self._run_progress_bar.configure(mode="determinate")
                            self._run_progress_pct.set(100.0)
                            self._run_progress_status.set(tr("done"))
                            self._set_running(False, tr("done"))
                    root.after(0, _done_simple)
            except Exception as exc:
                err_msg = str(exc)
                LOGGER.exception("background screen run failed: %s", exc)
                def _on_err() -> None:
                    self._set_running(False, tr("idle"))
                    messagebox.showerror(tr("run_screen_error"), err_msg)
                root.after(0, _on_err)

        threading.Thread(target=_bg, daemon=True).start()

    def _run_strategy(self) -> None:
        if self._running:
            return

        # Build config and run preflight on the main thread (accesses tkinter vars).
        try:
            funding_mode = self._funding_mode_internal()
            is_builder = self._is_strategy_builder_mode()
            cfg_path: str | None = None
            variants = compare_basis = custom_total = target_mode = None
            use_file_run = False  # run_strategy_backtest_from_config path

            if is_builder:
                cfg = self._build_strategy_builder_config()
                self._last_strategy_cfg = self._strategy_config_to_dict(cfg)
                self._last_screen_cfg = None
                ok, errs, warns = self._preflight_config(cfg)
                if not ok:
                    raise ValueError(tr("preflight_failed") + "\n" + "\n".join(errs[:20]))
                if warns:
                    self.builder_state_var.set("WARN: " + " | ".join(warns[:2]))
                if funding_mode == "compare":
                    variants = self._build_funding_variants()
                    self._last_funding_variants = [self._funding_config_to_dict(v) for v in variants]
                    compare_basis = self._funding_compare_basis_internal()
                    custom_total = self._parse_funding_optional_float(
                        self.funding_custom_total_var.get(),
                        field_label=tr("funding_custom_total"),
                    )
                    self._publish_funding_parse_warnings()
                    target_mode = self._funding_view_internal()
                else:
                    cfg.funding = self._build_funding_config(mode=funding_mode)
                    cfg.initial_cash = float(cfg.funding.initial_cash)
                    ok, errs, warns = self._preflight_config(cfg)
                    if not ok:
                        raise ValueError(tr("preflight_failed") + "\n" + "\n".join(errs[:20]))
                    if warns:
                        self.builder_state_var.set("WARN: " + " | ".join(warns[:2]))
                    self._last_strategy_cfg = self._strategy_config_to_dict(cfg)
                    self._last_funding_variants = None
                result_name = tr("strategy_builder_name")
            else:
                cfg_path = str(self.strategy_path_var.get() or "").strip()
                if not cfg_path:
                    raise ValueError(tr("strategy_path_required"))
                cfg = load_strategy_config(cfg_path)
                if bool(self.override_strategy_var.get()):
                    cfg = self._apply_strategy_overrides(cfg)
                ok, errs, warns = self._preflight_config(cfg)
                if not ok:
                    raise ValueError(tr("preflight_failed") + "\n" + "\n".join(errs[:20]))
                if warns:
                    self.builder_state_var.set("WARN: " + " | ".join(warns[:2]))
                self._last_strategy_cfg = self._strategy_config_to_dict(cfg)
                self._last_screen_cfg = None
                if funding_mode == "compare":
                    variants = self._build_funding_variants()
                    self._last_funding_variants = [self._funding_config_to_dict(v) for v in variants]
                    compare_basis = self._funding_compare_basis_internal()
                    custom_total = self._parse_funding_optional_float(
                        self.funding_custom_total_var.get(),
                        field_label=tr("funding_custom_total"),
                    )
                    self._publish_funding_parse_warnings()
                    target_mode = self._funding_view_internal()
                else:
                    if bool(self.override_strategy_var.get()) or funding_mode in {"dca", "va"}:
                        cfg.funding = self._build_funding_config(mode=funding_mode)
                        cfg.initial_cash = float(cfg.funding.initial_cash)
                        self._last_strategy_cfg = self._strategy_config_to_dict(cfg)
                        self._last_funding_variants = None
                    else:
                        use_file_run = True
                        self._last_funding_variants = None
                result_name = tr("source_strategy")
        except Exception as exc:
            messagebox.showerror(tr("run_strategy_error"), str(exc))
            return

        # Switch to results tab and prepare UI.
        self._set_running(True, tr("running_strategy"))
        self._reset_progress()
        self.sub_nb.select(1)
        root = self.parent.winfo_toplevel()

        def _bg() -> None:
            try:
                if funding_mode == "compare":
                    res_compare, meta = run_funding_comparison(
                        cfg, variants, alignment_mode=compare_basis,
                        custom_total=custom_total, return_meta=True,
                    )
                    strategy_result = res_compare.get(target_mode) or res_compare.get("lump_sum")
                    if strategy_result is None and res_compare:
                        strategy_result = next(iter(res_compare.values()))
                    if strategy_result is not None:
                        strategy_result.name = result_name
                    def _done_compare() -> None:
                        try:
                            self.strategy_result = strategy_result
                            self.funding_compare_results = res_compare
                            self._funding_compare_results_by_mode = res_compare
                            self.funding_compare_meta = meta
                            self.compare_result = None
                            self._render_all()
                        except Exception as render_exc:
                            LOGGER.exception("render error after strategy compare run: %s", render_exc)
                        finally:
                            if getattr(self, "_loading_phase", False):
                                self._loading_phase = False
                                self._run_progress_bar.stop()
                                self._run_progress_bar.configure(mode="determinate")
                            self._run_progress_pct.set(100.0)
                            self._run_progress_status.set(tr("done"))
                            self._set_running(False, tr("done"))
                    root.after(0, _done_compare)
                elif use_file_run:
                    strategy_result = run_strategy_backtest_from_config(cfg_path)
                    if strategy_result is not None:
                        strategy_result.name = result_name
                    def _done_file() -> None:
                        try:
                            self.strategy_result = strategy_result
                            self.funding_compare_results = None
                            self._funding_compare_results_by_mode = None
                            self.funding_compare_meta = {}
                            self.compare_result = None
                            self._render_all()
                        except Exception as render_exc:
                            LOGGER.exception("render error after file strategy run: %s", render_exc)
                        finally:
                            if getattr(self, "_loading_phase", False):
                                self._loading_phase = False
                                self._run_progress_bar.stop()
                                self._run_progress_bar.configure(mode="determinate")
                            self._run_progress_pct.set(100.0)
                            self._run_progress_status.set(tr("done"))
                            self._set_running(False, tr("done"))
                    root.after(0, _done_file)
                else:
                    def _loading_cb(i: int, total: int, symbol: str) -> None:
                        status_text = f"데이터 로딩 중... ({i}/{total})"
                        root.after(0, lambda t=status_text: self._run_progress_status.set(t))
                    def _prog(pct: float, date_str: str, trades: list) -> None:
                        root.after(0, lambda: self._on_backtest_progress(pct, date_str, trades))
                    strategy_result = run_backtest(cfg, progress_callback=_prog, loading_callback=_loading_cb)
                    if strategy_result is not None:
                        strategy_result.name = result_name
                    def _done_simple() -> None:
                        try:
                            self.strategy_result = strategy_result
                            self.funding_compare_results = None
                            self._funding_compare_results_by_mode = None
                            self.funding_compare_meta = {}
                            self.compare_result = None
                            self._render_all()
                        except Exception as render_exc:
                            LOGGER.exception("render error after strategy run: %s", render_exc)
                        finally:
                            if getattr(self, "_loading_phase", False):
                                self._loading_phase = False
                                self._run_progress_bar.stop()
                                self._run_progress_bar.configure(mode="determinate")
                            self._run_progress_pct.set(100.0)
                            self._run_progress_status.set(tr("done"))
                            self._set_running(False, tr("done"))
                    root.after(0, _done_simple)
            except Exception as exc:
                err_msg = str(exc)
                LOGGER.exception("background strategy run failed: %s", exc)
                def _on_err() -> None:
                    self._set_running(False, tr("idle"))
                    messagebox.showerror(tr("run_strategy_error"), err_msg)
                root.after(0, _on_err)

        threading.Thread(target=_bg, daemon=True).start()

    def _run_compare(self) -> None:
        if self._running:
            return

        # Build configs and run preflight on the main thread.
        try:
            screen_cfg = self._build_screen_config()
            ok, errs, warns = self._preflight_config(screen_cfg)
            if not ok:
                raise ValueError(tr("preflight_failed") + "\n" + "\n".join(errs[:20]))
            funding_mode = self._funding_mode_internal()
            if funding_mode == "compare":
                funding_mode = "lump_sum"
            screen_cfg.funding = self._build_funding_config(mode=funding_mode)
            screen_cfg.initial_cash = float(screen_cfg.funding.initial_cash)
            self._last_screen_cfg = self._strategy_config_to_dict(screen_cfg)

            cfg_path: str | None = None
            use_file_run = False
            if self._is_strategy_builder_mode():
                strategy_cfg = self._build_strategy_builder_config()
                strategy_cfg.funding = self._build_funding_config(mode=funding_mode)
                strategy_cfg.initial_cash = float(strategy_cfg.funding.initial_cash)
                ok2, errs2, _warns2 = self._preflight_config(strategy_cfg)
                if not ok2:
                    raise ValueError(tr("preflight_failed") + "\n" + "\n".join(errs2[:20]))
                self._last_strategy_cfg = self._strategy_config_to_dict(strategy_cfg)
                strat_name = tr("strategy_builder_name")
            else:
                cfg_path = str(self.strategy_path_var.get() or "").strip()
                if not cfg_path:
                    raise ValueError(tr("strategy_path_required"))
                base_cfg = load_strategy_config(cfg_path)
                if bool(self.override_strategy_var.get()):
                    strategy_cfg = self._apply_strategy_overrides(base_cfg)
                    strategy_cfg.funding = self._build_funding_config(mode=funding_mode)
                    strategy_cfg.initial_cash = float(strategy_cfg.funding.initial_cash)
                    ok2, errs2, _warns2 = self._preflight_config(strategy_cfg)
                    if not ok2:
                        raise ValueError(tr("preflight_failed") + "\n" + "\n".join(errs2[:20]))
                    self._last_strategy_cfg = self._strategy_config_to_dict(strategy_cfg)
                else:
                    if funding_mode in {"dca", "va"}:
                        base_cfg.funding = self._build_funding_config(mode=funding_mode)
                        base_cfg.initial_cash = float(base_cfg.funding.initial_cash)
                        ok2, errs2, _warns2 = self._preflight_config(base_cfg)
                        if not ok2:
                            raise ValueError(tr("preflight_failed") + "\n" + "\n".join(errs2[:20]))
                        self._last_strategy_cfg = self._strategy_config_to_dict(base_cfg)
                        strategy_cfg = base_cfg
                    else:
                        use_file_run = True
                        self._last_strategy_cfg = self._strategy_config_to_dict(base_cfg)
                        strategy_cfg = base_cfg
                strat_name = tr("source_strategy")
        except Exception as exc:
            messagebox.showerror(tr("compare_error"), str(exc))
            return

        # Switch to results tab and prepare UI.
        self._set_running(True, tr("running_compare"))
        self._reset_progress()
        self.sub_nb.select(1)
        root = self.parent.winfo_toplevel()

        def _bg() -> None:
            try:
                def _loading_cb(i: int, total: int, symbol: str) -> None:
                    status_text = f"데이터 로딩 중... ({i}/{total})"
                    root.after(0, lambda t=status_text: self._run_progress_status.set(t))
                def _prog(pct: float, date_str: str, trades: list) -> None:
                    root.after(0, lambda: self._on_backtest_progress(pct * 0.5, date_str, trades))
                screen_res = run_backtest(screen_cfg, progress_callback=_prog, loading_callback=_loading_cb)
                screen_res.name = tr("source_screen")
                if use_file_run:
                    strat_res = run_strategy_backtest_from_config(cfg_path)
                else:
                    def _prog2(pct: float, date_str: str, trades: list) -> None:
                        root.after(0, lambda: self._on_backtest_progress(0.5 + pct * 0.5, date_str, trades))
                    strat_res = run_backtest(strategy_cfg, progress_callback=_prog2, loading_callback=_loading_cb)
                strat_res.name = strat_name
                cmp_result = compare_results(screen_res, strat_res)
                def _done() -> None:
                    try:
                        self.screen_result = screen_res
                        self.strategy_result = strat_res
                        self.funding_compare_results = None
                        self._funding_compare_results_by_mode = None
                        self._last_funding_variants = None
                        self.funding_compare_meta = {}
                        self.compare_result = cmp_result
                        self._render_all()
                    except Exception as render_exc:
                        LOGGER.exception("render error after compare run: %s", render_exc)
                    finally:
                        if getattr(self, "_loading_phase", False):
                            self._loading_phase = False
                            self._run_progress_bar.stop()
                            self._run_progress_bar.configure(mode="determinate")
                        self._run_progress_pct.set(100.0)
                        self._run_progress_status.set(tr("done"))
                        self._set_running(False, tr("done"))
                root.after(0, _done)
            except Exception as exc:
                err_msg = str(exc)
                LOGGER.exception("background compare run failed: %s", exc)
                def _on_err() -> None:
                    self._set_running(False, tr("idle"))
                    messagebox.showerror(tr("compare_error"), err_msg)
                root.after(0, _on_err)

        threading.Thread(target=_bg, daemon=True).start()

    def _current_export_mode(self) -> str:
        if self.funding_compare_results and self.compare_result is None:
            return "funding_compare"
        if self.compare_result is not None:
            return "strategy_compare"
        return "single"

    def _build_ui_context_snapshot(self) -> dict[str, Any]:
        return {
            "strategy_source": self._strategy_source_internal(),
            "market": self._effective_market(),
            "start": str(self.start_var.get() or "").strip(),
            "end": str(self.end_var.get() or "").strip(),
            "frequency": str(self.freq_var.get() or "").strip(),
            "holdings": int(self._parse_int(self.holdings_var.get(), 3)),
            "benchmark": str(self.benchmark_var.get() or "").strip(),
            "execution_timing": str(self.exec_timing_var.get() or "").strip(),
            "execution_price_basis": self._exec_price_basis_internal(),
            "price_offset_pct": float(self._parse_float(self.price_offset_pct_var.get(), 0.0) or 0.0),
            "screen_rule": str(self.screen_rule_var.get() or "").strip(),
            "table_source": self._table_source_internal(),
            "funding_mode": self._funding_mode_internal(),
            "funding_view": self._funding_view_internal(),
            "funding_compare_basis": self._funding_compare_basis_internal(),
            "funding_flow_view": str(self.funding_flow_view_var.get() or "").strip(),
            "lower_chart_mode": self._chart_mode_internal(),
            "show_contributed": bool(self.show_contributed_var.get()),
            "strategy_builder_summary": str(self.strategy_builder_summary_var.get() or ""),
            "builder_summary": str(self.builder_summary_var.get() or ""),
            "position_weight_pct": self._parse_float(self.position_weight_pct_var.get(), None),
            "max_holdings_cfg": self._parse_int(self.max_holdings_cfg_var.get(), 0),
            "max_new_buys_per_day": self._parse_int(self.max_new_buys_var.get(), 0),
            "max_buy_amount_per_position": self._parse_float(self.max_buy_amount_var.get(), None),
            "min_cash_reserve_pct": float(self._parse_float(self.min_cash_reserve_pct_var.get(), 0.0) or 0.0),
            "universe_filter": {
                "exchanges": self._csv_values(self.universe_exchanges_var.get(), upper=True),
                "size_buckets": self._csv_values(self.universe_size_buckets_var.get(), lower=True),
                "include_sectors": self._csv_values(self.universe_include_sectors_var.get()),
                "exclude_sectors": self._csv_values(self.universe_exclude_sectors_var.get()),
                "include_subsectors": self._csv_values(self.universe_include_subsectors_var.get()),
                "exclude_subsectors": self._csv_values(self.universe_exclude_subsectors_var.get()),
                "exclude_tickers": self._csv_values(self.universe_exclude_tickers_var.get(), upper=True),
                "watchlist_name": str(self.universe_watchlist_name_var.get() or "").strip() or None,
                "stock_type_included": bool(self.stock_type_included_var.get()),
                "ptp_included": bool(self.ptp_included_var.get()),
                "universe_presets": self._collect_universe_preset_state()[0],
                "universe_bucket_selections": self._collect_universe_preset_state()[1],
            },
        }

    def _build_strategy_context_for_export(self) -> dict[str, Any]:
        cfg = self._last_strategy_cfg or self._last_screen_cfg or {}
        market = cfg.get("market", self._effective_market())
        start = cfg.get("start", str(self.start_var.get() or "").strip())
        end = cfg.get("end", str(self.end_var.get() or "").strip() or None)
        frequency = cfg.get("frequency", str(self.freq_var.get() or "").strip().upper())
        execution_timing = cfg.get("execution_timing", str(self.exec_timing_var.get() or "").strip())
        buy_rules = cfg.get("buy_rules", str(self.screen_rule_var.get() or "").strip())

        ranking_summary = "-"
        ranking = cfg.get("ranking", {})
        if isinstance(ranking, dict):
            factors = ranking.get("factors", [])
            if isinstance(factors, list) and factors:
                parts: list[str] = []
                for f in factors[:20]:
                    if not isinstance(f, dict):
                        continue
                    name = str(f.get("name", "")).strip()
                    if not name:
                        continue
                    weight = pd.to_numeric(pd.Series([f.get("weight")]), errors="coerce").iloc[0]
                    direction = str(f.get("direction", "")).strip()
                    if pd.notna(weight):
                        parts.append(f"{name}({float(weight):.3g} {direction})")
                    else:
                        parts.append(f"{name}({direction})")
                if parts:
                    ranking_summary = ", ".join(parts)

        universe_summary = "-"
        universe = cfg.get("universe", {})
        if isinstance(universe, dict):
            source = str(universe.get("source", "")).strip()
            symbols = universe.get("symbols", [])
            if isinstance(symbols, list) and symbols:
                universe_summary = f"{source}:{len(symbols)} symbols"
            elif source:
                universe_summary = source

        return {
            "name": cfg.get("name", str((self.strategy_result or self.screen_result).name if (self.strategy_result or self.screen_result) is not None else "strategy")),
            "market": market,
            "start": start,
            "end": end,
            "rebalance_frequency": frequency,
            "execution_timing": execution_timing,
            "execution_price_basis": (cfg.get("execution", {}) or {}).get("price_basis", self._exec_price_basis_internal())
            if isinstance(cfg.get("execution", {}), dict)
            else self._exec_price_basis_internal(),
            "execution_price_offset_pct": (cfg.get("execution", {}) or {}).get("price_offset_pct", 0.0)
            if isinstance(cfg.get("execution", {}), dict)
            else 0.0,
            "screen_rule": buy_rules,
            "ranking_summary": ranking_summary,
            "sell_rule_summary": cfg.get("sell_mode", ""),
            "universe_summary": universe_summary,
            "position_sizing": cfg.get("position_sizing", {}),
            "risk_limits": cfg.get("risk_limits", {}),
            "universe_filter": cfg.get("universe_filter", {}),
            "effective_config": cfg,
        }

    def _build_funding_context_for_export(self) -> dict[str, Any]:
        cfg = self._last_strategy_cfg or self._last_screen_cfg or {}
        funding = cfg.get("funding", {})
        if not isinstance(funding, dict):
            funding = {}
        context = {
            "enabled": True,
            "funding_mode": funding.get("mode", self._funding_mode_internal()),
            "contribution_frequency": funding.get("contribution_freq", self._funding_contribution_freq_internal()),
            "initial_cash": funding.get("initial_cash"),
            "dca_fixed_contribution": funding.get("fixed_contribution"),
            "va_target_step": funding.get("va_target_step"),
            "va_min_contribution": funding.get("va_min_contribution"),
            "va_max_contribution": funding.get("va_max_contribution"),
            "va_min_policy": funding.get("va_min_policy"),
            "va_allow_withdrawal": funding.get("va_allow_withdrawal"),
            "compare_alignment_mode": self._funding_compare_basis_internal() if self._current_export_mode() == "funding_compare" else "none",
        }
        if self._last_funding_variants:
            context["funding_variants"] = deepcopy(self._last_funding_variants)
        return context

    def _save_ai_review_bundle(self) -> None:
        mode = self._current_export_mode()
        if mode == "single" and self._target_result_for_tables() is None:
            messagebox.showwarning(tr("ai_export"), tr("ai_export_no_result"), parent=self.parent.winfo_toplevel())
            return
        if mode == "funding_compare" and not self.funding_compare_results:
            messagebox.showwarning(tr("ai_export"), tr("ai_export_no_result"), parent=self.parent.winfo_toplevel())
            return
        if mode == "strategy_compare" and (self.screen_result is None or self.strategy_result is None):
            messagebox.showwarning(tr("ai_export"), tr("ai_export_no_result"), parent=self.parent.winfo_toplevel())
            return

        default_root = Path("exports") / "ai_review"
        default_root.mkdir(parents=True, exist_ok=True)
        selected_dir = filedialog.askdirectory(
            title=tr("ai_export"),
            initialdir=str(default_root),
            parent=self.parent.winfo_toplevel(),
        )
        if not selected_dir:
            return

        try:
            strategy_ctx = self._build_strategy_context_for_export()
            funding_ctx = self._build_funding_context_for_export()
            ui_snapshot = self._build_ui_context_snapshot()

            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            strat_name = str(strategy_ctx.get("name", "strategy")).strip().replace(" ", "_")
            safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in strat_name) or "strategy"
            bundle_dir = Path(selected_dir) / f"{ts}_{safe_name}"

            single_result = self._target_result_for_tables() if mode == "single" else None
            funding_compare_results = self.funding_compare_results if mode == "funding_compare" else None
            strategy_compare_results = (
                {"screen": self.screen_result, "strategy": self.strategy_result}
                if mode == "strategy_compare"
                else None
            )
            compare_notes = deepcopy(self.funding_compare_meta) if mode == "funding_compare" else {}
            if self._last_funding_variants:
                compare_notes["funding_variants"] = deepcopy(self._last_funding_variants)

            out = export_ai_review_bundle(
                bundle_dir,
                mode=mode,
                strategy_context=strategy_ctx,
                funding_context=funding_ctx,
                single_result=single_result,
                funding_compare_results=funding_compare_results,
                strategy_compare_results=strategy_compare_results,
                compare_notes=compare_notes,
                ui_context_snapshot=ui_snapshot,
                app_version=APP_VERSION,
                snapshot_validation_mode="fail",
            )
            messagebox.showinfo(tr("ai_export_done"), str(out), parent=self.parent.winfo_toplevel())
        except Exception as exc:
            messagebox.showerror(tr("ai_export_error"), str(exc), parent=self.parent.winfo_toplevel())

    def _target_result_for_tables(self):
        if self.funding_compare_results:
            mode = self._funding_view_internal()
            target = self.funding_compare_results.get(mode)
            if target is None:
                target = self.funding_compare_results.get("lump_sum")
            if target is None:
                try:
                    target = next(iter(self.funding_compare_results.values()))
                except StopIteration:
                    target = None
            if target is not None:
                return target
        source = self._table_source_internal()
        if source == "screen":
            return self.screen_result
        if source == "strategy":
            return self.strategy_result
        if self.strategy_result is not None:
            return self.strategy_result
        return self.screen_result

    def _render_all(self) -> None:
        self._render_summary()
        self._render_charts()
        self._render_trade_analytics()
        self._render_tables()

    def _render_summary(self) -> None:
        target = self._target_result_for_tables()
        if target is None:
            self.compare_context_var.set("")
            self.kpi_vars["cagr"].set(f"{tr('kpi_cagr')} -")
            self.kpi_vars["mdd"].set(f"{tr('kpi_mdd')} -")
            self.kpi_vars["sharpe"].set(f"{tr('kpi_sharpe')} -")
            self.kpi_vars["turnover"].set(f"{tr('kpi_turnover')} -")
            self.kpi_vars["trades"].set(f"{tr('kpi_trades')} -")
            if hasattr(self, "kpi_labels"):
                for lbl in self.kpi_labels.values():
                    lbl.configure(style="KPI.TLabel")
            return

        if self.funding_compare_results and self.compare_result is None:
            self.compare_context_var.set(tr("context_funding_compare"))
            meta = self.funding_compare_meta if isinstance(self.funding_compare_meta, dict) else {}
            align_mode = str(meta.get("alignment_mode", "default"))
            applied = bool(meta.get("alignment_applied", False))
            if applied and align_mode == "align_lump_to_dca_total":
                aligned = pd.to_numeric(pd.Series([meta.get("aligned_lump_initial_cash")]), errors="coerce").iloc[0]
                dca_total = pd.to_numeric(pd.Series([meta.get("dca_total_contributed")]), errors="coerce").iloc[0]
                dca_init = float(self._parse_float(self.initial_cash_var.get(), 0.0) or 0.0)
                dca_fixed = pd.to_numeric(pd.Series([meta.get("dca_fixed_contribution")]), errors="coerce").fillna(0.0).iloc[0]
                dca_events = int(pd.to_numeric(pd.Series([meta.get("dca_event_count")]), errors="coerce").fillna(0).iloc[0])
                if pd.notna(aligned):
                    note = (
                        f"비교 기준: Lump 초기투자금 = DCA 총납입원금 "
                        f"| Lump 기준금액 = {float(aligned):,.0f} "
                        f"(초기 {dca_init:,.0f} + DCA {float(dca_fixed):,.0f} x {dca_events}회)"
                    )
                elif pd.notna(dca_total):
                    note = (
                        f"비교 기준: Lump 초기투자금 = DCA 총납입원금 "
                        f"| DCA 총원금 = {float(dca_total):,.0f}"
                    )
                else:
                    note = "비교 기준: Lump 초기투자금 = DCA 총납입원금"
                self.funding_info_var.set(note)
            elif applied and align_mode == "custom_total":
                aligned = pd.to_numeric(pd.Series([meta.get("aligned_lump_initial_cash")]), errors="coerce").iloc[0]
                note = (
                    f"비교 기준: 사용자 지정 총원금 "
                    f"| Lump 기준금액 = {float(aligned):,.0f}"
                    if pd.notna(aligned)
                    else "비교 기준: 사용자 지정 총원금"
                )
                self.funding_info_var.set(note)
            elif align_mode == "default":
                self.funding_info_var.set("비교 기준: 기본(각 방식 원설정 유지)")
            else:
                reason = str(meta.get("note", "")).strip()
                base = "비교 기준: Lump 정렬 미적용"
                self.funding_info_var.set(f"{base} | {reason}" if reason else base)
        elif self.compare_result is not None:
            self.compare_context_var.set(tr("context_strategy_compare"))
        else:
            self.compare_context_var.set("")

        m = target.metrics
        cagr_val = float(m.get("twr_cagr", m.get("cagr", 0.0)))
        mdd_val   = float(m.get("mdd", 0.0))
        self.kpi_vars["cagr"].set(f"{tr('kpi_cagr')} {cagr_val * 100.0:.2f}%")
        self.kpi_vars["mdd"].set(f"{tr('kpi_mdd')} {mdd_val * 100.0:.2f}%")
        self.kpi_vars["sharpe"].set(f"{tr('kpi_sharpe')} {float(m.get('sharpe', 0.0)):.2f}")
        self.kpi_vars["turnover"].set(f"{tr('kpi_turnover')} {float(m.get('turnover', 0.0)):.2f}")
        self.kpi_vars["trades"].set(f"{tr('kpi_trades')} {int(m.get('trades', 0))}")
        # Colour KPI cards: CAGR green/red, MDD always red shade
        if hasattr(self, "kpi_labels"):
            self.kpi_labels["cagr"].configure(style="KPI.Pos.TLabel" if cagr_val >= 0 else "KPI.Neg.TLabel")
            self.kpi_labels["mdd"].configure(style="KPI.Neg.TLabel" if mdd_val < -0.01 else "KPI.TLabel")

    def _render_charts(self) -> None:
        self.ax_top.clear()
        self._ax_top_twin.clear()
        self.ax_bottom.clear()

        if self.funding_compare_results:
            mode_map = {
                "lump_sum": self._funding_lump_mode_label(),
                "dca": tr("funding_mode_dca"),
                "va": tr("funding_mode_va"),
            }
            palette = {
                "lump_sum": "#2f7ed8",
                "dca": "#d94f4f",
                "va": "#0b7a3a",
            }
            plotted_bench = False
            for mode in ["lump_sum", "dca", "va"]:
                result = self.funding_compare_results.get(mode)
                if result is None or result.equity_curve.empty:
                    continue
                color = palette.get(mode, "#444444")
                label = mode_map.get(mode, mode)
                eq = result.equity_curve.copy()
                dates = pd.to_datetime(eq.get("date"), errors="coerce")
                vals = pd.to_numeric(eq.get("equity"), errors="coerce")
                self.ax_top.plot(dates, vals, label=label, linewidth=1.6, color=color)

                if bool(self.show_contributed_var.get()) and result.funding_flows is not None and not result.funding_flows.empty:
                    flows = result.funding_flows.copy()
                    flows["date"] = pd.to_datetime(flows.get("date"), errors="coerce")
                    flows["contribution"] = pd.to_numeric(flows.get("contribution"), errors="coerce").fillna(0.0)
                    cum_contrib = flows.groupby("date")["contribution"].sum().sort_index().cumsum()
                    cum_contrib = cum_contrib.reindex(pd.DatetimeIndex(dates.dropna().unique())).ffill().fillna(0.0)
                    self.ax_top.plot(
                        cum_contrib.index,
                        cum_contrib.values,
                        label=f"{label} {tr('legend_contributed')}",
                        linewidth=1.0,
                        linestyle=":",
                        color=color,
                        alpha=0.75,
                    )

                if not result.benchmark_curve.empty and not plotted_bench:
                    b = result.benchmark_curve.copy()
                    self.ax_top.plot(
                        pd.to_datetime(b["date"], errors="coerce"),
                        pd.to_numeric(b["equity"], errors="coerce"),
                        label=tr("benchmark"),
                        linewidth=1.2,
                        color="#444444",
                        linestyle="--",
                    )
                    plotted_bench = True

                mode_key = self._chart_mode_internal()
                if mode_key == "drawdown":
                    if not result.drawdown_curve.empty:
                        dd = result.drawdown_curve.copy()
                        self.ax_bottom.plot(
                            pd.to_datetime(dd["date"], errors="coerce"),
                            pd.to_numeric(dd["drawdown"], errors="coerce"),
                            label=f"{label} {tr('drawdown')}",
                            linewidth=1.2,
                            color=color,
                        )
                else:
                    if not result.excess_curve.empty:
                        ex = result.excess_curve.copy()
                        self.ax_bottom.plot(
                            pd.to_datetime(ex["date"], errors="coerce"),
                            pd.to_numeric(ex["cum_excess"], errors="coerce"),
                            label=f"{label} {tr('relative_curve')}",
                            linewidth=1.2,
                            color=color,
                        )

            if self.ax_top.has_data():
                self.ax_top.legend(loc="best")
                self.ax_top.set_title(tr("equity_curve"))
                self.ax_top.grid(True, alpha=0.2)
            else:
                self.ax_top.text(0.5, 0.5, tr("screen_or_strategy_hint"), ha="center", va="center", transform=self.ax_top.transAxes)

            mode_key = self._chart_mode_internal()
            self.ax_bottom.set_title(tr("drawdown") if mode_key == "drawdown" else tr("relative_curve"))
            if self.ax_bottom.has_data():
                self.ax_bottom.legend(loc="best")
                self.ax_bottom.grid(True, alpha=0.2)
            self._ax_top_twin.tick_params(right=False, labelright=False)
            for _sp in self._ax_top_twin.spines.values():
                _sp.set_visible(False)
            self.fig.tight_layout()
            self.canvas.draw_idle()
            return

        screen_label = str(self.screen_result.name) if self.screen_result is not None and str(self.screen_result.name).strip() else tr("source_screen")
        strategy_label = str(self.strategy_result.name) if self.strategy_result is not None and str(self.strategy_result.name).strip() else tr("source_strategy")
        plotted_bench = False
        for label, result, color in [
            (screen_label, self.screen_result, "#2f7ed8"),
            (strategy_label, self.strategy_result, "#d94f4f"),
        ]:
            if result is None or result.equity_curve.empty:
                continue

            eq = result.equity_curve.copy()
            self.ax_top.plot(
                pd.to_datetime(eq["date"], errors="coerce"),
                pd.to_numeric(eq["equity"], errors="coerce"),
                label=label,
                linewidth=1.6,
                color=color,
            )

            if not result.benchmark_curve.empty and not plotted_bench:
                b = result.benchmark_curve.copy()
                self.ax_top.plot(
                    pd.to_datetime(b["date"], errors="coerce"),
                    pd.to_numeric(b["equity"], errors="coerce"),
                    label=tr("benchmark"),
                    linewidth=1.2,
                    color="#444444",
                    linestyle="--",
                )
                plotted_bench = True

            mode = self._chart_mode_internal()
            if mode == "drawdown":
                if not result.drawdown_curve.empty:
                    dd = result.drawdown_curve.copy()
                    self.ax_bottom.plot(
                        pd.to_datetime(dd["date"], errors="coerce"),
                        pd.to_numeric(dd["drawdown"], errors="coerce"),
                        label=f"{label} {tr('drawdown')}",
                        linewidth=1.2,
                        color=color,
                    )
            else:
                if not result.excess_curve.empty:
                    ex = result.excess_curve.copy()
                    self.ax_bottom.plot(
                        pd.to_datetime(ex["date"], errors="coerce"),
                        pd.to_numeric(ex["cum_excess"], errors="coerce"),
                        label=f"{label} {tr('relative_curve')}",
                        linewidth=1.2,
                        color=color,
                    )

        # Buy/sell bars overlaid on equity curve via secondary y-axis
        _twin_has_data = False
        target_for_trades = self.strategy_result if self.strategy_result is not None else self.screen_result
        if target_for_trades is not None and target_for_trades.trades is not None and not target_for_trades.trades.empty:
            try:
                import matplotlib.dates as mdates_local
                import matplotlib.ticker as _mticker
                tr_df = target_for_trades.trades.copy()
                tr_df["exec_date"] = pd.to_datetime(tr_df.get("exec_date"), errors="coerce")
                tr_df["notional"] = pd.to_numeric(tr_df.get("notional"), errors="coerce").fillna(0.0)
                tr_df = tr_df.dropna(subset=["exec_date"])
                buy_by_date = tr_df[tr_df["side"] == "buy"].groupby("exec_date")["notional"].sum()
                sell_by_date = tr_df[tr_df["side"] == "sell"].groupby("exec_date")["notional"].sum()
                if not buy_by_date.empty or not sell_by_date.empty:
                    bar_w = 3.0
                    if not buy_by_date.empty:
                        bx = mdates_local.date2num(buy_by_date.index.to_pydatetime())
                        self._ax_top_twin.bar(bx, buy_by_date.values, width=bar_w, color="#0b7a3a", alpha=0.28, label="매수")
                    if not sell_by_date.empty:
                        sx = mdates_local.date2num(sell_by_date.index.to_pydatetime())
                        self._ax_top_twin.bar(sx, sell_by_date.values, width=bar_w, color="#b42318", alpha=0.28, label="매도")

                    def _fmt_money_axis(v, _):
                        if abs(v) >= 1e6:
                            return f"{v/1e6:.1f}M"
                        return f"{int(v/1e3)}K"

                    self._ax_top_twin.yaxis.set_major_formatter(_mticker.FuncFormatter(_fmt_money_axis))
                    self._ax_top_twin.set_ylabel("거래금액", fontsize=7)
                    self._ax_top_twin.tick_params(axis="y", labelright=True, right=True, labelsize=7)
                    for _sp_name, _sp in self._ax_top_twin.spines.items():
                        _sp.set_visible(_sp_name == "right")
                    _twin_has_data = True
            except Exception:
                pass

        if not _twin_has_data:
            self._ax_top_twin.tick_params(right=False, labelright=False)
            for _sp in self._ax_top_twin.spines.values():
                _sp.set_visible(False)

        if self.ax_top.has_data():
            _h1, _l1 = self.ax_top.get_legend_handles_labels()
            _h2, _l2 = self._ax_top_twin.get_legend_handles_labels()
            self.ax_top.legend(_h1 + _h2, _l1 + _l2, loc="upper left")
            self.ax_top.set_title(tr("equity_curve"))
            self.ax_top.grid(True, alpha=0.2)
        else:
            self.ax_top.text(0.5, 0.5, tr("screen_or_strategy_hint"), ha="center", va="center", transform=self.ax_top.transAxes)

        mode = self._chart_mode_internal()
        self.ax_bottom.set_title(tr("drawdown") if mode == "drawdown" else tr("relative_curve"))
        if self.ax_bottom.has_data():
            self.ax_bottom.legend(loc="best")
            self.ax_bottom.grid(True, alpha=0.2)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for iid in tree.get_children():
            tree.delete(iid)

    def _set_trade_reason_text(self, text: str) -> None:
        self.trade_reason_text.configure(state="normal")
        self.trade_reason_text.delete("1.0", tk.END)
        if text:
            self.trade_reason_text.insert("1.0", text)
        self.trade_reason_text.configure(state="disabled")

    def _on_trade_select(self, _event=None) -> None:
        selected = self.trades_tree.selection()
        if not selected:
            self._set_trade_reason_text("")
            return
        values = self.trades_tree.item(selected[0], "values")
        reason_text = str(values[8]) if values and len(values) > 8 else ""
        self._set_trade_reason_text(reason_text)

    def _fmt_pct(self, value: float | int | None, signed: bool = False) -> str:
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(num):
            return "-"
        if signed:
            return f"{float(num) * 100.0:+.2f}%"
        return f"{float(num) * 100.0:.2f}%"

    def _fmt_money(self, value: float | int | None) -> str:
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(num):
            return "-"
        return f"{float(num):,.0f}"

    def _fmt_number(self, value: float | int | None, digits: int = 2) -> str:
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(num):
            return "-"
        return f"{float(num):.{int(digits)}f}"

    def _format_side(self, value: Any) -> str:
        side = str(value or "").strip().lower()
        if side == "buy":
            return "매수"
        if side == "sell":
            return "매도"
        return str(value or "")

    def _prepare_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df is None or trades_df.empty:
            return pd.DataFrame()
        out = trades_df.copy()
        out["exec_date"] = pd.to_datetime(out.get("exec_date"), errors="coerce")
        out = out.dropna(subset=["exec_date"]).copy()
        out["ticker"] = out.get("ticker", "").astype(str).str.upper()
        out["side"] = out.get("side", "").astype(str).str.lower()
        out["notional"] = pd.to_numeric(out.get("notional"), errors="coerce").fillna(0.0)
        out["month"] = out["exec_date"].dt.to_period("M").astype(str)
        signed = np.where(out["side"] == "buy", 1.0, np.where(out["side"] == "sell", -1.0, 0.0))
        out["signed_count"] = signed
        out["signed_notional"] = out["notional"] * signed
        return out

    def _refresh_trade_filter_options(self, trades_df: pd.DataFrame) -> None:
        self._suspend_trade_filter_trace = True
        try:
            month_options = [tr("all")]
            ticker_options = [tr("all")]
            if trades_df is not None and not trades_df.empty:
                months = sorted(set(trades_df["month"].astype(str).tolist()))
                tickers = sorted(set(trades_df["ticker"].astype(str).tolist()))
                month_options += months
                ticker_options += tickers

            current_month = self.trade_filter_month_var.get()
            current_ticker = self.trade_filter_ticker_var.get()

            self.trade_month_combo.configure(values=month_options)
            self.trade_ticker_combo.configure(values=ticker_options)

            if current_month not in month_options:
                self.trade_filter_month_var.set(tr("all"))
            if current_ticker not in ticker_options:
                self.trade_filter_ticker_var.set(tr("all"))
        finally:
            self._suspend_trade_filter_trace = False

    def _filter_trades_by_controls(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df is None or trades_df.empty:
            return pd.DataFrame()

        out = trades_df.copy()
        month_sel = str(self.trade_filter_month_var.get() or tr("all"))
        ticker_sel = str(self.trade_filter_ticker_var.get() or tr("all")).upper()

        if month_sel != tr("all"):
            out = out.loc[out["month"] == month_sel]
        if ticker_sel != tr("all"):
            out = out.loc[out["ticker"] == ticker_sel]
        return out

    def _on_trade_filter_change(self) -> None:
        if self._suspend_trade_filter_trace:
            return
        self._render_tables()

    def _render_trade_analytics(self) -> None:
        target = self._target_result_for_tables()
        trades_raw = target.trades.copy() if target is not None and target.trades is not None and not target.trades.empty else pd.DataFrame()
        trades = self._prepare_trades(trades_raw)

        self._refresh_trade_filter_options(trades)
        self.heatmap_fig.clear()
        self.trade_summary_fig.clear()

        ax_heat = self.heatmap_fig.add_subplot(111)
        self._heatmap_ax = ax_heat
        summary_gs = self.trade_summary_fig.add_gridspec(2, 1, height_ratios=[1.0, 2.2], hspace=0.42)
        ax_sum = self.trade_summary_fig.add_subplot(summary_gs[0, 0])
        ax_ret = self.trade_summary_fig.add_subplot(summary_gs[1, 0])
        self._heatmap_rows = []
        self._heatmap_cols = []
        self._heatmap_values = np.empty((0, 0), dtype=float)
        self._heatmap_trades = trades.copy() if trades is not None and not trades.empty else pd.DataFrame()
        self._heatmap_hover_last_key = None
        self._heatmap_hover_pending_key = None
        self._heatmap_hover_pending_pointer = None
        self.heatmap_tooltip.hide()

        if trades.empty:
            ax_heat.text(0.5, 0.5, tr("no_trade_data"), ha="center", va="center", transform=ax_heat.transAxes)
            ax_sum.text(0.5, 0.5, tr("no_trade_data"), ha="center", va="center", transform=ax_sum.transAxes)
            ax_ret.text(0.5, 0.5, tr("no_trade_data"), ha="center", va="center", transform=ax_ret.transAxes)
            ax_heat.set_axis_off()
            ax_sum.set_axis_off()
            ax_ret.set_axis_off()
            self.heatmap_canvas_widget.configure(height=420)
            self.heatmap_canvas.draw_idle()
            self.trade_summary_canvas.draw_idle()
            self._set_hover_info(tr("hover_default"))
            self.heatmap_tooltip.hide()
            self._schedule_heatmap_scrollregion()
            return

        metric_mode = self._trade_metric_internal()
        sort_mode = self._trade_sort_internal()
        scope_mode = self._trade_scope_internal()
        top_n = max(1, self._parse_int(self.trade_top_n_var.get(), 20))

        metric_col = "signed_count" if metric_mode == "net_count" else "signed_notional"
        pivot = trades.groupby(["ticker", "month"])[metric_col].sum().unstack("month").fillna(0.0)

        if sort_mode == "count":
            score = trades.groupby("ticker").size().astype(float)
        else:
            score = trades.groupby("ticker")["notional"].sum().abs()

        ordered_tickers = score.sort_values(ascending=False).index
        if scope_mode == "top_n":
            ordered_tickers = ordered_tickers[:top_n]
        pivot = pivot.reindex(ordered_tickers).fillna(0.0)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        if pivot.empty:
            ax_heat.text(0.5, 0.5, tr("no_trade_data"), ha="center", va="center", transform=ax_heat.transAxes)
            ax_heat.set_axis_off()
            self.heatmap_canvas_widget.configure(height=420)
            self._set_hover_info(tr("hover_default"))
            self.heatmap_tooltip.hide()
        else:
            arr = pivot.to_numpy(dtype=float)
            self._heatmap_rows = [str(t) for t in pivot.index]
            self._heatmap_cols = [str(m) for m in pivot.columns]
            self._heatmap_values = arr
            n_tickers = int(arr.shape[0])
            heat_width_px = max(520, int(self.heatmap_scroll_canvas.winfo_width()) - 20)

            if n_tickers >= 500:
                dpi = 78
                row_height_px = 10
                y_step = 3
                y_font = 6.5
            elif n_tickers >= 300:
                dpi = 80
                row_height_px = 11
                y_step = 2
                y_font = 7.0
            elif n_tickers >= 150:
                dpi = 90
                row_height_px = 12
                y_step = 2
                y_font = 7.0
            else:
                dpi = 95
                row_height_px = 14
                y_step = 1
                y_font = 8.0

            heat_height_px = max(520, (n_tickers * row_height_px) + 120)
            self.heatmap_fig.set_dpi(dpi)
            self.heatmap_fig.set_size_inches(heat_width_px / dpi, heat_height_px / dpi, forward=True)

            vmax = float(np.nanmax(np.abs(arr))) if arr.size > 0 else 1.0
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0

            im = ax_heat.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            months = list(pivot.columns)
            tickers = list(pivot.index)

            y_ticks = list(range(0, len(tickers), y_step))
            if tickers and (len(tickers) - 1) not in y_ticks:
                y_ticks.append(len(tickers) - 1)
            ax_heat.set_yticks(y_ticks)
            ax_heat.set_yticklabels([tickers[i] for i in y_ticks], fontsize=y_font)

            if months:
                # Show x ticks every 3 months.
                x_ticks = list(range(0, len(months), 3))
                if (len(months) - 1) not in x_ticks:
                    x_ticks.append(len(months) - 1)
                ax_heat.set_xticks(x_ticks)
                ax_heat.set_xticklabels([months[i] for i in x_ticks], fontsize=7, rotation=35, ha="right")

            title_key = "heatmap_title_count" if metric_mode == "net_count" else "heatmap_title_notional"
            ax_heat.set_title(tr(title_key), fontsize=10)
            self.heatmap_fig.colorbar(im, ax=ax_heat, fraction=0.022, pad=0.01)
            self.heatmap_fig.subplots_adjust(top=0.95, bottom=0.08, left=0.12, right=0.98)
            self.heatmap_canvas_widget.configure(height=int(heat_height_px), width=int(heat_width_px))
            self._set_hover_info(tr("hover_default"))

        month_order = sorted(set(trades["month"].astype(str).tolist()))
        month_dt = pd.to_datetime(month_order, errors="coerce")
        valid_month = month_dt.notna()
        month_order = [m for m, keep in zip(month_order, valid_month) if bool(keep)]
        month_dt = month_dt[valid_month]

        buy_counts = trades.loc[trades["side"] == "buy"].groupby("month").size().reindex(month_order, fill_value=0)
        sell_counts = trades.loc[trades["side"] == "sell"].groupby("month").size().reindex(month_order, fill_value=0)

        x_sum = mdates.date2num(month_dt.to_pydatetime()) if len(month_dt) > 0 else np.array([], dtype=float)
        ax_sum.plot(x_sum, buy_counts.to_numpy(dtype=float), color="#0b7a3a", linewidth=1.5, label=tr("buy_count"))
        ax_sum.plot(x_sum, sell_counts.to_numpy(dtype=float), color="#b42318", linewidth=1.5, label=tr("sell_count"))
        ax_sum.set_title(tr("monthly_trade_summary"), fontsize=9)
        ax_sum.grid(True, alpha=0.2)
        if len(x_sum) > 0:
            ax_sum.set_xlim(float(x_sum.min() - 18.0), float(x_sum.max() + 18.0))
        self._apply_year_month_ticks(ax_sum, show_major_labels=False)
        ax_sum.legend(loc="upper left", fontsize=8)

        monthly_returns = target.monthly_returns.copy() if target is not None and target.monthly_returns is not None else pd.DataFrame()
        if monthly_returns is None or monthly_returns.empty:
            ax_ret.text(0.5, 0.5, tr("no_trade_data"), ha="center", va="center", transform=ax_ret.transAxes)
            ax_ret.set_axis_off()
        else:
            mdf = monthly_returns.copy()
            mdf["month"] = mdf.get("month", "").astype(str)
            mdf["month_dt"] = pd.to_datetime(mdf["month"], errors="coerce")
            mdf["strategy_ret"] = pd.to_numeric(mdf.get("strategy_ret"), errors="coerce")
            mdf["bench_ret"] = pd.to_numeric(mdf.get("bench_ret"), errors="coerce")
            mdf["excess_ret"] = pd.to_numeric(mdf.get("excess_ret"), errors="coerce")
            mdf = mdf.dropna(subset=["month_dt"]).sort_values("month_dt")

            if mdf.empty:
                ax_ret.text(0.5, 0.5, tr("no_trade_data"), ha="center", va="center", transform=ax_ret.transAxes)
                ax_ret.set_axis_off()
            else:
                ret_mode = self._return_plot_mode_internal()
                cum_mode = self._cum_line_mode_internal()

                x = mdates.date2num(mdf["month_dt"].dt.to_pydatetime())
                strategy_pct = mdf["strategy_ret"].fillna(0.0).to_numpy(dtype=float) * 100.0
                bench_pct = mdf["bench_ret"].fillna(0.0).to_numpy(dtype=float) * 100.0
                excess_pct = mdf["excess_ret"].fillna(0.0).to_numpy(dtype=float) * 100.0

                primary_colors = ["#2f7ed8" if v >= 0 else "#d94f4f" for v in strategy_pct]
                secondary_colors = ["#93c5fd" if v >= 0 else "#fca5a5" for v in (bench_pct if ret_mode == "strategy_bench" else excess_pct)]

                if ret_mode == "excess_only":
                    bar_w = 18.0
                    ax_ret.bar(x, excess_pct, width=bar_w, color=["#2f7ed8" if v >= 0 else "#d94f4f" for v in excess_pct], label=tr("legend_excess"), zorder=2, align="center")
                    bar_values_for_ylim = excess_pct
                else:
                    bar_w = 12.0
                    ax_ret.bar(x - (bar_w * 0.52), strategy_pct, width=bar_w, color=primary_colors, label=tr("legend_strategy"), zorder=2, align="center")
                    if ret_mode == "strategy_bench":
                        ax_ret.bar(x + (bar_w * 0.52), bench_pct, width=bar_w, color=secondary_colors, alpha=0.75, label=tr("legend_benchmark"), zorder=2, align="center")
                        bar_values_for_ylim = np.concatenate([strategy_pct, bench_pct])
                    else:
                        ax_ret.bar(x + (bar_w * 0.52), excess_pct, width=bar_w, color=secondary_colors, alpha=0.75, label=tr("legend_excess"), zorder=2, align="center")
                        bar_values_for_ylim = np.concatenate([strategy_pct, excess_pct])

                ax_ret.axhline(0.0, color="#4b5563", linewidth=0.9, alpha=0.75, zorder=1)
                ax_ret.set_title(tr("monthly_return_chart"), fontsize=9)
                ax_ret.set_ylabel("%", fontsize=8)
                ax_ret.grid(True, axis="y", alpha=0.22)

                if len(x) > 0:
                    ax_ret.set_xlim(float(np.min(x) - 18.0), float(np.max(x) + 18.0))

                min_ret = float(np.nanmin(bar_values_for_ylim)) if len(bar_values_for_ylim) > 0 else -2.0
                max_ret = float(np.nanmax(bar_values_for_ylim)) if len(bar_values_for_ylim) > 0 else 2.0
                if not np.isfinite(min_ret):
                    min_ret = -2.0
                if not np.isfinite(max_ret):
                    max_ret = 2.0
                span = max(max_ret - min_ret, 1e-6)
                pad = max(2.5, span * 0.12)
                y_min = min(min_ret - pad, -pad)
                y_max = max(max_ret + pad, pad)
                ax_ret.set_ylim(y_min, y_max)

                line_ax = None
                if cum_mode != "none":
                    line_ax = ax_ret.twinx()
                    if cum_mode == "cum_excess":
                        cum_line = mdf["excess_ret"].fillna(0.0).cumsum().to_numpy(dtype=float) * 100.0
                        cum_label = tr("cum_mode_excess")
                        line_color = "#111827"
                    else:
                        cum_line = ((1.0 + mdf["strategy_ret"].fillna(0.0)).cumprod() - 1.0).to_numpy(dtype=float) * 100.0
                        cum_label = tr("cum_mode_strategy")
                        line_color = "#0f766e"
                    line_ax.plot(x, cum_line, color=line_color, linewidth=1.8, label=cum_label, zorder=3)
                    line_ax.set_ylabel("%", fontsize=8)
                    line_ax.grid(False)
                    line_ax.tick_params(axis="x", which="both", labelbottom=False)

                self._apply_year_month_ticks(ax_ret, show_major_labels=True)

                h1, l1 = ax_ret.get_legend_handles_labels()
                if line_ax is not None:
                    h2, l2 = line_ax.get_legend_handles_labels()
                    ax_ret.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=2)
                else:
                    ax_ret.legend(loc="upper left", fontsize=8, ncol=2)

        self.trade_summary_fig.subplots_adjust(top=0.95, bottom=0.15, left=0.07, right=0.97)

        self.heatmap_canvas.draw_idle()
        self.trade_summary_canvas.draw_idle()
        self._schedule_heatmap_scrollregion()
        self.scrollable.schedule_scrollregion_update()

    def _render_tables(self) -> None:
        self._clear_tree(self.monthly_tree)
        self._clear_tree(self.yearly_tree)
        self._clear_tree(self.trades_tree)
        self._clear_tree(self.rebalance_tree)
        self._clear_tree(self.snapshot_tree)
        self._clear_tree(self.snapshot_preview_tree)
        self._clear_tree(self.metrics_tree)
        self._clear_tree(self.funding_compare_tree)
        for tree in getattr(self, "_funding_flow_trees", {}).values():
            self._clear_tree(tree)
        self._set_trade_reason_text("")

        monthly_frames: list[tuple[str, pd.DataFrame]] = []
        if self.compare_result is not None:
            if self.screen_result is not None and not self.screen_result.monthly_returns.empty:
                monthly_frames.append((str(self.screen_result.name or tr("source_screen")), self.screen_result.monthly_returns.copy()))
            if self.strategy_result is not None and not self.strategy_result.monthly_returns.empty:
                monthly_frames.append((str(self.strategy_result.name or tr("source_strategy")), self.strategy_result.monthly_returns.copy()))
        elif self.funding_compare_results:
            mode = self._funding_view_internal()
            label_map = {
                "lump_sum": tr("funding_mode_lump"),
                "dca": tr("funding_mode_dca"),
                "va": tr("funding_mode_va"),
            }
            target = self.funding_compare_results.get(mode) or self.funding_compare_results.get("lump_sum")
            if target is None and self.funding_compare_results:
                target = next(iter(self.funding_compare_results.values()))
            if target is not None and not target.monthly_returns.empty:
                monthly_frames.append((label_map.get(mode, mode), target.monthly_returns.copy()))
        else:
            target = self._target_result_for_tables()
            if target is not None and not target.monthly_returns.empty:
                monthly_frames.append((str(target.name), target.monthly_returns.copy()))

        for source, frame in monthly_frames:
            for _, row in frame.iterrows():
                excess = pd.to_numeric(pd.Series([row.get("excess_ret")]), errors="coerce").iloc[0]
                tags = ("pos",) if pd.notna(excess) and excess > 0 else (("neg",) if pd.notna(excess) and excess < 0 else ())
                self.monthly_tree.insert(
                    "",
                    "end",
                    values=(
                        source,
                        row.get("month", ""),
                        self._fmt_pct(row.get("strategy_ret"), signed=True),
                        self._fmt_pct(row.get("bench_ret"), signed=True),
                        self._fmt_pct(row.get("excess_ret"), signed=True),
                    ),
                    tags=tags,
                )

        if self.compare_result is not None and "yearly_returns" in self.compare_result:
            yearly = self.compare_result["yearly_returns"].copy()
            yearly_cols = [c for c in yearly.columns if str(c) != "year"]
            left_col = yearly_cols[0] if len(yearly_cols) >= 1 else None
            right_col = yearly_cols[1] if len(yearly_cols) >= 2 else None
            for _, row in yearly.iterrows():
                left_val = pd.to_numeric(pd.Series([row.get(left_col)]) if left_col else pd.Series([np.nan]), errors="coerce").iloc[0]
                right_val = pd.to_numeric(pd.Series([row.get(right_col)]) if right_col else pd.Series([np.nan]), errors="coerce").iloc[0]
                tag = "pos" if pd.notna(left_val) and left_val > 0 else ("neg" if pd.notna(left_val) and left_val < 0 else "")
                tags = (tag,) if tag else ()
                self.yearly_tree.insert(
                    "",
                    "end",
                    values=(
                        row.get("year", ""),
                        self._fmt_pct(left_val / 100.0 if pd.notna(left_val) else left_val, signed=True),
                        self._fmt_pct(right_val / 100.0 if pd.notna(right_val) else right_val, signed=True),
                    ),
                    tags=tags,
                )
        else:
            target_yearly = self._target_result_for_tables()
            if target_yearly is not None and not target_yearly.yearly_returns.empty:
                for _, row in target_yearly.yearly_returns.iterrows():
                    val = pd.to_numeric(pd.Series([row.get("return_pct")]), errors="coerce").iloc[0]
                    tag = "pos" if pd.notna(val) and val > 0 else ("neg" if pd.notna(val) and val < 0 else "")
                    tags = (tag,) if tag else ()
                    self.yearly_tree.insert("", "end", values=(row.get("year", ""), self._fmt_pct(val / 100.0 if pd.notna(val) else val, signed=True), ""), tags=tags)

        target = self._target_result_for_tables()
        trades_all = self._prepare_trades(target.trades.copy()) if target is not None and target.trades is not None and not target.trades.empty else pd.DataFrame()
        trades_filtered = self._filter_trades_by_controls(trades_all)

        if not trades_filtered.empty:
            trades_view = trades_filtered.sort_values("exec_date").tail(500)
            for _, row in trades_view.iterrows():
                side_val = str(row.get("side", "")).lower()
                tag = "buy" if side_val == "buy" else "sell"
                self.trades_tree.insert(
                    "",
                    "end",
                    values=(
                        str(pd.to_datetime(row.get("exec_date"), errors="coerce").date()) if pd.notna(row.get("exec_date")) else "",
                        row.get("ticker", ""),
                        self._format_side(row.get("side")),
                        f"{float(row.get('shares', 0.0)):.4f}",
                        f"{float(row.get('exec_price', 0.0)):.4f}",
                        f"{float(row.get('notional', 0.0)):,.0f}",
                        f"{float(row.get('commission', 0.0)):.2f}",
                        f"{float(row.get('slippage', 0.0)):.2f}",
                        row.get("reason_detail", row.get("reason", "")),
                    ),
                    tags=(tag,),
                )

        if target is not None and not target.rebalance_log.empty:
            for _, row in target.rebalance_log.iterrows():
                self.rebalance_tree.insert(
                    "",
                    "end",
                    values=(
                        str(pd.to_datetime(row.get("signal_date"), errors="coerce").date()) if pd.notna(row.get("signal_date")) else "",
                        str(pd.to_datetime(row.get("exec_date"), errors="coerce").date()) if pd.notna(row.get("exec_date")) else "",
                        row.get("mode", ""),
                        row.get("universe_count", ""),
                        row.get("candidate_count", ""),
                        row.get("selected_count", ""),
                        row.get("lookahead_ok", ""),
                        row.get("benchmark", ""),
                    ),
                )

        snap_sources: list[tuple[str, Any]] = []
        if self.screen_result is not None:
            snap_sources.append((tr("source_screen"), self.screen_result))
        if self.strategy_result is not None:
            snap_sources.append((tr("source_strategy"), self.strategy_result))

        for source, result in snap_sources:
            idx = result.rebalance_snapshots_index
            if idx is None or idx.empty:
                continue
            for _, row in idx.iterrows():
                signal_date = pd.to_datetime(row.get("signal_date"), errors="coerce")
                self.snapshot_tree.insert(
                    "",
                    "end",
                    values=(source, str(signal_date.date()) if pd.notna(signal_date) else "", row.get("snapshot_path", "")),
                )

        if self.funding_compare_results and self.compare_result is None:
            mode_order = ["lump_sum", "dca", "va"]
            mode_label = {
                "lump_sum": self._funding_lump_mode_label(),
                "dca": tr("funding_mode_dca"),
                "va": tr("funding_mode_va"),
            }
            rows_for_sort: list[dict[str, Any]] = []
            for mode in mode_order:
                result = self.funding_compare_results.get(mode)
                if result is None:
                    continue
                m = result.metrics or {}
                rows_for_sort.append(
                    {
                        "mode_name": mode_label.get(mode, mode),
                        "sort_ending_value": float(pd.to_numeric(pd.Series([m.get("ending_value")]), errors="coerce").fillna(0.0).iloc[0]),
                        "values": (
                            mode_label.get(mode, mode),
                            self._fmt_pct(m.get("twr_total_return"), signed=True),
                            self._fmt_pct(m.get("twr_cagr"), signed=False),
                            self._fmt_pct(m.get("twr_mdd", m.get("mdd")), signed=True),
                            self._fmt_number(m.get("twr_sharpe", m.get("sharpe")), 2),
                            self._fmt_pct(m.get("twr_vol", m.get("vol")), signed=False),
                            self._fmt_pct(m.get("account_cagr"), signed=False),
                            self._fmt_pct(m.get("account_mdd"), signed=True),
                            self._fmt_number(m.get("account_sharpe"), 2),
                            self._fmt_pct(m.get("account_vol"), signed=False),
                            self._fmt_number(m.get("turnover"), 2),
                            int(pd.to_numeric(pd.Series([m.get("trades")]), errors="coerce").fillna(0).iloc[0]),
                            self._fmt_money(m.get("total_contributed")),
                            self._fmt_money(m.get("ending_value")),
                            self._fmt_money(m.get("pnl")),
                            self._fmt_pct(m.get("mwr_irr"), signed=False),
                        ),
                    }
                )

            rows_for_sort.sort(key=lambda x: float(x.get("sort_ending_value", 0.0)), reverse=True)
            for row in rows_for_sort:
                self.funding_compare_tree.insert("", "end", values=row["values"])

            target_metrics = self._target_result_for_tables()
            if target_metrics is not None:
                for key, value in (target_metrics.metrics or {}).items():
                    self.metrics_tree.insert("", "end", values=(key, value, ""))
        elif self.compare_result is not None and "metrics" in self.compare_result:
            m = self.compare_result["metrics"].copy()
            metric_cols = [c for c in m.columns if str(c) != "metric"]
            left_col = metric_cols[0] if len(metric_cols) >= 1 else None
            right_col = metric_cols[1] if len(metric_cols) >= 2 else None
            for _, row in m.iterrows():
                self.metrics_tree.insert(
                    "",
                    "end",
                    values=(
                        row.get("metric", ""),
                        row.get(left_col, "") if left_col else "",
                        row.get(right_col, "") if right_col else "",
                    ),
                )
        else:
            if self.screen_result is not None:
                for key, value in self.screen_result.metrics.items():
                    self.metrics_tree.insert("", "end", values=(key, value, ""))
            if self.strategy_result is not None:
                existing = {self.metrics_tree.item(i, "values")[0]: i for i in self.metrics_tree.get_children()}
                for key, value in self.strategy_result.metrics.items():
                    if key in existing:
                        iid = existing[key]
                        vals = list(self.metrics_tree.item(iid, "values"))
                        vals[2] = value
                        self.metrics_tree.item(iid, values=vals)
                    else:
                        self.metrics_tree.insert("", "end", values=(key, "", value))

        self._render_monthly_chart(monthly_frames)
        self._render_yearly_chart()
        self._render_ticker_performance()
        self._render_funding_flows_tables()

    def _render_monthly_chart(self, monthly_frames: list) -> None:
        self.monthly_chart_ax.clear()
        if not monthly_frames:
            self.monthly_chart_ax.text(0.5, 0.5, tr("no_trade_data"), ha="center", va="center", transform=self.monthly_chart_ax.transAxes)
            self.monthly_chart_ax.set_axis_off()
            self.monthly_chart_canvas.draw_idle()
            return
        _, frame = monthly_frames[-1]
        mdf = frame.copy()
        mdf["month_dt"] = pd.to_datetime(mdf.get("month", ""), errors="coerce")
        mdf["strategy_ret"] = pd.to_numeric(mdf.get("strategy_ret"), errors="coerce")
        mdf = mdf.dropna(subset=["month_dt", "strategy_ret"]).sort_values("month_dt")
        if mdf.empty:
            self.monthly_chart_ax.set_axis_off()
            self.monthly_chart_canvas.draw_idle()
            return
        x = mdates.date2num(mdf["month_dt"].dt.to_pydatetime())
        vals = mdf["strategy_ret"].to_numpy(dtype=float) * 100.0
        colors = ["#0b7a3a" if v >= 0 else "#b42318" for v in vals]
        self.monthly_chart_ax.bar(x, vals, width=20, color=colors, alpha=0.85)
        self.monthly_chart_ax.axhline(0, color="#4b5563", linewidth=0.8)
        self.monthly_chart_ax.set_ylabel("%", fontsize=7)
        self.monthly_chart_ax.set_title("월별 전략 수익률", fontsize=9)
        self.monthly_chart_ax.grid(True, axis="y", alpha=0.2)
        if len(x) > 0:
            self.monthly_chart_ax.set_xlim(float(x.min() - 20), float(x.max() + 20))
        self.monthly_chart_ax.xaxis.set_major_locator(mdates.YearLocator())
        self.monthly_chart_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        self.monthly_chart_fig.tight_layout()
        self.monthly_chart_canvas.draw_idle()

    def _render_yearly_chart(self) -> None:
        self.yearly_chart_ax.clear()
        target = self._target_result_for_tables()
        if target is None or target.yearly_returns is None or target.yearly_returns.empty:
            self.yearly_chart_ax.set_axis_off()
            self.yearly_chart_canvas.draw_idle()
            return
        yr = target.yearly_returns.copy()
        if "return_pct" not in yr.columns:
            self.yearly_chart_ax.set_axis_off()
            self.yearly_chart_canvas.draw_idle()
            return
        yr["year"] = pd.to_numeric(yr.get("year"), errors="coerce")
        yr["return_pct"] = pd.to_numeric(yr.get("return_pct"), errors="coerce")
        yr = yr.dropna(subset=["year", "return_pct"]).sort_values("year")
        if yr.empty:
            self.yearly_chart_ax.set_axis_off()
            self.yearly_chart_canvas.draw_idle()
            return
        x = yr["year"].astype(int).tolist()
        vals = yr["return_pct"].tolist()
        colors = ["#0b7a3a" if v >= 0 else "#b42318" for v in vals]
        self.yearly_chart_ax.bar(x, vals, color=colors, alpha=0.85, width=0.6)
        self.yearly_chart_ax.axhline(0, color="#4b5563", linewidth=0.8)
        self.yearly_chart_ax.set_xticks(x)
        self.yearly_chart_ax.set_xticklabels([str(y) for y in x], fontsize=8, rotation=0)
        self.yearly_chart_ax.set_ylabel("%", fontsize=7)
        self.yearly_chart_ax.set_title("연도별 전략 수익률", fontsize=9)
        self.yearly_chart_ax.grid(True, axis="y", alpha=0.2)
        self.yearly_chart_fig.tight_layout()
        self.yearly_chart_canvas.draw_idle()
        self.yearly_chart_canvas.get_tk_widget().after(80, self.yearly_chart_canvas.draw_idle)

    def _on_tables_tab_change(self, event=None) -> None:
        """Force canvas redraw when a tab becomes visible (fixes zero-size render bug)."""
        try:
            selected = self.tables_notebook.select()
            if not selected:
                return
            tab_text = self.tables_notebook.tab(selected, "text")
            if tab_text == tr("yearly_returns_tab"):
                self.yearly_chart_canvas.draw_idle()
            elif tab_text == tr("monthly_returns_tab"):
                self.monthly_chart_canvas.draw_idle()
            elif tab_text == tr("ticker_perf_tab"):
                self.ticker_perf_canvas.draw_idle()
        except Exception:
            pass

    def _render_ticker_performance(self) -> None:
        self.ticker_perf_ax.clear()
        if hasattr(self, "ticker_perf_tree"):
            self._clear_tree(self.ticker_perf_tree)

        target = self._target_result_for_tables()
        if target is None or target.trades is None or target.trades.empty:
            self.ticker_perf_ax.text(0.5, 0.5, tr("ticker_no_data"), ha="center", va="center", transform=self.ticker_perf_ax.transAxes)
            self.ticker_perf_ax.set_axis_off()
            self.ticker_perf_canvas.draw_idle()
            return

        trades = self._prepare_trades(target.trades.copy())
        if trades.empty:
            self.ticker_perf_ax.set_axis_off()
            self.ticker_perf_canvas.draw_idle()
            return

        # Compute per-ticker P&L
        buy_cost = trades[trades["side"] == "buy"].groupby("ticker")["notional"].sum()
        sell_rev = trades[trades["side"] == "sell"].groupby("ticker")["notional"].sum()
        all_tickers = sorted(set(buy_cost.index) | set(sell_rev.index))

        rows = []
        for ticker in all_tickers:
            cost = float(buy_cost.get(ticker, 0.0))
            rev = float(sell_rev.get(ticker, 0.0))
            pnl = rev - cost
            ret_pct = (pnl / cost * 100.0) if cost > 0 else 0.0
            rows.append({"ticker": ticker, "cost": cost, "pnl": pnl, "ret_pct": ret_pct})

        if not rows:
            self.ticker_perf_ax.set_axis_off()
            self.ticker_perf_canvas.draw_idle()
            return

        df_perf = pd.DataFrame(rows).sort_values("ret_pct", ascending=False)
        n_show = min(20, len(df_perf))
        top_n = df_perf.head(n_show // 2 + (n_show % 2))
        bottom_n = df_perf.tail(n_show - len(top_n)).sort_values("ret_pct", ascending=False)
        shown = pd.concat([top_n, bottom_n]).drop_duplicates(subset=["ticker"])
        shown = shown.sort_values("ret_pct", ascending=False)

        tickers = shown["ticker"].tolist()
        vals = shown["ret_pct"].tolist()
        colors = ["#0b7a3a" if v >= 0 else "#b42318" for v in vals]

        x = range(len(tickers))
        self.ticker_perf_ax.bar(x, vals, color=colors, alpha=0.85)
        self.ticker_perf_ax.set_xticks(list(x))
        self.ticker_perf_ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=7)
        self.ticker_perf_ax.axhline(0, color="#4b5563", linewidth=0.8)
        self.ticker_perf_ax.set_ylabel("%", fontsize=7)
        self.ticker_perf_ax.set_title(f"{tr('ticker_top_label')} / {tr('ticker_bottom_label')} 수익률", fontsize=9)
        self.ticker_perf_ax.grid(True, axis="y", alpha=0.2)
        self.ticker_perf_fig.tight_layout()
        self.ticker_perf_canvas.draw_idle()

        # Fill table
        for _, row in df_perf.iterrows():
            tag = "pos" if row["ret_pct"] >= 0 else "neg"
            self.ticker_perf_tree.insert(
                "", "end",
                values=(
                    row["ticker"],
                    f"{row['cost']:,.0f}",
                    f"{row['pnl']:+,.0f}",
                    f"{row['ret_pct']:+.2f}%",
                ),
                tags=(tag,),
            )

    def _funding_mode_label_map(self) -> dict[str, str]:
        labels = _default_funding_mode_labels()
        labels["lump_sum"] = self._funding_lump_mode_label()
        return labels

    def _sorted_funding_flows(self, flows: pd.DataFrame) -> pd.DataFrame:
        if flows is None or flows.empty:
            return pd.DataFrame()
        ff = flows.copy()
        ff["date"] = pd.to_datetime(ff.get("date"), errors="coerce")
        ff = ff.dropna(subset=["date"])
        step_idx = pd.to_numeric(ff.get("step_index"), errors="coerce").fillna(0).astype(int)
        ff["step_index"] = step_idx
        ff = ff.sort_values(["date", "step_index"]).reset_index(drop=True)
        return ff

    def _apply_funding_flow_display_mode(self) -> None:
        detail_mode = self.funding_flow_view_var.get() == tr("funding_flow_view_detail")
        for mode_key, tree in self._funding_flow_trees.items():
            if detail_mode:
                cols = list(FUNDING_FLOW_COLUMNS)
                if mode_key != "all":
                    cols = [c for c in cols if c != "mode_label"]
            else:
                cols = list(FUNDING_FLOW_SUMMARY_COLUMNS)
                if mode_key != "all":
                    cols = [c for c in cols if c != "mode_label"]
            tree.configure(displaycolumns=cols)

    def _insert_funding_flow_row(
        self,
        tree: ttk.Treeview,
        row: pd.Series,
        mode_key: str | None = None,
        mode_label: str | None = None,
    ) -> None:
        date_txt = str(pd.Timestamp(row.get("date")).date()) if pd.notna(row.get("date")) else ""
        step_idx = int(pd.to_numeric(pd.Series([row.get("step_index")]), errors="coerce").fillna(0).iloc[0])
        raw_val = pd.to_numeric(pd.Series([row.get("raw_contribution")]), errors="coerce").iloc[0]
        c_val = pd.to_numeric(pd.Series([row.get("contribution")]), errors="coerce").fillna(0.0).iloc[0]
        cum_before = pd.to_numeric(pd.Series([row.get("cumulative_contributed_before")]), errors="coerce").iloc[0]
        cum_after = pd.to_numeric(pd.Series([row.get("cumulative_contributed_after")]), errors="coerce").iloc[0]
        min_val = pd.to_numeric(pd.Series([row.get("va_min_contribution")]), errors="coerce").iloc[0]
        max_val = pd.to_numeric(pd.Series([row.get("va_max_contribution")]), errors="coerce").iloc[0]
        cash_before = pd.to_numeric(pd.Series([row.get("cash_before_funding")]), errors="coerce").iloc[0]
        cash_after = pd.to_numeric(pd.Series([row.get("cash_after_funding")]), errors="coerce").iloc[0]
        pos_before = pd.to_numeric(pd.Series([row.get("positions_value_before_funding")]), errors="coerce").iloc[0]
        eq_before = pd.to_numeric(pd.Series([row.get("account_equity_before_funding")]), errors="coerce").iloc[0]
        pos_after = pd.to_numeric(pd.Series([row.get("positions_value_after_funding")]), errors="coerce").iloc[0]
        eq_after = pd.to_numeric(pd.Series([row.get("account_equity_after_funding")]), errors="coerce").iloc[0]
        cash_after_reb = pd.to_numeric(pd.Series([row.get("cash_after_rebalance")]), errors="coerce").iloc[0]
        pos_after_reb = pd.to_numeric(pd.Series([row.get("positions_value_after_rebalance")]), errors="coerce").iloc[0]
        eq_after_reb = pd.to_numeric(pd.Series([row.get("account_equity_after_rebalance")]), errors="coerce").iloc[0]
        invest_after = pd.to_numeric(pd.Series([row.get("invested_ratio_after_funding")]), errors="coerce").iloc[0]
        invest_after_reb = pd.to_numeric(pd.Series([row.get("invested_ratio_after_rebalance")]), errors="coerce").iloc[0]
        floor_applied = bool(row.get("floor_applied", row.get("cap_applied_min", False)))
        max_applied = bool(row.get("max_cap_applied", row.get("cap_applied_max", False)))
        same_day = bool(row.get("is_rebalance_same_day", False))
        had_rebalance_same_day = bool(row.get("had_rebalance_same_day", same_day))
        mode_name = mode_label or str(row.get("mode_label", mode_key or ""))
        row_map: dict[str, Any] = {
            "mode_label": mode_name,
            "date": date_txt,
            "step_index": step_idx,
            "funding_mode": row.get("funding_mode", ""),
            "contribution_freq": row.get("contribution_freq", ""),
            "contribution": self._fmt_money(c_val),
            "raw_contribution": self._fmt_money(raw_val),
            "reason": row.get("reason", ""),
            "cumulative_contributed_before": self._fmt_money(cum_before),
            "cumulative_contributed_after": self._fmt_money(cum_after),
            "cash_before_funding": self._fmt_money(cash_before),
            "positions_value_before_funding": self._fmt_money(pos_before),
            "account_equity_before_funding": self._fmt_money(eq_before),
            "cash_after_funding": self._fmt_money(cash_after),
            "positions_value_after_funding": self._fmt_money(pos_after),
            "account_equity_after_funding": self._fmt_money(eq_after),
            "cash_after_rebalance": self._fmt_money(cash_after_reb) if pd.notna(cash_after_reb) else "",
            "positions_value_after_rebalance": self._fmt_money(pos_after_reb) if pd.notna(pos_after_reb) else "",
            "account_equity_after_rebalance": self._fmt_money(eq_after_reb) if pd.notna(eq_after_reb) else "",
            "had_rebalance_same_day": "Y" if had_rebalance_same_day else "",
            "is_rebalance_same_day": "Y" if same_day else "",
            "event_order_tag": row.get("event_order_tag", ""),
            "rebalance_exec_timing": row.get("rebalance_exec_timing", ""),
            "invested_ratio_after_funding": self._fmt_pct(invest_after, signed=False),
            "invested_ratio_after_rebalance": self._fmt_pct(invest_after_reb, signed=False) if pd.notna(invest_after_reb) else "",
            "va_min_policy": row.get("va_min_policy", ""),
            "va_min_contribution": self._fmt_money(min_val) if pd.notna(min_val) else "",
            "va_max_contribution": self._fmt_money(max_val) if pd.notna(max_val) else "",
            "floor_applied": "Y" if floor_applied else "",
            "max_cap_applied": "Y" if max_applied else "",
        }
        columns = list(tree["columns"])
        values = tuple(row_map.get(col, "") for col in columns)

        tags = ()
        if mode_key and tree is self.funding_flows_all_tree:
            tags = (mode_key,)
        tree.insert("", "end", values=values, tags=tags)

    def _render_funding_flows_tables(self) -> None:
        labels = self._funding_mode_label_map()
        self._apply_funding_flow_display_mode()

        if self.funding_compare_results and self.compare_result is None:
            combined = build_combined_funding_flows(self.funding_compare_results, mode_label_map=labels)
            for _, row in combined.iterrows():
                mode_key = str(row.get("mode_key", ""))
                self._insert_funding_flow_row(
                    self.funding_flows_all_tree,
                    row,
                    mode_key=mode_key,
                    mode_label=labels.get(mode_key, mode_key),
                )

            for mode_key, tree in self._funding_flow_trees.items():
                if mode_key == "all":
                    continue
                result = self.funding_compare_results.get(mode_key)
                if result is None or result.funding_flows is None or result.funding_flows.empty:
                    continue
                mode_df = self._sorted_funding_flows(result.funding_flows)
                for _, row in mode_df.iterrows():
                    self._insert_funding_flow_row(tree, row, mode_key=mode_key, mode_label=labels.get(mode_key, mode_key))
            return

        target = self._target_result_for_tables()
        if target is None or target.funding_flows is None or target.funding_flows.empty:
            return
        target_mode = str(getattr(target, "name", "") or "").strip().lower()
        if target_mode not in {"lump_sum", "dca", "va"}:
            target_mode = str(target.funding_flows.get("funding_mode", pd.Series([""])).astype(str).str.lower().iloc[0] or "").strip()
        if target_mode not in {"lump_sum", "dca", "va"}:
            target_mode = "all"
        ff = self._sorted_funding_flows(target.funding_flows)
        for _, row in ff.iterrows():
            self._insert_funding_flow_row(
                self.funding_flows_all_tree,
                row,
                mode_key=target_mode if target_mode != "all" else None,
                mode_label=labels.get(target_mode, target_mode),
            )
            if target_mode in {"lump_sum", "dca", "va"}:
                self._insert_funding_flow_row(
                    self._funding_flow_trees[target_mode],
                    row,
                    mode_key=target_mode,
                    mode_label=labels.get(target_mode, target_mode),
                )

    def _on_snapshot_select(self, _event=None) -> None:
        selected = self.snapshot_tree.selection()
        if not selected:
            return
        values = self.snapshot_tree.item(selected[0], "values")
        if not values or len(values) < 3:
            return

        path = Path(str(values[2]))
        if not path.exists():
            return

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror(tr("snapshot_preview_error"), str(exc))
            return

        cols = ["ticker", "filter_pass", "rank_score", "rank", "selected", "target_weight", "lookahead_ok"]
        for col in cols:
            if col not in df.columns:
                df[col] = ""
        df = df[cols].head(250)

        self._clear_tree(self.snapshot_preview_tree)
        for _, row in df.iterrows():
            self.snapshot_preview_tree.insert("", "end", values=tuple(row.get(c, "") for c in cols))


def mount_backtest_tab(parent: ttk.Frame) -> BacktestTab:
    return BacktestTab(parent)
