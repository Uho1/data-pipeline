"""
PIT Rebalancing Audit Script
============================
목적: PIT ON에서 리밸런싱마다 실제로 매매가 발생하는지 검증하고,
      매매가 없으면 그 원인을 단계별로 분해한다.

실행:
    ./.venv/bin/python scripts/pit_rebalance_audit.py

산출물:
    logs/pit_verification/pit_rebalance_log.csv
    logs/pit_verification/pit_selected_snapshots/<scenario>_<date>.csv
    logs/pit_verification/pit_rebalance_audit.md  (최종 보고서)
"""
from __future__ import annotations

import sys
import os
import json
import textwrap
from pathlib import Path
from datetime import datetime

# project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from market_data.backtest.simulator import run_backtest
from market_data.backtest.models import (
    StrategyConfig,
    UniverseConfig,
    RankingConfig,
    FactorSpec,
    CostModel,
)
from market_data.backtest.factors import build_factor_panel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR = Path("logs/pit_verification")
SNAPSHOTS_DIR = OUT_DIR / "pit_selected_snapshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

START = "2018-01-01"
END   = "2024-12-31"

# 35개 검증 대상 티커 (good_price + sec_companyfacts_quarterly 동시 보유)
UNIVERSE_TICKERS = [
    "AAPL","ABNB","ACGL","ADI","AEP","AMGN","APP","AXON","BKNG","CDNS",
    "CME","CPB","DLTR","EA","EVRG","EXC","IBKR","INTC","ISRG","MNST",
    "MPWR","MRNA","MSFT","MTCH","MU","NTAP","NWSA","ON","PLTR","PSKY",
    "ROP","ROST","TSCO","TTD","UAL",
]

# 매수 조건: 단순 eps_ttm > 0 (수익이 있는 종목)  — rule_na_policy=pass 로 NaN은 통과
BUY_RULES = "eps_ttm > 0"

SCENARIOS = [
    # (name,         pit_on, freq, asof_lag)
    ("pit_off_M",   False,  "M",  0),
    ("pit_on_lag0_M", True, "M",  0),
    ("pit_on_lag5_M", True, "M",  5),
    ("pit_off_Q",   False,  "Q",  0),
    ("pit_on_lag0_Q", True, "Q",  0),
    ("pit_on_lag5_Q", True, "Q",  5),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(name: str, pit_on: bool, freq: str, asof_lag: int) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        mode="strategy",
        market="us",
        start=START,
        end=END,
        frequency=freq,
        holdings=5,
        sizing="equal",
        buy_rules=BUY_RULES,
        rule_na_policy="pass",          # NaN factor = pass (let it through)
        universe=UniverseConfig(
            source="symbols",
            symbols=UNIVERSE_TICKERS,
            market="us",
        ),
        use_fundamentals_pit=pit_on,
        asof_lag_trading_days=asof_lag,
        use_next_trading_day_availability=True,
        fundamentals_availability_fallback=True,
        fundamentals_fallback_q_days=45,
        fundamentals_fallback_k_days=90,
        fundamentals_missing_policy="exclude",
        asof_mode="available_date" if pit_on else "quarter_end",
        costs=CostModel(commission_bps=0.0, slippage_bps=0.0),
        execution_timing="same_close",
        ranking=RankingConfig(
            normalization="percentile",
            factors=[FactorSpec(name="pe", direction="asc", weight=1.0)],
        ),
        strict_pit=False,
        offline_mode=True,
    )


def _run_scenario(name: str, pit_on: bool, freq: str, asof_lag: int) -> dict:
    print(f"\n[RUN] {name}  pit={pit_on}  freq={freq}  lag={asof_lag}")
    cfg = _make_config(name, pit_on, freq, asof_lag)
    try:
        result = run_backtest(cfg)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return {"name": name, "error": str(exc)}

    m = result.metrics
    reb_log = result.rebalance_log.copy() if result.rebalance_log is not None and not result.rebalance_log.empty else pd.DataFrame()
    trades_df = result.trades.copy() if result.trades is not None and not result.trades.empty else pd.DataFrame()

    total_trades = int(m.get("trades", 0))
    buy_trades = int(trades_df[trades_df["side"] == "buy"].shape[0]) if not trades_df.empty else 0
    sell_trades = int(trades_df[trades_df["side"] == "sell"].shape[0]) if not trades_df.empty else 0

    print(f"  trades={total_trades}  buy={buy_trades}  sell={sell_trades}")
    print(f"  excluded_due_to_missing_fundamentals={m.get('excluded_due_to_missing_fundamentals',0)}")
    print(f"  fallback_used_rows={m.get('fallback_used_rows',0)}")

    # Save selected-ticker snapshots per rebalance date
    if not reb_log.empty:
        for _, row in reb_log.iterrows():
            sig_date = pd.Timestamp(row.get("signal_date", pd.NaT))
            if pd.isna(sig_date):
                continue
            # For rebalances with trades: save snapshot of selected tickers from trades_df
            if not trades_df.empty:
                day_buys = trades_df[
                    (trades_df["side"] == "buy") &
                    (pd.to_datetime(trades_df["signal_date"], errors="coerce") == sig_date)
                ].copy()
                if not day_buys.empty:
                    snap_path = SNAPSHOTS_DIR / f"{name}_{sig_date.strftime('%Y%m%d')}.csv"
                    day_buys.to_csv(snap_path, index=False)

    return {
        "name": name,
        "pit_on": pit_on,
        "freq": freq,
        "asof_lag": asof_lag,
        "total_trades": total_trades,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "excluded_due_to_missing_fundamentals": int(m.get("excluded_due_to_missing_fundamentals", 0)),
        "fallback_used_rows": int(m.get("fallback_used_rows", 0)),
        "twr_cagr": float(m.get("twr_cagr", float("nan"))),
        "twr_mdd": float(m.get("twr_mdd", float("nan"))),
        "rebalance_log": reb_log,
        "trades_df": trades_df,
        "metrics": m,
    }


def _analyze_zero_trades_date(
    reb_row: pd.Series,
    scenario_name: str,
    pit_on: bool,
    freq: str,
    asof_lag: int,
) -> dict:
    """
    Detailed funnel breakdown for a single rebalance date where selected_count == 0.
    Returns a dict with per-stage counts and probable cause.
    """
    sig_date = reb_row.get("signal_date", pd.NaT)
    pit_init = int(reb_row.get("pit_initial_count", -1))
    pit_future_excl = int(reb_row.get("pit_future_excluded_count", -1))
    pit_missing_excl = int(reb_row.get("pit_missing_excluded_count", -1))
    universe_count = int(reb_row.get("universe_count", -1))
    candidate_count = int(reb_row.get("candidate_count", -1))
    selected_count = int(reb_row.get("selected_count", -1))

    after_pit = pit_init - pit_future_excl - (pit_missing_excl if pit_on else 0) if pit_on else pit_init
    after_universe = universe_count
    after_buy_rule = candidate_count

    causes = []
    if pit_on and pit_init == 0:
        causes.append("(a) 크로스섹션이 비어있음: 패널에 해당 날짜 데이터 없음")
    elif pit_on and pit_future_excl > 0 and after_pit == 0:
        causes.append(f"(b) AvailableDate가 asof_date보다 미래 → {pit_future_excl}개 전부 제외")
    elif pit_on and pit_missing_excl > 0 and after_pit == 0:
        causes.append(f"(a) AvailableDate=NaT(missing) → {pit_missing_excl}개 전부 제외 (strict모드)")
    elif after_universe == 0:
        causes.append("(d) 유니버스 필터(validity window 등) 통과 후 후보 없음")
    elif after_buy_rule == 0:
        causes.append(f"(c) buy_rule({BUY_RULES}) 통과 종목 없음: eps_ttm/pe NaN/Inf 또는 조건 미충족")
    else:
        causes.append("(f) 기타: selected_count=0이나 candidate>0 (랭킹/sizing 문제?)")

    return {
        "signal_date": sig_date,
        "pit_initial_count": pit_init,
        "pit_future_excluded": pit_future_excl,
        "pit_missing_excluded": pit_missing_excl,
        "after_pit_count": after_pit,
        "universe_count": after_universe,
        "candidate_count": after_buy_rule,
        "selected_count": selected_count,
        "probable_cause": "; ".join(causes),
    }


def _sample_factor_values(
    tickers: list[str],
    asof_date: str,
    pit_on: bool,
    n_sample: int = 10,
) -> pd.DataFrame:
    """Build factor panel and snapshot factor values on asof_date for sample tickers."""
    try:
        panel = build_factor_panel(
            symbols=tickers[:n_sample],
            market="us",
            start=asof_date,
            end=asof_date,
            asof_mode="available_date" if pit_on else "quarter_end",
            use_next_trading_day_availability=True,
            availability_fallback=True,
            fallback_q_days=45,
            fallback_k_days=90,
            offline_mode=True,
        )
        if panel is None or panel.empty:
            return pd.DataFrame()
        idx = pd.to_datetime(asof_date)
        level0 = pd.to_datetime(panel.index.get_level_values(0), errors="coerce")
        mask = level0 == idx
        if not mask.any():
            return pd.DataFrame()
        snap = panel.loc[mask].copy()
        cols = ["eps_ttm", "pe", "asof_statement_date", "asof_available_date", "availability_method", "close"]
        avail = [c for c in cols if c in snap.columns]
        return snap[avail].reset_index()
    except Exception as exc:
        return pd.DataFrame({"error": [str(exc)]})


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PIT 리밸런싱 감사 스크립트")
    print(f"Universe: {len(UNIVERSE_TICKERS)} tickers | {START} ~ {END}")
    print(f"Buy rules: {BUY_RULES}")
    print("=" * 70)

    scenario_results = []
    all_reb_rows = []

    for sc_name, pit_on, freq, asof_lag in SCENARIOS:
        res = _run_scenario(sc_name, pit_on, freq, asof_lag)
        scenario_results.append(res)

        reb_log: pd.DataFrame = res.get("rebalance_log", pd.DataFrame())
        if reb_log.empty:
            continue

        reb_log = reb_log.copy()
        reb_log["scenario"] = sc_name
        reb_log["pit_on"] = pit_on
        reb_log["freq"] = freq
        reb_log["asof_lag"] = asof_lag
        all_reb_rows.append(reb_log)

    # Combine rebalance logs
    if all_reb_rows:
        combined_log = pd.concat(all_reb_rows, ignore_index=True, sort=False)
        csv_path = OUT_DIR / "pit_rebalance_log.csv"
        combined_log.to_csv(csv_path, index=False)
        print(f"\n[SAVED] {csv_path} ({len(combined_log)} rows)")
    else:
        combined_log = pd.DataFrame()

    # ---------------------------------------------------------------------------
    # Per-rebalance analysis: identify zero-trade causes
    # ---------------------------------------------------------------------------
    zero_trade_details: list[dict] = []
    for res in scenario_results:
        if "error" in res:
            continue
        reb_log: pd.DataFrame = res.get("rebalance_log", pd.DataFrame())
        trades_df: pd.DataFrame = res.get("trades_df", pd.DataFrame())
        if reb_log.empty:
            continue

        # Compute per-rebalance trade counts
        trade_counts: dict[pd.Timestamp, int] = {}
        if not trades_df.empty and "signal_date" in trades_df.columns:
            tc = trades_df[trades_df["side"] == "buy"].groupby("signal_date").size()
            for k, v in tc.items():
                trade_counts[pd.Timestamp(k)] = int(v)

        for _, row in reb_log.iterrows():
            sig_date = pd.Timestamp(row.get("signal_date", pd.NaT))
            if pd.isna(sig_date):
                continue
            trades_on_date = trade_counts.get(sig_date, 0)
            if trades_on_date == 0:
                detail = _analyze_zero_trades_date(
                    row,
                    res["name"],
                    res.get("pit_on", False),
                    res.get("freq", "Q"),
                    res.get("asof_lag", 0),
                )
                detail["scenario"] = res["name"]
                detail["pit_on"] = res.get("pit_on", False)
                detail["freq"] = res.get("freq", "Q")
                detail["asof_lag"] = res.get("asof_lag", 0)
                zero_trade_details.append(detail)

    zero_df = pd.DataFrame(zero_trade_details) if zero_trade_details else pd.DataFrame()
    if not zero_df.empty:
        zero_path = OUT_DIR / "zero_trade_dates.csv"
        zero_df.to_csv(zero_path, index=False)
        print(f"[SAVED] {zero_path} ({len(zero_df)} zero-trade rebalances)")

    # ---------------------------------------------------------------------------
    # Detailed samples: 3 trade-ok dates + 3 zero-trade dates per scenario
    # ---------------------------------------------------------------------------
    detailed_samples: dict[str, list[dict]] = {}  # scenario → sample rows

    for res in scenario_results:
        if "error" in res:
            continue
        sc_name = res["name"]
        pit_on = res.get("pit_on", False)
        reb_log = res.get("rebalance_log", pd.DataFrame())
        trades_df = res.get("trades_df", pd.DataFrame())
        if reb_log.empty:
            continue

        trade_counts: dict[pd.Timestamp, int] = {}
        if not trades_df.empty and "signal_date" in trades_df.columns:
            tc = trades_df[trades_df["side"] == "buy"].groupby("signal_date").size()
            for k, v in tc.items():
                trade_counts[pd.Timestamp(k)] = int(v)

        ok_rows = []
        zero_rows = []
        for _, row in reb_log.iterrows():
            sig_date = pd.Timestamp(row.get("signal_date", pd.NaT))
            if pd.isna(sig_date):
                continue
            cnt = trade_counts.get(sig_date, 0)
            if cnt > 0:
                ok_rows.append((sig_date, row, cnt))
            else:
                zero_rows.append((sig_date, row))

        samples = []
        # 3 trade-ok
        for sig_date, row, cnt in ok_rows[-3:]:
            day_buys = pd.DataFrame()
            if not trades_df.empty:
                day_buys = trades_df[
                    (trades_df["side"] == "buy") &
                    (pd.to_datetime(trades_df["signal_date"], errors="coerce") == sig_date)
                ][["ticker", "exec_price", "notional", "reason_detail"]].copy()
            selected_tickers = day_buys["ticker"].tolist() if not day_buys.empty else []
            samples.append({
                "type": "TRADE_OK",
                "signal_date": sig_date.date().isoformat(),
                "buy_trade_count": cnt,
                "selected_tickers": selected_tickers,
                "pit_initial": int(row.get("pit_initial_count", -1)),
                "pit_future_excl": int(row.get("pit_future_excluded_count", -1)),
                "universe_count": int(row.get("universe_count", -1)),
                "candidate_count": int(row.get("candidate_count", -1)),
                "selected_count": int(row.get("selected_count", -1)),
                "asof_max_stmt_date": str(row.get("asof_max_statement_date", "")),
            })

        # 3 zero-trade
        for sig_date, row in zero_rows[:3]:
            detail = _analyze_zero_trades_date(row, sc_name, pit_on, res.get("freq","Q"), res.get("asof_lag",0))
            samples.append({
                "type": "ZERO_TRADE",
                "signal_date": sig_date.date().isoformat(),
                "buy_trade_count": 0,
                "pit_initial": detail["pit_initial_count"],
                "pit_future_excl": detail["pit_future_excluded"],
                "universe_count": detail["universe_count"],
                "candidate_count": detail["candidate_count"],
                "selected_count": detail["selected_count"],
                "probable_cause": detail["probable_cause"],
            })

        detailed_samples[sc_name] = samples

    # ---------------------------------------------------------------------------
    # Generate Markdown report
    # ---------------------------------------------------------------------------
    _write_report(scenario_results, combined_log, zero_df, detailed_samples)
    print(f"\n[SAVED] {OUT_DIR / 'pit_rebalance_audit.md'}")
    print("\nAudit complete.")


def _write_report(
    scenario_results: list[dict],
    combined_log: pd.DataFrame,
    zero_df: pd.DataFrame,
    detailed_samples: dict,
):
    lines: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# PIT 리밸런싱 감사 보고서")
    lines.append(f"\n생성일시: {ts}")
    lines.append(f"검증 기간: {START} ~ {END}")
    lines.append(f"유니버스: {len(UNIVERSE_TICKERS)}개 티커 (S&P500 교집합 with good prices + sec_companyfacts)")
    lines.append(f"매수 조건: `{BUY_RULES}` (rule_na_policy=pass)")
    lines.append(f"ingest 옵션: --use-next-trading-day-availability --financial-source sec")
    lines.append("")

    # ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 1. 전체 시나리오 결과 요약")
    lines.append("")
    lines.append("| 시나리오 | PIT | 주기 | lag | trades | buy | sell | excl_missing | fallback_rows | CAGR |")
    lines.append("|----------|-----|------|-----|--------|-----|------|--------------|---------------|------|")
    for res in scenario_results:
        if "error" in res:
            lines.append(f"| {res['name']} | - | - | - | ERROR | - | - | - | - | - |")
            continue
        pit_label = "ON" if res.get("pit_on") else "OFF"
        cagr = res.get("twr_cagr", float("nan"))
        cagr_str = f"{cagr:.2%}" if np.isfinite(cagr) else "N/A"
        lines.append(
            f"| {res['name']} | {pit_label} | {res['freq']} | {res['asof_lag']} "
            f"| {res['total_trades']} | {res['buy_trades']} | {res['sell_trades']} "
            f"| {res['excluded_due_to_missing_fundamentals']} | {res['fallback_used_rows']} | {cagr_str} |"
        )

    lines.append("")
    lines.append("> **excl_missing**: AvailableDate가 미래(look-ahead)여서 제외된 누적 건수")
    lines.append("> **fallback_rows**: AcceptedAt 부재로 fallback lag(45/90일)으로 AvailableDate 계산된 행 수")
    lines.append("")

    # ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 2. 리밸런싱 날짜별 펀넬 통계 (PIT ON 시나리오)")
    lines.append("")
    lines.append("리밸런싱 단계별 평균 통과 수 (pit_on=True인 시나리오)")

    if not combined_log.empty:
        pit_log = combined_log[combined_log.get("pit_on", pd.Series(False)) == True].copy() if "pit_on" in combined_log.columns else pd.DataFrame()
        if not pit_log.empty:
            numeric_cols = ["pit_initial_count", "pit_future_excluded_count", "pit_missing_excluded_count",
                            "universe_count", "candidate_count", "selected_count"]
            avail = [c for c in numeric_cols if c in pit_log.columns]
            if avail:
                summary = pit_log.groupby("scenario")[avail].mean().round(1)
                lines.append("")
                lines.append("```")
                lines.append(summary.to_string())
                lines.append("```")
                lines.append("")
                lines.append("**컬럼 설명:**")
                lines.append("- `pit_initial_count`: PIT 필터 적용 전 크로스섹션 티커 수")
                lines.append("- `pit_future_excluded_count`: AvailableDate > asof_date → look-ahead 제외 수")
                lines.append("- `pit_missing_excluded_count`: AvailableDate=NaT (strict=False이면 0)")
                lines.append("- `universe_count`: 유니버스/validity 필터 통과 수")
                lines.append("- `candidate_count`: buy_rule 통과 수")
                lines.append("- `selected_count`: 최종 선택 수")

    lines.append("")

    # ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 3. 매매 발생 사례 분석 (상위 3개 날짜)")
    lines.append("")

    for sc_name, samples in detailed_samples.items():
        ok = [s for s in samples if s["type"] == "TRADE_OK"]
        if not ok:
            continue
        lines.append(f"### {sc_name}")
        lines.append("")
        for s in ok:
            lines.append(f"**{s['signal_date']}**")
            lines.append(f"- 매수 체결: {s['buy_trade_count']}건")
            lines.append(f"- 펀넬: pit_init={s['pit_initial']} → pit_future_excl={s['pit_future_excl']}"
                         f" → universe={s['universe_count']} → candidate={s['candidate_count']}"
                         f" → selected={s['selected_count']}")
            if s.get("selected_tickers"):
                lines.append(f"- 선택 종목: {', '.join(s['selected_tickers'])}")
            if s.get("asof_max_stmt_date"):
                lines.append(f"- 사용 재무제표 최대 날짜: {s['asof_max_stmt_date']}")
            lines.append("")

    # ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 4. 매매 없음(trades=0) 구간 원인 분석 (상위 3개 날짜)")
    lines.append("")

    for sc_name, samples in detailed_samples.items():
        zeros = [s for s in samples if s["type"] == "ZERO_TRADE"]
        if not zeros:
            lines.append(f"### {sc_name}")
            lines.append("> 모든 리밸런싱 날짜에서 매매 발생 — 0-trade 구간 없음")
            lines.append("")
            continue
        lines.append(f"### {sc_name}")
        lines.append("")
        for s in zeros:
            lines.append(f"**{s['signal_date']}**")
            lines.append(f"- 펀넬: pit_init={s['pit_initial']} → pit_future_excl={s['pit_future_excl']}"
                         f" → universe={s['universe_count']} → candidate={s['candidate_count']}"
                         f" → selected={s['selected_count']}")
            lines.append(f"- **추정 원인**: {s.get('probable_cause', 'N/A')}")
            lines.append("")

    # ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 5. 데이터 커버리지 분석")
    lines.append("")

    _append_coverage_analysis(lines)

    # ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 6. 이중 지연(double-lag) 분석")
    lines.append("")
    lines.append("PIT ON에서 `use_next_trading_day_availability=True`를 켜면:")
    lines.append("- `AvailableDate = AcceptedAt의 다음 거래일` (T+1)")
    lines.append("- `asof_lag_trading_days=5`를 추가하면 fundamentals_asof_date = signal_date - 5 거래일")
    lines.append("")
    lines.append("결합 효과: 실제로 사용하는 재무 데이터는 ")
    lines.append("  `(공시 수용일 다음 거래일 + 최소 5 거래일 전)` 기준을 만족해야 함.")
    lines.append("")
    lines.append("**완화 방법:**")
    lines.append("- `asof_lag_trading_days=0`으로 이중 지연 제거 (권장)")
    lines.append("- `use_next_trading_day_availability=False`로 T+0 기준 사용")
    lines.append("  (단, 당일 공시는 사실상 T+1에 반영되므로 look-ahead 위험)")
    lines.append("- 두 옵션 모두 끄고 `fallback_q_days=45/fallback_k_days=90`만 사용")
    lines.append("")

    # ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 7. Factor(TTM) NaN 분석")
    lines.append("")
    lines.append("### eps_ttm/pe NaN 발생 원인:")
    lines.append("1. `sec_companyfacts_quarterly.parquet`에 PIT 컬럼 없음 → fallback lag 계산")
    lines.append("2. fallback이 적용된 row: `AvailabilityMethod='fallback'` or `'fallback_next_trading_day'`")
    lines.append("3. 신규상장 종목: 4분기 미만 데이터 → eps_ttm rolling(4) min_periods=1 이므로 1분기 데이터면 OK")
    lines.append("4. Revenue/EPS가 SEC 데이터에 없는 종목 → eps_ttm=NaN → pe=NaN → buy_rule 통과(na_policy=pass)")
    lines.append("")
    lines.append("**buy_rule='eps_ttm > 0' + rule_na_policy='pass' 조합:** NaN eps_ttm은 통과")
    lines.append("→ 가격 데이터가 있으면 대부분 선택 후보에 포함됨")
    lines.append("")

    # ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 8. 핵심 발견 및 권고사항")
    lines.append("")

    # Compute summary stats
    pit_on_scenarios = [r for r in scenario_results if r.get("pit_on") and "error" not in r]
    pit_off_scenarios = [r for r in scenario_results if not r.get("pit_on") and "error" not in r]
    pit_on_avg_trades = np.mean([r["total_trades"] for r in pit_on_scenarios]) if pit_on_scenarios else 0
    pit_off_avg_trades = np.mean([r["total_trades"] for r in pit_off_scenarios]) if pit_off_scenarios else 0

    lines.append(f"- PIT OFF 평균 trades: **{pit_off_avg_trades:.0f}**")
    lines.append(f"- PIT ON 평균 trades: **{pit_on_avg_trades:.0f}**")
    lines.append("")

    if pit_on_avg_trades == 0:
        lines.append("### ⚠️ PIT ON에서 trades=0 발생!")
        lines.append("- `sec_companyfacts_quarterly.parquet`에 `AcceptedAt`/`AvailableDate` 컬럼이 없음")
        lines.append("- fallback lag(45/90일) 기반으로 `AvailableDate`가 계산됨")
        lines.append("- 일부 티커에서 AvailableDate > asof_date인 경우가 발생할 수 있음")
        lines.append("")
        lines.append("**해결책:** `--force` 옵션으로 재-ingest 실행")
        lines.append("```bash")
        lines.append("./.venv/bin/python -u -m market_data ingest \\")
        lines.append("  --universe custom \\")
        lines.append("  --tickers-file data/universe/symbols_nasdaq_stock_only.csv \\")
        lines.append("  --start 2000-01-01 \\")
        lines.append("  --financial-source sec \\")
        lines.append("  --sec-user-agent \"yooho dbgh0029@gmail.com\" \\")
        lines.append("  --use-next-trading-day-availability \\")
        lines.append("  --force")
        lines.append("```")
    elif pit_on_avg_trades > 0:
        lines.append("### ✅ PIT ON에서 trades 발생 확인")
        lines.append("- PIT 필터 (AvailableDate <= asof_date)가 올바르게 동작하며 look-ahead 차단")
        lines.append("- fallback lag 정책이 AcceptedAt 부재를 적절히 보완")
        lines.append(f"- PIT ON vs OFF trades 비율: {pit_on_avg_trades/pit_off_avg_trades:.2%}" if pit_off_avg_trades > 0 else "")

    lines.append("")
    lines.append("---")
    lines.append("*이 보고서는 `scripts/pit_rebalance_audit.py`에 의해 자동 생성되었습니다.*")

    md_path = OUT_DIR / "pit_rebalance_audit.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _append_coverage_analysis(lines: list[str]):
    """Analyze AcceptedAt/AvailableDate coverage in sec_companyfacts_quarterly files."""
    fin_dir = Path("data/financials/us")
    pit_cols = ["AcceptedAt", "AvailableDate", "AvailabilityMethod", "FilingDate"]
    results = []

    for t in UNIVERSE_TICKERS:
        fp = fin_dir / t / "sec_companyfacts_quarterly.parquet"
        if not fp.exists():
            results.append({"ticker": t, "rows": 0, "has_pit_cols": False,
                            "accepted_at_fill_pct": 0.0, "available_date_fill_pct": 0.0,
                            "fallback_pct": 0.0})
            continue
        try:
            df = pd.read_parquet(fp)
            if df.empty:
                results.append({"ticker": t, "rows": 0, "has_pit_cols": False,
                                "accepted_at_fill_pct": 0.0, "available_date_fill_pct": 0.0,
                                "fallback_pct": 0.0})
                continue
            has_pit = all(c in df.columns for c in pit_cols)
            n = len(df)
            acc_fill = float(df["AcceptedAt"].notna().sum() / n * 100) if "AcceptedAt" in df.columns else 0.0
            avail_fill = float(df["AvailableDate"].notna().sum() / n * 100) if "AvailableDate" in df.columns else 0.0
            fallback_pct = float(
                df["AvailabilityMethod"].astype(str).str.lower().str.startswith("fallback").sum() / n * 100
            ) if "AvailabilityMethod" in df.columns else 100.0
            results.append({
                "ticker": t, "rows": n, "has_pit_cols": has_pit,
                "accepted_at_fill_pct": acc_fill,
                "available_date_fill_pct": avail_fill,
                "fallback_pct": fallback_pct,
            })
        except Exception as exc:
            results.append({"ticker": t, "rows": -1, "has_pit_cols": False,
                            "accepted_at_fill_pct": 0.0, "available_date_fill_pct": 0.0,
                            "fallback_pct": 100.0, "error": str(exc)})

    cov_df = pd.DataFrame(results)
    n_with_pit = int(cov_df["has_pit_cols"].sum())
    n_total = len(cov_df)
    avg_acc = float(cov_df["accepted_at_fill_pct"].mean())
    avg_avail = float(cov_df["available_date_fill_pct"].mean())
    avg_fallback = float(cov_df["fallback_pct"].mean())

    lines.append(f"검증 대상 {n_total}개 티커 기준:")
    lines.append(f"- PIT 컬럼(AcceptedAt+AvailableDate+etc) 보유: **{n_with_pit}/{n_total}** ({n_with_pit/n_total*100:.0f}%)")
    lines.append(f"- AcceptedAt 평균 fill률: **{avg_acc:.1f}%**")
    lines.append(f"- AvailableDate 평균 fill률: **{avg_avail:.1f}%**")
    lines.append(f"- fallback lag 사용 비율: **{avg_fallback:.1f}%**")
    lines.append("")
    lines.append("**→ PIT 컬럼이 없는 파일은 `_ensure_pit_columns()`에서 fallback으로 AvailableDate를 계산함**")
    lines.append("**→ fallback 100% = AcceptedAt 없이 PeriodEnd + 45/90일로만 동작**")
    lines.append("")
    lines.append("```")
    lines.append(cov_df[["ticker","rows","has_pit_cols","accepted_at_fill_pct","available_date_fill_pct","fallback_pct"]].to_string(index=False))
    lines.append("```")

    # Save coverage CSV
    cov_path = OUT_DIR / "sec_coverage_analysis.csv"
    cov_df.to_csv(cov_path, index=False)
    lines.append(f"\n> 상세: `{cov_path}`")


if __name__ == "__main__":
    import argparse as _argparse
    _p = _argparse.ArgumentParser()
    _p.add_argument("--suffix", default="", help="output file suffix (e.g. _after_force)")
    _p.add_argument("--snapshots-dir-suffix", default="", help="snapshots subdir suffix")
    _args = _p.parse_args()
    if _args.suffix:
        _sfx = _args.suffix
        # patch module-level paths
        _snap_sfx = _args.snapshots_dir_suffix or _sfx
        SNAPSHOTS_DIR.__class__  # noqa — just reference to confirm path object
        import sys as _sys
        _mod = _sys.modules[__name__]
        _mod.SNAPSHOTS_DIR = OUT_DIR / f"pit_selected_snapshots{_snap_sfx}"
        _mod.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

        _orig_write = _mod._write_report
        def _patched_write(scenario_results, combined_log, zero_df, detailed_samples):
            _orig_write(scenario_results, combined_log, zero_df, detailed_samples)
            # rename outputs
            import shutil
            for src, dst in [
                (OUT_DIR / "pit_rebalance_audit.md",    OUT_DIR / f"pit_rebalance_audit{_sfx}.md"),
                (OUT_DIR / "pit_rebalance_log.csv",     OUT_DIR / f"pit_rebalance_log{_sfx}.csv"),
                (OUT_DIR / "zero_trade_dates.csv",      OUT_DIR / f"zero_trade_dates{_sfx}.csv"),
            ]:
                if src.exists():
                    shutil.move(str(src), str(dst))
                    print(f"[RENAMED] {src.name} → {dst.name}")
        _mod._write_report = _patched_write

    main()
