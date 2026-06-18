from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import json
import re
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from market_data.backtest.factors import available_price_symbols, build_factor_panel, load_factor_panel_from_local
from market_data.backtest.models import (
    BacktestResult,
    ConditionSet,
    FactorSpec,
    FundingConfig,
    Position,
    RankingConfig,
    StrategyConfig,
    Trade,
    UniverseConfig,
    load_ranking_config,
    load_strategy_config,
)
from market_data.backtest.condition_engine import evaluate_condition_set
from market_data.backtest.ranking import rank_cross_section
from market_data.backtest.report import (
    build_monthly_returns,
    normalize_metrics,
    build_yearly_returns,
    build_yearly_summary,
    compare_results,
    compute_drawdown_curve,
    compute_metrics,
    save_backtest_outputs,
)
from market_data.backtest.rules import evaluate_rule_detailed
from market_data.backtest.universe import apply_universe_filters, resolve_universe_symbols
from market_data.backtest.validation_symbol_time import load_ticker_validity_ranges
from market_data.reader import load_price_dataframe
from market_data.sp500_pit import (
    get_sp500_constituents_asof,
    get_sp500_symbol_universe_for_period,
    load_sp500_constituents_pit,
    validate_sp500_pit,
)
from market_data.universe import build_universe
from market_data.utils import ensure_dir, coerce_series_naive

LOGGER = logging.getLogger(__name__)


def _is_sp500_pit_source(config: StrategyConfig) -> bool:
    return str(config.universe.source or "").strip().lower() == "sp500_pit"


def _load_sp500_pit_run_context(config: StrategyConfig) -> tuple[pd.DataFrame, bool, float]:
    pit = load_sp500_constituents_pit()
    strict = bool(config.universe.sp500_pit_strict or config.universe.sp500_pit_fail_closed)
    min_conf = max(float(config.universe.sp500_pit_min_confidence or 0.0), 0.0)
    if pit.empty and strict:
        raise RuntimeError("sp500_pit source selected but PIT cache is missing/empty")
    if not pit.empty:
        report = validate_sp500_pit(
            pit,
            start=config.start,
            end=config.end,
            min_confidence=min_conf,
            fail_closed=False,
        )
        if strict and str(report.get("status", "warn")).lower() == "fail":
            raise RuntimeError(
                "sp500_pit validation failed in strict mode; "
                f"errors={len(report.get('summary', {}).get('errors', []))}"
            )
    return pit, strict, min_conf


def _resolve_panel_symbols(config: StrategyConfig) -> list[str]:
    avail = available_price_symbols(market=config.market)
    if not _is_sp500_pit_source(config):
        return resolve_universe_symbols(config.universe, avail)

    pit_symbols = get_sp500_symbol_universe_for_period(
        start=config.start,
        end=config.end,
        min_confidence=max(float(config.universe.sp500_pit_min_confidence or 0.0), 0.0),
    )
    if not pit_symbols:
        if bool(config.universe.sp500_pit_strict or config.universe.sp500_pit_fail_closed):
            raise RuntimeError("sp500_pit strict mode: no PIT symbols resolved for requested period")
        return avail
    avail_set = {str(s).upper() for s in avail}
    filtered = [s for s in pit_symbols if str(s).upper() in avail_set]
    if not filtered and bool(config.universe.sp500_pit_strict or config.universe.sp500_pit_fail_closed):
        raise RuntimeError("sp500_pit strict mode: no PIT symbols have local price data")
    return filtered or avail


def _apply_sp500_pit_asof_filter(
    frame_t: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
    pit_df: pd.DataFrame,
    strict: bool,
    min_confidence: float,
) -> tuple[pd.DataFrame, int]:
    if frame_t is None or frame_t.empty:
        return pd.DataFrame(), 0
    snap = get_sp500_constituents_asof(
        signal_date,
        pit_df=pit_df,
        min_confidence=min_confidence,
        strict=False,
    )
    if snap.empty:
        if strict:
            raise RuntimeError(
                "sp500_pit strict mode: empty as-of constituent snapshot "
                f"for signal_date={pd.Timestamp(signal_date).date().isoformat()}"
            )
        # Fallback to current universe (allow mode)
        return frame_t.copy(), 0

    def _norm_ticker(s: str) -> str:
        return str(s).strip().upper().replace(".", "").replace("-", "")

    snap_idx = snap.set_index("ticker")
    member_set = {_norm_ticker(s) for s in snap_idx.index.astype(str).tolist()}
    
    before = int(len(frame_t))
    
    # Use normalized ticker comparison for robustness
    current_tickers = frame_t.index.astype(str).to_series().map(_norm_ticker)
    out = frame_t.loc[current_tickers.isin(member_set).values].copy()
    dropped = max(before - int(len(out)), 0)
    if out.empty:
        return out, dropped
    evidence_cols = {
        "sp500_pit_member": True,
        "sp500_pit_included": True,
        "sp500_pit_source_tier": "source_tier",
        "sp500_pit_source_name": "source_name",
        "sp500_pit_source_url": "source_url",
        "sp500_pit_source_doc_id": "source_doc_id",
        "sp500_pit_evidence_text": "evidence_text",
        "sp500_pit_source_ref": "source_ref",
        "sp500_pit_confidence": "confidence",
        "sp500_pit_valid_from": "valid_from",
        "sp500_pit_valid_to": "valid_to",
        "sp500_pit_effective_date": "effective_date",
        "sp500_pit_announcement_date": "announcement_date",
    }
    reindex = out.index.astype(str).str.upper()
    out["sp500_pit_member"] = True
    for out_col, src_col in evidence_cols.items():
        if out_col == "sp500_pit_member":
            continue
        if src_col in snap_idx.columns:
            out[out_col] = snap_idx.reindex(reindex)[src_col].to_numpy()
        else:
            out[out_col] = np.nan
    # Friendly aliases used by AI export validation/report.
    if "sp500_pit_source_name" in out.columns:
        out["sp500_pit_source"] = out["sp500_pit_source_name"]
    elif "sp500_pit_source_tier" in out.columns:
        out["sp500_pit_source"] = out["sp500_pit_source_tier"]
    else:
        out["sp500_pit_source"] = np.nan
    if "sp500_pit_evidence_text" in out.columns:
        out["sp500_pit_provenance"] = out["sp500_pit_evidence_text"]
    elif "sp500_pit_source_ref" in out.columns:
        out["sp500_pit_provenance"] = out["sp500_pit_source_ref"]
    else:
        out["sp500_pit_provenance"] = np.nan
    return out, dropped


def normalize_frequency(freq: str) -> str:
    text = str(freq or "Q").strip().upper()
    if text in {"W", "WEEK", "WEEKLY"}:
        return "W"
    if text in {"M", "MONTH", "MONTHLY"}:
        return "M"
    if text in {"Y", "A", "YEAR", "ANNUAL", "YEARLY"}:
        return "Y"
    return "Q"


def generate_rebalance_dates(trading_dates: pd.DatetimeIndex, freq: str = "Q") -> pd.DatetimeIndex:
    if trading_dates is None or len(trading_dates) == 0:
        return pd.DatetimeIndex([])

    idx = pd.DatetimeIndex(pd.to_datetime(trading_dates, errors="coerce")).dropna().sort_values().unique()
    if len(idx) == 0:
        return pd.DatetimeIndex([])

    freq_norm = str(freq).strip().upper()
    
    # Standard frequencies (last trading day of period)
    # Only use this logic if it's EXACTLY W, M, Q, or Y to avoid catching W-SAT etc.
    if freq_norm in {"W", "M", "Q", "Y"}:
        period = freq_norm
        s = pd.Series(idx, index=idx)
        reb = s.groupby(idx.to_period(period)).max().sort_values()
        return pd.DatetimeIndex(reb.to_numpy())
    
    # Custom frequencies (e.g. W-SAT, 2W, etc.)
    # Generate dates starting slightly earlier to catch rebalances that might snap into the range
    start_anchor = idx.min() - pd.Timedelta(days=7)
    raw_dates = pd.date_range(start=start_anchor, end=idx.max(), freq=freq)
    snapped = []
    for rd in raw_dates:
        # searchsorted side='left' returns the first element >= rd
        pos = int(np.searchsorted(idx.values, np.datetime64(rd), side="left"))
        if pos < len(idx):
            # Ensure the snapped date is within the actual data range
            if idx[pos] >= idx.min():
                snapped.append(idx[pos])
            
    return pd.DatetimeIndex(snapped).unique().sort_values()


def generate_funding_dates(trading_dates: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    return generate_rebalance_dates(trading_dates, freq=freq)


def _shift_trading_day(trading_dates: pd.DatetimeIndex, anchor: pd.Timestamp, shift: int) -> pd.Timestamp:
    idx = pd.DatetimeIndex(pd.to_datetime(trading_dates, errors="coerce")).dropna().sort_values().unique()
    if len(idx) == 0:
        return pd.Timestamp(anchor)
    pos = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(anchor)), side="left"))
    if pos >= len(idx) or idx[pos] != pd.Timestamp(anchor):
        pos = max(pos - 1, 0)
    shifted = min(max(pos + int(shift), 0), len(idx) - 1)
    return pd.Timestamp(idx[shifted])


def count_contribution_events(
    trading_dates: pd.DatetimeIndex,
    contribution_freq: str,
    execution_timing: str = "next_open",
) -> int:
    idx = pd.DatetimeIndex(pd.to_datetime(trading_dates, errors="coerce")).dropna().sort_values().unique()
    if len(idx) == 0:
        return 0
    funding_dates = generate_funding_dates(idx, freq=contribution_freq)
    exec_map = _build_execution_map(idx, funding_dates, execution_timing=execution_timing)
    return int(len(exec_map))


def _normalize_funding_alignment_mode(mode: str | None) -> str:
    text = str(mode or "default").strip().lower()
    if text in {"align_lump_to_dca_total", "custom_total", "default"}:
        return text
    return "default"


def _normalize_funding_mode(mode: str | None) -> str:
    text = str(mode or "lump_sum").strip().lower()
    if text in {"dca", "va", "lump_sum"}:
        return text
    return "lump_sum"


def _normalize_va_min_policy(policy: str | None) -> str:
    text = str(policy or "positive_raw_only").strip().lower()
    if text in {"positive_raw_only", "every_rebalance"}:
        return text
    return "positive_raw_only"


def _effective_funding_config(config: StrategyConfig) -> FundingConfig:
    default_funding = FundingConfig()
    raw = config.funding if isinstance(config.funding, FundingConfig) else FundingConfig()
    mode = _normalize_funding_mode(raw.mode)
    raw_initial = raw.initial_cash if raw.initial_cash is not None else default_funding.initial_cash
    if float(raw_initial) == float(default_funding.initial_cash) and float(config.initial_cash) != float(default_funding.initial_cash):
        initial_cash = float(config.initial_cash)
    else:
        initial_cash = float(raw_initial)
    freq = normalize_frequency(raw.contribution_freq or config.frequency)
    if freq not in {"M", "Q", "Y"}:
        freq = normalize_frequency(config.frequency)
    if freq not in {"M", "Q", "Y"}:
        freq = "M"
    out = FundingConfig(
        mode=mode,
        initial_cash=max(initial_cash, 0.0),
        contribution_freq=freq,
        fixed_contribution=(float(raw.fixed_contribution) if raw.fixed_contribution is not None else None),
        va_target_step=(float(raw.va_target_step) if raw.va_target_step is not None else None),
        va_target_base=float(raw.va_target_base or 0.0),
        va_min_contribution=max(float(raw.va_min_contribution or 0.0), 0.0),
        va_min_policy=_normalize_va_min_policy(raw.va_min_policy),
        va_max_contribution=(float(raw.va_max_contribution) if raw.va_max_contribution is not None else None),
        va_allow_withdrawal=bool(raw.va_allow_withdrawal),
    )
    if out.va_max_contribution is not None and out.va_max_contribution < 0:
        out.va_max_contribution = 0.0
    if out.va_max_contribution is not None and out.va_max_contribution < out.va_min_contribution:
        LOGGER.warning(
            "Funding config adjusted: va_max_contribution(%.4f) < va_min_contribution(%.4f); using max=min",
            float(out.va_max_contribution),
            float(out.va_min_contribution),
        )
        out.va_max_contribution = float(out.va_min_contribution)
    return out


def _resolve_symbol_identity_overrides_path() -> Path | None:
    for cand in [Path("config") / "symbol_identity_overrides.csv", Path("config") / "symbol_identity_overrides.json"]:
        if cand.exists():
            return cand
    return None


def _apply_symbol_validity_window(panel: pd.DataFrame, *, market: str) -> pd.DataFrame:
    if panel is None or panel.empty:
        return panel

    index_names = [str(n) for n in panel.index.names]
    if "Ticker" not in index_names:
        return panel

    def _norm_ticker(s: str) -> str:
        return str(s).strip().upper().replace(".", "").replace("-", "")

    tickers = panel.index.get_level_values("Ticker").astype(str).str.upper()
    tickers_norm = pd.Series(tickers).map(_norm_ticker).values
    unique_tickers = sorted(set(tickers))
    if not unique_tickers:
        return panel

    override_path = _resolve_symbol_identity_overrides_path()
    ranges = load_ticker_validity_ranges(
        unique_tickers,
        market=market,
        price_root=Path("data") / "prices",
        overrides_path=override_path,
    )
    if not ranges:
        return panel

    date_level = pd.to_datetime(panel.index.get_level_values(0), errors="coerce")
    if hasattr(date_level, "tz") and date_level.tz is not None:
        date_level = date_level.tz_localize(None)
    
    keep_mask = np.ones(len(panel), dtype=bool)
    filtered_by_override = 0
    filtered_total = 0
    for sym, info in ranges.items():
        first_valid = info.get("first_valid_date")
        last_valid = info.get("last_valid_date")
        source_used = str(info.get("source_used", ""))
        
        target_sym_norm = _norm_ticker(sym)
        sym_mask = tickers_norm == target_sym_norm
        
        if not sym_mask.any():
            continue
            
        pre_cnt = int(sym_mask.sum())
        if first_valid is not None:
            fv_ts = pd.Timestamp(first_valid)
            if fv_ts.tz is not None:
                fv_ts = fv_ts.tz_localize(None)
            drop_mask = sym_mask & (date_level < fv_ts)
            if source_used == "override":
                filtered_by_override += int(drop_mask.sum())
            keep_mask &= ~drop_mask
        if last_valid is not None:
            lv_ts = pd.Timestamp(last_valid)
            if lv_ts.tz is not None:
                lv_ts = lv_ts.tz_localize(None)
            drop_mask = sym_mask & (date_level > lv_ts)
            if source_used == "override":
                filtered_by_override += int(drop_mask.sum())
            keep_mask &= ~drop_mask
        post_cnt = int((sym_mask & keep_mask).sum())
        filtered_total += max(pre_cnt - post_cnt, 0)

    if not keep_mask.any():
        LOGGER.warning("Symbol validity filter removed all rows; falling back to original panel.")
        return panel

    if filtered_total > 0:
        LOGGER.info(
            "Applied symbol validity window filter: removed_rows=%d (override_rows=%d)",
            filtered_total,
            filtered_by_override,
        )
    return panel.loc[keep_mask]


def _runtime_snapshot_root(config: StrategyConfig) -> Path:
    if config.out_dir:
        return Path(config.out_dir).expanduser()
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(config.name or "strategy")).strip("_") or "strategy"
    run_id = f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}_{safe_name}_{uuid4().hex[:8]}"
    return Path("logs") / "backtests" / "_runtime_snapshots" / run_id


def compute_contribution(
    funding_cfg: FundingConfig,
    step_index: int,
    rebalance_date: pd.Timestamp,
    portfolio_value_before: float,
) -> tuple[float, float, float | None, str]:
    mode = _normalize_funding_mode(funding_cfg.mode)
    step = max(int(step_index), 1)
    value_before = float(portfolio_value_before)

    if mode == "dca":
        raw = float(funding_cfg.fixed_contribution or 0.0)
        contrib = max(raw, 0.0)
        reason = f"dca_fixed={contrib:.2f}"
        return contrib, raw, None, reason

    if mode == "va":
        target_step = float(funding_cfg.va_target_step or 0.0)
        target_value = float(funding_cfg.va_target_base) + float(step * target_step)
        raw = float(target_value - value_before)
        min_cap = max(float(funding_cfg.va_min_contribution or 0.0), 0.0)
        max_cap = float(funding_cfg.va_max_contribution) if funding_cfg.va_max_contribution is not None else None
        policy = _normalize_va_min_policy(funding_cfg.va_min_policy)
        reason_tokens: list[str] = [f"va_raw={raw:.2f}"]

        if max_cap is not None and max_cap < min_cap:
            LOGGER.warning(
                "VA max cap lower than min floor at compute step; adjusted max to min (min=%.4f, max=%.4f)",
                min_cap,
                max_cap,
            )
            max_cap = float(min_cap)
            reason_tokens.append("max_lt_min_adjusted")

        effective_policy = policy
        if policy == "every_rebalance" and bool(funding_cfg.va_allow_withdrawal):
            # Withdrawal semantics conflict with unconditional floor semantics; keep backward-compatible behavior.
            effective_policy = "positive_raw_only"
            reason_tokens.append("policy_fallback_positive_raw_only_allow_withdrawal")

        if effective_policy == "every_rebalance":
            contrib = max(raw, min_cap)
            if contrib > raw and min_cap > 0.0:
                reason_tokens.append("va_every_rebalance_floor")
            elif raw <= 0.0 and min_cap <= 0.0:
                reason_tokens.append("va_every_rebalance_zero_floor")
            else:
                reason_tokens.append("va_raw_pass_through")
        else:
            # positive_raw_only
            if raw <= 0.0 and not funding_cfg.va_allow_withdrawal:
                return 0.0, raw, target_value, "va_positive_raw_only_clipped_zero"
            contrib = raw
            if contrib > 0.0 and min_cap > 0.0 and contrib < min_cap:
                contrib = min_cap
                reason_tokens.append("va_capped_to_min")
            elif contrib > 0.0:
                reason_tokens.append("va_raw_pass_through")
            elif contrib < 0.0:
                reason_tokens.append("va_withdrawal_allowed")
            else:
                reason_tokens.append("va_zero")

        if max_cap is not None and contrib > max_cap:
            contrib = float(max_cap)
            reason_tokens.append("va_capped_to_max")

        return float(contrib), raw, target_value, " | ".join(reason_tokens)

    # lump_sum (default): no periodic contributions after t=0 initial deposit.
    return 0.0, 0.0, None, "lump_sum_no_flow"


def _build_execution_map(
    trading_dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    execution_timing: str,
) -> dict[pd.Timestamp, pd.Timestamp]:
    idx = pd.DatetimeIndex(trading_dates).sort_values()
    out: dict[pd.Timestamp, pd.Timestamp] = {}
    same_close = str(execution_timing).strip().lower() == "same_close"

    for reb in pd.DatetimeIndex(rebalance_dates):
        if same_close:
            if reb in idx:
                out[pd.Timestamp(reb)] = pd.Timestamp(reb)
            continue

        pos = int(np.searchsorted(idx.values, np.datetime64(reb), side="right"))
        if pos >= len(idx):
            continue
        out[pd.Timestamp(idx[pos])] = pd.Timestamp(reb)
    return out


def _resolve_execution_price_row(
    *,
    signal_date: pd.Timestamp,
    exec_date: pd.Timestamp,
    timing: str,
    price_basis: str,
    price_offset_pct: float,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
) -> tuple[pd.Series, str, float]:
    # Ensure dates are timezone-naive for index lookup.
    sig_ts = pd.Timestamp(signal_date)
    if sig_ts.tz is not None:
        sig_ts = sig_ts.tz_localize(None)
    ex_ts = pd.Timestamp(exec_date)
    if ex_ts.tz is not None:
        ex_ts = ex_ts.tz_localize(None)

    timing_norm = str(timing or "next_open").strip().lower()
    basis_norm = str(price_basis or "auto").strip().lower()
    if basis_norm not in {"auto", "open", "close", "prev_close"}:
        basis_norm = "auto"
    if basis_norm == "auto":
        basis_norm = "close" if timing_norm == "same_close" else "open"

    if basis_norm == "open":
        row = open_px.loc[ex_ts]
    elif basis_norm == "close":
        row = close_px.loc[ex_ts]
    elif basis_norm == "prev_close":
        if sig_ts in close_px.index:
            row = close_px.loc[sig_ts]
        else:
            row = close_px.loc[:sig_ts].iloc[-1]
    else:
        row = open_px.loc[ex_ts]

    offset = float(price_offset_pct or 0.0) / 100.0
    if offset != 0.0:
        row = pd.to_numeric(row, errors="coerce") * (1.0 + offset)
    return row, basis_norm, float(price_offset_pct or 0.0)


def _effective_execution_timing(config: StrategyConfig) -> str:
    legacy = str(config.execution_timing or "next_open").strip().lower() or "next_open"
    new_timing = str(getattr(config.execution, "timing", "") or "").strip().lower()
    if new_timing in {"next_open", "same_close"}:
        # Preserve backward compatibility: explicit legacy same_close should win
        # when the new nested field is just default next_open.
        if not (new_timing == "next_open" and legacy == "same_close"):
            return new_timing
    return legacy


def _cross_section(panel: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    ts = pd.Timestamp(date)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    if ts not in panel.index.get_level_values(0):
        return pd.DataFrame()
    out = panel.xs(ts, level=0).copy()
    out.index = out.index.astype(str).str.upper()
    return out


def _default_ranking_for_screen() -> RankingConfig:
    return RankingConfig(
        normalization="percentile",
        factors=[
            FactorSpec(name="pe", direction="asc", weight=1.0),
            FactorSpec(name="ps", direction="asc", weight=1.0),
            FactorSpec(name="fcf_yield", direction="desc", weight=0.5),
        ],
    )


def _compile_indicator_inputs(config: StrategyConfig) -> tuple[str, RankingConfig]:
    if not config.indicators:
        ranking = config.ranking if config.ranking and config.ranking.factors else _default_ranking_for_screen()
        return config.buy_rules or "", ranking

    filter_parts_g1: list[str] = []
    filter_parts_g2: list[str] = []
    factor_specs: list[FactorSpec] = []

    for item in config.indicators:
        if not isinstance(item, dict):
            continue
        metric_id = str(item.get("id", "")).strip()
        if not metric_id:
            continue
        enabled = bool(item.get("enabled", True))
        if not enabled:
            continue

        if bool(item.get("as_filter", False)):
            op = str(item.get("op", "<=")).strip()
            threshold = item.get("threshold")
            try:
                filter_group = int(item.get("filter_group", 1))
            except Exception:
                filter_group = 1
            if filter_group not in {1, 2}:
                filter_group = 1
            if threshold is not None and op in {"<=", "<", ">=", ">", "==", "!="}:
                token = f"{metric_id} {op} {float(threshold)}"
                if filter_group == 2:
                    filter_parts_g2.append(token)
                else:
                    filter_parts_g1.append(token)

        if bool(item.get("as_rank", False)):
            factor_specs.append(
                FactorSpec(
                    name=metric_id,
                    direction=str(item.get("direction_override") or item.get("direction") or "asc"),
                    weight=float(item.get("weight", 1.0)),
                    winsorize_low=(float(item["winsorize_low"]) if item.get("winsorize_low") is not None else None),
                    winsorize_high=(float(item["winsorize_high"]) if item.get("winsorize_high") is not None else None),
                )
            )

    if not filter_parts_g1 and not filter_parts_g2 and config.buy_rules:
        rule_expr = config.buy_rules
    else:
        g1_expr = " & ".join(filter_parts_g1)
        g2_expr = " & ".join(filter_parts_g2)
        if g1_expr and g2_expr:
            rule_expr = f"({g1_expr}) | ({g2_expr})"
        else:
            rule_expr = g1_expr or g2_expr

    ranking = RankingConfig(normalization="percentile", factors=factor_specs) if factor_specs else _default_ranking_for_screen()
    return rule_expr, ranking


def _pick_targets(
    frame_t: pd.DataFrame,
    config: StrategyConfig,
    positions: dict[str, Position],
    buy_rule_expr: str,
    ranking_cfg: RankingConfig,
    panel: pd.DataFrame | None = None,
    signal_date: pd.Timestamp | None = None,
) -> tuple[list[str], pd.DataFrame, dict[str, str], dict[str, str], dict[str, str], pd.DataFrame, pd.Series]:
    if frame_t is None or frame_t.empty:
        return [], pd.DataFrame(), {}, {}, {}, pd.DataFrame(), pd.Series(dtype=bool)

    universe_frame = apply_universe_filters(frame_t, config.universe, filter_config=config.universe_filter)
    if universe_frame.empty:
        return [], pd.DataFrame(), {}, {}, {}, pd.DataFrame(), pd.Series(dtype=bool)

    buy_condition_set: ConditionSet = config.buy_condition_set
    if buy_condition_set and buy_condition_set.conditions:
        buy_mask, buy_detail = evaluate_condition_set(
            buy_condition_set,
            universe_frame,
            panel=panel,
            signal_date=signal_date,
            na_policy=config.rule_na_policy,
        )
    else:
        buy_mask, buy_detail = evaluate_rule_detailed(buy_rule_expr, universe_frame, na_policy=config.rule_na_policy)

    sell_set_mask = pd.Series(False, index=universe_frame.index)
    if config.sell_condition_set and config.sell_condition_set.conditions:
        sell_set_mask, _ = evaluate_condition_set(
            config.sell_condition_set,
            universe_frame,
            panel=panel,
            signal_date=signal_date,
            na_policy=config.rule_na_policy,
        )

    candidates = universe_frame.loc[buy_mask].copy()

    ranked = rank_cross_section(candidates, ranking_cfg)
    if not ranked.empty:
        if str(config.missing_policy).lower() == "drop":
            ranked = ranked.loc[ranked["rank_score"].notna()].copy()
        else:
            ranked["rank_score"] = pd.to_numeric(ranked["rank_score"], errors="coerce").fillna(0.0)
        ranked = ranked.sort_values("rank_score", ascending=False).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)

    sell_reasons: dict[str, str] = {}
    sell_reason_details: dict[str, str] = {}
    buy_reason_details: dict[str, str] = {}
    requested_holdings = max(int(config.holdings), 1)
    holdings_n = requested_holdings
    max_holdings = config.position_sizing.max_holdings
    if max_holdings is not None and max_holdings > 0:
        holdings_n = min(holdings_n, int(max_holdings))

    rank_map = pd.Series(dtype=float)
    score_map = pd.Series(dtype=float)
    if not ranked.empty and "rank" in ranked.columns:
        rank_map = pd.to_numeric(ranked["rank"], errors="coerce")
    if not ranked.empty and "rank_score" in ranked.columns:
        score_map = pd.to_numeric(ranked["rank_score"], errors="coerce")

    def _rule_detail_for(sym: str) -> dict[str, bool] | None:
        if not isinstance(buy_detail, dict):
            return None
        return buy_detail.get(str(sym).upper()) or buy_detail.get(str(sym))

    def _format_rank(sym: str) -> str:
        rank_val = pd.to_numeric(pd.Series([rank_map.get(sym, np.nan)]), errors="coerce").iloc[0]
        return str(int(rank_val)) if np.isfinite(rank_val) else "NA"

    def _format_score(sym: str) -> str:
        score_val = pd.to_numeric(pd.Series([score_map.get(sym, np.nan)]), errors="coerce").iloc[0]
        return f"{float(score_val):.4f}" if np.isfinite(score_val) else "NA"

    def _condition_lists(sym: str) -> tuple[list[str], list[str]]:
        detail = _rule_detail_for(sym)
        if not detail:
            token = str(buy_rule_expr or "").replace(" ", "")
            return ([token] if token else []), []
        passed = [cond for cond, ok in detail.items() if bool(ok)]
        failed = [cond for cond, ok in detail.items() if not bool(ok)]
        return passed, failed

    def _build_buy_detail(sym: str) -> str:
        passed, failed = _condition_lists(sym)
        parts = [f"buy: selected rank={_format_rank(sym)} score={_format_score(sym)}"]
        if holdings_n < requested_holdings:
            parts.append("capped_by_max_holdings")
        if passed:
            parts.append(f"passed: {','.join(passed)}")
        if failed:
            parts.append(f"failed: {','.join(failed)}")
        return " | ".join(parts)

    sell_mode = str(config.sell_mode or "B").strip().upper()
    if config.mode == "screen" or sell_mode == "A":
        selected = ranked.index.tolist()[:holdings_n]
        for sym in positions:
            if positions[sym].shares > 0:
                sell_reasons[sym] = "screen_rebalance"
                sell_reason_details[sym] = "sell: screen_rebalance (rebalance refresh)"
        for sym in selected:
            buy_reason_details[sym] = _build_buy_detail(sym)
        return selected, ranked, sell_reasons, sell_reason_details, buy_reason_details, universe_frame, buy_mask

    current_symbols = [sym for sym, pos in positions.items() if pos.shares > 0]
    keep: list[str] = []

    for sym in current_symbols:
        if sym not in universe_frame.index:
            sell_reasons[sym] = "universe_exit"
            sell_reason_details[sym] = "sell: universe_exit"
            continue
        if bool(sell_set_mask.get(sym, False)):
            sell_reasons[sym] = "sell_rule_trigger"
            sell_reason_details[sym] = f"sell: sell_condition_set_trigger rank={_format_rank(sym)}"
            continue
        buy_ok = bool(buy_mask.get(sym, False))
        if not buy_ok:
            sell_reasons[sym] = "buy_rule_fail"
            _passed, failed = _condition_lists(sym)
            failed_txt = ",".join([f"{item}:False" for item in failed]) if failed else f"{str(buy_rule_expr or '').replace(' ', '')}:False"
            sell_reason_details[sym] = f"sell: filter_fail ({failed_txt}) rank={_format_rank(sym)} cutoff={2 * holdings_n}"
            continue
        if sell_mode == "C":
            multiplier = max(float(config.rank_drop_multiplier), 1.0)
            rank_cutoff = int(np.ceil(multiplier * holdings_n))
            rank_val = rank_map.get(sym, np.nan)
            if not np.isfinite(rank_val) or float(rank_val) > float(rank_cutoff):
                sell_reasons[sym] = "rank_drop"
                sell_reason_details[sym] = f"sell: rank_drop rank={_format_rank(sym)} cutoff={rank_cutoff}"
                continue
        keep.append(sym)

    if len(keep) > holdings_n:
        keep_sorted = sorted(keep, key=lambda s: float(rank_map.get(s, 1e12)))
        keep = keep_sorted[:holdings_n]
        for sym in keep_sorted[holdings_n:]:
            sell_reasons[sym] = "rank_trim"
            sell_reason_details[sym] = f"sell: rank_trim rank={_format_rank(sym)} cutoff={holdings_n}"

    selected = keep.copy()
    for sym in ranked.index.tolist():
        if sym in selected:
            continue
        selected.append(sym)
        if len(selected) >= holdings_n:
            break

    max_new_limit = config.position_sizing.max_new_buys_per_day
    if max_new_limit is not None and max_new_limit > 0:
        current_set = {str(sym).upper() for sym in current_symbols}
        kept_selected: list[str] = []
        new_count = 0
        for sym in selected:
            is_new = str(sym).upper() not in current_set
            if is_new and new_count >= int(max_new_limit):
                sell_reasons[sym] = "daily_new_buy_limit"
                sell_reason_details[sym] = "sell: capped_by_daily_new_buy_limit"
                continue
            if is_new:
                new_count += 1
            kept_selected.append(sym)
        selected = kept_selected

    for sym in selected:
        buy_reason_details[sym] = _build_buy_detail(sym)

    return selected, ranked, sell_reasons, sell_reason_details, buy_reason_details, universe_frame, buy_mask


def _target_weights(
    selected: list[str],
    ranked: pd.DataFrame,
    sizing: str,
    position_weight_pct: float | None = None,
) -> dict[str, float]:
    if not selected:
        return {}

    n = len(selected)
    sizing_norm = str(sizing or "equal").strip().lower()
    if sizing_norm != "rank_weight" or ranked.empty or "rank_score" not in ranked.columns:
        w = 1.0 / n
        base = {sym: w for sym in selected}
    else:
        scores = pd.to_numeric(ranked.reindex(selected)["rank_score"], errors="coerce").fillna(0.0)
        scores = scores.clip(lower=0.0)
        total = float(scores.sum())
        if not np.isfinite(total) or total <= 0.0:
            w = 1.0 / n
            base = {sym: w for sym in selected}
        else:
            base = {sym: float(scores.loc[sym] / total) for sym in selected}

    if position_weight_pct is None:
        return base

    cap = max(float(position_weight_pct), 0.0) / 100.0
    if cap <= 0.0:
        return base
    capped = {sym: min(float(w), cap) for sym, w in base.items()}
    total = float(sum(capped.values()))
    if total <= 0.0:
        return base
    if total > 1.0:
        scale = 1.0 / total
        capped = {sym: float(w * scale) for sym, w in capped.items()}
    return capped


def _portfolio_value(close_row: pd.Series, positions: dict[str, Position], cash: float) -> float:
    value = float(cash)
    if positions:
        for sym, pos in positions.items():
            if pos.shares <= 0:
                continue
            px = pd.to_numeric(close_row.get(sym), errors="coerce")
            if pd.isna(px):
                continue
            value += float(pos.shares) * float(px)
    return float(value)


def _execute_rebalance(
    signal_date: pd.Timestamp,
    exec_date: pd.Timestamp,
    positions: dict[str, Position],
    cash: float,
    exec_row: pd.Series,
    target_weights: dict[str, float],
    signal_equity: float,
    sell_reasons: dict[str, str],
    sell_reason_details: dict[str, str],
    buy_reason_details: dict[str, str],
    commission_bps: float,
    slippage_bps: float,
    share_mode: str = "fractional",
    cash_buffer_pct: float = 0.0,
    min_trade_notional: float = 0.0,
    max_new_buys_per_day: int | None = None,
    max_buy_amount_per_position: float | None = None,
    min_cash_reserve_pct: float = 0.0,
    execution_timing: str = "next_open",
    price_basis: str = "auto",
    price_offset_pct: float = 0.0,
) -> tuple[float, list[Trade]]:
    comm_rate = float(commission_bps) / 10_000.0
    slip_rate = float(slippage_bps) / 10_000.0
    integer_mode = str(share_mode or "fractional").strip().lower() == "integer"
    cash_keep_ratio = min(max(max(float(cash_buffer_pct), float(min_cash_reserve_pct)), 0.0), 1.0)
    alloc_equity = float(signal_equity) * (1.0 - cash_keep_ratio)
    min_notional = max(float(min_trade_notional), 0.0)
    max_buy_amt = float(max_buy_amount_per_position) if max_buy_amount_per_position is not None else None
    new_buy_limit = int(max_new_buys_per_day) if max_new_buys_per_day is not None else None
    if new_buy_limit is not None and new_buy_limit <= 0:
        new_buy_limit = None

    all_tickers = set(target_weights.keys()) | set(positions.keys())
    desired_shares: dict[str, float] = {}

    for sym in all_tickers:
        px = pd.to_numeric(exec_row.get(sym), errors="coerce")
        if pd.isna(px) or float(px) <= 0.0:
            desired_shares[sym] = 0.0 if sym not in target_weights else float(positions.get(sym, Position(sym, 0.0, 0.0, exec_date)).shares)
            continue
        if sym in target_weights:
            raw_shares = float(alloc_equity * target_weights[sym] / float(px))
            desired_shares[sym] = float(np.floor(raw_shares)) if integer_mode else raw_shares
        else:
            desired_shares[sym] = 0.0

    trades: list[Trade] = []

    new_buys_today = 0

    for sym in sorted(all_tickers):
        pos = positions.get(sym)
        current_shares = float(pos.shares if pos is not None else 0.0)
        target = float(desired_shares.get(sym, 0.0))
        delta = target - current_shares
        if delta >= -1e-12:
            continue

        px = pd.to_numeric(exec_row.get(sym), errors="coerce")
        if pd.isna(px) or float(px) <= 0.0:
            continue

        shares = min(current_shares, -delta)
        if integer_mode:
            shares = float(np.floor(shares))
        if shares <= 1e-12:
            continue

        price = float(px)
        notional = shares * price
        if notional < min_notional:
            continue
        commission = notional * comm_rate
        slippage = notional * slip_rate
        cash_delta = notional - commission - slippage

        avg_cost = float(pos.avg_cost if pos is not None else price)
        realized = (price - avg_cost) * shares - commission - slippage

        before = current_shares
        after = max(current_shares - shares, 0.0)
        cash += cash_delta

        if pos is not None:
            if after <= 1e-10:
                positions.pop(sym, None)
            else:
                pos.shares = after

        reason = sell_reasons.get(sym, "rebalance_reduce")
        reason_detail = sell_reason_details.get(sym, f"sell: {reason}")
        trades.append(
            Trade(
                signal_date=signal_date,
                exec_date=exec_date,
                ticker=sym,
                side="sell",
                shares=float(shares),
                exec_price=price,
                notional=float(notional),
                commission=float(commission),
                slippage=float(slippage),
                cash_delta=float(cash_delta),
                reason=reason,
                before_shares=float(before),
                after_shares=float(after),
                realized_pnl=float(realized),
                reason_detail=reason_detail,
            )
        )

    for sym in sorted(all_tickers):
        pos = positions.get(sym)
        current_shares = float(pos.shares if pos is not None else 0.0)
        target = float(desired_shares.get(sym, 0.0))
        delta = target - current_shares
        if delta <= 1e-12:
            continue

        px = pd.to_numeric(exec_row.get(sym), errors="coerce")
        if pd.isna(px) or float(px) <= 0.0:
            continue

        price = float(px)
        per_share_cash = price * (1.0 + comm_rate + slip_rate)
        if per_share_cash <= 0:
            continue

        detail_flags: list[str] = []
        if current_shares <= 1e-12 and new_buy_limit is not None and new_buys_today >= int(new_buy_limit):
            continue

        affordable = float(cash / per_share_cash)
        if integer_mode:
            affordable = float(np.floor(affordable))
        shares = min(delta, affordable)
        if max_buy_amt is not None and max_buy_amt > 0.0:
            share_cap = float(max_buy_amt / price)
            if integer_mode:
                share_cap = float(np.floor(share_cap))
            if share_cap < shares:
                shares = share_cap
                detail_flags.append("capped_by_max_buy_amount")
        if integer_mode:
            shares = float(np.floor(shares))
        if shares <= 1e-12:
            continue

        notional = shares * price
        if notional < min_notional:
            continue
        # Preserve minimum cash reserve even after buy.
        reserve_floor = float(signal_equity) * cash_keep_ratio
        total_buy_cash = notional * (1.0 + comm_rate + slip_rate)
        if float(cash - total_buy_cash) < reserve_floor - 1e-9:
            max_spend = max(float(cash - reserve_floor), 0.0)
            shares_reserve = max_spend / per_share_cash
            if integer_mode:
                shares_reserve = float(np.floor(shares_reserve))
            if shares_reserve <= 1e-12:
                continue
            if shares_reserve < shares:
                shares = float(shares_reserve)
                notional = shares * price
                detail_flags.append("capped_by_cash_reserve")
            if notional < min_notional:
                continue

        commission = notional * comm_rate
        slippage = notional * slip_rate
        cash_delta = -(notional + commission + slippage)

        before = current_shares
        after = current_shares + shares
        cash += cash_delta

        if pos is None:
            avg_cost = float((notional + commission + slippage) / shares)
            positions[sym] = Position(ticker=sym, shares=float(after), avg_cost=avg_cost, entry_date=exec_date)
        else:
            total_cost = float(pos.avg_cost * pos.shares) + float(notional + commission + slippage)
            pos.shares = float(after)
            pos.avg_cost = float(total_cost / after) if after > 0 else float(pos.avg_cost)

        reason = "target_entry" if before <= 1e-12 else "rebalance_add"
        detail_parts = [buy_reason_details.get(sym, f"buy: {reason}"), f"target_w={float(target_weights.get(sym, 0.0)):.4f}"]
        if detail_flags:
            detail_parts.extend(detail_flags)
        if shares + 1e-12 < delta:
            detail_parts.append("partial_fill cash_limit")
            if integer_mode:
                detail_parts.append("integer_rounding floor_shares")
        reason_detail = " | ".join(detail_parts)
        trades.append(
            Trade(
                signal_date=signal_date,
                exec_date=exec_date,
                ticker=sym,
                side="buy",
                shares=float(shares),
                exec_price=price,
                notional=float(notional),
                commission=float(commission),
                slippage=float(slippage),
                cash_delta=float(cash_delta),
                reason=reason,
                before_shares=float(before),
                after_shares=float(after),
                realized_pnl=0.0,
                reason_detail=reason_detail,
            )
        )
        if before <= 1e-12:
            new_buys_today += 1

    return float(cash), trades


def _load_benchmark_equity(
    benchmark: str,
    market: str,
    trading_index: pd.DatetimeIndex,
    initial_cash: float,
    funding_flows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ticker = str(benchmark or "").strip()
    if not ticker:
        return pd.DataFrame(columns=["date", "equity"])

    map_ticker = "^GSPC" if ticker.upper() in {"GSPC", "SP500"} else ticker
    try:
        px, _ = load_price_dataframe(ticker=map_ticker, market=market)
    except Exception:
        return pd.DataFrame(columns=["date", "equity"])

    if px is None or px.empty:
        return pd.DataFrame(columns=["date", "equity"])

    close_col = "Adj Close" if "Adj Close" in px.columns else "Close"
    series = pd.to_numeric(px[close_col], errors="coerce")
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.loc[~series.index.isna()].sort_index()
    series = series.reindex(trading_index).ffill().dropna()
    if series.empty:
        return pd.DataFrame(columns=["date", "equity"])

    flows = pd.DataFrame(columns=["date", "contribution"])
    if funding_flows is not None and not funding_flows.empty:
        flows = funding_flows.copy()
        flows["date"] = pd.to_datetime(flows.get("date"), errors="coerce")
        flows["contribution"] = pd.to_numeric(flows.get("contribution"), errors="coerce").fillna(0.0)
        flows = flows.dropna(subset=["date"])
        flows = flows.groupby("date", as_index=False)["contribution"].sum()

    if flows.empty:
        base = float(series.iloc[0])
        if base <= 0:
            return pd.DataFrame(columns=["date", "equity"])
        equity = (series / base) * float(initial_cash)
        return pd.DataFrame({"date": equity.index, "equity": equity.values})

    flow_map = flows.set_index("date")["contribution"].to_dict()
    units = 0.0
    cash = 0.0
    rows: list[dict[str, float | pd.Timestamp]] = []
    for dt, px in series.items():
        contribution = float(flow_map.get(pd.Timestamp(dt), 0.0))
        if contribution != 0.0:
            cash += contribution
        price = float(px)
        if cash > 0.0 and price > 0.0:
            units += cash / price
            cash = 0.0
        equity = (units * price) + cash
        rows.append({"date": pd.Timestamp(dt), "equity": float(equity)})
    return pd.DataFrame(rows)


def _build_excess_curve(equity_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty or benchmark_df.empty:
        return pd.DataFrame(columns=["date", "cum_excess", "relative_ratio"])

    left = equity_df[["date", "equity"]].copy()
    right = benchmark_df[["date", "equity"]].copy().rename(columns={"equity": "bench_equity"})
    merged = left.merge(right, on="date", how="inner").sort_values("date")
    if merged.empty:
        return pd.DataFrame(columns=["date", "cum_excess", "relative_ratio"])

    strat_ret = pd.to_numeric(merged["equity"], errors="coerce").pct_change().fillna(0.0)
    bench_ret = pd.to_numeric(merged["bench_equity"], errors="coerce").pct_change().fillna(0.0)

    merged["cum_excess"] = (strat_ret - bench_ret).cumsum()
    merged["relative_ratio"] = (merged["equity"] / merged["bench_equity"].replace(0.0, np.nan)) - 1.0
    return merged[["date", "cum_excess", "relative_ratio"]]


def _build_snapshot_frame(
    universe_frame: pd.DataFrame,
    ranked: pd.DataFrame,
    buy_mask: pd.Series,
    selected: list[str],
    weights: dict[str, float],
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    if universe_frame is None or universe_frame.empty:
        return pd.DataFrame()

    out = universe_frame.copy()
    out.index = out.index.astype(str)
    out["in_universe"] = True
    out["filter_pass"] = buy_mask.reindex(out.index).fillna(False).astype(bool)

    out["rank_score"] = np.nan
    out["rank"] = np.nan
    if ranked is not None and not ranked.empty:
        out.loc[ranked.index, "rank_score"] = pd.to_numeric(ranked["rank_score"], errors="coerce")
        if "rank" in ranked.columns:
            out.loc[ranked.index, "rank"] = pd.to_numeric(ranked["rank"], errors="coerce")

    out["selected"] = out.index.isin(selected)
    out["target_weight"] = [float(weights.get(sym, 0.0)) for sym in out.index]
    asof = coerce_series_naive(out.get("asof_statement_date"))
    out["asof_statement_date"] = asof
    out["lookahead_ok"] = asof.isna() | (asof <= pd.Timestamp(signal_date))
    src_accept = (
        coerce_series_naive(out["source_acceptance_datetime"])
        if "source_acceptance_datetime" in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    src_filing = (
        coerce_series_naive(out["source_filing_date"])
        if "source_filing_date" in out.columns
        else pd.Series(pd.NaT, index=out.index)
    )
    src_date = src_accept.fillna(src_filing)
    out["sector_source_date"] = src_date
    out["sector_lookahead_ok"] = src_date.isna() | (src_date <= pd.Timestamp(signal_date))
    out["lookahead_ok"] = out["lookahead_ok"] & out["sector_lookahead_ok"]

    cols = [
        "in_universe",
        "filter_pass",
        "rank_score",
        "rank",
        "selected",
        "target_weight",
        "asof_statement_date",
        "sector_source_date",
        "sector_lookahead_ok",
        "lookahead_ok",
    ]
    extra_cols_by_name: list[str] = []
    for name in [
        "sec_sic",
        "sec_sic_description",
        "sector_l1_kr",
        "sector_l2_kr",
        "sector_l1_en",
        "sector_l2_en",
        "mapping_source",
        "mapping_confidence",
        "rule_id",
        "mapping_version",
        "source_filing_date",
        "source_acceptance_datetime",
        "sector_valid_from",
        "sector_valid_to",
        "sp500_pit_member",
        "sp500_pit_included",
        "sp500_pit_source",
        "sp500_pit_source_tier",
        "sp500_pit_source_name",
        "sp500_pit_source_ref",
        "sp500_pit_source_url",
        "sp500_pit_source_doc_id",
        "sp500_pit_evidence_text",
        "sp500_pit_provenance",
        "sp500_pit_confidence",
        "sp500_pit_valid_from",
        "sp500_pit_valid_to",
        "sp500_pit_effective_date",
        "sp500_pit_announcement_date",
        "updated_at",
        "note",
    ]:
        if name in out.columns:
            extra_cols_by_name.append(name)
    extra_metric_cols: list[str] = []
    for col in out.columns:
        if col in cols or col in extra_cols_by_name:
            continue
        if col in {"in_universe", "filter_pass", "rank_score", "rank", "selected", "target_weight", "asof_statement_date", "lookahead_ok"}:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            extra_metric_cols.append(col)
    cols.extend(extra_cols_by_name)
    cols.extend(sorted(set(extra_metric_cols)))
    out = out[cols].reset_index().rename(columns={"index": "ticker", "Ticker": "ticker"})
    if "ticker" not in out.columns:
        # Fallback for unexpected index name casing.
        first_col = out.columns[0]
        out = out.rename(columns={first_col: "ticker"})
    return out


def run_backtest(
    config: StrategyConfig,
    panel: pd.DataFrame | None = None,
    progress_callback: "Callable[[float, str, list], None] | None" = None,
    loading_callback: "Callable[[int, int, str], None] | None" = None,
) -> BacktestResult:
    if panel is None or panel.empty:
        symbols = _resolve_panel_symbols(config)
        panel_asof_mode = "available_date" if bool(config.use_fundamentals_pit) else config.asof_mode
        panel = build_factor_panel(
            symbols=symbols,
            market=config.market,
            start=config.start,
            end=config.end,
            asof_mode=panel_asof_mode,
            use_next_trading_day_availability=bool(config.use_next_trading_day_availability),
            availability_fallback=bool(config.fundamentals_availability_fallback),
            fallback_q_days=int(config.fundamentals_fallback_q_days),
            fallback_k_days=int(config.fundamentals_fallback_k_days),
            offline_mode=getattr(config, "offline_mode", False),
            loading_callback=loading_callback,
        )

    if panel is None or panel.empty:
        empty_df = pd.DataFrame()
        return BacktestResult(
            name=config.name,
            mode=config.mode,
            metrics=normalize_metrics(
                {
                "cagr": 0.0,
                "mdd": 0.0,
                "sharpe": 0.0,
                "vol": 0.0,
                "turnover": 0.0,
                "hit_rate": 0.0,
                "total_return": 0.0,
                "twr_total_return": 0.0,
                "twr_cagr": 0.0,
                "mwr_irr": np.nan,
                "total_contributed": 0.0,
                "ending_value": 0.0,
                "pnl": 0.0,
                "trades": 0,
                "fundamentals_pit_enabled": bool(config.use_fundamentals_pit),
                "asof_lag_trading_days": int(max(0, int(config.asof_lag_trading_days))),
                "use_next_trading_day_availability": bool(config.use_next_trading_day_availability),
                "availability_fallback_enabled": bool(config.fundamentals_availability_fallback),
                "fallback_used_rows": 0,
                "excluded_due_to_missing_fundamentals": 0,
                }
            ),
            equity_curve=empty_df,
            drawdown_curve=empty_df,
            trades=empty_df,
            holdings=empty_df,
            yearly_returns=empty_df,
            benchmark_curve=empty_df,
            excess_curve=empty_df,
            monthly_returns=empty_df,
            yearly_summary=empty_df,
            rebalance_snapshots_index=empty_df,
            rebalance_log=empty_df,
            funding_flows=empty_df,
            out_dir=Path(config.out_dir).expanduser() if config.out_dir else None,
        )

    panel = panel.sort_index()
    # Ensure index is timezone-naive to prevent comparison errors with config dates.
    if hasattr(panel.index.get_level_values(0), "tz") and panel.index.get_level_values(0).tz is not None:
        new_levels = [
            panel.index.levels[0].tz_localize(None),
            panel.index.levels[1]
        ]
        panel.index = panel.index.set_levels(new_levels)

    dates = pd.DatetimeIndex(panel.index.get_level_values(0).unique()).sort_values()
    if hasattr(dates, "tz") and dates.tz is not None:
        dates = dates.tz_localize(None)

    start_ts = pd.to_datetime(config.start, errors="coerce")
    if pd.notna(start_ts) and hasattr(start_ts, "tz") and start_ts.tz is not None:
        start_ts = start_ts.tz_localize(None)

    end_ts = pd.to_datetime(config.end, errors="coerce")
    if pd.notna(end_ts) and hasattr(end_ts, "tz") and end_ts.tz is not None:
        end_ts = end_ts.tz_localize(None)

    if pd.notna(start_ts):
        dates = dates[dates >= pd.Timestamp(start_ts).normalize()]
    if pd.notna(end_ts):
        dates = dates[dates <= pd.Timestamp(end_ts).normalize()]
    if len(dates) == 0:
        raise ValueError("No trading dates available for the selected period")

    date_level = pd.to_datetime(panel.index.get_level_values(0), errors="coerce")
    if hasattr(date_level, "tz") and date_level.tz is not None:
        date_level = date_level.tz_localize(None)
    mask = (date_level >= dates.min()) & (date_level <= dates.max())
    panel = panel.loc[mask]
    panel = _apply_symbol_validity_window(panel, market=config.market)

    use_sp500_pit = _is_sp500_pit_source(config)
    sp500_pit_df = pd.DataFrame()
    sp500_pit_strict = False
    sp500_pit_min_conf = 0.0
    if use_sp500_pit:
        sp500_pit_df, sp500_pit_strict, sp500_pit_min_conf = _load_sp500_pit_run_context(config)

    close_px = panel["close"].unstack("Ticker").sort_index().ffill()
    open_px = panel["open"].unstack("Ticker").sort_index()
    open_px = open_px.where(open_px.notna(), close_px).ffill()

    exec_timing_cfg = _effective_execution_timing(config)
    config.execution_timing = exec_timing_cfg
    strategy_rebalance_dates = generate_rebalance_dates(close_px.index, freq=config.frequency)
    strategy_exec_map = _build_execution_map(close_px.index, strategy_rebalance_dates, execution_timing=exec_timing_cfg)
    funding_cfg = _effective_funding_config(config)
    config.initial_cash = float(funding_cfg.initial_cash)
    funding_dates = generate_funding_dates(close_px.index, freq=funding_cfg.contribution_freq)
    funding_exec_map = _build_execution_map(close_px.index, funding_dates, execution_timing=exec_timing_cfg)

    positions: dict[str, Position] = {}
    cash = float(funding_cfg.initial_cash)

    buy_rule_expr, ranking_cfg = _compile_indicator_inputs(config)

    run_root = _runtime_snapshot_root(config)
    snapshots_root = run_root / "rebalance_snapshots"
    ensure_dir(snapshots_root)

    equity_rows: list[dict] = []
    holding_rows: list[dict] = []
    trade_rows: list[dict] = []
    rebalance_rows: list[dict] = []
    snapshot_index_rows: list[dict] = []
    funding_rows: list[dict] = []
    cumulative_contributed = float(funding_cfg.initial_cash)

    if len(close_px.index) > 0:
        init_date = pd.Timestamp(close_px.index[0])
        funding_rows.append(
            {
                "date": init_date,
                "step_index": 0,
                "funding_mode": funding_cfg.mode,
                "contribution_freq": funding_cfg.contribution_freq,
                "contribution": float(funding_cfg.initial_cash),
                "raw_contribution": float(funding_cfg.initial_cash),
                "target_value": np.nan,
                "cash_before_funding": 0.0,
                "positions_value_before_funding": 0.0,
                "account_equity_before_funding": 0.0,
                "cumulative_contributed_before": 0.0,
                "cash_after_funding": float(funding_cfg.initial_cash),
                "positions_value_after_funding": 0.0,
                "account_equity_after_funding": float(funding_cfg.initial_cash),
                "cumulative_contributed_after": float(funding_cfg.initial_cash),
                "invested_ratio_after_funding": 0.0,
                "cash_after_rebalance": np.nan,
                "positions_value_after_rebalance": np.nan,
                "account_equity_after_rebalance": np.nan,
                "invested_ratio_after_rebalance": np.nan,
                "had_rebalance_same_day": False,
                "rebalance_exec_timing": "",
                "portfolio_value_before": 0.0,
                "portfolio_value_after": float(funding_cfg.initial_cash),
                "is_rebalance_same_day": False,
                "event_order_tag": "initial_cash",
                "va_min_policy": "",
                "va_min_contribution": np.nan,
                "va_max_contribution": np.nan,
                "floor_applied": False,
                "max_cap_applied": False,
                "cap_applied_min": False,
                "cap_applied_max": False,
                "reason": "initial_cash",
            }
        )
    funding_step_index = 0
    fallback_used_rows_total = 0
    excluded_due_to_missing_fundamentals = 0
    total_days = len(close_px.index)

    for day_idx, day in enumerate(close_px.index):
        ts_day = pd.Timestamp(day)

        funding_event_today = ts_day in funding_exec_map
        rebalance_event_today = ts_day in strategy_exec_map
        contribution_today = 0.0
        funding_row_idx: int | None = None
        if funding_event_today:
            funding_signal_date = pd.Timestamp(funding_exec_map[ts_day])
            funding_signal_row = close_px.loc[funding_signal_date] if funding_signal_date in close_px.index else close_px.loc[:funding_signal_date].iloc[-1]
            portfolio_before = _portfolio_value(funding_signal_row, positions, cash)
            cash_before_funding = float(cash)
            positions_value_before = float(portfolio_before - cash_before_funding)
            cumulative_before = float(cumulative_contributed)
            funding_step_index += 1
            contribution, raw_contribution, target_value, funding_reason = compute_contribution(
                funding_cfg=funding_cfg,
                step_index=funding_step_index,
                rebalance_date=funding_signal_date,
                portfolio_value_before=portfolio_before,
            )
            contribution_today = float(contribution)
            cash += contribution_today
            cash_after_funding = float(cash)
            positions_value_after_funding = float(portfolio_before - cash_before_funding)
            account_equity_after_funding = float(cash_after_funding + positions_value_after_funding)
            cumulative_after = float(cumulative_before + contribution_today)
            cumulative_contributed = cumulative_after
            invested_ratio_after_funding = (
                float(positions_value_after_funding / account_equity_after_funding)
                if account_equity_after_funding > 0.0
                else 0.0
            )
            floor_applied = bool("va_every_rebalance_floor" in str(funding_reason) or "va_capped_to_min" in str(funding_reason))
            max_cap_applied = bool("va_capped_to_max" in str(funding_reason))
            cap_applied_min = floor_applied
            cap_applied_max = max_cap_applied
            va_min_contribution = np.nan
            va_max_contribution = np.nan
            va_min_policy = ""
            if str(funding_cfg.mode).strip().lower() == "va":
                va_min_contribution = float(funding_cfg.va_min_contribution or 0.0)
                va_max_contribution = (
                    float(funding_cfg.va_max_contribution)
                    if funding_cfg.va_max_contribution is not None
                    else np.nan
                )
                va_min_policy = _normalize_va_min_policy(funding_cfg.va_min_policy)
            event_order_tag = "funding_before_rebalance" if rebalance_event_today else "funding_only"
            funding_rows.append(
                {
                    "date": ts_day,
                    "step_index": int(funding_step_index),
                    "funding_mode": funding_cfg.mode,
                    "contribution_freq": funding_cfg.contribution_freq,
                    "contribution": float(contribution_today),
                    "raw_contribution": float(raw_contribution),
                    "target_value": (float(target_value) if target_value is not None else np.nan),
                    "cash_before_funding": float(cash_before_funding),
                    "positions_value_before_funding": float(positions_value_before),
                    "account_equity_before_funding": float(portfolio_before),
                    "cumulative_contributed_before": float(cumulative_before),
                    "cash_after_funding": float(cash_after_funding),
                    "positions_value_after_funding": float(positions_value_after_funding),
                    "account_equity_after_funding": float(account_equity_after_funding),
                    "cumulative_contributed_after": float(cumulative_after),
                    "invested_ratio_after_funding": float(invested_ratio_after_funding),
                    "cash_after_rebalance": np.nan,
                    "positions_value_after_rebalance": np.nan,
                    "account_equity_after_rebalance": np.nan,
                    "invested_ratio_after_rebalance": np.nan,
                    "had_rebalance_same_day": bool(rebalance_event_today),
                    "rebalance_exec_timing": str(config.execution_timing),
                    "portfolio_value_before": float(portfolio_before),
                    "portfolio_value_after": float(portfolio_before + contribution_today),
                    "is_rebalance_same_day": bool(rebalance_event_today),
                    "event_order_tag": event_order_tag,
                    "va_min_policy": va_min_policy,
                    "va_min_contribution": va_min_contribution,
                    "va_max_contribution": va_max_contribution,
                    "floor_applied": floor_applied,
                    "max_cap_applied": max_cap_applied,
                    "cap_applied_min": cap_applied_min,
                    "cap_applied_max": cap_applied_max,
                    "reason": funding_reason,
                }
            )
            funding_row_idx = len(funding_rows) - 1

        if rebalance_event_today:
            signal_date = pd.Timestamp(strategy_exec_map[ts_day])
            fundamentals_asof_date = pd.Timestamp(signal_date)
            if bool(config.use_fundamentals_pit) and int(config.asof_lag_trading_days) > 0:
                fundamentals_asof_date = _shift_trading_day(
                    close_px.index,
                    signal_date,
                    -int(config.asof_lag_trading_days),
                )
            signal_frame = _cross_section(panel, fundamentals_asof_date)
            sp500_membership_excluded = 0
            if use_sp500_pit:
                signal_frame, sp500_membership_excluded = _apply_sp500_pit_asof_filter(
                    signal_frame,
                    signal_date=fundamentals_asof_date,
                    pit_df=sp500_pit_df,
                    strict=sp500_pit_strict,
                    min_confidence=sp500_pit_min_conf,
                )
            # Per-rebalance PIT filter diagnostic counts
            pit_initial_count = int(len(signal_frame))
            pit_future_excluded_count = 0
            pit_missing_excluded_count = 0
            if bool(config.use_fundamentals_pit) and not signal_frame.empty:
                if "asof_available_date" in signal_frame.columns:
                    asof_avail = coerce_series_naive(signal_frame["asof_available_date"])
                else:
                    asof_avail = pd.Series(pd.NaT, index=signal_frame.index, dtype="datetime64[ns]")

                # Check for BOTH missing (NaT) AND future availability (look-ahead)
                missing_mask = asof_avail.isna()
                future_mask = asof_avail > fundamentals_asof_date
                pit_future_excluded_count = int(future_mask.sum())
                pit_missing_excluded_count = int(missing_mask.sum())

                if getattr(config, "strict_pit", False):
                    bad_mask = missing_mask | future_mask
                    bad_reason = "missing or not yet available"
                else:
                    # Allow missing (NaT) data but strictly exclude future (look-ahead)
                    bad_mask = future_mask
                    bad_reason = "not yet available (look-ahead)"

                bad_cnt = int(bad_mask.sum())
                if bad_cnt > 0:
                    excluded_due_to_missing_fundamentals += bad_cnt
                    policy = str(config.fundamentals_missing_policy or "exclude").strip().lower()
                    if policy == "error":
                        raise RuntimeError(
                            f"fundamentals PIT enabled but financial data is {bad_reason} "
                            f"on asof_date={fundamentals_asof_date.date().isoformat()} bad_count={bad_cnt}"
                        )
                    signal_frame = signal_frame.loc[~bad_mask].copy()
                if "availability_method" in signal_frame.columns:
                    fallback_used_rows_total += int(
                        signal_frame["availability_method"]
                        .astype(str)
                        .str.lower()
                        .str.startswith("fallback")
                        .sum()
                    )
            sector_future_excluded = 0
            if not signal_frame.empty:
                src_accept = (
                    coerce_series_naive(signal_frame["source_acceptance_datetime"])
                    if "source_acceptance_datetime" in signal_frame.columns
                    else pd.Series(pd.NaT, index=signal_frame.index)
                )
                src_filing = (
                    coerce_series_naive(signal_frame["source_filing_date"])
                    if "source_filing_date" in signal_frame.columns
                    else pd.Series(pd.NaT, index=signal_frame.index)
                )
                src_date = src_accept.fillna(src_filing)
                bad_sector = src_date.notna() & (src_date > fundamentals_asof_date)
                sector_future_excluded = int(bad_sector.sum())
                if sector_future_excluded > 0:
                    if getattr(config, "strict_pit", False):
                        raise RuntimeError(
                            f"Sector proxy lookahead detected for {sector_future_excluded} symbols "
                            f"on signal_date={fundamentals_asof_date.date().isoformat()} in strict mode."
                        )
                    signal_frame = signal_frame.loc[~bad_sector].copy()
                    LOGGER.warning(
                        "Sector look-ahead guard excluded %d symbols on %s",
                        sector_future_excluded,
                        fundamentals_asof_date.date().isoformat(),
                    )
            selected, ranked, sell_reasons, sell_reason_details, buy_reason_details, universe_frame, buy_mask = _pick_targets(
                frame_t=signal_frame,
                config=config,
                positions=positions,
                buy_rule_expr=buy_rule_expr,
                ranking_cfg=ranking_cfg,
                panel=panel,
                signal_date=fundamentals_asof_date,
            )

            weights = _target_weights(
                selected=selected,
                ranked=ranked,
                sizing=config.sizing,
                position_weight_pct=config.position_sizing.position_weight_pct,
            )
            signal_close_row = (
                close_px.loc[signal_date]
                if signal_date in close_px.index
                else close_px.loc[:signal_date].iloc[-1]
            )
            signal_equity = _portfolio_value(signal_close_row, positions, cash)

            exec_cfg = config.execution
            exec_timing = _effective_execution_timing(config)
            exec_basis = str(exec_cfg.price_basis or "auto").strip().lower()
            exec_offset_pct = float(exec_cfg.price_offset_pct or 0.0)
            exec_row, exec_basis_used, exec_offset_used = _resolve_execution_price_row(
                signal_date=signal_date,
                exec_date=ts_day,
                timing=exec_timing,
                price_basis=exec_basis,
                price_offset_pct=exec_offset_pct,
                open_px=open_px,
                close_px=close_px,
            )
            comm_bps = float(config.costs.commission_bps)
            slip_bps = float(config.costs.slippage_bps)
            if exec_cfg.commission_pct is not None:
                comm_bps = float(exec_cfg.commission_pct) * 100.0
            if exec_cfg.slippage_pct is not None:
                slip_bps = float(exec_cfg.slippage_pct) * 100.0
            cash_before_rebalance = float(cash)
            cash, trades = _execute_rebalance(
                signal_date=signal_date,
                exec_date=ts_day,
                positions=positions,
                cash=cash,
                exec_row=exec_row,
                target_weights=weights,
                signal_equity=signal_equity,
                sell_reasons=sell_reasons,
                sell_reason_details=sell_reason_details,
                buy_reason_details=buy_reason_details,
                commission_bps=comm_bps,
                slippage_bps=slip_bps,
                share_mode=config.share_mode,
                cash_buffer_pct=config.cash_buffer_pct,
                min_trade_notional=config.min_trade_notional,
                max_new_buys_per_day=config.position_sizing.max_new_buys_per_day,
                max_buy_amount_per_position=config.position_sizing.max_buy_amount_per_position,
                min_cash_reserve_pct=config.risk_limits.min_cash_reserve_pct / 100.0,
                execution_timing=exec_timing,
                price_basis=exec_basis_used,
                price_offset_pct=exec_offset_used,
            )
            exec_equity_after = _portfolio_value(exec_row, positions, cash)
            exec_positions_after = float(exec_equity_after - cash)
            invested_ratio_after_rebalance = (
                float(exec_positions_after / exec_equity_after)
                if exec_equity_after > 0.0
                else 0.0
            )
            if funding_row_idx is not None and 0 <= funding_row_idx < len(funding_rows):
                funding_rows[funding_row_idx]["cash_after_rebalance"] = float(cash)
                funding_rows[funding_row_idx]["positions_value_after_rebalance"] = float(exec_positions_after)
                funding_rows[funding_row_idx]["account_equity_after_rebalance"] = float(exec_equity_after)
                funding_rows[funding_row_idx]["invested_ratio_after_rebalance"] = float(invested_ratio_after_rebalance)
                funding_rows[funding_row_idx]["had_rebalance_same_day"] = True
                funding_rows[funding_row_idx]["rebalance_exec_timing"] = str(config.execution_timing)

            for t in trades:
                trade_rows.append(
                    {
                        "signal_date": pd.Timestamp(t.signal_date),
                        "exec_date": pd.Timestamp(t.exec_date),
                        "ticker": t.ticker,
                        "side": t.side,
                        "shares": t.shares,
                        "exec_price": t.exec_price,
                        "notional": t.notional,
                        "gross_notional": t.notional,
                        "net_notional": (t.notional - t.commission - t.slippage) if t.side == "sell" else (t.notional + t.commission + t.slippage),
                        "commission": t.commission,
                        "slippage": t.slippage,
                        "cash_delta": t.cash_delta,
                        "execution_timing": exec_timing,
                        "price_basis": exec_basis_used,
                        "price_offset_pct": exec_offset_used,
                        "reason": t.reason,
                        "reason_short": t.reason,
                        "reason_detail": t.reason_detail or t.reason,
                        "before_shares": t.before_shares,
                        "after_shares": t.after_shares,
                        "realized_pnl": t.realized_pnl,
                    }
                )

            def _uniq_tickers(values: list[str]) -> list[str]:
                seen: set[str] = set()
                out: list[str] = []
                for v in values:
                    s = str(v).strip().upper()
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    out.append(s)
                return out

            buy_tickers = _uniq_tickers(
                [t.ticker for t in trades if str(t.side).strip().lower() == "buy" and float(t.shares) > 0.0]
            )
            sell_tickers = _uniq_tickers(
                [t.ticker for t in trades if str(t.side).strip().lower() == "sell" and float(t.shares) > 0.0]
            )
            holding_tickers = _uniq_tickers(
                [sym for sym, pos in positions.items() if float(pos.shares) > 1e-12]
            )
            holding_weights: dict[str, float] = {}
            if float(exec_equity_after) > 0.0:
                for sym in holding_tickers:
                    pos = positions.get(sym)
                    if pos is None or float(pos.shares) <= 1e-12:
                        continue
                    px = pd.to_numeric(exec_row.get(sym), errors="coerce")
                    if pd.isna(px):
                        continue
                    w = float(float(pos.shares) * float(px) / float(exec_equity_after))
                    if np.isfinite(w) and w > 0.0:
                        holding_weights[sym] = w

            max_asof = pd.NaT
            lookahead_ok = True
            if not ranked.empty and "asof_statement_date" in ranked.columns:
                asof = coerce_series_naive(ranked["asof_statement_date"])
                if not asof.dropna().empty:
                    max_asof = pd.Timestamp(asof.max())
                    if max_asof.tz is not None:
                        max_asof = max_asof.tz_localize(None)
                    lookahead_ok = bool(max_asof <= fundamentals_asof_date)
            sector_source_max = pd.NaT
            sector_lookahead_ok = True
            if not ranked.empty:
                src_accept = (
                    coerce_series_naive(ranked["source_acceptance_datetime"])
                    if "source_acceptance_datetime" in ranked.columns
                    else pd.Series(pd.NaT, index=ranked.index)
                )
                src_filing = (
                    coerce_series_naive(ranked["source_filing_date"])
                    if "source_filing_date" in ranked.columns
                    else pd.Series(pd.NaT, index=ranked.index)
                )
                src_date = src_accept.fillna(src_filing)
                if not src_date.dropna().empty:
                    sector_source_max = pd.Timestamp(src_date.max())
                    sector_lookahead_ok = bool(sector_source_max <= fundamentals_asof_date)
            lookahead_ok = bool(lookahead_ok and sector_lookahead_ok)

            rebalance_rows.append(
                {
                    "date": ts_day,
                    "signal_date": signal_date,
                    "fundamentals_asof_date": fundamentals_asof_date,
                    "exec_date": ts_day,
                    "mode": config.mode,
                    "buys": buy_tickers,
                    "sells": sell_tickers,
                    "holdings": holding_tickers,
                    "holding_weights": holding_weights,
                    "universe_size": int(len(universe_frame)),
                    "pit_initial_count": pit_initial_count,
                    "pit_future_excluded_count": pit_future_excluded_count,
                    "pit_missing_excluded_count": pit_missing_excluded_count,
                    "universe_count": int(len(universe_frame)),
                    "candidate_count": int(int(buy_mask.sum()) if len(buy_mask) > 0 else 0),
                    "selected_count": int(len(selected)),
                    "asof_max_statement_date": max_asof,
                    "sector_source_max_date": sector_source_max,
                    "sector_lookahead_ok": bool(sector_lookahead_ok),
                    "sector_future_excluded_count": int(sector_future_excluded),
                    "sp500_pit_membership_excluded_count": int(sp500_membership_excluded),
                    "sp500_pit_strict_mode": bool(sp500_pit_strict),
                    "lookahead_ok": lookahead_ok,
                    "benchmark": config.benchmark,
                    "funding_contribution": float(contribution_today),
                    "cash_before_rebalance": float(cash_before_rebalance),
                    "cash_after_rebalance": float(cash),
                    "had_funding_same_day": bool(funding_event_today),
                }
            )

            if progress_callback is not None:
                _cb_trades = [
                    {"ticker": t.ticker, "side": t.side, "price": t.exec_price, "shares": t.shares}
                    for t in trades
                ]
                progress_callback(
                    (day_idx + 1) / total_days if total_days > 0 else 1.0,
                    ts_day.date().isoformat(),
                    _cb_trades,
                )

            if snapshots_root is not None:
                snap = _build_snapshot_frame(
                    universe_frame=universe_frame,
                    ranked=ranked,
                    buy_mask=buy_mask,
                    selected=selected,
                    weights=weights,
                    signal_date=signal_date,
                )
                snap_name = f"{signal_date.date().isoformat()}.csv"
                snap_path = snapshots_root / snap_name
                snap.to_csv(snap_path, index=False)
                snapshot_index_rows.append({"signal_date": signal_date, "snapshot_path": str(snap_path)})

        close_row = close_px.loc[ts_day]
        equity = _portfolio_value(close_row, positions, cash)
        positions_value = equity - cash

        equity_rows.append(
            {
                "date": ts_day,
                "cash": float(cash),
                "positions_value": float(positions_value),
                "equity": float(equity),
            }
        )

        if positions_value > 0:
            for sym, pos in sorted(positions.items()):
                if pos.shares <= 0:
                    continue
                px = pd.to_numeric(close_row.get(sym), errors="coerce")
                if pd.isna(px):
                    continue
                value = float(pos.shares * float(px))
                weight = float(value / equity) if equity > 0 else 0.0
                holding_rows.append(
                    {
                        "date": ts_day,
                        "ticker": sym,
                        "shares": float(pos.shares),
                        "price": float(px),
                        "value": value,
                        "weight": weight,
                    }
                )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)
    holdings_df = pd.DataFrame(holding_rows)
    rebalance_df = pd.DataFrame(rebalance_rows)
    snapshots_index_df = pd.DataFrame(snapshot_index_rows)
    funding_df = pd.DataFrame(funding_rows)

    if not equity_df.empty:
        equity_df["date"] = pd.to_datetime(equity_df["date"], errors="coerce")
        equity_df = equity_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        eq_idx = equity_df.set_index("date")
    else:
        eq_idx = pd.DataFrame(columns=["equity"])

    if not funding_df.empty:
        funding_df["date"] = pd.to_datetime(funding_df.get("date"), errors="coerce")
        freq_series = funding_df.get("contribution_freq")
        if freq_series is None:
            freq_series = pd.Series("", index=funding_df.index)
        order_series = funding_df.get("event_order_tag")
        if order_series is None:
            order_series = pd.Series("", index=funding_df.index)
        same_day_series = funding_df.get("is_rebalance_same_day")
        if same_day_series is None:
            same_day_series = pd.Series(False, index=funding_df.index)
        funding_df["contribution_freq"] = freq_series.astype(str)
        funding_df["contribution"] = pd.to_numeric(funding_df.get("contribution"), errors="coerce").fillna(0.0)
        funding_df["raw_contribution"] = pd.to_numeric(funding_df.get("raw_contribution"), errors="coerce").fillna(0.0)
        funding_df["cash_before_funding"] = pd.to_numeric(funding_df.get("cash_before_funding"), errors="coerce")
        funding_df["positions_value_before_funding"] = pd.to_numeric(funding_df.get("positions_value_before_funding"), errors="coerce")
        funding_df["account_equity_before_funding"] = pd.to_numeric(funding_df.get("account_equity_before_funding"), errors="coerce")
        funding_df["cumulative_contributed_before"] = pd.to_numeric(funding_df.get("cumulative_contributed_before"), errors="coerce")
        funding_df["cash_after_funding"] = pd.to_numeric(funding_df.get("cash_after_funding"), errors="coerce")
        funding_df["positions_value_after_funding"] = pd.to_numeric(funding_df.get("positions_value_after_funding"), errors="coerce")
        funding_df["account_equity_after_funding"] = pd.to_numeric(funding_df.get("account_equity_after_funding"), errors="coerce")
        funding_df["cumulative_contributed_after"] = pd.to_numeric(funding_df.get("cumulative_contributed_after"), errors="coerce")
        funding_df["cash_after_rebalance"] = pd.to_numeric(funding_df.get("cash_after_rebalance"), errors="coerce")
        funding_df["positions_value_after_rebalance"] = pd.to_numeric(funding_df.get("positions_value_after_rebalance"), errors="coerce")
        funding_df["account_equity_after_rebalance"] = pd.to_numeric(funding_df.get("account_equity_after_rebalance"), errors="coerce")
        funding_df["invested_ratio_after_funding"] = pd.to_numeric(funding_df.get("invested_ratio_after_funding"), errors="coerce")
        funding_df["invested_ratio_after_rebalance"] = pd.to_numeric(funding_df.get("invested_ratio_after_rebalance"), errors="coerce")
        had_reb_series = funding_df.get("had_rebalance_same_day")
        if had_reb_series is None:
            had_reb_series = pd.Series(False, index=funding_df.index)
        funding_df["had_rebalance_same_day"] = had_reb_series.fillna(False).astype(bool)
        reb_exec_series = funding_df.get("rebalance_exec_timing")
        if reb_exec_series is None:
            reb_exec_series = pd.Series("", index=funding_df.index)
        funding_df["rebalance_exec_timing"] = reb_exec_series.astype(str)
        funding_df["is_rebalance_same_day"] = same_day_series.fillna(False).astype(bool)
        funding_df["event_order_tag"] = order_series.astype(str)
        funding_df["va_min_contribution"] = pd.to_numeric(funding_df.get("va_min_contribution"), errors="coerce")
        funding_df["va_max_contribution"] = pd.to_numeric(funding_df.get("va_max_contribution"), errors="coerce")
        floor_series = funding_df.get("floor_applied")
        if floor_series is None:
            floor_series = pd.Series(False, index=funding_df.index)
        max_cap_series = funding_df.get("max_cap_applied")
        if max_cap_series is None:
            max_cap_series = pd.Series(False, index=funding_df.index)
        cap_min_series = funding_df.get("cap_applied_min")
        if cap_min_series is None:
            cap_min_series = pd.Series(False, index=funding_df.index)
        cap_max_series = funding_df.get("cap_applied_max")
        if cap_max_series is None:
            cap_max_series = pd.Series(False, index=funding_df.index)
        funding_df["floor_applied"] = floor_series.fillna(False).astype(bool)
        funding_df["max_cap_applied"] = max_cap_series.fillna(False).astype(bool)
        funding_df["cap_applied_min"] = cap_min_series.fillna(False).astype(bool)
        funding_df["cap_applied_max"] = cap_max_series.fillna(False).astype(bool)
        funding_df = funding_df.dropna(subset=["date"]).sort_values(["date", "step_index"]).reset_index(drop=True)
    else:
        funding_df = pd.DataFrame(
            columns=[
                "date",
                "step_index",
                "funding_mode",
                "contribution_freq",
                "contribution",
                "raw_contribution",
                "target_value",
                "cash_before_funding",
                "positions_value_before_funding",
                "account_equity_before_funding",
                "cumulative_contributed_before",
                "cash_after_funding",
                "positions_value_after_funding",
                "account_equity_after_funding",
                "cumulative_contributed_after",
                "cash_after_rebalance",
                "positions_value_after_rebalance",
                "account_equity_after_rebalance",
                "invested_ratio_after_funding",
                "invested_ratio_after_rebalance",
                "had_rebalance_same_day",
                "rebalance_exec_timing",
                "portfolio_value_before",
                "portfolio_value_after",
                "is_rebalance_same_day",
                "event_order_tag",
                "va_min_policy",
                "va_min_contribution",
                "va_max_contribution",
                "floor_applied",
                "max_cap_applied",
                "cap_applied_min",
                "cap_applied_max",
                "reason",
            ]
        )

    drawdown_df = compute_drawdown_curve(eq_idx[["equity"]] if "equity" in eq_idx.columns else pd.DataFrame())
    if not drawdown_df.empty:
        drawdown_df = drawdown_df.reset_index().rename(columns={"index": "date"})

    benchmark_curve = _load_benchmark_equity(
        benchmark=config.benchmark,
        market=config.market,
        trading_index=eq_idx.index if not eq_idx.empty else pd.DatetimeIndex([]),
        initial_cash=float(funding_cfg.initial_cash),
        funding_flows=funding_df,
    )
    if not benchmark_curve.empty:
        benchmark_curve = benchmark_curve.sort_values("date").reset_index(drop=True)
        bench_idx = benchmark_curve.set_index("date")
    else:
        bench_idx = pd.DataFrame(columns=["equity"])

    excess_curve = _build_excess_curve(equity_df, benchmark_curve)
    monthly_returns = build_monthly_returns(eq_idx, bench_idx)
    yearly_summary = build_yearly_summary(monthly_returns)

    yearly_df = build_yearly_returns(eq_idx[["equity"]] if "equity" in eq_idx.columns else pd.DataFrame())
    metrics = normalize_metrics(
        compute_metrics(
            eq_idx[["equity"]] if "equity" in eq_idx.columns else pd.DataFrame(),
            trades_df,
            funding_flows=funding_df,
        )
    )
    metrics["fundamentals_pit_enabled"] = bool(config.use_fundamentals_pit)
    metrics["asof_lag_trading_days"] = int(max(0, int(config.asof_lag_trading_days)))
    metrics["use_next_trading_day_availability"] = bool(config.use_next_trading_day_availability)
    metrics["availability_fallback_enabled"] = bool(config.fundamentals_availability_fallback)
    metrics["fallback_used_rows"] = int(fallback_used_rows_total)
    metrics["excluded_due_to_missing_fundamentals"] = int(excluded_due_to_missing_fundamentals)
    if not bool(config.use_fundamentals_pit):
        metrics["warning"] = "fundamentals PIT disabled (look-ahead risk)"

    result = BacktestResult(
        name=config.name,
        mode=config.mode,
        metrics=metrics,
        equity_curve=equity_df,
        drawdown_curve=drawdown_df,
        trades=trades_df,
        holdings=holdings_df,
        yearly_returns=yearly_df,
        benchmark_curve=benchmark_curve,
        excess_curve=excess_curve,
        monthly_returns=monthly_returns,
        yearly_summary=yearly_summary,
        rebalance_snapshots_index=snapshots_index_df,
        rebalance_log=rebalance_df,
        funding_flows=funding_df,
        out_dir=run_root,
    )
    return result


def run_screen_backtest(
    screen_expr: str,
    universe_source: str = "symbols",
    freq: str = "Q",
    start: str = "2000-01-01",
    end: str | None = None,
    holdings: int = 3,
    sizing: str = "equal",
    market: str = "us",
    out_dir: str | None = None,
    ranking: RankingConfig | None = None,
    symbols: list[str] | None = None,
    benchmark: str = "SPY",
    execution_timing: str = "next_open",
    use_fundamentals_pit: bool = True,
    asof_lag_trading_days: int = 0,
    use_next_trading_day_availability: bool = False,
    fundamentals_availability_fallback: bool = True,
    fundamentals_missing_policy: str = "exclude",
    fundamentals_fallback_q_days: int = 45,
    fundamentals_fallback_k_days: int = 90,
    offline_mode: bool = False,
    strict: bool = False,
) -> BacktestResult:
    cfg = StrategyConfig(
        name="screen_backtest",
        mode="screen",
        market=market,
        start=start,
        end=end,
        frequency=freq,
        holdings=max(1, int(holdings)),
        sizing=sizing,
        buy_rules=screen_expr,
        sell_mode="A",
        ranking=ranking or _default_ranking_for_screen(),
        universe=UniverseConfig(
            source=universe_source,
            symbols=symbols or [],
            market=market,
            sp500_pit_strict=strict,
            sp500_pit_fail_closed=strict,
        ),
        benchmark=benchmark,
        execution_timing=execution_timing,
        use_fundamentals_pit=bool(use_fundamentals_pit),
        asof_lag_trading_days=max(0, int(asof_lag_trading_days)),
        use_next_trading_day_availability=bool(use_next_trading_day_availability),
        fundamentals_availability_fallback=bool(fundamentals_availability_fallback),
        fundamentals_missing_policy=str(fundamentals_missing_policy or "exclude").strip().lower() or "exclude",
        fundamentals_fallback_q_days=max(0, int(fundamentals_fallback_q_days)),
        fundamentals_fallback_k_days=max(0, int(fundamentals_fallback_k_days)),
        out_dir=out_dir,
        offline_mode=offline_mode,
        strict_pit=strict,
    )
    
    # If source is sp500 or sp500_pit, let _resolve_panel_symbols determine the initial symbols.
    resolved_symbols = _resolve_panel_symbols(cfg) if universe_source in ["sp500", "sp500_pit"] else symbols

    panel_asof_mode = "available_date" if bool(cfg.use_fundamentals_pit) else cfg.asof_mode
    panel = load_factor_panel_from_local(
        market=market,
        start=start,
        end=end,
        asof_mode=panel_asof_mode,
        symbols=resolved_symbols,
        use_next_trading_day_availability=bool(cfg.use_next_trading_day_availability),
        availability_fallback=bool(cfg.fundamentals_availability_fallback),
        fallback_q_days=int(cfg.fundamentals_fallback_q_days),
        fallback_k_days=int(cfg.fundamentals_fallback_k_days),
        offline_mode=offline_mode,
    )
    result = run_backtest(cfg, panel=panel)
    if out_dir:
        save_backtest_outputs(result, out_dir)
    return result


def run_strategy_backtest_from_config(config_path: str | Path, offline_mode_override: bool | None = None) -> BacktestResult:
    cfg = load_strategy_config(config_path)
    if offline_mode_override is not None:
        cfg.offline_mode = offline_mode_override
    symbols = _resolve_panel_symbols(cfg)
    panel_asof_mode = "available_date" if bool(cfg.use_fundamentals_pit) else cfg.asof_mode
    panel = build_factor_panel(
        symbols=symbols,
        market=cfg.market,
        start=cfg.start,
        end=cfg.end,
        asof_mode=panel_asof_mode,
        use_next_trading_day_availability=bool(cfg.use_next_trading_day_availability),
        availability_fallback=bool(cfg.fundamentals_availability_fallback),
        fallback_q_days=int(cfg.fundamentals_fallback_q_days),
        fallback_k_days=int(cfg.fundamentals_fallback_k_days),
        offline_mode=getattr(cfg, "offline_mode", False),
    )
    result = run_backtest(cfg, panel=panel)
    if cfg.out_dir:
        save_backtest_outputs(result, cfg.out_dir)
    return result


def run_funding_comparison(
    base_strategy_cfg: StrategyConfig,
    funding_variants: list[FundingConfig],
    panel: pd.DataFrame | None = None,
    alignment_mode: str = "default",
    custom_total: float | None = None,
    return_meta: bool = False,
) -> dict[str, BacktestResult] | tuple[dict[str, BacktestResult], dict[str, Any]]:
    if not funding_variants:
        empty_meta = {
            "alignment_mode": _normalize_funding_alignment_mode(alignment_mode),
            "alignment_applied": False,
            "aligned_lump_initial_cash": np.nan,
            "dca_event_count": 0,
            "dca_fixed_contribution": 0.0,
            "dca_total_contributed": np.nan,
            "note": "",
        }
        return ({}, empty_meta) if return_meta else {}

    shared_panel = panel
    if shared_panel is None or shared_panel.empty:
        symbols = _resolve_panel_symbols(base_strategy_cfg)
        panel_asof_mode = "available_date" if bool(base_strategy_cfg.use_fundamentals_pit) else base_strategy_cfg.asof_mode
        shared_panel = build_factor_panel(
            symbols=symbols,
            market=base_strategy_cfg.market,
            start=base_strategy_cfg.start,
            end=base_strategy_cfg.end,
            asof_mode=panel_asof_mode,
            use_next_trading_day_availability=bool(base_strategy_cfg.use_next_trading_day_availability),
            availability_fallback=bool(base_strategy_cfg.fundamentals_availability_fallback),
            fallback_q_days=int(base_strategy_cfg.fundamentals_fallback_q_days),
            fallback_k_days=int(base_strategy_cfg.fundamentals_fallback_k_days),
        )

    alignment_key = _normalize_funding_alignment_mode(alignment_mode)
    meta: dict[str, Any] = {
        "alignment_mode": alignment_key,
        "alignment_applied": False,
        "aligned_lump_initial_cash": np.nan,
        "dca_event_count": 0,
        "dca_fixed_contribution": 0.0,
        "dca_total_contributed": np.nan,
        "note": "",
    }

    aligned_lump_initial_cash: float | None = None
    dca_variant = next((v for v in funding_variants if _normalize_funding_mode(v.mode) == "dca"), None)
    if dca_variant is not None and shared_panel is not None and not shared_panel.empty:
        trading_dates = pd.DatetimeIndex(shared_panel.index.get_level_values(0).unique()).sort_values()
        dca_freq = normalize_frequency(dca_variant.contribution_freq or base_strategy_cfg.frequency)
        dca_events = count_contribution_events(
            trading_dates=trading_dates,
            contribution_freq=dca_freq,
            execution_timing=base_strategy_cfg.execution_timing,
        )
        dca_fixed = float(dca_variant.fixed_contribution or 0.0)
        dca_total = float(dca_variant.initial_cash) + (dca_fixed * float(dca_events))
        meta["dca_event_count"] = int(dca_events)
        meta["dca_fixed_contribution"] = float(dca_fixed)
        meta["dca_total_contributed"] = float(dca_total)

        if alignment_key == "align_lump_to_dca_total":
            if dca_fixed > 0.0 and dca_events > 0:
                aligned_lump_initial_cash = float(dca_total)
                meta["alignment_applied"] = True
                meta["aligned_lump_initial_cash"] = float(aligned_lump_initial_cash)
                meta["note"] = (
                    f"Lump aligned to DCA total: "
                    f"{dca_total:.2f} = {float(dca_variant.initial_cash):.2f} + {dca_fixed:.2f} x {int(dca_events)}"
                )
            else:
                meta["note"] = "Lump alignment skipped (DCA fixed contribution <= 0 or no contribution events)"
    elif alignment_key == "align_lump_to_dca_total":
        meta["note"] = "Lump alignment skipped (no DCA variant available)"

    if alignment_key == "custom_total":
        custom_value = pd.to_numeric(pd.Series([custom_total]), errors="coerce").iloc[0]
        if pd.notna(custom_value) and float(custom_value) > 0.0:
            aligned_lump_initial_cash = float(custom_value)
            meta["alignment_applied"] = True
            meta["aligned_lump_initial_cash"] = float(aligned_lump_initial_cash)
            meta["note"] = f"Lump aligned to custom total: {float(custom_value):.2f}"
        else:
            meta["note"] = "Lump alignment skipped (invalid custom total)"

    results: dict[str, BacktestResult] = {}
    for variant in funding_variants:
        cfg = deepcopy(base_strategy_cfg)
        cfg.funding = deepcopy(variant)
        mode = _normalize_funding_mode(cfg.funding.mode)
        if mode == "lump_sum" and aligned_lump_initial_cash is not None:
            cfg.funding.initial_cash = float(aligned_lump_initial_cash)
        cfg.initial_cash = float(cfg.funding.initial_cash)
        cfg.name = f"{base_strategy_cfg.name}_{mode}"
        result = run_backtest(cfg, panel=shared_panel)
        result.name = mode
        results[mode] = result
    if return_meta:
        return results, meta
    return results


def run_rank_test(
    ranking_config_path: str | Path,
    universe: str,
    freq: str,
    start: str,
    end: str | None = None,
    market: str = "us",
    tickers_file: str | None = None,
    out_dir: str | None = None,
) -> int:
    ranking = load_ranking_config(ranking_config_path)

    if universe == "local":
        symbols = available_price_symbols(market=market)
    else:
        try:
            symbols, inferred_market, _ = build_universe(
                universe=universe,
                tickers_file=tickers_file,
                kospi_external_url="",
                kospi_top_n=None,
            )
            market = inferred_market
        except Exception:
            symbols = available_price_symbols(market=market)

    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    panel = build_factor_panel(symbols=symbols, market=market, start=start, end=end, asof_mode="quarter_end")
    if panel.empty:
        print("[RANK TEST] no panel rows")
        return 2

    dates = pd.DatetimeIndex(panel.index.get_level_values(0).unique()).sort_values()
    rebalance_dates = generate_rebalance_dates(dates, freq=freq)

    run_root = Path(out_dir).expanduser() if out_dir else Path("logs") / "backtests" / pd.Timestamp.utcnow().strftime("ranktest_%Y%m%d_%H%M%S")
    snap_dir = run_root / "rank_snapshots"
    ensure_dir(snap_dir)

    saved = 0
    for dt in rebalance_dates:
        frame = _cross_section(panel, pd.Timestamp(dt))
        if frame.empty:
            continue
        ranked = rank_cross_section(frame, ranking)
        if ranked.empty:
            continue
        out = ranked.reset_index().rename(columns={"index": "Ticker"})
        out.insert(0, "date", pd.Timestamp(dt))
        out.to_csv(snap_dir / f"{pd.Timestamp(dt).date().isoformat()}.csv", index=False)
        saved += 1

    manifest = {
        "saved_snapshots": saved,
        "start": start,
        "end": end,
        "freq": normalize_frequency(freq),
        "universe": universe,
        "market": market,
        "symbols": len(symbols),
        "snapshot_dir": str(snap_dir),
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RANK TEST] snapshots={saved} dir={snap_dir}")
    return 0 if saved > 0 else 2


def run_strategy_and_screen_compare(
    config_path: str | Path,
    screen_expr: str | None = None,
    screen_result: BacktestResult | None = None,
    strategy_result: BacktestResult | None = None,
    screen_config: StrategyConfig | None = None,
) -> dict[str, pd.DataFrame]:
    strategy = strategy_result if strategy_result is not None else run_strategy_backtest_from_config(config_path)
    if screen_result is not None:
        screen = screen_result
    elif screen_config is not None:
        screen = run_backtest(screen_config)
    else:
        cfg = load_strategy_config(config_path)
        expr = screen_expr if screen_expr is not None else cfg.buy_rules
        screen = run_screen_backtest(
            screen_expr=expr,
            freq=cfg.frequency,
            start=cfg.start,
            end=cfg.end,
            holdings=cfg.holdings,
            sizing=cfg.sizing,
            market=cfg.market,
            ranking=cfg.ranking,
            benchmark=cfg.benchmark,
            execution_timing=cfg.execution_timing,
            use_fundamentals_pit=bool(cfg.use_fundamentals_pit),
            asof_lag_trading_days=int(cfg.asof_lag_trading_days),
            use_next_trading_day_availability=bool(cfg.use_next_trading_day_availability),
            fundamentals_availability_fallback=bool(cfg.fundamentals_availability_fallback),
            fundamentals_missing_policy=str(cfg.fundamentals_missing_policy),
            fundamentals_fallback_q_days=int(cfg.fundamentals_fallback_q_days),
            fundamentals_fallback_k_days=int(cfg.fundamentals_fallback_k_days),
        )
    return compare_results(screen, strategy)
