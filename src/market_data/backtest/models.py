from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json

import pandas as pd


@dataclass
class UniverseConfig:
    source: str = "symbols"  # symbols | screen | sp500 | sp500_pit
    symbols: list[str] = field(default_factory=list)
    symbols_file: str | None = None
    screen: str | None = None
    market: str = "us"
    min_market_cap: float | None = None
    min_dollar_volume_20d: float | None = None
    exclude_adr: bool = False
    exclude_preferred: bool = False
    exclude_distress: bool = False
    include_sectors: list[str] = field(default_factory=list)
    exclude_sectors: list[str] = field(default_factory=list)
    include_subsectors: list[str] = field(default_factory=list)
    exclude_subsectors: list[str] = field(default_factory=list)
    exchanges: list[str] = field(default_factory=list)
    size_buckets: list[str] = field(default_factory=list)
    exclude_tickers: list[str] = field(default_factory=list)
    watchlist_name: str | None = None
    sp500_pit_strict: bool = False
    sp500_pit_fail_closed: bool = False
    sp500_pit_min_confidence: float = 0.0


@dataclass
class UniverseFilterConfig:
    exchanges: list[str] = field(default_factory=list)
    size_buckets: list[str] = field(default_factory=list)
    include_sectors: list[str] = field(default_factory=list)
    exclude_sectors: list[str] = field(default_factory=list)
    include_subsectors: list[str] = field(default_factory=list)
    exclude_subsectors: list[str] = field(default_factory=list)
    exclude_tickers: list[str] = field(default_factory=list)
    watchlist_name: str | None = None
    universe_presets: list[str] = field(default_factory=list)  # sp500 | etf_all | nasdaq_all | nyse_all
    universe_bucket_selections: dict[str, list[str]] = field(default_factory=dict)  # ETF/NASDAQ/NYSE -> buckets
    stock_type_included: bool | None = None
    ptp_included: bool | None = None
    financial_data_only: bool = False


@dataclass
class ConditionClause:
    label: str
    left: str
    op: str
    right: float


@dataclass
class ConditionSet:
    conditions: list[ConditionClause] = field(default_factory=list)
    logical_expression: str = ""
    na_policy: str = "fail"


@dataclass
class PositionSizingConfig:
    # Percent of allocatable equity per position. None keeps strategy defaults.
    position_weight_pct: float | None = None
    max_holdings: int | None = None
    max_new_buys_per_day: int | None = None
    max_buy_amount_per_position: float | None = None


@dataclass
class RiskLimitsConfig:
    min_cash_reserve_pct: float = 0.0


@dataclass
class ExecutionConfig:
    timing: str = "next_open"  # next_open | same_close
    price_basis: str = "auto"  # auto | open | close | prev_close
    price_offset_pct: float = 0.0
    # pct units (0.10 = 0.10%), falls back to costs bps when None.
    commission_pct: float | None = None
    slippage_pct: float | None = None


@dataclass
class FactorSpec:
    name: str
    direction: str = "asc"  # asc=lower better, desc=higher better
    weight: float = 1.0
    winsorize_low: float | None = None
    winsorize_high: float | None = None


@dataclass
class RankingConfig:
    normalization: str = "percentile"
    factors: list[FactorSpec] = field(default_factory=list)


@dataclass
class CostModel:
    commission_bps: float = 0.0
    slippage_bps: float = 0.0


@dataclass
class FundingConfig:
    mode: str = "lump_sum"  # lump_sum | dca | va
    initial_cash: float = 100_000.0
    contribution_freq: str = "M"  # M | Q | Y
    fixed_contribution: float | None = None
    va_target_step: float | None = None
    va_target_base: float = 0.0
    va_min_contribution: float = 0.0
    # positive_raw_only: apply minimum only when raw>0
    # every_rebalance: enforce minimum floor every rebalance
    va_min_policy: str = "positive_raw_only"
    va_max_contribution: float | None = None
    va_allow_withdrawal: bool = False


@dataclass
class StrategyConfig:
    name: str = "strategy"
    mode: str = "strategy"  # strategy | screen
    market: str = "us"
    start: str = "2000-01-01"
    end: str | None = None
    frequency: str = "Q"  # W | M | Q
    holdings: int = 3
    initial_cash: float = 100_000.0
    sizing: str = "equal"  # equal | rank_weight
    buy_rules: str = ""
    sell_rules: str | None = None
    sell_mode: str = "B"  # A(screen) | B(strategy)
    rank_drop_multiplier: float = 2.0
    rule_na_policy: str = "fail"
    ranking: RankingConfig = field(default_factory=RankingConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    costs: CostModel = field(default_factory=CostModel)
    asof_mode: str = "quarter_end"
    use_fundamentals_pit: bool = False
    asof_lag_trading_days: int = 0
    use_next_trading_day_availability: bool = False
    fundamentals_availability_fallback: bool = True
    fundamentals_missing_policy: str = "exclude"  # exclude | error
    fundamentals_fallback_q_days: int = 45
    fundamentals_fallback_k_days: int = 90
    missing_policy: str = "drop"
    execution_timing: str = "next_open"  # next_open | same_close
    benchmark: str = "SPY"
    indicators: list[dict[str, Any]] = field(default_factory=list)
    buy_condition_set: ConditionSet = field(default_factory=ConditionSet)
    sell_condition_set: ConditionSet = field(default_factory=ConditionSet)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    risk_limits: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    universe_filter: UniverseFilterConfig = field(default_factory=UniverseFilterConfig)
    share_mode: str = "fractional"  # fractional | integer
    cash_buffer_pct: float = 0.0
    min_trade_notional: float = 0.0
    out_dir: str | None = None
    funding: FundingConfig = field(default_factory=FundingConfig)
    offline_mode: bool = False
    strict_pit: bool = False


@dataclass
class Trade:
    signal_date: pd.Timestamp
    exec_date: pd.Timestamp
    ticker: str
    side: str
    shares: float
    exec_price: float
    notional: float
    commission: float
    slippage: float
    cash_delta: float
    reason: str
    before_shares: float
    after_shares: float
    realized_pnl: float
    reason_detail: str = ""


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float
    entry_date: pd.Timestamp


@dataclass
class BacktestResult:
    name: str
    mode: str
    # Metrics schema includes both strategy(TWR) and account(equity) viewpoints.
    # TWR: twr_total_return, twr_cagr, twr_mdd, twr_sharpe, twr_vol
    # Account: account_total_return, account_cagr, account_mdd, account_sharpe, account_vol
    # Cashflow-aware investor metrics: mwr_irr, total_contributed, ending_value, pnl
    # Backward-compatible aliases remain: cagr/mdd/sharpe/vol/total_return -> TWR values.
    metrics: dict[str, float | int]
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    yearly_returns: pd.DataFrame
    benchmark_curve: pd.DataFrame
    excess_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    yearly_summary: pd.DataFrame
    rebalance_snapshots_index: pd.DataFrame
    rebalance_log: pd.DataFrame
    funding_flows: pd.DataFrame
    out_dir: Path | None = None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return out


def _to_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except Exception:
        return default
    return out


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_raw_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path).expanduser()
    text = cfg_path.read_text(encoding="utf-8")
    suffix = cfg_path.suffix.lower()

    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON config root must be an object")
        return data

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyYAML is required to read YAML config files") from exc
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("YAML config root must be a mapping")
        return data

    raise ValueError(f"Unsupported config extension: {cfg_path.suffix}")


def ranking_config_from_dict(data: dict[str, Any] | None) -> RankingConfig:
    raw = data or {}
    factors: list[FactorSpec] = []
    for item in raw.get("factors", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        factors.append(
            FactorSpec(
                name=name,
                direction=str(item.get("direction", "asc")).strip().lower() or "asc",
                weight=float(item.get("weight", 1.0)),
                winsorize_low=_to_float(item.get("winsorize_low")),
                winsorize_high=_to_float(item.get("winsorize_high")),
            )
        )
    return RankingConfig(
        normalization=str(raw.get("normalization", "percentile")).strip().lower() or "percentile",
        factors=factors,
    )


def strategy_config_from_dict(data: dict[str, Any]) -> StrategyConfig:
    universe_raw = data.get("universe", {})
    if not isinstance(universe_raw, dict):
        universe_raw = {}

    ranking = ranking_config_from_dict(data.get("ranking") if isinstance(data.get("ranking"), dict) else None)

    costs_raw = data.get("costs", {})
    if not isinstance(costs_raw, dict):
        costs_raw = {}
    funding_raw = data.get("funding", {})
    if not isinstance(funding_raw, dict):
        funding_raw = {}

    universe = UniverseConfig(
        source=str(universe_raw.get("source", "symbols")).strip().lower() or "symbols",
        symbols=[str(s).strip().upper() for s in universe_raw.get("symbols", []) or [] if str(s).strip()],
        symbols_file=universe_raw.get("symbols_file"),
        screen=universe_raw.get("screen"),
        market=str(universe_raw.get("market", data.get("market", "us"))).strip().lower() or "us",
        min_market_cap=_to_float(universe_raw.get("min_market_cap")),
        min_dollar_volume_20d=_to_float(universe_raw.get("min_dollar_volume_20d")),
        exclude_adr=_to_bool(universe_raw.get("exclude_adr"), False),
        exclude_preferred=_to_bool(universe_raw.get("exclude_preferred"), False),
        exclude_distress=_to_bool(universe_raw.get("exclude_distress"), False),
        include_sectors=[str(s).strip() for s in universe_raw.get("include_sectors", []) or [] if str(s).strip()],
        exclude_sectors=[str(s).strip() for s in universe_raw.get("exclude_sectors", []) or [] if str(s).strip()],
        include_subsectors=[str(s).strip() for s in universe_raw.get("include_subsectors", []) or [] if str(s).strip()],
        exclude_subsectors=[str(s).strip() for s in universe_raw.get("exclude_subsectors", []) or [] if str(s).strip()],
        exchanges=[str(s).strip() for s in universe_raw.get("exchanges", []) or [] if str(s).strip()],
        size_buckets=[str(s).strip() for s in universe_raw.get("size_buckets", []) or [] if str(s).strip()],
        exclude_tickers=[
            str(s).strip().upper() for s in universe_raw.get("exclude_tickers", []) or [] if str(s).strip()
        ],
        watchlist_name=(str(universe_raw.get("watchlist_name")).strip() if universe_raw.get("watchlist_name") else None),
        sp500_pit_strict=_to_bool(universe_raw.get("sp500_pit_strict"), False),
        sp500_pit_fail_closed=_to_bool(universe_raw.get("sp500_pit_fail_closed"), False),
        sp500_pit_min_confidence=float(universe_raw.get("sp500_pit_min_confidence", 0.0) or 0.0),
    )

    universe_filter_raw = data.get("universe_filter", {})
    if not isinstance(universe_filter_raw, dict):
        universe_filter_raw = {}
    universe_filter = UniverseFilterConfig(
        exchanges=[
            str(s).strip().upper()
            for s in (universe_filter_raw.get("exchanges", universe.exchanges) or [])
            if str(s).strip()
        ],
        size_buckets=[
            str(s).strip().lower()
            for s in (universe_filter_raw.get("size_buckets", universe.size_buckets) or [])
            if str(s).strip()
        ],
        include_sectors=[
            str(s).strip()
            for s in (universe_filter_raw.get("include_sectors", universe.include_sectors) or [])
            if str(s).strip()
        ],
        exclude_sectors=[
            str(s).strip()
            for s in (universe_filter_raw.get("exclude_sectors", universe.exclude_sectors) or [])
            if str(s).strip()
        ],
        include_subsectors=[
            str(s).strip()
            for s in (universe_filter_raw.get("include_subsectors", universe.include_subsectors) or [])
            if str(s).strip()
        ],
        exclude_subsectors=[
            str(s).strip()
            for s in (universe_filter_raw.get("exclude_subsectors", universe.exclude_subsectors) or [])
            if str(s).strip()
        ],
        exclude_tickers=[
            str(s).strip().upper()
            for s in (universe_filter_raw.get("exclude_tickers", universe.exclude_tickers) or [])
            if str(s).strip()
        ],
        watchlist_name=(
            str(universe_filter_raw.get("watchlist_name")).strip()
            if universe_filter_raw.get("watchlist_name")
            else universe.watchlist_name
        ),
        universe_presets=[
            str(s).strip().lower()
            for s in (universe_filter_raw.get("universe_presets", []) or [])
            if str(s).strip()
        ],
        universe_bucket_selections=(
            {
                str(k).strip().upper(): [str(v).strip().lower() for v in (vals or []) if str(v).strip()]
                for k, vals in (universe_filter_raw.get("universe_bucket_selections") or {}).items()
                if str(k).strip()
            }
            if isinstance(universe_filter_raw.get("universe_bucket_selections"), dict)
            else {}
        ),
        stock_type_included=(
            _to_bool(universe_filter_raw.get("stock_type_included"), True)
            if universe_filter_raw.get("stock_type_included") is not None
            else None
        ),
        ptp_included=(
            _to_bool(universe_filter_raw.get("ptp_included"), False)
            if universe_filter_raw.get("ptp_included") is not None
            else None
        ),
    )

    def _condition_set_from_dict(raw: Any, default_na: str = "fail") -> ConditionSet:
        if not isinstance(raw, dict):
            return ConditionSet()
        clauses: list[ConditionClause] = []
        for item in raw.get("conditions", []) or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip().upper()
            left = str(item.get("left", "")).strip()
            op = str(item.get("op", "")).strip()
            right = _to_float(item.get("right"))
            if not label or not left or op not in {"<", "<=", ">", ">=", "==", "!="} or right is None:
                continue
            clauses.append(ConditionClause(label=label, left=left, op=op, right=float(right)))
        return ConditionSet(
            conditions=clauses,
            logical_expression=str(raw.get("logical_expression", "")).strip(),
            na_policy=str(raw.get("na_policy", default_na)).strip().lower() or default_na,
        )

    buy_condition_set = _condition_set_from_dict(data.get("buy_condition_set"), default_na=str(data.get("rule_na_policy", "fail")))
    sell_condition_set = _condition_set_from_dict(data.get("sell_condition_set"), default_na=str(data.get("rule_na_policy", "fail")))

    position_sizing_raw = data.get("position_sizing", {})
    if not isinstance(position_sizing_raw, dict):
        position_sizing_raw = {}
    position_sizing = PositionSizingConfig(
        position_weight_pct=_to_float(position_sizing_raw.get("position_weight_pct")),
        max_holdings=(
            max(1, int(position_sizing_raw["max_holdings"]))
            if _to_float(position_sizing_raw.get("max_holdings")) is not None
            else None
        ),
        max_new_buys_per_day=(
            max(1, int(position_sizing_raw["max_new_buys_per_day"]))
            if _to_float(position_sizing_raw.get("max_new_buys_per_day")) is not None
            else None
        ),
        max_buy_amount_per_position=_to_float(position_sizing_raw.get("max_buy_amount_per_position")),
    )

    risk_limits_raw = data.get("risk_limits", {})
    if not isinstance(risk_limits_raw, dict):
        risk_limits_raw = {}
    risk_limits = RiskLimitsConfig(
        min_cash_reserve_pct=float(risk_limits_raw.get("min_cash_reserve_pct", 0.0)),
    )

    execution_raw = data.get("execution", {})
    if not isinstance(execution_raw, dict):
        execution_raw = {}
    execution = ExecutionConfig(
        timing=str(execution_raw.get("timing", data.get("execution_timing", "next_open"))).strip().lower() or "next_open",
        price_basis=str(execution_raw.get("price_basis", "auto")).strip().lower() or "auto",
        price_offset_pct=float(execution_raw.get("price_offset_pct", 0.0)),
        commission_pct=_to_float(execution_raw.get("commission_pct")),
        slippage_pct=_to_float(execution_raw.get("slippage_pct")),
    )

    legacy_initial_cash = float(data.get("initial_cash", 100_000.0))
    funding_mode = str(funding_raw.get("mode", "lump_sum")).strip().lower() or "lump_sum"
    if funding_mode not in {"lump_sum", "dca", "va"}:
        funding_mode = "lump_sum"
    contribution_freq = str(funding_raw.get("contribution_freq", data.get("frequency", "Q"))).strip().upper() or "Q"
    if contribution_freq not in {"M", "Q", "Y"}:
        contribution_freq = str(data.get("frequency", "Q")).strip().upper() or "Q"
    if contribution_freq not in {"M", "Q", "Y"}:
        contribution_freq = "Q"

    funding_initial_cash = _to_float(funding_raw.get("initial_cash"))
    if funding_initial_cash is None:
        funding_initial_cash = legacy_initial_cash

    funding = FundingConfig(
        mode=funding_mode,
        initial_cash=float(funding_initial_cash),
        contribution_freq=contribution_freq,
        fixed_contribution=_to_float(funding_raw.get("fixed_contribution")),
        va_target_step=_to_float(funding_raw.get("va_target_step")),
        va_target_base=float(funding_raw.get("va_target_base", 0.0)),
        va_min_contribution=float(funding_raw.get("va_min_contribution", 0.0)),
        va_min_policy=(
            str(funding_raw.get("va_min_policy", "positive_raw_only")).strip().lower() or "positive_raw_only"
        ),
        va_max_contribution=_to_float(funding_raw.get("va_max_contribution")),
        va_allow_withdrawal=_to_bool(funding_raw.get("va_allow_withdrawal"), False),
    )
    if funding.va_min_policy not in {"positive_raw_only", "every_rebalance"}:
        funding.va_min_policy = "positive_raw_only"

    cfg = StrategyConfig(
        name=str(data.get("name", "strategy")).strip() or "strategy",
        mode=str(data.get("mode", "strategy")).strip().lower() or "strategy",
        market=str(data.get("market", universe.market)).strip().lower() or "us",
        start=str(data.get("start", "2000-01-01")),
        end=data.get("end"),
        frequency=str(data.get("frequency", "Q")).strip().upper() or "Q",
        holdings=max(1, _to_int(data.get("holdings", 3), 3)),
        initial_cash=legacy_initial_cash,
        sizing=str(data.get("sizing", "equal")).strip().lower() or "equal",
        buy_rules=str(data.get("buy_rules", "")).strip(),
        sell_rules=(str(data["sell_rules"]).strip() if data.get("sell_rules") is not None else None),
        sell_mode=str(data.get("sell_mode", "B")).strip().upper() or "B",
        rank_drop_multiplier=float(data.get("rank_drop_multiplier", 2.0)),
        rule_na_policy=str(data.get("rule_na_policy", "fail")).strip().lower() or "fail",
        ranking=ranking,
        universe=universe,
        costs=CostModel(
            commission_bps=float(costs_raw.get("commission_bps", 0.0)),
            slippage_bps=float(costs_raw.get("slippage_bps", 0.0)),
        ),
        asof_mode=str(data.get("asof_mode", "quarter_end")).strip().lower() or "quarter_end",
        use_fundamentals_pit=_to_bool(data.get("use_fundamentals_pit"), False),
        strict_pit=_to_bool(data.get("strict_pit"), False),
        asof_lag_trading_days=max(0, _to_int(data.get("asof_lag_trading_days", 0), 0)),
        use_next_trading_day_availability=_to_bool(data.get("use_next_trading_day_availability"), False),
        fundamentals_availability_fallback=_to_bool(data.get("fundamentals_availability_fallback"), True),
        fundamentals_missing_policy=(
            str(data.get("fundamentals_missing_policy", "exclude")).strip().lower() or "exclude"
        ),
        fundamentals_fallback_q_days=max(0, _to_int(data.get("fundamentals_fallback_q_days", 45), 45)),
        fundamentals_fallback_k_days=max(0, _to_int(data.get("fundamentals_fallback_k_days", 90), 90)),
        missing_policy=str(data.get("missing_policy", "drop")).strip().lower() or "drop",
        execution_timing=str(data.get("execution_timing", "next_open")).strip().lower() or "next_open",
        benchmark=str(data.get("benchmark", "SPY")).strip() or "SPY",
        indicators=data.get("indicators", []) if isinstance(data.get("indicators"), list) else [],
        buy_condition_set=buy_condition_set,
        sell_condition_set=sell_condition_set,
        position_sizing=position_sizing,
        risk_limits=risk_limits,
        execution=execution,
        universe_filter=universe_filter,
        share_mode=str(data.get("share_mode", "fractional")).strip().lower() or "fractional",
        cash_buffer_pct=float(data.get("cash_buffer_pct", 0.0)),
        min_trade_notional=float(data.get("min_trade_notional", 0.0)),
        out_dir=data.get("out_dir"),
        funding=funding,
    )
    if cfg.funding.initial_cash is None:
        cfg.funding.initial_cash = cfg.initial_cash
    cfg.initial_cash = float(cfg.funding.initial_cash)
    if cfg.fundamentals_missing_policy not in {"exclude", "error"}:
        cfg.fundamentals_missing_policy = "exclude"
    return cfg


def load_strategy_config(path: str | Path) -> StrategyConfig:
    data = _load_raw_config(path)
    return strategy_config_from_dict(data)


def load_ranking_config(path: str | Path) -> RankingConfig:
    data = _load_raw_config(path)
    return ranking_config_from_dict(data)
