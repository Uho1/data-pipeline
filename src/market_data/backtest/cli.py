from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_data.backtest.models import load_ranking_config, load_strategy_config
from market_data.backtest.report import save_backtest_outputs
from market_data.backtest.simulator import run_backtest, run_rank_test, run_screen_backtest


def _resolve_run_dir(base_dir: str | None, name: str) -> Path:
    base = Path(base_dir).expanduser() if base_dir else Path("logs") / "backtests"
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name).strip())
    return base / f"{safe_name}_{ts}"


def run_backtest_screen_cli(args) -> int:  # type: ignore[no-untyped-def]
    ranking = None
    if getattr(args, "ranking", None):
        ranking = load_ranking_config(args.ranking)

    run_dir = _resolve_run_dir(args.out_dir, "screen")
    
    # Resolve universe
    universe_source = getattr(args, "universe", "sp500")
    if getattr(args, "use_sp500_pit", False):
        universe_source = "sp500_pit"

    # Set strict policy for missing fundamentals
    missing_policy = "error" if getattr(args, "strict", False) else str(getattr(args, "fundamentals_missing_policy", "exclude"))

    result = run_screen_backtest(
        screen_expr=args.screen,
        universe_source=universe_source,
        freq=args.freq,
        start=args.start,
        end=args.end,
        holdings=max(1, int(args.holdings)),
        sizing=args.sizing,
        market=args.market,
        out_dir=str(run_dir),
        ranking=ranking,
        benchmark=args.benchmark,
        execution_timing=args.execution_timing,
        use_fundamentals_pit=bool(args.use_fundamentals_pit),
        asof_lag_trading_days=max(0, int(args.asof_lag_trading_days)),
        use_next_trading_day_availability=bool(args.use_next_trading_day_availability),
        fundamentals_availability_fallback=bool(args.fundamentals_availability_fallback),
        fundamentals_missing_policy=missing_policy,
        fundamentals_fallback_q_days=max(0, int(args.fundamentals_fallback_q_days)),
        fundamentals_fallback_k_days=max(0, int(args.fundamentals_fallback_k_days)),
        strict=getattr(args, "strict", False),
    )

    print(f"[BACKTEST SCREEN] output={run_dir}")
    print(result.metrics)
    return 0 if not result.equity_curve.empty else 2


def run_backtest_strategy_cli(args) -> int:  # type: ignore[no-untyped-def]
    cfg = load_strategy_config(args.config)
    if getattr(args, "use_fundamentals_pit", None) is not None:
        cfg.use_fundamentals_pit = bool(args.use_fundamentals_pit)
    if getattr(args, "asof_lag_trading_days", None) is not None:
        cfg.asof_lag_trading_days = max(0, int(args.asof_lag_trading_days))
    if getattr(args, "use_next_trading_day_availability", None) is not None:
        cfg.use_next_trading_day_availability = bool(args.use_next_trading_day_availability)
    if getattr(args, "fundamentals_availability_fallback", None) is not None:
        cfg.fundamentals_availability_fallback = bool(args.fundamentals_availability_fallback)
    if getattr(args, "fundamentals_missing_policy", None) is not None:
        cfg.fundamentals_missing_policy = str(args.fundamentals_missing_policy)
    if getattr(args, "fundamentals_fallback_q_days", None) is not None:
        cfg.fundamentals_fallback_q_days = max(0, int(args.fundamentals_fallback_q_days))
    if getattr(args, "fundamentals_fallback_k_days", None) is not None:
        cfg.fundamentals_fallback_k_days = max(0, int(args.fundamentals_fallback_k_days))

    run_dir = _resolve_run_dir(args.out_dir or cfg.out_dir, cfg.name)
    cfg.out_dir = str(run_dir)

    result = run_backtest(cfg)
    save_backtest_outputs(result, run_dir)

    print(f"[BACKTEST STRATEGY] output={run_dir}")
    print(result.metrics)
    return 0 if not result.equity_curve.empty else 2


def run_rank_test_cli(args) -> int:  # type: ignore[no-untyped-def]
    run_dir = _resolve_run_dir(args.out_dir, "ranktest")
    return run_rank_test(
        ranking_config_path=args.ranking,
        universe=args.universe,
        freq=args.freq,
        start=args.start,
        end=args.end,
        market=args.market,
        tickers_file=args.tickers_file,
        out_dir=str(run_dir),
    )
