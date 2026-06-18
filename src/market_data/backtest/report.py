from __future__ import annotations

import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

from market_data.backtest.models import BacktestResult
from market_data.backtest.validation_symbol_time import (
    detect_ticker_time_inconsistency,
    summarize_ticker_time_issues,
)
from market_data.backtest.validation_sp500_pit import (
    detect_sp500_pit_membership_inconsistency,
    summarize_sp500_pit_issues,
)
from market_data.backtest.validation_snapshots import validate_rebalance_snapshots
from market_data.sec_sector_proxy import (
    SECTOR_SNAPSHOT_REQUIRED_COLUMNS,
    detect_sector_pit_time_inconsistency,
)
from market_data.sp500_pit import SP500_SNAPSHOT_REQUIRED_COLUMNS
from market_data.utils import ensure_dir


def compute_drawdown_curve(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve is None or equity_curve.empty:
        return pd.DataFrame(columns=["equity", "peak", "drawdown"])
    out = equity_curve.copy()
    eq = pd.to_numeric(out["equity"], errors="coerce")
    peak = eq.cummax()
    dd = (eq / peak.replace(0.0, np.nan)) - 1.0
    out["peak"] = peak
    out["drawdown"] = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_yearly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve is None or equity_curve.empty:
        return pd.DataFrame(columns=["year", "return_pct"])
    eq = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
    if eq.empty:
        return pd.DataFrame(columns=["year", "return_pct"])
    yearly = eq.resample("YE").last().pct_change()
    year_vals = yearly.index.year.tolist()
    ret_vals = yearly.values.tolist()
    out = pd.DataFrame({"year": year_vals, "return_pct": [v * 100.0 if v is not None else float("nan") for v in ret_vals]})
    out = out.dropna(subset=["return_pct"]).reset_index(drop=True)
    return out


def build_monthly_returns(strategy_equity: pd.DataFrame, benchmark_equity: pd.DataFrame | None = None) -> pd.DataFrame:
    if strategy_equity is None or strategy_equity.empty:
        return pd.DataFrame(columns=["month", "strategy_ret", "bench_ret", "excess_ret", "year", "month_num"])

    strat = strategy_equity.copy()
    strat.index = pd.to_datetime(strat.index, errors="coerce")
    strat = strat.loc[~strat.index.isna()]
    strat_ret = pd.to_numeric(strat["equity"], errors="coerce").pct_change().dropna()
    strat_monthly = (1.0 + strat_ret).resample("ME").prod() - 1.0

    bench_monthly = pd.Series(index=strat_monthly.index, dtype=float)
    if benchmark_equity is not None and not benchmark_equity.empty and "equity" in benchmark_equity.columns:
        bench = benchmark_equity.copy()
        bench.index = pd.to_datetime(bench.index, errors="coerce")
        bench = bench.loc[~bench.index.isna()]
        bret = pd.to_numeric(bench["equity"], errors="coerce").pct_change().dropna()
        bench_monthly = (1.0 + bret).resample("ME").prod() - 1.0
        bench_monthly = bench_monthly.reindex(strat_monthly.index)

    monthly = pd.DataFrame(
        {
            "month": strat_monthly.index.strftime("%Y-%m"),
            "strategy_ret": strat_monthly.values,
            "bench_ret": bench_monthly.values,
        },
        index=strat_monthly.index,
    )
    monthly["excess_ret"] = monthly["strategy_ret"] - monthly["bench_ret"]
    monthly["year"] = monthly.index.year
    monthly["month_num"] = monthly.index.month
    return monthly.reset_index(drop=True)


def build_yearly_summary(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    if monthly_returns is None or monthly_returns.empty:
        return pd.DataFrame(columns=["year", "strategy_ret", "bench_ret", "excess_ret"])

    df = monthly_returns.copy()
    out_rows: list[dict[str, float | int]] = []
    for yr, g in df.groupby("year"):
        strat = pd.to_numeric(g["strategy_ret"], errors="coerce").dropna()
        bench = pd.to_numeric(g["bench_ret"], errors="coerce").dropna()

        strat_y = float((1.0 + strat).prod() - 1.0) if not strat.empty else np.nan
        bench_y = float((1.0 + bench).prod() - 1.0) if not bench.empty else np.nan

        out_rows.append(
            {
                "year": int(yr),
                "strategy_ret": strat_y,
                "bench_ret": bench_y,
                "excess_ret": strat_y - bench_y if np.isfinite(strat_y) and np.isfinite(bench_y) else np.nan,
            }
        )

    return pd.DataFrame(out_rows).sort_values("year").reset_index(drop=True)


def _empty_metrics() -> dict[str, float | int]:
    return {
        "cagr": 0.0,
        "mdd": 0.0,
        "sharpe": 0.0,
        "vol": 0.0,
        "twr_mdd": 0.0,
        "twr_sharpe": 0.0,
        "twr_vol": 0.0,
        "account_total_return": np.nan,
        "account_cagr": np.nan,
        "account_mdd": np.nan,
        "account_sharpe": np.nan,
        "account_vol": np.nan,
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
    }


def normalize_metrics(metrics: dict[str, float | int] | None) -> dict[str, float | int]:
    out = _empty_metrics()
    if not isinstance(metrics, dict):
        return out
    for key in out:
        if key in metrics:
            out[key] = metrics[key]
    return out


def _normalize_funding_flows(funding_flows: pd.DataFrame | None) -> pd.DataFrame:
    if funding_flows is None or funding_flows.empty:
        return pd.DataFrame(columns=["date", "contribution"])
    out = funding_flows.copy()
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce")
    out["contribution"] = pd.to_numeric(out.get("contribution"), errors="coerce").fillna(0.0)
    out = out.dropna(subset=["date"]).sort_values("date")
    return out[["date", "contribution"]]


def _compute_twr_daily_returns(eq: pd.Series, funding_flows: pd.DataFrame | None) -> pd.Series:
    if eq is None or eq.empty:
        return pd.Series(dtype=float)
    flow_df = _normalize_funding_flows(funding_flows)
    flow = pd.Series(0.0, index=eq.index)
    if not flow_df.empty:
        grouped = flow_df.groupby("date")["contribution"].sum()
        flow = flow.add(grouped.reindex(eq.index).fillna(0.0), fill_value=0.0)
    twr_ret = ((eq - flow) / eq.shift(1)) - 1.0
    twr_ret = twr_ret.replace([np.inf, -np.inf], np.nan)
    twr_ret.iloc[0] = 0.0
    return twr_ret.fillna(0.0)


def _build_twr_curve(eq: pd.Series, funding_flows: pd.DataFrame | None) -> pd.Series:
    twr_ret = _compute_twr_daily_returns(eq, funding_flows)
    if twr_ret.empty:
        return pd.Series(dtype=float)
    return (1.0 + twr_ret).cumprod()


def compute_account_metrics(equity_curve: pd.DataFrame) -> dict[str, float]:
    if equity_curve is None or equity_curve.empty or "equity" not in equity_curve.columns:
        return {
            "account_total_return": np.nan,
            "account_cagr": np.nan,
            "account_mdd": np.nan,
            "account_sharpe": np.nan,
            "account_vol": np.nan,
        }

    eq = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
    if eq.empty:
        return {
            "account_total_return": np.nan,
            "account_cagr": np.nan,
            "account_mdd": np.nan,
            "account_sharpe": np.nan,
            "account_vol": np.nan,
        }

    day_count = max((eq.index[-1] - eq.index[0]).days, 1)
    years = day_count / 365.25
    start_value = float(eq.iloc[0])
    end_value = float(eq.iloc[-1])

    if np.isfinite(start_value) and start_value > 0.0 and np.isfinite(end_value):
        account_total_return = float((end_value / start_value) - 1.0)
    else:
        account_total_return = np.nan

    if (
        np.isfinite(start_value)
        and start_value > 0.0
        and np.isfinite(end_value)
        and end_value > 0.0
        and years > 0.0
    ):
        account_cagr = float((end_value / start_value) ** (1.0 / years) - 1.0)
    else:
        account_cagr = np.nan

    dd_curve = compute_drawdown_curve(pd.DataFrame({"equity": eq}, index=eq.index))
    account_mdd = float(dd_curve["drawdown"].min()) if not dd_curve.empty else np.nan

    ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) >= 2 and float(ret.std()) > 0.0:
        account_sharpe = float((ret.mean() / ret.std()) * np.sqrt(252.0))
        account_vol = float(ret.std() * np.sqrt(252.0))
    else:
        account_sharpe = 0.0
        account_vol = 0.0

    return {
        "account_total_return": account_total_return,
        "account_cagr": account_cagr,
        "account_mdd": account_mdd,
        "account_sharpe": account_sharpe,
        "account_vol": account_vol,
    }


def _xnpv(rate: float, amounts: np.ndarray, years: np.ndarray) -> float:
    if rate <= -1.0:
        return np.inf
    return float(np.sum(amounts / np.power(1.0 + rate, years)))


def _compute_xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    if len(cashflows) < 2:
        return float("nan")

    ordered = sorted([(pd.Timestamp(d), float(v)) for d, v in cashflows], key=lambda x: x[0])
    amounts = np.array([v for _, v in ordered], dtype=float)
    if not (np.any(amounts > 0.0) and np.any(amounts < 0.0)):
        return float("nan")

    base_date = ordered[0][0]
    years = np.array([(d - base_date).days / 365.25 for d, _ in ordered], dtype=float)

    grid = np.array([-0.999, -0.9, -0.7, -0.5, -0.2, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    vals = np.array([_xnpv(float(r), amounts, years) for r in grid], dtype=float)

    low = float("nan")
    high = float("nan")
    for i in range(len(grid) - 1):
        v0 = float(vals[i])
        v1 = float(vals[i + 1])
        if np.isnan(v0) or np.isnan(v1):
            continue
        if v0 == 0.0:
            return float(grid[i])
        if v0 * v1 < 0.0:
            low = float(grid[i])
            high = float(grid[i + 1])
            break

    if not np.isfinite(low) or not np.isfinite(high):
        return float("nan")

    f_low = _xnpv(low, amounts, years)
    f_high = _xnpv(high, amounts, years)
    if not np.isfinite(f_low) or not np.isfinite(f_high):
        return float("nan")

    for _ in range(120):
        mid = (low + high) * 0.5
        f_mid = _xnpv(mid, amounts, years)
        if not np.isfinite(f_mid):
            return float("nan")
        if abs(f_mid) < 1e-10:
            return float(mid)
        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
        if abs(high - low) < 1e-8:
            break
    return float((low + high) * 0.5)


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    funding_flows: pd.DataFrame | None = None,
) -> dict[str, float | int]:
    if equity_curve is None or equity_curve.empty:
        return _empty_metrics()

    eq = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
    if eq.empty:
        return _empty_metrics()

    twr_curve = _build_twr_curve(eq, funding_flows)
    twr_ret = twr_curve.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    twr_total_return = float(twr_curve.iloc[-1] - 1.0) if not twr_curve.empty else 0.0

    day_count = max((eq.index[-1] - eq.index[0]).days, 1)
    years = day_count / 365.25
    growth = 1.0 + twr_total_return
    cagr = float(growth ** (1.0 / years) - 1.0) if growth > 0.0 else -1.0

    twr_df = pd.DataFrame({"equity": twr_curve}, index=twr_curve.index)
    dd_curve = compute_drawdown_curve(twr_df)
    twr_mdd = float(dd_curve["drawdown"].min()) if not dd_curve.empty else 0.0

    ret = twr_ret.dropna()
    if len(ret) >= 2 and float(ret.std()) > 0.0:
        twr_sharpe = float((ret.mean() / ret.std()) * np.sqrt(252.0))
        twr_vol = float(ret.std() * np.sqrt(252.0))
    else:
        twr_sharpe = 0.0
        twr_vol = 0.0

    turnover = 0.0
    if trades is not None and not trades.empty:
        traded = pd.to_numeric(trades.get("notional"), errors="coerce").abs().sum()
        avg_equity = float(eq.mean()) if float(eq.mean()) > 0 else np.nan
        turnover = float(traded / avg_equity) if np.isfinite(avg_equity) else 0.0

    hit_rate = 0.0
    if trades is not None and not trades.empty and "side" in trades.columns and "realized_pnl" in trades.columns:
        sells = trades.loc[trades["side"].astype(str).str.lower() == "sell"].copy()
        if not sells.empty:
            pnl = pd.to_numeric(sells["realized_pnl"], errors="coerce")
            valid = pnl.dropna()
            if not valid.empty:
                hit_rate = float((valid > 0.0).mean())

    ending_value = float(eq.iloc[-1])
    flow_df = _normalize_funding_flows(funding_flows)
    total_contributed = float(flow_df.loc[flow_df["contribution"] > 0, "contribution"].sum()) if not flow_df.empty else float(eq.iloc[0])
    pnl = float(ending_value - total_contributed)

    cashflows: list[tuple[pd.Timestamp, float]] = []
    if flow_df.empty:
        cashflows = [(pd.Timestamp(eq.index[0]), -float(eq.iloc[0])), (pd.Timestamp(eq.index[-1]), float(ending_value))]
    else:
        for _, row in flow_df.iterrows():
            contrib = float(row["contribution"])
            if contrib == 0.0:
                continue
            # Investor perspective: deposits are negative cash flows.
            cashflows.append((pd.Timestamp(row["date"]), -contrib))
        cashflows.append((pd.Timestamp(eq.index[-1]), float(ending_value)))

    mwr_irr = _compute_xirr(cashflows)
    account_metrics = compute_account_metrics(pd.DataFrame({"equity": eq}, index=eq.index))

    return normalize_metrics(
        {
            "cagr": cagr,  # backward compatibility alias for twr_cagr
            "mdd": twr_mdd,  # backward compatibility alias for twr_mdd
            "sharpe": twr_sharpe,  # backward compatibility alias for twr_sharpe
            "vol": twr_vol,  # backward compatibility alias for twr_vol
            "twr_total_return": twr_total_return,
            "total_return": twr_total_return,  # backward compatibility alias
            "twr_cagr": cagr,
            "twr_mdd": twr_mdd,
            "twr_sharpe": twr_sharpe,
            "twr_vol": twr_vol,
            "account_total_return": account_metrics["account_total_return"],
            "account_cagr": account_metrics["account_cagr"],
            "account_mdd": account_metrics["account_mdd"],
            "account_sharpe": account_metrics["account_sharpe"],
            "account_vol": account_metrics["account_vol"],
            "turnover": turnover,
            "hit_rate": hit_rate,
            "mwr_irr": mwr_irr,
            "total_contributed": total_contributed,
            "ending_value": ending_value,
            "pnl": pnl,
            "trades": int(len(trades) if trades is not None else 0),
        }
    )


def save_backtest_outputs(result: BacktestResult, out_dir: str | Path) -> Path:
    root = Path(out_dir).expanduser()
    ensure_dir(root)

    trades_path = root / "trades.csv"
    equity_path = root / "equity_curve.parquet"
    benchmark_path = root / "benchmark_curve.parquet"
    excess_path = root / "excess_curve.parquet"
    drawdown_path = root / "drawdown_curve.parquet"
    holdings_path = root / "holdings.csv"
    yearly_path = root / "yearly_returns.csv"
    monthly_path = root / "monthly_returns.csv"
    yearly_summary_path = root / "yearly_summary.csv"
    funding_path = root / "funding_flows.csv"
    metrics_path = root / "metrics.json"
    rebalance_path = root / "rebalance_log.csv"
    snapshot_index_path = root / "rebalance_snapshots_index.csv"

    result.trades.to_csv(trades_path, index=False)
    result.equity_curve.to_parquet(equity_path, index=False)
    result.benchmark_curve.to_parquet(benchmark_path, index=False)
    result.excess_curve.to_parquet(excess_path, index=False)
    result.drawdown_curve.to_parquet(drawdown_path, index=False)
    result.holdings.to_csv(holdings_path, index=False)
    result.yearly_returns.to_csv(yearly_path, index=False)
    result.monthly_returns.to_csv(monthly_path, index=False)
    result.yearly_summary.to_csv(yearly_summary_path, index=False)
    result.funding_flows.to_csv(funding_path, index=False)
    result.rebalance_log.to_csv(rebalance_path, index=False)
    result.rebalance_snapshots_index.to_csv(snapshot_index_path, index=False)
    metrics_path.write_text(json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return root


def compare_results(left: BacktestResult, right: BacktestResult) -> dict[str, pd.DataFrame]:
    metrics = pd.DataFrame(
        {
            "metric": sorted(set(left.metrics.keys()) | set(right.metrics.keys())),
        }
    )
    metrics[left.name] = metrics["metric"].map(lambda k: left.metrics.get(k))
    metrics[right.name] = metrics["metric"].map(lambda k: right.metrics.get(k))

    eq_left = (
        left.equity_curve[["date", "equity"]].rename(columns={"equity": left.name})
        if not left.equity_curve.empty
        else pd.DataFrame(columns=["date", left.name])
    )
    eq_right = (
        right.equity_curve[["date", "equity"]].rename(columns={"equity": right.name})
        if not right.equity_curve.empty
        else pd.DataFrame(columns=["date", right.name])
    )
    equity = eq_left.merge(eq_right, on="date", how="outer").sort_values("date")

    yr_left = (
        left.yearly_returns.rename(columns={"return_pct": left.name})
        if not left.yearly_returns.empty
        else pd.DataFrame(columns=["year", left.name])
    )
    yr_right = (
        right.yearly_returns.rename(columns={"return_pct": right.name})
        if not right.yearly_returns.empty
        else pd.DataFrame(columns=["year", right.name])
    )
    yearly = yr_left.merge(yr_right, on="year", how="outer").sort_values("year")

    return {
        "metrics": metrics,
        "equity": equity,
        "yearly_returns": yearly,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        return ts.isoformat() if pd.notna(ts) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return None if not np.isfinite(val) else val
    if pd.isna(value):
        return None
    return value


def _frame_to_iso_csv(df: pd.DataFrame | None, path: Path) -> None:
    out = pd.DataFrame() if df is None else df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            ts = pd.to_datetime(out[col], errors="coerce")
            out[col] = ts.map(lambda x: x.isoformat() if pd.notna(x) else "")
        elif isinstance(out[col].dtype, pd.PeriodDtype):
            out[col] = out[col].astype(str)
    out.to_csv(path, index=False, encoding="utf-8")


def _try_write_parquet(df: pd.DataFrame | None, path: Path) -> bool:
    if df is None:
        return False
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def _core_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "twr_total_return",
        "twr_cagr",
        "twr_mdd",
        "twr_sharpe",
        "twr_vol",
        "account_cagr",
        "account_mdd",
        "account_sharpe",
        "account_vol",
        "mwr_irr",
        "total_contributed",
        "ending_value",
        "pnl",
        "turnover",
        "trades",
    ]
    return {k: _json_safe(metrics.get(k)) for k in keys}


def _collect_sp500_pit_snapshot_stats(results_by_mode: dict[str, BacktestResult]) -> dict[str, Any]:
    source_mix: dict[str, int] = {}
    confidences: list[float] = []
    snapshot_rows = 0
    snapshot_files_checked = 0
    snapshot_files_missing = 0

    for mode_key, result in (results_by_mode or {}).items():
        idx = result.rebalance_snapshots_index if result is not None else pd.DataFrame()
        if idx is None or idx.empty or "snapshot_path" not in idx.columns:
            continue
        for _, row in idx.iterrows():
            raw_path = str(row.get("snapshot_path", "") or "").strip()
            if not raw_path:
                continue
            p = Path(raw_path).expanduser()
            if not p.exists():
                snapshot_files_missing += 1
                continue
            snapshot_files_checked += 1
            try:
                snap = pd.read_csv(p)
            except Exception:
                continue
            if snap.empty:
                continue
            row_count = int(len(snap))
            snapshot_rows += row_count

            src_col = None
            for cand in ["sp500_pit_source_tier", "sp500_pit_source", "sp500_pit_source_name"]:
                if cand in snap.columns:
                    src_col = cand
                    break
            if src_col is not None:
                vc = snap[src_col].astype(str).str.strip().replace("", "unknown").value_counts()
                for k, v in vc.to_dict().items():
                    source_mix[str(k)] = source_mix.get(str(k), 0) + int(v)

            if "sp500_pit_confidence" in snap.columns:
                conf = pd.to_numeric(snap["sp500_pit_confidence"], errors="coerce").dropna()
                if not conf.empty:
                    confidences.extend(conf.astype(float).tolist())

    conf_series = pd.Series(confidences, dtype=float) if confidences else pd.Series(dtype=float)
    confidence_summary = {
        "count": int(conf_series.shape[0]),
        "min": float(conf_series.min()) if not conf_series.empty else None,
        "max": float(conf_series.max()) if not conf_series.empty else None,
        "mean": float(conf_series.mean()) if not conf_series.empty else None,
        "median": float(conf_series.median()) if not conf_series.empty else None,
        "p10": float(conf_series.quantile(0.10)) if not conf_series.empty else None,
        "p90": float(conf_series.quantile(0.90)) if not conf_series.empty else None,
    }

    return {
        "snapshot_rows": int(snapshot_rows),
        "snapshot_files_checked": int(snapshot_files_checked),
        "snapshot_files_missing": int(snapshot_files_missing),
        "source_mix": source_mix,
        "confidence_summary": confidence_summary,
    }


def _summarize_sp500_pit_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [c for c in (checks or []) if "sp500_pit" in str(c.get("name", ""))]
    if not rows:
        return {
            "status": "not_applicable",
            "check_count": 0,
            "pass_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "failed_checks": [],
        }

    pass_count = sum(1 for c in rows if str(c.get("status", "")).lower() == "pass")
    warn_count = sum(1 for c in rows if str(c.get("status", "")).lower() == "warn")
    fail_count = sum(1 for c in rows if str(c.get("status", "")).lower() == "fail")
    overall = "pass"
    if fail_count > 0:
        overall = "fail"
    elif warn_count > 0:
        overall = "warn"

    return {
        "status": overall,
        "check_count": int(len(rows)),
        "pass_count": int(pass_count),
        "warn_count": int(warn_count),
        "fail_count": int(fail_count),
        "failed_checks": [str(c.get("name", "")) for c in rows if str(c.get("status", "")).lower() == "fail"],
    }


def _git_commit_hash() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        out = str(proc.stdout).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _run_consistency_checks(
    result: BacktestResult,
    mode_key: str,
    *,
    market: str = "us",
    price_root: str | Path | None = None,
    overrides_path: str | Path | None = None,
    mode_label: str | None = None,
    run_dir: str | Path | None = None,
    snapshot_validation_mode: str = "warn",
    sector_validation_mode: str = "warn",
    require_sp500_pit_evidence: bool = False,
    sp500_pit_validation_mode: str = "warn",
    sp500_pit_min_confidence: float = 0.0,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    metrics = result.metrics or {}

    pnl = pd.to_numeric(pd.Series([metrics.get("pnl")]), errors="coerce").iloc[0]
    ending = pd.to_numeric(pd.Series([metrics.get("ending_value")]), errors="coerce").iloc[0]
    total_contrib = pd.to_numeric(pd.Series([metrics.get("total_contributed")]), errors="coerce").iloc[0]
    if pd.notna(pnl) and pd.notna(ending) and pd.notna(total_contrib):
        diff = float(pnl - (ending - total_contrib))
        checks.append(
            {
                "name": f"{mode_key}.pnl_identity",
                "status": "pass" if abs(diff) <= 1e-6 else "fail",
                "details": f"diff={diff:.8f}",
            }
        )
    else:
        checks.append({"name": f"{mode_key}.pnl_identity", "status": "warn", "details": "missing metrics fields"})

    eq = result.equity_curve if result.equity_curve is not None else pd.DataFrame()
    if not eq.empty and {"equity", "cash", "positions_value"}.issubset(set(eq.columns)):
        lhs = pd.to_numeric(eq["cash"], errors="coerce").fillna(0.0) + pd.to_numeric(eq["positions_value"], errors="coerce").fillna(0.0)
        rhs = pd.to_numeric(eq["equity"], errors="coerce").fillna(0.0)
        ok = bool(np.allclose(lhs.to_numpy(dtype=float), rhs.to_numpy(dtype=float), atol=1e-6))
        checks.append(
            {
                "name": f"{mode_key}.equity_cash_positions_identity",
                "status": "pass" if ok else "fail",
                "sample_rows_checked": int(len(eq)),
            }
        )
    else:
        checks.append(
            {
                "name": f"{mode_key}.equity_cash_positions_identity",
                "status": "warn",
                "details": "equity/cash/positions_value columns not available",
            }
        )

    flows = result.funding_flows if result.funding_flows is not None else pd.DataFrame()
    if not flows.empty:
        if "cumulative_contributed_after" in flows.columns:
            seq = pd.to_numeric(flows["cumulative_contributed_after"], errors="coerce").dropna()
            ff_total = float(seq.iloc[-1]) if not seq.empty else float(pd.to_numeric(flows.get("contribution"), errors="coerce").fillna(0.0).sum())
        else:
            ff_total = float(pd.to_numeric(flows.get("contribution"), errors="coerce").fillna(0.0).sum())
        if pd.notna(total_contrib):
            diff = float(ff_total - float(total_contrib))
            status = "pass" if abs(diff) <= 1e-6 else "warn"
            checks.append(
                {
                    "name": f"{mode_key}.funding_total_matches_metrics",
                    "status": status,
                    "details": f"funding_total={ff_total:.6f}, metrics_total={float(total_contrib):.6f}, diff={diff:.6f}",
                }
            )
        else:
            checks.append({"name": f"{mode_key}.funding_total_matches_metrics", "status": "warn", "details": "metrics total_contributed missing"})

        if {"is_rebalance_same_day", "event_order_tag"}.issubset(set(flows.columns)):
            mask = flows.get("is_rebalance_same_day", pd.Series(False)).fillna(False).astype(bool)
            tagged = flows.loc[mask, "event_order_tag"].astype(str)
            ok = bool(tagged.empty or tagged.eq("funding_before_rebalance").all())
            checks.append(
                {
                    "name": f"{mode_key}.same_day_order_tag",
                    "status": "pass" if ok else "fail",
                    "details": "all same-day rows tagged funding_before_rebalance",
                }
            )
    else:
        checks.append({"name": f"{mode_key}.funding_presence", "status": "warn", "details": "funding_flows empty"})

    if mode_key == "va":
        needed = {"floor_applied", "max_cap_applied", "reason"}
        missing = sorted(needed.difference(set(flows.columns)))
        checks.append(
            {
                "name": f"{mode_key}.va_cap_columns",
                "status": "pass" if not missing else "warn",
                "details": "missing=" + ",".join(missing) if missing else "ok",
            }
        )

    detected = detect_ticker_time_inconsistency(
        result.trades if result.trades is not None else pd.DataFrame(),
        market=market,
        price_root=price_root,
        overrides_path=overrides_path,
        tolerance_days=7,
        warn_days=30,
        fail_days=180,
        check_last_valid=False,
        mode_label=mode_label or mode_key,
    )
    checks.append(
        summarize_ticker_time_issues(
            detected,
            check_name=f"{mode_key}.ticker_time_consistency",
            max_examples=5,
        )
    )
    sector_detected = detect_sector_pit_time_inconsistency(
        result.trades if result.trades is not None else pd.DataFrame(),
        market=market,
        mode_label=mode_label or mode_key,
    )
    checks.append(
        summarize_ticker_time_issues(
            sector_detected,
            check_name=f"{mode_key}.sector_pit_time_consistency",
            max_examples=5,
        )
    )
    issues_df = detected.get("issues")
    if not isinstance(issues_df, pd.DataFrame):
        issues_df = pd.DataFrame()
    sector_issues_df = sector_detected.get("issues")
    if isinstance(sector_issues_df, pd.DataFrame) and not sector_issues_df.empty:
        issues_df = pd.concat([issues_df, sector_issues_df], ignore_index=True, sort=False)

    if require_sp500_pit_evidence:
        low_conf_result = "fail" if str(sp500_pit_validation_mode or "warn").strip().lower() == "fail" else "warn"
        sp500_detected = detect_sp500_pit_membership_inconsistency(
            result.trades if result.trades is not None else pd.DataFrame(),
            mode_label=mode_label or mode_key,
            min_confidence=0.0,
            low_confidence_threshold=float(sp500_pit_min_confidence) if float(sp500_pit_min_confidence or 0.0) > 0.0 else None,
            low_confidence_result=low_conf_result,
        )
        checks.append(
            summarize_sp500_pit_issues(
                sp500_detected,
                check_name=f"{mode_key}.sp500_pit_membership_consistency",
                max_examples=5,
            )
        )
        sp500_issues_df = sp500_detected.get("issues")
        if isinstance(sp500_issues_df, pd.DataFrame) and not sp500_issues_df.empty:
            if issues_df.empty:
                issues_df = sp500_issues_df.copy()
            else:
                issues_df = pd.concat([issues_df, sp500_issues_df], ignore_index=True, sort=False)

    snapshot_base = Path(run_dir).expanduser() if run_dir is not None else None
    snapshot_index_path = (snapshot_base / "rebalance_snapshots_index.csv") if snapshot_base is not None else None
    snapshot_result = validate_rebalance_snapshots(
        snapshot_index_path if snapshot_index_path is not None else result.rebalance_snapshots_index,
        rebalance_log_df=result.rebalance_log if result.rebalance_log is not None else pd.DataFrame(),
        base_dir=snapshot_base,
        validation_mode=snapshot_validation_mode,
    )
    snapshot_check = {
        "name": f"{mode_key}.rebalance_snapshots_integrity",
        "status": snapshot_result.get("status", "warn"),
        "details": (
            f"rebalance_rows={snapshot_result.get('counts', {}).get('rebalance_rows', 0)}, "
            f"index_rows={snapshot_result.get('counts', {}).get('snapshot_index_rows', 0)}, "
            f"existing={snapshot_result.get('counts', {}).get('existing_snapshot_files', 0)}, "
            f"missing={snapshot_result.get('counts', {}).get('missing_snapshot_files', 0)}"
        ),
        "counts": snapshot_result.get("counts", {}),
        "summary": snapshot_result.get("summary", {}),
        "examples": snapshot_result.get("bad_rows", [])[:5],
    }
    checks.append(snapshot_check)
    sector_snapshot = validate_rebalance_snapshots(
        snapshot_index_path if snapshot_index_path is not None else result.rebalance_snapshots_index,
        rebalance_log_df=result.rebalance_log if result.rebalance_log is not None else pd.DataFrame(),
        base_dir=snapshot_base,
        required_columns=SECTOR_SNAPSHOT_REQUIRED_COLUMNS,
        validation_mode=sector_validation_mode,
    )
    checks.append(
        {
            "name": f"{mode_key}.sector_snapshot_evidence",
            "status": sector_snapshot.get("status", "warn"),
            "details": (
                f"rebalance_rows={sector_snapshot.get('counts', {}).get('rebalance_rows', 0)}, "
                f"index_rows={sector_snapshot.get('counts', {}).get('snapshot_index_rows', 0)}, "
                f"existing={sector_snapshot.get('counts', {}).get('existing_snapshot_files', 0)}, "
                f"missing={sector_snapshot.get('counts', {}).get('missing_snapshot_files', 0)}"
            ),
            "counts": sector_snapshot.get("counts", {}),
            "summary": sector_snapshot.get("summary", {}),
            "examples": sector_snapshot.get("bad_rows", [])[:5],
        }
    )
    if require_sp500_pit_evidence:
        sp500_snapshot = validate_rebalance_snapshots(
            snapshot_index_path if snapshot_index_path is not None else result.rebalance_snapshots_index,
            rebalance_log_df=result.rebalance_log if result.rebalance_log is not None else pd.DataFrame(),
            base_dir=snapshot_base,
            required_columns=SP500_SNAPSHOT_REQUIRED_COLUMNS,
            validation_mode=sp500_pit_validation_mode,
        )
        checks.append(
            {
                "name": f"{mode_key}.sp500_pit_snapshot_evidence",
                "status": sp500_snapshot.get("status", "warn"),
                "details": (
                    f"rebalance_rows={sp500_snapshot.get('counts', {}).get('rebalance_rows', 0)}, "
                    f"index_rows={sp500_snapshot.get('counts', {}).get('snapshot_index_rows', 0)}, "
                    f"existing={sp500_snapshot.get('counts', {}).get('existing_snapshot_files', 0)}, "
                    f"missing={sp500_snapshot.get('counts', {}).get('missing_snapshot_files', 0)}"
                ),
                "counts": sp500_snapshot.get("counts", {}),
                "summary": sp500_snapshot.get("summary", {}),
                "examples": sp500_snapshot.get("bad_rows", [])[:5],
            }
        )
    return checks, issues_df, snapshot_check


def _write_result_files(result: BacktestResult, out_dir: Path, files_map: dict[str, str], key_prefix: str) -> None:
    ensure_dir(out_dir)
    mapping: list[tuple[str, pd.DataFrame]] = [
        ("metrics.json", pd.DataFrame()),
        ("equity_curve.csv", result.equity_curve),
        ("benchmark_curve.csv", result.benchmark_curve),
        ("excess_curve.csv", result.excess_curve),
        ("drawdown_curve.csv", result.drawdown_curve),
        ("monthly_returns.csv", result.monthly_returns),
        ("yearly_returns.csv", result.yearly_returns),
        ("yearly_summary.csv", result.yearly_summary),
        ("trades.csv", result.trades),
        ("funding_flows.csv", result.funding_flows),
        ("rebalance_log.csv", result.rebalance_log),
        ("rebalance_snapshots_index.csv", result.rebalance_snapshots_index),
        ("holdings.csv", result.holdings),
    ]

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_json_safe(result.metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    files_map[f"{key_prefix}.metrics"] = str(metrics_path.relative_to(out_dir.parent.parent if out_dir.parent.name == "runs" else out_dir.parent))

    for fname, df in mapping[1:]:
        p = out_dir / fname
        _frame_to_iso_csv(df, p)
        rel_base = out_dir.parent.parent if out_dir.parent.name == "runs" else out_dir.parent
        files_map[f"{key_prefix}.{fname.replace('.csv', '')}"] = str(p.relative_to(rel_base))

    # Export rebalance snapshot files referenced by index into bundle-local directory.
    # This allows offline replay/AI validation without depending on original paths.
    try:
        idx = result.rebalance_snapshots_index.copy() if result.rebalance_snapshots_index is not None else pd.DataFrame()
    except Exception:
        idx = pd.DataFrame()
    if not idx.empty and "snapshot_path" in idx.columns:
        snap_dir = out_dir / "rebalance_snapshots"
        ensure_dir(snap_dir)
        exported_rows: list[dict[str, Any]] = []
        for i, row in idx.iterrows():
            src_raw = str(row.get("snapshot_path", "") or "").strip()
            signal_date = row.get("signal_date")
            src = Path(src_raw).expanduser() if src_raw else Path("")
            target_name = src.name if src_raw else f"snapshot_{i}.csv"
            if not target_name.lower().endswith(".csv"):
                target_name = f"{target_name}.csv"
            dst = snap_dir / target_name

            copied = False
            if src_raw:
                if src.is_absolute() and src.exists():
                    try:
                        shutil.copy2(src, dst)
                        copied = True
                    except Exception:
                        copied = False
                else:
                    # try relative to current cwd and out_dir
                    for cand in [Path.cwd() / src_raw, out_dir / src_raw]:
                        if cand.exists():
                            try:
                                shutil.copy2(cand, dst)
                                copied = True
                                break
                            except Exception:
                                copied = False
            exported_rows.append(
                {
                    "signal_date": signal_date,
                    "snapshot_path": f"rebalance_snapshots/{target_name}" if copied else src_raw,
                    "source_snapshot_path": src_raw,
                    "copied_to_bundle": bool(copied),
                }
            )

        exported_idx = pd.DataFrame(exported_rows)
        idx_path = out_dir / "rebalance_snapshots_index.csv"
        _frame_to_iso_csv(exported_idx, idx_path)

    for parquet_name, df in [
        ("equity_curve.parquet", result.equity_curve),
        ("benchmark_curve.parquet", result.benchmark_curve),
        ("excess_curve.parquet", result.excess_curve),
        ("drawdown_curve.parquet", result.drawdown_curve),
    ]:
        p = out_dir / parquet_name
        if _try_write_parquet(df, p):
            rel_base = out_dir.parent.parent if out_dir.parent.name == "runs" else out_dir.parent
            files_map[f"{key_prefix}.{parquet_name.replace('.parquet', '')}_parquet"] = str(p.relative_to(rel_base))


def export_ai_review_bundle(
    export_root: str | Path,
    *,
    mode: str,
    strategy_context: dict[str, Any] | None,
    funding_context: dict[str, Any] | None,
    single_result: BacktestResult | None = None,
    funding_compare_results: dict[str, BacktestResult] | None = None,
    strategy_compare_results: dict[str, BacktestResult] | None = None,
    compare_notes: dict[str, Any] | None = None,
    ui_context_snapshot: dict[str, Any] | None = None,
    app_version: str = "unknown",
    snapshot_validation_mode: str = "fail",
    sector_validation_mode: str = "warn",
    sp500_pit_validation_mode: str = "warn",
) -> Path:
    root = Path(export_root).expanduser()
    ensure_dir(root)

    files_map: dict[str, str] = {}
    checks: list[dict[str, Any]] = []
    mode_key = str(mode or "single")
    compare_rows: list[dict[str, Any]] = []
    ticker_issue_frames: list[pd.DataFrame] = []
    snapshot_status_by_mode: dict[str, str] = {}
    strategy_ctx_raw = strategy_context or {}
    effective_cfg = strategy_ctx_raw.get("effective_config", {}) if isinstance(strategy_ctx_raw, dict) else {}
    if not isinstance(effective_cfg, dict):
        effective_cfg = {}
    universe_raw = effective_cfg.get("universe", {}) if isinstance(effective_cfg.get("universe", {}), dict) else {}
    universe_source = str(
        universe_raw.get("source")
        or strategy_ctx_raw.get("universe_source")
        or strategy_ctx_raw.get("universe_summary", "")
        or ""
    ).strip().lower()
    try:
        sp500_pit_min_confidence = float(universe_raw.get("sp500_pit_min_confidence", 0.0) or 0.0)
    except Exception:
        sp500_pit_min_confidence = 0.0
    require_sp500_pit_evidence = universe_source == "sp500_pit"
    market = str(strategy_ctx_raw.get("market", "us") or "us").strip().lower()
    price_root = Path("data") / "prices"
    override_path: Path | None = None
    for cand in [Path("config") / "symbol_identity_overrides.csv", Path("config") / "symbol_identity_overrides.json"]:
        if cand.exists():
            override_path = cand
            break

    if mode_key == "single":
        if single_result is None:
            raise ValueError("single mode export requires single_result")
        result_dir = root / "result"
        _write_result_files(single_result, result_dir, files_map, "result")
        run_checks, ticker_issues, snapshot_check = _run_consistency_checks(
            single_result,
            "single",
            market=market,
            price_root=price_root,
            overrides_path=override_path,
            mode_label="single",
            run_dir=result_dir,
            snapshot_validation_mode=snapshot_validation_mode,
            sector_validation_mode=sector_validation_mode,
            require_sp500_pit_evidence=require_sp500_pit_evidence,
            sp500_pit_validation_mode=sp500_pit_validation_mode,
            sp500_pit_min_confidence=sp500_pit_min_confidence,
        )
        checks.extend(run_checks)
        snapshot_status_by_mode["single"] = str(snapshot_check.get("status", "warn"))
        if not ticker_issues.empty:
            issue_path = result_dir / "ticker_time_issues.csv"
            _frame_to_iso_csv(ticker_issues, issue_path)
            files_map["result.ticker_time_issues"] = "result/ticker_time_issues.csv"
            ticker_issue_frames.append(ticker_issues.copy())
    elif mode_key == "funding_compare":
        if not funding_compare_results:
            raise ValueError("funding_compare export requires funding_compare_results")
        runs_dir = root / "runs"
        ensure_dir(runs_dir)
        mode_label_map = {
            "lump_sum": "일시불",
            "dca": "정액적립(DCA)",
            "va": "가치평균(VA)",
        }
        for mk in ["lump_sum", "dca", "va"]:
            res = funding_compare_results.get(mk)
            if res is None:
                checks.append({"name": f"{mk}.result_exists", "status": "fail", "details": "missing result"})
                continue
            _write_result_files(res, runs_dir / mk, files_map, f"runs.{mk}")
            run_checks, ticker_issues, snapshot_check = _run_consistency_checks(
                res,
                mk,
                market=market,
                price_root=price_root,
                overrides_path=override_path,
                mode_label=mode_label_map.get(mk, mk),
                run_dir=runs_dir / mk,
                snapshot_validation_mode=snapshot_validation_mode,
                sector_validation_mode=sector_validation_mode,
                require_sp500_pit_evidence=require_sp500_pit_evidence,
                sp500_pit_validation_mode=sp500_pit_validation_mode,
                sp500_pit_min_confidence=sp500_pit_min_confidence,
            )
            checks.extend(run_checks)
            snapshot_status_by_mode[mk] = str(snapshot_check.get("status", "warn"))
            if not ticker_issues.empty:
                issue_path = runs_dir / mk / "ticker_time_issues.csv"
                _frame_to_iso_csv(ticker_issues, issue_path)
                files_map[f"runs.{mk}.ticker_time_issues"] = f"runs/{mk}/ticker_time_issues.csv"
                ticker_issue_frames.append(ticker_issues.copy())
            compare_rows.append({"mode_key": mk, "mode_label": mode_label_map.get(mk, mk), **_core_metrics(res.metrics)})

        compare_dir = root / "compare"
        ensure_dir(compare_dir)
        cmp_df = pd.DataFrame(compare_rows)
        _frame_to_iso_csv(cmp_df, compare_dir / "compare_metrics.csv")
        (compare_dir / "compare_metrics.json").write_text(json.dumps(_json_safe(compare_rows), ensure_ascii=False, indent=2), encoding="utf-8")
        files_map["compare.metrics_csv"] = "compare/compare_metrics.csv"
        files_map["compare.metrics_json"] = "compare/compare_metrics.json"

        combined = pd.DataFrame()
        for mk, res in (funding_compare_results or {}).items():
            ff = res.funding_flows.copy() if res.funding_flows is not None else pd.DataFrame()
            if ff.empty:
                continue
            ff["mode_key"] = mk
            combined = pd.concat([combined, ff], ignore_index=True, sort=False)
        if not combined.empty:
            _frame_to_iso_csv(combined, compare_dir / "funding_flows_combined.csv")
            files_map["compare.funding_flows_combined"] = "compare/funding_flows_combined.csv"
        if ticker_issue_frames:
            combined_ticker = pd.concat(ticker_issue_frames, ignore_index=True, sort=False)
            _frame_to_iso_csv(combined_ticker, compare_dir / "ticker_time_issues_combined.csv")
            files_map["compare.ticker_time_issues_combined"] = "compare/ticker_time_issues_combined.csv"
    elif mode_key == "strategy_compare":
        if not strategy_compare_results:
            raise ValueError("strategy_compare export requires strategy_compare_results")
        runs_dir = root / "runs"
        ensure_dir(runs_dir)
        mode_label_map = {
            "screen": "스크리너",
            "strategy": "전략",
        }
        for mk in ["screen", "strategy"]:
            res = strategy_compare_results.get(mk)
            if res is None:
                checks.append({"name": f"{mk}.result_exists", "status": "fail", "details": "missing result"})
                continue
            _write_result_files(res, runs_dir / mk, files_map, f"runs.{mk}")
            run_checks, ticker_issues, snapshot_check = _run_consistency_checks(
                res,
                mk,
                market=market,
                price_root=price_root,
                overrides_path=override_path,
                mode_label=mode_label_map.get(mk, mk),
                run_dir=runs_dir / mk,
                snapshot_validation_mode=snapshot_validation_mode,
                sector_validation_mode=sector_validation_mode,
                require_sp500_pit_evidence=require_sp500_pit_evidence,
                sp500_pit_validation_mode=sp500_pit_validation_mode,
                sp500_pit_min_confidence=sp500_pit_min_confidence,
            )
            checks.extend(run_checks)
            snapshot_status_by_mode[mk] = str(snapshot_check.get("status", "warn"))
            if not ticker_issues.empty:
                issue_path = runs_dir / mk / "ticker_time_issues.csv"
                _frame_to_iso_csv(ticker_issues, issue_path)
                files_map[f"runs.{mk}.ticker_time_issues"] = f"runs/{mk}/ticker_time_issues.csv"
                ticker_issue_frames.append(ticker_issues.copy())
            compare_rows.append({"mode_key": mk, "mode_label": mode_label_map.get(mk, mk), **_core_metrics(res.metrics)})
        compare_dir = root / "compare"
        ensure_dir(compare_dir)
        cmp_df = pd.DataFrame(compare_rows)
        _frame_to_iso_csv(cmp_df, compare_dir / "compare_metrics.csv")
        (compare_dir / "compare_metrics.json").write_text(json.dumps(_json_safe(compare_rows), ensure_ascii=False, indent=2), encoding="utf-8")
        files_map["compare.metrics_csv"] = "compare/compare_metrics.csv"
        files_map["compare.metrics_json"] = "compare/compare_metrics.json"
        if ticker_issue_frames:
            combined_ticker = pd.concat(ticker_issue_frames, ignore_index=True, sort=False)
            _frame_to_iso_csv(combined_ticker, compare_dir / "ticker_time_issues_combined.csv")
            files_map["compare.ticker_time_issues_combined"] = "compare/ticker_time_issues_combined.csv"
    else:
        raise ValueError(f"Unsupported export mode: {mode_key}")

    passed = sum(1 for c in checks if c.get("status") == "pass")
    warned = sum(1 for c in checks if c.get("status") == "warn")
    failed = sum(1 for c in checks if c.get("status") == "fail")
    consistency = {
        "checks": checks,
        "passed_count": passed,
        "warn_count": warned,
        "failed_count": failed,
    }
    (root / "consistency_checks.json").write_text(json.dumps(_json_safe(consistency), ensure_ascii=False, indent=2), encoding="utf-8")
    files_map["consistency_checks"] = "consistency_checks.json"

    mode_norm = str(snapshot_validation_mode or "warn").strip().lower()
    snapshot_fail_checks = [
        c
        for c in checks
        if str(c.get("name", "")).endswith("rebalance_snapshots_integrity") and str(c.get("status", "")).lower() == "fail"
    ]
    if mode_norm == "fail" and snapshot_fail_checks:
        targets = ", ".join(str(c.get("name", "")) for c in snapshot_fail_checks)
        raise RuntimeError(
            "Snapshot validation failed in fail-closed mode; "
            f"bundle export aborted. failed_checks=[{targets}]"
        )
    sector_mode_norm = str(sector_validation_mode or "warn").strip().lower()
    sector_fail_checks = [
        c
        for c in checks
        if str(c.get("name", "")).endswith("sector_snapshot_evidence") and str(c.get("status", "")).lower() == "fail"
    ]
    if sector_mode_norm == "fail" and sector_fail_checks:
        targets = ", ".join(str(c.get("name", "")) for c in sector_fail_checks)
        raise RuntimeError(
            "Sector snapshot evidence validation failed in fail-closed mode; "
            f"bundle export aborted. failed_checks=[{targets}]"
        )
    sp500_mode_norm = str(sp500_pit_validation_mode or "warn").strip().lower()
    sp500_fail_checks = [
        c
        for c in checks
        if str(c.get("name", "")).endswith("sp500_pit_snapshot_evidence") and str(c.get("status", "")).lower() == "fail"
    ]
    if sp500_mode_norm == "fail" and sp500_fail_checks:
        targets = ", ".join(str(c.get("name", "")) for c in sp500_fail_checks)
        raise RuntimeError(
            "SP500 PIT snapshot evidence validation failed in fail-closed mode; "
            f"bundle export aborted. failed_checks=[{targets}]"
        )

    strategy_ctx = _json_safe(strategy_context or {})
    funding_ctx = _json_safe(funding_context or {})
    results_overview: dict[str, Any] = {"single": None, "compare": []}
    if mode_key == "single" and single_result is not None:
        results_overview["single"] = {
            "mode_label": "single",
            "mode_key": "single",
            "metrics": _core_metrics(single_result.metrics),
        }
    if mode_key in {"funding_compare", "strategy_compare"}:
        schema_keys = list(_core_metrics({}).keys())
        normalized_compare: list[dict[str, Any]] = []
        for row in compare_rows:
            metrics_map = {k: row.get(k) for k in schema_keys}
            normalized_compare.append(
                {
                    "mode_label": row.get("mode_label"),
                    "mode_key": row.get("mode_key"),
                    "metrics": _json_safe(metrics_map),
                }
            )
        results_overview["compare"] = normalized_compare

    sp500_issue_count = 0
    for c in checks:
        name = str(c.get("name", ""))
        if name.endswith("sp500_pit_membership_consistency"):
            summary = c.get("summary", {}) if isinstance(c.get("summary"), dict) else {}
            sp500_issue_count += int(summary.get("total_issues", 0) or 0)

    sp500_results_map: dict[str, BacktestResult] = {}
    if mode_key == "single" and single_result is not None:
        sp500_results_map["single"] = single_result
    elif mode_key == "funding_compare" and funding_compare_results:
        sp500_results_map = dict(funding_compare_results)
    elif mode_key == "strategy_compare" and strategy_compare_results:
        sp500_results_map = dict(strategy_compare_results)
    sp500_snapshot_stats = _collect_sp500_pit_snapshot_stats(sp500_results_map)
    sp500_validation_summary = _summarize_sp500_pit_checks(checks)

    summary_json = {
        "bundle_version": "1.0",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "app_version": app_version,
        "mode": mode_key,
        "strategy_context": strategy_ctx,
        "funding_context": funding_ctx,
        "results_overview": results_overview,
        "files": files_map,
        "consistency_checks_summary": {
            "passed": failed == 0,
            "warnings": [c for c in checks if c.get("status") == "warn"],
            "failed_checks": [c for c in checks if c.get("status") == "fail"],
        },
        "sp500_pit_validation_summary": sp500_validation_summary,
        "sp500_pit_issue_count": int(sp500_issue_count),
        "sp500_pit_source_mix": sp500_snapshot_stats.get("source_mix", {}),
        "sp500_pit_confidence_summary": sp500_snapshot_stats.get("confidence_summary", {}),
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit_hash(),
        },
        "snapshot_validation": {
            "mode": snapshot_validation_mode,
            "sector_mode": sector_validation_mode,
            "sp500_pit_mode": sp500_pit_validation_mode,
            "status_by_mode": snapshot_status_by_mode,
        },
    }
    (root / "ai_review_summary.json").write_text(json.dumps(_json_safe(summary_json), ensure_ascii=False, indent=2), encoding="utf-8")
    files_map["ai_review_summary_json"] = "ai_review_summary.json"

    if strategy_context is not None:
        (root / "strategy_config_effective.json").write_text(json.dumps(_json_safe(strategy_context), ensure_ascii=False, indent=2), encoding="utf-8")
        files_map["strategy_config_effective"] = "strategy_config_effective.json"
    if ui_context_snapshot is not None:
        (root / "ui_context_snapshot.json").write_text(json.dumps(_json_safe(ui_context_snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
        files_map["ui_context_snapshot"] = "ui_context_snapshot.json"

    if compare_notes is not None:
        compare_dir = root / "compare"
        ensure_dir(compare_dir)
        (compare_dir / "compare_notes.json").write_text(json.dumps(_json_safe(compare_notes), ensure_ascii=False, indent=2), encoding="utf-8")
        files_map["compare.notes"] = "compare/compare_notes.json"

    md_lines: list[str] = []
    md_lines.append("# AI Review Summary")
    md_lines.append("")
    md_lines.append(f"- mode: `{mode_key}`")
    md_lines.append(f"- created_at: `{summary_json['created_at']}`")
    md_lines.append(f"- app_version: `{app_version}`")
    md_lines.append("")
    md_lines.append("## 실행 요약")
    md_lines.append(f"- strategy: `{strategy_ctx.get('name', '')}`")
    md_lines.append(f"- market: `{strategy_ctx.get('market', '')}`")
    md_lines.append(f"- period: `{strategy_ctx.get('start', '')}` ~ `{strategy_ctx.get('end', '')}`")
    md_lines.append(f"- rebalance: `{strategy_ctx.get('rebalance_frequency', '')}`")
    md_lines.append(f"- execution_timing: `{strategy_ctx.get('execution_timing', '')}`")
    md_lines.append(f"- funding_mode: `{funding_ctx.get('funding_mode', '')}`")
    md_lines.append("")
    md_lines.append("## 방식별 핵심 성과")
    if results_overview["single"] is not None:
        s = results_overview["single"]
        m = s["metrics"]
        md_lines.append(f"- single: TWR CAGR={m.get('twr_cagr')}, IRR={m.get('mwr_irr')}, total_contributed={m.get('total_contributed')}, ending_value={m.get('ending_value')}, pnl={m.get('pnl')}")
    else:
        for row in results_overview["compare"]:
            m = row["metrics"]
            md_lines.append(f"- {row.get('mode_label')}({row.get('mode_key')}): TWR CAGR={m.get('twr_cagr')}, IRR={m.get('mwr_irr')}, total_contributed={m.get('total_contributed')}, ending_value={m.get('ending_value')}, pnl={m.get('pnl')}")
    md_lines.append("")
    md_lines.append("## 검증 포인트 체크리스트")
    md_lines.append("- 손익 = 최종평가금액 - 누적납입원금")
    md_lines.append("- funding 순서 = funding -> rebalance (same-day)")
    md_lines.append("- equity = cash + positions_value")
    md_lines.append("- VA min/max cap 적용 여부(reason/floor/max_cap)")
    ticker_checks = [c for c in checks if str(c.get("name", "")).endswith("ticker_time_consistency")]
    if ticker_checks:
        md_lines.append("")
        md_lines.append("## 티커-시점 불일치 검사")
        for chk in ticker_checks:
            md_lines.append(f"- {chk.get('name')}: {chk.get('status')} ({chk.get('details', '')})")
            for ex in (chk.get("examples", []) or [])[:3]:
                md_lines.append(
                    f"  - {ex.get('mode_label','')} | {ex.get('ticker','')} | trade={ex.get('trade_date','')} | "
                    f"first_valid={ex.get('first_valid_date','')} | delta_days={ex.get('delta_days_from_first','')}"
                )
    snapshot_checks = [c for c in checks if str(c.get("name", "")).endswith("rebalance_snapshots_integrity")]
    if snapshot_checks:
        md_lines.append("")
        md_lines.append("## 리밸런싱 스냅샷 무결성")
        for chk in snapshot_checks:
            status = str(chk.get("status", "warn")).upper()
            counts = chk.get("counts", {})
            if status == "PASS":
                md_lines.append(
                    f"- {chk.get('name')}: PASS (전략 조건 재검산 가능) "
                    f"[index={counts.get('snapshot_index_rows', 0)}, missing={counts.get('missing_snapshot_files', 0)}]"
                )
            else:
                md_lines.append(
                    f"- {chk.get('name')}: {status} (전략 조건 수치 재검산 제한) "
                    f"[index={counts.get('snapshot_index_rows', 0)}, missing={counts.get('missing_snapshot_files', 0)}]"
                )
                for ex in (chk.get("examples", []) or [])[:3]:
                    md_lines.append(f"  - row={ex.get('row', '')} issue={ex.get('issue', '')} path={ex.get('snapshot_path', '')}")
    sp500_snapshot_checks = [c for c in checks if str(c.get("name", "")).endswith("sp500_pit_snapshot_evidence")]
    if sp500_snapshot_checks:
        md_lines.append("")
        md_lines.append("## S&P500 PIT 스냅샷 근거")
        for chk in sp500_snapshot_checks:
            status = str(chk.get("status", "warn")).upper()
            counts = chk.get("counts", {})
            md_lines.append(
                f"- {chk.get('name')}: {status} "
                f"[index={counts.get('snapshot_index_rows', 0)}, missing={counts.get('missing_snapshot_files', 0)}]"
            )
    if summary_json.get("sp500_pit_source_mix") or summary_json.get("sp500_pit_confidence_summary", {}).get("count"):
        md_lines.append("")
        md_lines.append("## S&P500 PIT 품질 요약")
        md_lines.append(f"- validation: {summary_json.get('sp500_pit_validation_summary', {}).get('status')}")
        md_lines.append(f"- issue_count: {summary_json.get('sp500_pit_issue_count')}")
        src_mix = summary_json.get("sp500_pit_source_mix", {}) or {}
        if src_mix:
            md_lines.append("- source_mix:")
            for k, v in src_mix.items():
                md_lines.append(f"  - {k}: {v}")
        conf = summary_json.get("sp500_pit_confidence_summary", {}) or {}
        md_lines.append(
            "- confidence_summary: "
            f"count={conf.get('count')}, min={conf.get('min')}, median={conf.get('median')}, max={conf.get('max')}"
        )
    (root / "ai_review_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    files_map["ai_review_summary_md"] = "ai_review_summary.md"

    prompt_lines = [
        "# AI 검증 프롬프트 템플릿",
        "",
        "1. 먼저 `ai_review_summary.json`을 읽고 실행 컨텍스트를 파악하세요.",
        "2. `metrics.json`, `trades.csv`, `funding_flows.csv`, `equity_curve.csv`를 교차검증하세요.",
        "3. 아래 항목을 검증하세요:",
        "   - pnl = ending_value - total_contributed",
        "   - equity = cash + positions_value",
        "   - same-day funding/rebalance 순서 태그",
        "   - VA 최소/최대 cap 적용 여부(reason, floor_applied, max_cap_applied)",
        "4. 이상치/불일치가 있으면 재현 가능한 날짜/행을 함께 제시하세요.",
    ]
    (root / "validate_with_ai_prompt.md").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    files_map["validate_with_ai_prompt"] = "validate_with_ai_prompt.md"

    manifest = {
        "bundle_version": "1.0",
        "created_at": summary_json["created_at"],
        "mode": mode_key,
        "snapshot_validation_mode": snapshot_validation_mode,
        "sector_validation_mode": sector_validation_mode,
        "sp500_pit_validation_mode": sp500_pit_validation_mode,
        "snapshots_integrity_status": snapshot_status_by_mode,
        "files": files_map,
    }
    (root / "export_manifest.json").write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return root
