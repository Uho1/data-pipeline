from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from market_data.backtest.models import RankingConfig


def _winsorize(series: pd.Series, p_low: float | None, p_high: float | None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo = p_low
    hi = p_high
    if lo is None and hi is None:
        return s

    q_lo = float(s.quantile(lo)) if lo is not None else None
    q_hi = float(s.quantile(hi)) if hi is not None else None

    out = s.copy()
    if q_lo is not None and np.isfinite(q_lo):
        out = out.clip(lower=q_lo)
    if q_hi is not None and np.isfinite(q_hi):
        out = out.clip(upper=q_hi)
    return out


def _percentile_score(series: pd.Series, direction: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    pct = s.rank(method="average", pct=True, ascending=True)
    direction_norm = str(direction or "asc").strip().lower()
    if direction_norm == "asc":
        score = (1.0 - pct) * 100.0
    else:
        score = pct * 100.0
    return score


def rank_cross_section(
    frame: pd.DataFrame,
    ranking: RankingConfig,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if not ranking.factors:
        out = frame.copy()
        out["rank_score"] = np.nan
        return out

    out = frame.copy()
    score_cols: list[str] = []

    for factor in ranking.factors:
        col = str(factor.name)
        if col not in out.columns:
            out[f"score_{col}"] = np.nan
            continue
        w_series = _winsorize(out[col], factor.winsorize_low, factor.winsorize_high)
        score = _percentile_score(w_series, factor.direction)
        out[f"score_{col}"] = score
        score_cols.append(col)

    weighted_sum = pd.Series(0.0, index=out.index, dtype=float)
    weight_sum = pd.Series(0.0, index=out.index, dtype=float)

    for factor in ranking.factors:
        col = str(factor.name)
        score_col = f"score_{col}"
        if score_col not in out.columns:
            continue
        score = pd.to_numeric(out[score_col], errors="coerce")
        valid = score.notna()
        w = float(factor.weight)
        if not np.isfinite(w) or w <= 0:
            continue
        weighted_sum.loc[valid] = weighted_sum.loc[valid] + (score.loc[valid] * w)
        weight_sum.loc[valid] = weight_sum.loc[valid] + w

    out["rank_score"] = (weighted_sum / weight_sum.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out = out.sort_values("rank_score", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def ranking_to_dict(ranking: RankingConfig) -> dict:
    return {
        "normalization": ranking.normalization,
        "factors": [asdict(f) for f in ranking.factors],
    }
