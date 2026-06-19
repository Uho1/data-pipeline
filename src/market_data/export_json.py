"""Export local market-data storage to per-ticker JSON files + ticker_master meta.

Usage:
    python -m market_data.export_json [--market kr] [--tickers 005930,000270]

Reads from local parquet-backed storage (plus compatibility readers) and writes:
  data/tickers/{market}/{ticker}.json   (one per ticker)
  data/meta/ticker_master_{market}.json (full ticker list)
  data/meta/last_updated.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from market_data.config import STORAGE_BACKEND
from market_data.fiscal_periods import infer_fiscal_period_meta
from market_data.valuation_ttm import build_valuation_ttm_payload

# ---------------------------------------------------------------------------
# Paths — always resolve to the MAIN repo root (not worktree)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = Path(os.environ.get("MDL_EXPORT_DIR", _REPO_ROOT / "data"))
_TICKERS_DIR = _DATA_ROOT / "tickers"
_META_DIR = _DATA_ROOT / "meta"
_CONFIG_DIR = _REPO_ROOT / "config"
_REFERENCE_DIR = _DATA_ROOT / "reference"

# Global date filter — set by run_export() via --start-date
_START_DATE: str = "2013-06-01"


def _safe_val(v: Any) -> Any:
    """Convert numpy/pandas types to JSON-safe Python types."""
    if v is None:
        return None
    # Handle pandas NA / NaT / numpy NaN
    try:
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()[:10]
    return v


def _series_to_list(s: pd.Series) -> list:
    """Convert a pandas Series to a JSON-safe list."""
    return [_safe_val(v) for v in s]


def _snap_to_quarter_end(ts: pd.Timestamp, max_days: int = 15) -> pd.Timestamp:
    """Snap a timestamp to the nearest quarter-end (3/31, 6/30, 9/30, 12/31).

    Only snaps if within *max_days* of a quarter-end; otherwise returns the
    original timestamp unchanged.
    """
    if pd.isna(ts):
        return ts
    year, month = ts.year, ts.month
    quarter = (month - 1) // 3 + 1
    end_month = quarter * 3
    qe = pd.Timestamp(year, end_month, {3: 31, 6: 30, 9: 30, 12: 31}[end_month])
    if abs((ts - qe).days) <= max_days:
        return qe
    return ts


def _sanitize_income_series(
    income: dict[str, list],
    df: pd.DataFrame,
    *,
    market: str = "kr",
) -> None:
    """Post-process income metrics to fix three anomaly types in-place.

    1. Q4 annual cumulative: FY value appearing as Q4 (~4x spike).
       Fix: replace with FY - (Q1+Q2+Q3) if prior quarters exist, else null.
    2. Extreme unit spike: value >20x neighbors from unit mismatch.
       Fix: try dividing by 1e3 or 1e6; null if no factor works.
    3. Negative revenue: impossible values from YTD deaccumulation errors.
       Fix: null out.
    """
    rev = income.get("revenue")
    if not rev or len(rev) < 4:
        return

    n = len(rev)
    fiscal_q = list(df["fiscal_quarter"]) if "fiscal_quarter" in df.columns else [None] * n
    form_type = list(df["FormType"]) if "FormType" in df.columns else [""] * n

    # --- Fix 3: Negative revenue → null (simplest, do first) ---
    for i in range(n):
        if rev[i] is not None and rev[i] < 0:
            rev[i] = None

    # Build numeric array for neighbor comparisons
    import numpy as _np
    nums = _np.array([float(v) if v is not None else _np.nan for v in rev])

    # --- Fix 1: Q4 annual cumulative ---
    # Detect: Q4 (or FY) value that is ~3-6x the median of neighboring quarters
    for i in range(1, n):
        if _np.isnan(nums[i]):
            continue
        fq = fiscal_q[i] if i < len(fiscal_q) else None
        ft = str(form_type[i] if i < len(form_type) else "").upper()
        if fq != 4 and ft != "FY":
            continue

        # Gather up to 4 neighboring quarterly values (non-Q4)
        neighbors = []
        for j in [i - 1, i - 2, i - 3, i + 1]:
            if 0 <= j < n and not _np.isnan(nums[j]):
                jq = fiscal_q[j] if j < len(fiscal_q) else None
                jt = str(form_type[j] if j < len(form_type) else "").upper()
                if jq != 4 and jt != "FY":
                    neighbors.append(nums[j])
        if len(neighbors) < 2:
            continue

        median_n = _np.median(neighbors)
        if median_n <= 0:
            continue
        ratio = nums[i] / median_n

        # US needs higher threshold — biotech milestones cause legitimate Q4 spikes
        q4_spike_threshold = 2.8 if market == "kr" else 5.0
        if ratio < q4_spike_threshold:
            continue  # Not a spike

        # Try Q4 = FY - (Q1+Q2+Q3) if 3 prior quarters exist within same fiscal year
        corrected = False
        if i >= 3:
            prior_3 = [nums[i - 3], nums[i - 2], nums[i - 1]]
            if all(not _np.isnan(p) for p in prior_3):
                q4_est = nums[i] - sum(prior_3)
                if median_n > 0 and 0.1 <= q4_est / median_n <= 4.0:
                    # Apply correction to all income metrics at this index
                    fy_val = nums[i]
                    prior_sum = sum(prior_3)
                    for key, vals in income.items():
                        if i >= len(vals) or vals[i] is None:
                            continue
                        # Compute same-metric Q1+Q2+Q3
                        prior_metric = [vals[i - 3], vals[i - 2], vals[i - 1]]
                        if all(v is not None for v in prior_metric):
                            vals[i] = vals[i] - sum(prior_metric)
                        else:
                            vals[i] = None
                    corrected = True
                    nums[i] = q4_est  # update local tracker

        if not corrected and ratio > q4_spike_threshold:
            # Can't deaccumulate — null out the spiking metrics
            for key, vals in income.items():
                if i < len(vals) and vals[i] is not None:
                    if median_n > 0 and abs(vals[i]) / median_n > 2.5:
                        pass  # only null revenue for now to be conservative
            rev[i] = None
            nums[i] = _np.nan

    # --- Fix 2: Extreme unit spike (unit mismatch) ---
    # US uses much higher threshold — biotech milestones cause legitimate 100x+ spikes
    unit_spike_threshold = 20 if market == "kr" else 500
    nums = _np.array([float(v) if v is not None else _np.nan for v in rev])
    for i in range(1, n - 1):
        if _np.isnan(nums[i]):
            continue
        neighbors = []
        for j in [i - 1, i - 2, i + 1, i + 2]:
            if 0 <= j < n and not _np.isnan(nums[j]):
                neighbors.append(nums[j])
        if len(neighbors) < 2:
            continue
        median_n = _np.median([abs(x) for x in neighbors if x != 0])
        if median_n <= 0:
            continue
        ratio = abs(nums[i]) / median_n
        if ratio < unit_spike_threshold:
            continue

        # Try scale factors: 1e3, 1e6
        # For US, also require that the original ratio is close to a power of 1000
        # to avoid false positives on legitimate milestone revenue
        fixed = False
        for factor in (1e3, 1e6):
            adjusted = nums[i] / factor
            adj_ratio = abs(adjusted) / median_n
            ratio_to_factor = ratio / factor
            # After adjustment, value should be reasonable vs neighbors
            if 0.2 <= adj_ratio <= 5.0:
                # For US: only scale if the ratio itself is close to the factor
                # (e.g., 987x spike → factor 1000 is plausible; 901x → could be real)
                if market != "kr" and abs(ratio / factor - 1.0) > 0.5:
                    continue  # ratio doesn't match factor well enough
                for key, vals in income.items():
                    if i < len(vals) and vals[i] is not None:
                        vals[i] = vals[i] / factor
                nums[i] = adjusted
                fixed = True
                break
        if not fixed and market == "kr":
            rev[i] = None
            nums[i] = _np.nan


def _collapse_financial_period_rows(
    df: pd.DataFrame | None,
    *,
    prefer_latest_dates: bool = True,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "PeriodEnd" not in out.columns:
        return out
    raw_dates = pd.to_datetime(out.get("PeriodEnd"), errors="coerce").dt.normalize()
    out["__period_key"] = raw_dates.apply(_snap_to_quarter_end)
    out = out.loc[out["__period_key"].notna()].copy()
    if out.empty:
        return out.drop(columns=["__period_key"], errors="ignore")

    meta_cols = {
        "ticker",
        "market",
        "PeriodEnd",
        "PeriodStart",
        "StatementDate",
        "FormType",
        "FilingDate",
        "AcceptedAt",
        "AvailableDate",
        "AvailabilityMethod",
        "term",
        "fiscal_year",
        "fiscal_quarter",
        "fiscal_label",
        "name",
        "industry",
        "sector",
        "subsector",
        "source",
    }
    score_cols = [col for col in out.columns if col not in meta_cols and not col.startswith("__")]
    if score_cols:
        out["__nonnull_score"] = out[score_cols].notna().sum(axis=1)
    else:
        out["__nonnull_score"] = 0
    for col in ("AvailableDate", "FilingDate", "AcceptedAt", "StatementDate"):
        if col in out.columns:
            out[f"__sort_{col}"] = pd.to_datetime(out[col], errors="coerce")
        else:
            out[f"__sort_{col}"] = pd.NaT
    sort_cols = [
        "__period_key",
        "__nonnull_score",
        "__sort_StatementDate",
        "__sort_FilingDate",
        "__sort_AvailableDate",
        "__sort_AcceptedAt",
    ]
    ascending = [True, True, True, True, True, True]
    if prefer_latest_dates:
        ascending = [True, True, True, True, True, True]
    out = out.sort_values(sort_cols, ascending=ascending)
    out = out.drop_duplicates(subset=["__period_key"], keep="last")
    return out.drop(columns=[col for col in out.columns if col.startswith("__")], errors="ignore").reset_index(drop=True)


def _ensure_fiscal_metadata(
    df: pd.DataFrame | None,
    *,
    force_recompute: bool = False,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    out = df.copy()
    has_fiscal = {"fiscal_year", "fiscal_quarter", "fiscal_label"}.issubset(out.columns)
    if force_recompute or not has_fiscal or out.get("fiscal_label") is None or pd.Series(out.get("fiscal_label")).dropna().empty:
        fiscal_meta = infer_fiscal_period_meta(out.get("PeriodEnd"), out.get("FormType"), out.get("PeriodStart"))
        if not fiscal_meta.empty:
            fiscal_meta = fiscal_meta.drop_duplicates(subset=["period_end"], keep="last").set_index("period_end")
            period_index = pd.to_datetime(out.get("PeriodEnd"), errors="coerce").dt.normalize()
            for col in ("fiscal_year", "fiscal_quarter", "fiscal_label"):
                out[col] = fiscal_meta[col].reindex(period_index).to_numpy()
    if "fiscal_label" in out.columns:
        if force_recompute:
            out["term"] = out["fiscal_label"]
        elif "term" not in out.columns:
            out["term"] = out["fiscal_label"]
        else:
            term_series = pd.Series(out["term"], index=out.index, dtype="object")
            fiscal_series = pd.Series(out["fiscal_label"], index=out.index, dtype="object")
            blank_mask = term_series.isna() | term_series.astype("string").isin(["", "nan", "<NA>"])
            term_series.loc[blank_mask] = fiscal_series.loc[blank_mask]
            out["term"] = term_series
    return out


def _attach_fiscal_block_metadata(result: dict[str, Any], df: pd.DataFrame) -> None:
    for col in ("term", "fiscal_year", "fiscal_quarter", "fiscal_label"):
        if col in df.columns:
            result[col] = _series_to_list(df[col])


def _parse_fiscal_label(text: Any) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d{4})Q([1-4])", str(text or "").strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _quarter_ord(year: int, quarter: int) -> int:
    return int(year) * 4 + int(quarter)


def _fiscal_quality_score(labels: list[Any]) -> tuple[int, int, int]:
    parsed = [_parse_fiscal_label(label) for label in labels]
    invalid = sum(value is None for value in parsed)
    valid = [value for value in parsed if value is not None]
    if not valid:
        return (invalid, 10_000, 10_000)
    counts = pd.Series(valid).value_counts()
    duplicates = int((counts > 1).sum())
    uniq = sorted(set(valid))
    missing = 0
    year, quarter = uniq[0]
    target = uniq[-1]
    while (year, quarter) != target:
        if quarter < 4:
            quarter += 1
        else:
            year += 1
            quarter = 1
        if (year, quarter) not in counts.index:
            missing += 1
    return (invalid, duplicates, missing)


def _financial_block_quality_score(block: dict[str, Any] | None) -> tuple[int, int, int, int]:
    if not block:
        return (10_000, 10_000, 10_000, 10_000)
    labels = list(block.get("fiscal_label") or [])
    invalid, duplicates, missing = _fiscal_quality_score(labels)
    valid = [label for label in labels if _parse_fiscal_label(label) is not None]
    unique_count = len(set(valid))
    # Prefer fewer missing/duplicates first, then more usable quarters.
    return (missing, duplicates, invalid, -unique_count)


def _merge_financial_blocks_by_fiscal_label(
    primary: dict[str, Any] | None,
    supplemental: dict[str, Any] | None,
    *,
    market: str,
) -> dict[str, Any] | None:
    if not primary:
        return supplemental
    if not supplemental:
        return primary

    primary_labels = list(primary.get("fiscal_label") or [])
    supplemental_labels = list(supplemental.get("fiscal_label") or [])
    merged_labels: list[str] = []
    for label in primary_labels + supplemental_labels:
        if _parse_fiscal_label(label) and label not in merged_labels:
            merged_labels.append(label)
    if not merged_labels:
        return primary
    merged_labels = sorted(merged_labels, key=lambda item: _quarter_ord(*_parse_fiscal_label(item)))

    def _series_map(block: dict[str, Any], values: list[Any]) -> dict[str, Any]:
        return {
            str(label): value
            for label, value in zip(block.get("fiscal_label") or [], values, strict=False)
            if _parse_fiscal_label(label)
        }

    out: dict[str, Any] = {
        "fiscal_label": merged_labels,
        "term": [],
        "fiscal_year": [],
        "fiscal_quarter": [],
        "periods": [],
    }
    primary_period_map = _series_map(primary, list(primary.get("periods") or []))
    supplemental_period_map = _series_map(supplemental, list(supplemental.get("periods") or []))
    for label in merged_labels:
        parsed = _parse_fiscal_label(label)
        if parsed is None:
            continue
        year, quarter = parsed
        out["fiscal_year"].append(year)
        out["fiscal_quarter"].append(quarter)
        out["term"].append(f"Q{quarter}" if market == "kr" else label)
        out["periods"].append(primary_period_map.get(label, supplemental_period_map.get(label)))

    for section in ("income", "balance", "cashflow"):
        payload = primary.get(section)
        supplemental_payload = supplemental.get(section)
        merged_payload: dict[str, list[Any]] = {}
        keys = set(payload.keys() if isinstance(payload, dict) else []) | set(
            supplemental_payload.keys() if isinstance(supplemental_payload, dict) else []
        )
        for key in keys:
            primary_values = _series_map(primary, list((payload or {}).get(key, [])))
            supplemental_values = _series_map(supplemental, list((supplemental_payload or {}).get(key, [])))
            merged_payload[key] = [
                primary_values.get(label, supplemental_values.get(label))
                for label in merged_labels
            ]
        if merged_payload:
            out[section] = merged_payload

    for key in ("shares_outstanding", "market_cap"):
        primary_values = _series_map(primary, list(primary.get(key, []))) if isinstance(primary.get(key), list) else {}
        supplemental_values = _series_map(supplemental, list(supplemental.get(key, []))) if isinstance(supplemental.get(key), list) else {}
        if primary_values or supplemental_values:
            out[key] = [primary_values.get(label, supplemental_values.get(label)) for label in merged_labels]
    return out


def _choose_preferred_fiscal_frame(
    existing_frame: pd.DataFrame,
    recomputed_frame: pd.DataFrame,
) -> pd.DataFrame:
    existing_labels = existing_frame.get("fiscal_label", pd.Series(dtype=object)).tolist()
    recomputed_labels = recomputed_frame.get("fiscal_label", pd.Series(dtype=object)).tolist()
    existing_score = _fiscal_quality_score(existing_labels)
    recomputed_score = _fiscal_quality_score(recomputed_labels)
    if recomputed_score < existing_score:
        return recomputed_frame
    return existing_frame


def _build_financials_block_from_normalized_df(
    df: pd.DataFrame,
    *,
    market: str,
    price_df: pd.DataFrame | None = None,
) -> dict | None:
    if df is None or df.empty:
        return None
    out_df = df.copy()
    out_df = _ensure_fiscal_metadata(out_df)
    if "PeriodEnd" not in out_df.columns:
        return None
    out_df["PeriodEnd"] = pd.to_datetime(out_df["PeriodEnd"], errors="coerce")
    if _START_DATE:
        out_df = out_df[out_df["PeriodEnd"] >= pd.Timestamp(_START_DATE)]
    out_df = out_df.sort_values("PeriodEnd")
    if out_df.empty:
        return None
    periods = [_safe_val(d) for d in out_df["PeriodEnd"]]
    result: dict[str, Any] = {"periods": periods}
    _attach_fiscal_block_metadata(result, out_df)

    income: dict[str, list] = {}
    for key, col in _INCOME_METRICS.items():
        if market == "kr" and key == "rd":
            continue
        if col in out_df.columns:
            income[key] = _series_to_list(out_df[col])

    # Sanitize income: Q4 annual cumulative, unit spikes, negative revenue
    _sanitize_income_series(income, out_df, market=market)

    if income:
        result["income"] = income

    balance: dict[str, list] = {}
    for key, col in _BALANCE_METRICS.items():
        if col in out_df.columns:
            balance[key] = _series_to_list(out_df[col])
    if balance:
        result["balance"] = balance

    cashflow: dict[str, list] = {}
    for key, col in _CASHFLOW_METRICS.items():
        if col in out_df.columns:
            cashflow[key] = _series_to_list(out_df[col])
    if cashflow:
        result["cashflow"] = cashflow

    for shares_col in ("Shares", "Diluted Shares", "Basic Shares"):
        if shares_col in out_df.columns:
            vals = _series_to_list(out_df[shares_col])
            if any(v is not None for v in vals):
                result["shares_outstanding"] = vals
                break

    # Fallback: if financials have no shares data, pull from price_df (pykrx/parquet)
    if "shares_outstanding" not in result and price_df is not None and "SharesOutstanding" in price_df.columns:
        shares_from_price: list[Any] = []
        for p in periods:
            try:
                target = pd.Timestamp(p)
                mask = price_df.index <= target
                if mask.any():
                    shares_from_price.append(_safe_val(price_df.loc[mask, "SharesOutstanding"].iloc[-1]))
                else:
                    shares_from_price.append(None)
            except Exception:
                shares_from_price.append(None)
        if any(v is not None for v in shares_from_price):
            result["shares_outstanding"] = shares_from_price

    # Market cap: prefer MarketCap column from prices, fallback to shares × Close
    mcaps: list[Any] = []
    _close_col = None
    for _cc in ("Close", "Adj Close", "close"):
        if price_df is not None and _cc in price_df.columns:
            _close_col = _cc
            break

    if price_df is not None and not price_df.empty and "MarketCap" in price_df.columns:
        for p in periods:
            try:
                target = pd.Timestamp(p)
                mask = price_df.index <= target
                if mask.any():
                    mcaps.append(_safe_val(price_df.loc[mask, "MarketCap"].iloc[-1]))
                else:
                    mcaps.append(None)
            except Exception:
                mcaps.append(None)
    elif "shares_outstanding" in result and _close_col is not None:
        # Fallback: shares × close price at each period end
        shares_list = result["shares_outstanding"]
        for i, p in enumerate(periods):
            try:
                target = pd.Timestamp(p)
                mask = price_df.index <= target
                s = shares_list[i] if i < len(shares_list) else None
                if mask.any() and s is not None:
                    close_val = float(price_df.loc[mask, _close_col].iloc[-1])
                    mcaps.append(_safe_val(s * close_val))
                else:
                    mcaps.append(None)
            except Exception:
                mcaps.append(None)

    if mcaps and any(v is not None for v in mcaps):
        result["market_cap"] = mcaps
    return result




# ---------------------------------------------------------------------------
# Income / Balance / Cashflow metric grouping
# ---------------------------------------------------------------------------
_INCOME_METRICS = {
    "revenue": "Revenue",
    "cogs": "COGS",
    "gross_profit": "Gross Profit",
    "sga": "SG&A",
    "rd": "R&D",
    "operating_income": "Operating Income",
    "net_income": "Net Income",
    "eps": "EPS",
    "diluted_eps": "Diluted EPS",
    "da": "D&A",
    "sbc": "SBC",
    "interest": "Interest",
    "pretax_income": "Pretax Income",
    "tax": "Tax",
}

_BALANCE_METRICS = {
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities",
    "shareholders_equity": "Shareholders Equity",
    "current_assets": "Current Assets",
    "current_liabilities": "Current Liabilities",
    "ar": "AR",
    "ap": "AP",
    "inventory": "Inventory",
    "cash": "Cash",
    "debt_short": "Debt Short",
    "debt_long": "Debt Long",
    "deferred_revenue": "Deferred Revenue",
    "goodwill": "Goodwill",
    "intangibles": "Intangibles",
}

_CASHFLOW_METRICS = {
    "cfo": "Operating Cash Flow",
    "cfi": "Investing Cash Flow",
    "cff": "Financing Cash Flow",
    "capex": "Capital Expenditure",
    "ppe_capex": "PPE CapEx",
    "dividends_paid": "Dividends Paid",
    "repurchases": "Repurchases",
}

# ---------------------------------------------------------------------------
# US direct DuckDB access (no db_reader_kr)
# ---------------------------------------------------------------------------

def _us_connect(db_type: str = "main"):
    """Get a read-only DuckDB connection for US data (supports parquet backend)."""
    from market_data.config import STORAGE_BACKEND
    if STORAGE_BACKEND == "parquet":
        import duckdb
        from market_data.parquet_views import register_parquet_views
        con = duckdb.connect(":memory:")
        register_parquet_views(con, market="us", db_type=db_type)
        return con
    import duckdb
    data_dir = os.environ.get("MDL_DATA_DIR", str(_REPO_ROOT / "data"))
    if db_type == "prices":
        path = os.path.join(data_dir, "market_data_prices.duckdb")
    else:
        path = os.path.join(data_dir, "market_data.duckdb")
    return duckdb.connect(path, read_only=True)


def _build_prices_block_us(ticker: str) -> tuple[dict | None, pd.DataFrame | None]:
    """Build prices block for US ticker. Returns (block, raw_df)."""
    try:
        from market_data import db_reader

        loaded = db_reader.load_price_from_db(ticker=ticker, market="us")
        if loaded is None:
            return None, None
        df, _market = loaded
    except Exception:
        return None, None
    if df is None or df.empty:
        return None, None
    if _START_DATE:
        df = df[df.index >= pd.Timestamp(_START_DATE)]
    if df.empty:
        return None, None
    dates = [_safe_val(d) for d in df.index]
    out: dict[str, Any] = {"dates": dates}
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "MarketCap": "market_cap",
    }
    for src, dst in col_map.items():
        if src in df.columns:
            out[dst] = _series_to_list(df[src])
    return out, df.copy()


def _build_financials_block_us(ticker: str, price_df: pd.DataFrame | None = None) -> dict | None:
    """Build financials block for US ticker."""
    try:
        from market_data import db_reader

        df = db_reader.load_financials_from_db(ticker=ticker, market="us")
    except Exception:
        df = None

    sec_block: dict[str, Any] | None = None
    if df is not None and not df.empty and "PeriodEnd" in df.columns:
        df = _collapse_financial_period_rows(df)
        df_existing = _ensure_fiscal_metadata(df, force_recompute=False)
        df_recomputed = _ensure_fiscal_metadata(df, force_recompute=True)
        if df_existing is not None and df_recomputed is not None:
            df = _choose_preferred_fiscal_frame(df_existing, df_recomputed)
        elif df_recomputed is not None:
            df = df_recomputed
        else:
            df = df_existing
        sec_block = _build_financials_block_from_normalized_df(df, market="us", price_df=price_df)

    # Keep SEC output when it already forms a contiguous quarterly series.
    if sec_block is not None and _financial_block_quality_score(sec_block)[0] == 0:
        return sec_block

    yf_block = _build_financials_block_us_yfinance(ticker=ticker, price_df=price_df)
    if yf_block is None:
        return sec_block
    if sec_block is None:
        return yf_block

    sec_score = _financial_block_quality_score(sec_block)
    yf_score = _financial_block_quality_score(yf_block)
    merged_block = _merge_financial_blocks_by_fiscal_label(sec_block, yf_block, market="us")
    merged_score = _financial_block_quality_score(merged_block)
    if merged_score < sec_score:
        return merged_block
    if yf_score < sec_score and len(set(yf_block.get("fiscal_label") or [])) >= 4:
        return _merge_financial_blocks_by_fiscal_label(sec_block, yf_block, market="us")
    return sec_block


_YF_INCOME_MAP = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cogs": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "sga": [
        "Selling General And Administration",
        "General And Administrative Expense",
        "Selling And Marketing Expense",
    ],
    "rd": ["Research And Development"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported", "EBIT"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "eps": ["Diluted EPS", "Basic EPS"],
    "diluted_eps": ["Diluted EPS"],
    "da": ["Depreciation And Amortization", "Reconciled Depreciation"],
    "interest": ["Interest Expense", "Net Interest Income"],
    "pretax_income": ["Pretax Income"],
    "tax": ["Tax Provision"],
}

_YF_BALANCE_MAP = {
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "shareholders_equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "ar": ["Accounts Receivable", "Receivables"],
    "ap": ["Accounts Payable", "Payables"],
    "inventory": ["Inventory"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Federal Funds Sold"],
    "debt_short": ["Current Debt", "Current Debt And Capital Lease Obligation"],
    "debt_long": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "goodwill": ["Goodwill"],
    "intangibles": ["Other Intangible Assets"],
}

_YF_CASHFLOW_MAP = {
    "cfo": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "cfi": ["Investing Cash Flow", "Cash Flow From Continuing Investing Activities"],
    "cff": ["Financing Cash Flow", "Cash Flow From Continuing Financing Activities"],
    "capex": ["Capital Expenditure", "Purchase Of PPE"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "repurchases": ["Repurchase Of Capital Stock", "Common Stock Payments"],
}


def _first_present_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _build_financials_block_us_yfinance(ticker: str, price_df: pd.DataFrame | None = None) -> dict | None:
    try:
        from market_data.ingest import _download_financials_yfinance, _merge_yf_quarterly
    except Exception:
        return None

    try:
        fin_map = _download_financials_yfinance(ticker, retries=1, backoff=0.5, financial_workers=1)
        merged = _merge_yf_quarterly(fin_map)
    except Exception:
        return None
    if merged is None or merged.empty:
        return None

    df = merged.copy()
    if "StatementDate" not in df.columns:
        return None
    df["StatementDate"] = pd.to_datetime(df["StatementDate"], errors="coerce")
    df["PeriodEnd"] = pd.to_datetime(df.get("PeriodEnd", df["StatementDate"]), errors="coerce")
    df = df.loc[~df["PeriodEnd"].isna()].sort_values("PeriodEnd").reset_index(drop=True)
    if df.empty:
        return None

    out = pd.DataFrame(
        {
            "PeriodEnd": df["PeriodEnd"],
            "StatementDate": df["StatementDate"],
            "PeriodStart": pd.to_datetime(df.get("PeriodStart"), errors="coerce"),
            "FormType": df.get("FormType"),
            "FilingDate": pd.to_datetime(df.get("FilingDate"), errors="coerce"),
            "AcceptedAt": pd.to_datetime(df.get("AcceptedAt"), errors="coerce"),
            "AvailableDate": pd.to_datetime(df.get("AvailableDate"), errors="coerce"),
            "AvailabilityMethod": df.get("AvailabilityMethod"),
            "Source": "yfinance_quarterly",
        }
    )

    for target, candidates in _YF_INCOME_MAP.items():
        source_col = _first_present_column(df, candidates)
        if source_col:
            out[_INCOME_METRICS[target]] = pd.to_numeric(df[source_col], errors="coerce")
    for target, candidates in _YF_BALANCE_MAP.items():
        source_col = _first_present_column(df, candidates)
        if source_col:
            out[_BALANCE_METRICS[target]] = pd.to_numeric(df[source_col], errors="coerce")
    for target, candidates in _YF_CASHFLOW_MAP.items():
        source_col = _first_present_column(df, candidates)
        if source_col:
            out[_CASHFLOW_METRICS[target]] = pd.to_numeric(df[source_col], errors="coerce")

    shares_col = _first_present_column(df, ["Diluted Average Shares", "Basic Average Shares", "Ordinary Shares Number"])
    if shares_col:
        out["Shares"] = pd.to_numeric(df[shares_col], errors="coerce")
    return _build_financials_block_from_normalized_df(out, market="us", price_df=price_df)


def _get_ticker_info_us(ticker: str) -> dict[str, str]:
    """Get US ticker metadata."""
    if STORAGE_BACKEND == "parquet":
        try:
            from market_data import parquet_reader

            rows = parquet_reader.load_sec_issuer_registry(ticker=ticker, market="us")
            if rows is not None and not rows.empty:
                row = rows.iloc[0]
                return {
                    "name": str(row.get("company_name", "") or ""),
                    "sector": "",
                    "industry": "",
                    "subsector": "",
                    "market_tier": "NYSE/NASDAQ",
                }
        except Exception:
            pass
        return {"name": "", "sector": "", "industry": "", "subsector": "", "market_tier": ""}

    try:
        con = _us_connect("main")
        rows = con.execute(
            "SELECT company_name, gsector, ggroup, gind, gsubind FROM entity_master WHERE current_ticker = ?",
            [ticker]
        ).fetchdf()
        con.close()
        if not rows.empty:
            r = rows.iloc[0]
            return {
                "name": str(r.get("company_name", "") or ""),
                "sector": str(r.get("gsector", "") or ""),
                "industry": str(r.get("gind", "") or ""),
                "subsector": str(r.get("gsubind", "") or ""),
                "market_tier": "NYSE/NASDAQ",
            }
    except Exception:
        pass
    # Fallback to sec_issuer_registry
    try:
        con = _us_connect("prices")
        rows = con.execute(
            "SELECT company_name FROM sec_issuer_registry WHERE ticker = ?", [ticker]
        ).fetchdf()
        con.close()
        if not rows.empty:
            return {"name": str(rows.iloc[0]["company_name"] or ""), "sector": "", "industry": "", "subsector": "", "market_tier": ""}
    except Exception:
        pass
    return {"name": "", "sector": "", "industry": "", "subsector": "", "market_tier": ""}


# ---------------------------------------------------------------------------
# Export: single ticker
# ---------------------------------------------------------------------------

def _build_prices_block(ticker: str, market: str) -> tuple[dict | None, pd.DataFrame | None]:
    """Build the prices block from the reader layer."""
    try:
        from market_data import db_reader_kr

        result = db_reader_kr.load_price_from_db(ticker, market=market)
        if result is None:
            return None, None
        df, _resolved_market = result
    except Exception:
        return None, None

    if df.empty:
        return None, None

    # Apply date filter
    if _START_DATE:
        df = df[df.index >= pd.Timestamp(_START_DATE)]
    if df.empty:
        return None, None

    dates = [_safe_val(d) for d in df.index]
    out: dict[str, Any] = {"dates": dates}

    col_map = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close",
        "Volume": "volume", "MarketCap": "market_cap",
        "SharesOutstanding": "shares_outstanding",
    }
    for src_col, dst_key in col_map.items():
        if src_col in df.columns:
            out[dst_key] = _series_to_list(df[src_col])

    return out, df


# ---------------------------------------------------------------------------
# Pre-compute TTM (trailing 4-quarter rolling) and annual aggregations
# ---------------------------------------------------------------------------

# Flow metrics: sum over 4 quarters for TTM / sum per fiscal year for annual
_FLOW_KEYS = {
    "revenue", "cogs", "gross_profit", "sga", "rd", "operating_income",
    "net_income", "da", "sbc", "interest", "pretax_income", "tax",
    "ebitda", "ebit",
    "cfo", "cfi", "cff", "capex", "ppe_capex", "dividends_paid", "repurchases",
}

# Stock/ratio metrics: take last value of the 4-quarter window (or fiscal year)
_STOCK_KEYS = {
    "total_assets", "total_liabilities", "shareholders_equity",
    "current_assets", "current_liabilities", "ar", "ap", "inventory",
    "cash", "debt_short", "debt_long", "deferred_revenue",
    "goodwill", "intangibles", "shares_outstanding", "market_cap",
}


def _compute_ttm_block(quarterly_block: dict) -> dict:
    """Compute TTM (trailing 4-quarter rolling) from a quarterly block.

    Input: {"periods": [...], "income": {"revenue": [v1,v2,...], ...}, ...}
    Output: same structure but with TTM values (first 3 quarters will be null).
    """
    periods = quarterly_block.get("periods", [])
    if len(periods) < 4:
        return {}

    result: dict[str, Any] = {"periods": periods}

    for section_key in ("income", "balance", "cashflow"):
        section = quarterly_block.get(section_key)
        if not section:
            continue
        ttm_section: dict[str, list] = {}
        for metric_key, values in section.items():
            is_flow = metric_key in _FLOW_KEYS
            ttm_vals: list = []
            for i in range(len(values)):
                if i < 3:
                    ttm_vals.append(None)
                    continue
                window = values[i - 3 : i + 1]
                if is_flow:
                    # Sum 4 quarters
                    nums = [v for v in window if v is not None]
                    ttm_vals.append(sum(nums) if len(nums) == 4 else None)
                else:
                    # Stock: take latest non-null
                    ttm_vals.append(values[i])
            ttm_section[metric_key] = ttm_vals
        result[section_key] = ttm_section

    # Copy non-section keys (shares_outstanding, market_cap)
    for key in ("shares_outstanding", "market_cap", "term", "fiscal_year", "fiscal_quarter", "fiscal_label"):
        if key in quarterly_block:
            result[key] = quarterly_block[key]

    return result


def _compute_annual_block(quarterly_block: dict) -> dict:
    """Compute annual aggregations from a quarterly block.

    Groups quarters into fiscal years and sums flow / takes last stock.
    """
    periods = quarterly_block.get("periods", [])
    if not periods:
        return {}

    # Group period indices by fiscal year when available; otherwise fallback to calendar year.
    year_groups: dict[str, list[int]] = {}
    fiscal_years = list(quarterly_block.get("fiscal_year") or [])
    fiscal_quarters = list(quarterly_block.get("fiscal_quarter") or [])
    use_fiscal_year = len(fiscal_years) == len(periods) and any(v is not None for v in fiscal_years)
    if use_fiscal_year:
        for i, fiscal_year in enumerate(fiscal_years):
            try:
                if fiscal_year is None or pd.isna(fiscal_year):
                    continue
                year_key = str(int(float(fiscal_year)))
                year_groups.setdefault(year_key, []).append(i)
            except Exception:
                continue
    else:
        for i, p in enumerate(periods):
            try:
                dt = pd.Timestamp(p)
                year_key = str(dt.year)
                year_groups.setdefault(year_key, []).append(i)
            except Exception:
                continue

    complete_years: dict[str, list[int]] = {}
    for year, idxs in sorted(year_groups.items()):
        if len(idxs) != 4:
            continue
        if len(fiscal_quarters) == len(periods):
            quarter_values = {
                int(float(fiscal_quarters[i]))
                for i in idxs
                if i < len(fiscal_quarters) and fiscal_quarters[i] is not None and not pd.isna(fiscal_quarters[i])
            }
            if quarter_values and quarter_values != {1, 2, 3, 4}:
                continue
        complete_years[year] = idxs
    if not complete_years:
        return {}

    annual_periods = list(complete_years.keys())
    result: dict[str, Any] = {"periods": annual_periods}

    for section_key in ("income", "balance", "cashflow"):
        section = quarterly_block.get(section_key)
        if not section:
            continue
        annual_section: dict[str, list] = {}
        for metric_key, values in section.items():
            is_flow = metric_key in _FLOW_KEYS
            annual_vals: list = []
            for year in annual_periods:
                idxs = complete_years[year]
                window = [values[i] if i < len(values) else None for i in idxs]
                nums = [v for v in window if v is not None]
                if is_flow:
                    annual_vals.append(sum(nums) if len(nums) == 4 else None)
                else:
                    # Stock: take last quarter of the year
                    annual_vals.append(values[idxs[-1]] if idxs[-1] < len(values) else None)
            annual_section[metric_key] = annual_vals
        result[section_key] = annual_section

    for key in ("shares_outstanding", "market_cap"):
        if key in quarterly_block:
            vals = quarterly_block[key]
            annual_vals = []
            for year in annual_periods:
                idxs = complete_years[year]
                annual_vals.append(vals[idxs[-1]] if idxs[-1] < len(vals) else None)
            result[key] = annual_vals

    result["fiscal_year"] = [int(year) for year in annual_periods]
    result["fiscal_quarter"] = [4 for _ in annual_periods]
    result["fiscal_label"] = annual_periods
    result["term"] = annual_periods

    return result


def _build_financials_block(ticker: str, market: str, price_df: pd.DataFrame | None = None) -> dict | None:
    """Build the financials block from the reader layer."""
    try:
        if market == "kr" and STORAGE_BACKEND == "parquet":
            from market_data import parquet_reader

            df = parquet_reader.load_financials(ticker=ticker, market=market, include_extra=False)
        else:
            from market_data import db_reader_kr

            df = db_reader_kr.load_financials_from_db(ticker, market=market)
    except Exception:
        return None

    if df is None or df.empty:
        return None
    df = _ensure_fiscal_metadata(df)

    # Sort by PeriodEnd and apply date filter
    if "PeriodEnd" in df.columns:
        df["PeriodEnd"] = pd.to_datetime(df["PeriodEnd"], errors="coerce")
        if _START_DATE:
            df = df[df["PeriodEnd"] >= pd.Timestamp(_START_DATE)]
        df = df.sort_values("PeriodEnd")
        if df.empty:
            return None
        periods = [_safe_val(d) for d in df["PeriodEnd"]]
    else:
        return None

    result: dict[str, Any] = {"periods": periods}
    _attach_fiscal_block_metadata(result, df)

    # Income
    income: dict[str, list] = {}
    for key, col in _INCOME_METRICS.items():
        if market == "kr" and key == "rd":
            continue
        if col in df.columns:
            income[key] = _series_to_list(df[col])

    # Fallback: fill net_income gaps from Net Income Common (지배기업 귀속)
    # DART XBRL sometimes omits ifrs-full_ProfitLoss while reporting
    # the parent-attributable figure separately.
    if "net_income" in income and market == "kr":
        fallback_col = None
        for candidate in ("net_income_common", "Net Income Common"):
            if candidate in df.columns:
                fallback_col = candidate
                break
        if fallback_col is not None:
            fb = _series_to_list(df[fallback_col])
            ni = income["net_income"]
            fq = list(df["fiscal_quarter"]) if "fiscal_quarter" in df.columns else []
            for i in range(len(ni)):
                if i >= len(fb) or fb[i] is None:
                    continue
                # Case 1: NI missing → use NIC
                if ni[i] is None:
                    ni[i] = fb[i]
                    continue
                # Case 2: Q4/FY row where NI is the FY annual total (not
                # deaccumulated) while NIC was correctly deaccumulated.
                # Detect by NI/NIC ratio > 3 at Q4 positions.
                if (i < len(fq) and fq[i] == 4
                        and fb[i] != 0 and abs(ni[i] / fb[i]) > 3):
                    ni[i] = fb[i]

            # Guard: null out NIC-sourced values that are DART unit spikes.
            # Some DART XBRL filings have unitRef mismatches (원 vs 천원)
            # producing values 1000x+ larger than neighbors.  Only touch
            # indices that were just filled from NIC (where original NI was
            # None), so existing NI data is never affected.
            import numpy as _np
            _ni_arr = _np.array([float(v) if v is not None else _np.nan for v in ni])
            for i in range(len(ni)):
                if ni[i] is None or _np.isnan(_ni_arr[i]):
                    continue
                # Gather up to 4 nearest non-None neighbors
                neighbors = []
                for j in (i - 1, i - 2, i + 1, i + 2):
                    if 0 <= j < len(ni) and not _np.isnan(_ni_arr[j]):
                        neighbors.append(abs(_ni_arr[j]))
                if len(neighbors) < 2:
                    continue
                median_n = _np.median(neighbors)
                if median_n <= 0:
                    continue
                if abs(_ni_arr[i]) / median_n > 100:
                    ni[i] = None

    # Derive revenue from gross_profit + cogs where revenue is missing
    if "revenue" in income and "gross_profit" in income and "cogs" in income:
        rev, gp, cogs = income["revenue"], income["gross_profit"], income["cogs"]
        for i in range(len(rev)):
            if rev[i] is None and gp[i] is not None and cogs[i] is not None:
                rev[i] = gp[i] + cogs[i]

    # Null out gross_profit when it equals revenue and cogs is missing
    # (DART non-COGS reporters have GP = Revenue which is misleading)
    if "revenue" in income and "gross_profit" in income:
        rev, gp = income["revenue"], income["gross_profit"]
        cogs = income.get("cogs", [None] * len(rev))
        for i in range(len(rev)):
            if (rev[i] is not None and gp[i] is not None
                    and abs(rev[i] - gp[i]) < 1
                    and (i >= len(cogs) or cogs[i] is None)):
                gp[i] = None

    # Derive pretax_income from net_income + tax where pretax is missing
    if "net_income" in income and "tax" in income:
        ni, tx = income["net_income"], income["tax"]
        pt = income.get("pretax_income", [None] * len(ni))
        changed = False
        for i in range(len(ni)):
            if (i < len(pt) and pt[i] is None
                    and ni[i] is not None and tx[i] is not None):
                pt[i] = ni[i] + tx[i]
                changed = True
        if changed:
            income["pretax_income"] = pt

    # Sanitize income: Q4 annual cumulative, unit spikes, negative revenue
    _sanitize_income_series(income, df, market=market)

    if income:
        result["income"] = income

    # Balance
    balance: dict[str, list] = {}
    for key, col in _BALANCE_METRICS.items():
        if col in df.columns:
            balance[key] = _series_to_list(df[col])
    if balance:
        result["balance"] = balance

    # Cashflow
    cashflow: dict[str, list] = {}
    for key, col in _CASHFLOW_METRICS.items():
        if col in df.columns:
            cashflow[key] = _series_to_list(df[col])
    if cashflow:
        result["cashflow"] = cashflow

    # Shares outstanding: prefer financials, fallback to price_df (pykrx)
    for shares_col in ("Shares", "Diluted Shares", "Basic Shares"):
        if shares_col in df.columns:
            vals = _series_to_list(df[shares_col])
            if any(v is not None for v in vals):
                result["shares_outstanding"] = vals
                break

    if "shares_outstanding" not in result and price_df is not None and "SharesOutstanding" in price_df.columns:
        shares_from_price: list[Any] = []
        for p in periods:
            try:
                target = pd.Timestamp(p)
                mask = price_df.index <= target
                if mask.any():
                    shares_from_price.append(_safe_val(price_df.loc[mask, "SharesOutstanding"].iloc[-1]))
                else:
                    shares_from_price.append(None)
            except Exception:
                shares_from_price.append(None)
        if any(v is not None for v in shares_from_price):
            result["shares_outstanding"] = shares_from_price

    # Market cap at each period_end: prefer MarketCap column, fallback shares × Close
    mcaps: list[Any] = []
    _close_col2 = None
    for _cc2 in ("Close", "Adj Close", "close"):
        if price_df is not None and _cc2 in price_df.columns:
            _close_col2 = _cc2
            break

    if price_df is not None and not price_df.empty and "MarketCap" in price_df.columns:
        for p in periods:
            try:
                target = pd.Timestamp(p)
                mask = price_df.index <= target
                if mask.any():
                    mcaps.append(_safe_val(price_df.loc[mask, "MarketCap"].iloc[-1]))
                else:
                    mcaps.append(None)
            except Exception:
                mcaps.append(None)
    elif "shares_outstanding" in result and _close_col2 is not None and price_df is not None:
        shares_list2 = result["shares_outstanding"]
        for i, p in enumerate(periods):
            try:
                target = pd.Timestamp(p)
                mask = price_df.index <= target
                s = shares_list2[i] if i < len(shares_list2) else None
                if mask.any() and s is not None:
                    close_val = float(price_df.loc[mask, _close_col2].iloc[-1])
                    mcaps.append(_safe_val(s * close_val))
                else:
                    mcaps.append(None)
            except Exception:
                mcaps.append(None)

    if mcaps and any(v is not None for v in mcaps):
        result["market_cap"] = mcaps

    return result


def _get_ticker_info(ticker: str) -> dict[str, str]:
    """Get ticker metadata (name, sector, industry).

    load_ticker_master_from_db(ticker) — no market param.
    """
    from market_data import db_reader_kr

    try:
        info = db_reader_kr.load_ticker_master_from_db(ticker)
        if info is not None and not info.empty:
            row = info.iloc[0]
            # Prefer detailed KSIC classification over generic sector_name
            sector = str(row.get("sector_name", "") or "")
            industry = str(row.get("industry_name", "") or "")
            ksic = str(row.get("ksic_name_ko", "") or "")
            krx_ind = str(row.get("krx_industry_name", "") or "")
            subsector = str(row.get("subsector_name", "") or "")
            if ksic:
                industry = ksic
            elif krx_ind:
                industry = krx_ind
            if subsector and not sector:
                sector = subsector
            return {
                "name": str(row.get("ticker_name", "") or ""),
                "sector": sector,
                "industry": industry,
                "subsector": subsector,
                "market_tier": str(row.get("market_tier", "") or ""),
            }
    except Exception:
        pass
    return {"name": "", "sector": "", "industry": "", "subsector": "", "market_tier": ""}


def export_ticker(ticker: str, market: str = "kr") -> dict | None:
    """Build and return the full JSON dict for a single ticker."""
    if market == "us":
        return _export_ticker_us(ticker)
    return _export_ticker_kr(ticker)


def _export_ticker_kr(ticker: str) -> dict | None:
    """Build JSON for a KR ticker."""
    info = _get_ticker_info(ticker)
    data: dict[str, Any] = {
        "ticker": ticker,
        "market": "kr",
        "name": info["name"],
        "sector": info["sector"],
        "industry": info["industry"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    # Build financials first (without price data), then fetch quarter-end prices
    # via pykrx on-the-fly — avoids loading full daily price parquet (851MB).
    financials = _build_financials_block(ticker, "kr", price_df=None)

    # Fetch quarter-end prices only for market_cap calc and valuation_ttm
    prices_block, price_df = None, None
    if financials and "periods" in financials and os.environ.get("MDL_SKIP_KR_PRICES") != "1":
        prices_block, price_df = _kr_prices_for_dates(ticker, financials["periods"])
        if price_df is not None:
            financials = _build_financials_block(ticker, "kr", price_df=price_df)
    if financials:
        data["financials"] = financials
        ttm = _compute_ttm_block(financials)
        if ttm and len(ttm) > 1:
            data["financials_ttm"] = ttm
        annual = _compute_annual_block(financials)
        if annual and len(annual) > 1:
            data["financials_annual"] = annual

    # Attach prices_block temporarily for valuation_ttm generation (removed before JSON write)
    if prices_block:
        data["_prices_for_valuation"] = prices_block
    return data


def _yf_prices_for_dates(ticker: str, period_dates: list[str]) -> tuple[dict | None, pd.DataFrame | None]:
    """Fetch Adj Close prices from yfinance for specific quarter-end dates.

    Uses Adj Close (split + dividend adjusted) which is consistent with
    yfinance's per-share financial metrics (EPS, BPS) that are also
    retroactively adjusted for splits.

    Returns (prices_block, raw_df) matching _build_prices_block_us signature.
    """
    if not period_dates:
        return None, None
    try:
        import yfinance as yf
        dates_dt = pd.to_datetime(period_dates).drop_duplicates().sort_values()
        start = (dates_dt.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        end = (dates_dt.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        hist = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
        if hist is None or hist.empty:
            return None, None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        # Use Adj Close for consistency with split-adjusted financial metrics
        adj_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
        close_series = pd.to_numeric(hist[adj_col], errors="coerce").sort_index()
        # Reindex to quarter-end dates using ffill
        aligned = close_series.reindex(dates_dt, method="ffill")
        valid = aligned.dropna()
        if valid.empty:
            return None, None
        dates_out = [d.strftime("%Y-%m-%d") for d in valid.index]
        block: dict[str, Any] = {
            "dates": dates_out,
            "close": [round(float(v), 2) for v in valid.values],
        }
        raw_df = pd.DataFrame({"Close": valid.values}, index=valid.index)
        raw_df.index.name = "date"
        return block, raw_df
    except Exception:
        return None, None


def _kr_prices_for_dates(ticker: str, period_dates: list[str]) -> tuple[dict | None, pd.DataFrame | None]:
    """Fetch KR prices from pykrx for specific quarter-end dates only.

    Mirrors _yf_prices_for_dates but uses pykrx (Close, MarketCap, SharesOutstanding).
    Avoids loading the full daily price parquet.
    """
    if not period_dates:
        return None, None
    try:
        from market_data.krx.prices import fetch_price_frame
        from market_data.db_router import normalize_kr_ticker

        dates_dt = pd.to_datetime(period_dates).drop_duplicates().sort_values()
        start = (dates_dt.min() - pd.Timedelta(days=10)).strftime("%Y%m%d")
        end = (dates_dt.max() + pd.Timedelta(days=10)).strftime("%Y%m%d")
        ticker_code = normalize_kr_ticker(ticker)

        frame = fetch_price_frame(ticker=ticker_code, start=start, end=end)
        if frame is None or frame.empty:
            return None, None

        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame = frame.sort_index()

        # Align to quarter-end dates via ffill
        close_series = pd.to_numeric(frame.get("Close"), errors="coerce")
        aligned_close = close_series.reindex(dates_dt, method="ffill")

        mcap_series = pd.to_numeric(frame.get("MarketCap"), errors="coerce") if "MarketCap" in frame.columns else None
        aligned_mcap = mcap_series.reindex(dates_dt, method="ffill") if mcap_series is not None else None

        shares_series = pd.to_numeric(frame.get("SharesOutstanding"), errors="coerce") if "SharesOutstanding" in frame.columns else None
        aligned_shares = shares_series.reindex(dates_dt, method="ffill") if shares_series is not None else None

        valid = aligned_close.dropna()
        if valid.empty:
            return None, None

        # Build prices_block
        dates_out = [d.strftime("%Y-%m-%d") for d in valid.index]
        block: dict[str, Any] = {
            "dates": dates_out,
            "close": [_safe_val(v) for v in valid.values],
        }
        # Build raw_df
        raw_data: dict[str, Any] = {"Close": valid.values}

        if aligned_mcap is not None:
            mcap_valid = aligned_mcap.reindex(valid.index)
            block["market_cap"] = [_safe_val(v) for v in mcap_valid.values]
            raw_data["MarketCap"] = mcap_valid.values

        if aligned_shares is not None:
            shares_valid = aligned_shares.reindex(valid.index)
            block["shares_outstanding"] = [_safe_val(v) for v in shares_valid.values]
            raw_data["SharesOutstanding"] = shares_valid.values

        raw_df = pd.DataFrame(raw_data, index=valid.index)
        raw_df.index.name = "Date"
        return block, raw_df
    except Exception:
        return None, None


def _export_ticker_us(ticker: str) -> dict | None:
    """Build JSON for a US ticker."""
    info = _get_ticker_info_us(ticker)
    data: dict[str, Any] = {
        "ticker": ticker,
        "market": "us",
        "name": info["name"],
        "sector": info["sector"],
        "industry": info["industry"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    # Load prices for market_cap calc and valuation_ttm (not stored in main JSON)
    prices_block, price_df = _build_prices_block_us(ticker)
    financials = _build_financials_block_us(ticker, price_df=price_df)

    # yfinance fallback: fetch quarter-end prices when parquet has no price data
    if price_df is None and financials and "periods" in financials:
        prices_block, price_df = _yf_prices_for_dates(ticker, financials["periods"])
        if price_df is not None:
            financials = _build_financials_block_us(ticker, price_df=price_df)

    if financials:
        data["financials"] = financials
        ttm = _compute_ttm_block(financials)
        if ttm and len(ttm) > 1:
            data["financials_ttm"] = ttm
        annual = _compute_annual_block(financials)
        if annual and len(annual) > 1:
            data["financials_annual"] = annual

    if prices_block:
        data["_prices_for_valuation"] = prices_block
    return data


def export_ticker_to_file(ticker: str, market: str = "kr") -> Path | None:
    """Export a single ticker to JSON file. Returns the output path."""
    data = export_ticker(ticker, market)
    if data is None:
        return None

    # Generate valuation_ttm sidecar BEFORE stripping prices
    export_valuation_ttm_to_file(ticker, market, ticker_data=data)

    # Overwrite financials.market_cap with split-adjusted values from valuation_ttm
    _patch_market_cap_from_valuation_ttm(data, ticker, market)

    # Strip temporary prices (not needed in main JSON — V2 uses Yahoo for charts)
    data.pop("_prices_for_valuation", None)

    out_dir = _TICKERS_DIR / market
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def _patch_market_cap_from_valuation_ttm(data: dict, ticker: str, market: str) -> None:
    """Overwrite financials.market_cap with split-adjusted values from valuation_ttm.

    The valuation_ttm sidecar computes market_cap using split-adjusted prices,
    which avoids discontinuities around stock splits (e.g. Samsung 2018 split).
    """
    fin = data.get("financials")
    if not fin or not isinstance(fin, dict):
        return

    val_path = _TICKERS_DIR / market / f"{ticker}.valuation_ttm.json"
    if not val_path.exists():
        return

    try:
        val_data = json.loads(val_path.read_bytes())
    except Exception:
        return

    windows = val_data.get("windows", {})
    # Use longest window (10y > 5y) for most coverage
    val_periods: list[str] = []
    val_mc: list[float | None] = []
    for wkey in ("10y", "5y"):
        w = windows.get(wkey, {})
        vp = w.get("periods", [])
        vm = w.get("series", {}).get("market_cap", [])
        if vp and vm and len(vp) == len(vm):
            val_periods = vp
            val_mc = vm
            break

    if not val_periods or not val_mc:
        return

    # Build a lookup: quarter label → market_cap
    # val_periods are like ["2016Q1", "2016Q2", ...]
    # fin["periods"] are like ["2016-03-31", "2016-06-30", ...]
    val_lookup: dict[str, float | None] = dict(zip(val_periods, val_mc))

    fin_periods = fin.get("periods", [])
    if not fin_periods:
        return

    # Convert fin period dates to quarter labels for matching
    patched_mc: list[float | None] = []
    for p in fin_periods:
        try:
            ts = pd.Timestamp(p)
            q_label = f"{ts.year}Q{(ts.month - 1) // 3 + 1}"
            mc_val = val_lookup.get(q_label)
            patched_mc.append(mc_val)
        except Exception:
            patched_mc.append(None)

    # Only overwrite if we got at least some values
    if any(v is not None for v in patched_mc):
        fin["market_cap"] = patched_mc


def _valuation_ttm_file_path(ticker: str, market: str) -> Path:
    return _TICKERS_DIR / market / f"{ticker}.valuation_ttm.json"


def export_valuation_ttm_to_file(
    ticker: str,
    market: str = "kr",
    *,
    ticker_data: dict[str, Any] | None = None,
) -> Path | None:
    data = ticker_data if ticker_data is not None else export_ticker(ticker, market)
    if data is None:
        return None

    payload = build_valuation_ttm_payload(
        ticker=ticker,
        market=market,
        prices_block=data.get("_prices_for_valuation") or data.get("prices"),
        financials_ttm_block=data.get("financials_ttm"),
        quarterly_block=data.get("financials"),
        updated_at=str(data.get("updated_at") or datetime.now().isoformat(timespec="seconds")),
    )

    out_path = _valuation_ttm_file_path(ticker, market)
    if payload is None:
        out_path.unlink(missing_ok=True)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Export: ticker master meta
# ---------------------------------------------------------------------------

def export_ticker_master(market: str = "kr") -> Path:
    """Export ticker_master to a JSON meta file."""
    items = []
    if market == "kr":
        from market_data import db_reader_kr
        df = db_reader_kr.load_ticker_master_all()
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                items.append({
                    "ticker": str(row.get("ticker", "")),
                    "name": str(row.get("ticker_name", "") or ""),
                    "market_tier": str(row.get("market_tier", "") or ""),
                    "sector": str(row.get("sector_name", "") or ""),
                    "industry": str(row.get("industry_name", "") or ""),
                })
    elif market == "us":
        try:
            if STORAGE_BACKEND == "parquet":
                from market_data import parquet_reader

                prices_df = parquet_reader.read_table("us", "prices", columns=["ticker"])
                issuer_df = parquet_reader.load_sec_issuer_registry_all(market="us")
                company_map: dict[str, str] = {}
                if issuer_df is not None and not issuer_df.empty:
                    issuer_df = issuer_df.dropna(subset=["ticker"]).copy()
                    if "collected_at" in issuer_df.columns:
                        issuer_df = issuer_df.sort_values(["ticker", "collected_at"])
                    company_map = (
                        issuer_df.drop_duplicates(subset=["ticker"], keep="last")
                        .set_index("ticker")["company_name"]
                        .fillna("")
                        .astype(str)
                        .to_dict()
                    )
                tickers = sorted(prices_df["ticker"].dropna().astype(str).unique().tolist()) if not prices_df.empty else []
                for ticker in tickers:
                    items.append({
                        "ticker": ticker,
                        "name": company_map.get(ticker, ""),
                        "market_tier": "NYSE/NASDAQ",
                        "sector": "",
                        "industry": "",
                    })
            else:
                con = _us_connect("prices")
                df = con.execute(
                    "SELECT DISTINCT p.ticker, COALESCE(sir.company_name, '') as company_name "
                    "FROM prices p LEFT JOIN sec_issuer_registry sir ON sir.ticker = p.ticker "
                    "WHERE p.market = 'us' ORDER BY p.ticker"
                ).df()
                con.close()
                for _, row in df.iterrows():
                    items.append({
                        "ticker": str(row.get("ticker", "")),
                        "name": str(row.get("company_name", "") or ""),
                        "market_tier": "NYSE/NASDAQ",
                        "sector": "",
                        "industry": "",
                    })
        except Exception:
            pass

    # Preserve name_kr from existing file (populated by external script)
    _META_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _META_DIR / f"ticker_master_{market}.json"
    if market == "us" and out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            kr_map = {it["ticker"]: it["name_kr"] for it in existing.get("items", []) if it.get("name_kr")}
            if kr_map:
                for item in items:
                    kr = kr_map.get(item["ticker"])
                    if kr:
                        item["name_kr"] = kr
        except Exception:
            pass

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"market": market, "count": len(items), "items": items}, f, ensure_ascii=False, indent=2)
    return out_path


def _update_last_updated(market: str, tickers_exported: int) -> None:
    """Update the last_updated.json meta file."""
    meta_path = _META_DIR / "last_updated.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    meta[market] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "tickers_exported": tickers_exported,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_export(
    market: str = "kr",
    tickers: list[str] | None = None,
    start_date: str = "2013-06-01",
    top_n: int = 0,
) -> None:
    """Run full export for a market.

    Args:
        start_date: Only include data from this date onwards (default: 2013-06-01)
        top_n: If >0, only export top N tickers by market cap (for US market)
    """
    global _START_DATE
    _START_DATE = start_date
    print(f"[export] Starting export for market={market}, start_date={start_date}, top_n={top_n or 'all'}")

    # 1. Get ticker list
    if tickers:
        ticker_list = tickers
    elif market == "kr":
        # Default: use universe file to limit to curated ~2000 tickers
        universe_path = _REPO_ROOT / ".codex_tmp" / "kr_2000_universe.csv"
        if universe_path.exists():
            with open(universe_path, encoding="utf-8") as uf:
                ticker_list = [line.strip() for line in uf if line.strip() and not line.strip().startswith("ticker")]
            print(f"[export] Using KR universe: {len(ticker_list)} tickers from {universe_path.name}")
        else:
            from market_data import db_reader_kr
            master_df = db_reader_kr.load_ticker_master_all()
            if master_df is None or master_df.empty:
                print("[export] No tickers found in KR ticker_master")
                return
            ticker_list = master_df["ticker"].tolist()
            print(f"[export] WARNING: No universe file found, using all {len(ticker_list)} tickers from ticker_master")
    elif market == "us":
        try:
            if STORAGE_BACKEND == "parquet":
                from market_data import parquet_reader

                price_cols = parquet_reader.read_table("us", "prices", columns=["ticker", "date", "market_cap"])
                if price_cols.empty:
                    ticker_list = []
                elif top_n > 0 and {"ticker", "date", "market_cap"}.issubset(price_cols.columns):
                    ranked = price_cols.dropna(subset=["ticker"]).copy()
                    ranked["date"] = pd.to_datetime(ranked["date"], errors="coerce")
                    ranked["market_cap"] = pd.to_numeric(ranked["market_cap"], errors="coerce")
                    ranked = ranked.dropna(subset=["date"])
                    ranked = ranked.sort_values(["ticker", "date"]).groupby("ticker", sort=False).tail(1)
                    ranked = ranked[(ranked["market_cap"].notna()) & (ranked["market_cap"] > 0)]
                    ranked = ranked.sort_values("market_cap", ascending=False)
                    ticker_list = ranked["ticker"].astype(str).head(top_n).tolist()
                else:
                    ticker_list = sorted(price_cols["ticker"].dropna().astype(str).unique().tolist())
            else:
                con = _us_connect("prices")
                if top_n > 0:
                    df = con.execute(f"""
                        WITH latest AS (
                            SELECT ticker, market_cap,
                                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
                            FROM prices WHERE market = 'us' AND market_cap IS NOT NULL AND market_cap > 0
                        )
                        SELECT ticker FROM latest WHERE rn = 1
                        ORDER BY market_cap DESC LIMIT {top_n}
                    """).df()
                else:
                    df = con.execute("SELECT DISTINCT ticker FROM prices WHERE market = 'us' ORDER BY ticker").df()
                con.close()
                ticker_list = df["ticker"].tolist()
        except Exception as e:
            print(f"[export] Failed to get US ticker list: {e}")
            return
    else:
        print(f"[export] Unknown market: {market}")
        return

    print(f"[export] {len(ticker_list)} tickers to export")

    # 2. Export each ticker
    exported = 0
    errors = 0
    for i, ticker in enumerate(ticker_list):
        try:
            path = export_ticker_to_file(ticker, market)
            if path:
                exported += 1
            if (i + 1) % 50 == 0:
                print(f"[export] {i + 1}/{len(ticker_list)} done ({exported} exported, {errors} errors)")
        except Exception as e:
            errors += 1
            print(f"[export] ERROR {ticker}: {e}")

    print(f"[export] Exported {exported}/{len(ticker_list)} tickers ({errors} errors)")

    # 3. Export ticker master (only when exporting all tickers, not a subset)
    if tickers is None or len(ticker_list) > 100:
        master_path = export_ticker_master(market)
        print(f"[export] Ticker master written to {master_path}")
    else:
        print(f"[export] Skipping ticker_master regeneration (partial export of {len(ticker_list)} tickers)")

    # 4. Update meta
    _update_last_updated(market, exported)
    print("[export] Done!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DuckDB to JSON ticker files")
    parser.add_argument("--market", default="kr", choices=["kr", "us"])
    parser.add_argument("--tickers", help="Comma-separated ticker list", default="")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    run_export(market=args.market, tickers=tickers)


if __name__ == "__main__":
    main()
