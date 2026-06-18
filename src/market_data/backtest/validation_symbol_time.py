from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _coerce_ts(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _read_override_rows(overrides_path: str | Path | None, market: str | None = None) -> pd.DataFrame:
    if overrides_path is None:
        return pd.DataFrame()
    path = Path(overrides_path).expanduser()
    if not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    elif path.suffix.lower() in {".json", ".jsn"}:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return pd.DataFrame()
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            if "rows" in raw and isinstance(raw["rows"], list):
                df = pd.DataFrame(raw["rows"])
            else:
                df = pd.DataFrame(list(raw.values()))
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    if market and "market" in out.columns:
        out_market = out["market"].astype(str).str.lower().str.strip()
        out = out.loc[(out_market == str(market).lower()) | (out_market == "")]
    return out


def _extract_date_range_from_price(price_df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if price_df is None or price_df.empty:
        return None, None

    dates: pd.Series
    if "Date" in price_df.columns:
        dates = pd.to_datetime(price_df["Date"], errors="coerce")
    else:
        idx = pd.to_datetime(price_df.index, errors="coerce")
        dates = pd.Series(idx, index=price_df.index)

    if dates.isna().all():
        return None, None

    candidates = ["Adj Close", "Close", "adj_close", "close", "open", "price"]
    valid_mask = pd.Series(False, index=price_df.index)
    for col in candidates:
        if col in price_df.columns:
            valid_mask = valid_mask | pd.to_numeric(price_df[col], errors="coerce").notna()
    if not valid_mask.any():
        valid_mask = dates.notna()

    valid_dates = pd.to_datetime(dates.loc[valid_mask], errors="coerce").dropna()
    if valid_dates.empty:
        return None, None
    return pd.Timestamp(valid_dates.min()).normalize(), pd.Timestamp(valid_dates.max()).normalize()


def _resolve_market_dir(price_root: str | Path | None, market: str) -> Path:
    if price_root is None:
        base = Path("data") / "prices"
    else:
        base = Path(price_root).expanduser()
    market_dir = base / market
    if market_dir.exists():
        return market_dir
    return base


def load_ticker_validity_ranges(
    tickers: list[str] | set[str] | tuple[str, ...],
    *,
    market: str = "us",
    price_root: str | Path | None = None,
    overrides_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    ticker_list = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
    if not ticker_list:
        return {}

    out: dict[str, dict[str, Any]] = {}

    # Bulk-load date ranges from DuckDB (primary)
    db_ranges: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] = {}
    try:
        from market_data.db_router import db_available_for_market, get_prices_connection_for_market
        if db_available_for_market(market):
            con = get_prices_connection_for_market(market)
            ticker_list_sql = ", ".join(f"'{t}'" for t in ticker_list)
            rows = con.execute(f"""
                SELECT ticker, MIN(date) AS first_date, MAX(date) AS last_date
                FROM prices
                WHERE ticker IN ({ticker_list_sql}) AND market = '{str(market).lower()}'
                GROUP BY ticker
            """).fetchall()
            for row in rows:
                t, d_first, d_last = row
                db_ranges[str(t).upper()] = (
                    pd.Timestamp(d_first).normalize() if d_first else None,
                    pd.Timestamp(d_last).normalize() if d_last else None,
                )
    except Exception:
        pass

    overrides = _read_override_rows(overrides_path=overrides_path, market=market)
    override_map = {
        str(row.get("ticker", "")).upper(): row
        for _, row in overrides.iterrows()
        if str(row.get("ticker", "")).strip()
    }

    for ticker in ticker_list:
        first_valid: pd.Timestamp | None = None
        last_valid: pd.Timestamp | None = None
        source_used = "missing"
        canonical_id = None
        note = ""

        if ticker in db_ranges:
            first_valid, last_valid = db_ranges[ticker]
            if first_valid is not None or last_valid is not None:
                source_used = "db_price"

        ovr = override_map.get(ticker)
        if ovr is not None:
            o_first = _coerce_ts(ovr.get("first_valid_date"))
            o_last = _coerce_ts(ovr.get("last_valid_date"))
            if o_first is not None:
                first_valid = o_first
            if o_last is not None:
                last_valid = o_last
            canonical_id = ovr.get("canonical_id")
            note = str(ovr.get("note", "") or "")
            source_used = "override"

        out[ticker] = {
            "first_valid_date": first_valid,
            "last_valid_date": last_valid,
            "source_used": source_used,
            "canonical_id": canonical_id,
            "note": note,
        }
    return out


def detect_ticker_time_inconsistency(
    trades_df: pd.DataFrame,
    *,
    market: str = "us",
    price_root: str | Path | None = None,
    overrides_path: str | Path | None = None,
    tolerance_days: int = 7,
    warn_days: int = 30,
    fail_days: int = 180,
    check_last_valid: bool = False,
    mode_label: str | None = None,
) -> dict[str, Any]:
    cols = [
        "mode_label",
        "ticker",
        "trade_date",
        "first_valid_date",
        "last_valid_date",
        "delta_days_from_first",
        "delta_days_from_last",
        "check_result",
        "check_type",
        "source_used",
        "note",
    ]
    if trades_df is None or trades_df.empty:
        return {
            "status": "pass",
            "issues": pd.DataFrame(columns=cols),
            "summary": {"total_issues": 0, "warn_count": 0, "fail_count": 0},
        }

    out = trades_df.copy()
    trade_date_col = "exec_date" if "exec_date" in out.columns else ("trade_date" if "trade_date" in out.columns else ("date" if "date" in out.columns else ""))
    if not trade_date_col or "ticker" not in out.columns:
        return {
            "status": "warn",
            "issues": pd.DataFrame(columns=cols),
            "summary": {"total_issues": 0, "warn_count": 1, "fail_count": 0, "note": "trades schema missing ticker/date"},
        }

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["trade_date"] = pd.to_datetime(out[trade_date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=["ticker", "trade_date"]).copy()
    if out.empty:
        return {
            "status": "pass",
            "issues": pd.DataFrame(columns=cols),
            "summary": {"total_issues": 0, "warn_count": 0, "fail_count": 0},
        }

    ranges = load_ticker_validity_ranges(
        tickers=out["ticker"].dropna().astype(str).unique().tolist(),
        market=market,
        price_root=price_root,
        overrides_path=overrides_path,
    )

    issues: list[dict[str, Any]] = []
    tol = int(max(0, tolerance_days))
    warn_d = int(max(0, warn_days))
    fail_d = int(max(warn_d, fail_days))

    for _, row in out.iterrows():
        ticker = str(row["ticker"])
        trade_date = pd.Timestamp(row["trade_date"]).normalize()
        info = ranges.get(ticker, {})
        first_valid = info.get("first_valid_date")
        last_valid = info.get("last_valid_date")
        source_used = str(info.get("source_used", "missing"))
        note = str(info.get("note", ""))

        if first_valid is None and last_valid is None:
            issues.append(
                {
                    "mode_label": mode_label or "",
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "first_valid_date": pd.NaT,
                    "last_valid_date": pd.NaT,
                    "delta_days_from_first": np.nan,
                    "delta_days_from_last": np.nan,
                    "check_result": "warn",
                    "check_type": "missing_validity_range",
                    "source_used": source_used,
                    "note": note or "ticker validity range unavailable",
                }
            )
            continue

        if first_valid is not None:
            delta_first = int((pd.Timestamp(first_valid) - trade_date).days)
            if delta_first > tol:
                result = "fail" if delta_first > fail_d else "warn"
                issues.append(
                    {
                        "mode_label": mode_label or "",
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "first_valid_date": pd.Timestamp(first_valid),
                        "last_valid_date": pd.Timestamp(last_valid) if last_valid is not None else pd.NaT,
                        "delta_days_from_first": delta_first,
                        "delta_days_from_last": np.nan,
                        "check_result": result,
                        "check_type": "before_first_valid",
                        "source_used": source_used,
                        "note": note,
                    }
                )

        if check_last_valid and last_valid is not None:
            delta_last = int((trade_date - pd.Timestamp(last_valid)).days)
            if delta_last > tol:
                result = "fail" if delta_last > fail_d else ("warn" if delta_last > warn_d else "warn")
                issues.append(
                    {
                        "mode_label": mode_label or "",
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "first_valid_date": pd.Timestamp(first_valid) if first_valid is not None else pd.NaT,
                        "last_valid_date": pd.Timestamp(last_valid),
                        "delta_days_from_first": np.nan,
                        "delta_days_from_last": delta_last,
                        "check_result": result,
                        "check_type": "after_last_valid",
                        "source_used": source_used,
                        "note": note,
                    }
                )

    issues_df = pd.DataFrame(issues, columns=cols)
    warn_count = int((issues_df.get("check_result", pd.Series(dtype=str)) == "warn").sum()) if not issues_df.empty else 0
    fail_count = int((issues_df.get("check_result", pd.Series(dtype=str)) == "fail").sum()) if not issues_df.empty else 0
    status = "fail" if fail_count > 0 else ("warn" if warn_count > 0 else "pass")
    return {
        "status": status,
        "issues": issues_df,
        "summary": {
            "total_issues": int(len(issues_df)),
            "warn_count": warn_count,
            "fail_count": fail_count,
            "tolerance_days": tol,
            "warn_days": warn_d,
            "fail_days": fail_d,
        },
    }


def summarize_ticker_time_issues(
    detected: dict[str, Any],
    *,
    check_name: str = "ticker_time_consistency",
    max_examples: int = 5,
) -> dict[str, Any]:
    issues_df = detected.get("issues")
    if issues_df is None or not isinstance(issues_df, pd.DataFrame):
        issues_df = pd.DataFrame()
    summary = detected.get("summary", {}) if isinstance(detected.get("summary"), dict) else {}
    status = str(detected.get("status", "pass")).lower()
    if status not in {"pass", "warn", "fail"}:
        status = "warn"

    examples: list[dict[str, Any]] = []
    if not issues_df.empty:
        head = issues_df.head(max(0, int(max_examples)))
        for _, row in head.iterrows():
            examples.append(
                {
                    "mode_label": row.get("mode_label", ""),
                    "ticker": row.get("ticker", ""),
                    "trade_date": pd.Timestamp(row.get("trade_date")).date().isoformat() if pd.notna(row.get("trade_date")) else "",
                    "first_valid_date": pd.Timestamp(row.get("first_valid_date")).date().isoformat() if pd.notna(row.get("first_valid_date")) else "",
                    "last_valid_date": pd.Timestamp(row.get("last_valid_date")).date().isoformat() if pd.notna(row.get("last_valid_date")) else "",
                    "delta_days_from_first": None if pd.isna(row.get("delta_days_from_first")) else int(row.get("delta_days_from_first")),
                    "delta_days_from_last": None if pd.isna(row.get("delta_days_from_last")) else int(row.get("delta_days_from_last")),
                    "check_result": row.get("check_result", ""),
                    "source_used": row.get("source_used", ""),
                    "note": row.get("note", ""),
                }
            )

    total_issues = int(summary.get("total_issues", len(issues_df)))
    warn_count = int(summary.get("warn_count", 0))
    fail_count = int(summary.get("fail_count", 0))
    details = f"issues={total_issues}, warn={warn_count}, fail={fail_count}"
    return {
        "name": check_name,
        "status": status,
        "details": details,
        "summary": summary,
        "examples": examples,
    }
