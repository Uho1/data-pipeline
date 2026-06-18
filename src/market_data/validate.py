from __future__ import annotations

import pandas as pd

from market_data.config import LOGS_DIR, PRICE_REQUIRED_COLUMNS
from market_data.utils import ensure_dir


def _validate_prices() -> tuple[pd.DataFrame, dict[str, int]]:
    from market_data.db import db_available
    from market_data.db_prices import get_connection as get_prices_connection
    if not db_available():
        return pd.DataFrame(), {"price_tickers": 0, "error": "DuckDB not available"}

    con = get_prices_connection()
    # Per-ticker stats
    stats = con.execute("""
        SELECT
            market,
            ticker,
            COUNT(*) AS rows,
            MIN(date) AS date_min,
            MAX(date) AS date_max,
            AVG(CASE WHEN close IS NULL THEN 1.0 ELSE 0.0 END) AS missing_close,
            AVG(CASE WHEN adj_close IS NULL THEN 1.0 ELSE 0.0 END) AS missing_adj_close,
            AVG(CASE WHEN volume IS NULL THEN 1.0 ELSE 0.0 END) AS missing_volume
        FROM prices
        GROUP BY market, ticker
        ORDER BY market, ticker
    """).df()

    total = len(stats)
    summary = {
        "price_tickers": total,
        "price_rows_total": int(stats["rows"].sum()) if not stats.empty else 0,
        "price_tickers_with_missing_close": int((stats.get("missing_close", pd.Series(dtype=float)) > 0).sum()) if not stats.empty else 0,
    }
    return stats, summary


def _validate_financials() -> tuple[pd.DataFrame, dict[str, int]]:
    from market_data.db import db_available, get_connection
    if not db_available():
        return pd.DataFrame(), {"financial_tickers": 0, "error": "DuckDB not available"}

    con = get_connection()
    stats = con.execute("""
        SELECT
            market,
            ticker,
            COUNT(*) AS rows,
            MIN("PeriodEnd") AS period_min,
            MAX("PeriodEnd") AS period_max,
            COUNT("Revenue") AS revenue_non_null,
            COUNT("Net Income") AS net_income_non_null,
            COUNT("Total Assets") AS assets_non_null
        FROM financials_quarterly
        GROUP BY market, ticker
        ORDER BY market, ticker
    """).df()

    summary = {
        "financial_tickers": len(stats),
        "financial_rows_total": int(stats["rows"].sum()) if not stats.empty else 0,
        "financial_tickers_no_revenue": int((stats.get("revenue_non_null", pd.Series(dtype=int)) == 0).sum()) if not stats.empty else 0,
    }
    return stats, summary


def run_validate() -> int:
    ensure_dir(LOGS_DIR)
    price_rep, price_summary = _validate_prices()
    fin_rep, fin_summary = _validate_financials()

    price_out = LOGS_DIR / "validate_prices_report.csv"
    fin_out = LOGS_DIR / "validate_financials_report.csv"
    price_rep.to_csv(price_out, index=False)
    fin_rep.to_csv(fin_out, index=False)

    print("[VALIDATE] prices")
    print(price_summary)
    print(f"[REPORT] {price_out}")
    print("[VALIDATE] financials")
    print(fin_summary)
    print(f"[REPORT] {fin_out}")

    return 0
