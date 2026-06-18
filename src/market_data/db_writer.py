"""Thread-safe DuckDB write operations for market_data_lake.

Data is written directly to DuckDB during ingest — no intermediate parquet files.

Thread safety:
    A single global write connection is protected by a threading.Lock.
    Worker threads download data concurrently (IO-bound); DB writes are
    serialized by the lock (CPU-bound but fast).

Tables managed here:
    prices                — OHLCV price data, PRIMARY KEY (ticker, market, date)
    financials_quarterly  — Quarterly financials (SEC + yfinance merged)
    ingest_checkpoints    — Replaces per-ticker JSON checkpoint files
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from market_data.config import STORAGE_BACKEND
from market_data.db import DB_PATH
from market_data.db_prices import DB_PATH as PRICES_DB_PATH

def _use_parquet() -> bool:
    return STORAGE_BACKEND == "parquet"

_lock = threading.Lock()
_con = None  # Single global write connection
_prices_con = None  # Prices DB write connection


def _get_con():
    global _con  # noqa: PLW0603
    if _con is None:
        import duckdb

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _con = duckdb.connect(str(DB_PATH))
    return _con


def _get_prices_con():
    global _prices_con  # noqa: PLW0603
    if _prices_con is None:
        import duckdb

        PRICES_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _prices_con = duckdb.connect(str(PRICES_DB_PATH))
    return _prices_con


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_schema() -> None:
    """Create all tables if they don't exist. Call once at ingest start."""
    if _use_parquet():
        from market_data import parquet_store
        parquet_store.init_dirs("us")
        parquet_store.init_dirs("kr")
        return
    with _lock:
        con = _get_con()
        prices_con = _get_prices_con()
        prices_con.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                date         DATE    NOT NULL,
                ticker       VARCHAR NOT NULL,
                market       VARCHAR NOT NULL,
                open         DOUBLE,
                high         DOUBLE,
                low          DOUBLE,
                close        DOUBLE,
                adj_close    DOUBLE,
                volume       BIGINT,
                dividends    DOUBLE,
                stock_splits DOUBLE,
                collected_at VARCHAR,
                market_cap   DOUBLE,
                PRIMARY KEY (ticker, market, date)
            )
        """)
        # 기존 테이블에 market_cap 컬럼이 없으면 추가
        try:
            existing_cols = [r[0] for r in prices_con.execute("PRAGMA table_info('prices')").fetchall()]
            if "market_cap" not in existing_cols:
                prices_con.execute("ALTER TABLE prices ADD COLUMN market_cap DOUBLE")
        except Exception:
            pass
        con.execute("""
            CREATE TABLE IF NOT EXISTS financials_quarterly (
                ticker                VARCHAR  NOT NULL,
                market                VARCHAR  NOT NULL,
                term                  VARCHAR,
                fiscal_year           INTEGER,
                fiscal_quarter        INTEGER,
                fiscal_label          VARCHAR,
                "StatementDate"       DATE,
                "PeriodEnd"           DATE     NOT NULL,
                "PeriodStart"         DATE,
                "FormType"            VARCHAR,
                "FilingDate"          DATE,
                "AcceptedAt"          TIMESTAMPTZ,
                "AvailableDate"       DATE,
                "AvailabilityMethod"  VARCHAR,
                "Revenue"             DOUBLE,
                "COGS"                DOUBLE,
                "Gross Profit"        DOUBLE,
                "SG&A"                DOUBLE,
                "R&D"                 DOUBLE,
                "Operating Income"    DOUBLE,
                "Net Income"          DOUBLE,
                "Net Income Common"   DOUBLE,
                "EPS"                 DOUBLE,
                "Diluted EPS"         DOUBLE,
                "D&A"                 DOUBLE,
                "Amortization"        DOUBLE,
                "SBC"                 DOUBLE,
                "Interest"            DOUBLE,
                "Pretax Income"       DOUBLE,
                "Tax"                 DOUBLE,
                diluted_eps           DOUBLE,
                diluted_shares        DOUBLE,
                basic_shares          DOUBLE,
                net_income_common     DOUBLE,
                eps_source            VARCHAR,
                "Operating Cash Flow"  DOUBLE,
                "Investing Cash Flow"  DOUBLE,
                "Financing Cash Flow"  DOUBLE,
                "Capital Expenditure"  DOUBLE,
                "Dividends Paid"       DOUBLE,
                "Repurchases"          DOUBLE,
                "Total Assets"         DOUBLE,
                "Total Liabilities"    DOUBLE,
                "Shareholders Equity"  DOUBLE,
                "Current Assets"       DOUBLE,
                "Current Liabilities"  DOUBLE,
                "AR"                   DOUBLE,
                "AP"                   DOUBLE,
                "Inventory"            DOUBLE,
                "Cash"                 DOUBLE,
                "Debt Short"           DOUBLE,
                "Debt Long"            DOUBLE,
                "Deferred Revenue"     DOUBLE,
                "Goodwill"             DOUBLE,
                "Intangibles"          DOUBLE,
                "Common Stock"         DOUBLE,
                "APIC"                 DOUBLE,
                "Retained Earnings"    DOUBLE,
                "AOCI"                 DOUBLE,
                "Shares"               DOUBLE,
                "Diluted Shares"       DOUBLE,
                "Basic Shares"         DOUBLE,
                "Price"                DOUBLE,
                name                  VARCHAR,
                sector                VARCHAR,
                industry              VARCHAR,
                "Source"              VARCHAR,
                collected_at          VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS segment_revenue_quarterly (
                ticker              VARCHAR NOT NULL,
                market              VARCHAR NOT NULL,
                period_end          DATE    NOT NULL,
                available_date      DATE,
                period_start        DATE,
                form_type           VARCHAR,
                filing_date         DATE,
                accepted_at         TIMESTAMPTZ,
                availability_method VARCHAR,
                segment_type        VARCHAR NOT NULL,
                segment_name        VARCHAR NOT NULL,
                revenue             DOUBLE,
                op_income           DOUBLE,
                source              VARCHAR,
                collected_at        VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                ticker          VARCHAR NOT NULL,
                market          VARCHAR NOT NULL,
                accession       VARCHAR NOT NULL,
                form_type       VARCHAR,
                period_end      DATE,
                report_date     DATE,
                available_date  DATE,
                filing_date     DATE,
                accepted_at     TIMESTAMPTZ,
                primary_doc_url VARCHAR,
                index_url       VARCHAR,
                is_amendment    BOOLEAN,
                is_nt           BOOLEAN,
                collected_at    VARCHAR,
                PRIMARY KEY (ticker, market, accession)
            )
        """)
        prices_con.execute("""
            CREATE TABLE IF NOT EXISTS sec_issuer_registry (
                ticker        VARCHAR NOT NULL,
                market        VARCHAR NOT NULL,
                cik           VARCHAR,
                company_name  VARCHAR,
                exchange      VARCHAR,
                security_category VARCHAR,
                is_common_stock BOOLEAN,
                source        VARCHAR,
                collected_at  TIMESTAMPTZ,
                PRIMARY KEY (ticker, market)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS segment_facts_quarterly (
                ticker              VARCHAR NOT NULL,
                market              VARCHAR NOT NULL,
                period_end          DATE    NOT NULL,
                available_date      DATE,
                segment_type        VARCHAR NOT NULL,
                segment_name        VARCHAR NOT NULL,
                metric              VARCHAR NOT NULL,
                value               DOUBLE,
                currency            VARCHAR,
                accession           VARCHAR,
                period_start        DATE,
                form_type           VARCHAR,
                filing_date         DATE,
                accepted_at         TIMESTAMPTZ,
                availability_method VARCHAR,
                source              VARCHAR,
                collected_at        VARCHAR,
                PRIMARY KEY (ticker, market, period_end, segment_type, segment_name, metric, accession)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS segment_extract_log (
                ticker      VARCHAR NOT NULL,
                market      VARCHAR NOT NULL,
                accession   VARCHAR,
                method      VARCHAR,
                status      VARCHAR,
                reason      VARCHAR,
                created_at  TIMESTAMPTZ
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS sec_facts_raw_normalized (
                ticker               VARCHAR,
                market               VARCHAR,
                accession            VARCHAR,
                form_type            VARCHAR,
                fact_name            VARCHAR,
                taxonomy             VARCHAR,
                unit                 VARCHAR,
                scale                VARCHAR,
                period_start         DATE,
                period_end           DATE,
                instant_date         DATE,
                value                DOUBLE,
                context_id           VARCHAR,
                dimension_json       VARCHAR,
                filing_date          DATE,
                accepted_at          TIMESTAMPTZ,
                available_date       DATE,
                availability_method  VARCHAR,
                source               VARCHAR,
                source_url           VARCHAR,
                collected_at         TIMESTAMPTZ,
                PRIMARY KEY (ticker, accession, fact_name, unit, period_start, period_end, instant_date, context_id)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS financials_quarterly_extra (
                ticker                          VARCHAR,
                market                          VARCHAR,
                period_end                      DATE,
                available_date                  DATE,
                filing_date                     DATE,
                accepted_at                     TIMESTAMPTZ,
                form_type                       VARCHAR,
                dividends_paid                  DOUBLE,
                share_repurchases               DOUBLE,
                sbc                             DOUBLE,
                r_and_d                         DOUBLE,
                shares_outstanding              DOUBLE,
                shares_eop                      DOUBLE,
                ar                              DOUBLE,
                inventory                       DOUBLE,
                ap                              DOUBLE,
                cash                            DOUBLE,
                debt_total                      DOUBLE,
                net_income                      DOUBLE,
                cfo                             DOUBLE,
                total_assets                    DOUBLE,
                owner_equity                    DOUBLE,
                owner_net_income                DOUBLE,
                common_stock                    DOUBLE,
                additional_paid_in_capital      DOUBLE,
                retained_earnings               DOUBLE,
                aoci                            DOUBLE,
                ppe                             DOUBLE,
                ppe_capex                       DOUBLE,
                intangibles                     DOUBLE,
                intangible_capex                DOUBLE,
                amortization                    DOUBLE,
                other_gain                      DOUBLE,
                financial_gain                  DOUBLE,
                equity_method_gain              DOUBLE,
                other_income                    DOUBLE,
                other_expense                   DOUBLE,
                financial_income                DOUBLE,
                financial_expense               DOUBLE,
                current_fin_assets              DOUBLE,
                non_current_fin_assets          DOUBLE,
                current_fin_liabilities         DOUBLE,
                non_current_fin_liabilities     DOUBLE,
                source                          VARCHAR,
                confidence                      DOUBLE,
                collected_at                    TIMESTAMPTZ,
                PRIMARY KEY (ticker, market, period_end, available_date, form_type)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS filing_events_8k (
                ticker          VARCHAR,
                market          VARCHAR,
                accession       VARCHAR,
                item_code       VARCHAR,
                filing_date     DATE,
                accepted_at     TIMESTAMPTZ,
                available_date  DATE,
                title           VARCHAR,
                source_url      VARCHAR,
                collected_at    TIMESTAMPTZ,
                PRIMARY KEY (ticker, accession, item_code)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS nt_filings (
                ticker             VARCHAR,
                market             VARCHAR,
                accession          VARCHAR,
                nt_form_type       VARCHAR,
                related_form_type  VARCHAR,
                filing_date        DATE,
                accepted_at        TIMESTAMPTZ,
                reason             VARCHAR,
                source_url         VARCHAR,
                collected_at       TIMESTAMPTZ,
                PRIMARY KEY (ticker, accession)
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS derived_factors_quarterly (
                ticker                   VARCHAR,
                market                   VARCHAR,
                period_end               DATE,
                available_date           DATE,
                basis                    VARCHAR,
                roe                      DOUBLE,
                roa                      DOUBLE,
                roic                     DOUBLE,
                gpa                      DOUBLE,
                asset_turnover           DOUBLE,
                leverage                 DOUBLE,
                debt_ratio               DOUBLE,
                current_ratio            DOUBLE,
                gross_margin             DOUBLE,
                op_margin                DOUBLE,
                net_margin               DOUBLE,
                cogs_ratio               DOUBLE,
                sga_ratio                DOUBLE,
                total_cost_ratio         DOUBLE,
                revenue_growth           DOUBLE,
                gross_profit_growth      DOUBLE,
                operating_income_growth  DOUBLE,
                net_income_growth        DOUBLE,
                cogs_growth              DOUBLE,
                sga_growth               DOUBLE,
                price_return             DOUBLE,
                ar_turnover              DOUBLE,
                inventory_turnover       DOUBLE,
                ap_turnover              DOUBLE,
                dso                      DOUBLE,
                dio                      DOUBLE,
                dpo                      DOUBLE,
                operating_cycle          DOUBLE,
                cash_cycle               DOUBLE,
                ccr                      DOUBLE,
                fcf                      DOUBLE,
                eps                      DOUBLE,
                bps                      DOUBLE,
                sps                      DOUBLE,
                ops                      DOUBLE,
                oofps                    DOUBLE,
                fcfps                    DOUBLE,
                per                      DOUBLE,
                pbr                      DOUBLE,
                psr                      DOUBLE,
                por                      DOUBLE,
                pfcfr                    DOUBLE,
                peg                      DOUBLE,
                accruals_ratio           DOUBLE,
                cfo_to_ni                DOUBLE,
                ar_delta                 DOUBLE,
                inv_delta                DOUBLE,
                ap_delta                 DOUBLE,
                net_wc                   DOUBLE,
                net_wc_delta             DOUBLE,
                filing_lag_days          DOUBLE,
                is_amendment             BOOLEAN,
                is_nt                    BOOLEAN,
                punctuality_score        DOUBLE,
                source                   VARCHAR,
                collected_at             TIMESTAMPTZ,
                PRIMARY KEY (ticker, market, period_end, available_date, basis)
            )
        """)
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS asset_turnover DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS leverage DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS debt_ratio DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS current_ratio DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS price_return DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS accruals_ratio DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS cfo_to_ni DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS ar_delta DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS inv_delta DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS ap_delta DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS net_wc DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS net_wc_delta DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS filing_lag_days DOUBLE")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS is_amendment BOOLEAN")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS is_nt BOOLEAN")
        con.execute("ALTER TABLE derived_factors_quarterly ADD COLUMN IF NOT EXISTS punctuality_score DOUBLE")
        con.execute("ALTER TABLE filings ADD COLUMN IF NOT EXISTS period_end DATE")
        con.execute("ALTER TABLE filings ADD COLUMN IF NOT EXISTS available_date DATE")
        con.execute("ALTER TABLE filings ADD COLUMN IF NOT EXISTS is_nt BOOLEAN")
        prices_con.execute("ALTER TABLE sec_issuer_registry ADD COLUMN IF NOT EXISTS security_category VARCHAR")
        prices_con.execute("ALTER TABLE sec_issuer_registry ADD COLUMN IF NOT EXISTS is_common_stock BOOLEAN")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS dividends_paid DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS share_repurchases DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS sbc DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS r_and_d DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS shares_outstanding DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS shares_eop DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS ar DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS inventory DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS ap DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS cash DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS debt_total DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS net_income DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS cfo DOUBLE")
        con.execute("ALTER TABLE financials_quarterly_extra ADD COLUMN IF NOT EXISTS total_assets DOUBLE")
        con.execute("""
            CREATE TABLE IF NOT EXISTS ingest_checkpoints (
                ticker       VARCHAR NOT NULL,
                market       VARCHAR NOT NULL,
                run_id       VARCHAR,
                completed_at VARCHAR,
                fresh_days   INTEGER,
                payload      VARCHAR,
                PRIMARY KEY (ticker, market)
            )
        """)
        # Migrate legacy tables built by build_duckdb.py (missing collected_at)
        prices_con.execute("ALTER TABLE prices ADD COLUMN IF NOT EXISTS collected_at VARCHAR")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS collected_at VARCHAR")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS fiscal_year INTEGER")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS fiscal_quarter INTEGER")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS fiscal_label VARCHAR")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"D&A\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Amortization\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"SBC\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"R&D\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Interest\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Pretax Income\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Tax\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Dividends Paid\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Repurchases\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Current Assets\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Current Liabilities\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"AR\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"AP\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Inventory\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Cash\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Debt Short\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Debt Long\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Deferred Revenue\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Goodwill\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Intangibles\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Common Stock\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"APIC\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"Retained Earnings\" DOUBLE")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS \"AOCI\" DOUBLE")
        con.execute("""
            CREATE OR REPLACE VIEW segment_revenue_quarterly_v AS
            SELECT
              ticker,
              market,
              period_end,
              available_date,
              period_start,
              form_type,
              filing_date,
              accepted_at,
              availability_method,
              segment_type,
              segment_name,
              SUM(CASE WHEN lower(metric)='revenue' THEN value END) AS revenue,
              SUM(CASE WHEN lower(metric) IN ('operating_income','op_income') THEN value END) AS op_income,
              any_value(source) AS source,
              MAX(collected_at) AS collected_at
            FROM segment_facts_quarterly
            GROUP BY
              ticker,
              market,
              period_end,
              available_date,
              period_start,
              form_type,
              filing_date,
              accepted_at,
              availability_method,
              segment_type,
              segment_name
        """)

        # Migrate legacy prices table that lacks PRIMARY KEY (created by build_duckdb.py)
        pk_count = prices_con.execute("""
            SELECT COUNT(*) FROM information_schema.table_constraints
            WHERE table_name = 'prices' AND constraint_type = 'PRIMARY KEY'
        """).fetchone()[0]
        if pk_count == 0:
            prices_con.execute("ALTER TABLE prices RENAME TO _prices_v0")
            prices_con.execute("""
                CREATE TABLE prices (
                    date         DATE    NOT NULL,
                    ticker       VARCHAR NOT NULL,
                    market       VARCHAR NOT NULL,
                    open         DOUBLE,
                    high         DOUBLE,
                    low          DOUBLE,
                    close        DOUBLE,
                    adj_close    DOUBLE,
                    volume       BIGINT,
                    dividends    DOUBLE,
                    stock_splits DOUBLE,
                    collected_at VARCHAR,
                    PRIMARY KEY (ticker, market, date)
                )
            """)
            prices_con.execute("""
                INSERT INTO prices
                SELECT date, ticker, market, open, high, low, close,
                       adj_close, volume, dividends, stock_splits, collected_at
                FROM _prices_v0
                QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, market, date ORDER BY date) = 1
            """)
            prices_con.execute("DROP TABLE _prices_v0")


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

_PRICE_COL_MAP = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
    "CollectedAt": "collected_at",
    "MarketCap": "market_cap",
}

_PRICE_SCHEMA = [
    "date", "ticker", "market", "open", "high", "low", "close",
    "adj_close", "volume", "dividends", "stock_splits", "collected_at",
    "market_cap",
]


def upsert_prices(df: pd.DataFrame, ticker: str, market: str) -> int:
    """Upsert price rows for one ticker. Returns row count written."""
    if df is None or df.empty:
        return 0

    out = df.copy()
    # Reset DatetimeIndex → column if needed
    if isinstance(out.index, pd.DatetimeIndex) and "Date" not in out.columns:
        out = out.reset_index()

    out = out.rename(columns={k: v for k, v in _PRICE_COL_MAP.items() if k in out.columns})
    out["ticker"] = str(ticker).strip().upper()
    out["market"] = str(market).strip().lower()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"])
    if out.empty:
        return 0

    for col in _PRICE_SCHEMA:
        if col not in out.columns:
            out[col] = None
    out = out[_PRICE_SCHEMA]

    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_prices(out, ticker=ticker, market=market)

    with _lock:
        con = _get_prices_con()
        con.execute(
            "DELETE FROM prices WHERE ticker = ? AND market = ?",
            [str(ticker).strip().upper(), str(market).strip().lower()],
        )
        con.register("_tmp_prices", out)
        con.execute("INSERT INTO prices SELECT * FROM _tmp_prices")
        con.unregister("_tmp_prices")
    return len(out)


def append_prices(df: pd.DataFrame, ticker: str, market: str) -> int:
    """Insert new price rows without removing existing ones (for incremental refresh).

    Uses INSERT OR REPLACE so duplicate (ticker, market, date) rows are updated.
    Returns row count inserted/replaced.
    """
    if df is None or df.empty:
        return 0

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and "Date" not in out.columns:
        out = out.reset_index()

    out = out.rename(columns={k: v for k, v in _PRICE_COL_MAP.items() if k in out.columns})
    out["ticker"] = str(ticker).strip().upper()
    out["market"] = str(market).strip().lower()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"])
    if out.empty:
        return 0

    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_prices(out, ticker=ticker, market=market)

    for col in _PRICE_SCHEMA:
        if col not in out.columns:
            out[col] = None
    out = out[_PRICE_SCHEMA]

    with _lock:
        con = _get_prices_con()
        con.register("_tmp_prices_append", out)
        con.execute("INSERT OR REPLACE INTO prices SELECT * FROM _tmp_prices_append")
        con.unregister("_tmp_prices_append")
    return len(out)


def update_market_cap(ticker: str, market: str, market_cap: float, date: str | None = None) -> None:
    """Update market_cap on the latest (or specified) price row for a ticker."""
    if _use_parquet():
        return  # handled during price upsert
    with _lock:
        con = _get_prices_con()
        if date:
            con.execute(
                "UPDATE prices SET market_cap=? WHERE ticker=? AND market=? AND date=?",
                [market_cap, ticker.upper(), market.lower(), date],
            )
        else:
            con.execute("""
                UPDATE prices SET market_cap=?
                WHERE ticker=? AND market=?
                  AND date=(SELECT MAX(date) FROM prices WHERE ticker=? AND market=?)
            """, [market_cap, ticker.upper(), market.lower(), ticker.upper(), market.lower()])


def get_latest_price_date(ticker: str, market: str) -> pd.Timestamp | None:
    """Return the most recent date stored for a ticker, or None if not found."""
    if _use_parquet():
        from market_data import parquet_store
        d = parquet_store.get_latest_price_date(ticker, market)
        return pd.Timestamp(d) if d else None
    try:
        with _lock:
            con = _get_prices_con()
            row = con.execute(
                "SELECT MAX(date) FROM prices WHERE ticker = ? AND market = ?",
                [str(ticker).strip().upper(), str(market).strip().lower()],
            ).fetchone()
        if row and row[0] is not None:
            return pd.Timestamp(row[0])
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Financials
# ---------------------------------------------------------------------------

_FIN_SCHEMA = [
    "ticker", "market", "term", "fiscal_year", "fiscal_quarter", "fiscal_label", "StatementDate", "PeriodEnd", "PeriodStart",
    "FormType", "FilingDate", "AcceptedAt", "AvailableDate", "AvailabilityMethod",
    "Revenue", "COGS", "Gross Profit", "SG&A", "R&D", "Operating Income", "Net Income",
    "Net Income Common", "EPS", "Diluted EPS", "D&A", "Amortization", "SBC", "Interest", "Pretax Income", "Tax",
    "diluted_eps", "diluted_shares",
    "basic_shares", "net_income_common", "eps_source",
    "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Capital Expenditure",
    "Dividends Paid", "Repurchases",
    "Total Assets", "Total Liabilities", "Shareholders Equity",
    "Current Assets", "Current Liabilities", "AR", "AP", "Inventory", "Cash", "Debt Short", "Debt Long",
    "Deferred Revenue", "Goodwill", "Intangibles",
    "Shares",
    "Diluted Shares", "Basic Shares", "Price", "name", "sector", "industry",
    "Source", "collected_at",
]

_FILING_SCHEMA = [
    "ticker",
    "market",
    "accession",
    "form_type",
    "period_end",
    "report_date",
    "available_date",
    "filing_date",
    "accepted_at",
    "primary_doc_url",
    "index_url",
    "is_amendment",
    "is_nt",
    "collected_at",
]

_SEC_ISSUER_SCHEMA = [
    "ticker",
    "market",
    "cik",
    "company_name",
    "exchange",
    "security_category",
    "is_common_stock",
    "source",
    "collected_at",
]

_SEC_RAW_FACT_SCHEMA = [
    "ticker",
    "market",
    "accession",
    "form_type",
    "fact_name",
    "taxonomy",
    "unit",
    "scale",
    "period_start",
    "period_end",
    "instant_date",
    "value",
    "context_id",
    "dimension_json",
    "filing_date",
    "accepted_at",
    "available_date",
    "availability_method",
    "source",
    "source_url",
    "collected_at",
]

_FIN_EXTRA_SCHEMA = [
    "ticker",
    "market",
    "period_end",
    "available_date",
    "filing_date",
    "accepted_at",
    "form_type",
    "dividends_paid",
    "share_repurchases",
    "sbc",
    "r_and_d",
    "shares_outstanding",
    "shares_eop",
    "ar",
    "inventory",
    "ap",
    "cash",
    "debt_total",
    "net_income",
    "cfo",
    "total_assets",
    "owner_equity",
    "owner_net_income",
    "common_stock",
    "additional_paid_in_capital",
    "retained_earnings",
    "aoci",
    "ppe",
    "ppe_capex",
    "intangibles",
    "intangible_capex",
    "amortization",
    "other_gain",
    "financial_gain",
    "equity_method_gain",
    "other_income",
    "other_expense",
    "financial_income",
    "financial_expense",
    "current_fin_assets",
    "non_current_fin_assets",
    "current_fin_liabilities",
    "non_current_fin_liabilities",
    "source",
    "confidence",
    "collected_at",
]

_DERIVED_SCHEMA = [
    "ticker",
    "market",
    "period_end",
    "available_date",
    "basis",
    "roe",
    "roa",
    "roic",
    "gpa",
    "asset_turnover",
    "leverage",
    "debt_ratio",
    "current_ratio",
    "gross_margin",
    "op_margin",
    "net_margin",
    "cogs_ratio",
    "sga_ratio",
    "total_cost_ratio",
    "revenue_growth",
    "gross_profit_growth",
    "operating_income_growth",
    "net_income_growth",
    "cogs_growth",
    "sga_growth",
    "price_return",
    "ar_turnover",
    "inventory_turnover",
    "ap_turnover",
    "dso",
    "dio",
    "dpo",
    "operating_cycle",
    "cash_cycle",
    "ccr",
    "fcf",
    "eps",
    "bps",
    "sps",
    "ops",
    "oofps",
    "fcfps",
    "per",
    "pbr",
    "psr",
    "por",
    "pfcfr",
    "peg",
    "accruals_ratio",
    "cfo_to_ni",
    "ar_delta",
    "inv_delta",
    "ap_delta",
    "net_wc",
    "net_wc_delta",
    "filing_lag_days",
    "is_amendment",
    "is_nt",
    "punctuality_score",
    "source",
    "collected_at",
]

_CHECKPOINT_WRITE_SCHEMA = [
    "ticker",
    "market",
    "run_id",
    "completed_at",
    "fresh_days",
    "payload",
]
_REMOVED_NONOP_FINANCIAL_COLUMNS = {
    "Other Gain",
    "Financial Gain",
    "Equity Method Gain",
    "Other Income",
    "Other Expense",
    "Financial Income",
    "Financial Expense",
}


def upsert_financials(df: pd.DataFrame, ticker: str, market: str) -> int:
    """Upsert quarterly financial rows for one ticker. Returns row count written."""
    if df is None or df.empty:
        return 0
    out = df.copy()
    drop_cols = [column for column in out.columns if column in _REMOVED_NONOP_FINANCIAL_COLUMNS]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_financials(out, ticker=ticker, market=market)

    out["ticker"] = str(ticker).strip().upper()
    out["market"] = str(market).strip().lower()

    if "symbol" in out.columns:
        out = out.drop(columns=["symbol"])
    if "CollectedAt" in out.columns and "collected_at" not in out.columns:
        out["collected_at"] = out["CollectedAt"]
    if "PeriodEnd" not in out.columns and "StatementDate" in out.columns:
        out["PeriodEnd"] = out["StatementDate"]

    out = out.dropna(subset=["PeriodEnd"])
    if out.empty:
        return 0

    available = [c for c in _FIN_SCHEMA if c in out.columns]
    out = out[available]

    with _lock:
        con = _get_con()
        con.execute(
            "DELETE FROM financials_quarterly WHERE ticker = ? AND market = ?",
            [str(ticker).strip().upper(), str(market).strip().lower()],
        )
        quoted = ", ".join(f'"{c}"' for c in available)
        con.register("_tmp_fin", out)
        con.execute(f"INSERT INTO financials_quarterly ({quoted}) SELECT {quoted} FROM _tmp_fin")
        con.unregister("_tmp_fin")
    return len(out)


def upsert_filings(df: pd.DataFrame, ticker: str, market: str) -> int:
    """Replace filing metadata rows for one ticker."""
    if df is None or df.empty:
        return 0
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_filings(df, ticker=ticker, market=market)
    out = df.copy()
    out["ticker"] = str(ticker).strip().upper()
    out["market"] = str(market).strip().lower()
    out["accession"] = out.get("accession", pd.Series(dtype=object)).astype(str).str.strip()
    out = out.loc[out["accession"] != ""].copy()
    if out.empty:
        return 0
    out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce").dt.date
    out["report_date"] = pd.to_datetime(out.get("report_date"), errors="coerce").dt.date
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce").dt.date
    out["filing_date"] = pd.to_datetime(out.get("filing_date"), errors="coerce").dt.date
    out["accepted_at"] = pd.to_datetime(out.get("accepted_at"), errors="coerce", utc=True)
    out = out.sort_values([c for c in ["filing_date", "accepted_at", "available_date"] if c in out.columns])
    out = out.drop_duplicates(subset=["ticker", "market", "accession"], keep="last")
    for col in _FILING_SCHEMA:
        if col not in out.columns:
            out[col] = None
    out = out[_FILING_SCHEMA]
    with _lock:
        con = _get_con()
        con.execute(
            "DELETE FROM filings WHERE ticker = ? AND market = ?",
            [str(ticker).strip().upper(), str(market).strip().lower()],
        )
        quoted = ", ".join(f'"{c}"' for c in _FILING_SCHEMA)
        con.register("_tmp_filing", out)
        con.execute(f"INSERT INTO filings ({quoted}) SELECT {quoted} FROM _tmp_filing")
        con.unregister("_tmp_filing")
    return len(out)


def upsert_sec_issuer_registry(df: pd.DataFrame, ticker: str, market: str) -> int:
    """Replace SEC issuer metadata row for one ticker."""
    if df is None or df.empty:
        return 0
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_sec_issuer_registry(df, ticker=ticker, market=market)
    out = df.copy()
    out["ticker"] = str(ticker).strip().upper()
    out["market"] = str(market).strip().lower()
    out["cik"] = out.get("cik", pd.Series(dtype=object)).where(
        lambda s: s.notna(),
        None,
    )
    out["company_name"] = out.get("company_name", pd.Series(dtype=object)).where(
        lambda s: s.notna(),
        None,
    )
    out["exchange"] = out.get("exchange", pd.Series(dtype=object)).where(
        lambda s: s.notna(),
        None,
    )
    out["security_category"] = out.get("security_category", pd.Series(dtype=object)).where(
        lambda s: s.notna(),
        None,
    )
    out["is_common_stock"] = out.get("is_common_stock", pd.Series(dtype=object)).astype("boolean")
    out["source"] = out.get("source", pd.Series(dtype=object)).where(
        lambda s: s.notna(),
        None,
    )
    out["collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out = out.dropna(subset=["ticker", "market"])
    if out.empty:
        return 0
    out = out.sort_values([c for c in ["collected_at"] if c in out.columns])
    out = out.drop_duplicates(subset=["ticker", "market"], keep="last")
    for col in _SEC_ISSUER_SCHEMA:
        if col not in out.columns:
            out[col] = None
    out = out[_SEC_ISSUER_SCHEMA]
    with _lock:
        con = _get_prices_con()
        con.execute(
            "DELETE FROM sec_issuer_registry WHERE ticker = ? AND market = ?",
            [str(ticker).strip().upper(), str(market).strip().lower()],
        )
        quoted = ", ".join(f'"{c}"' for c in _SEC_ISSUER_SCHEMA)
        con.register("_tmp_sec_issuer", out)
        con.execute(f"INSERT INTO sec_issuer_registry ({quoted}) SELECT {quoted} FROM _tmp_sec_issuer")
        con.unregister("_tmp_sec_issuer")
    return len(out)


def replace_sec_issuer_registry_bulk(df: pd.DataFrame) -> int:
    """Replace SEC issuer metadata rows for many tickers at once."""
    if df is None or df.empty:
        return 0
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.replace_sec_issuer_registry_bulk(df)
    out = df.copy()
    out["ticker"] = out.get("ticker", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
    out["market"] = out.get("market", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    out["cik"] = out.get("cik", pd.Series(dtype=object)).where(lambda s: s.notna(), None)
    out["company_name"] = out.get("company_name", pd.Series(dtype=object)).where(lambda s: s.notna(), None)
    out["exchange"] = out.get("exchange", pd.Series(dtype=object)).where(lambda s: s.notna(), None)
    out["security_category"] = out.get("security_category", pd.Series(dtype=object)).where(lambda s: s.notna(), None)
    out["is_common_stock"] = out.get("is_common_stock", pd.Series(dtype=object)).astype("boolean")
    out["source"] = out.get("source", pd.Series(dtype=object)).where(lambda s: s.notna(), None)
    out["collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out = out.dropna(subset=["ticker", "market"])
    if out.empty:
        return 0
    out = out.sort_values([c for c in ["collected_at"] if c in out.columns])
    out = out.drop_duplicates(subset=["ticker", "market"], keep="last")
    for col in _SEC_ISSUER_SCHEMA:
        if col not in out.columns:
            out[col] = None
    out = out[_SEC_ISSUER_SCHEMA]
    with _lock:
        con = _get_prices_con()
        con.execute("DELETE FROM sec_issuer_registry")
        quoted = ", ".join(f'"{c}"' for c in _SEC_ISSUER_SCHEMA)
        con.register("_tmp_sec_issuer_bulk", out)
        con.execute(f"INSERT INTO sec_issuer_registry ({quoted}) SELECT {quoted} FROM _tmp_sec_issuer_bulk")
        con.unregister("_tmp_sec_issuer_bulk")
    return len(out)


def upsert_sec_facts_raw_normalized(df: pd.DataFrame, ticker: str, market: str) -> int:
    """Replace SEC raw normalized fact rows for one ticker."""
    if df is None or df.empty:
        return 0
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_sec_facts_raw_normalized(df, ticker=ticker, market=market)
    out = df.copy()
    out["ticker"] = str(ticker).strip().upper()
    out["market"] = str(market).strip().lower()
    out["period_start"] = pd.to_datetime(out.get("period_start"), errors="coerce").dt.date
    out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce").dt.date
    out["instant_date"] = pd.to_datetime(out.get("instant_date"), errors="coerce").dt.date
    out["filing_date"] = pd.to_datetime(out.get("filing_date"), errors="coerce").dt.date
    out["accepted_at"] = pd.to_datetime(out.get("accepted_at"), errors="coerce", utc=True)
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce").dt.date
    out["collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out = out.dropna(subset=["period_end", "fact_name", "value"])
    if out.empty:
        return 0
    out["accession"] = out.get("accession", pd.Series(dtype=object)).fillna("").astype(str).str.strip()
    out["context_id"] = out.get("context_id", pd.Series(dtype=object)).fillna("").astype(str).str.strip()
    dedupe_keys = [
        "ticker",
        "accession",
        "fact_name",
        "unit",
        "period_start",
        "period_end",
        "instant_date",
        "context_id",
    ]
    sort_cols = [c for c in ["filing_date", "accepted_at", "available_date", "collected_at"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    out = out.drop_duplicates(subset=dedupe_keys, keep="last")
    for col in _SEC_RAW_FACT_SCHEMA:
        if col not in out.columns:
            out[col] = None
    out = out[_SEC_RAW_FACT_SCHEMA]
    with _lock:
        con = _get_con()
        con.execute(
            "DELETE FROM sec_facts_raw_normalized WHERE ticker = ? AND market = ?",
            [str(ticker).strip().upper(), str(market).strip().lower()],
        )
        quoted = ", ".join(f'"{c}"' for c in _SEC_RAW_FACT_SCHEMA)
        con.register("_tmp_sec_raw", out)
        con.execute(f"INSERT INTO sec_facts_raw_normalized ({quoted}) SELECT {quoted} FROM _tmp_sec_raw")
        con.unregister("_tmp_sec_raw")
    return len(out)


def upsert_financials_extra(df: pd.DataFrame, ticker: str, market: str) -> int:
    """Replace extended quarterly financial rows for one ticker."""
    if df is None or df.empty:
        return 0
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_financials_extra(df, ticker=ticker, market=market)
    out = df.copy()
    out["ticker"] = str(ticker).strip().upper()
    out["market"] = str(market).strip().lower()
    out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce").dt.date
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce").dt.date
    out["filing_date"] = pd.to_datetime(out.get("filing_date"), errors="coerce").dt.date
    out["accepted_at"] = pd.to_datetime(out.get("accepted_at"), errors="coerce", utc=True)
    out["collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out = out.dropna(subset=["period_end", "available_date"])
    if out.empty:
        return 0
    out = out.sort_values([c for c in ["filing_date", "accepted_at", "available_date", "collected_at"] if c in out.columns])
    out = out.drop_duplicates(
        subset=["ticker", "market", "period_end", "available_date", "form_type"],
        keep="last",
    )
    for col in _FIN_EXTRA_SCHEMA:
        if col not in out.columns:
            out[col] = None
    out = out[_FIN_EXTRA_SCHEMA]
    with _lock:
        con = _get_con()
        con.execute(
            "DELETE FROM financials_quarterly_extra WHERE ticker = ? AND market = ?",
            [str(ticker).strip().upper(), str(market).strip().lower()],
        )
        quoted = ", ".join(f'"{c}"' for c in _FIN_EXTRA_SCHEMA)
        con.register("_tmp_fin_extra", out)
        con.execute(f"INSERT INTO financials_quarterly_extra ({quoted}) SELECT {quoted} FROM _tmp_fin_extra")
        con.unregister("_tmp_fin_extra")
    return len(out)


def upsert_derived_factors(df: pd.DataFrame, ticker: str, market: str) -> int:
    """Replace derived factor rows for one ticker."""
    if df is None or df.empty:
        return 0
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_derived_factors(df, ticker=ticker, market=market)
    out = df.copy()
    out["ticker"] = str(ticker).strip().upper()
    out["market"] = str(market).strip().lower()
    out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce").dt.date
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce").dt.date
    out["basis"] = out.get("basis", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    out["collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out = out.dropna(subset=["period_end", "available_date", "basis"])
    out = out.loc[out["basis"].isin({"quarter", "ttm", "annual"})].copy()
    if out.empty:
        return 0
    out = out.sort_values([c for c in ["available_date", "collected_at"] if c in out.columns])
    out = out.drop_duplicates(
        subset=["ticker", "market", "period_end", "available_date", "basis"],
        keep="last",
    )
    for col in _DERIVED_SCHEMA:
        if col not in out.columns:
            out[col] = None
    out = out[_DERIVED_SCHEMA]
    with _lock:
        con = _get_con()
        con.execute(
            "DELETE FROM derived_factors_quarterly WHERE ticker = ? AND market = ?",
            [str(ticker).strip().upper(), str(market).strip().lower()],
        )
        quoted = ", ".join(f'"{c}"' for c in _DERIVED_SCHEMA)
        con.register("_tmp_derived", out)
        con.execute(f"INSERT INTO derived_factors_quarterly ({quoted}) SELECT {quoted} FROM _tmp_derived")
        con.unregister("_tmp_derived")
    return len(out)


def _normalize_financials_bulk_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_FIN_SCHEMA)

    out = df.copy()
    out["ticker"] = out.get("ticker", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
    out["market"] = out.get("market", pd.Series(dtype=object)).astype(str).str.strip().str.lower()

    if "symbol" in out.columns:
        out = out.drop(columns=["symbol"])
    if "CollectedAt" in out.columns and "collected_at" not in out.columns:
        out["collected_at"] = out["CollectedAt"]
    if "PeriodEnd" not in out.columns and "StatementDate" in out.columns:
        out["PeriodEnd"] = out["StatementDate"]

    out = out.dropna(subset=["ticker", "market", "PeriodEnd"])
    if out.empty:
        return pd.DataFrame(columns=_FIN_SCHEMA)

    available = [c for c in _FIN_SCHEMA if c in out.columns]
    return out[available]


def _normalize_derived_bulk_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_DERIVED_SCHEMA)

    out = df.copy()
    out["ticker"] = out.get("ticker", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
    out["market"] = out.get("market", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce").dt.date
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce").dt.date
    out["basis"] = out.get("basis", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    out["collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out = out.dropna(subset=["ticker", "market", "period_end", "available_date", "basis"])
    out = out.loc[out["basis"].isin({"quarter", "ttm", "annual"})].copy()
    if out.empty:
        return pd.DataFrame(columns=_DERIVED_SCHEMA)
    out = out.sort_values([c for c in ["available_date", "collected_at"] if c in out.columns])
    out = out.drop_duplicates(
        subset=["ticker", "market", "period_end", "available_date", "basis"],
        keep="last",
    )
    for col in _DERIVED_SCHEMA:
        if col not in out.columns:
            out[col] = None
    return out[_DERIVED_SCHEMA]


def _normalize_sec_issuer_bulk_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_SEC_ISSUER_SCHEMA)

    out = df.copy()
    out["ticker"] = out.get("ticker", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
    out["market"] = out.get("market", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    out["cik"] = out.get("cik", pd.Series(dtype=object)).where(lambda s: s.notna(), None)
    out["company_name"] = out.get("company_name", pd.Series(dtype=object)).where(lambda s: s.notna(), None)
    out["exchange"] = out.get("exchange", pd.Series(dtype=object)).where(lambda s: s.notna(), None)
    out["security_category"] = out.get("security_category", pd.Series(dtype=object)).where(lambda s: s.notna(), None)
    out["is_common_stock"] = out.get("is_common_stock", pd.Series(dtype=object)).astype("boolean")
    out["source"] = out.get("source", pd.Series(dtype=object)).where(lambda s: s.notna(), None)
    out["collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out = out.dropna(subset=["ticker", "market"])
    if out.empty:
        return pd.DataFrame(columns=_SEC_ISSUER_SCHEMA)
    out = out.sort_values([c for c in ["collected_at"] if c in out.columns])
    out = out.drop_duplicates(subset=["ticker", "market"], keep="last")
    for col in _SEC_ISSUER_SCHEMA:
        if col not in out.columns:
            out[col] = None
    return out[_SEC_ISSUER_SCHEMA]


def _normalize_filings_bulk_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_FILING_SCHEMA)
    out = df.copy()
    out["ticker"] = out.get("ticker", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
    out["market"] = out.get("market", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    out["accession"] = out.get("accession", pd.Series(dtype=object)).astype(str).str.strip()
    out = out.loc[out["accession"] != ""].copy()
    if out.empty:
        return pd.DataFrame(columns=_FILING_SCHEMA)
    out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce").dt.date
    out["report_date"] = pd.to_datetime(out.get("report_date"), errors="coerce").dt.date
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce").dt.date
    out["filing_date"] = pd.to_datetime(out.get("filing_date"), errors="coerce").dt.date
    out["accepted_at"] = pd.to_datetime(out.get("accepted_at"), errors="coerce", utc=True)
    out = out.sort_values([c for c in ["filing_date", "accepted_at", "available_date"] if c in out.columns])
    out = out.drop_duplicates(subset=["ticker", "market", "accession"], keep="last")
    for col in _FILING_SCHEMA:
        if col not in out.columns:
            out[col] = None
    return out[_FILING_SCHEMA]


def _normalize_sec_raw_bulk_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_SEC_RAW_FACT_SCHEMA)
    out = df.copy()
    out["ticker"] = out.get("ticker", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
    out["market"] = out.get("market", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    out["period_start"] = pd.to_datetime(out.get("period_start"), errors="coerce").dt.date
    out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce").dt.date
    out["instant_date"] = pd.to_datetime(out.get("instant_date"), errors="coerce").dt.date
    out["filing_date"] = pd.to_datetime(out.get("filing_date"), errors="coerce").dt.date
    out["accepted_at"] = pd.to_datetime(out.get("accepted_at"), errors="coerce", utc=True)
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce").dt.date
    out["collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out = out.dropna(subset=["ticker", "market", "period_end", "fact_name", "value"])
    if out.empty:
        return pd.DataFrame(columns=_SEC_RAW_FACT_SCHEMA)
    out["accession"] = out.get("accession", pd.Series(dtype=object)).fillna("").astype(str).str.strip()
    out["context_id"] = out.get("context_id", pd.Series(dtype=object)).fillna("").astype(str).str.strip()
    dedupe_keys = [
        "ticker",
        "accession",
        "fact_name",
        "unit",
        "period_start",
        "period_end",
        "instant_date",
        "context_id",
    ]
    sort_cols = [c for c in ["filing_date", "accepted_at", "available_date", "collected_at"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    out = out.drop_duplicates(subset=dedupe_keys, keep="last")
    for col in _SEC_RAW_FACT_SCHEMA:
        if col not in out.columns:
            out[col] = None
    return out[_SEC_RAW_FACT_SCHEMA]


def _normalize_financials_extra_bulk_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_FIN_EXTRA_SCHEMA)
    out = df.copy()
    out["ticker"] = out.get("ticker", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
    out["market"] = out.get("market", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce").dt.date
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce").dt.date
    out["filing_date"] = pd.to_datetime(out.get("filing_date"), errors="coerce").dt.date
    out["accepted_at"] = pd.to_datetime(out.get("accepted_at"), errors="coerce", utc=True)
    out["collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out = out.dropna(subset=["ticker", "market", "period_end", "available_date"])
    if out.empty:
        return pd.DataFrame(columns=_FIN_EXTRA_SCHEMA)
    out = out.sort_values([c for c in ["filing_date", "accepted_at", "available_date", "collected_at"] if c in out.columns])
    out = out.drop_duplicates(
        subset=["ticker", "market", "period_end", "available_date", "form_type"],
        keep="last",
    )
    for col in _FIN_EXTRA_SCHEMA:
        if col not in out.columns:
            out[col] = None
    return out[_FIN_EXTRA_SCHEMA]


def _normalize_checkpoint_bulk_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_CHECKPOINT_WRITE_SCHEMA)

    out = pd.DataFrame(rows).copy()
    out["ticker"] = out.get("ticker", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
    out["market"] = out.get("market", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
    out["run_id"] = out.get("run_id", pd.Series(dtype=object)).astype(str)
    out["completed_at"] = out.get("completed_at", pd.Series(dtype=object)).astype(str)
    out["payload"] = out.get("payload", pd.Series(dtype=object)).apply(lambda v: json.dumps(v, default=str))
    out = out.dropna(subset=["ticker", "market"])
    if out.empty:
        return pd.DataFrame(columns=_CHECKPOINT_WRITE_SCHEMA)
    out = out.drop_duplicates(subset=["ticker", "market"], keep="last")
    for col in _CHECKPOINT_WRITE_SCHEMA:
        if col not in out.columns:
            out[col] = None
    return out[_CHECKPOINT_WRITE_SCHEMA]


def replace_sec_financials_batch(
    *,
    financials: pd.DataFrame | None,
    derived: pd.DataFrame | None,
    issuer_registry: pd.DataFrame | None,
    filings: pd.DataFrame | None = None,
    raw_facts: pd.DataFrame | None = None,
    financials_extra: pd.DataFrame | None = None,
    checkpoints: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Batch replace SEC financial outputs in a single transaction."""
    if _use_parquet():
        from market_data import parquet_store
        counts = {}
        for key, df_arg, func in [
            ("financials", financials, parquet_store.upsert_financials),
            ("derived", derived, parquet_store.upsert_derived_factors),
            ("issuer_registry", issuer_registry, parquet_store.upsert_sec_issuer_registry),
            ("filings", filings, parquet_store.upsert_filings),
            ("raw_facts", raw_facts, parquet_store.upsert_sec_facts_raw_normalized),
            ("financials_extra", financials_extra, parquet_store.upsert_financials_extra),
        ]:
            if df_arg is not None and not df_arg.empty:
                for ticker, chunk in df_arg.groupby("ticker"):
                    mkt = chunk["market"].iloc[0] if "market" in chunk.columns else "us"
                    func(chunk, ticker=str(ticker), market=str(mkt))
                counts[key] = len(df_arg)
            else:
                counts[key] = 0
        if checkpoints:
            for cp in checkpoints:
                parquet_store.save_checkpoint(cp["ticker"], cp.get("market", "us"), cp.get("payload", {}))
            counts["checkpoints"] = len(checkpoints)
        return counts
    fin_out = _normalize_financials_bulk_frame(financials if financials is not None else pd.DataFrame())
    derived_out = _normalize_derived_bulk_frame(derived if derived is not None else pd.DataFrame())
    issuer_out = _normalize_sec_issuer_bulk_frame(issuer_registry if issuer_registry is not None else pd.DataFrame())
    filings_out = _normalize_filings_bulk_frame(filings if filings is not None else pd.DataFrame())
    raw_out = _normalize_sec_raw_bulk_frame(raw_facts if raw_facts is not None else pd.DataFrame())
    extra_out = _normalize_financials_extra_bulk_frame(financials_extra if financials_extra is not None else pd.DataFrame())
    checkpoint_out = _normalize_checkpoint_bulk_rows(checkpoints or [])

    temp_tables: list[str] = []

    def _register(con, name: str, frame: pd.DataFrame) -> None:
        con.register(name, frame)
        temp_tables.append(name)

    def _delete_pairs(con, table: str, pairs: pd.DataFrame, temp_name: str) -> None:
        if pairs.empty:
            return
        _register(con, temp_name, pairs)
        con.execute(
            f"""
            DELETE FROM {table}
            WHERE EXISTS (
                SELECT 1
                FROM {temp_name} p
                WHERE {table}.ticker = p.ticker
                  AND {table}.market = p.market
            )
            """
        )

    with _lock:
        con = _get_con()
        try:
            con.execute("BEGIN")

            if not fin_out.empty:
                fin_pairs = fin_out[["ticker", "market"]].drop_duplicates().reset_index(drop=True)
                _delete_pairs(con, "financials_quarterly", fin_pairs, "_tmp_fast_fin_pairs")
                quoted = ", ".join(f'"{c}"' for c in fin_out.columns)
                _register(con, "_tmp_fast_financials", fin_out)
                con.execute(f"INSERT INTO financials_quarterly ({quoted}) SELECT {quoted} FROM _tmp_fast_financials")

            if not derived_out.empty:
                derived_pairs = derived_out[["ticker", "market"]].drop_duplicates().reset_index(drop=True)
                _delete_pairs(con, "derived_factors_quarterly", derived_pairs, "_tmp_fast_derived_pairs")
                quoted = ", ".join(f'"{c}"' for c in _DERIVED_SCHEMA)
                _register(con, "_tmp_fast_derived", derived_out)
                con.execute(f"INSERT INTO derived_factors_quarterly ({quoted}) SELECT {quoted} FROM _tmp_fast_derived")

            if not filings_out.empty:
                filings_pairs = filings_out[["ticker", "market"]].drop_duplicates().reset_index(drop=True)
                _delete_pairs(con, "filings", filings_pairs, "_tmp_fast_filings_pairs")
                quoted = ", ".join(f'"{c}"' for c in _FILING_SCHEMA)
                _register(con, "_tmp_fast_filings", filings_out)
                con.execute(f"INSERT INTO filings ({quoted}) SELECT {quoted} FROM _tmp_fast_filings")

            if not raw_out.empty:
                raw_pairs = raw_out[["ticker", "market"]].drop_duplicates().reset_index(drop=True)
                _delete_pairs(con, "sec_facts_raw_normalized", raw_pairs, "_tmp_fast_raw_pairs")
                quoted = ", ".join(f'"{c}"' for c in _SEC_RAW_FACT_SCHEMA)
                _register(con, "_tmp_fast_raw", raw_out)
                con.execute(f"INSERT INTO sec_facts_raw_normalized ({quoted}) SELECT {quoted} FROM _tmp_fast_raw")

            if not extra_out.empty:
                extra_pairs = extra_out[["ticker", "market"]].drop_duplicates().reset_index(drop=True)
                _delete_pairs(con, "financials_quarterly_extra", extra_pairs, "_tmp_fast_extra_pairs")
                quoted = ", ".join(f'"{c}"' for c in _FIN_EXTRA_SCHEMA)
                _register(con, "_tmp_fast_extra", extra_out)
                con.execute(f"INSERT INTO financials_quarterly_extra ({quoted}) SELECT {quoted} FROM _tmp_fast_extra")

            if not checkpoint_out.empty:
                quoted = ", ".join(f'"{c}"' for c in _CHECKPOINT_WRITE_SCHEMA)
                _register(con, "_tmp_fast_checkpoint", checkpoint_out)
                con.execute(
                    f"""
                    INSERT OR REPLACE INTO ingest_checkpoints ({quoted})
                    SELECT {quoted} FROM _tmp_fast_checkpoint
                    """
                )

            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            for name in reversed(temp_tables):
                try:
                    con.unregister(name)
                except Exception:
                    pass

        # Write sec_issuer_registry to prices DB (separate connection)
        if not issuer_out.empty:
            pcon = _get_prices_con()
            issuer_pairs = issuer_out[["ticker", "market"]].drop_duplicates().reset_index(drop=True)
            _delete_pairs(pcon, "sec_issuer_registry", issuer_pairs, "_tmp_fast_issuer_pairs")
            quoted = ", ".join(f'"{c}"' for c in _SEC_ISSUER_SCHEMA)
            pcon.register("_tmp_fast_issuer", issuer_out)
            pcon.execute(f"INSERT INTO sec_issuer_registry ({quoted}) SELECT {quoted} FROM _tmp_fast_issuer")
            pcon.unregister("_tmp_fast_issuer")

    return {
        "financials": int(len(fin_out)),
        "derived": int(len(derived_out)),
        "issuer_registry": int(len(issuer_out)),
        "filings": int(len(filings_out)),
        "raw_facts": int(len(raw_out)),
        "financials_extra": int(len(extra_out)),
        "checkpoints": int(len(checkpoint_out)),
    }


def replace_sec_financials_fast_batch(
    *,
    financials: pd.DataFrame | None,
    derived: pd.DataFrame | None,
    issuer_registry: pd.DataFrame | None,
    checkpoints: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Backward-compatible wrapper for fast-mode SEC batch writes."""
    return replace_sec_financials_batch(
        financials=financials,
        derived=derived,
        issuer_registry=issuer_registry,
        checkpoints=checkpoints,
    )


def get_financial_null_rate_summary(market: str, columns: list[str]) -> dict[str, dict[str, float]]:
    """Return null-rate summary for requested financial columns."""
    out: dict[str, dict[str, float]] = {}
    if not columns:
        return out
    if _use_parquet():
        from market_data import parquet_store
        try:
            df = parquet_store.read_parquet(market, "financials_quarterly")
        except Exception:
            df = pd.DataFrame()
        total = len(df)
        for col in columns:
            non_null = int(df[col].notna().sum()) if col in df.columns else 0
            null_count = max(total - non_null, 0)
            null_rate = (float(null_count) / float(total)) if total > 0 else 1.0
            out[col] = {
                "total": float(total),
                "non_null": float(non_null),
                "null_count": float(null_count),
                "null_rate": float(null_rate),
            }
        return out
    with _lock:
        con = _get_con()
        total_row = con.execute(
            'SELECT COUNT(*) FROM financials_quarterly WHERE market = ?',
            [str(market).strip().lower()],
        ).fetchone()
        total = int(total_row[0]) if total_row else 0
        for col in columns:
            safe_col = str(col).replace('"', '""')
            non_null_row = con.execute(
                f'SELECT COUNT(*) FROM financials_quarterly WHERE market = ? AND "{safe_col}" IS NOT NULL',
                [str(market).strip().lower()],
            ).fetchone()
            non_null = int(non_null_row[0]) if non_null_row else 0
            null_count = max(total - non_null, 0)
            null_rate = (float(null_count) / float(total)) if total > 0 else 1.0
            out[col] = {
                "total": float(total),
                "non_null": float(non_null),
                "null_count": float(null_count),
                "null_rate": float(null_rate),
            }
    return out


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(ticker: str, market: str, payload: dict[str, Any]) -> None:
    """Persist ingest checkpoint to DuckDB (replaces JSON file)."""
    if _use_parquet():
        from market_data import parquet_store
        parquet_store.save_checkpoint(ticker, market, payload)
        return
    with _lock:
        con = _get_con()
        con.execute(
            """
            INSERT OR REPLACE INTO ingest_checkpoints
                (ticker, market, run_id, completed_at, fresh_days, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                str(ticker).strip().upper(),
                str(market).strip().lower(),
                str(payload.get("run_id", "")),
                str(payload.get("completed_at", "")),
                payload.get("fresh_days"),
                json.dumps(payload, default=str),
            ],
        )


def get_checkpoint(ticker: str, market: str) -> dict[str, Any] | None:
    """Return checkpoint payload dict or None if not found."""
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.get_checkpoint(ticker, market)
    try:
        with _lock:
            con = _get_con()
            row = con.execute(
                "SELECT payload FROM ingest_checkpoints WHERE ticker = ? AND market = ? LIMIT 1",
                [str(ticker).strip().upper(), str(market).strip().lower()],
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def get_fresh_tickers(market: str, fresh_days: int | None) -> set[str]:
    """Return set of tickers whose checkpoint is within fresh_days.

    If fresh_days is None, returns all tickers that have any checkpoint.
    """
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.get_fresh_tickers(market, fresh_days)
    try:
        with _lock:
            con = _get_con()
            if fresh_days is None:
                rows = con.execute(
                    "SELECT ticker FROM ingest_checkpoints WHERE market = ?",
                    [str(market).strip().lower()],
                ).fetchall()
                return {r[0] for r in rows}

            # Filter by age: completed_at within fresh_days
            rows = con.execute(
                """
                SELECT ticker, completed_at
                FROM ingest_checkpoints
                WHERE market = ?
                """,
                [str(market).strip().lower()],
            ).fetchall()

        result: set[str] = set()
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=fresh_days)
        for ticker_val, completed_str in rows:
            try:
                completed = pd.to_datetime(completed_str, utc=True)
                if completed >= cutoff:
                    result.add(str(ticker_val))
            except Exception:
                continue
        return result
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def close() -> None:
    """Close the global write connections."""
    if _use_parquet():
        return  # No connections to close
    global _con, _prices_con  # noqa: PLW0603
    if _con is not None:
        try:
            _con.close()
        except Exception:
            pass
        _con = None
    if _prices_con is not None:
        try:
            _prices_con.close()
        except Exception:
            pass
        _prices_con = None
