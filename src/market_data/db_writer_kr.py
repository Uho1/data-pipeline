"""Thread-safe DuckDB write operations for Korea market data."""
from __future__ import annotations

import json
import threading

import pandas as pd

from market_data.config import STORAGE_BACKEND
from market_data.db_kr import DB_PATH
from market_data.db_kr_prices import DB_PATH as PRICES_DB_PATH
from market_data.db_router import normalize_kr_ticker

_lock = threading.Lock()
_con = None
_prices_con = None


def _use_parquet() -> bool:
    return STORAGE_BACKEND == "parquet"


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


def checkpoint() -> None:
    with _lock:
        con = _get_con()
        con.commit()
        con.execute("CHECKPOINT")


def close() -> None:
    global _con  # noqa: PLW0603
    with _lock:
        if _con is None:
            return
        try:
            _con.commit()
        except Exception:
            pass
        try:
            _con.execute("CHECKPOINT")
        except Exception:
            pass
        try:
            _con.close()
        except Exception:
            pass
        _con = None


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
    "Ticker": "ticker",
    "TickerName": "ticker_name",
    "PriceChange": "price_change",
    "PctChange": "pct_change",
    "TradedValue": "traded_value",
    "MarketCap": "market_cap",
    "SharesOutstanding": "shares_outstanding",
    "BPS": "bps",
    "PER": "per",
    "PBR": "pbr",
    "EPS": "eps",
    "DividendYield": "dividend_yield",
    "DPS": "dps",
    "MarketTier": "market_tier",
}

_PRICE_SCHEMA = [
    "date",
    "ticker",
    "market",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
    "collected_at",
    "ticker_name",
    "price_change",
    "pct_change",
    "traded_value",
    "market_cap",
    "shares_outstanding",
    "bps",
    "per",
    "pbr",
    "eps",
    "dividend_yield",
    "dps",
    "market_tier",
]

_TICKER_MASTER_SCHEMA = [
    "ticker",
    "market",
    "market_tier",
    "ticker_name",
    "short_name",
    "is_common_stock",
    "common_stock_filter_reason",
    "listed_date",
    "delisted_date",
    "shares_outstanding",
    "par_value",
    "industry_name",
    "sector_name",
    "subsector_name",
    "krx_industry_name",
    "induty_code",
    "ksic_name_ko",
    "ksic_name_en",
    "sector_code",
    "subsector_code",
    "classification_source",
    "kind_code",
    "kind_name",
    "dart_corp_code",
    "dart_corp_name",
    "representative_index",
    "source",
    "collected_at",
]

_INVESTOR_FLOW_SCHEMA = [
    "date",
    "ticker",
    "market",
    "ticker_name",
    "market_tier",
    "investor_type",
    "investor_type_label",
    "buy_volume",
    "sell_volume",
    "net_volume",
    "buy_value",
    "sell_value",
    "net_value",
    "collected_at",
]

_INDEX_PRICE_SCHEMA = [
    "date",
    "index_code",
    "index_name",
    "market",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "traded_value",
    "price_change",
    "pct_change",
    "collected_at",
]

_FILING_SCHEMA = [
    "ticker",
    "market",
    "accession",
    "corp_code",
    "corp_name",
    "stock_code",
    "report_name",
    "report_code",
    "period_end",
    "filing_date",
    "available_date",
    "accepted_at",
    "receipt_no",
    "filer_name",
    "remarks",
    "source_url",
    "raw_payload",
    "collected_at",
]

_DART_CORP_SCHEMA = [
    "corp_code",
    "corp_name",
    "stock_code",
    "stock_name",
    "corp_name_eng",
    "corp_cls",
    "induty_code",
    "ceo_name",
    "accounting_month",
    "established_date",
    "homepage_url",
    "address",
    "modify_date",
    "market_tier",
    "ticker",
    "is_common_stock",
    "raw_payload",
    "collected_at",
]

_KSIC_DIM_SCHEMA = [
    "ksic_code",
    "name_ko",
    "name_en",
    "level",
    "depth",
    "parent_code",
    "parent_name_ko",
    "section_code",
    "section_name_ko",
    "division_code",
    "division_name_ko",
    "group_code",
    "group_name_ko",
    "class_code",
    "class_name_ko",
    "subclass_code",
    "subclass_name_ko",
    "revision",
    "source_url",
    "collected_at",
]

_DART_FINANCIAL_RAW_SCHEMA = [
    "corp_code",
    "ticker",
    "market",
    "bsns_year",
    "reprt_code",
    "fs_div",
    "sj_div",
    "sj_nm",
    "account_id",
    "account_nm",
    "account_detail",
    "account_key",
    "currency",
    "thstrm_amount",
    "thstrm_add_amount",
    "frmtrm_amount",
    "frmtrm_add_amount",
    "frmtrm_q_amount",
    "bfefrmtrm_amount",
    "receipt_no",
    "ord",
    "raw_payload",
    "filing_date",
    "period_end",
    "source",
    "collected_at",
]

_FIN_SCHEMA = [
    "ticker",
    "market",
    "term",
    "fiscal_year",
    "fiscal_quarter",
    "fiscal_label",
    "StatementDate",
    "PeriodEnd",
    "PeriodStart",
    "FormType",
    "FilingDate",
    "AcceptedAt",
    "AvailableDate",
    "AvailabilityMethod",
    "Revenue",
    "COGS",
    "Gross Profit",
    "SG&A",
    "R&D",
    "Operating Income",
    "Net Income",
    "Net Income Common",
    "EPS",
    "Diluted EPS",
    "D&A",
    "Amortization",
    "SBC",
    "Interest",
    "Pretax Income",
    "Tax",
    "diluted_eps",
    "diluted_shares",
    "basic_shares",
    "net_income_common",
    "eps_source",
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Capital Expenditure",
    "PPE CapEx",
    "Dividends Paid",
    "Repurchases",
    "Total Assets",
    "Total Liabilities",
    "Shareholders Equity",
    "Current Assets",
    "Current Liabilities",
    "AR",
    "AP",
    "Inventory",
    "Cash",
    "Debt Short",
    "Debt Long",
    "Deferred Revenue",
    "Goodwill",
    "Intangibles",
    "Shares",
    "Diluted Shares",
    "Basic Shares",
    "Price",
    "name",
    "sector",
    "industry",
    "Source",
    "collected_at",
]
_REMOVED_KR_FINANCIAL_COLUMNS = {
    "R&D",
    "Trading Gain",
    "Trading Loss",
    "Investment Gain/Loss",
    "Insurance Finance Income",
    "Insurance Finance Expense",
    "Reinsurance Finance Income",
    "Reinsurance Finance Expense",
    "Other Operating Income Component",
}


def init_schema() -> None:
    if _use_parquet():
        from market_data import parquet_store
        parquet_store.init_dirs("kr")
        return
    with _lock:
        con = _get_con()
        prices_con = _get_prices_con()
        prices_con.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                date DATE NOT NULL,
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                dividends DOUBLE,
                stock_splits DOUBLE,
                collected_at VARCHAR,
                ticker_name VARCHAR,
                price_change DOUBLE,
                pct_change DOUBLE,
                traded_value DOUBLE,
                market_cap DOUBLE,
                shares_outstanding DOUBLE,
                bps DOUBLE,
                per DOUBLE,
                pbr DOUBLE,
                eps DOUBLE,
                dividend_yield DOUBLE,
                dps DOUBLE,
                market_tier VARCHAR,
                PRIMARY KEY (ticker, market, date)
            )
            """
        )
        prices_con.execute(
            """
            CREATE TABLE IF NOT EXISTS ticker_master (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                market_tier VARCHAR,
                ticker_name VARCHAR,
                short_name VARCHAR,
                is_common_stock BOOLEAN,
                common_stock_filter_reason VARCHAR,
                listed_date DATE,
                delisted_date DATE,
                shares_outstanding DOUBLE,
                par_value DOUBLE,
                industry_name VARCHAR,
                sector_name VARCHAR,
                subsector_name VARCHAR,
                krx_industry_name VARCHAR,
                induty_code VARCHAR,
                ksic_name_ko VARCHAR,
                ksic_name_en VARCHAR,
                sector_code VARCHAR,
                subsector_code VARCHAR,
                classification_source VARCHAR,
                kind_code VARCHAR,
                kind_name VARCHAR,
                dart_corp_code VARCHAR,
                dart_corp_name VARCHAR,
                representative_index VARCHAR,
                source VARCHAR,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (ticker)
            )
            """
        )
        prices_con.execute(
            """
            CREATE TABLE IF NOT EXISTS index_prices (
                date DATE NOT NULL,
                index_code VARCHAR NOT NULL,
                index_name VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                traded_value DOUBLE,
                price_change DOUBLE,
                pct_change DOUBLE,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (index_code, date)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS investor_flows (
                date DATE NOT NULL,
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                ticker_name VARCHAR,
                market_tier VARCHAR,
                investor_type VARCHAR NOT NULL,
                investor_type_label VARCHAR,
                buy_volume BIGINT,
                sell_volume BIGINT,
                net_volume BIGINT,
                buy_value DOUBLE,
                sell_value DOUBLE,
                net_value DOUBLE,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (ticker, market, date, investor_type)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS filings (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                accession VARCHAR NOT NULL,
                corp_code VARCHAR,
                corp_name VARCHAR,
                stock_code VARCHAR,
                report_name VARCHAR,
                report_code VARCHAR,
                period_end DATE,
                filing_date DATE,
                available_date DATE,
                accepted_at TIMESTAMPTZ,
                receipt_no VARCHAR,
                filer_name VARCHAR,
                remarks VARCHAR,
                source_url VARCHAR,
                raw_payload VARCHAR,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (ticker, market, accession)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS dart_corp_master (
                corp_code VARCHAR NOT NULL,
                corp_name VARCHAR,
                stock_code VARCHAR,
                stock_name VARCHAR,
                corp_name_eng VARCHAR,
                corp_cls VARCHAR,
                induty_code VARCHAR,
                ceo_name VARCHAR,
                accounting_month VARCHAR,
                established_date DATE,
                homepage_url VARCHAR,
                address VARCHAR,
                modify_date DATE,
                market_tier VARCHAR,
                ticker VARCHAR,
                is_common_stock BOOLEAN,
                raw_payload VARCHAR,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (corp_code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ksic_dim (
                ksic_code VARCHAR NOT NULL,
                name_ko VARCHAR,
                name_en VARCHAR,
                level VARCHAR,
                depth INTEGER,
                parent_code VARCHAR,
                parent_name_ko VARCHAR,
                section_code VARCHAR,
                section_name_ko VARCHAR,
                division_code VARCHAR,
                division_name_ko VARCHAR,
                group_code VARCHAR,
                group_name_ko VARCHAR,
                class_code VARCHAR,
                class_name_ko VARCHAR,
                subclass_code VARCHAR,
                subclass_name_ko VARCHAR,
                revision INTEGER,
                source_url VARCHAR,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (ksic_code)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS dart_financials_raw (
                corp_code VARCHAR NOT NULL,
                ticker VARCHAR,
                market VARCHAR,
                bsns_year INTEGER NOT NULL,
                reprt_code VARCHAR NOT NULL,
                fs_div VARCHAR NOT NULL,
                sj_div VARCHAR NOT NULL,
                sj_nm VARCHAR,
                account_id VARCHAR,
                account_nm VARCHAR,
                account_detail VARCHAR,
                account_key VARCHAR NOT NULL,
                currency VARCHAR,
                thstrm_amount DOUBLE,
                thstrm_add_amount DOUBLE,
                frmtrm_amount DOUBLE,
                frmtrm_add_amount DOUBLE,
                frmtrm_q_amount DOUBLE,
                bfefrmtrm_amount DOUBLE,
                receipt_no VARCHAR,
                ord DOUBLE,
                raw_payload VARCHAR,
                filing_date DATE,
                period_end DATE,
                source VARCHAR,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (corp_code, bsns_year, reprt_code, fs_div, sj_div, account_key)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS financials_quarterly (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                term VARCHAR,
                fiscal_year INTEGER,
                fiscal_quarter INTEGER,
                fiscal_label VARCHAR,
                "StatementDate" DATE,
                "PeriodEnd" DATE NOT NULL,
                "PeriodStart" DATE,
                "FormType" VARCHAR,
                "FilingDate" DATE,
                "AcceptedAt" TIMESTAMPTZ,
                "AvailableDate" DATE,
                "AvailabilityMethod" VARCHAR,
                "Revenue" DOUBLE,
                "COGS" DOUBLE,
                "Gross Profit" DOUBLE,
                "SG&A" DOUBLE,
                "R&D" DOUBLE,
                "Operating Income" DOUBLE,
                "Net Income" DOUBLE,
                "Net Income Common" DOUBLE,
                "EPS" DOUBLE,
                "Diluted EPS" DOUBLE,
                "D&A" DOUBLE,
                "Amortization" DOUBLE,
                "SBC" DOUBLE,
                "Interest" DOUBLE,
                "Pretax Income" DOUBLE,
                "Tax" DOUBLE,
                diluted_eps DOUBLE,
                diluted_shares DOUBLE,
                basic_shares DOUBLE,
                net_income_common DOUBLE,
                eps_source VARCHAR,
                "Operating Cash Flow" DOUBLE,
                "Investing Cash Flow" DOUBLE,
                "Financing Cash Flow" DOUBLE,
                "Capital Expenditure" DOUBLE,
                "PPE CapEx" DOUBLE,
                "Dividends Paid" DOUBLE,
                "Repurchases" DOUBLE,
                "Total Assets" DOUBLE,
                "Total Liabilities" DOUBLE,
                "Shareholders Equity" DOUBLE,
                "Current Assets" DOUBLE,
                "Current Liabilities" DOUBLE,
                "AR" DOUBLE,
                "AP" DOUBLE,
                "Inventory" DOUBLE,
                "Cash" DOUBLE,
                "Debt Short" DOUBLE,
                "Debt Long" DOUBLE,
                "Deferred Revenue" DOUBLE,
                "Goodwill" DOUBLE,
                "Intangibles" DOUBLE,
                "Shares" DOUBLE,
                "Diluted Shares" DOUBLE,
                "Basic Shares" DOUBLE,
                "Price" DOUBLE,
                name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                "Source" VARCHAR,
                collected_at VARCHAR
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_checkpoints (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                run_id VARCHAR,
                completed_at VARCHAR,
                fresh_days INTEGER,
                payload VARCHAR,
                PRIMARY KEY (ticker, market)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS segment_facts_quarterly (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                period_end DATE NOT NULL,
                available_date DATE,
                segment_type VARCHAR NOT NULL,
                segment_name VARCHAR NOT NULL,
                metric VARCHAR NOT NULL,
                value DOUBLE,
                currency VARCHAR,
                accession VARCHAR,
                period_start DATE,
                form_type VARCHAR,
                filing_date DATE,
                accepted_at TIMESTAMPTZ,
                availability_method VARCHAR,
                source VARCHAR,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (ticker, market, period_end, segment_type, segment_name, metric)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS segment_revenue_quarterly (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                period_end DATE NOT NULL,
                available_date DATE,
                period_start DATE,
                form_type VARCHAR,
                filing_date DATE,
                accepted_at TIMESTAMPTZ,
                availability_method VARCHAR,
                segment_type VARCHAR NOT NULL,
                segment_name VARCHAR NOT NULL,
                revenue DOUBLE,
                op_income DOUBLE,
                source VARCHAR,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (ticker, market, period_end, segment_type, segment_name)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS segment_customer_facts_quarterly (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                period_end DATE NOT NULL,
                available_date DATE,
                segment_type VARCHAR NOT NULL,
                segment_name VARCHAR NOT NULL,
                metric VARCHAR NOT NULL,
                value DOUBLE,
                currency VARCHAR,
                accession VARCHAR,
                period_start DATE,
                form_type VARCHAR,
                filing_date DATE,
                accepted_at TIMESTAMPTZ,
                availability_method VARCHAR,
                source VARCHAR,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (ticker, market, period_end, segment_type, segment_name, metric)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS segment_customer_revenue_quarterly (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                period_end DATE NOT NULL,
                available_date DATE,
                period_start DATE,
                form_type VARCHAR,
                filing_date DATE,
                accepted_at TIMESTAMPTZ,
                availability_method VARCHAR,
                segment_type VARCHAR NOT NULL,
                segment_name VARCHAR NOT NULL,
                revenue DOUBLE,
                op_income DOUBLE,
                source VARCHAR,
                collected_at TIMESTAMPTZ,
                PRIMARY KEY (ticker, market, period_end, segment_type, segment_name)
            )
            """
        )
        # YTD → quarterly derived view
        # DART reports: Q1 (3m), H1 (6m YTD), Q3 (9m YTD), FY (12m)
        # BUT ~30% of tickers report H1/Q3 as already-quarterly (not cumulative).
        # Detection: if majority of segments have H1_revenue < Q1_revenue * 1.3,
        # then this (ticker, segment_type, year) group is quarterly, not cumulative.
        # Samsung switched from quarterly→cumulative in 2024, so per-year detection is critical.
        # Build canonical alias CASE expression from segment_aliases module
        try:
            from market_data.kr_dart.segment_aliases import build_alias_case_sql
            _alias_case = build_alias_case_sql("segment_name_clean")
        except ImportError:
            _alias_case = "segment_name_clean"  # fallback: no alias resolution

        con.execute(
            f"""
            CREATE OR REPLACE VIEW segment_revenue_quarterly_derived AS
            WITH
            -- Step 0: Clean segment names and filter out comparative-period FY rows.
            cleaned AS (
                SELECT
                    ticker, market, period_end, available_date, segment_type,
                    -- Basic cleanup: remove HTML artifacts, collapse whitespace
                    REGEXP_REPLACE(
                        REGEXP_REPLACE(REPLACE(segment_name, '&cr', ' '), '\\s+', ' ', 'g'),
                        '^\\s+|\\s+$', '', 'g'
                    ) AS segment_name_raw,
                    -- Also strip common suffixes (부문/사업부/본부/등) for alias matching
                    REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(
                                REGEXP_REPLACE(REPLACE(segment_name, '&cr', ' '), '\\s+', ' ', 'g'),
                                '^\\s+|\\s+$', '', 'g'
                            ),
                            '\\s*(사업부문|사업부|부문|본부)\\s*$', '', 'g'
                        ),
                        '\\s+', ' ', 'g'
                    ) AS segment_name_clean,
                    metric, value, currency, accession, period_start, form_type,
                    filing_date, accepted_at, availability_method, source, collected_at
                FROM segment_facts_quarterly
                WHERE form_type IN ('Q1', 'H1', 'Q3', 'FY')
                  AND NOT (
                      form_type = 'FY'
                      AND period_start IS NOT NULL
                      AND EXTRACT(YEAR FROM period_start) < EXTRACT(YEAR FROM period_end)
                  )
            ),
            -- Step 0b: Apply canonical segment name aliases (반도체→DS, CE→DX, DP→SDC, etc.)
            -- This maps old/new segment era names to a single canonical form.
            -- The alias CASE expression is generated from segment_aliases.py at schema init time.
            canonicalized AS (
                SELECT
                    ticker, market, period_end, available_date, segment_type,
                    segment_name_raw,
                    {_alias_case} AS segment_name,
                    metric, value, currency, accession, period_start, form_type,
                    filing_date, accepted_at, availability_method, source, collected_at
                FROM cleaned
            ),
            -- Step 0c: Deduplicate after canonical name resolution.
            -- When old-name (반도체) and new-name (DS) both map to "DS" for the same period,
            -- keep only the most recently filed one.
            deduped_canonical AS (
                SELECT *
                FROM (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY ticker, market, segment_type, segment_name,
                                         period_end, metric
                            ORDER BY accepted_at DESC NULLS LAST, filing_date DESC NULLS LAST
                        ) AS _dedup_rn
                    FROM canonicalized
                ) sub
                WHERE _dedup_rn = 1
            ),
            ranked AS (
                SELECT
                    ticker, market, period_end, available_date, segment_type,
                    segment_name, segment_name_raw,
                    metric, value, currency, accession, period_start, form_type,
                    filing_date, accepted_at, availability_method, source, collected_at,
                    CASE form_type
                        WHEN 'Q1' THEN 1
                        WHEN 'H1' THEN 2
                        WHEN 'Q3' THEN 3
                        WHEN 'FY' THEN 4
                        ELSE NULL
                    END AS report_order,
                    EXTRACT(YEAR FROM period_end) AS report_year
                FROM deduped_canonical
            ),
            -- Detect if H1 values are YTD-cumulative or already-quarterly
            -- per (ticker, market, segment_type, year) using majority voting on revenue.
            -- Uses normalized segment names so name variants (아시아 및&cr아프리카
            -- vs 아시아 및아프리카) don't prevent Q1↔H1 matching.
            ytd_detect AS (
                SELECT ticker, market, segment_type, report_year,
                    CASE
                        WHEN SUM(CASE WHEN h1_rev > q1_rev * 1.3 THEN 1 ELSE 0 END) >=
                             SUM(CASE WHEN h1_rev <= q1_rev * 1.3 THEN 1 ELSE 0 END)
                        THEN TRUE
                        ELSE FALSE
                    END AS is_ytd
                FROM (
                    SELECT ticker, market, segment_type, report_year, segment_name,
                        MAX(CASE WHEN report_order = 1 THEN value END) AS q1_rev,
                        MAX(CASE WHEN report_order = 2 THEN value END) AS h1_rev
                    FROM ranked
                    WHERE metric = 'revenue'
                    GROUP BY ticker, market, segment_type, report_year, segment_name
                ) sub
                WHERE q1_rev IS NOT NULL AND h1_rev IS NOT NULL AND q1_rev > 0
                GROUP BY ticker, market, segment_type, report_year
            ),
            with_detect AS (
                SELECT r.*,
                    COALESCE(d.is_ytd, TRUE) AS is_ytd_mode
                FROM ranked r
                LEFT JOIN ytd_detect d
                    ON r.ticker = d.ticker AND r.market = d.market
                    AND r.segment_type = d.segment_type AND r.report_year = d.report_year
            ),
            -- For non-cumulative mode: sum of Q1+Q2(=H1)+Q3 to derive Q4
            q123_sums AS (
                SELECT ticker, market, segment_type, segment_name, metric, report_year,
                    SUM(value) AS sum_q123
                FROM with_detect
                WHERE NOT is_ytd_mode AND report_order IN (1, 2, 3)
                GROUP BY ticker, market, segment_type, segment_name, metric, report_year
            ),
            -- Detect FY=Q4 standalone at GROUP level (ticker, segment_type, year).
            -- If majority of revenue segments have FY < sum_q123 * 0.6,
            -- then ALL FY values in this group are Q4 standalone.
            -- This handles edge cases where individual segments have FY close to sum_q123
            -- but the overall pattern clearly shows Q4-only reporting.
            fy_standalone_detect AS (
                SELECT s.ticker, s.market, s.segment_type, s.report_year,
                    CASE WHEN SUM(CASE WHEN fy.value < s.sum_q123 * 0.6 AND s.sum_q123 > 0 THEN 1 ELSE 0 END) >
                              SUM(CASE WHEN fy.value >= s.sum_q123 * 0.6 OR s.sum_q123 <= 0 THEN 1 ELSE 0 END)
                         THEN TRUE ELSE FALSE
                    END AS is_fy_q4_standalone
                FROM q123_sums s
                INNER JOIN with_detect fy
                    ON s.ticker = fy.ticker AND s.market = fy.market
                    AND s.segment_type = fy.segment_type AND s.segment_name = fy.segment_name
                    AND s.metric = fy.metric AND s.report_year = fy.report_year
                    AND fy.report_order = 4
                WHERE s.metric = 'revenue' AND NOT fy.is_ytd_mode
                GROUP BY s.ticker, s.market, s.segment_type, s.report_year
            ),
            with_lag AS (
                SELECT
                    d.*,
                    LAG(value) OVER (
                        PARTITION BY d.ticker, d.market, d.segment_type, d.segment_name, d.metric, d.report_year
                        ORDER BY d.report_order
                    ) AS prev_value,
                    LAG(d.report_order) OVER (
                        PARTITION BY d.ticker, d.market, d.segment_type, d.segment_name, d.metric, d.report_year
                        ORDER BY d.report_order
                    ) AS prev_order,
                    s.sum_q123,
                    COALESCE(fsd.is_fy_q4_standalone, FALSE) AS is_fy_q4_standalone
                FROM with_detect d
                LEFT JOIN q123_sums s
                    ON d.ticker = s.ticker AND d.market = s.market
                    AND d.segment_type = s.segment_type AND d.segment_name = s.segment_name
                    AND d.metric = s.metric AND d.report_year = s.report_year
                LEFT JOIN fy_standalone_detect fsd
                    ON d.ticker = fsd.ticker AND d.market = fsd.market
                    AND d.segment_type = fsd.segment_type AND d.report_year = fsd.report_year
            ),
            quarterly AS (
                SELECT
                    ticker, market, period_end, available_date, segment_type, segment_name,
                    segment_name_raw,
                    metric, currency, accession, period_start, form_type, filing_date,
                    accepted_at, availability_method, source, collected_at,
                    report_order, prev_order, value, prev_value, is_ytd_mode,
                    CASE
                        -- Q1: always use directly
                        WHEN report_order = 1 THEN value

                        -- === Cumulative (YTD) mode ===
                        -- Q2 = H1_ytd - Q1, Q3 = Q3_ytd - H1_ytd
                        WHEN is_ytd_mode AND report_order IN (2, 3) AND prev_order = report_order - 1
                            THEN value - prev_value
                        -- Q4 = FY - Q3_ytd
                        WHEN is_ytd_mode AND report_order = 4 AND prev_order = 3
                            THEN value - prev_value

                        -- === Non-cumulative (quarterly) mode ===
                        -- H1 and Q3 are already quarterly values
                        WHEN NOT is_ytd_mode AND report_order IN (2, 3)
                            THEN value
                        -- Q4: detect if FY is annual total or Q4 standalone
                        -- Group-level detection: if majority of segments show FY << sum_q123,
                        -- ALL segments in this group use FY directly as Q4
                        WHEN NOT is_ytd_mode AND report_order = 4 AND is_fy_q4_standalone
                            THEN value  -- FY IS Q4 standalone (group-level detected)
                        -- Normal case: Q4 = FY(annual) - (Q1 + Q2 + Q3)
                        WHEN NOT is_ytd_mode AND report_order = 4 AND sum_q123 IS NOT NULL
                            THEN value - sum_q123

                        -- FY-only data (no Q1/H1/Q3): use FY value directly.
                        -- Common for geographic disclosures which are annual-only.
                        WHEN report_order = 4 AND prev_order IS NULL
                            THEN value
                        ELSE NULL
                    END AS value_q,
                    CASE
                        WHEN report_order = 1 THEN FALSE
                        -- Cumulative mode: missing predecessor = can't derive
                        WHEN is_ytd_mode AND report_order IN (2, 3, 4) AND prev_order IS NULL THEN TRUE
                        -- Non-cumulative mode: H1/Q3 always valid (already quarterly)
                        WHEN NOT is_ytd_mode AND report_order IN (2, 3) THEN FALSE
                        -- Non-cumulative FY without all 3 prior quarters
                        WHEN NOT is_ytd_mode AND report_order = 4 AND sum_q123 IS NULL THEN TRUE
                        ELSE FALSE
                    END AS is_ytd_only
                FROM with_lag
            )
            SELECT
                ticker, market, period_end, available_date, segment_type, segment_name,
                segment_name_raw,
                metric, value_q AS value, currency, accession, period_start, form_type,
                filing_date, accepted_at, availability_method, source, collected_at,
                is_ytd_only
            FROM quarterly
            WHERE value_q IS NOT NULL
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS segment_ingest_tickers (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                attempted_at TIMESTAMPTZ,
                found_segments INTEGER,
                PRIMARY KEY (ticker, market)
            )
            """
        )
        con.execute("ALTER TABLE dart_financials_raw ADD COLUMN IF NOT EXISTS sj_nm VARCHAR")
        con.execute("ALTER TABLE dart_financials_raw ADD COLUMN IF NOT EXISTS account_detail VARCHAR")
        con.execute("ALTER TABLE dart_financials_raw ADD COLUMN IF NOT EXISTS thstrm_add_amount DOUBLE")
        con.execute("ALTER TABLE dart_financials_raw ADD COLUMN IF NOT EXISTS frmtrm_add_amount DOUBLE")
        con.execute("ALTER TABLE dart_financials_raw ADD COLUMN IF NOT EXISTS frmtrm_q_amount DOUBLE")
        con.execute("ALTER TABLE dart_financials_raw ADD COLUMN IF NOT EXISTS receipt_no VARCHAR")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS fiscal_year INTEGER")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS fiscal_quarter INTEGER")
        con.execute("ALTER TABLE financials_quarterly ADD COLUMN IF NOT EXISTS fiscal_label VARCHAR")
        prices_con.execute("ALTER TABLE ticker_master ADD COLUMN IF NOT EXISTS subsector_name VARCHAR")
        prices_con.execute("ALTER TABLE ticker_master ADD COLUMN IF NOT EXISTS krx_industry_name VARCHAR")
        prices_con.execute("ALTER TABLE ticker_master ADD COLUMN IF NOT EXISTS induty_code VARCHAR")
        prices_con.execute("ALTER TABLE ticker_master ADD COLUMN IF NOT EXISTS ksic_name_ko VARCHAR")
        prices_con.execute("ALTER TABLE ticker_master ADD COLUMN IF NOT EXISTS ksic_name_en VARCHAR")
        prices_con.execute("ALTER TABLE ticker_master ADD COLUMN IF NOT EXISTS sector_code VARCHAR")
        prices_con.execute("ALTER TABLE ticker_master ADD COLUMN IF NOT EXISTS subsector_code VARCHAR")
        prices_con.execute("ALTER TABLE ticker_master ADD COLUMN IF NOT EXISTS classification_source VARCHAR")
        con.execute("ALTER TABLE dart_corp_master ADD COLUMN IF NOT EXISTS stock_name VARCHAR")
        con.execute("ALTER TABLE dart_corp_master ADD COLUMN IF NOT EXISTS corp_name_eng VARCHAR")
        con.execute("ALTER TABLE dart_corp_master ADD COLUMN IF NOT EXISTS corp_cls VARCHAR")
        con.execute("ALTER TABLE dart_corp_master ADD COLUMN IF NOT EXISTS induty_code VARCHAR")
        con.execute("ALTER TABLE dart_corp_master ADD COLUMN IF NOT EXISTS ceo_name VARCHAR")
        con.execute("ALTER TABLE dart_corp_master ADD COLUMN IF NOT EXISTS accounting_month VARCHAR")
        con.execute("ALTER TABLE dart_corp_master ADD COLUMN IF NOT EXISTS established_date DATE")
        con.execute("ALTER TABLE dart_corp_master ADD COLUMN IF NOT EXISTS homepage_url VARCHAR")
        con.execute("ALTER TABLE dart_corp_master ADD COLUMN IF NOT EXISTS address VARCHAR")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS capacity_production_quarterly (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                period_end DATE NOT NULL,
                available_date DATE,
                section VARCHAR NOT NULL,
                product_name VARCHAR NOT NULL,
                unit VARCHAR,
                value DOUBLE,
                currency VARCHAR DEFAULT 'KRW',
                accession VARCHAR,
                form_type VARCHAR,
                filing_date DATE,
                source VARCHAR DEFAULT 'dart_html',
                collected_at TIMESTAMPTZ
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS capacity_ingest_tickers (
                ticker VARCHAR NOT NULL,
                market VARCHAR NOT NULL,
                attempted_at TIMESTAMPTZ,
                rows_written INTEGER DEFAULT 0,
                PRIMARY KEY (ticker, market)
            )
            """
        )


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out[columns]


def _write_table(table: str, out: pd.DataFrame, columns: list[str]) -> int:
    if out.empty:
        return 0
    out = out.loc[:, columns].copy()
    quoted = ", ".join(f'"{col}"' for col in columns)
    with _lock:
        con = _get_con()
        con.register("_tmp_write", out)
        con.execute(f"INSERT INTO {table} ({quoted}) SELECT {quoted} FROM _tmp_write")
        con.unregister("_tmp_write")
        con.commit()
    return len(out)


def _write_table_prices(table: str, out: pd.DataFrame, columns: list[str]) -> int:
    """Write to prices DB (prices, ticker_master, index_prices)."""
    if out.empty:
        return 0
    out = out.loc[:, columns].copy()
    quoted = ", ".join(f'"{col}"' for col in columns)
    with _lock:
        con = _get_prices_con()
        con.register("_tmp_write", out)
        con.execute(f"INSERT INTO {table} ({quoted}) SELECT {quoted} FROM _tmp_write")
        con.unregister("_tmp_write")
        con.commit()
    return len(out)


def replace_ticker_master(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.replace_ticker_master(df, market='kr')
    out = _ensure_columns(df.copy(), _TICKER_MASTER_SCHEMA)
    out["ticker"] = out["ticker"].astype(str).map(normalize_kr_ticker)
    out["market"] = out["market"].fillna("kr").astype(str).str.lower()
    out["listed_date"] = pd.to_datetime(out["listed_date"], errors="coerce").dt.date
    out["delisted_date"] = pd.to_datetime(out["delisted_date"], errors="coerce").dt.date
    out["collected_at"] = pd.to_datetime(out["collected_at"], errors="coerce", utc=True)
    out = out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="last")
    with _lock:
        con = _get_prices_con()
        con.execute("DELETE FROM ticker_master")
    return _write_table_prices("ticker_master", out, _TICKER_MASTER_SCHEMA)


def upsert_prices(df: pd.DataFrame, ticker: str, market: str = "kr") -> int:
    if df is None or df.empty:
        return 0
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and "Date" not in out.columns:
        out = out.reset_index()
    out = out.rename(columns={src: dst for src, dst in _PRICE_COL_MAP.items() if src in out.columns})
    out["ticker"] = normalize_kr_ticker(ticker)
    out["market"] = str(market).strip().lower()
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce").dt.date
    out = out.dropna(subset=["date"])
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_prices(out, ticker=ticker, market=market)
    out = _ensure_columns(out, _PRICE_SCHEMA)
    with _lock:
        con = _get_prices_con()
        con.execute("DELETE FROM prices WHERE ticker = ? AND market = ?", [normalize_kr_ticker(ticker), str(market).strip().lower()])
    return _write_table_prices("prices", out, _PRICE_SCHEMA)


def upsert_investor_flows(df: pd.DataFrame, ticker: str, market: str = "kr") -> int:
    if df is None or df.empty:
        return 0
    out = _ensure_columns(df.copy(), _INVESTOR_FLOW_SCHEMA)
    out["ticker"] = normalize_kr_ticker(ticker)
    out["market"] = str(market).strip().lower()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["collected_at"] = pd.to_datetime(out["collected_at"], errors="coerce", utc=True)
    out = out.dropna(subset=["date", "investor_type"])
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_investor_flows(out, ticker=ticker, market=market)
    with _lock:
        con = _get_con()
        con.execute("DELETE FROM investor_flows WHERE ticker = ? AND market = ?", [normalize_kr_ticker(ticker), str(market).strip().lower()])
    return _write_table("investor_flows", out, _INVESTOR_FLOW_SCHEMA)


def upsert_index_prices(df: pd.DataFrame, index_code: str) -> int:
    if df is None or df.empty:
        return 0
    out = _ensure_columns(df.copy(), _INDEX_PRICE_SCHEMA)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["collected_at"] = pd.to_datetime(out["collected_at"], errors="coerce", utc=True)
    out = out.dropna(subset=["date", "index_code"])
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_index_prices(out, index_code=index_code, market='kr')
    with _lock:
        con = _get_prices_con()
        con.execute("DELETE FROM index_prices WHERE index_code = ?", [str(index_code).strip()])
    return _write_table_prices("index_prices", out, _INDEX_PRICE_SCHEMA)


def replace_dart_corp_master(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.replace_dart_corp_master(df, market='kr')
    out = _ensure_columns(df.copy(), _DART_CORP_SCHEMA)
    out["established_date"] = pd.to_datetime(out["established_date"], errors="coerce").dt.date
    out["modify_date"] = pd.to_datetime(out["modify_date"], errors="coerce").dt.date
    out["collected_at"] = pd.to_datetime(out["collected_at"], errors="coerce", utc=True)
    out = out.dropna(subset=["corp_code"]).drop_duplicates(subset=["corp_code"], keep="last")
    with _lock:
        con = _get_con()
        con.execute("DELETE FROM dart_corp_master")
    return _write_table("dart_corp_master", out, _DART_CORP_SCHEMA)


def replace_ksic_dim(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.replace_ksic_dim(df, market='kr')
    out = _ensure_columns(df.copy(), _KSIC_DIM_SCHEMA)
    out["collected_at"] = pd.to_datetime(out["collected_at"], errors="coerce", utc=True)
    out = out.dropna(subset=["ksic_code"]).drop_duplicates(subset=["ksic_code"], keep="last")
    with _lock:
        con = _get_con()
        con.execute("DELETE FROM ksic_dim")
    return _write_table("ksic_dim", out, _KSIC_DIM_SCHEMA)


def upsert_filings(df: pd.DataFrame, ticker: str, market: str = "kr") -> int:
    if df is None or df.empty:
        return 0
    out = _ensure_columns(df.copy(), _FILING_SCHEMA)
    out["ticker"] = normalize_kr_ticker(ticker)
    out["market"] = str(market).strip().lower()
    for column in ("period_end", "filing_date", "available_date"):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.date
    out["accepted_at"] = pd.to_datetime(out["accepted_at"], errors="coerce", utc=True)
    out["collected_at"] = pd.to_datetime(out["collected_at"], errors="coerce", utc=True)
    out = out.dropna(subset=["accession"]).drop_duplicates(subset=["ticker", "market", "accession"], keep="last")
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_filings(out, ticker=ticker, market=market)
    with _lock:
        con = _get_con()
        con.execute("DELETE FROM filings WHERE ticker = ? AND market = ?", [normalize_kr_ticker(ticker), str(market).strip().lower()])
    return _write_table("filings", out, _FILING_SCHEMA)


def upsert_dart_financials_raw(df: pd.DataFrame, *, corp_code: str) -> int:
    if df is None or df.empty:
        return 0
    out = _ensure_columns(df.copy(), _DART_FINANCIAL_RAW_SCHEMA)
    out["corp_code"] = str(corp_code).strip()
    for column in ("filing_date", "period_end"):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.date
    out["collected_at"] = pd.to_datetime(out["collected_at"], errors="coerce", utc=True)
    out = out.dropna(subset=["corp_code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_key"])
    out = out.drop_duplicates(
        subset=["corp_code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_key"],
        keep="last",
    )
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_dart_financials_raw(out, corp_code=corp_code, market='kr')
    with _lock:
        con = _get_con()
        con.execute("DELETE FROM dart_financials_raw WHERE corp_code = ?", [str(corp_code).strip()])
    return _write_table("dart_financials_raw", out, _DART_FINANCIAL_RAW_SCHEMA)


def upsert_financials(df: pd.DataFrame, ticker: str, market: str = "kr") -> int:
    if df is None or df.empty:
        return 0
    out = df.copy()
    drop_cols = [column for column in out.columns if column in _REMOVED_KR_FINANCIAL_COLUMNS]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.upsert_financials(out, ticker=ticker, market=market)
    out = _ensure_columns(out, _FIN_SCHEMA)
    out["ticker"] = normalize_kr_ticker(ticker)
    out["market"] = str(market).strip().lower()
    for column in ("StatementDate", "PeriodEnd", "PeriodStart", "FilingDate", "AvailableDate"):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.date
    out["AcceptedAt"] = pd.to_datetime(out["AcceptedAt"], errors="coerce", utc=True)
    out = out.dropna(subset=["ticker", "market", "PeriodEnd"])
    with _lock:
        con = _get_con()
        con.execute("DELETE FROM financials_quarterly WHERE ticker = ? AND market = ?", [normalize_kr_ticker(ticker), str(market).strip().lower()])
    return _write_table("financials_quarterly", out, _FIN_SCHEMA)


def get_latest_price_date(ticker: str, market: str = "kr") -> pd.Timestamp | None:
    if _use_parquet():
        from market_data import parquet_store
        d = parquet_store.get_latest_price_date(ticker, market)
        return pd.Timestamp(d) if d else None
    try:
        with _lock:
            con = _get_prices_con()
            row = con.execute(
                "SELECT MAX(date) FROM prices WHERE ticker = ? AND market = ?",
                [normalize_kr_ticker(ticker), str(market).strip().lower()],
            ).fetchone()
        if row and row[0] is not None:
            return pd.Timestamp(row[0])
        return None
    except Exception:
        return None


def save_checkpoint(ticker: str, market: str, payload: dict[str, object]) -> None:
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
                normalize_kr_ticker(ticker),
                str(market).strip().lower(),
                str(payload.get("run_id", "")),
                str(payload.get("completed_at", "")),
                payload.get("fresh_days"),
                json.dumps(payload, ensure_ascii=False, default=str),
            ],
        )


def get_checkpoint(ticker: str, market: str) -> dict[str, object] | None:
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.get_checkpoint(ticker, market)
    try:
        with _lock:
            con = _get_con()
            row = con.execute(
                "SELECT payload FROM ingest_checkpoints WHERE ticker = ? AND market = ? LIMIT 1",
                [normalize_kr_ticker(ticker), str(market).strip().lower()],
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
    except Exception:
        return None


_CAPACITY_SCHEMA: list[str] = [
    "ticker", "market", "period_end", "available_date", "section",
    "product_name", "unit", "value", "currency", "accession",
    "form_type", "filing_date", "source", "collected_at",
]


def upsert_dart_capacity(df: pd.DataFrame, ticker: str, market: str = "kr") -> int:
    """Write capacity/production/utilization rows to capacity_production_quarterly."""
    if df is None or df.empty:
        return 0
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store._upsert_by_ticker(df, ticker=ticker, market=market, table='dart_capacity')
    out = _ensure_columns(df.copy(), _CAPACITY_SCHEMA)
    out["ticker"] = normalize_kr_ticker(ticker)
    out["market"] = str(market).strip().lower()
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce").dt.date
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce").dt.date
    out["filing_date"] = pd.to_datetime(out["filing_date"], errors="coerce").dt.date
    out["collected_at"] = pd.to_datetime(out["collected_at"], errors="coerce", utc=True)
    out = out.dropna(subset=["ticker", "market", "period_end", "section", "product_name"])
    out = out.drop_duplicates(
        subset=["ticker", "market", "period_end", "section", "product_name"],
        keep="last",
    )
    with _lock:
        con = _get_con()
        con.execute(
            "DELETE FROM capacity_production_quarterly WHERE ticker = ? AND market = ?",
            [normalize_kr_ticker(ticker), str(market).strip().lower()],
        )
    return _write_table("capacity_production_quarterly", out, _CAPACITY_SCHEMA)


def mark_capacity_ingest_done(ticker: str, market: str, rows_written: int) -> None:
    """Record that capacity ingest was attempted for this ticker."""
    if _use_parquet():
        return
    tk = normalize_kr_ticker(ticker)
    mk = str(market).strip().lower()
    with _lock:
        con = _get_con()
        con.execute(
            """
            INSERT OR REPLACE INTO capacity_ingest_tickers
                (ticker, market, attempted_at, rows_written)
            VALUES (?, ?, now(), ?)
            """,
            [tk, mk, int(rows_written)],
        )
        con.commit()


def get_fresh_tickers(market: str, fresh_days: int | None) -> set[str]:
    if _use_parquet():
        from market_data import parquet_store
        return parquet_store.get_fresh_tickers(market, fresh_days)
    try:
        with _lock:
            con = _get_con()
            rows = con.execute(
                "SELECT ticker, completed_at FROM ingest_checkpoints WHERE market = ?",
                [str(market).strip().lower()],
            ).fetchall()
        if fresh_days is None:
            return {str(row[0]) for row in rows if row and row[0]}
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=fresh_days)
        out: set[str] = set()
        for ticker, completed_at in rows:
            completed = pd.to_datetime(completed_at, errors="coerce", utc=True)
            if pd.notna(completed) and completed >= cutoff:
                out.add(str(ticker))
        return out
    except Exception:
        return set()


def close() -> None:
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
