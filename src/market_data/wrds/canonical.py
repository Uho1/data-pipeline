from __future__ import annotations

import logging
from typing import Iterable

from market_data.wrds.duckdb_io import DuckDBManager

CANONICAL_BUILD_ORDER: tuple[str, ...] = (
    "security_link_history",
    "entity_master",
    "prices_daily_canonical",
    "financials_quarterly_canonical",
    "financials_annual_canonical",
    "segments_historical_canonical",
    "universe_membership_history",
)


def _security_link_history_sql() -> str:
    return """
        SELECT
            CONCAT(
                COALESCE(c.gvkey, ''),
                '|',
                COALESCE(CAST(c.lpermno AS VARCHAR), ''),
                '|',
                COALESCE(CAST(GREATEST(COALESCE(c.linkdt, DATE '1900-01-01'), COALESCE(s.namedt, DATE '1900-01-01')) AS VARCHAR), ''),
                '|',
                COALESCE(c.liid, '')
            ) AS link_id,
            c.gvkey AS gvkey,
            c.liid AS iid,
            c.lpermno AS permno,
            c.lpermco AS permco,
            s.ticker AS ticker,
            COALESCE(s.ncusip, s.cusip) AS cusip,
            comp.cik AS cik,
            s.exchcd AS exchange_code,
            s.shrcd AS share_code,
            s.company_name AS security_name,
            c.linktype AS link_type,
            c.linkprim AS link_primary,
            GREATEST(COALESCE(c.linkdt, DATE '1900-01-01'), COALESCE(s.namedt, DATE '1900-01-01')) AS effective_from,
            CASE
                WHEN LEAST(COALESCE(c.linkenddt, DATE '9999-12-31'), COALESCE(s.nameenddt, DATE '9999-12-31')) = DATE '9999-12-31'
                    THEN NULL
                ELSE LEAST(COALESCE(c.linkenddt, DATE '9999-12-31'), COALESCE(s.nameenddt, DATE '9999-12-31'))
            END AS effective_to,
            'wrds_ccm_link + wrds_security_master + wrds_company_master' AS source_summary,
            CURRENT_TIMESTAMP AS refreshed_at
        FROM (
            SELECT
                c.*,
                s.namedt AS security_namedt,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        c.gvkey,
                        c.lpermno,
                        GREATEST(COALESCE(c.linkdt, DATE '1900-01-01'), COALESCE(s.namedt, DATE '1900-01-01'))
                    ORDER BY
                        CASE WHEN c.linkprim = 'P' THEN 0 ELSE 1 END,
                        CASE WHEN c.linktype IN ('LU', 'LC', 'LS') THEN 0 ELSE 1 END,
                        COALESCE(s.namedt, DATE '1900-01-01') DESC
                ) AS rn
            FROM wrds_ccm_link c
            LEFT JOIN wrds_security_master s
                ON s.permno = c.lpermno
               AND COALESCE(s.nameenddt, DATE '9999-12-31') >= COALESCE(c.linkdt, DATE '1900-01-01')
               AND COALESCE(s.namedt, DATE '1900-01-01') <= COALESCE(c.linkenddt, DATE '9999-12-31')
            WHERE c.lpermno IS NOT NULL
        ) c
        LEFT JOIN wrds_security_master s
            ON s.permno = c.lpermno
           AND COALESCE(s.namedt, DATE '1900-01-01') = COALESCE(c.security_namedt, DATE '1900-01-01')
        LEFT JOIN wrds_company_master comp
            ON comp.gvkey = c.gvkey
        WHERE c.rn = 1
    """


def _entity_master_sql() -> str:
    return """
        WITH link_stats AS (
            SELECT
                gvkey,
                COUNT(DISTINCT permno) AS permno_count,
                COUNT(DISTINCT permco) AS permco_count,
                MIN(effective_from) AS first_seen_date,
                MAX(COALESCE(effective_to, CURRENT_DATE)) AS last_seen_date
            FROM security_link_history
            GROUP BY gvkey
        ),
        company_ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY gvkey
                    ORDER BY COALESCE(collected_at, CURRENT_TIMESTAMP) DESC
                ) AS rn
            FROM wrds_company_master
        )
        SELECT
            COALESCE(gvkey, CONCAT('CIK:', cik)) AS entity_id,
            gvkey,
            cik,
            company_name,
            tic AS current_ticker,
            cusip AS current_cusip,
            sic,
            naics,
            gsector,
            ggroup,
            gind,
            gsubind,
            fic AS incorp_country,
            state_incorp AS incorp_state,
            link_stats.first_seen_date,
            link_stats.last_seen_date,
            COALESCE(link_stats.permno_count, 0) AS permno_count,
            COALESCE(link_stats.permco_count, 0) AS permco_count,
            'wrds_company_master + security_link_history' AS source_summary,
            CURRENT_TIMESTAMP AS refreshed_at
        FROM company_ranked
        LEFT JOIN link_stats USING (gvkey)
        WHERE rn = 1
    """


def _prices_daily_sql() -> str:
    return """
        SELECT
            trade_date,
            trade_date AS available_date,
            permno,
            permco,
            gvkey,
            iid,
            ticker,
            cusip,
            cik,
            exchange_code,
            close_price,
            return_total,
            return_ex_dividend,
            shares_outstanding,
            market_cap,
            price_adjust_factor,
            share_adjust_factor,
            delist_code,
            delist_date,
            delist_return,
            delist_return_ex_dividend,
            delist_price,
            'wrds_crsp_daily' AS source_table,
            CURRENT_TIMESTAMP AS refreshed_at
        FROM (
            SELECT
                d.trade_date,
                d.permno,
                d.permco,
                sl.gvkey,
                sl.iid,
                COALESCE(sm.ticker, sl.ticker) AS ticker,
                COALESCE(sm.ncusip, sm.cusip, sl.cusip) AS cusip,
                sl.cik AS cik,
                COALESCE(sm.exchcd, sl.exchange_code) AS exchange_code,
                d.price_close AS close_price,
                d.return_total,
                d.return_ex_dividend,
                d.shares_outstanding_k * 1000.0 AS shares_outstanding,
                ABS(d.price_close) * d.shares_outstanding_k * 1000.0 AS market_cap,
                d.price_adjust_factor,
                d.share_adjust_factor,
                d.delist_code,
                d.delist_date,
                d.delist_return,
                d.delist_return_ex_dividend,
                d.delist_price,
                ROW_NUMBER() OVER (
                    PARTITION BY d.permno, d.trade_date
                    ORDER BY
                        CASE WHEN sl.link_primary = 'P' THEN 0 ELSE 1 END,
                        CASE WHEN sl.link_type IN ('LU', 'LC', 'LS') THEN 0 ELSE 1 END,
                        COALESCE(sl.effective_from, DATE '1900-01-01') DESC
                ) AS rn
            FROM wrds_crsp_daily d
            LEFT JOIN security_link_history sl
                ON sl.permno = d.permno
               AND d.trade_date >= COALESCE(sl.effective_from, DATE '1900-01-01')
               AND (sl.effective_to IS NULL OR d.trade_date <= sl.effective_to)
            LEFT JOIN wrds_security_master sm
                ON sm.permno = d.permno
               AND d.trade_date >= COALESCE(sm.namedt, DATE '1900-01-01')
               AND d.trade_date <= COALESCE(sm.nameenddt, DATE '9999-12-31')
        ) ranked
        WHERE rn = 1 OR rn IS NULL
    """


def _financials_quarterly_sql() -> str:
    return """
        WITH source_rows AS (
            SELECT *
            FROM (
                SELECT
                    q.*,
                    COALESCE(
                        q.fyearq,
                        TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                        EXTRACT(YEAR FROM q.datadate)::INTEGER
                    ) AS fiscal_year_resolved,
                    COALESCE(
                        q.fqtr,
                        TRY_CAST(REGEXP_EXTRACT(q.datafqtr, 'Q([1-4])', 1) AS INTEGER),
                        EXTRACT(QUARTER FROM q.datadate)::INTEGER
                    ) AS fiscal_quarter_resolved,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            q.gvkey,
                            q.datadate,
                            COALESCE(
                                q.fyearq,
                                TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                                EXTRACT(YEAR FROM q.datadate)::INTEGER
                            ),
                            COALESCE(
                                q.fqtr,
                                TRY_CAST(REGEXP_EXTRACT(q.datafqtr, 'Q([1-4])', 1) AS INTEGER),
                                EXTRACT(QUARTER FROM q.datadate)::INTEGER
                            )
                        ORDER BY
                            CASE
                                WHEN q.datafqtr = CONCAT(
                                    COALESCE(
                                        q.fyearq,
                                        TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                                        EXTRACT(YEAR FROM q.datadate)::INTEGER
                                    )::text,
                                    'Q',
                                    COALESCE(
                                        q.fqtr,
                                        TRY_CAST(REGEXP_EXTRACT(q.datafqtr, 'Q([1-4])', 1) AS INTEGER),
                                        EXTRACT(QUARTER FROM q.datadate)::INTEGER
                                    )::text
                                ) THEN 0
                                ELSE 1
                            END,
                            COALESCE(q.rdq, q.datadate) DESC,
                            COALESCE(q.datafqtr, '') DESC
                    ) AS canonical_rn
                FROM wrds_compustat_quarterly q
            ) ranked
            WHERE canonical_rn = 1
        ),
        base AS (
            SELECT
                q.*,
                CASE
                    WHEN COALESCE(
                        q.fqtr,
                        TRY_CAST(REGEXP_EXTRACT(q.datafqtr, 'Q([1-4])', 1) AS INTEGER),
                        EXTRACT(QUARTER FROM q.datadate)::INTEGER
                    ) = 1 THEN q.oancfy
                    WHEN LAG(q.oancfy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    ) IS NULL THEN NULL
                    ELSE q.oancfy - LAG(q.oancfy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    )
                END AS operating_cash_flow_q,
                CASE
                    WHEN COALESCE(
                        q.fqtr,
                        TRY_CAST(REGEXP_EXTRACT(q.datafqtr, 'Q([1-4])', 1) AS INTEGER),
                        EXTRACT(QUARTER FROM q.datadate)::INTEGER
                    ) = 1 THEN q.capxy
                    WHEN LAG(q.capxy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    ) IS NULL THEN NULL
                    ELSE q.capxy - LAG(q.capxy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    )
                END AS capital_expenditure_q
                ,
                CASE
                    WHEN COALESCE(
                        q.fqtr,
                        TRY_CAST(REGEXP_EXTRACT(q.datafqtr, 'Q([1-4])', 1) AS INTEGER),
                        EXTRACT(QUARTER FROM q.datadate)::INTEGER
                    ) = 1 THEN q.ivncfy
                    WHEN LAG(q.ivncfy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    ) IS NULL THEN NULL
                    ELSE q.ivncfy - LAG(q.ivncfy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    )
                END AS investing_cash_flow_q,
                CASE
                    WHEN COALESCE(
                        q.fqtr,
                        TRY_CAST(REGEXP_EXTRACT(q.datafqtr, 'Q([1-4])', 1) AS INTEGER),
                        EXTRACT(QUARTER FROM q.datadate)::INTEGER
                    ) = 1 THEN q.fincfy
                    WHEN LAG(q.fincfy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    ) IS NULL THEN NULL
                    ELSE q.fincfy - LAG(q.fincfy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    )
                END AS financing_cash_flow_q,
                CASE
                    WHEN COALESCE(
                        q.fqtr,
                        TRY_CAST(REGEXP_EXTRACT(q.datafqtr, 'Q([1-4])', 1) AS INTEGER),
                        EXTRACT(QUARTER FROM q.datadate)::INTEGER
                    ) = 1 THEN q.dvy
                    WHEN LAG(q.dvy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    ) IS NULL THEN NULL
                    ELSE q.dvy - LAG(q.dvy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    )
                END AS dividends_paid_q,
                CASE
                    WHEN COALESCE(
                        q.fqtr,
                        TRY_CAST(REGEXP_EXTRACT(q.datafqtr, 'Q([1-4])', 1) AS INTEGER),
                        EXTRACT(QUARTER FROM q.datadate)::INTEGER
                    ) = 1 THEN q.prstkcy
                    WHEN LAG(q.prstkcy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    ) IS NULL THEN NULL
                    ELSE q.prstkcy - LAG(q.prstkcy) OVER (
                        PARTITION BY q.gvkey, COALESCE(
                            q.fyearq,
                            TRY_CAST(REGEXP_EXTRACT(q.datafqtr, '([0-9]{4})Q[1-4]', 1) AS INTEGER),
                            EXTRACT(YEAR FROM q.datadate)::INTEGER
                        )
                        ORDER BY q.datadate
                    )
                END AS repurchases_q
            FROM source_rows q
        )
        SELECT
            gvkey,
            datadate AS period_end,
            COALESCE(rdq, datadate) AS available_date,
            fiscal_year_resolved AS fiscal_year,
            fiscal_quarter_resolved AS fiscal_quarter,
            curcdq AS currency,
            tic AS ticker,
            cusip,
            cik,
            COALESCE(revtq, saleq) AS revenue,
            cogsq AS cogs,
            CASE
                WHEN COALESCE(revtq, saleq) IS NOT NULL AND cogsq IS NOT NULL THEN COALESCE(revtq, saleq) - cogsq
                ELSE NULL
            END AS gross_profit,
            xsgaq AS sga,
            oiadpq AS operating_income,
            piq AS pretax_income,
            txtq AS tax,
            COALESCE(niq, ibq) AS net_income,
            COALESCE(ibcomq, niq, ibq) AS net_income_common,
            xintq AS interest,
            dpq AS d_and_a,
            epspxq AS eps_basic,
            epsfxq AS eps_diluted,
            cshoq AS shares_basic,
            cshfdq AS shares_diluted,
            atq AS assets,
            ltq AS liabilities,
            seqq AS equity,
            actq AS current_assets,
            lctq AS current_liabilities,
            cheq AS cash,
            COALESCE(finivstq, ivstq) AS current_fin_assets,
            COALESCE(finaoq, ivaoq) AS non_current_fin_assets,
            COALESCE(finlcoq, findlcq, dlcq) AS current_fin_liabilities,
            COALESCE(finltoq, findltq, dlttq) AS non_current_fin_liabilities,
            dlcq AS debt_short,
            dlttq AS debt_long,
            rectq AS accounts_receivable,
            invtq AS inventory,
            apq AS accounts_payable,
            drcq AS deferred_revenue,
            gdwlq AS goodwill,
            intanq AS intangibles,
            cstkq AS common_stock,
            retq AS retained_earnings,
            CASE
                WHEN aociderglq IS NULL
                 AND aociotherq IS NULL
                 AND aocipenq IS NULL
                 AND aocisecglq IS NULL THEN NULL
                ELSE COALESCE(aociderglq, 0.0)
                   + COALESCE(aociotherq, 0.0)
                   + COALESCE(aocipenq, 0.0)
                   + COALESCE(aocisecglq, 0.0)
            END AS aoci,
            amq AS amortization,
            operating_cash_flow_q AS operating_cash_flow,
            investing_cash_flow_q AS investing_cash_flow,
            financing_cash_flow_q AS financing_cash_flow,
            capital_expenditure_q AS capital_expenditure,
            COALESCE(dividends_paid_q, dvq) AS dividends,
            COALESCE(dividends_paid_q, dvq) AS dividends_paid,
            repurchases_q AS repurchases,
            prccq AS price_period_end,
            'wrds_compustat_quarterly' AS source_table,
            CURRENT_TIMESTAMP AS refreshed_at
        FROM base
    """


def _financials_annual_sql() -> str:
    return """
        SELECT
            gvkey,
            datadate AS period_end,
            datadate AS available_date,
            fyear AS fiscal_year,
            curcd AS currency,
            tic AS ticker,
            cusip,
            cik,
            COALESCE(revt, sale) AS revenue,
            cogs AS cogs,
            CASE
                WHEN COALESCE(revt, sale) IS NOT NULL AND cogs IS NOT NULL THEN COALESCE(revt, sale) - cogs
                ELSE NULL
            END AS gross_profit,
            xsga AS sga,
            oiadp AS operating_income,
            pi AS pretax_income,
            txt AS tax,
            COALESCE(ni, ib) AS net_income,
            COALESCE(ibcom, ni, ib) AS net_income_common,
            xint AS interest,
            CASE
                WHEN dp IS NOT NULL AND am IS NOT NULL THEN dp + am
                ELSE COALESCE(dp, am)
            END AS d_and_a,
            epspx AS eps_basic,
            epsfx AS eps_diluted,
            csho AS shares_outstanding,
            "at" AS assets,
            "lt" AS liabilities,
            "seq" AS equity,
            act AS current_assets,
            lct AS current_liabilities,
            che AS cash,
            COALESCE(finivst, ivst) AS current_fin_assets,
            COALESCE(finao, ivao) AS non_current_fin_assets,
            COALESCE(finlco, findlc, dlc) AS current_fin_liabilities,
            COALESCE(finlto, findlt, dltt) AS non_current_fin_liabilities,
            dlc AS debt_short,
            dltt AS debt_long,
            rect AS accounts_receivable,
            invt AS inventory,
            ap AS accounts_payable,
            drc AS deferred_revenue,
            gdwl AS goodwill,
            intan AS intangibles,
            cstk AS common_stock,
            ret AS retained_earnings,
            CASE
                WHEN aocidergl IS NULL
                 AND aociother IS NULL
                 AND aocipen IS NULL
                 AND aocisecgl IS NULL THEN NULL
                ELSE COALESCE(aocidergl, 0.0)
                   + COALESCE(aociother, 0.0)
                   + COALESCE(aocipen, 0.0)
                   + COALESCE(aocisecgl, 0.0)
            END AS aoci,
            am AS amortization,
            oancf AS operating_cash_flow,
            ivncf AS investing_cash_flow,
            fincf AS financing_cash_flow,
            capx AS capital_expenditure,
            COALESCE(dvc, dv) AS dividends,
            COALESCE(dvc, dv) AS dividends_paid,
            prstkc AS repurchases,
            prcc_f AS price_period_end,
            'wrds_compustat_annual' AS source_table,
            CURRENT_TIMESTAMP AS refreshed_at
        FROM wrds_compustat_annual
    """


def _universe_membership_sql() -> str:
    return """
        SELECT
            membership_type,
            membership_code,
            membership_name,
            member_key,
            permno,
            permco,
            gvkey,
            iid,
            ticker,
            cusip,
            cik,
            effective_from,
            effective_to,
            source_kind,
            source_summary,
            CURRENT_TIMESTAMP AS refreshed_at
        FROM (
            SELECT
                src.membership_type,
                src.membership_code,
                src.membership_name,
                src.member_key,
                COALESCE(src.permno, sl.permno) AS permno,
                COALESCE(src.permco, sl.permco) AS permco,
                COALESCE(src.gvkey, sl.gvkey) AS gvkey,
                COALESCE(src.iid, sl.iid) AS iid,
                COALESCE(src.ticker, sl.ticker) AS ticker,
                COALESCE(src.cusip, sl.cusip) AS cusip,
                COALESCE(src.cik, sl.cik) AS cik,
                src.effective_from,
                src.effective_to,
                src.source_kind,
                CONCAT(
                    COALESCE(src.source_relation, ''),
                    ' | ',
                    COALESCE(src.source_query_name, ''),
                    ' | ',
                    COALESCE(src.assumptions, '')
                ) AS source_summary,
                ROW_NUMBER() OVER (
                    PARTITION BY src.membership_type, src.membership_code, src.member_key, src.effective_from
                    ORDER BY
                        CASE WHEN sl.link_primary = 'P' THEN 0 ELSE 1 END,
                        COALESCE(sl.effective_from, DATE '1900-01-01') DESC
                ) AS rn
            FROM wrds_universe_membership_history src
            LEFT JOIN security_link_history sl
                ON src.gvkey = sl.gvkey
               AND (src.iid IS NULL OR src.iid = sl.iid)
               AND COALESCE(src.effective_from, DATE '1900-01-01') <= COALESCE(sl.effective_to, DATE '9999-12-31')
               AND COALESCE(src.effective_to, DATE '9999-12-31') >= COALESCE(sl.effective_from, DATE '1900-01-01')
        ) ranked
        WHERE rn = 1 OR rn IS NULL
    """


def _segments_historical_sql() -> str:
    return """
        SELECT
            s.segment_row_id,
            s.gvkey,
            COALESCE(comp.tic, ent.current_ticker) AS ticker,
            COALESCE(comp.cusip, ent.current_cusip) AS cusip,
            COALESCE(comp.cik, ent.cik) AS cik,
            s.datadate AS period_end,
            COALESCE(s.srcdate, s.datadate) AS available_date,
            LOWER(COALESCE(NULLIF(TRIM(s.segment_type), ''), 'other')) AS segment_type,
            CONCAT(
                LOWER(COALESCE(NULLIF(TRIM(s.segment_type), ''), 'other')),
                '|',
                COALESCE(NULLIF(TRIM(s.sid), ''), NULLIF(TRIM(s.segment_name), ''), 'unknown')
            ) AS segment_key,
            s.segment_name,
            s.segment_name_secondary,
            s.curcds AS currency,
            s.revenue,
            s.sales,
            s.operating_income,
            s.operating_profit_before_dep,
            s.assets,
            s.capex,
            s.r_and_d,
            s.goodwill,
            s.employees,
            'wrds_compustat_segments_historical' AS source_table,
            CURRENT_TIMESTAMP AS refreshed_at
        FROM wrds_compustat_segments_historical s
        LEFT JOIN wrds_company_master comp
          ON comp.gvkey = s.gvkey
        LEFT JOIN entity_master ent
          ON ent.gvkey = s.gvkey
    """


CANONICAL_SQL: dict[str, str] = {
    "security_link_history": _security_link_history_sql(),
    "entity_master": _entity_master_sql(),
    "prices_daily_canonical": _prices_daily_sql(),
    "financials_quarterly_canonical": _financials_quarterly_sql(),
    "financials_annual_canonical": _financials_annual_sql(),
    "segments_historical_canonical": _segments_historical_sql(),
    "universe_membership_history": _universe_membership_sql(),
}


def build_canonical_tables(
    duckdb_manager: DuckDBManager,
    logger: logging.Logger,
    *,
    selected_tables: Iterable[str] | None = None,
) -> dict[str, dict[str, int | str]]:
    """Rebuild canonical WRDS research tables from the source layer."""

    tables = tuple(selected_tables or CANONICAL_BUILD_ORDER)
    summary: dict[str, dict[str, int | str]] = {}
    for table in tables:
        sql = CANONICAL_SQL[table]
        row_count = duckdb_manager.overwrite_with_query(table, sql)
        summary[table] = {"row_count": row_count}
        logger.info("Canonical table refreshed: %s rows=%s", table, row_count)
    return summary
