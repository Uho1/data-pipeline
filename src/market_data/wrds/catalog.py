from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import pandas as pd

from market_data.wrds.config import WRDSSettings


@dataclass(frozen=True)
class ChunkWindow:
    """Inclusive chunk window used for restartable historical pulls."""

    start: date | None
    end: date | None

    @property
    def key(self) -> str:
        if self.start is None and self.end is None:
            return "full_refresh"
        if self.start is None:
            return f"until_{self.end.isoformat()}"
        if self.end is None:
            return f"from_{self.start.isoformat()}"
        return f"{self.start.isoformat()}__{self.end.isoformat()}"


QueryBuilder = Callable[[WRDSSettings, ChunkWindow | None], tuple[str, dict[str, Any]]]


@dataclass(frozen=True)
class DatasetSpec:
    """Source dataset metadata for the WRDS ingestion service."""

    name: str
    target_table: str
    key_columns: tuple[str, ...]
    query_builder: QueryBuilder | None
    chunk_months: int | None
    date_column: str | None
    summary_distinct_columns: tuple[str, ...]
    full_refresh_only: bool = False


DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "all": (
        "company_master",
        "security_master",
        "ccm_link",
        "crsp_daily",
        "compustat_quarterly",
        "compustat_annual",
        "universe_membership_history",
    ),
    "lake_all": (
        "company_master",
        "security_master",
        "ccm_link",
        "compustat_quarterly",
        "compustat_annual",
        "compustat_segments_historical",
        "compustat_quarterly_variant_metrics",
        "ibes_actuals_epsus",
        "ibes_summary_epsus",
        "source_access_registry",
        "universe_membership_history",
    ),
    "links": ("company_master", "security_master", "ccm_link", "universe_membership_history"),
}


def _override_or_default(settings: WRDSSettings, dataset_name: str, sql: str) -> str:
    return settings.sql_override(dataset_name) or sql


def _quoted_literal_list(values: tuple[str, ...]) -> str:
    if not values:
        return "('')"
    payload = ", ".join("'" + item.replace("'", "''") + "'" for item in values)
    return f"({payload})"


def build_crsp_daily_sql(settings: WRDSSettings, window: ChunkWindow | None) -> tuple[str, dict[str, Any]]:
    relations = settings.relations
    sql = _override_or_default(
        settings,
        "crsp_daily",
        f"""
        SELECT
            d.permno::bigint AS permno,
            d.permco::bigint AS permco,
            d.date::date AS trade_date,
            d.bidlo::double precision AS bid_low,
            d.askhi::double precision AS ask_high,
            ABS(d.prc)::double precision AS price_close,
            d.vol::double precision AS volume,
            d.ret::double precision AS return_total,
            d.retx::double precision AS return_ex_dividend,
            d.shrout::double precision AS shares_outstanding_k,
            d.cfacpr::double precision AS price_adjust_factor,
            d.cfacshr::double precision AS share_adjust_factor,
            d.numtrd::bigint AS trade_count,
            e.dlstcd::integer AS delist_code,
            e.dlstdt::date AS delist_date,
            e.dlret::double precision AS delist_return,
            e.dlretx::double precision AS delist_return_ex_dividend,
            e.dlprc::double precision AS delist_price
        FROM {relations.crsp_daily} d
        LEFT JOIN {relations.crsp_delist} e
          ON d.permno = e.permno
         AND d.date = e.dlstdt
        WHERE d.date BETWEEN %(start)s AND %(end)s
        ORDER BY d.date, d.permno
        """,
    )
    return sql, {"start": window.start.isoformat(), "end": window.end.isoformat()}


def build_compustat_quarterly_sql(settings: WRDSSettings, window: ChunkWindow | None) -> tuple[str, dict[str, Any]]:
    relation = settings.relations.compustat_quarterly
    sql = _override_or_default(
        settings,
        "compustat_quarterly",
        f"""
        WITH iq_dedup AS (
            SELECT
                gvkey,
                datadate,
                indfmt,
                consol,
                popsrc,
                datafmt,
                MAX(amq)::double precision AS amq
            FROM comp.co_ifndq
            GROUP BY 1, 2, 3, 4, 5, 6
        )
        SELECT
            q.gvkey::text AS gvkey,
            q.datadate::date AS datadate,
            q.fyearq::integer AS fyearq,
            q.fqtr::integer AS fqtr,
            COALESCE(
                q.datafqtr::text,
                CASE
                    WHEN q.fyearq IS NOT NULL AND q.fqtr IS NOT NULL
                    THEN CONCAT(q.fyearq::text, 'Q', q.fqtr::text)
                    WHEN q.datadate IS NOT NULL
                    THEN CONCAT('D', REPLACE(q.datadate::date::text, '-', ''))
                    ELSE NULL
                END
            ) AS datafqtr,
            q.rdq::date AS rdq,
            q.fyr::integer AS fyr,
            q.indfmt::text AS indfmt,
            q.consol::text AS consol,
            q.popsrc::text AS popsrc,
            q.datafmt::text AS datafmt,
            q.curcdq::text AS curcdq,
            q.tic::text AS tic,
            q.cusip::text AS cusip,
            q.cik::text AS cik,
            q.saleq::double precision AS saleq,
            q.revtq::double precision AS revtq,
            q.cogsq::double precision AS cogsq,
            q.xsgaq::double precision AS xsgaq,
            q.oiadpq::double precision AS oiadpq,
            q.ibq::double precision AS ibq,
            q.niq::double precision AS niq,
            q.ibcomq::double precision AS ibcomq,
            q.piq::double precision AS piq,
            q.txtq::double precision AS txtq,
            q.xintq::double precision AS xintq,
            q.dpq::double precision AS dpq,
            q.xrdq::double precision AS xrdq,
            q.epspxq::double precision AS epspxq,
            q.epsfxq::double precision AS epsfxq,
            q.cshoq::double precision AS cshoq,
            q.cshfdq::double precision AS cshfdq,
            q.stkcoq::double precision AS stkcoq,
            q.ucapsq::double precision AS ucapsq,
            q.atq::double precision AS atq,
            q.ltq::double precision AS ltq,
            q.seqq::double precision AS seqq,
            q.actq::double precision AS actq,
            q.lctq::double precision AS lctq,
            q.cheq::double precision AS cheq,
            q.finivstq::double precision AS finivstq,
            q.ivstq::double precision AS ivstq,
            q.finaoq::double precision AS finaoq,
            q.ivaoq::double precision AS ivaoq,
            q.finlcoq::double precision AS finlcoq,
            q.findlcq::double precision AS findlcq,
            q.finltoq::double precision AS finltoq,
            q.findltq::double precision AS findltq,
            q.dlcq::double precision AS dlcq,
            q.dlttq::double precision AS dlttq,
            q.rectq::double precision AS rectq,
            q.invtq::double precision AS invtq,
            q.apq::double precision AS apq,
            q.drcq::double precision AS drcq,
            q.gdwlq::double precision AS gdwlq,
            q.intanq::double precision AS intanq,
            iq.amq::double precision AS amq,
            q.cstkq::double precision AS cstkq,
            q.retq::double precision AS retq,
            q.aociderglq::double precision AS aociderglq,
            q.aociotherq::double precision AS aociotherq,
            q.aocipenq::double precision AS aocipenq,
            q.aocisecglq::double precision AS aocisecglq,
            q.oancfy::double precision AS oancfy,
            q.capxy::double precision AS capxy,
            NULL::double precision AS dvq,
            q.dvy::double precision AS dvy,
            q.ivncfy::double precision AS ivncfy,
            q.fincfy::double precision AS fincfy,
            q.prstkcy::double precision AS prstkcy,
            q.prccq::double precision AS prccq
        FROM {relation} q
        LEFT JOIN iq_dedup iq
          ON iq.gvkey = q.gvkey
         AND iq.datadate = q.datadate
         AND iq.indfmt = q.indfmt
         AND iq.consol = q.consol
         AND iq.popsrc = q.popsrc
         AND iq.datafmt = q.datafmt
        WHERE q.datadate BETWEEN %(start)s AND %(end)s
          AND q.indfmt = 'INDL'
          AND q.datafmt = 'STD'
          AND q.consol = 'C'
          AND q.popsrc = 'D'
        ORDER BY q.datadate, q.gvkey
        """,
    )
    return sql, {"start": window.start.isoformat(), "end": window.end.isoformat()}


def build_compustat_annual_sql(settings: WRDSSettings, window: ChunkWindow | None) -> tuple[str, dict[str, Any]]:
    relation = settings.relations.compustat_annual
    sql = _override_or_default(
        settings,
        "compustat_annual",
        f"""
        SELECT
            gvkey::text AS gvkey,
            datadate::date AS datadate,
            fyear::integer AS fyear,
            fyr::integer AS fyr,
            indfmt::text AS indfmt,
            consol::text AS consol,
            popsrc::text AS popsrc,
            datafmt::text AS datafmt,
            curcd::text AS curcd,
            tic::text AS tic,
            cusip::text AS cusip,
            cik::text AS cik,
            sale::double precision AS sale,
            revt::double precision AS revt,
            cogs::double precision AS cogs,
            xsga::double precision AS xsga,
            oiadp::double precision AS oiadp,
            ib::double precision AS ib,
            ni::double precision AS ni,
            ibcom::double precision AS ibcom,
            pi::double precision AS pi,
            txt::double precision AS txt,
            xint::double precision AS xint,
            dp::double precision AS dp,
            am::double precision AS am,
            xrd::double precision AS xrd,
            epspx::double precision AS epspx,
            epsfx::double precision AS epsfx,
            stkco::double precision AS stkco,
            ucaps::double precision AS ucaps,
            csho::double precision AS csho,
            at::double precision AS at,
            lt::double precision AS lt,
            seq::double precision AS seq,
            act::double precision AS act,
            lct::double precision AS lct,
            che::double precision AS che,
            finivst::double precision AS finivst,
            ivst::double precision AS ivst,
            finao::double precision AS finao,
            ivao::double precision AS ivao,
            finlco::double precision AS finlco,
            findlc::double precision AS findlc,
            finlto::double precision AS finlto,
            findlt::double precision AS findlt,
            dlc::double precision AS dlc,
            dltt::double precision AS dltt,
            rect::double precision AS rect,
            invt::double precision AS invt,
            ap::double precision AS ap,
            drc::double precision AS drc,
            gdwl::double precision AS gdwl,
            intan::double precision AS intan,
            cstk::double precision AS cstk,
            ret::double precision AS ret,
            aocidergl::double precision AS aocidergl,
            aociother::double precision AS aociother,
            aocipen::double precision AS aocipen,
            aocisecgl::double precision AS aocisecgl,
            oancf::double precision AS oancf,
            capx::double precision AS capx,
            dv::double precision AS dv,
            dvc::double precision AS dvc,
            ivncf::double precision AS ivncf,
            fincf::double precision AS fincf,
            prstkc::double precision AS prstkc,
            prcc_f::double precision AS prcc_f
        FROM {relation}
        WHERE datadate BETWEEN %(start)s AND %(end)s
          AND indfmt = 'INDL'
          AND datafmt = 'STD'
          AND consol = 'C'
          AND popsrc = 'D'
        ORDER BY datadate, gvkey
        """,
    )
    return sql, {"start": window.start.isoformat(), "end": window.end.isoformat()}


def build_ccm_link_sql(settings: WRDSSettings, window: ChunkWindow | None) -> tuple[str, dict[str, Any]]:
    relation = settings.relations.ccm_link
    sql = _override_or_default(
        settings,
        "ccm_link",
        f"""
        SELECT
            gvkey::text AS gvkey,
            liid::text AS liid,
            lpermno::bigint AS lpermno,
            lpermco::bigint AS lpermco,
            linktype::text AS linktype,
            linkprim::text AS linkprim,
            COALESCE(linkdt, DATE '1900-01-01')::date AS linkdt,
            linkenddt::date AS linkenddt,
            usedflag::integer AS usedflag
        FROM {relation}
        ORDER BY gvkey, lpermno, linkdt
        """,
    )
    return sql, {}


def build_security_master_sql(settings: WRDSSettings, window: ChunkWindow | None) -> tuple[str, dict[str, Any]]:
    relation = settings.relations.security_master
    sql = _override_or_default(
        settings,
        "security_master",
        f"""
        SELECT
            permno::bigint AS permno,
            permco::bigint AS permco,
            namedt::date AS namedt,
            nameenddt::date AS nameenddt,
            ticker::text AS ticker,
            comnam::text AS company_name,
            cusip::text AS cusip,
            ncusip::text AS ncusip,
            shrcls::text AS shrcls,
            exchcd::integer AS exchcd,
            shrcd::integer AS shrcd,
            siccd::integer AS siccd
        FROM {relation}
        ORDER BY permno, namedt
        """,
    )
    return sql, {}


def build_company_master_sql(settings: WRDSSettings, window: ChunkWindow | None) -> tuple[str, dict[str, Any]]:
    relation = settings.relations.company_master
    sql = _override_or_default(
        settings,
        "company_master",
        f"""
        WITH names_ranked AS (
            SELECT
                n.*,
                ROW_NUMBER() OVER (
                    PARTITION BY n.gvkey
                    ORDER BY
                        CASE WHEN n.year2 IS NULL THEN 0 ELSE 1 END,
                        COALESCE(n.year2, 9999) DESC,
                        COALESCE(n.year1, 0) DESC,
                        COALESCE(n.tic, '') DESC
                ) AS rn
            FROM comp.names n
        )
        SELECT
            c.gvkey::text AS gvkey,
            COALESCE(n.conm, c.conm)::text AS company_name,
            n.tic::text AS tic,
            n.cusip::text AS cusip,
            COALESCE(n.cik, c.cik)::text AS cik,
            COALESCE(n.sic, c.sic)::text AS sic,
            COALESCE(n.naics, c.naics)::text AS naics,
            c.fic::text AS fic,
            COALESCE(c.incorp, c.state)::text AS state_incorp,
            c.loc::text AS location_country,
            c.gsector::text AS gsector,
            c.ggroup::text AS ggroup,
            COALESCE(n.gind, c.gind)::text AS gind,
            COALESCE(n.gsubind, c.gsubind)::text AS gsubind
        FROM {relation} c
        LEFT JOIN names_ranked n
          ON n.gvkey = c.gvkey
         AND n.rn = 1
        ORDER BY c.gvkey
        """,
    )
    return sql, {}


def build_compustat_segments_historical_sql(
    settings: WRDSSettings,
    window: ChunkWindow | None,
) -> tuple[str, dict[str, Any]]:
    merged = settings.relations.compustat_segments_merged
    seg_geo = settings.relations.compustat_segments_geo
    seg_product = settings.relations.compustat_segments_product
    seg_customer = settings.relations.compustat_segments_customer
    sql = _override_or_default(
        settings,
        "compustat_segments_historical",
        f"""
        WITH geo_dim AS (
            SELECT
                gvkey,
                datadate,
                sid,
                stype,
                MAX(gareag)::text AS gareag,
                MAX(gareat)::text AS gareat
            FROM {seg_geo}
            GROUP BY 1, 2, 3, 4
        ),
        product_dim AS (
            SELECT
                gvkey,
                datadate,
                sid,
                stype,
                MAX(naicsp)::text AS naicsp,
                MAX(pnms)::text AS pnms
            FROM {seg_product}
            GROUP BY 1, 2, 3, 4
        ),
        customer_dim AS (
            SELECT
                gvkey,
                datadate,
                sid,
                stype,
                MAX(cnms)::text AS cnms,
                MAX(ctype)::text AS ctype,
                MAX(gareat)::text AS gareat
            FROM {seg_customer}
            GROUP BY 1, 2, 3, 4
        )
        SELECT
            CONCAT(
                m.gvkey::text, '|',
                m.datadate::date::text, '|',
                COALESCE(m.stype::text, ''), '|',
                COALESCE(m.sid::text, ''), '|',
                COALESCE(m.srcdate::date::text, '')
            ) AS segment_row_id,
            m.gvkey::text AS gvkey,
            m.datadate::date AS datadate,
            m.srcdate::date AS srcdate,
            m.stype::text AS stype,
            m.sid::text AS sid,
            CASE
                WHEN g.sid IS NOT NULL THEN 'geography'
                WHEN p.sid IS NOT NULL THEN 'product'
                WHEN c.sid IS NOT NULL THEN 'customer'
                ELSE COALESCE(NULLIF(TRIM(m.stype::text), ''), 'other')
            END AS segment_type,
            COALESCE(
                NULLIF(TRIM(g.gareat::text), ''),
                NULLIF(TRIM(g.gareag::text), ''),
                NULLIF(TRIM(p.pnms::text), ''),
                NULLIF(TRIM(c.cnms::text), ''),
                m.sid::text
            ) AS segment_name,
            COALESCE(
                NULLIF(TRIM(p.naicsp::text), ''),
                NULLIF(TRIM(c.ctype::text), ''),
                NULLIF(TRIM(c.gareat::text), '')
            ) AS segment_name_secondary,
            m.curcds::text AS curcds,
            m.revts::double precision AS revenue,
            m.sales::double precision AS sales,
            COALESCE(m.oiadps, m.oibdps)::double precision AS operating_income,
            m.oibdps::double precision AS operating_profit_before_dep,
            m.atlls::double precision AS assets,
            m.capxs::double precision AS capex,
            m.rds::double precision AS r_and_d,
            m.gdwls::double precision AS goodwill,
            m.emps::double precision AS employees
        FROM {merged} m
        LEFT JOIN geo_dim g
          ON g.gvkey = m.gvkey
         AND g.datadate = m.datadate
         AND g.sid = m.sid
         AND g.stype = m.stype
        LEFT JOIN product_dim p
          ON p.gvkey = m.gvkey
         AND p.datadate = m.datadate
         AND p.sid = m.sid
         AND p.stype = m.stype
        LEFT JOIN customer_dim c
          ON c.gvkey = m.gvkey
         AND c.datadate = m.datadate
         AND c.sid = m.sid
         AND c.stype = m.stype
        WHERE m.datadate BETWEEN %(start)s AND %(end)s
        ORDER BY m.datadate, m.gvkey, m.stype, m.sid
        """,
    )
    return sql, {"start": window.start.isoformat(), "end": window.end.isoformat()}


def build_compustat_quarterly_variant_metrics_sql(
    settings: WRDSSettings,
    window: ChunkWindow | None,
) -> tuple[str, dict[str, Any]]:
    ytd = settings.relations.compustat_quarterly_ytd
    semi = settings.relations.compustat_quarterly_semi
    flags = settings.relations.compustat_quarterly_flags
    sec_values = settings.relations.compustat_security_quarterly
    sec_flags = settings.relations.compustat_security_quarterly_flags
    sql = _override_or_default(
        settings,
        "compustat_quarterly_variant_metrics",
        f"""
        SELECT
            variant_row_id,
            MAX(source_variant) AS source_variant,
            MAX(period_basis) AS period_basis,
            MAX(provenance_class) AS provenance_class,
            MAX(gvkey) AS gvkey,
            MAX(iid) AS iid,
            MAX(datadate) AS datadate,
            MAX(fiscal_year) AS fiscal_year,
            MAX(fiscal_quarter) AS fiscal_quarter,
            MAX(fyr) AS fyr,
            MAX(indfmt) AS indfmt,
            MAX(consol) AS consol,
            MAX(popsrc) AS popsrc,
            MAX(datafmt) AS datafmt,
            MAX(ticker) AS ticker,
            MAX(cusip) AS cusip,
            MAX(metric_name) AS metric_name,
            MAX(metric_value) AS metric_value,
            MAX(metric_flag) AS metric_flag
        FROM (
            SELECT
                CONCAT(
                    'co_ifndytd|', gvkey::text, '|', datadate::date::text, '|',
                    indfmt::text, '|', consol::text, '|', popsrc::text, '|', datafmt::text, '|operating_cash_flow'
                ) AS variant_row_id,
                'co_ifndytd'::text AS source_variant,
                'ytd'::text AS period_basis,
                'company_level_value'::text AS provenance_class,
                gvkey::text AS gvkey,
                NULL::text AS iid,
                datadate::date AS datadate,
                NULL::integer AS fiscal_year,
                NULL::integer AS fiscal_quarter,
                fyr::integer AS fyr,
                indfmt::text AS indfmt,
                consol::text AS consol,
                popsrc::text AS popsrc,
                datafmt::text AS datafmt,
                NULL::text AS ticker,
                NULL::text AS cusip,
                'operating_cash_flow'::text AS metric_name,
                cfoy::double precision AS metric_value,
                cfoy_dc::text AS metric_flag
            FROM {ytd}
            WHERE datadate BETWEEN %(start)s AND %(end)s
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND consol = 'C'
              AND popsrc = 'D'

            UNION ALL

            SELECT
                CONCAT(
                    'co_ifndytd|', gvkey::text, '|', datadate::date::text, '|',
                    indfmt::text, '|', consol::text, '|', popsrc::text, '|', datafmt::text, '|capital_expenditure'
                ) AS variant_row_id,
                'co_ifndytd'::text AS source_variant,
                'ytd'::text AS period_basis,
                'company_level_value'::text AS provenance_class,
                gvkey::text AS gvkey,
                NULL::text AS iid,
                datadate::date AS datadate,
                NULL::integer AS fiscal_year,
                NULL::integer AS fiscal_quarter,
                fyr::integer AS fyr,
                indfmt::text AS indfmt,
                consol::text AS consol,
                popsrc::text AS popsrc,
                datafmt::text AS datafmt,
                NULL::text AS ticker,
                NULL::text AS cusip,
                'capital_expenditure'::text AS metric_name,
                capxy::double precision AS metric_value,
                capxy_dc::text AS metric_flag
            FROM {ytd}
            WHERE datadate BETWEEN %(start)s AND %(end)s
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND consol = 'C'
              AND popsrc = 'D'

            UNION ALL

            SELECT
                CONCAT(
                    'co_ifndsa|', gvkey::text, '|', datadate::date::text, '|',
                    indfmt::text, '|', consol::text, '|', popsrc::text, '|', datafmt::text, '|operating_cash_flow'
                ) AS variant_row_id,
                'co_ifndsa'::text AS source_variant,
                'semi_annual'::text AS period_basis,
                'company_level_value'::text AS provenance_class,
                gvkey::text AS gvkey,
                NULL::text AS iid,
                datadate::date AS datadate,
                NULL::integer AS fiscal_year,
                NULL::integer AS fiscal_quarter,
                fyr::integer AS fyr,
                indfmt::text AS indfmt,
                consol::text AS consol,
                popsrc::text AS popsrc,
                datafmt::text AS datafmt,
                NULL::text AS ticker,
                NULL::text AS cusip,
                'operating_cash_flow'::text AS metric_name,
                cfosa::double precision AS metric_value,
                cfosa_dc::text AS metric_flag
            FROM {semi}
            WHERE datadate BETWEEN %(start)s AND %(end)s
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND consol = 'C'
              AND popsrc = 'D'

            UNION ALL

            SELECT
                CONCAT(
                    'co_ifndsa|', gvkey::text, '|', datadate::date::text, '|',
                    indfmt::text, '|', consol::text, '|', popsrc::text, '|', datafmt::text, '|cogs'
                ) AS variant_row_id,
                'co_ifndsa'::text AS source_variant,
                'semi_annual'::text AS period_basis,
                'company_level_value'::text AS provenance_class,
                gvkey::text AS gvkey,
                NULL::text AS iid,
                datadate::date AS datadate,
                NULL::integer AS fiscal_year,
                NULL::integer AS fiscal_quarter,
                fyr::integer AS fyr,
                indfmt::text AS indfmt,
                consol::text AS consol,
                popsrc::text AS popsrc,
                datafmt::text AS datafmt,
                NULL::text AS ticker,
                NULL::text AS cusip,
                'cogs'::text AS metric_name,
                cogssa::double precision AS metric_value,
                cogssa_dc::text AS metric_flag
            FROM {semi}
            WHERE datadate BETWEEN %(start)s AND %(end)s
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND consol = 'C'
              AND popsrc = 'D'

            UNION ALL

            SELECT
                CONCAT(
                    'co_ifntq|', gvkey::text, '|', datadate::date::text, '|',
                    indfmt::text, '|', consol::text, '|', popsrc::text, '|', datafmt::text, '|operating_cash_flow'
                ) AS variant_row_id,
                'co_ifntq'::text AS source_variant,
                'quarter_flag'::text AS period_basis,
                'company_level_flag'::text AS provenance_class,
                gvkey::text AS gvkey,
                NULL::text AS iid,
                datadate::date AS datadate,
                NULL::integer AS fiscal_year,
                NULL::integer AS fiscal_quarter,
                fyr::integer AS fyr,
                indfmt::text AS indfmt,
                consol::text AS consol,
                popsrc::text AS popsrc,
                datafmt::text AS datafmt,
                NULL::text AS ticker,
                NULL::text AS cusip,
                'operating_cash_flow'::text AS metric_name,
                NULL::double precision AS metric_value,
                cfoq_fn1::text AS metric_flag
            FROM {flags}
            WHERE datadate BETWEEN %(start)s AND %(end)s
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND consol = 'C'
              AND popsrc = 'D'

            UNION ALL

            SELECT
                CONCAT(
                    'co_ifntq|', gvkey::text, '|', datadate::date::text, '|',
                    indfmt::text, '|', consol::text, '|', popsrc::text, '|', datafmt::text, '|cash'
                ) AS variant_row_id,
                'co_ifntq'::text AS source_variant,
                'quarter_flag'::text AS period_basis,
                'company_level_flag'::text AS provenance_class,
                gvkey::text AS gvkey,
                NULL::text AS iid,
                datadate::date AS datadate,
                NULL::integer AS fiscal_year,
                NULL::integer AS fiscal_quarter,
                fyr::integer AS fyr,
                indfmt::text AS indfmt,
                consol::text AS consol,
                popsrc::text AS popsrc,
                datafmt::text AS datafmt,
                NULL::text AS ticker,
                NULL::text AS cusip,
                'cash'::text AS metric_name,
                NULL::double precision AS metric_value,
                cheq_fn1::text AS metric_flag
            FROM {flags}
            WHERE datadate BETWEEN %(start)s AND %(end)s
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND consol = 'C'
              AND popsrc = 'D'

            UNION ALL

            SELECT
                CONCAT(
                    'sec_ifnd|', gvkey::text, '|', COALESCE(iid::text, ''), '|', datadate::date::text, '|',
                    indfmt::text, '|', consol::text, '|', popsrc::text, '|', datafmt::text, '|net_income'
                ) AS variant_row_id,
                'sec_ifnd'::text AS source_variant,
                'security_quarter'::text AS period_basis,
                'security_issue_value'::text AS provenance_class,
                gvkey::text AS gvkey,
                iid::text AS iid,
                datadate::date AS datadate,
                NULL::integer AS fiscal_year,
                NULL::integer AS fiscal_quarter,
                fyr::integer AS fyr,
                indfmt::text AS indfmt,
                consol::text AS consol,
                popsrc::text AS popsrc,
                datafmt::text AS datafmt,
                NULL::text AS ticker,
                NULL::text AS cusip,
                'net_income'::text AS metric_name,
                COALESCE(niq, nincq)::double precision AS metric_value,
                COALESCE(niq_dc, nincq_dc)::text AS metric_flag
            FROM {sec_values}
            WHERE datadate BETWEEN %(start)s AND %(end)s
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND consol = 'C'
              AND popsrc = 'D'

            UNION ALL

            SELECT
                CONCAT(
                    'sec_ifnt|', gvkey::text, '|', COALESCE(iid::text, ''), '|', datadate::date::text, '|',
                    indfmt::text, '|', consol::text, '|', popsrc::text, '|', datafmt::text, '|net_income'
                ) AS variant_row_id,
                'sec_ifnt'::text AS source_variant,
                'security_quarter_flag'::text AS period_basis,
                'security_issue_flag'::text AS provenance_class,
                gvkey::text AS gvkey,
                iid::text AS iid,
                datadate::date AS datadate,
                NULL::integer AS fiscal_year,
                NULL::integer AS fiscal_quarter,
                fyr::integer AS fyr,
                indfmt::text AS indfmt,
                consol::text AS consol,
                popsrc::text AS popsrc,
                datafmt::text AS datafmt,
                NULL::text AS ticker,
                NULL::text AS cusip,
                'net_income'::text AS metric_name,
                NULL::double precision AS metric_value,
                COALESCE(niq_fn1, nincq_fn1)::text AS metric_flag
            FROM {sec_flags}
            WHERE datadate BETWEEN %(start)s AND %(end)s
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND consol = 'C'
              AND popsrc = 'D'
        ) variants
        GROUP BY variant_row_id
        """,
    )
    return sql, {"start": window.start.isoformat(), "end": window.end.isoformat()}


def build_ibes_actuals_epsus_sql(settings: WRDSSettings, window: ChunkWindow | None) -> tuple[str, dict[str, Any]]:
    relation = settings.relations.ibes_actuals_epsus
    sql = _override_or_default(
        settings,
        "ibes_actuals_epsus",
        f"""
        SELECT
            actual_row_id,
            MAX(ticker) AS ticker,
            MAX(cusip) AS cusip,
            MAX(oftic) AS oftic,
            MAX(company_name) AS company_name,
            MAX(period_end) AS period_end,
            MAX(measure) AS measure,
            MAX(periodicity) AS periodicity,
            MAX(announcement_date) AS announcement_date,
            MAX(announcement_time) AS announcement_time,
            MAX(actual_date) AS actual_date,
            MAX(actual_time) AS actual_time,
            MAX(actual_value) AS actual_value,
            MAX(currency) AS currency,
            MAX(usfirm) AS usfirm
        FROM (
            SELECT
                CONCAT(
                    COALESCE(ticker::text, ''), '|',
                    COALESCE(cusip::text, ''), '|',
                    COALESCE(measure::text, ''), '|',
                    COALESCE(pends::date::text, ''), '|',
                    COALESCE(anndats::date::text, '')
                ) AS actual_row_id,
                ticker::text AS ticker,
                cusip::text AS cusip,
                oftic::text AS oftic,
                cname::text AS company_name,
                pends::date AS period_end,
                measure::text AS measure,
                pdicity::text AS periodicity,
                anndats::date AS announcement_date,
                anntims::text AS announcement_time,
                actdats::date AS actual_date,
                acttims::text AS actual_time,
                value::double precision AS actual_value,
                curr_act::text AS currency,
                usfirm::integer AS usfirm
            FROM {relation}
            WHERE pends BETWEEN %(start)s AND %(end)s
        ) actuals
        GROUP BY actual_row_id
        ORDER BY period_end, ticker
        """,
    )
    return sql, {"start": window.start.isoformat(), "end": window.end.isoformat()}


def build_ibes_summary_epsus_sql(settings: WRDSSettings, window: ChunkWindow | None) -> tuple[str, dict[str, Any]]:
    relation = settings.relations.ibes_summary_epsus
    sql = _override_or_default(
        settings,
        "ibes_summary_epsus",
        f"""
        SELECT
            summary_row_id,
            MAX(ticker) AS ticker,
            MAX(cusip) AS cusip,
            MAX(oftic) AS oftic,
            MAX(company_name) AS company_name,
            MAX(statpers) AS statpers,
            MAX(measure) AS measure,
            MAX(fiscal_period) AS fiscal_period,
            MAX(forecast_period_code) AS forecast_period_code,
            MAX(estimate_flag) AS estimate_flag,
            MAX(currency) AS currency,
            MAX(num_estimates) AS num_estimates,
            MAX(num_up) AS num_up,
            MAX(num_down) AS num_down,
            MAX(median_estimate) AS median_estimate,
            MAX(mean_estimate) AS mean_estimate,
            MAX(stdev_estimate) AS stdev_estimate,
            MAX(highest_estimate) AS highest_estimate,
            MAX(lowest_estimate) AS lowest_estimate,
            MAX(usfirm) AS usfirm,
            MAX(period_end) AS period_end,
            MAX(actual_value) AS actual_value,
            MAX(actual_date) AS actual_date,
            MAX(announcement_date) AS announcement_date
        FROM (
            SELECT
                CONCAT(
                    COALESCE(ticker::text, ''), '|',
                    COALESCE(measure::text, ''), '|',
                    COALESCE(statpers::date::text, ''), '|',
                    COALESCE(fpi::text, ''), '|',
                    COALESCE(fpedats::date::text, '')
                ) AS summary_row_id,
                ticker::text AS ticker,
                cusip::text AS cusip,
                oftic::text AS oftic,
                cname::text AS company_name,
                statpers::date AS statpers,
                measure::text AS measure,
                fiscalp::text AS fiscal_period,
                fpi::text AS forecast_period_code,
                estflag::text AS estimate_flag,
                curcode::text AS currency,
                numest::integer AS num_estimates,
                numup::integer AS num_up,
                numdown::integer AS num_down,
                medest::double precision AS median_estimate,
                meanest::double precision AS mean_estimate,
                stdev::double precision AS stdev_estimate,
                highest::double precision AS highest_estimate,
                lowest::double precision AS lowest_estimate,
                usfirm::integer AS usfirm,
                fpedats::date AS period_end,
                actual::double precision AS actual_value,
                actdats_act::date AS actual_date,
                anndats_act::date AS announcement_date
            FROM {relation}
            WHERE statpers BETWEEN %(start)s AND %(end)s
        ) summaries
        GROUP BY summary_row_id
        ORDER BY statpers, ticker
        """,
    )
    return sql, {"start": window.start.isoformat(), "end": window.end.isoformat()}


def build_source_access_registry_sql(settings: WRDSSettings, window: ChunkWindow | None) -> tuple[str, dict[str, Any]]:
    del window
    return "SELECT NULL WHERE FALSE", {}


def build_index_membership_sql(
    settings: WRDSSettings,
    *,
    membership_code: str,
    membership_name: str,
    index_ids: tuple[str, ...],
    window: ChunkWindow,
) -> tuple[str, dict[str, Any]]:
    relation = settings.relations.index_membership
    if not index_ids:
        return "SELECT NULL WHERE FALSE", {}
    sql = f"""
        SELECT
            'index'::text AS membership_type,
            '{membership_code}'::text AS membership_code,
            '{membership_name}'::text AS membership_name,
            gvkey::text AS gvkey,
            iid::text AS iid,
            NULL::text AS ticker,
            "from"::date AS effective_from,
            "thru"::date AS effective_to
        FROM {relation}
        WHERE gvkeyx IN {_quoted_literal_list(index_ids)}
          AND "from" <= %(end)s
          AND ("thru" IS NULL OR "thru" >= %(start)s)
        ORDER BY "from", gvkey, iid
    """
    override_key = f"universe_index_{membership_code.lower()}"
    return _override_or_default(settings, override_key, sql), {
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
    }


def source_dataset_specs(settings: WRDSSettings) -> dict[str, DatasetSpec]:
    """Return the default source dataset specifications."""

    return {
        "crsp_daily": DatasetSpec(
            name="crsp_daily",
            target_table="wrds_crsp_daily",
            key_columns=("permno", "trade_date"),
            query_builder=build_crsp_daily_sql,
            chunk_months=settings.chunking.crsp_daily_months,
            date_column="trade_date",
            summary_distinct_columns=("permno", "permco"),
        ),
        "compustat_quarterly": DatasetSpec(
            name="compustat_quarterly",
            target_table="wrds_compustat_quarterly",
            key_columns=("gvkey", "datadate", "datafqtr", "indfmt", "consol", "popsrc", "datafmt"),
            query_builder=build_compustat_quarterly_sql,
            chunk_months=settings.chunking.compustat_quarterly_months,
            date_column="datadate",
            summary_distinct_columns=("gvkey", "tic"),
        ),
        "compustat_annual": DatasetSpec(
            name="compustat_annual",
            target_table="wrds_compustat_annual",
            key_columns=("gvkey", "datadate", "indfmt", "consol", "popsrc", "datafmt"),
            query_builder=build_compustat_annual_sql,
            chunk_months=settings.chunking.compustat_annual_months,
            date_column="datadate",
            summary_distinct_columns=("gvkey", "tic"),
        ),
        "compustat_segments_historical": DatasetSpec(
            name="compustat_segments_historical",
            target_table="wrds_compustat_segments_historical",
            key_columns=("segment_row_id",),
            query_builder=build_compustat_segments_historical_sql,
            chunk_months=settings.chunking.compustat_segments_months,
            date_column="datadate",
            summary_distinct_columns=("gvkey", "segment_type"),
        ),
        "compustat_quarterly_variant_metrics": DatasetSpec(
            name="compustat_quarterly_variant_metrics",
            target_table="wrds_compustat_quarterly_variant_metrics",
            key_columns=("variant_row_id",),
            query_builder=build_compustat_quarterly_variant_metrics_sql,
            chunk_months=settings.chunking.compustat_quarterly_variant_months,
            date_column="datadate",
            summary_distinct_columns=("gvkey", "source_variant", "metric_name"),
        ),
        "ibes_actuals_epsus": DatasetSpec(
            name="ibes_actuals_epsus",
            target_table="wrds_ibes_actuals_epsus",
            key_columns=("actual_row_id",),
            query_builder=build_ibes_actuals_epsus_sql,
            chunk_months=settings.chunking.ibes_actuals_months,
            date_column="period_end",
            summary_distinct_columns=("ticker", "measure"),
        ),
        "ibes_summary_epsus": DatasetSpec(
            name="ibes_summary_epsus",
            target_table="wrds_ibes_summary_epsus",
            key_columns=("summary_row_id",),
            query_builder=build_ibes_summary_epsus_sql,
            chunk_months=settings.chunking.ibes_summary_months,
            date_column="statpers",
            summary_distinct_columns=("ticker", "measure", "forecast_period_code"),
        ),
        "source_access_registry": DatasetSpec(
            name="source_access_registry",
            target_table="wrds_source_access_registry",
            key_columns=("source_key",),
            query_builder=build_source_access_registry_sql,
            chunk_months=None,
            date_column="checked_at",
            summary_distinct_columns=("dataset_group", "relation_name", "access_status"),
            full_refresh_only=True,
        ),
        "ccm_link": DatasetSpec(
            name="ccm_link",
            target_table="wrds_ccm_link",
            key_columns=("gvkey", "liid", "lpermno", "linkdt"),
            query_builder=build_ccm_link_sql,
            chunk_months=None,
            date_column="linkdt",
            summary_distinct_columns=("gvkey", "lpermno"),
            full_refresh_only=True,
        ),
        "security_master": DatasetSpec(
            name="security_master",
            target_table="wrds_security_master",
            key_columns=("permno", "namedt"),
            query_builder=build_security_master_sql,
            chunk_months=None,
            date_column="namedt",
            summary_distinct_columns=("permno", "permco", "ticker"),
            full_refresh_only=True,
        ),
        "company_master": DatasetSpec(
            name="company_master",
            target_table="wrds_company_master",
            key_columns=("gvkey",),
            query_builder=build_company_master_sql,
            chunk_months=None,
            date_column=None,
            summary_distinct_columns=("gvkey", "tic"),
            full_refresh_only=True,
        ),
        "universe_membership_history": DatasetSpec(
            name="universe_membership_history",
            target_table="wrds_universe_membership_history",
            key_columns=("membership_type", "membership_code", "member_key", "effective_from"),
            query_builder=None,
            chunk_months=None,
            date_column="effective_from",
            summary_distinct_columns=("membership_code", "member_key"),
            full_refresh_only=True,
        ),
    }


def resolve_dataset_names(raw_names: str | list[str]) -> list[str]:
    """Normalize dataset selections and expand aliases such as `all` and `links`."""

    if isinstance(raw_names, str):
        tokens = [item.strip() for item in raw_names.split(",") if item.strip()]
    else:
        tokens = [str(item).strip() for item in raw_names if str(item).strip()]

    resolved: list[str] = []
    for token in tokens or ["all"]:
        names = DATASET_ALIASES.get(token, (token,))
        for name in names:
            if name not in resolved:
                resolved.append(name)
    return resolved


def append_audit_columns(frame: pd.DataFrame, *, source_relation: str, run_id: str, collected_at: pd.Timestamp) -> pd.DataFrame:
    """Attach standard ingestion audit columns to a WRDS chunk."""

    enriched = frame.copy()
    enriched["source_relation"] = source_relation
    enriched["ingest_run_id"] = run_id
    enriched["collected_at"] = collected_at
    return enriched
