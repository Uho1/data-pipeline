from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from market_data.wrds.client import WRDSClient
from market_data.wrds.config import WRDSSettings, WRDS_KR_LAKE_DB_PATH
from market_data.wrds.duckdb_io import DuckDBManager

KR_EXCHANGE_TIER_MAP: dict[int, str] = {
    248: "KOSPI",
    298: "KOSDAQ",
}
KR_CORE_RELATIONS: tuple[tuple[str, str, str, str], ...] = (
    ("comp_global_daily", "g_company", "company_master", "kr_company_master"),
    ("comp_global_daily", "g_security", "security_master", "kr_company_master"),
    ("comp_global_daily", "g_funda", "annual_fundamentals", "kr_financials_annual_canonical"),
    ("comp_global_daily", "g_fundq", "quarterly_fundamentals", "kr_financials_quarterly_canonical"),
    ("comp_global_daily", "g_secd", "security_status_history", "kr_security_status_history"),
    ("comp_global_daily", "g_funda_fncd", "annual_status_flags", ""),
    ("comp_global_daily", "g_fundq_fncd", "quarterly_status_flags", ""),
    ("comp_global_daily", "g_co_ifndq", "quarterly_unrestated_like", ""),
    ("comp_global_daily", "g_co_ifndsa", "quarterly_semi_like", ""),
    ("comp_global_daily", "g_co_ifndytd", "quarterly_ytd_like", ""),
    ("comp_global_daily", "g_co_ifntq", "quarterly_flag_like", ""),
    ("comp_segments_hist_daily", "wrds_segmerged", "historical_segments", ""),
    ("comp_segments_hist_daily", "seg_geo", "historical_segments_geo", ""),
    ("comp_segments_hist_daily", "seg_product", "historical_segments_product", ""),
    ("comp_segments_hist_daily", "seg_customer", "historical_segments_customer", ""),
    ("ibes", "act_epsint", "ibes_actuals_intl", ""),
    ("ibes", "statsum_epsint", "ibes_summary_intl", ""),
)


@dataclass(frozen=True)
class KRLakeSummary:
    db_path: Path
    company_rows: int
    annual_rows: int
    quarterly_rows: int
    status_rows: int


def _local_code_from_isin(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if len(text) == 12 and text.startswith("KR") and text[3:9].isdigit():
        return text[3:9]
    return None


def _market_tier(exchange_code: Any, exchange_country: Any) -> str | None:
    try:
        numeric = int(exchange_code)
    except Exception:  # noqa: BLE001
        numeric = None
    if numeric in KR_EXCHANGE_TIER_MAP:
        return KR_EXCHANGE_TIER_MAP[numeric]
    if exchange_country is not None and not pd.isna(exchange_country) and str(exchange_country).upper() == "KOR" and numeric is not None:
        return f"KR_OTHER_{numeric}"
    return None


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _first_not_null(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        result = result.where(result.notna(), frame[column])
    return result


class WRDSKRLakeBuilder:
    """Build a Korea-focused WRDS reference lake in a separate DuckDB."""

    def __init__(
        self,
        settings: WRDSSettings,
        *,
        db_path: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.db_path = Path(db_path or WRDS_KR_LAKE_DB_PATH).expanduser()
        self.db = DuckDBManager(self.db_path)
        self.logger = logger or logging.getLogger(__name__)

    def survey_availability(self) -> dict[str, Any]:
        self.db.init_schema()
        collected_at = pd.Timestamp.utcnow()
        with WRDSClient(self.settings) as client:
            inventory = self._build_relation_inventory(client, collected_at=collected_at)
        inventory_frame = pd.DataFrame(inventory)
        self.db.replace_dataframe("kr_relation_inventory", inventory_frame)
        company_row = next((row for row in inventory if row["relation_name"] == "g_company"), None)
        annual_row = next((row for row in inventory if row["relation_name"] == "g_funda"), None)
        quarterly_row = next((row for row in inventory if row["relation_name"] == "g_fundq"), None)
        summary = {
            "db_path": str(self.db_path),
            "inventory_rows": int(len(inventory_frame)),
            "kospi_supported": True,
            "kosdaq_supported": True,
            "korea_company_rows": int(company_row.get("row_count", 0) if company_row else 0),
            "annual_fundamental_rows": int(annual_row.get("row_count", 0) if annual_row else 0),
            "quarterly_fundamental_rows": int(quarterly_row.get("row_count", 0) if quarterly_row else 0),
            "sample_relations": json.loads(inventory_frame.head(10).to_json(orient="records", date_format="iso")),
        }
        return summary

    def build(self) -> dict[str, Any]:
        self.db.init_schema()
        collected_at = pd.Timestamp.utcnow()
        with WRDSClient(self.settings) as client:
            company_master = self._fetch_company_master(client, collected_at=collected_at)
            annual = self._fetch_annual_fundamentals(client, company_master, collected_at=collected_at)
            quarterly = self._fetch_quarterly_fundamentals(client, company_master, collected_at=collected_at)
            status = self._fetch_security_status_history(client, company_master, collected_at=collected_at)
            inventory = self._build_relation_inventory(
                client,
                collected_at=collected_at,
                materialized_counts={
                    "kr_company_master": len(company_master),
                    "kr_financials_annual_canonical": len(annual),
                    "kr_financials_quarterly_canonical": len(quarterly),
                    "kr_security_status_history": len(status),
                },
            )

        self.db.replace_dataframe("kr_company_master", company_master)
        self.db.replace_dataframe("kr_financials_annual_canonical", annual)
        self.db.replace_dataframe("kr_financials_quarterly_canonical", quarterly)
        self.db.replace_dataframe("kr_security_status_history", status)
        self.db.replace_dataframe("kr_relation_inventory", pd.DataFrame(inventory))

        summary = {
            "db_path": str(self.db_path),
            "tables": {
                "kr_company_master": self._table_summary(company_master, "gvkey", "market_tier"),
                "kr_financials_annual_canonical": self._date_table_summary(annual, "gvkey", "period_end"),
                "kr_financials_quarterly_canonical": self._date_table_summary(quarterly, "gvkey", "period_end"),
                "kr_security_status_history": self._date_table_summary(status, "gvkey", "status_date"),
                "kr_relation_inventory": {"row_count": int(len(inventory))},
            },
            "market_tier_counts": (
                company_master["market_tier"].fillna("UNKNOWN").value_counts(dropna=False).to_dict()
                if not company_master.empty
                else {}
            ),
            "core_metrics_available": [
                "revenue",
                "gross_profit",
                "operating_income",
                "net_income",
                "operating_cash_flow",
                "capex",
                "cash",
                "receivables",
                "inventory",
                "current_assets",
                "non_current_assets",
                "current_liabilities",
                "non_current_liabilities",
                "total_assets",
                "total_liabilities",
                "equity",
                "retained_earnings",
                "common_stock",
                "r_and_d_annual",
                "d_and_a",
                "pretax_income",
                "tax",
                "interest",
            ],
        }
        return summary

    def _fetch_company_master(self, client: WRDSClient, *, collected_at: pd.Timestamp) -> pd.DataFrame:
        sql = """
        WITH ranked AS (
            SELECT
                c.gvkey::text AS gvkey,
                c.conm::text AS company_name,
                c.conml::text AS company_name_long,
                c.cik::text AS cik,
                c.fic::text AS fic,
                c.loc::text AS loc,
                c.costat::text AS company_status,
                c.gsector::text AS gsector,
                c.gind::text AS gind,
                c.gsubind::text AS gsubind,
                c.sic::text AS sic,
                c.naics::text AS naics,
                c.stko::text AS stko,
                c.prirow::text AS prirow,
                c.ipodate::date AS ipodate,
                c.dldte::date AS dldte,
                s.iid::text AS iid,
                s.tic::text AS ticker_raw,
                s.ibtic::text AS ibtic,
                s.cusip::text AS cusip,
                s.isin::text AS isin,
                s.exchg::integer AS exchange_code,
                s.excntry::text AS exchange_country,
                s.secstat::text AS security_status,
                s.dldtei::date AS dldtei,
                ex.exchgdesc::text AS exchange_desc,
                ROW_NUMBER() OVER (
                    PARTITION BY c.gvkey
                    ORDER BY
                        CASE WHEN s.iid = c.prirow THEN 0 ELSE 1 END,
                        CASE WHEN s.secstat = 'A' THEN 0 ELSE 1 END,
                        CASE WHEN s.exchg IN (248, 298) THEN 0 ELSE 1 END,
                        COALESCE(s.dldtei, DATE '9999-12-31') DESC,
                        COALESCE(s.iid, '')
                ) AS rn
            FROM comp_global_daily.g_company c
            LEFT JOIN comp_global_daily.g_security s
              ON s.gvkey = c.gvkey
             AND s.excntry = 'KOR'
            LEFT JOIN comp_global_daily.r_ex_codes ex
              ON ex.exchgcd = s.exchg
            WHERE c.fic = 'KOR' OR c.loc = 'KOR'
        )
        SELECT * FROM ranked WHERE rn = 1
        ORDER BY gvkey
        """
        frame = client.raw_sql(sql).rows
        if frame.empty:
            return pd.DataFrame(columns=[
                "company_key", "gvkey", "iid", "company_name", "company_name_long", "cik", "fic", "loc",
                "ticker_raw", "ibtic", "cusip", "isin", "local_code_6", "exchange_code", "exchange_desc",
                "exchange_country", "market_tier", "company_status", "security_status", "active_flag",
                "gsector", "gind", "gsubind", "sic", "naics", "stko", "prirow", "ipodate", "dldte", "dldtei",
                "source_relation", "collected_at",
            ])
        for column in ("ipodate", "dldte", "dldtei"):
            frame[column] = _to_date(frame[column])
        frame["local_code_6"] = frame["isin"].map(_local_code_from_isin)
        frame["market_tier"] = [
            _market_tier(exchange_code, exchange_country)
            for exchange_code, exchange_country in zip(frame["exchange_code"], frame["exchange_country"], strict=False)
        ]
        frame["active_flag"] = (
            frame["dldte"].isna()
            & frame["dldtei"].isna()
            & frame["security_status"].fillna("A").ne("I")
        )
        frame["company_key"] = frame.apply(
            lambda row: f"{row['gvkey']}|{row['iid'] if pd.notna(row['iid']) and str(row['iid']).strip() else 'PRIMARY'}",
            axis=1,
        )
        frame["source_relation"] = "comp_global_daily.g_company + comp_global_daily.g_security"
        frame["collected_at"] = collected_at
        return frame[
            [
                "company_key",
                "gvkey",
                "iid",
                "company_name",
                "company_name_long",
                "cik",
                "fic",
                "loc",
                "ticker_raw",
                "ibtic",
                "cusip",
                "isin",
                "local_code_6",
                "exchange_code",
                "exchange_desc",
                "exchange_country",
                "market_tier",
                "company_status",
                "security_status",
                "active_flag",
                "gsector",
                "gind",
                "gsubind",
                "sic",
                "naics",
                "stko",
                "prirow",
                "ipodate",
                "dldte",
                "dldtei",
                "source_relation",
                "collected_at",
            ]
        ]

    def _fetch_annual_fundamentals(
        self,
        client: WRDSClient,
        company_master: pd.DataFrame,
        *,
        collected_at: pd.Timestamp,
    ) -> pd.DataFrame:
        sql = """
        WITH korea AS (
            SELECT DISTINCT gvkey
            FROM comp_global_daily.g_company
            WHERE fic = 'KOR' OR loc = 'KOR'
        ),
        ranked AS (
            SELECT
                a.*,
                ROW_NUMBER() OVER (
                    PARTITION BY a.gvkey, a.datadate, a.fyear
                    ORDER BY
                        CASE WHEN a.indfmt = 'INDL' THEN 0 ELSE 1 END,
                        CASE WHEN a.consol = 'C' THEN 0 ELSE 1 END,
                        CASE WHEN a.datafmt = 'STD' THEN 0 ELSE 1 END,
                        CASE WHEN a.final = 'Y' THEN 0 ELSE 1 END,
                        CASE WHEN a.popsrc = 'D' THEN 0 ELSE 1 END,
                        COALESCE(a.pdate, a.fdate, a.datadate) DESC
                ) AS rn
            FROM comp_global_daily.g_funda a
            INNER JOIN korea k
              ON k.gvkey = a.gvkey
        )
        SELECT
            gvkey::text AS gvkey,
            datadate::date AS period_end,
            COALESCE(pdate, fdate, datadate)::date AS available_date,
            fdate::date AS filing_date,
            fyear::integer AS fiscal_year,
            fyr::integer AS fiscal_year_end_month,
            curcd::text AS currency_code,
            indfmt::text AS indfmt,
            consol::text AS consol,
            popsrc::text AS popsrc,
            datafmt::text AS datafmt,
            final::text AS final_flag,
            upd::text AS update_code,
            src::text AS source_code,
            sale::double precision AS sale,
            revt::double precision AS revt,
            cogs::double precision AS cogs,
            oiadp::double precision AS operating_income,
            pi::double precision AS pretax_income,
            txt::double precision AS tax,
            ib::double precision AS net_income,
            cfo::double precision AS cfo,
            oancf::double precision AS oancf,
            capx::double precision AS capex,
            che::double precision AS cash,
            rect::double precision AS receivables,
            invt::double precision AS inventory,
            act::double precision AS current_assets,
            lct::double precision AS current_liabilities,
            at::double precision AS total_assets,
            lt::double precision AS total_liabilities,
            ceq::double precision AS ceq,
            seq::double precision AS seq,
            re::double precision AS retained_earnings,
            cstk::double precision AS common_stock,
            xrd::double precision AS r_and_d,
            dp::double precision AS dp,
            am::double precision AS am,
            xint::double precision AS interest
        FROM ranked
        WHERE rn = 1
        ORDER BY period_end, gvkey
        """
        frame = client.raw_sql(sql).rows
        if frame.empty:
            return pd.DataFrame(columns=[
                "row_id", "gvkey", "iid", "company_key", "period_end", "available_date", "filing_date", "fiscal_year",
                "fiscal_year_end_month", "currency_code", "indfmt", "consol", "popsrc", "datafmt", "final_flag",
                "update_code", "source_code", "ticker_raw", "ibtic", "isin", "local_code_6", "market_tier",
                "exchange_code", "cik", "revenue", "gross_profit", "operating_income", "pretax_income", "tax",
                "net_income", "operating_cash_flow", "capex", "cash", "receivables", "inventory", "current_assets",
                "non_current_assets", "current_liabilities", "non_current_liabilities", "total_assets",
                "total_liabilities", "equity", "retained_earnings", "common_stock", "r_and_d", "d_and_a", "interest",
                "source_relation", "collected_at",
            ])
        frame = frame.merge(
            company_master[
                ["gvkey", "iid", "company_key", "ticker_raw", "ibtic", "isin", "local_code_6", "market_tier", "exchange_code", "cik"]
            ].drop_duplicates(subset=["gvkey"]),
            on="gvkey",
            how="left",
            suffixes=("", "_company"),
        )
        frame["period_end"] = _to_date(frame["period_end"])
        frame["available_date"] = _to_date(frame["available_date"])
        frame["filing_date"] = _to_date(frame["filing_date"])
        frame["revenue"] = _first_not_null(frame, ["sale", "revt"])
        frame["gross_profit"] = frame["revenue"] - frame["cogs"]
        frame.loc[frame["revenue"].isna() | frame["cogs"].isna(), "gross_profit"] = pd.NA
        frame["operating_cash_flow"] = _first_not_null(frame, ["cfo", "oancf"])
        frame["non_current_assets"] = frame["total_assets"] - frame["current_assets"]
        frame.loc[frame["total_assets"].isna() | frame["current_assets"].isna(), "non_current_assets"] = pd.NA
        frame["non_current_liabilities"] = frame["total_liabilities"] - frame["current_liabilities"]
        frame.loc[frame["total_liabilities"].isna() | frame["current_liabilities"].isna(), "non_current_liabilities"] = pd.NA
        frame["equity"] = _first_not_null(frame, ["seq", "ceq"])
        frame["d_and_a"] = frame[["dp", "am"]].fillna(0.0).sum(axis=1)
        frame.loc[frame[["dp", "am"]].isna().all(axis=1), "d_and_a"] = pd.NA
        frame["source_relation"] = "comp_global_daily.g_funda"
        frame["collected_at"] = collected_at
        frame["row_id"] = frame.apply(
            lambda row: uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"annual|{row['gvkey']}|{row['period_end']}",
            ).hex,
            axis=1,
        )
        return frame[
            [
                "row_id",
                "gvkey",
                "iid",
                "company_key",
                "period_end",
                "available_date",
                "filing_date",
                "fiscal_year",
                "fiscal_year_end_month",
                "currency_code",
                "indfmt",
                "consol",
                "popsrc",
                "datafmt",
                "final_flag",
                "update_code",
                "source_code",
                "ticker_raw",
                "ibtic",
                "isin",
                "local_code_6",
                "market_tier",
                "exchange_code",
                "cik",
                "revenue",
                "gross_profit",
                "operating_income",
                "pretax_income",
                "tax",
                "net_income",
                "operating_cash_flow",
                "capex",
                "cash",
                "receivables",
                "inventory",
                "current_assets",
                "non_current_assets",
                "current_liabilities",
                "non_current_liabilities",
                "total_assets",
                "total_liabilities",
                "equity",
                "retained_earnings",
                "common_stock",
                "r_and_d",
                "d_and_a",
                "interest",
                "source_relation",
                "collected_at",
            ]
        ]

    def _fetch_quarterly_fundamentals(
        self,
        client: WRDSClient,
        company_master: pd.DataFrame,
        *,
        collected_at: pd.Timestamp,
    ) -> pd.DataFrame:
        sql = """
        WITH korea AS (
            SELECT DISTINCT gvkey
            FROM comp_global_daily.g_company
            WHERE fic = 'KOR' OR loc = 'KOR'
        ),
        ranked AS (
            SELECT
                q.*,
                ROW_NUMBER() OVER (
                    PARTITION BY q.gvkey, q.datadate, q.fyearq, q.fqtr
                    ORDER BY
                        CASE WHEN q.indfmt = 'INDL' THEN 0 ELSE 1 END,
                        CASE WHEN q.consol = 'C' THEN 0 ELSE 1 END,
                        CASE WHEN q.datafmt = 'STD' THEN 0 ELSE 1 END,
                        CASE WHEN q.popsrc = 'D' THEN 0 ELSE 1 END,
                        COALESCE(q.pdateq, q.fdateq, q.datadate) DESC
                ) AS rn
            FROM comp_global_daily.g_fundq q
            INNER JOIN korea k
              ON k.gvkey = q.gvkey
        )
        SELECT
            gvkey::text AS gvkey,
            datadate::date AS period_end,
            COALESCE(pdateq, fdateq, datadate)::date AS available_date,
            fdateq::date AS filing_date,
            fyearq::integer AS fiscal_year,
            fqtr::integer AS fiscal_quarter,
            fyr::integer AS fiscal_year_end_month,
            curcdq::text AS currency_code,
            indfmt::text AS indfmt,
            consol::text AS consol,
            popsrc::text AS popsrc,
            datafmt::text AS datafmt,
            updq::text AS update_code,
            srcq::text AS source_code,
            saleq::double precision AS saleq,
            revtq::double precision AS revtq,
            gpq::double precision AS gpq,
            cogsq::double precision AS cogsq,
            oiadpq::double precision AS operating_income,
            piq::double precision AS pretax_income,
            txtq::double precision AS tax,
            ibq::double precision AS net_income,
            cfoq::double precision AS cfoq,
            oancfy::double precision AS oancfy,
            capxy::double precision AS capxy,
            cheq::double precision AS cash,
            rectq::double precision AS receivables,
            invtq::double precision AS inventory,
            actq::double precision AS current_assets,
            lctq::double precision AS current_liabilities,
            atq::double precision AS total_assets,
            ltq::double precision AS total_liabilities,
            ceqq::double precision AS ceqq,
            seqq::double precision AS seqq,
            req::double precision AS retained_earnings,
            cstkq::double precision AS common_stock,
            NULL::double precision AS r_and_d,
            dpq::double precision AS dpq,
            amq::double precision AS amq,
            xintq::double precision AS interest
        FROM ranked
        WHERE rn = 1
        ORDER BY gvkey, fiscal_year, fiscal_quarter, period_end
        """
        frame = client.raw_sql(sql).rows
        if frame.empty:
            return pd.DataFrame(columns=[
                "row_id", "gvkey", "iid", "company_key", "period_end", "available_date", "filing_date",
                "fiscal_year", "fiscal_quarter", "fiscal_year_end_month", "currency_code", "indfmt", "consol",
                "popsrc", "datafmt", "update_code", "source_code", "ticker_raw", "ibtic", "isin", "local_code_6",
                "market_tier", "exchange_code", "cik", "revenue", "gross_profit", "operating_income", "pretax_income",
                "tax", "net_income", "operating_cash_flow", "capex", "cash", "receivables", "inventory",
                "current_assets", "non_current_assets", "current_liabilities", "non_current_liabilities",
                "total_assets", "total_liabilities", "equity", "retained_earnings", "common_stock", "r_and_d",
                "d_and_a", "interest", "source_relation", "collected_at",
            ])
        frame = frame.merge(
            company_master[
                ["gvkey", "iid", "company_key", "ticker_raw", "ibtic", "isin", "local_code_6", "market_tier", "exchange_code", "cik"]
            ].drop_duplicates(subset=["gvkey"]),
            on="gvkey",
            how="left",
        )
        frame["period_end"] = _to_date(frame["period_end"])
        frame["available_date"] = _to_date(frame["available_date"])
        frame["filing_date"] = _to_date(frame["filing_date"])
        frame["revenue"] = _first_not_null(frame, ["saleq", "revtq"])
        frame["gross_profit"] = _first_not_null(frame, ["gpq"])
        gross_profit_mask = frame["gross_profit"].isna() & frame["revenue"].notna() & frame["cogsq"].notna()
        frame.loc[gross_profit_mask, "gross_profit"] = frame.loc[gross_profit_mask, "revenue"] - frame.loc[gross_profit_mask, "cogsq"]
        frame["operating_cash_flow"] = frame["cfoq"]
        frame = frame.sort_values(["gvkey", "fiscal_year", "fiscal_quarter", "period_end"]).reset_index(drop=True)
        grouped_ytd = frame.groupby(["gvkey", "fiscal_year"], dropna=False)["oancfy"]
        derived_oancf = grouped_ytd.diff()
        first_quarter_mask = frame["fiscal_quarter"].fillna(1).astype("Int64") == 1
        derived_oancf = derived_oancf.where(~first_quarter_mask, frame["oancfy"])
        frame["operating_cash_flow"] = frame["operating_cash_flow"].where(frame["operating_cash_flow"].notna(), derived_oancf)
        grouped_capx = frame.groupby(["gvkey", "fiscal_year"], dropna=False)["capxy"]
        derived_capx = grouped_capx.diff()
        derived_capx = derived_capx.where(~first_quarter_mask, frame["capxy"])
        frame["capex"] = derived_capx
        frame["non_current_assets"] = frame["total_assets"] - frame["current_assets"]
        frame.loc[frame["total_assets"].isna() | frame["current_assets"].isna(), "non_current_assets"] = pd.NA
        frame["non_current_liabilities"] = frame["total_liabilities"] - frame["current_liabilities"]
        frame.loc[frame["total_liabilities"].isna() | frame["current_liabilities"].isna(), "non_current_liabilities"] = pd.NA
        frame["equity"] = _first_not_null(frame, ["seqq", "ceqq"])
        frame["d_and_a"] = frame[["dpq", "amq"]].fillna(0.0).sum(axis=1)
        frame.loc[frame[["dpq", "amq"]].isna().all(axis=1), "d_and_a"] = pd.NA
        frame["source_relation"] = "comp_global_daily.g_fundq"
        frame["collected_at"] = collected_at
        frame["row_id"] = frame.apply(
            lambda row: uuid.uuid5(
                uuid.NAMESPACE_URL,
                "|".join(
                    [
                        "quarterly",
                        str(row["gvkey"]),
                        str(row["period_end"]),
                        "" if pd.isna(row["fiscal_year"]) else str(int(row["fiscal_year"])),
                        "" if pd.isna(row["fiscal_quarter"]) else str(int(row["fiscal_quarter"])),
                    ]
                ),
            ).hex,
            axis=1,
        )
        return frame[
            [
                "row_id",
                "gvkey",
                "iid",
                "company_key",
                "period_end",
                "available_date",
                "filing_date",
                "fiscal_year",
                "fiscal_quarter",
                "fiscal_year_end_month",
                "currency_code",
                "indfmt",
                "consol",
                "popsrc",
                "datafmt",
                "update_code",
                "source_code",
                "ticker_raw",
                "ibtic",
                "isin",
                "local_code_6",
                "market_tier",
                "exchange_code",
                "cik",
                "revenue",
                "gross_profit",
                "operating_income",
                "pretax_income",
                "tax",
                "net_income",
                "operating_cash_flow",
                "capex",
                "cash",
                "receivables",
                "inventory",
                "current_assets",
                "non_current_assets",
                "current_liabilities",
                "non_current_liabilities",
                "total_assets",
                "total_liabilities",
                "equity",
                "retained_earnings",
                "common_stock",
                "r_and_d",
                "d_and_a",
                "interest",
                "source_relation",
                "collected_at",
            ]
        ]

    def _fetch_security_status_history(
        self,
        client: WRDSClient,
        company_master: pd.DataFrame,
        *,
        collected_at: pd.Timestamp,
    ) -> pd.DataFrame:
        start_year = int(self.settings.start_date.year if self.settings.start_date else 1981)
        end_year = int(self.settings.end_date.year if self.settings.end_date else date.today().year)
        chunks: list[pd.DataFrame] = []
        for year in range(start_year, end_year + 1, 5):
            chunk_start = date(year, 1, 1)
            chunk_end = date(min(year + 4, end_year), 12, 31)
            sql = """
            SELECT
                d.gvkey::text AS gvkey,
                d.iid::text AS iid,
                d.datadate::date AS status_date,
                d.exchg::integer AS exchange_code,
                d.secstat::text AS security_status,
                d.fic::text AS fic,
                d.loc::text AS loc,
                d.monthend::double precision AS monthend,
                d.cshoc::double precision AS shares_outstanding,
                s.isin::text AS isin,
                ex.exchgdesc::text AS exchange_desc,
                c.costat::text AS company_status
            FROM comp_global_daily.g_secd d
            LEFT JOIN comp_global_daily.g_security s
              ON s.gvkey = d.gvkey
             AND s.iid = d.iid
            LEFT JOIN comp_global_daily.g_company c
              ON c.gvkey = d.gvkey
            LEFT JOIN comp_global_daily.r_ex_codes ex
              ON ex.exchgcd = d.exchg
            WHERE d.fic = 'KOR'
              AND d.monthend = 1
              AND d.datadate BETWEEN %(start)s AND %(end)s
            ORDER BY d.datadate, d.gvkey, d.iid
            """
            frame = client.raw_sql(
                sql,
                params={"start": chunk_start.isoformat(), "end": chunk_end.isoformat()},
            ).rows
            if frame.empty:
                continue
            chunks.append(frame)
        if not chunks:
            return pd.DataFrame(columns=[
                "status_row_id", "gvkey", "iid", "company_key", "status_date", "isin", "local_code_6", "exchange_code",
                "exchange_desc", "market_tier", "security_status", "company_status", "active_flag", "shares_outstanding",
                "monthend_flag", "fic", "loc", "source_relation", "collected_at",
            ])
        frame = pd.concat(chunks, ignore_index=True)
        frame = frame.merge(
            company_master[["gvkey", "iid", "company_key"]],
            on=["gvkey", "iid"],
            how="left",
        )
        frame["status_date"] = _to_date(frame["status_date"])
        frame["local_code_6"] = frame["isin"].map(_local_code_from_isin)
        frame["market_tier"] = [
            _market_tier(exchange_code, "KOR")
            for exchange_code in frame["exchange_code"]
        ]
        frame["active_flag"] = frame["security_status"].fillna("A").ne("I")
        frame["monthend_flag"] = frame["monthend"].fillna(0).astype(float).eq(1.0)
        frame["source_relation"] = "comp_global_daily.g_secd"
        frame["collected_at"] = collected_at
        frame["status_row_id"] = frame.apply(
            lambda row: uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"status|{row['gvkey']}|{row['iid']}|{row['status_date']}",
            ).hex,
            axis=1,
        )
        return frame[
            [
                "status_row_id",
                "gvkey",
                "iid",
                "company_key",
                "status_date",
                "isin",
                "local_code_6",
                "exchange_code",
                "exchange_desc",
                "market_tier",
                "security_status",
                "company_status",
                "active_flag",
                "shares_outstanding",
                "monthend_flag",
                "fic",
                "loc",
                "source_relation",
                "collected_at",
            ]
        ]

    def _build_relation_inventory(
        self,
        client: WRDSClient,
        *,
        collected_at: pd.Timestamp,
        materialized_counts: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        materialized_counts = materialized_counts or {}
        rows: list[dict[str, Any]] = []
        measures = {
            "g_company": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT gvkey) AS issuer_count
                FROM comp_global_daily.g_company
                WHERE fic = 'KOR' OR loc = 'KOR'
            """,
            "g_security": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT gvkey) AS issuer_count
                FROM comp_global_daily.g_security
                WHERE excntry = 'KOR'
            """,
            "g_funda": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT gvkey) AS issuer_count, MIN(datadate) AS min_date, MAX(datadate) AS max_date
                FROM comp_global_daily.g_funda
                WHERE gvkey IN (
                    SELECT gvkey FROM comp_global_daily.g_company WHERE fic = 'KOR' OR loc = 'KOR'
                )
            """,
            "g_fundq": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT gvkey) AS issuer_count, MIN(datadate) AS min_date, MAX(datadate) AS max_date
                FROM comp_global_daily.g_fundq
                WHERE gvkey IN (
                    SELECT gvkey FROM comp_global_daily.g_company WHERE fic = 'KOR' OR loc = 'KOR'
                )
            """,
            "g_secd": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT gvkey) AS issuer_count, MIN(datadate) AS min_date, MAX(datadate) AS max_date
                FROM comp_global_daily.g_secd
                WHERE fic = 'KOR' AND monthend = 1
            """,
            "wrds_segmerged": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT gvkey) AS issuer_count, MIN(srcdate) AS min_date, MAX(srcdate) AS max_date
                FROM comp_segments_hist_daily.wrds_segmerged
                WHERE gvkey IN (
                    SELECT gvkey FROM comp_global_daily.g_company WHERE fic = 'KOR' OR loc = 'KOR'
                )
            """,
            "seg_geo": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT gvkey) AS issuer_count, MIN(datadate) AS min_date, MAX(datadate) AS max_date
                FROM comp_segments_hist_daily.seg_geo
                WHERE gvkey IN (
                    SELECT gvkey FROM comp_global_daily.g_company WHERE fic = 'KOR' OR loc = 'KOR'
                )
            """,
            "seg_product": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT gvkey) AS issuer_count, MIN(datadate) AS min_date, MAX(datadate) AS max_date
                FROM comp_segments_hist_daily.seg_product
                WHERE gvkey IN (
                    SELECT gvkey FROM comp_global_daily.g_company WHERE fic = 'KOR' OR loc = 'KOR'
                )
            """,
            "seg_customer": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT gvkey) AS issuer_count, MIN(datadate) AS min_date, MAX(datadate) AS max_date
                FROM comp_segments_hist_daily.seg_customer
                WHERE gvkey IN (
                    SELECT gvkey FROM comp_global_daily.g_company WHERE fic = 'KOR' OR loc = 'KOR'
                )
            """,
            "act_epsint": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT oftic) AS issuer_count, MIN(pends) AS min_date, MAX(pends) AS max_date
                FROM ibes.act_epsint
                WHERE usfirm = 0 AND curr_act = 'KRW'
            """,
            "statsum_epsint": """
                SELECT COUNT(*) AS row_count, COUNT(DISTINCT oftic) AS issuer_count, MIN(fpedats) AS min_date, MAX(fpedats) AS max_date
                FROM ibes.statsum_epsint
                WHERE usfirm = 0 AND curr_act = 'KRW'
            """,
        }
        for library_name, relation_name, purpose, materialized_table in KR_CORE_RELATIONS:
            access_status = "available"
            notes = ""
            row_count = issuer_count = None
            min_date = max_date = None
            try:
                client.raw_sql(f"SELECT * FROM {library_name}.{relation_name} LIMIT 1")
                measure_sql = measures.get(relation_name)
                if measure_sql:
                    measured = client.raw_sql(measure_sql).rows
                    if not measured.empty:
                        row = measured.iloc[0]
                        raw_row_count = row.get("row_count")
                        raw_issuer_count = row.get("issuer_count")
                        row_count = int(raw_row_count) if pd.notna(raw_row_count) else 0
                        issuer_count = int(raw_issuer_count) if pd.notna(raw_issuer_count) else 0
                        min_date = row.get("min_date")
                        max_date = row.get("max_date")
                if relation_name in {"act_epsint", "statsum_epsint"}:
                    notes = "KRW/usfirm=0 proxy only; issuer linkage to company master still pending."
                elif relation_name.startswith("seg_") or relation_name == "wrds_segmerged":
                    notes = "Historical segments relation is accessible; not materialized in KR lake yet."
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                if "permission denied" in message or "insufficientprivilege" in message:
                    access_status = "permission_denied"
                elif "does not exist" in message:
                    access_status = "not_found"
                else:
                    access_status = "unavailable"
                notes = str(exc)
            rows.append(
                {
                    "relation_key": f"{library_name}.{relation_name}",
                    "library_name": library_name,
                    "relation_name": relation_name,
                    "purpose": purpose,
                    "materialized_table": materialized_table,
                    "access_status": access_status,
                    "row_count": row_count,
                    "issuer_count": issuer_count,
                    "min_date": pd.to_datetime(min_date, errors="coerce").date() if pd.notna(pd.to_datetime(min_date, errors="coerce")) else None,
                    "max_date": pd.to_datetime(max_date, errors="coerce").date() if pd.notna(pd.to_datetime(max_date, errors="coerce")) else None,
                    "key_columns": json.dumps(self._key_columns_for_relation(relation_name)),
                    "market_coverage": self._market_coverage_note(relation_name),
                    "notes": notes if not materialized_table else self._materialized_note(materialized_table, materialized_counts, notes),
                    "collected_at": collected_at,
                }
            )
        return rows

    @staticmethod
    def _materialized_note(materialized_table: str, materialized_counts: dict[str, int], notes: str) -> str:
        prefix = f"materialized_rows={int(materialized_counts.get(materialized_table, 0))}"
        return f"{prefix}; {notes}".strip("; ")

    @staticmethod
    def _key_columns_for_relation(relation_name: str) -> list[str]:
        mapping = {
            "g_company": ["gvkey"],
            "g_security": ["gvkey", "iid"],
            "g_funda": ["gvkey", "datadate", "fyear"],
            "g_fundq": ["gvkey", "datadate", "fyearq", "fqtr"],
            "g_secd": ["gvkey", "iid", "datadate"],
            "wrds_segmerged": ["gvkey", "sid", "srcdate"],
            "seg_geo": ["gvkey", "sid", "srcdate"],
            "seg_product": ["gvkey", "sid", "srcdate"],
            "seg_customer": ["gvkey", "sid", "srcdate"],
            "act_epsint": ["oftic", "pends", "measure"],
            "statsum_epsint": ["oftic", "fpedats", "measure"],
        }
        return mapping.get(relation_name, [])

    @staticmethod
    def _market_coverage_note(relation_name: str) -> str:
        if relation_name in {"g_company", "g_security", "g_funda", "g_fundq", "g_secd"}:
            return "Korean issuers filtered via fic='KOR' or loc='KOR' / excntry='KOR'."
        if relation_name in {"act_epsint", "statsum_epsint"}:
            return "International IBES proxy via curr_act='KRW' and usfirm=0."
        return "Korean issuer coverage estimated via gvkey join against g_company."

    @staticmethod
    def _table_summary(frame: pd.DataFrame, issuer_column: str, split_column: str) -> dict[str, Any]:
        return {
            "row_count": int(len(frame)),
            "issuer_count": int(frame[issuer_column].nunique()) if not frame.empty else 0,
            "split_counts": frame[split_column].fillna("UNKNOWN").value_counts(dropna=False).to_dict() if split_column in frame.columns else {},
        }

    @staticmethod
    def _date_table_summary(frame: pd.DataFrame, issuer_column: str, date_column: str) -> dict[str, Any]:
        series = pd.to_datetime(frame[date_column], errors="coerce")
        return {
            "row_count": int(len(frame)),
            "issuer_count": int(frame[issuer_column].nunique()) if not frame.empty else 0,
            "min_date": series.min().date().isoformat() if not series.empty and pd.notna(series.min()) else None,
            "max_date": series.max().date().isoformat() if not series.empty and pd.notna(series.max()) else None,
        }
