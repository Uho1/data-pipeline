from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from market_data.wrds.canonical import CANONICAL_BUILD_ORDER, build_canonical_tables
from market_data.wrds.catalog import (
    ChunkWindow,
    append_audit_columns,
    build_index_membership_sql,
    resolve_dataset_names,
    source_dataset_specs,
)
from market_data.wrds.client import WRDSClient, WRDSExecutionError
from market_data.wrds.config import WRDSSettings
from market_data.wrds.config import WRDS_KR_LAKE_DB_PATH, WRDS_LAKE_DB_PATH
from market_data.wrds.duckdb_io import DuckDBManager
from market_data.wrds.kr_lake import WRDSKRLakeBuilder
from market_data.wrds.logging_utils import setup_wrds_logger
from market_data.wrds.schemas import SCHEMA_BY_TABLE
from market_data.wrds.sec_validation import (
    SEC_VALIDATION_START_DATE,
    SECVsWRDSFinancialValidationService,
    seed_default_metric_mapping_registry,
)
from market_data.wrds.sec_segment_validation import (
    SECSegmentValidationService,
    seed_default_segment_cluster_policy_registry,
    seed_default_segment_metric_policy_registry,
)

DEFAULT_START_DATES: dict[str, date] = {
    "crsp_daily": date(1925, 1, 1),
    "compustat_quarterly": date(1950, 1, 1),
    "compustat_annual": date(1920, 1, 1),
    "compustat_segments_historical": date(1960, 1, 1),
    "compustat_quarterly_variant_metrics": date(1960, 1, 1),
    "ibes_actuals_epsus": date(1976, 1, 1),
    "ibes_summary_epsus": date(1976, 1, 1),
}
SAMPLE_LOOKBACK_DAYS: dict[str, int] = {
    "crsp_daily": 14,
    "compustat_quarterly": 365,
    "compustat_annual": 365 * 5,
    "compustat_segments_historical": 365 * 2,
    "compustat_quarterly_variant_metrics": 365,
    "ibes_actuals_epsus": 365 * 2,
    "ibes_summary_epsus": 365 * 2,
}
CANONICAL_PROFILE_CONFIG: dict[str, dict[str, Any]] = {
    "entity_master": {"date_column": "last_seen_date", "distinct_columns": ("gvkey", "current_ticker")},
    "security_link_history": {"date_column": "effective_from", "distinct_columns": ("gvkey", "permno", "ticker")},
    "prices_daily_canonical": {"date_column": "trade_date", "distinct_columns": ("permno", "gvkey", "ticker")},
    "financials_quarterly_canonical": {
        "date_column": "period_end",
        "distinct_columns": ("gvkey", "ticker", "cik"),
    },
    "financials_annual_canonical": {
        "date_column": "period_end",
        "distinct_columns": ("gvkey", "ticker", "cik"),
    },
    "segments_historical_canonical": {
        "date_column": "period_end",
        "distinct_columns": ("gvkey", "ticker", "segment_type", "segment_key"),
    },
    "universe_membership_history": {
        "date_column": "effective_from",
        "distinct_columns": ("membership_code", "member_key", "gvkey", "permno"),
    },
}


@dataclass(frozen=True)
class RunContext:
    """Per-run metadata persisted into wrds_ingest_runs."""

    run_id: str
    started_at: pd.Timestamp
    command_name: str
    dataset_scope: str


@dataclass(frozen=True)
class WRDSDatasetQueryError(RuntimeError):
    """Rich, safe dataset-level diagnostics for WRDS query failures."""

    dataset_name: str
    relation_name: str
    sql_template_name: str
    sql_text: str
    chunk_window: dict[str, Any]
    exception_class: str
    exception_message: str

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "relation_name": self.relation_name,
            "sql_template_name": self.sql_template_name,
            "sql_text": self.sql_text,
            "chunk_window": self.chunk_window,
            "exception_class": self.exception_class,
            "exception_message": self.exception_message,
        }


class WRDSIngestionService:
    """Orchestrates WRDS source ingestion and canonical table materialization."""

    def __init__(self, settings: WRDSSettings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or setup_wrds_logger(settings.log_dir)
        self.db = DuckDBManager(settings.db_path)
        self.specs = source_dataset_specs(settings)
        self._source_max_date_cache: dict[str, date | None] = {}

    def init_schema(self) -> dict[str, Any]:
        """Initialize WRDS tables inside DuckDB."""

        self.db.init_schema()
        registry_rows = seed_default_metric_mapping_registry(self.db)
        segment_registry_rows = seed_default_segment_metric_policy_registry(self.db)
        summary = {
            "db_path": str(self.settings.db_path),
            "tables_created": len(SCHEMA_BY_TABLE),
            "metric_registry_rows_seeded": int(registry_rows),
            "segment_metric_policy_rows_seeded": int(segment_registry_rows),
            "segment_cluster_policy_rows_seeded": int(seed_default_segment_cluster_policy_registry(self.db)),
        }
        self.logger.info("WRDS schema initialized at %s", self.settings.db_path)
        return summary

    def ingest(self, datasets: str | list[str], *, build_canonical_after: bool = False) -> dict[str, Any]:
        """Ingest one or more WRDS source datasets into DuckDB."""

        dataset_names = resolve_dataset_names(datasets)
        if self._requires_wrds_client(dataset_names):
            WRDSClient(self.settings).preflight_credentials()
        if not self.settings.dry_run:
            self.db.init_schema()
            run = self._start_run("wrds_ingest", dataset_names)
            run_id = run.run_id
        else:
            run = RunContext(
                run_id="dry_run",
                started_at=pd.Timestamp.utcnow(),
                command_name="wrds_ingest",
                dataset_scope=",".join(dataset_names),
            )
            run_id = "dry_run"
        summary: dict[str, Any] = {"run_id": run_id, "datasets": {}, "dry_run": self.settings.dry_run}
        try:
            if self._requires_wrds_client(dataset_names):
                with WRDSClient(self.settings) as client:
                    for dataset_name in dataset_names:
                        if dataset_name == "universe_membership_history":
                            dataset_summary = self._refresh_universe_membership_source(run, client)
                        elif dataset_name == "source_access_registry":
                            dataset_summary = self._refresh_source_access_registry(run, client)
                        else:
                            dataset_summary = self._ingest_source_dataset(run, client, dataset_name)
                        summary["datasets"][dataset_name] = dataset_summary
            else:
                for dataset_name in dataset_names:
                    if dataset_name == "universe_membership_history":
                        dataset_summary = self._refresh_universe_membership_source(run, client=None)
                    elif dataset_name == "source_access_registry":
                        dataset_summary = self._refresh_source_access_registry(run, client=None)
                    else:
                        raise ValueError(f"Dataset {dataset_name} requires WRDS access")
                    summary["datasets"][dataset_name] = dataset_summary

            if build_canonical_after and not self.settings.dry_run:
                summary["canonical"] = self.build_canonical(CANONICAL_BUILD_ORDER, command_name="wrds_build_canonical")["tables"]

            if not self.settings.dry_run:
                self._finish_run(run, "completed", summary=summary)
                self._write_manifest(run.run_id, summary)
            return summary
        except Exception as exc:  # noqa: BLE001
            if not self.settings.dry_run:
                self._finish_run(run, "failed", summary=summary, error_text=str(exc))
            raise

    def probe_connection(self, *, live: bool = False) -> dict[str, Any]:
        """Return safe credential diagnostics and optionally execute a tiny live WRDS query."""

        client = WRDSClient(self.settings)
        return client.probe(live=live)

    def list_relations(self, libraries: str | list[str], *, preview_limit: int = 20) -> dict[str, Any]:
        """List accessible WRDS libraries and a preview of relation names."""

        tokens = self._split_csv(libraries)
        with WRDSClient(self.settings) as client:
            return client.list_relations(tokens, preview_limit=preview_limit)

    def test_query(self, dataset: str, *, limit: int = 1) -> dict[str, Any]:
        """Execute a tiny live test query for a dataset and report safe diagnostics."""

        if dataset not in self.specs:
            raise ValueError(f"Unknown WRDS dataset: {dataset}")
        spec = self.specs[dataset]
        if spec.query_builder is None:
            raise ValueError(f"Dataset {dataset} does not have a direct WRDS SQL query")

        with WRDSClient(self.settings) as client:
            effective_end_date = self._effective_end_date(dataset, client)
        window = self._build_windows(
            spec.name,
            spec.chunk_months,
            spec.full_refresh_only,
            use_checkpoints=False,
            resolved_end_date=effective_end_date,
        )
        query_window = window[0] if window else None
        sql, params = self._build_dataset_sql(spec, dataset, query_window)
        test_sql = f"SELECT * FROM ({sql}) wrds_probe LIMIT {max(1, int(limit))}"
        relation_name = self._source_relation_for_dataset(dataset)
        try:
            with WRDSClient(self.settings) as client:
                result = client.raw_sql(test_sql, params=params)
                if dataset == "ccm_link":
                    relation_name = self._source_relation_for_dataset(dataset)
        except Exception as exc:  # noqa: BLE001
            if dataset == "ccm_link" and self._should_use_ccm_fallback(exc):
                relation_name = "heuristic:comp.names+crsp.dsenames"
                fallback_sql = self._build_ccm_fallback_sql(limit=max(250, min(1000, int(limit) * 100)))
                try:
                    with WRDSClient(self.settings) as client:
                        result = client.raw_sql(f"SELECT * FROM ({fallback_sql}) wrds_probe LIMIT {max(1, int(limit))}")
                    test_sql = f"SELECT * FROM ({fallback_sql}) wrds_probe LIMIT {max(1, int(limit))}"
                except Exception as fallback_exc:  # noqa: BLE001
                    raise self._dataset_query_error(dataset, query_window, fallback_sql, fallback_exc) from fallback_exc
            else:
                raise self._dataset_query_error(dataset, query_window, test_sql, exc) from exc

        sample_rows = result.rows.to_dict(orient="records")
        return {
            "dataset_name": dataset,
            "relation_name": relation_name,
            "sql_template_name": self._sql_template_name(dataset),
            "chunk_window": self._window_payload(query_window),
            "row_count": int(len(result.rows)),
            "columns": result.rows.columns.tolist(),
            "sample_rows": sample_rows,
        }

    def validate_source(self, datasets: str | list[str]) -> dict[str, Any]:
        """Return validation summaries for WRDS source tables."""

        dataset_names = resolve_dataset_names(datasets)
        payload: dict[str, Any] = {"datasets": {}}
        for dataset_name in dataset_names:
            spec = self.specs[dataset_name]
            payload["datasets"][dataset_name] = self.db.table_profile(
                spec.target_table,
                date_column=spec.date_column,
                key_columns=spec.key_columns,
                distinct_columns=spec.summary_distinct_columns,
            )
        return payload

    def validate_canonical(self, tables: str | list[str] | tuple[str, ...]) -> dict[str, Any]:
        """Return validation summaries for canonical WRDS tables."""

        table_names = self._resolve_canonical_tables(tables)
        payload: dict[str, Any] = {"tables": {}}
        for table_name in table_names:
            config = CANONICAL_PROFILE_CONFIG[table_name]
            payload["tables"][table_name] = self.db.table_profile(
                table_name,
                date_column=config["date_column"],
                key_columns=SCHEMA_BY_TABLE[table_name].primary_key,
                distinct_columns=config["distinct_columns"],
            )
        return payload

    def inspect(self, dataset: str, *, layer: str = "auto", limit: int = 5) -> dict[str, Any]:
        """Return schema/profile/sample rows for a source dataset or canonical table."""

        table_name, date_column, key_columns, distinct_columns, resolved_layer = self._resolve_inspection_target(
            dataset,
            layer=layer,
        )
        return {
            "layer": resolved_layer,
            "table_name": table_name,
            "profile": self.db.table_profile(
                table_name,
                date_column=date_column,
                key_columns=key_columns,
                distinct_columns=distinct_columns,
            ),
            "sample_rows": self.db.inspect_rows(table_name, limit=limit),
        }

    def build_canonical(
        self,
        tables: str | list[str] | tuple[str, ...],
        *,
        command_name: str = "wrds_build_canonical",
    ) -> dict[str, Any]:
        """Materialize canonical research tables from WRDS source tables."""

        self.db.init_schema()
        selected_tables = self._resolve_canonical_tables(tables)
        run = self._start_run(command_name, selected_tables)
        summary: dict[str, Any] = {"run_id": run.run_id, "tables": {}}
        try:
            summary["tables"] = build_canonical_tables(self.db, self.logger, selected_tables=selected_tables)
            self._finish_run(run, "completed", summary=summary)
            self._write_manifest(run.run_id, summary)
            return summary
        except Exception as exc:  # noqa: BLE001
            self._finish_run(run, "failed", summary=summary, error_text=str(exc))
            raise

    def compare_sec_crossdb(
        self,
        *,
        sec_db_path: str | Path | None = None,
        wrds_db_path: str | Path | None = None,
        tickers: str | list[str] | None = None,
        market: str = "us",
        limit: int | None = None,
        start_date: date | None = None,
        statement_types: tuple[str, ...] = ("quarterly", "annual"),
        compare_mode: str = "default",
        metric_names: tuple[str, ...] | None = None,
        include_inactive: bool = False,
    ) -> dict[str, Any]:
        sec_db = DuckDBManager(Path(sec_db_path).expanduser() if sec_db_path is not None else self.settings.db_path)
        wrds_db = Path(wrds_db_path).expanduser() if wrds_db_path is not None else WRDS_LAKE_DB_PATH
        sec_db.init_schema()
        run = self._start_run("wrds_compare_sec_crossdb", list(statement_types))
        engine = SECVsWRDSFinancialValidationService(sec_db, self.logger, wrds_db_path=wrds_db)
        summary: dict[str, Any] = {}
        try:
            summary = engine.compare(
                comparison_run_id=run.run_id,
                tickers=tickers,
                market=market,
                start_date=start_date or SEC_VALIDATION_START_DATE,
                limit=limit,
                statement_types=statement_types,
                compare_mode=compare_mode,
                metric_names=metric_names,
                active_only=not include_inactive,
            )
            summary["sec_db_path"] = str(sec_db.db_path)
            summary["wrds_db_path"] = str(wrds_db)
            summary["reference_source_kind"] = "wrds_duckdb_lake"
            self._finish_run(run, "completed", summary=summary)
            self._write_manifest(run.run_id, summary)
            return summary
        except Exception as exc:  # noqa: BLE001
            self._finish_run(run, "failed", summary=summary, error_text=str(exc))
            raise

    def survey_source_access_registry(self) -> dict[str, Any]:
        self.db.init_schema()
        WRDSClient(self.settings).preflight_credentials()
        run = self._start_run("wrds_survey_source_access", ["source_access_registry"])
        summary: dict[str, Any] = {"survey_run_id": run.run_id, "rows": []}
        try:
            with WRDSClient(self.settings) as client:
                rows = self._survey_source_access_registry(client, checked_at=run.started_at)
            frame = pd.DataFrame(rows)
            if not frame.empty:
                self.db.merge_dataframe("wrds_source_access_registry", frame, ("source_key",))
            summary["rows"] = self._json_safe_payload(rows)
            self._finish_run(run, "completed", summary=summary)
            self._write_manifest(run.run_id, summary)
            return summary
        except Exception as exc:  # noqa: BLE001
            self._finish_run(run, "failed", summary=summary, error_text=str(exc))
            raise

    def survey_kr_availability(self, *, db_path: str | Path | None = None) -> dict[str, Any]:
        builder = WRDSKRLakeBuilder(
            self.settings,
            db_path=Path(db_path).expanduser() if db_path is not None else WRDS_KR_LAKE_DB_PATH,
            logger=self.logger,
        )
        return builder.survey_availability()

    def build_kr_lake(self, *, db_path: str | Path | None = None) -> dict[str, Any]:
        builder = WRDSKRLakeBuilder(
            self.settings,
            db_path=Path(db_path).expanduser() if db_path is not None else WRDS_KR_LAKE_DB_PATH,
            logger=self.logger,
        )
        return builder.build()

    def compare_sec(
        self,
        *,
        tickers: str | list[str] | None = None,
        market: str = "us",
        limit: int | None = None,
        start_date: date | None = None,
        statement_types: tuple[str, ...] = ("quarterly", "annual"),
        compare_mode: str = "default",
        metric_names: tuple[str, ...] | None = None,
        include_inactive: bool = False,
    ) -> dict[str, Any]:
        self.db.init_schema()
        seed_default_metric_mapping_registry(self.db)
        run = self._start_run("wrds_compare_sec", list(statement_types))
        engine = SECVsWRDSFinancialValidationService(self.db, self.logger)
        summary: dict[str, Any] = {}
        try:
            summary = engine.compare(
                comparison_run_id=run.run_id,
                tickers=tickers,
                market=market,
                start_date=start_date or SEC_VALIDATION_START_DATE,
                limit=limit,
                statement_types=statement_types,
                compare_mode=compare_mode,
                metric_names=metric_names,
                active_only=not include_inactive,
            )
            self._finish_run(run, "completed", summary=summary)
            self._write_manifest(run.run_id, summary)
            return summary
        except Exception as exc:  # noqa: BLE001
            self._finish_run(run, "failed", summary=summary, error_text=str(exc))
            raise

    def compare_sec_report(
        self,
        *,
        group_by: str = "metric",
        comparison_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        engine = SECVsWRDSFinancialValidationService(self.db, self.logger)
        return engine.report(group_by=group_by, comparison_run_id=comparison_run_id, limit=limit)

    def compare_sec_policy_summary(
        self,
        *,
        comparison_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        engine = SECVsWRDSFinancialValidationService(self.db, self.logger)
        return engine.policy_summary(comparison_run_id=comparison_run_id, limit=limit)

    def compare_sec_segments(
        self,
        *,
        tickers: str | list[str] | None = None,
        market: str = "us",
        start_date: date | None = None,
    ) -> dict[str, Any]:
        self.db.init_schema()
        seed_default_segment_metric_policy_registry(self.db)
        run = self._start_run("wrds_compare_sec_segments", ["quarterly_segments"])
        engine = SECSegmentValidationService(self.db, self.logger)
        summary: dict[str, Any] = {}
        try:
            summary = engine.compare(
                comparison_run_id=run.run_id,
                tickers=tickers,
                market=market,
                start_date=start_date or SEC_VALIDATION_START_DATE,
            )
            self._finish_run(run, "completed", summary=summary)
            self._write_manifest(run.run_id, summary)
            return summary
        except Exception as exc:  # noqa: BLE001
            self._finish_run(run, "failed", summary=summary, error_text=str(exc))
            raise

    def compare_sec_segments_availability(
        self,
        *,
        tickers: str | list[str] | None = None,
        market: str = "us",
        start_date: date | None = None,
    ) -> dict[str, Any]:
        self.db.init_schema()
        seed_default_segment_metric_policy_registry(self.db)
        seed_default_segment_cluster_policy_registry(self.db)
        run = self._start_run("wrds_compare_sec_segments_availability", ["quarterly_segments"])
        engine = SECSegmentValidationService(self.db, self.logger)
        summary: dict[str, Any] = {}
        try:
            summary = engine.survey_source_availability(
                survey_run_id=run.run_id,
                tickers=tickers,
                market=market,
                start_date=start_date or SEC_VALIDATION_START_DATE,
            )
            self._finish_run(run, "completed", summary=summary)
            self._write_manifest(run.run_id, summary)
            return summary
        except Exception as exc:  # noqa: BLE001
            self._finish_run(run, "failed", summary=summary, error_text=str(exc))
            raise

    def compare_sec_segments_report(
        self,
        *,
        group_by: str = "metric",
        comparison_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        engine = SECSegmentValidationService(self.db, self.logger)
        return engine.report(group_by=group_by, comparison_run_id=comparison_run_id, limit=limit)

    def compare_sec_segments_policy_summary(
        self,
        *,
        comparison_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        engine = SECSegmentValidationService(self.db, self.logger)
        return engine.policy_summary(comparison_run_id=comparison_run_id, limit=limit)

    def compare_sec_segments_availability_summary(
        self,
        *,
        survey_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        engine = SECSegmentValidationService(self.db, self.logger)
        return engine.availability_summary(survey_run_id=survey_run_id, limit=limit)

    def compare_sec_segments_cluster_summary(
        self,
        *,
        survey_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        engine = SECSegmentValidationService(self.db, self.logger)
        return engine.cluster_summary(survey_run_id=survey_run_id, limit=limit)

    def compare_sec_segments_recovery_summary(
        self,
        *,
        survey_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        engine = SECSegmentValidationService(self.db, self.logger)
        return engine.recovery_summary(survey_run_id=survey_run_id, limit=limit)

    def survey_wrds_aux_metric_readiness(
        self,
        *,
        tickers: str | list[str],
        start_date: date | None = None,
    ) -> dict[str, Any]:
        self.db.init_schema()
        WRDSClient(self.settings).preflight_credentials()
        basket = [ticker.upper() for ticker in self._split_csv(tickers)]
        run = self._start_run("wrds_survey_aux_metric_readiness", basket)
        summary: dict[str, Any] = {"survey_run_id": run.run_id, "rows": []}
        try:
            gvkey_map = self._wrds_aux_ticker_gvkey_map(basket)
            if not gvkey_map:
                summary["rows"] = []
                summary["blocker"] = "no_local_gvkey_mapping"
            else:
                with WRDSClient(self.settings) as client:
                    rows = self._run_wrds_aux_readiness_survey(
                        client=client,
                        survey_run_id=run.run_id,
                        basket=basket,
                        ticker_gvkeys=gvkey_map,
                        start_date=start_date or SEC_VALIDATION_START_DATE,
                    )
                frame = pd.DataFrame(rows)
                if not frame.empty:
                    self.db.merge_dataframe(
                        "wrds_aux_metric_readiness_summary",
                        frame,
                        ("readiness_result_id",),
                    )
                summary["rows"] = self._json_safe_payload(rows)
            self._finish_run(run, "completed", summary=summary)
            self._write_manifest(run.run_id, summary)
            return summary
        except Exception as exc:  # noqa: BLE001
            self._finish_run(run, "failed", summary=summary, error_text=str(exc))
            raise

    def _ingest_source_dataset(self, run: RunContext, client: WRDSClient, dataset_name: str) -> dict[str, Any]:
        spec = self.specs[dataset_name]
        windows = self._build_windows(
            spec.name,
            spec.chunk_months,
            spec.full_refresh_only,
            use_checkpoints=not self.settings.dry_run,
            resolved_end_date=self._effective_end_date(dataset_name, client),
        )
        completed_keys = self.db.checkpoint_completed_keys(spec.name) if spec.chunk_months and not self.settings.dry_run else set()
        total_fetched = 0
        total_written = 0
        chunks: list[dict[str, Any]] = []

        for window in windows:
            chunk_key = window.key if window is not None else "full_refresh"
            if spec.chunk_months and not self.settings.force and chunk_key in completed_keys:
                self.logger.info("Skipping completed chunk %s for %s", chunk_key, dataset_name)
                continue

            if not self.settings.dry_run:
                self._upsert_checkpoint(
                    dataset_name=dataset_name,
                    chunk_key=chunk_key,
                    status="running",
                    run_id=run.run_id,
                    chunk_start=window.start if window else None,
                    chunk_end=window.end if window else None,
                    rows_fetched=0,
                    rows_written=0,
                    notes=None,
                )

            sql, params = self._build_dataset_sql(spec, dataset_name, window)
            source_relation = self._source_relation_for_dataset(dataset_name)
            try:
                result = client.raw_sql(sql, params=params)
            except Exception as exc:  # noqa: BLE001
                if dataset_name == "ccm_link" and self._should_use_ccm_fallback(exc):
                    sql = self._build_ccm_fallback_sql(limit=self._ccm_fallback_limit())
                    if spec.full_refresh_only and self.settings.sample_mode:
                        sql = f"SELECT * FROM ({sql}) sample_sub LIMIT {self._ccm_fallback_limit() or int(self.settings.sample_row_limit)}"
                    params = {}
                    source_relation = "heuristic:comp.names+crsp.dsenames"
                    result = client.raw_sql(sql, params=params)
                else:
                    if not self.settings.dry_run:
                        self._upsert_checkpoint(
                            dataset_name=dataset_name,
                            chunk_key=chunk_key,
                            status="failed",
                            run_id=run.run_id,
                            chunk_start=window.start if window else None,
                            chunk_end=window.end if window else None,
                            rows_fetched=0,
                            rows_written=0,
                            notes=str(self._dataset_query_error(dataset_name, window, sql, exc)),
                        )
                    raise self._dataset_query_error(dataset_name, window, sql, exc) from exc
            frame = append_audit_columns(
                result.rows,
                source_relation=source_relation,
                run_id=run.run_id,
                collected_at=run.started_at,
            )

            rows_fetched = int(len(frame))
            rows_written = 0
            if not self.settings.dry_run and not frame.empty:
                rows_written = self.db.merge_dataframe(spec.target_table, frame, spec.key_columns)

            total_fetched += rows_fetched
            total_written += rows_written
            chunk_summary = {
                "chunk_key": chunk_key,
                "rows_fetched": rows_fetched,
                "rows_written": rows_written,
                "attempts": result.attempts,
                "source_relation": source_relation,
            }
            chunks.append(chunk_summary)
            if not self.settings.dry_run:
                self._upsert_checkpoint(
                    dataset_name=dataset_name,
                    chunk_key=chunk_key,
                    status="completed",
                    run_id=run.run_id,
                    chunk_start=window.start if window else None,
                    chunk_end=window.end if window else None,
                    rows_fetched=rows_fetched,
                    rows_written=rows_written,
                    notes=None,
                )
            self.logger.info(
                "WRDS chunk complete dataset=%s chunk=%s fetched=%s written=%s attempts=%s",
                dataset_name,
                chunk_key,
                rows_fetched,
                rows_written,
                result.attempts,
            )

        if self.settings.dry_run:
            validation = self._frame_profile(
                frame=result.rows if "result" in locals() else pd.DataFrame(),
                date_column=spec.date_column,
                distinct_columns=spec.summary_distinct_columns,
                key_columns=spec.key_columns,
            )
        else:
            validation = self.db.table_profile(
                spec.target_table,
                date_column=spec.date_column,
                key_columns=spec.key_columns,
                distinct_columns=spec.summary_distinct_columns,
            )
        return {
            "target_table": spec.target_table,
            "rows_fetched": total_fetched,
            "rows_written": total_written,
            "chunks": chunks,
            "validation": validation,
        }

    def _refresh_universe_membership_source(self, run: RunContext, client: WRDSClient | None) -> dict[str, Any]:
        chunk_key = "full_refresh"
        dataset_name = "universe_membership_history"
        self._upsert_checkpoint(
            dataset_name=dataset_name,
            chunk_key=chunk_key,
            status="running",
            run_id=run.run_id,
            chunk_start=self.settings.resolved_start_date(),
            chunk_end=self.settings.resolved_end_date(),
            rows_fetched=0,
            rows_written=0,
            notes=None,
        )

        exchange_rows = self._exchange_membership_rows(run)
        index_rows: list[pd.DataFrame] = []
        date_window = ChunkWindow(
            start=self.settings.resolved_start_date() or date(1950, 1, 1),
            end=self.settings.resolved_end_date(),
        )

        if client is not None and self.settings.indexes.enable_sp500:
            index_rows.append(
                self._index_membership_rows(
                    run,
                    client,
                    membership_code="SP500",
                    membership_name="S&P 500",
                    index_ids=self.settings.indexes.sp500_index_ids,
                    date_window=date_window,
                )
            )
        if client is not None and self.settings.indexes.enable_nasdaq100:
            index_rows.append(
                self._index_membership_rows(
                    run,
                    client,
                    membership_code="NASDAQ100",
                    membership_name="NASDAQ-100",
                    index_ids=self.settings.indexes.nasdaq100_index_ids,
                    date_window=date_window,
                )
            )

        frames = [exchange_rows] + [frame for frame in index_rows if frame is not None]
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        combined = combined.drop_duplicates(
            subset=["membership_type", "membership_code", "member_key", "effective_from"],
            keep="last",
        )
        rows_fetched = int(len(combined))
        rows_written = 0
        if not self.settings.dry_run:
            rows_written = self.db.replace_dataframe("wrds_universe_membership_history", combined)

        self._upsert_checkpoint(
            dataset_name=dataset_name,
            chunk_key=chunk_key,
            status="completed",
            run_id=run.run_id,
            chunk_start=date_window.start,
            chunk_end=date_window.end,
            rows_fetched=rows_fetched,
            rows_written=rows_written,
            notes="exchange membership derived from wrds_security_master; index membership from configurable WRDS adapter",
        )
        validation = self.db.table_profile(
            "wrds_universe_membership_history",
            date_column="effective_from",
            key_columns=("membership_type", "membership_code", "member_key", "effective_from"),
            distinct_columns=("membership_code", "member_key"),
        )
        return {
            "target_table": "wrds_universe_membership_history",
            "rows_fetched": rows_fetched,
            "rows_written": rows_written,
            "validation": validation,
        }

    def _refresh_source_access_registry(self, run: RunContext, client: WRDSClient | None) -> dict[str, Any]:
        chunk_key = "full_refresh"
        dataset_name = "source_access_registry"
        self._upsert_checkpoint(
            dataset_name=dataset_name,
            chunk_key=chunk_key,
            status="running",
            run_id=run.run_id,
            chunk_start=None,
            chunk_end=None,
            rows_fetched=0,
            rows_written=0,
            notes=None,
        )
        rows = self._survey_source_access_registry(client, checked_at=run.started_at)
        frame = pd.DataFrame(rows)
        rows_fetched = int(len(frame))
        rows_written = 0
        if not self.settings.dry_run and not frame.empty:
            rows_written = self.db.replace_dataframe("wrds_source_access_registry", frame)
        self._upsert_checkpoint(
            dataset_name=dataset_name,
            chunk_key=chunk_key,
            status="completed",
            run_id=run.run_id,
            chunk_start=None,
            chunk_end=None,
            rows_fetched=rows_fetched,
            rows_written=rows_written,
            notes="WRDS relation accessibility and subscription surface summary",
        )
        validation = self.db.table_profile(
            "wrds_source_access_registry",
            date_column="checked_at",
            key_columns=("source_key",),
            distinct_columns=("dataset_group", "relation_name", "access_status"),
        )
        return {
            "target_table": "wrds_source_access_registry",
            "rows_fetched": rows_fetched,
            "rows_written": rows_written,
            "validation": validation,
        }

    def _survey_source_access_registry(
        self,
        client: WRDSClient | None,
        *,
        checked_at: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        sources = [
            ("compustat_fundamentals", "comp", self.settings.relations.compustat_quarterly),
            ("compustat_fundamentals", "comp", self.settings.relations.compustat_annual),
            ("compustat_segments", "comp", self.settings.relations.compustat_segments_merged),
            ("compustat_segments", "comp", self.settings.relations.compustat_segments_geo),
            ("compustat_segments", "comp", self.settings.relations.compustat_segments_product),
            ("compustat_segments", "comp", self.settings.relations.compustat_segments_customer),
            ("compustat_pit_like", "comp", self.settings.relations.compustat_quarterly_ytd),
            ("compustat_pit_like", "comp", self.settings.relations.compustat_quarterly_semi),
            ("compustat_pit_like", "comp", self.settings.relations.compustat_quarterly_flags),
            ("compustat_pit_like", "comp", self.settings.relations.compustat_security_quarterly),
            ("compustat_pit_like", "comp", self.settings.relations.compustat_security_quarterly_flags),
            ("compustat_status", "comp", self.settings.relations.compustat_quarterly_fncd),
            ("compustat_status", "comp", self.settings.relations.compustat_annual_fncd),
            ("ibes_estimates", "ibes", self.settings.relations.ibes_actuals_epsus),
            ("ibes_estimates", "ibes", self.settings.relations.ibes_summary_epsus),
            ("ibes_guidance", "ibes", self.settings.relations.ibes_guidance),
            ("sec_13f", "sec13f", "sec13f.holdings"),
            ("sec_13f", "tr_13f", "tr_13f.holdings"),
        ]
        if client is None:
            return [
                {
                    "source_key": f"{group}:{relation}",
                    "dataset_group": group,
                    "library_name": library,
                    "relation_name": relation,
                    "access_status": "not_checked",
                    "notes": "WRDS client unavailable for access survey",
                    "checked_at": checked_at,
                }
                for group, library, relation in sources
            ]

        rows: list[dict[str, Any]] = []
        for dataset_group, library_name, relation_name in sources:
            try:
                client.raw_sql(f"SELECT * FROM {relation_name} LIMIT 1")
                access_status = "available"
                notes = ""
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                if "insufficientprivilege" in message or "permission denied" in message:
                    access_status = "permission_denied"
                elif "relation" in message and "does not exist" in message:
                    access_status = "not_found"
                else:
                    access_status = "unavailable"
                notes = str(exc)
            rows.append(
                {
                    "source_key": f"{dataset_group}:{relation_name}",
                    "dataset_group": dataset_group,
                    "library_name": library_name,
                    "relation_name": relation_name,
                    "access_status": access_status,
                    "notes": notes,
                    "checked_at": checked_at,
                }
            )
        return rows

    def _exchange_membership_rows(self, run: RunContext) -> pd.DataFrame:
        sql = """
            SELECT
                'exchange' AS membership_type,
                CASE
                    WHEN exchcd = 1 THEN 'NYSE'
                    WHEN exchcd = 3 THEN 'NASDAQ'
                    ELSE NULL
                END AS membership_code,
                CASE
                    WHEN exchcd = 1 THEN 'New York Stock Exchange'
                    WHEN exchcd = 3 THEN 'NASDAQ'
                    ELSE NULL
                END AS membership_name,
                CONCAT('PERMNO:', CAST(permno AS VARCHAR)) AS member_key,
                'exchange' AS source_kind,
                permno,
                permco,
                NULL::VARCHAR AS gvkey,
                NULL::VARCHAR AS iid,
                ticker,
                COALESCE(ncusip, cusip) AS cusip,
                NULL::VARCHAR AS cik,
                exchcd AS exchange_code,
                namedt AS effective_from,
                nameenddt AS effective_to,
                'wrds_security_master' AS source_relation,
                'derived_exchange_membership' AS source_query_name,
                'Derived from CRSP security history exchange codes; exchcd=1->NYSE, exchcd=3->NASDAQ' AS assumptions
            FROM wrds_security_master
            WHERE exchcd IN (1, 3)
              AND namedt IS NOT NULL
        """
        frame = self.db.fetch_df(sql)
        return append_audit_columns(
            frame,
            source_relation="wrds_security_master",
            run_id=run.run_id,
            collected_at=run.started_at,
        )

    def _index_membership_rows(
        self,
        run: RunContext,
        client: WRDSClient,
        *,
        membership_code: str,
        membership_name: str,
        index_ids: tuple[str, ...],
        date_window: ChunkWindow,
    ) -> pd.DataFrame:
        sql, params = build_index_membership_sql(
            self.settings,
            membership_code=membership_code,
            membership_name=membership_name,
            index_ids=index_ids,
            window=date_window,
        )
        if "SELECT NULL WHERE FALSE" in sql:
            return pd.DataFrame()
        try:
            result = client.raw_sql(sql, params=params)
        except Exception as exc:  # noqa: BLE001
            raise self._dataset_query_error("universe_membership_history", date_window, sql, exc) from exc
        frame = result.rows.copy()
        if frame.empty:
            return append_audit_columns(
                pd.DataFrame(
                    columns=[
                        "membership_type",
                        "membership_code",
                        "membership_name",
                        "member_key",
                        "source_kind",
                        "permno",
                        "permco",
                        "gvkey",
                        "iid",
                        "ticker",
                        "cusip",
                        "cik",
                        "exchange_code",
                        "effective_from",
                        "effective_to",
                        "source_query_name",
                        "assumptions",
                    ]
                ),
                source_relation=self.settings.relations.index_membership,
                run_id=run.run_id,
                collected_at=run.started_at,
            )

        frame["source_kind"] = "index"
        frame["permno"] = None
        frame["permco"] = None
        frame["cusip"] = None
        frame["cik"] = None
        frame["exchange_code"] = None
        frame["source_query_name"] = f"{membership_code.lower()}_membership_adapter"
        frame["assumptions"] = (
            "Configurable WRDS index adapter. Adjust relation/SQL if your WRDS subscription exposes a different table."
        )
        frame["member_key"] = frame.apply(self._member_key_from_row, axis=1)
        return append_audit_columns(
            frame,
            source_relation=self.settings.relations.index_membership,
            run_id=run.run_id,
            collected_at=run.started_at,
        )

    @staticmethod
    def _member_key_from_row(row: pd.Series) -> str:
        if pd.notna(row.get("permno")):
            return f"PERMNO:{int(row['permno'])}"
        gvkey = row.get("gvkey")
        iid = row.get("iid")
        ticker = row.get("ticker")
        if pd.notna(gvkey) and pd.notna(iid):
            return f"GVKEY_IID:{gvkey}:{iid}"
        if pd.notna(gvkey):
            return f"GVKEY:{gvkey}"
        return f"TICKER:{ticker}"

    def _build_windows(
        self,
        dataset_name: str,
        chunk_months: int | None,
        full_refresh_only: bool,
        *,
        use_checkpoints: bool = True,
        resolved_end_date: date | None = None,
    ) -> list[ChunkWindow | None]:
        if full_refresh_only or chunk_months is None:
            return [None]

        end = resolved_end_date or self.settings.resolved_end_date()
        start = self.settings.resolved_start_date() or DEFAULT_START_DATES[dataset_name]
        if self.settings.sample_mode and self.settings.start_date is None:
            lookback_days = SAMPLE_LOOKBACK_DAYS.get(dataset_name)
            if lookback_days is not None:
                start = max(DEFAULT_START_DATES[dataset_name], end - timedelta(days=lookback_days))
        latest_completed = self.db.checkpoint_latest_end(dataset_name) if use_checkpoints else None
        if latest_completed is not None and not self.settings.force:
            start = max(start, latest_completed + timedelta(days=1))
        if start > end:
            return []

        windows: list[ChunkWindow] = []
        cursor = pd.Timestamp(start)
        final_end = pd.Timestamp(end)
        while cursor <= final_end:
            next_end = min(cursor + pd.DateOffset(months=chunk_months) - pd.DateOffset(days=1), final_end)
            windows.append(ChunkWindow(start=cursor.date(), end=next_end.date()))
            cursor = next_end + pd.DateOffset(days=1)
        return windows

    def _frame_profile(
        self,
        *,
        frame: pd.DataFrame,
        date_column: str | None,
        distinct_columns: tuple[str, ...],
        key_columns: tuple[str, ...],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"row_count": int(len(frame))}
        if frame.empty:
            payload["distinct_counts"] = {column: 0 for column in distinct_columns}
            payload["null_ratios"] = {column: 0.0 for column in key_columns}
            payload["duplicate_primary_key_rows"] = 0
            payload["schema"] = [{"column_name": column, "column_type": str(dtype)} for column, dtype in frame.dtypes.items()]
            return payload
        if date_column and date_column in frame.columns:
            series = pd.to_datetime(frame[date_column], errors="coerce")
            payload["min_date"] = series.min().date().isoformat() if series.notna().any() else None
            payload["max_date"] = series.max().date().isoformat() if series.notna().any() else None
        payload["distinct_counts"] = {
            column: int(frame[column].nunique(dropna=True)) if column in frame.columns else 0
            for column in distinct_columns
        }
        payload["null_ratios"] = {
            column: float(frame[column].isna().mean()) if column in frame.columns else 1.0
            for column in key_columns
        }
        if key_columns and all(column in frame.columns for column in key_columns):
            dupes = frame.duplicated(subset=list(key_columns), keep=False).sum()
            payload["duplicate_primary_key_rows"] = int(dupes)
        else:
            payload["duplicate_primary_key_rows"] = 0
        payload["schema"] = [{"column_name": column, "column_type": str(dtype)} for column, dtype in frame.dtypes.items()]
        return payload

    def _build_dataset_sql(
        self,
        spec: Any,
        dataset_name: str,
        window: ChunkWindow | None,
    ) -> tuple[str, dict[str, Any]]:
        sql, params = spec.query_builder(self.settings, window)
        if spec.full_refresh_only and self.settings.sample_mode:
            sql = f"SELECT * FROM ({sql}) sample_sub LIMIT {int(self.settings.sample_row_limit)}"
        return sql, params

    def _dataset_query_error(
        self,
        dataset_name: str,
        window: ChunkWindow | None,
        sql: str,
        exc: Exception,
    ) -> WRDSDatasetQueryError:
        if isinstance(exc, WRDSDatasetQueryError):
            return exc
        if isinstance(exc, WRDSExecutionError):
            exception_class = exc.exception_class
            exception_message = exc.exception_message
        else:
            exception_class = exc.__class__.__name__
            exception_message = str(exc)
        return WRDSDatasetQueryError(
            dataset_name=dataset_name,
            relation_name=self._source_relation_for_dataset(dataset_name),
            sql_template_name=self._sql_template_name(dataset_name),
            sql_text=sql,
            chunk_window=self._window_payload(window),
            exception_class=exception_class,
            exception_message=exception_message,
        )

    @staticmethod
    def _window_payload(window: ChunkWindow | None) -> dict[str, Any]:
        if window is None:
            return {"chunk_key": "full_refresh", "start": None, "end": None}
        return {
            "chunk_key": window.key,
            "start": window.start.isoformat() if window.start else None,
            "end": window.end.isoformat() if window.end else None,
        }

    @staticmethod
    def _split_csv(values: str | list[str]) -> list[str]:
        if isinstance(values, str):
            return [item.strip() for item in values.split(",") if item.strip()]
        return [str(item).strip() for item in values if str(item).strip()]

    def _requires_wrds_client(self, dataset_names: list[str]) -> bool:
        if any(name != "universe_membership_history" for name in dataset_names):
            return True
        return bool(self.settings.indexes.enable_sp500 or self.settings.indexes.enable_nasdaq100)

    def _resolve_inspection_target(
        self,
        dataset: str,
        *,
        layer: str,
    ) -> tuple[str, str | None, tuple[str, ...], tuple[str, ...], str]:
        if layer in {"auto", "source"} and dataset in self.specs:
            spec = self.specs[dataset]
            return spec.target_table, spec.date_column, spec.key_columns, spec.summary_distinct_columns, "source"
        if layer in {"auto", "canonical"} and dataset in CANONICAL_PROFILE_CONFIG:
            config = CANONICAL_PROFILE_CONFIG[dataset]
            return (
                dataset,
                config["date_column"],
                SCHEMA_BY_TABLE[dataset].primary_key,
                config["distinct_columns"],
                "canonical",
            )
        if layer == "source":
            raise ValueError(f"Unknown WRDS source dataset: {dataset}")
        if layer == "canonical":
            raise ValueError(f"Unknown WRDS canonical table: {dataset}")
        raise ValueError(f"Unknown inspect target: {dataset}")

    def _resolve_canonical_tables(self, tables: str | list[str] | tuple[str, ...]) -> list[str]:
        if isinstance(tables, str):
            tokens = [item.strip() for item in tables.split(",") if item.strip()]
        else:
            tokens = [str(item).strip() for item in tables if str(item).strip()]
        if not tokens or tokens == ["all"]:
            return list(CANONICAL_BUILD_ORDER)
        invalid = [token for token in tokens if token not in CANONICAL_BUILD_ORDER]
        if invalid:
            raise ValueError(f"Unknown canonical tables: {', '.join(invalid)}")
        return tokens

    def _source_relation_for_dataset(self, dataset_name: str) -> str:
        mapping = {
            "crsp_daily": f"{self.settings.relations.crsp_daily}|{self.settings.relations.crsp_delist}",
            "compustat_quarterly": self.settings.relations.compustat_quarterly,
            "compustat_annual": self.settings.relations.compustat_annual,
            "compustat_segments_historical": self.settings.relations.compustat_segments_merged,
            "compustat_quarterly_variant_metrics": (
                f"{self.settings.relations.compustat_quarterly_ytd}|{self.settings.relations.compustat_quarterly_semi}|"
                f"{self.settings.relations.compustat_quarterly_flags}|{self.settings.relations.compustat_security_quarterly}|"
                f"{self.settings.relations.compustat_security_quarterly_flags}"
            ),
            "ibes_actuals_epsus": self.settings.relations.ibes_actuals_epsus,
            "ibes_summary_epsus": self.settings.relations.ibes_summary_epsus,
            "source_access_registry": "wrds_accessibility_probe",
            "ccm_link": self.settings.relations.ccm_link,
            "security_master": self.settings.relations.security_master,
            "company_master": f"{self.settings.relations.company_master}|comp.names",
            "universe_membership_history": self.settings.relations.index_membership,
        }
        return mapping[dataset_name]

    def _wrds_aux_ticker_gvkey_map(self, tickers: list[str]) -> dict[str, str]:
        if not tickers:
            return {}
        placeholders = ", ".join(["?"] * len(tickers))
        df = self.db.fetch_df(
            f"""
            SELECT UPPER(ticker) AS ticker, gvkey
            FROM (
                SELECT tic AS ticker, gvkey
                FROM wrds_company_master
                WHERE tic IS NOT NULL AND gvkey IS NOT NULL
                UNION ALL
                SELECT ticker, gvkey
                FROM financials_quarterly_canonical
                WHERE ticker IS NOT NULL AND gvkey IS NOT NULL
            ) src
            WHERE UPPER(ticker) IN ({placeholders})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY UPPER(ticker) ORDER BY gvkey) = 1
            """,
            tickers,
        )
        if df.empty:
            return {}
        return {str(row["ticker"]).upper(): str(row["gvkey"]) for _, row in df.iterrows() if pd.notna(row["ticker"]) and pd.notna(row["gvkey"])}

    def _run_wrds_aux_readiness_survey(
        self,
        *,
        client: WRDSClient,
        survey_run_id: str,
        basket: list[str],
        ticker_gvkeys: dict[str, str],
        start_date: date,
    ) -> list[dict[str, Any]]:
        gvkeys = sorted({str(value) for value in ticker_gvkeys.values() if value})
        if not gvkeys:
            return []
        gvkey_to_tickers: dict[str, set[str]] = {}
        for ticker, gvkey in ticker_gvkeys.items():
            gvkey_to_tickers.setdefault(str(gvkey), set()).add(str(ticker).upper())
        gvkey_sql = ", ".join("'" + gvkey.replace("'", "''") + "'" for gvkey in gvkeys)
        candidates: list[dict[str, Any]] = [
            {
                "metric_name": "sbc",
                "statement_type": "quarterly",
                "sources": [
                    ("comp.fundq", "stkcoq"),
                    ("comp.co_ifndq", "stkcoq"),
                    ("comp.bank_fundq", "stkcoq"),
                ],
            },
            {
                "metric_name": "sbc",
                "statement_type": "annual",
                "sources": [
                    ("comp.funda", "stkco"),
                    ("comp.bank_funda", "stkco"),
                ],
            },
            {
                "metric_name": "apic",
                "statement_type": "quarterly",
                "sources": [
                    ("comp.fundq", "ucapsq"),
                    ("comp.co_ifndq", "ucapsq"),
                ],
            },
            {
                "metric_name": "apic",
                "statement_type": "annual",
                "sources": [
                    ("comp.funda", "ucaps"),
                ],
            },
        ]
        created_at = pd.Timestamp.utcnow()
        rows: list[dict[str, Any]] = []
        for spec in candidates:
            candidate_summaries: list[dict[str, Any]] = []
            best: dict[str, Any] | None = None
            for table_name, column_name in spec["sources"]:
                sql = f"""
                    SELECT
                        gvkey,
                        datadate::date AS datadate,
                        {column_name} AS candidate_value
                    FROM {table_name}
                    WHERE gvkey IN ({gvkey_sql})
                      AND datadate >= '{start_date.isoformat()}'
                """
                try:
                    result = client.raw_sql(sql)
                    frame = result.rows if result.rows is not None else pd.DataFrame()
                except Exception as exc:  # noqa: BLE001
                    candidate_summaries.append(
                        {
                            "table": table_name,
                            "column": column_name,
                            "error": exc.__class__.__name__,
                            "message": str(exc),
                        }
                    )
                    continue
                if frame.empty:
                    candidate_summaries.append(
                        {
                            "table": table_name,
                            "column": column_name,
                            "row_count": 0,
                            "issuer_count": 0,
                            "non_null_rows": 0,
                            "non_null_issuers": 0,
                        }
                    )
                    continue
                frame["candidate_value"] = pd.to_numeric(frame["candidate_value"], errors="coerce")
                non_null = frame.loc[frame["candidate_value"].notna()].copy()
                non_null_pairs: set[tuple[str, date]] = set()
                for _, item in non_null.iterrows():
                    gvkey = str(item.get("gvkey", "") or "")
                    datadate = pd.to_datetime(item.get("datadate"), errors="coerce")
                    if not gvkey or pd.isna(datadate):
                        continue
                    for ticker in gvkey_to_tickers.get(gvkey, set()):
                        non_null_pairs.add((ticker, datadate.date()))
                summary = {
                    "table": table_name,
                    "column": column_name,
                    "row_count": int(len(frame)),
                    "issuer_count": int(frame["gvkey"].nunique()),
                    "non_null_rows": int(len(non_null)),
                    "non_null_issuers": int(non_null["gvkey"].nunique()) if not non_null.empty else 0,
                    "coverage_start": pd.to_datetime(frame["datadate"], errors="coerce").min(),
                    "coverage_end": pd.to_datetime(frame["datadate"], errors="coerce").max(),
                    "non_null_pairs": non_null_pairs,
                    "non_null_tickers": sorted({ticker for ticker, _ in non_null_pairs}),
                }
                candidate_summaries.append(summary)
                if best is None or summary["non_null_rows"] > best["non_null_rows"]:
                    best = summary
            sec_coverage = self._build_aux_sec_coverage(
                metric_name=str(spec["metric_name"]),
                statement_type=str(spec["statement_type"]),
                basket=basket,
                start_date=start_date,
            )
            row = self._build_aux_readiness_row(
                survey_run_id=survey_run_id,
                metric_name=str(spec["metric_name"]),
                statement_type=str(spec["statement_type"]),
                candidate_summaries=candidate_summaries,
                sec_coverage=sec_coverage,
                created_at=created_at,
            )
            rows.append(row)
        return rows

    def _build_aux_readiness_row(
        self,
        *,
        survey_run_id: str,
        metric_name: str,
        statement_type: str,
        candidate_summaries: list[dict[str, Any]],
        sec_coverage: dict[str, Any],
        created_at: pd.Timestamp,
    ) -> dict[str, Any]:
        successful = [item for item in candidate_summaries if "error" not in item]
        best = max(successful, key=lambda item: int(item.get("non_null_rows", 0)), default=None)
        coverage_note = json.dumps(candidate_summaries, ensure_ascii=True, default=str)
        if best is None or int(best.get("non_null_rows", 0)) <= 0:
            blocker_type = "wrds_reference_missing"
            blocker_reason = "No non-null WRDS auxiliary candidate rows were found for the selected basket/date window."
            next_action = "keep_deferred_until_wrds_aux_source_confirmed"
            candidate_name = None
            table_name = None
            column_name = None
            row_count = issuer_count = non_null_rows = non_null_issuers = 0
            coverage_start = coverage_end = None
            candidate_overlap_rows = 0
            candidate_overlap_issuers = 0
            readiness_class = "reference_missing"
            pilot_compare_ready = False
        else:
            table_name = str(best["table"])
            column_name = str(best["column"])
            candidate_name = f"{table_name}.{column_name}"
            row_count = int(best.get("row_count", 0))
            issuer_count = int(best.get("issuer_count", 0))
            non_null_rows = int(best.get("non_null_rows", 0))
            non_null_issuers = int(best.get("non_null_issuers", 0))
            coverage_start = pd.to_datetime(best.get("coverage_start"), errors="coerce")
            coverage_end = pd.to_datetime(best.get("coverage_end"), errors="coerce")
            candidate_pairs = {
                (str(ticker).upper(), pd.to_datetime(period_end, errors="coerce").date())
                for ticker, period_end in best.get("non_null_pairs", set())
                if ticker and pd.notna(pd.to_datetime(period_end, errors="coerce"))
            }
            sec_pairs = set(sec_coverage.get("sec_any_pairs", set()))
            overlap_pairs = candidate_pairs & sec_pairs
            candidate_overlap_rows = len(overlap_pairs)
            candidate_overlap_issuers = len({ticker for ticker, _ in overlap_pairs})
            if metric_name == "sbc":
                blocker_type = "wrds_aux_candidate_needs_validation"
                blocker_reason = "WRDS auxiliary SBC candidate exists, but canonical onboarding and semantics validation are still pending."
                next_action = "prototype_wrds_aux_sbc_compare"
                readiness_class = (
                    "usable_for_pilot_compare"
                    if candidate_overlap_issuers >= 10
                    and int(sec_coverage.get("sec_any_non_null_issuers", 0)) >= 10
                    else "needs_validation"
                )
                pilot_compare_ready = readiness_class == "usable_for_pilot_compare"
            else:
                blocker_type = "wrds_aux_candidate_semantics_risk"
                blocker_reason = "WRDS paid-in-capital candidate exists, but semantics may not match isolated APIC and requires validation before compare activation."
                next_action = "validate_ucaps_semantics_before_compare"
                readiness_class = "deferred_semantics_risk"
                pilot_compare_ready = False
        payload = {
            "survey_run_id": survey_run_id,
            "metric_name": metric_name,
            "statement_type": statement_type,
            "wrds_reference_candidate": candidate_name,
            "wrds_reference_table": table_name,
            "wrds_reference_column": column_name,
            "candidate_row_count": row_count,
            "candidate_issuer_count": issuer_count,
            "candidate_non_null_rows": non_null_rows,
            "candidate_non_null_issuers": non_null_issuers,
            "sec_base_non_null_rows": int(sec_coverage.get("sec_base_non_null_rows", 0)),
            "sec_base_non_null_issuers": int(sec_coverage.get("sec_base_non_null_issuers", 0)),
            "sec_extra_non_null_rows": int(sec_coverage.get("sec_extra_non_null_rows", 0)),
            "sec_extra_non_null_issuers": int(sec_coverage.get("sec_extra_non_null_issuers", 0)),
            "sec_raw_non_null_rows": int(sec_coverage.get("sec_raw_non_null_rows", 0)),
            "sec_raw_non_null_issuers": int(sec_coverage.get("sec_raw_non_null_issuers", 0)),
            "candidate_overlap_rows": candidate_overlap_rows,
            "candidate_overlap_issuers": candidate_overlap_issuers,
            "coverage_start_date": coverage_start.date() if pd.notna(coverage_start) else None,
            "coverage_end_date": coverage_end.date() if pd.notna(coverage_end) else None,
            "coverage_note": coverage_note,
            "blocker_type": blocker_type,
            "blocker_reason": blocker_reason,
            "readiness_class": readiness_class,
            "pilot_compare_ready": pilot_compare_ready,
            "recommended_next_action": next_action,
            "created_at": created_at,
        }
        payload["readiness_result_id"] = uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(
                {
                    "survey_run_id": survey_run_id,
                    "metric_name": metric_name,
                    "statement_type": statement_type,
                },
                sort_keys=True,
                ensure_ascii=True,
            ),
        ).hex
        return payload

    def _build_aux_sec_coverage(
        self,
        *,
        metric_name: str,
        statement_type: str,
        basket: list[str],
        start_date: date,
    ) -> dict[str, Any]:
        metric_key = str(metric_name).strip().lower()
        basket_upper = [str(ticker).upper() for ticker in basket if str(ticker).strip()]
        if not basket_upper:
            return {
                "sec_base_non_null_rows": 0,
                "sec_base_non_null_issuers": 0,
                "sec_extra_non_null_rows": 0,
                "sec_extra_non_null_issuers": 0,
                "sec_raw_non_null_rows": 0,
                "sec_raw_non_null_issuers": 0,
                "sec_any_non_null_issuers": 0,
                "sec_any_pairs": set(),
            }
        tickers_sql = ", ".join("'" + ticker.replace("'", "''") + "'" for ticker in basket_upper)
        mapping = {
            "sbc": {
                "base_table": "financials_quarterly",
                "base_column": '"SBC"',
                "base_date": '"PeriodEnd"',
                "extra_table": "financials_quarterly_extra",
                "extra_column": "sbc",
                "extra_date": "period_end",
                "raw_fact_name": "SBC",
            },
            "apic": {
                "base_table": "financials_quarterly",
                "base_column": '"APIC"',
                "base_date": '"PeriodEnd"',
                "extra_table": "financials_quarterly_extra",
                "extra_column": "additional_paid_in_capital",
                "extra_date": "period_end",
                "raw_fact_name": "APIC",
            },
        }
        spec = mapping.get(metric_key)
        if spec is None:
            return {
                "sec_base_non_null_rows": 0,
                "sec_base_non_null_issuers": 0,
                "sec_extra_non_null_rows": 0,
                "sec_extra_non_null_issuers": 0,
                "sec_raw_non_null_rows": 0,
                "sec_raw_non_null_issuers": 0,
                "sec_any_non_null_issuers": 0,
                "sec_any_pairs": set(),
            }

        def _fetch_pairs(table: str, value_column: str, date_column: str, *, extra_filter: str = "") -> tuple[set[tuple[str, date]], int, int]:
            if not self.db.table_exists(table):
                return set(), 0, 0
            frame = self.db.fetch_df(
                f"""
                SELECT UPPER(ticker) AS ticker, CAST({date_column} AS DATE) AS period_end
                FROM {table}
                WHERE UPPER(ticker) IN ({tickers_sql})
                  AND CAST({date_column} AS DATE) >= ?
                  AND {value_column} IS NOT NULL
                  {extra_filter}
                """,
                [start_date],
            )
            pairs = {
                (str(row["ticker"]).upper(), pd.to_datetime(row["period_end"], errors="coerce").date())
                for _, row in frame.iterrows()
                if row.get("ticker") is not None and pd.notna(pd.to_datetime(row.get("period_end"), errors="coerce"))
            }
            return pairs, int(len(frame)), int(frame["ticker"].nunique()) if not frame.empty else 0

        base_pairs, base_rows, base_issuers = _fetch_pairs(
            str(spec["base_table"]),
            str(spec["base_column"]),
            str(spec["base_date"]),
        )
        extra_pairs, extra_rows, extra_issuers = _fetch_pairs(
            str(spec["extra_table"]),
            str(spec["extra_column"]),
            str(spec["extra_date"]),
        )
        raw_pairs, raw_rows, raw_issuers = set(), 0, 0
        if self.db.table_exists("sec_facts_raw_normalized"):
            raw_frame = self.db.fetch_df(
                """
                SELECT UPPER(ticker) AS ticker, CAST(period_end AS DATE) AS period_end
                FROM sec_facts_raw_normalized
                WHERE UPPER(ticker) IN ("""
                + tickers_sql
                + """
                )
                  AND CAST(period_end AS DATE) >= ?
                  AND LOWER(fact_name) = ?
                  AND value IS NOT NULL
                """,
                [start_date, str(spec["raw_fact_name"]).lower()],
            )
            raw_pairs = {
                (str(row["ticker"]).upper(), pd.to_datetime(row["period_end"], errors="coerce").date())
                for _, row in raw_frame.iterrows()
                if row.get("ticker") is not None and pd.notna(pd.to_datetime(row.get("period_end"), errors="coerce"))
            }
            raw_rows = int(len(raw_frame))
            raw_issuers = int(raw_frame["ticker"].nunique()) if not raw_frame.empty else 0

        sec_any_pairs = base_pairs | extra_pairs | raw_pairs
        sec_any_non_null_issuers = len({ticker for ticker, _ in sec_any_pairs})
        return {
            "statement_type": statement_type,
            "sec_base_non_null_rows": base_rows,
            "sec_base_non_null_issuers": base_issuers,
            "sec_extra_non_null_rows": extra_rows,
            "sec_extra_non_null_issuers": extra_issuers,
            "sec_raw_non_null_rows": raw_rows,
            "sec_raw_non_null_issuers": raw_issuers,
            "sec_any_non_null_issuers": sec_any_non_null_issuers,
            "sec_any_pairs": sec_any_pairs,
        }

    @staticmethod
    def _json_safe_payload(payload: Any) -> Any:
        return json.loads(json.dumps(payload, ensure_ascii=False, default=str))

    @staticmethod
    def _sql_template_name(dataset_name: str) -> str:
        return f"build_{dataset_name}_sql"

    def _effective_end_date(self, dataset_name: str, client: WRDSClient) -> date:
        configured_end = self.settings.resolved_end_date()
        if self.settings.end_date is not None:
            return configured_end
        source_max = self._source_max_date(dataset_name, client)
        if source_max is None:
            return configured_end
        return min(configured_end, source_max)

    def _source_max_date(self, dataset_name: str, client: WRDSClient) -> date | None:
        if dataset_name in self._source_max_date_cache:
            return self._source_max_date_cache[dataset_name]
        sql_map = {
            "crsp_daily": f"SELECT MAX(date)::date AS max_date FROM {self.settings.relations.crsp_daily}",
            "compustat_quarterly": (
                f"SELECT MAX(datadate)::date AS max_date FROM {self.settings.relations.compustat_quarterly} "
                "WHERE indfmt='INDL' AND datafmt='STD' AND consol='C' AND popsrc='D'"
            ),
            "compustat_annual": (
                f"SELECT MAX(datadate)::date AS max_date FROM {self.settings.relations.compustat_annual} "
                "WHERE indfmt='INDL' AND datafmt='STD' AND consol='C' AND popsrc='D'"
            ),
            "compustat_segments_historical": (
                f"SELECT MAX(datadate)::date AS max_date FROM {self.settings.relations.compustat_segments_merged}"
            ),
            "compustat_quarterly_variant_metrics": (
                f"""
                SELECT GREATEST(
                    (SELECT MAX(datadate)::date FROM {self.settings.relations.compustat_quarterly_ytd}),
                    (SELECT MAX(datadate)::date FROM {self.settings.relations.compustat_quarterly_semi}),
                    (SELECT MAX(datadate)::date FROM {self.settings.relations.compustat_quarterly_flags}),
                    (SELECT MAX(datadate)::date FROM {self.settings.relations.compustat_security_quarterly}),
                    (SELECT MAX(datadate)::date FROM {self.settings.relations.compustat_security_quarterly_flags})
                ) AS max_date
                """
            ),
            "ibes_actuals_epsus": (
                f"SELECT MAX(pends)::date AS max_date FROM {self.settings.relations.ibes_actuals_epsus}"
            ),
            "ibes_summary_epsus": (
                f"SELECT MAX(statpers)::date AS max_date FROM {self.settings.relations.ibes_summary_epsus}"
            ),
        }
        sql = sql_map.get(dataset_name)
        if sql is None:
            self._source_max_date_cache[dataset_name] = None
            return None
        result = client.raw_sql(sql)
        value = None
        if not result.rows.empty:
            raw = result.rows.iloc[0].get("max_date")
            if pd.notna(raw):
                value = pd.Timestamp(raw).date()
        self._source_max_date_cache[dataset_name] = value
        return value

    @staticmethod
    def _should_use_ccm_fallback(exc: Exception) -> bool:
        message = str(exc).lower()
        return "permission denied for schema crsp_a_ccm" in message or "insufficientprivilege" in message

    @staticmethod
    def _build_ccm_fallback_sql(limit: int | None = None) -> str:
        comp_seed_sql = "SELECT * FROM comp.names"
        if limit is not None:
            comp_seed_sql = f"SELECT * FROM comp.names ORDER BY gvkey LIMIT {int(limit)}"
        outer_limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
        order_by_sql = "" if limit is not None else "ORDER BY matched.gvkey, matched.lpermno, matched.linkdt"
        base_sql = f"""
            WITH comp_seed AS (
                {comp_seed_sql}
            )
            SELECT
                matched.gvkey::text AS gvkey,
                COALESCE(matched.iid, '01')::text AS liid,
                matched.lpermno::bigint AS lpermno,
                matched.lpermco::bigint AS lpermco,
                'HX'::text AS linktype,
                'P'::text AS linkprim,
                matched.linkdt::date AS linkdt,
                matched.linkenddt::date AS linkenddt,
                1::integer AS usedflag
            FROM (
                SELECT DISTINCT
                    n.gvkey,
                    sec.iid,
                    ds.permno AS lpermno,
                    ds.permco AS lpermco,
                    GREATEST(ds.namedt, MAKE_DATE(COALESCE(n.year1::int, 1900), 1, 1)) AS linkdt,
                    CASE
                        WHEN LEAST(
                            COALESCE(ds.nameendt, DATE '9999-12-31'),
                            MAKE_DATE(COALESCE(n.year2::int, EXTRACT(YEAR FROM CURRENT_DATE)::int), 12, 31)
                        ) = DATE '9999-12-31'
                            THEN NULL
                        ELSE LEAST(
                            COALESCE(ds.nameendt, DATE '9999-12-31'),
                            MAKE_DATE(COALESCE(n.year2::int, EXTRACT(YEAR FROM CURRENT_DATE)::int), 12, 31)
                        )
                    END AS linkenddt
                FROM comp_seed n
                LEFT JOIN comp.security sec
                  ON sec.gvkey = n.gvkey
                 AND (
                        (LEFT(COALESCE(sec.cusip, ''), 8) <> ''
                         AND LEFT(COALESCE(sec.cusip, ''), 8) = LEFT(COALESCE(n.cusip, ''), 8))
                     OR (COALESCE(sec.tic, '') <> ''
                         AND UPPER(TRIM(sec.tic)) = UPPER(TRIM(n.tic)))
                 )
                JOIN crsp.dsenames ds
                  ON (
                        (LEFT(COALESCE(n.cusip, ''), 8) <> ''
                         AND LEFT(COALESCE(ds.ncusip, ds.cusip, ''), 8) = LEFT(COALESCE(n.cusip, ''), 8))
                     OR (COALESCE(n.tic, '') <> ''
                         AND UPPER(TRIM(ds.ticker)) = UPPER(TRIM(n.tic)))
                  )
                 AND ds.namedt <= MAKE_DATE(COALESCE(n.year2::int, EXTRACT(YEAR FROM CURRENT_DATE)::int), 12, 31)
                 AND COALESCE(ds.nameendt, DATE '9999-12-31') >= MAKE_DATE(COALESCE(n.year1::int, 1900), 1, 1)
            ) matched
            {order_by_sql}
            {outer_limit_sql}
        """
        return base_sql

    def _ccm_fallback_limit(self) -> int | None:
        if not self.settings.sample_mode:
            return None
        return max(250, min(1000, int(self.settings.sample_row_limit)))

    def _start_run(self, command_name: str, dataset_scope: list[str]) -> RunContext:
        run = RunContext(
            run_id=uuid.uuid4().hex,
            started_at=pd.Timestamp.utcnow(),
            command_name=command_name,
            dataset_scope=",".join(dataset_scope),
        )
        self.db.upsert_run(
            {
                "run_id": run.run_id,
                "started_at": run.started_at,
                "finished_at": None,
                "status": "running",
                "command_name": command_name,
                "dataset_scope": run.dataset_scope,
                "sample_mode": self.settings.sample_mode,
                "dry_run": self.settings.dry_run,
                "force": self.settings.force,
                "config_json": json.dumps(self._config_snapshot(), ensure_ascii=True, sort_keys=True),
                "summary_json": None,
                "error_text": None,
            }
        )
        return run

    def _finish_run(self, run: RunContext, status: str, *, summary: dict[str, Any], error_text: str | None = None) -> None:
        self.db.upsert_run(
            {
                "run_id": run.run_id,
                "started_at": run.started_at,
                "finished_at": pd.Timestamp.utcnow(),
                "status": status,
                "command_name": run.command_name,
                "dataset_scope": run.dataset_scope,
                "sample_mode": self.settings.sample_mode,
                "dry_run": self.settings.dry_run,
                "force": self.settings.force,
                "config_json": json.dumps(self._config_snapshot(), ensure_ascii=True, sort_keys=True),
                "summary_json": json.dumps(summary, ensure_ascii=True, sort_keys=True),
                "error_text": error_text,
            }
        )

    def _upsert_checkpoint(
        self,
        *,
        dataset_name: str,
        chunk_key: str,
        status: str,
        run_id: str,
        chunk_start: date | None,
        chunk_end: date | None,
        rows_fetched: int,
        rows_written: int,
        notes: str | None,
    ) -> None:
        self.db.upsert_checkpoint(
            {
                "dataset_name": dataset_name,
                "chunk_key": chunk_key,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "last_source_value": chunk_end.isoformat() if chunk_end else None,
                "status": status,
                "rows_fetched": rows_fetched,
                "rows_written": rows_written,
                "run_id": run_id,
                "updated_at": pd.Timestamp.utcnow(),
                "notes": notes,
            }
        )

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "db_path": str(self.settings.db_path),
            "wrds_username_present": bool(self.settings.wrds_username),
            "wrds_password_present": bool(self.settings.wrds_password),
            "start_date": self.settings.start_date.isoformat() if self.settings.start_date else None,
            "end_date": self.settings.end_date.isoformat() if self.settings.end_date else None,
            "sample_mode": self.settings.sample_mode,
            "sample_years": self.settings.sample_years,
            "dry_run": self.settings.dry_run,
            "force": self.settings.force,
            "relations": self.settings.relations.__dict__,
            "indexes": {
                "sp500_index_ids": list(self.settings.indexes.sp500_index_ids),
                "nasdaq100_index_ids": list(self.settings.indexes.nasdaq100_index_ids),
                "enable_sp500": self.settings.indexes.enable_sp500,
                "enable_nasdaq100": self.settings.indexes.enable_nasdaq100,
            },
            "chunking": self.settings.chunking.__dict__,
        }

    def _write_manifest(self, run_id: str, summary: dict[str, Any]) -> Path:
        self.settings.manifest_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.manifest_dir / f"{run_id}.json"
        path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        return path
