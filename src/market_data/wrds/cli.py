from __future__ import annotations

import argparse
import json

from market_data.wrds.service import WRDSIngestionService
from market_data.wrds.config import load_wrds_settings


def add_wrds_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the `market_data wrds ...` command tree to the root CLI."""

    parser = subparsers.add_parser("wrds", help="WRDS-first DuckDB ingestion and canonicalization")
    wrds_sub = parser.add_subparsers(dest="wrds_command", required=True)

    p_init = wrds_sub.add_parser("init-schema", help="Create WRDS source/canonical tables in DuckDB")
    _add_common_options(p_init, include_runtime=False)

    p_probe = wrds_sub.add_parser("probe", help="Inspect credential resolution and optionally run a tiny live WRDS query")
    _add_common_options(p_probe, include_runtime=True)
    p_probe.add_argument("--live", action="store_true", help="Run a tiny live query after credential checks")

    p_rel = wrds_sub.add_parser("list-relations", help="List accessible WRDS libraries and relation previews")
    _add_common_options(p_rel, include_runtime=True)
    p_rel.add_argument("--libraries", default="crsp,comp", help="Comma-separated WRDS libraries to inspect")
    p_rel.add_argument("--preview-limit", type=int, default=20, help="Max relation names to preview per library")

    p_test = wrds_sub.add_parser("test-query", help="Run a limit-1 live query for a single WRDS dataset")
    _add_common_options(p_test, include_runtime=True)
    p_test.add_argument("--dataset", required=True, help="Dataset name: crsp_daily, compustat_quarterly, compustat_annual, ccm_link, ...")
    p_test.add_argument("--limit", type=int, default=1, help="Maximum rows to fetch for the probe query")

    p_ingest = wrds_sub.add_parser("ingest", help="Ingest WRDS source tables directly into DuckDB")
    _add_common_options(p_ingest, include_runtime=True)
    p_ingest.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset list: all, crsp_daily, compustat_quarterly, compustat_annual, company_master, security_master, ccm_link, universe_membership_history, links",
    )
    p_ingest.add_argument("--build-canonical", action="store_true", help="Rebuild canonical tables after source ingest")

    p_links = wrds_sub.add_parser("refresh-links", help="Refresh company/security master, CCM, and universe source tables")
    _add_common_options(p_links, include_runtime=True)
    p_links.add_argument("--build-canonical", action="store_true", help="Rebuild canonical tables after refresh")

    p_canonical = wrds_sub.add_parser("build-canonical", help="Rebuild canonical research tables from WRDS source tables")
    _add_common_options(p_canonical, include_runtime=False)
    p_canonical.add_argument(
        "--tables",
        default="all",
        help="Comma-separated canonical tables: all, entity_master, security_link_history, prices_daily_canonical, financials_quarterly_canonical, financials_annual_canonical, universe_membership_history",
    )

    p_validate_source = wrds_sub.add_parser("validate-source", help="Validate WRDS source tables in DuckDB")
    _add_common_options(p_validate_source, include_runtime=False)
    p_validate_source.add_argument("--datasets", default="all", help="Comma-separated source dataset names")

    p_validate_canonical = wrds_sub.add_parser("validate-canonical", help="Validate canonical WRDS tables in DuckDB")
    _add_common_options(p_validate_canonical, include_runtime=False)
    p_validate_canonical.add_argument("--tables", default="all", help="Comma-separated canonical table names")

    p_inspect = wrds_sub.add_parser("inspect", help="Inspect schema/profile/sample rows for a source or canonical table")
    _add_common_options(p_inspect, include_runtime=False)
    p_inspect.add_argument("--dataset", required=True, help="Source dataset name or canonical table name")
    p_inspect.add_argument("--layer", choices=["auto", "source", "canonical"], default="auto")
    p_inspect.add_argument("--limit", type=int, default=5, help="Number of sample rows to return")

    p_compare = wrds_sub.add_parser("compare-sec", help="Compare WRDS canonical financials against SEC-derived financials")
    _add_common_options(p_compare, include_runtime=True)
    p_compare.add_argument("--tickers", default=None, help="Comma-separated ticker list for focused comparison")
    p_compare.add_argument("--market", default="us", help="Target market on SEC tables (default: us)")
    p_compare.add_argument("--limit", type=int, default=None, help="Limit common tickers when --tickers is omitted")
    p_compare.add_argument(
        "--statement-types",
        default="quarterly,annual",
        help="Comma-separated statement types: quarterly, annual",
    )
    p_compare.add_argument(
        "--compare-mode",
        choices=["default", "reported", "normalized"],
        default="default",
        help="Comparison value mode: registry default, direct reported SEC, or WRDS-aligned normalized SEC",
    )
    p_compare.add_argument(
        "--metrics",
        default=None,
        help="Optional comma-separated canonical metric names for focused compare runs (for example: sbc,apic,r_and_d)",
    )
    p_compare.add_argument(
        "--include-inactive",
        action="store_true",
        help="Allow pilot/deferred metric rows from the registry to participate in compare runs",
    )

    p_report = wrds_sub.add_parser("compare-sec-report", help="Summarize SEC-vs-WRDS comparison results")
    _add_common_options(p_report, include_runtime=False)
    p_report.add_argument(
        "--group-by",
        choices=["metric", "ticker", "statement_type", "mismatch_class", "compare_mode", "value_mode", "time_regime"],
        default="metric",
    )
    p_report.add_argument("--run-id", default=None, help="Optional comparison_run_id; defaults to latest")
    p_report.add_argument("--limit", type=int, default=20, help="Maximum rows per summary block")

    p_policy = wrds_sub.add_parser("compare-sec-policy-summary", help="Summarize SEC-vs-WRDS results using policy classes")
    _add_common_options(p_policy, include_runtime=False)
    p_policy.add_argument("--run-id", default=None, help="Optional comparison_run_id; defaults to latest")
    p_policy.add_argument("--limit", type=int, default=20, help="Maximum metric policy rows to return")

    p_segment_compare = wrds_sub.add_parser("compare-sec-segments", help="Compare SEC companyfacts-derived segments against filing-derived segments")
    _add_common_options(p_segment_compare, include_runtime=False)
    p_segment_compare.add_argument("--tickers", default=None, help="Comma-separated ticker list for focused comparison")
    p_segment_compare.add_argument("--market", default="us", help="Target market on SEC tables (default: us)")
    p_segment_compare.add_argument("--start", default=None, help="Inclusive start date (YYYY-MM-DD) for segment comparison")

    p_segment_avail = wrds_sub.add_parser("compare-sec-segments-availability", help="Survey segment source availability and issuer clusters")
    _add_common_options(p_segment_avail, include_runtime=False)
    p_segment_avail.add_argument("--tickers", default=None, help="Comma-separated ticker list for focused survey")
    p_segment_avail.add_argument("--market", default="us", help="Target market on SEC tables (default: us)")
    p_segment_avail.add_argument("--start", default=None, help="Inclusive start date (YYYY-MM-DD) for segment survey")
    p_segment_avail.add_argument("--limit", type=int, default=20, help="Maximum sample issuer rows to return")

    p_segment_report = wrds_sub.add_parser("compare-sec-segments-report", help="Summarize SEC segment comparison results")
    _add_common_options(p_segment_report, include_runtime=False)
    p_segment_report.add_argument(
        "--group-by",
        choices=[
            "metric",
            "ticker",
            "segment_type",
            "mismatch_class",
            "issuer_cluster",
            "issuer_cluster_detail",
            "source_lane",
            "segment_type_profile",
            "filing_source_confidence",
            "coverage_reason",
        ],
        default="metric",
    )
    p_segment_report.add_argument("--run-id", default=None, help="Optional comparison_run_id; defaults to latest")
    p_segment_report.add_argument("--limit", type=int, default=20, help="Maximum rows per summary block")

    p_segment_policy = wrds_sub.add_parser("compare-sec-segments-policy-summary", help="Summarize SEC segment comparison policy classes")
    _add_common_options(p_segment_policy, include_runtime=False)
    p_segment_policy.add_argument("--run-id", default=None, help="Optional comparison_run_id; defaults to latest")
    p_segment_policy.add_argument("--limit", type=int, default=20, help="Maximum metric policy rows to return")

    p_segment_cluster = wrds_sub.add_parser("compare-sec-segments-cluster-summary", help="Summarize issuer clusters and ticker membership for segment sources")
    _add_common_options(p_segment_cluster, include_runtime=False)
    p_segment_cluster.add_argument("--run-id", default=None, help="Optional survey_run_id; defaults to latest")
    p_segment_cluster.add_argument("--limit", type=int, default=20, help="Maximum cluster rows to return")

    p_segment_recovery = wrds_sub.add_parser("compare-sec-segments-recovery-summary", help="Summarize recover_now / recover_next candidates for segment source recovery")
    _add_common_options(p_segment_recovery, include_runtime=False)
    p_segment_recovery.add_argument("--run-id", default=None, help="Optional survey_run_id; defaults to latest recovery or coverage run")
    p_segment_recovery.add_argument("--limit", type=int, default=20, help="Maximum recovery rows to return")

    p_aux = wrds_sub.add_parser("survey-wrds-aux-readiness", help="Survey WRDS auxiliary metric readiness for deferred SEC compare metrics")
    _add_common_options(p_aux, include_runtime=True)
    p_aux.add_argument("--tickers", required=True, help="Comma-separated ticker list to survey")

    p_cross = wrds_sub.add_parser("compare-sec-crossdb", help="Compare SEC DuckDB against an attached WRDS DuckDB lake")
    _add_common_options(p_cross, include_runtime=True)
    p_cross.add_argument("--sec-db-path", default=None, help="SEC/materialized DuckDB path (default: current project DB)")
    p_cross.add_argument("--wrds-db-path", default=None, help="WRDS lake DuckDB path (default: data/wrds_market_data.duckdb)")
    p_cross.add_argument("--tickers", default=None, help="Comma-separated ticker list for focused comparison")
    p_cross.add_argument("--market", default="us", help="Target market on SEC tables (default: us)")
    p_cross.add_argument("--limit", type=int, default=None, help="Limit common tickers when --tickers is omitted")
    p_cross.add_argument(
        "--statement-types",
        default="quarterly,annual",
        help="Comma-separated statement types: quarterly, annual",
    )
    p_cross.add_argument(
        "--compare-mode",
        choices=["default", "reported", "normalized"],
        default="default",
        help="Comparison value mode: registry default, direct reported SEC, or WRDS-aligned normalized SEC",
    )
    p_cross.add_argument(
        "--metrics",
        default=None,
        help="Optional comma-separated canonical metric names for focused compare runs",
    )
    p_cross.add_argument(
        "--include-inactive",
        action="store_true",
        help="Allow pilot/deferred metric rows from the registry to participate in compare runs",
    )

    p_access = wrds_sub.add_parser("survey-source-access", help="Persist WRDS relation access/permission survey into DuckDB")
    _add_common_options(p_access, include_runtime=True)

    p_kr_survey = wrds_sub.add_parser("survey-kr-availability", help="Survey WRDS Korea-company availability and relation coverage")
    _add_common_options(p_kr_survey, include_runtime=True)
    p_kr_survey.add_argument("--kr-db-path", default=None, help="KR WRDS lake DuckDB path (default: data/wrds_kr_market_data.duckdb)")

    p_kr_build = wrds_sub.add_parser("build-kr-lake", help="Build a Korea-focused WRDS reference DuckDB lake")
    _add_common_options(p_kr_build, include_runtime=True)
    p_kr_build.add_argument("--kr-db-path", default=None, help="KR WRDS lake DuckDB path (default: data/wrds_kr_market_data.duckdb)")


def run_wrds_command(args: argparse.Namespace) -> int:
    """Execute the selected WRDS subcommand."""

    try:
        settings = load_wrds_settings(
            config_path=getattr(args, "config", None),
            db_path=getattr(args, "db_path", None),
            wrds_username=getattr(args, "wrds_username", None),
            start_date=getattr(args, "start", None),
            end_date=getattr(args, "end", None),
            sample_mode=bool(getattr(args, "sample", False)),
            dry_run=bool(getattr(args, "dry_run", False)),
            force=bool(getattr(args, "force", False)),
        )
        service = WRDSIngestionService(settings)

        if args.wrds_command == "init-schema":
            print(json.dumps(service.init_schema(), ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "probe":
            print(json.dumps(service.probe_connection(live=bool(args.live)), ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "list-relations":
            print(
                json.dumps(
                    service.list_relations(args.libraries, preview_limit=max(1, int(args.preview_limit))),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.wrds_command == "test-query":
            print(
                json.dumps(
                    service.test_query(str(args.dataset), limit=max(1, int(args.limit))),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.wrds_command == "ingest":
            payload = service.ingest(args.datasets, build_canonical_after=bool(args.build_canonical))
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "refresh-links":
            payload = service.ingest("links", build_canonical_after=bool(args.build_canonical))
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "build-canonical":
            payload = service.build_canonical(args.tables)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "validate-source":
            print(json.dumps(service.validate_source(args.datasets), ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "validate-canonical":
            print(json.dumps(service.validate_canonical(args.tables), ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "inspect":
            print(
                json.dumps(
                    service.inspect(args.dataset, layer=str(args.layer), limit=max(1, int(args.limit))),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.wrds_command == "compare-sec":
            statement_types = tuple(item.strip().lower() for item in str(args.statement_types).split(",") if item.strip())
            metric_names = tuple(item.strip().lower() for item in str(args.metrics or "").split(",") if item.strip())
            payload = service.compare_sec(
                tickers=getattr(args, "tickers", None),
                market=str(getattr(args, "market", "us")),
                limit=getattr(args, "limit", None),
                start_date=getattr(settings, "start_date", None),
                statement_types=statement_types,
                compare_mode=str(getattr(args, "compare_mode", "default")),
                metric_names=metric_names or None,
                include_inactive=bool(getattr(args, "include_inactive", False)),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "compare-sec-crossdb":
            statement_types = tuple(item.strip().lower() for item in str(args.statement_types).split(",") if item.strip())
            metric_names = tuple(item.strip().lower() for item in str(args.metrics or "").split(",") if item.strip())
            payload = service.compare_sec_crossdb(
                sec_db_path=getattr(args, "sec_db_path", None),
                wrds_db_path=getattr(args, "wrds_db_path", None),
                tickers=getattr(args, "tickers", None),
                market=str(getattr(args, "market", "us")),
                limit=getattr(args, "limit", None),
                start_date=getattr(settings, "start_date", None),
                statement_types=statement_types,
                compare_mode=str(getattr(args, "compare_mode", "default")),
                metric_names=metric_names or None,
                include_inactive=bool(getattr(args, "include_inactive", False)),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "compare-sec-report":
            payload = service.compare_sec_report(
                group_by=str(args.group_by),
                comparison_run_id=getattr(args, "run_id", None),
                limit=max(1, int(args.limit)),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "compare-sec-policy-summary":
            payload = service.compare_sec_policy_summary(
                comparison_run_id=getattr(args, "run_id", None),
                limit=max(1, int(args.limit)),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "compare-sec-segments":
            payload = service.compare_sec_segments(
                tickers=getattr(args, "tickers", None),
                market=str(getattr(args, "market", "us")),
                start_date=getattr(settings, "start_date", None),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "compare-sec-segments-availability":
            payload = service.compare_sec_segments_availability(
                tickers=getattr(args, "tickers", None),
                market=str(getattr(args, "market", "us")),
                start_date=getattr(settings, "start_date", None),
            )
            summary = service.compare_sec_segments_availability_summary(
                survey_run_id=payload.get("survey_run_id"),
                limit=max(1, int(getattr(args, "limit", 20))),
            )
            merged = dict(payload)
            merged["availability_summary"] = summary
            print(json.dumps(merged, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "compare-sec-segments-report":
            payload = service.compare_sec_segments_report(
                group_by=str(args.group_by),
                comparison_run_id=getattr(args, "run_id", None),
                limit=max(1, int(args.limit)),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "compare-sec-segments-policy-summary":
            payload = service.compare_sec_segments_policy_summary(
                comparison_run_id=getattr(args, "run_id", None),
                limit=max(1, int(args.limit)),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "compare-sec-segments-cluster-summary":
            payload = service.compare_sec_segments_cluster_summary(
                survey_run_id=getattr(args, "run_id", None),
                limit=max(1, int(args.limit)),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "compare-sec-segments-recovery-summary":
            payload = service.compare_sec_segments_recovery_summary(
                survey_run_id=getattr(args, "run_id", None),
                limit=max(1, int(args.limit)),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "survey-wrds-aux-readiness":
            payload = service.survey_wrds_aux_metric_readiness(
                tickers=str(args.tickers),
                start_date=getattr(settings, "start_date", None),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "survey-source-access":
            payload = service.survey_source_access_registry()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "survey-kr-availability":
            payload = service.survey_kr_availability(db_path=getattr(args, "kr_db_path", None))
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        if args.wrds_command == "build-kr-lake":
            payload = service.build_kr_lake(db_path=getattr(args, "kr_db_path", None))
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        raise ValueError(f"Unsupported wrds command: {args.wrds_command}")
    except Exception as exc:  # noqa: BLE001
        payload = {"wrds_command": str(args.wrds_command), "error": str(exc)}
        if hasattr(exc, "to_dict") and callable(getattr(exc, "to_dict")):
            payload["details"] = exc.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2


def _add_common_options(parser: argparse.ArgumentParser, *, include_runtime: bool) -> None:
    parser.add_argument("--config", default=None, help="Optional JSON config file for WRDS relation/index overrides")
    parser.add_argument("--db-path", default=None, help="Optional DuckDB path override (default: data/market_data.duckdb)")
    if include_runtime:
        parser.add_argument("--wrds-username", default=None, help="WRDS username (defaults to WRDS_USERNAME env var)")
        parser.add_argument("--start", default=None, help="Inclusive start date (YYYY-MM-DD) for chunked datasets")
        parser.add_argument("--end", default=None, help="Inclusive end date (YYYY-MM-DD) for chunked datasets")
        parser.add_argument("--sample", action="store_true", help="Sample mode: narrower time window and smaller full-refresh pulls")
        parser.add_argument("--dry-run", action="store_true", help="Fetch and summarize without writing to DuckDB")
        parser.add_argument("--force", action="store_true", help="Ignore completed checkpoints and rerun all chunks")
