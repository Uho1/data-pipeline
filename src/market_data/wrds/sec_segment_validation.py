from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd

from market_data.wrds.duckdb_io import DuckDBManager

MISMATCH_CLASS_NONE = "no_mismatch"
MISMATCH_CLASS_PARSER_BUG = "parser_bug"
MISMATCH_CLASS_NORMALIZATION_GAP = "normalization_gap"
MISMATCH_CLASS_SOURCE_SEMANTICS_GAP = "source_semantics_gap"
MISMATCH_CLASS_SOURCE_COVERAGE_GAP = "source_coverage_gap"
MISMATCH_CLASS_REFERENCE_GAP = "reference_gap"
COMPARE_MODE_DEFAULT = "default"
VALUE_MODE_REPORTED = "reported_sec"
COMPARISON_STATUS_FILING_ONLY_VALIDATED = "filing_only_validated"
SEGMENT_COMPANYFACTS_SOURCE = "sec_companyfacts_segment"
SEGMENT_FILING_SOURCE_PRIORITY = {
    "ixbrl_dimension": 1,
    "xbrl_instance_dimension": 2,
    "xbrl_instance": 2,
    "html_table_mvp": 3,
}
SEGMENT_FILING_SOURCE_CONFIDENCE = {
    "ixbrl_dimension": "high",
    "xbrl_instance_dimension": "medium",
    "xbrl_instance": "medium",
    "html_table_mvp": "low",
}
SEGMENT_TYPE_CANONICAL = {
    "business": "business",
    "product": "product",
    "geography": "geography",
    "geographic": "geography",
    "geo": "geography",
    "operating": "business",
}
SEGMENT_NAME_STOPWORDS = {
    "member",
    "members",
    "segment",
    "segments",
    "reportable",
    "reportables",
    "group",
    "groups",
    "business",
    "businesses",
    "geographic",
    "geography",
    "region",
    "regions",
    "operation",
    "operations",
    "division",
    "divisions",
}


def _quote(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _quoted_literal_list(values: list[str]) -> str:
    payload = ", ".join("'" + value.replace("'", "''") + "'" for value in values)
    return f"({payload})" if payload else "('')"


@dataclass(frozen=True)
class SegmentMetricPolicy:
    metric_name: str
    statement_type: str
    compare_pair: str
    metric_class: str
    value_mode_default: str
    dimension_normalization_rule: str
    member_normalization_rule: str
    reported_value_policy: str
    normalized_value_policy: str
    mismatch_policy: str
    tolerance_type: str
    tolerance_value: float
    known_gap_flag: bool
    known_gap_reason: str
    recommended_compare_mode: str
    notes: str
    is_active: bool = True


@dataclass(frozen=True)
class SegmentClusterPolicy:
    issuer_cluster: str
    source_lane: str
    companyfacts_required: bool
    filing_only_validated_allowed: bool
    revenue_supported: bool
    operating_income_supported: bool
    preferred_segment_types: str
    source_coverage_policy: str
    mismatch_policy: str
    low_confidence_flag: bool
    recovery_policy: str
    notes: str
    is_active: bool = True


def default_segment_metric_policies() -> list[SegmentMetricPolicy]:
    return [
        SegmentMetricPolicy(
            metric_name="revenue",
            statement_type="quarterly",
            compare_pair="companyfacts_vs_filing",
            metric_class="segment_direct_match",
            value_mode_default=VALUE_MODE_REPORTED,
            dimension_normalization_rule="normalize_segment_type_to_business_product_geography",
            member_normalization_rule="normalize_member_name_slug_with_generic_suffix_stripping",
            reported_value_policy="companyfacts_vs_best_filing_source",
            normalized_value_policy="normalized_segment_key_match_only",
            mismatch_policy="compare_same_metric_same_period_same_segment_key",
            tolerance_type="relative",
            tolerance_value=0.05,
            known_gap_flag=True,
            known_gap_reason="CompanyFacts segment rollups and filing segment tables can diverge when issuers omit intersegment eliminations or present custom segment member names.",
            recommended_compare_mode=COMPARE_MODE_DEFAULT,
            notes="Filing-derived rows prefer iXBRL explicit-dimension extraction, then XBRL instance facts, then HTML table fallback.",
        ),
        SegmentMetricPolicy(
            metric_name="operating_income",
            statement_type="quarterly",
            compare_pair="companyfacts_vs_filing",
            metric_class="segment_scope_sensitive",
            value_mode_default=VALUE_MODE_REPORTED,
            dimension_normalization_rule="normalize_segment_type_to_business_product_geography",
            member_normalization_rule="normalize_member_name_slug_with_generic_suffix_stripping",
            reported_value_policy="companyfacts_vs_best_filing_source",
            normalized_value_policy="normalized_segment_key_match_only",
            mismatch_policy="compare_same_metric_same_period_same_segment_key",
            tolerance_type="relative",
            tolerance_value=0.08,
            known_gap_flag=True,
            known_gap_reason="Segment operating income is more sensitive to issuer-specific profitability definitions and segment disclosure scope than segment revenue.",
            recommended_compare_mode=COMPARE_MODE_DEFAULT,
            notes="Differences between companyfacts and filing-derived segment operating income are often semantic rather than parser-only.",
        ),
    ]


def default_segment_cluster_policies() -> list[SegmentClusterPolicy]:
    return [
        SegmentClusterPolicy(
            issuer_cluster="companyfacts_overlap",
            source_lane="companyfacts_vs_filing",
            companyfacts_required=True,
            filing_only_validated_allowed=False,
            revenue_supported=True,
            operating_income_supported=True,
            preferred_segment_types="business|geography|product",
            source_coverage_policy="full_compare",
            mismatch_policy="missing_companyfacts_can_be_parser_bug",
            low_confidence_flag=False,
            recovery_policy="no_fetch_needed",
            notes="Best-case issuer cluster: companyfacts and filing-derived segment rows overlap on at least one usable period.",
        ),
        SegmentClusterPolicy(
            issuer_cluster="filing_only_full",
            source_lane="filing_only_validation",
            companyfacts_required=False,
            filing_only_validated_allowed=True,
            revenue_supported=True,
            operating_income_supported=True,
            preferred_segment_types="business|geography|product",
            source_coverage_policy="treat_missing_companyfacts_as_source_coverage_gap",
            mismatch_policy="allow_filing_only_validated_when_temporally_consistent",
            low_confidence_flag=False,
            recovery_policy="no_fetch_needed",
            notes="No companyfacts segment rows, but filing-derived segment history is deep enough to validate internally by normalized segment-key consistency.",
        ),
        SegmentClusterPolicy(
            issuer_cluster="filing_only_revenue_only",
            source_lane="filing_only_validation",
            companyfacts_required=False,
            filing_only_validated_allowed=True,
            revenue_supported=True,
            operating_income_supported=False,
            preferred_segment_types="business|geography|product",
            source_coverage_policy="revenue_only_filing_validation",
            mismatch_policy="operating_income_missing_is_source_coverage_gap",
            low_confidence_flag=False,
            recovery_policy="candidate_for_operating_income_recovery",
            notes="Filing-derived segment revenue is usable, but operating-income support is sparse or absent. Lower-confidence filing sources may still recover some issuers.",
        ),
        SegmentClusterPolicy(
            issuer_cluster="filing_only_partial",
            source_lane="filing_only_low_confidence",
            companyfacts_required=False,
            filing_only_validated_allowed=False,
            revenue_supported=True,
            operating_income_supported=False,
            preferred_segment_types="business|geography|product",
            source_coverage_policy="partial_filing_only",
            mismatch_policy="classify_missing_companyfacts_as_source_coverage_gap",
            low_confidence_flag=True,
            recovery_policy="recover_next_when_parser_candidate",
            notes="Issuer has some filing-derived segment history, but not enough period depth or metric breadth for filing-only validation.",
        ),
        SegmentClusterPolicy(
            issuer_cluster="segment_source_poor",
            source_lane="source_poor",
            companyfacts_required=False,
            filing_only_validated_allowed=False,
            revenue_supported=False,
            operating_income_supported=False,
            preferred_segment_types="none",
            source_coverage_policy="reference_gap_dominant",
            mismatch_policy="missing_sources_are_reference_gap",
            low_confidence_flag=True,
            recovery_policy="recover_only_when_cache_gap_or_parser_candidate",
            notes="Neither companyfacts nor filing-derived segment history is rich enough for meaningful compare on this basket window.",
        ),
    ]


def seed_default_segment_metric_policy_registry(db: DuckDBManager) -> int:
    frame = pd.DataFrame([asdict(item) for item in default_segment_metric_policies()])
    frame["updated_at"] = pd.Timestamp.utcnow()
    if frame.empty:
        return 0
    return db.merge_dataframe("segment_metric_policy_registry", frame, ("metric_name", "statement_type"))


def seed_default_segment_cluster_policy_registry(db: DuckDBManager) -> int:
    frame = pd.DataFrame([asdict(item) for item in default_segment_cluster_policies()])
    frame["updated_at"] = pd.Timestamp.utcnow()
    if frame.empty:
        return 0
    return db.merge_dataframe("segment_cluster_policy_registry", frame, ("issuer_cluster",))


class SECSegmentValidationService:
    def __init__(self, db: DuckDBManager, logger: logging.Logger | None = None) -> None:
        self.db = db
        self.logger = logger or logging.getLogger(__name__)

    def ensure_registry_seeded(self) -> dict[str, int]:
        return {
            "metric_policy_rows": int(seed_default_segment_metric_policy_registry(self.db)),
            "cluster_policy_rows": int(seed_default_segment_cluster_policy_registry(self.db)),
        }

    def survey_source_availability(
        self,
        *,
        survey_run_id: str,
        tickers: str | list[str] | None,
        market: str,
        start_date: date,
    ) -> dict[str, Any]:
        self.ensure_registry_seeded()
        targets = self._resolve_target_tickers(tickers=tickers, market=market, start_date=start_date)
        facts = self._load_segment_rows(tickers=targets, market=market, start_date=start_date)
        logs = self._load_segment_extract_logs(tickers=targets, market=market)
        filing_meta = self._load_recent_filing_meta(tickers=targets, market=market, start_date=start_date)
        cluster_policies = self._load_cluster_policies()
        coverage = self._build_source_coverage_summary(
            facts=facts,
            logs=logs,
            filing_meta=filing_meta,
            tickers=targets,
            market=market,
            start_date=start_date,
            survey_run_id=survey_run_id,
            cluster_policies=cluster_policies,
        )
        self.db.merge_dataframe("segment_issuer_source_coverage", coverage, ("coverage_result_id",))
        recovery = self._build_recovery_candidate_frame(coverage=coverage, survey_run_id=survey_run_id)
        self.db.merge_dataframe("segment_recovery_candidate_registry", recovery, ("candidate_result_id",))
        return self._summarize_source_coverage(
            coverage=coverage,
            survey_run_id=survey_run_id,
            tickers=targets,
            market=market,
            start_date=start_date,
        )

    def compare(
        self,
        *,
        comparison_run_id: str,
        tickers: str | list[str] | None,
        market: str,
        start_date: date,
    ) -> dict[str, Any]:
        self.ensure_registry_seeded()
        policies = self._load_policies()
        cluster_policies = self._load_cluster_policies()
        targets = self._resolve_target_tickers(tickers=tickers, market=market, start_date=start_date)
        facts = self._load_segment_rows(tickers=targets, market=market, start_date=start_date)
        logs = self._load_segment_extract_logs(tickers=targets, market=market)
        filing_meta = self._load_recent_filing_meta(tickers=targets, market=market, start_date=start_date)
        coverage = self._build_source_coverage_summary(
            facts=facts,
            logs=logs,
            filing_meta=filing_meta,
            tickers=targets,
            market=market,
            start_date=start_date,
            survey_run_id=comparison_run_id,
            cluster_policies=cluster_policies,
        )
        self.db.merge_dataframe("segment_issuer_source_coverage", coverage, ("coverage_result_id",))
        recovery = self._build_recovery_candidate_frame(coverage=coverage, survey_run_id=comparison_run_id)
        self.db.merge_dataframe("segment_recovery_candidate_registry", recovery, ("candidate_result_id",))
        results = self._build_result_frame(
            facts=facts,
            policies=policies,
            comparison_run_id=comparison_run_id,
            coverage=coverage,
            cluster_policies=cluster_policies,
        )
        self.db.merge_dataframe("validation_sec_segment_quality", results, ("result_id",))
        return self._summarize_results(
            results,
            comparison_run_id=comparison_run_id,
            tickers=targets,
            market=market,
            start_date=start_date,
            coverage=coverage,
        )

    def report(
        self,
        *,
        group_by: str = "metric",
        comparison_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        run_id = comparison_run_id or self._latest_comparison_run_id()
        if not run_id:
            return {"comparison_run_id": None, "group_by": group_by, "rows": [], "status_counts": {}}
        df = self.db.fetch_df(
            """
            SELECT *
            FROM validation_sec_segment_quality
            WHERE comparison_run_id = ?
            """,
            [run_id],
        )
        if df.empty:
            return {"comparison_run_id": run_id, "group_by": group_by, "rows": [], "status_counts": {}}

        group_map = {
            "metric": ["metric_name"],
            "ticker": ["ticker"],
            "segment_type": ["segment_type_normalized"],
            "mismatch_class": ["mismatch_class"],
            "issuer_cluster": ["issuer_cluster"],
            "issuer_cluster_detail": ["issuer_cluster_detail"],
            "source_lane": ["source_lane"],
            "segment_type_profile": ["segment_type_profile"],
            "filing_source_confidence": ["filing_source_confidence"],
            "coverage_reason": ["coverage_reason"],
            "issuer_confidence_band": ["issuer_confidence_band"],
        }
        group_cols = group_map.get(str(group_by or "metric"), ["metric_name"])
        summary = (
            df.groupby(group_cols, dropna=False)
            .agg(
                rows=("result_id", "count"),
                matches=(
                    "comparison_status",
                    lambda s: int(
                        pd.Series(s, copy=False).isin(["match", "matched_with_ambiguity", COMPARISON_STATUS_FILING_ONLY_VALIDATED]).sum()
                    ),
                ),
                filing_only_validated_rows=("comparison_status", lambda s: int((pd.Series(s, copy=False) == COMPARISON_STATUS_FILING_ONLY_VALIDATED).sum())),
                tolerance_breaches=("comparison_status", lambda s: int((pd.Series(s, copy=False) == "tolerance_breach").sum())),
                missing_on_companyfacts=("missing_on_companyfacts", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                missing_on_filing=("missing_on_filing", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                parser_bug_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_PARSER_BUG).sum())),
                source_coverage_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_SOURCE_COVERAGE_GAP).sum())),
                normalization_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_NORMALIZATION_GAP).sum())),
                source_semantics_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_SOURCE_SEMANTICS_GAP).sum())),
                reference_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_REFERENCE_GAP).sum())),
            )
            .reset_index()
            .sort_values(
                ["source_coverage_gap_rows", "source_semantics_gap_rows", "parser_bug_rows", "tolerance_breaches", "rows"],
                ascending=[False, False, False, False, False],
            )
        )
        top_breaches = (
            df.loc[df["comparison_status"] == "tolerance_breach"]
            .sort_values(["pct_diff", "abs_diff"], ascending=[False, False], na_position="last")
            .head(max(1, int(limit)))
        )
        return self._json_safe(
            {
                "comparison_run_id": run_id,
                "group_by": group_by,
                "status_counts": df["comparison_status"].value_counts(dropna=False).to_dict(),
                "rows": summary.head(max(1, int(limit))).to_dict(orient="records"),
                "top_tolerance_breaches": top_breaches[
                    [
                        "ticker",
                        "period_end",
                        "metric_name",
                        "segment_type_normalized",
                        "segment_name_normalized",
                        "companyfacts_value",
                        "filing_value",
                        "abs_diff",
                        "pct_diff",
                        "mismatch_class",
                    "semantic_gap_class",
                    "issuer_cluster",
                    "issuer_cluster_detail",
                    "source_lane",
                    "filing_source_confidence",
                    "coverage_reason",
                    "diagnostic_code",
                ]
                ].to_dict(orient="records"),
            }
        )

    def policy_summary(
        self,
        *,
        comparison_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        run_id = comparison_run_id or self._latest_comparison_run_id()
        policies = self._load_policies()
        if not run_id:
            return {
                "comparison_run_id": None,
                "rows": policies.head(max(1, int(limit))).to_dict(orient="records"),
                "mismatch_class_summary": [],
                "semantic_gap_summary": [],
            }
        df = self.db.fetch_df(
            """
            SELECT *
            FROM validation_sec_segment_quality
            WHERE comparison_run_id = ?
            """,
            [run_id],
        )
        if df.empty:
            return {
                "comparison_run_id": run_id,
                "rows": policies.head(max(1, int(limit))).to_dict(orient="records"),
                "mismatch_class_summary": [],
                "semantic_gap_summary": [],
            }
        summary = (
            df.groupby(["metric_name", "statement_type"], dropna=False)
            .agg(
                comparisons=("result_id", "count"),
                matches=(
                    "comparison_status",
                    lambda s: int(
                        pd.Series(s, copy=False).isin(["match", "matched_with_ambiguity", COMPARISON_STATUS_FILING_ONLY_VALIDATED]).sum()
                    ),
                ),
                tolerance_breaches=("comparison_status", lambda s: int((pd.Series(s, copy=False) == "tolerance_breach").sum())),
                missing_on_companyfacts=("missing_on_companyfacts", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                missing_on_filing=("missing_on_filing", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                parser_bug_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_PARSER_BUG).sum())),
                source_coverage_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_SOURCE_COVERAGE_GAP).sum())),
                normalization_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_NORMALIZATION_GAP).sum())),
                source_semantics_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_SOURCE_SEMANTICS_GAP).sum())),
                reference_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_REFERENCE_GAP).sum())),
            )
            .reset_index()
        )
        joined = summary.merge(policies, on=["metric_name", "statement_type"], how="left")
        mismatch_summary = (
            df.groupby("mismatch_class", dropna=False)["result_id"]
            .count()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
        )
        semantic_summary = (
            df.loc[df["semantic_gap_class"].notna()]
            .groupby("semantic_gap_class", dropna=False)["result_id"]
            .count()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
        )
        cluster_summary = (
            df.groupby(["issuer_cluster", "issuer_cluster_detail", "source_lane"], dropna=False)["result_id"]
            .count()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
        )
        confidence_summary = (
            df.groupby("filing_source_confidence", dropna=False)["result_id"]
            .count()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
        )
        return self._json_safe(
            {
                "comparison_run_id": run_id,
                "rows": joined.head(max(1, int(limit))).to_dict(orient="records"),
                "mismatch_class_summary": mismatch_summary.to_dict(orient="records"),
                "semantic_gap_summary": semantic_summary.to_dict(orient="records"),
                "confidence_summary": confidence_summary.to_dict(orient="records"),
                "cluster_summary": cluster_summary.head(max(1, int(limit))).to_dict(orient="records"),
            }
        )

    def availability_summary(
        self,
        *,
        survey_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        run_id = survey_run_id or self._latest_coverage_run_id()
        if not run_id:
            return {"survey_run_id": None, "rows": [], "cluster_counts": [], "metric_coverage": [], "segment_type_coverage": []}
        df = self.db.fetch_df(
            """
            SELECT *
            FROM segment_issuer_source_coverage
            WHERE survey_run_id = ?
            ORDER BY ticker
            """,
            [run_id],
        )
        if df.empty:
            return {"survey_run_id": run_id, "rows": [], "cluster_counts": [], "metric_coverage": [], "segment_type_coverage": []}
        cluster_counts = (
            df.groupby("issuer_cluster", dropna=False)["ticker"]
            .count()
            .reset_index(name="ticker_count")
            .sort_values("ticker_count", ascending=False)
        )
        cluster_detail_counts = (
            df.groupby("issuer_cluster_detail", dropna=False)["ticker"]
            .count()
            .reset_index(name="ticker_count")
            .sort_values("ticker_count", ascending=False)
        )
        source_poor_reason_counts = (
            df.loc[df["source_poor_reason"].notna()]
            .groupby("source_poor_reason", dropna=False)["ticker"]
            .count()
            .reset_index(name="ticker_count")
            .sort_values("ticker_count", ascending=False)
        )
        metric_coverage = [
            {
                "metric_name": "revenue",
                "companyfacts_periods": int(df["companyfacts_revenue_periods"].fillna(0).sum()),
                "filing_periods": int(df["filing_revenue_periods"].fillna(0).sum()),
                "supported_issuers": int((df["filing_revenue_periods"].fillna(0) > 0).sum()),
            },
            {
                "metric_name": "operating_income",
                "companyfacts_periods": int(df["companyfacts_operating_income_periods"].fillna(0).sum()),
                "filing_periods": int(df["filing_operating_income_periods"].fillna(0).sum()),
                "supported_issuers": int((df["filing_operating_income_periods"].fillna(0) > 0).sum()),
            },
        ]
        segment_type_coverage = [
            {"segment_type": "business", "rows": int(df["business_rows"].fillna(0).sum()), "issuers": int((df["business_rows"].fillna(0) > 0).sum())},
            {"segment_type": "geography", "rows": int(df["geography_rows"].fillna(0).sum()), "issuers": int((df["geography_rows"].fillna(0) > 0).sum())},
            {"segment_type": "product", "rows": int(df["product_rows"].fillna(0).sum()), "issuers": int((df["product_rows"].fillna(0) > 0).sum())},
        ]
        confidence_summary = [
            {
                "confidence": "high",
                "rows": int(df["high_confidence_rows"].fillna(0).sum()),
                "issuers": int((df["high_confidence_rows"].fillna(0) > 0).sum()),
            },
            {
                "confidence": "medium",
                "rows": int(df["medium_confidence_rows"].fillna(0).sum()),
                "issuers": int((df["medium_confidence_rows"].fillna(0) > 0).sum()),
            },
            {
                "confidence": "low",
                "rows": int(df["low_confidence_rows"].fillna(0).sum()),
                "issuers": int((df["low_confidence_rows"].fillna(0) > 0).sum()),
            },
        ]
        confidence_band_counts = (
            df["confidence_band"].value_counts(dropna=False).to_dict()
            if "confidence_band" in df.columns
            else {}
        )
        recovery_bucket_counts = (
            df["recovery_bucket"].value_counts(dropna=False).to_dict()
            if "recovery_bucket" in df.columns
            else {}
        )
        recoverability_class_counts = (
            df["recoverability_class"].value_counts(dropna=False).to_dict()
            if "recoverability_class" in df.columns
            else {}
        )
        recovery_confidence_counts = (
            df["recovery_confidence"].value_counts(dropna=False).to_dict()
            if "recovery_confidence" in df.columns
            else {}
        )
        return self._json_safe(
            {
                "survey_run_id": run_id,
                "rows": df.head(max(1, int(limit))).to_dict(orient="records"),
                "cluster_counts": cluster_counts.to_dict(orient="records"),
                "cluster_detail_counts": cluster_detail_counts.to_dict(orient="records"),
                "source_poor_reason_counts": source_poor_reason_counts.to_dict(orient="records"),
                "metric_coverage": metric_coverage,
                "segment_type_coverage": segment_type_coverage,
                "confidence_summary": confidence_summary,
                "confidence_band_counts": confidence_band_counts,
                "recovery_bucket_counts": recovery_bucket_counts,
                "recoverability_class_counts": recoverability_class_counts,
                "recovery_confidence_counts": recovery_confidence_counts,
                "fetch_required_count": int(df["fetch_required"].fillna(False).sum()) if "fetch_required" in df.columns else 0,
                "companyfacts_available_issuers": int(df["companyfacts_available"].fillna(False).sum()),
                "filing_only_issuers": int(((~df["companyfacts_available"].fillna(False)) & (df["filing_available"].fillna(False))).sum()),
                "poor_source_issuers": int((df["issuer_cluster"] == "segment_source_poor").sum()),
                "overlap_issuers": int(df["overlap_available"].fillna(False).sum()),
            }
        )

    def cluster_summary(
        self,
        *,
        survey_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        run_id = survey_run_id or self._latest_coverage_run_id()
        if not run_id:
            return {"survey_run_id": None, "rows": [], "cluster_tickers": {}}
        df = self.db.fetch_df(
            """
            SELECT *
            FROM segment_issuer_source_coverage
            WHERE survey_run_id = ?
            ORDER BY issuer_cluster, ticker
            """,
            [run_id],
        )
        if df.empty:
            return {"survey_run_id": run_id, "rows": [], "cluster_tickers": {}}
        summary = (
            df.groupby(
                [
                    "issuer_cluster",
                    "issuer_cluster_detail",
                    "source_lane",
                    "metric_cluster",
                    "segment_type_profile",
                    "source_poor_reason",
                    "confidence_band",
                    "recovery_bucket",
                    "recoverability_class",
                ],
                dropna=False,
            )
            .agg(
                ticker_count=("ticker", "count"),
                companyfacts_issuers=("companyfacts_available", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                filing_issuers=("filing_available", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                overlap_issuers=("overlap_available", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                filing_only_validated_candidates=("filing_only_validated_candidate", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
            )
            .reset_index()
            .sort_values(["ticker_count", "overlap_issuers"], ascending=[False, False])
        )
        cluster_tickers = {
            str(cluster): group["ticker"].astype(str).tolist()
            for cluster, group in df.groupby("issuer_cluster", dropna=False)
        }
        cluster_detail_tickers = {
            str(cluster): group["ticker"].astype(str).tolist()
            for cluster, group in df.groupby("issuer_cluster_detail", dropna=False)
        }
        return self._json_safe(
            {
                "survey_run_id": run_id,
                "rows": summary.head(max(1, int(limit))).to_dict(orient="records"),
                "cluster_tickers": cluster_tickers,
                "cluster_detail_tickers": cluster_detail_tickers,
                "recoverability_class_counts": df["recoverability_class"].value_counts(dropna=False).to_dict() if "recoverability_class" in df.columns else {},
            }
        )

    def recovery_summary(
        self,
        *,
        survey_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        run_id = survey_run_id or self._latest_recovery_run_id() or self._latest_coverage_run_id()
        if not run_id or not self.db.table_exists("segment_recovery_candidate_registry"):
            return {"survey_run_id": run_id, "rows": [], "bucket_counts": {}, "candidate_tickers": {}}
        df = self.db.fetch_df(
            """
            SELECT *
            FROM segment_recovery_candidate_registry
            WHERE survey_run_id = ?
            ORDER BY recovery_score DESC, ticker
            """,
            [run_id],
        )
        if df.empty:
            return {"survey_run_id": run_id, "rows": [], "bucket_counts": {}, "candidate_tickers": {}}
        summary = (
            df.groupby(
                [
                    "recovery_bucket",
                    "recoverability_class",
                    "recovery_reason",
                    "recommended_action",
                    "target_source",
                    "expected_gain",
                    "recovery_confidence",
                ],
                dropna=False,
            )
            .agg(
                ticker_count=("ticker", "count"),
                avg_score=("recovery_score", "mean"),
                max_score=("recovery_score", "max"),
            )
            .reset_index()
            .sort_values(["ticker_count", "avg_score"], ascending=[False, False])
        )
        candidate_tickers = {
            str(bucket): group["ticker"].astype(str).tolist()
            for bucket, group in df.groupby("recovery_bucket", dropna=False)
        }
        class_tickers = {
            str(bucket): group["ticker"].astype(str).tolist()
            for bucket, group in df.groupby("recoverability_class", dropna=False)
        }
        return self._json_safe(
            {
                "survey_run_id": run_id,
                "rows": summary.head(max(1, int(limit))).to_dict(orient="records"),
                "bucket_counts": df["recovery_bucket"].value_counts(dropna=False).to_dict(),
                "recoverability_class_counts": df["recoverability_class"].value_counts(dropna=False).to_dict() if "recoverability_class" in df.columns else {},
                "recovery_confidence_counts": df["recovery_confidence"].value_counts(dropna=False).to_dict() if "recovery_confidence" in df.columns else {},
                "fetch_required_count": int(df["fetch_required"].fillna(False).sum()) if "fetch_required" in df.columns else 0,
                "candidate_tickers": candidate_tickers,
                "class_tickers": class_tickers,
                "sample_rows": df.head(max(1, int(limit))).to_dict(orient="records"),
            }
        )

    def _build_recovery_candidate_frame(
        self,
        *,
        coverage: pd.DataFrame,
        survey_run_id: str,
    ) -> pd.DataFrame:
        if coverage.empty:
            return pd.DataFrame(columns=self._recovery_result_columns())
        created_at = pd.Timestamp.utcnow()
        rows: list[dict[str, Any]] = []
        for item in coverage.to_dict(orient="records"):
            ticker = self._coerce_str(item.get("ticker"))
            market = self._coerce_str(item.get("market"))
            rows.append(
                {
                    "candidate_result_id": self._recovery_result_id(
                        survey_run_id=survey_run_id,
                        ticker=ticker,
                        market=market,
                    ),
                    "survey_run_id": survey_run_id,
                    "ticker": ticker,
                    "market": market,
                    "issuer_cluster": self._coerce_str(item.get("issuer_cluster")),
                    "issuer_cluster_detail": self._coerce_str(item.get("issuer_cluster_detail")),
                    "source_poor_reason": self._coerce_str(item.get("source_poor_reason")),
                    "confidence_score": self._coerce_float(item.get("confidence_score")),
                    "confidence_band": self._coerce_str(item.get("confidence_band")),
                    "recovery_bucket": self._coerce_str(item.get("recovery_bucket")),
                    "recoverability_class": self._coerce_str(item.get("recoverability_class")),
                    "recovery_score": self._coerce_float(item.get("recovery_score")),
                    "recovery_reason": self._coerce_str(item.get("recovery_reason")),
                    "recommended_action": self._coerce_str(item.get("recommended_action")),
                    "target_source": self._coerce_str(item.get("target_source")) or self._coerce_str(item.get("best_source_name")) or "filing_cache_or_instance",
                    "fetch_required": bool(item.get("fetch_required", False)),
                    "expected_gain": self._coerce_str(item.get("expected_gain")),
                    "priority_rank": self._coerce_int(item.get("priority_rank")),
                    "recovery_confidence": self._coerce_str(item.get("recovery_confidence")),
                    "cache_presence_detail": self._coerce_str(item.get("cache_presence_detail")),
                    "filing_meta_strength": self._coerce_str(item.get("filing_meta_strength")),
                    "source_signal_strength": self._coerce_str(item.get("source_signal_strength")),
                    "filing_meta_count": self._coerce_int(item.get("filing_meta_count")),
                    "filing_10q_count": self._coerce_int(item.get("filing_10q_count")),
                    "filing_10k_count": self._coerce_int(item.get("filing_10k_count")),
                    "parser_candidate_flag": bool(item.get("parser_candidate_flag", False)),
                    "created_at": created_at,
                }
            )
        frame = pd.DataFrame(rows)
        return frame[self._recovery_result_columns()].reset_index(drop=True) if not frame.empty else pd.DataFrame(columns=self._recovery_result_columns())

    def _load_policies(self) -> pd.DataFrame:
        return self.db.fetch_df(
            """
            SELECT *
            FROM segment_metric_policy_registry
            WHERE is_active
            ORDER BY metric_name, statement_type
            """
        )

    def _load_cluster_policies(self) -> pd.DataFrame:
        return self.db.fetch_df(
            """
            SELECT *
            FROM segment_cluster_policy_registry
            WHERE is_active
            ORDER BY issuer_cluster
            """
        )

    def _resolve_target_tickers(
        self,
        *,
        tickers: str | list[str] | None,
        market: str,
        start_date: date,
    ) -> list[str]:
        if tickers:
            if isinstance(tickers, str):
                return [item.strip().upper() for item in tickers.split(",") if item.strip()]
            return [str(item).strip().upper() for item in tickers if str(item).strip()]
        frame = self.db.fetch_df(
            """
            SELECT DISTINCT UPPER(ticker) AS ticker
            FROM segment_facts_quarterly
            WHERE market = ?
              AND period_end >= ?
            ORDER BY 1
            """,
            [market.lower(), start_date],
        )
        return frame["ticker"].astype(str).str.upper().tolist() if not frame.empty else []

    def _load_segment_rows(
        self,
        *,
        tickers: list[str],
        market: str,
        start_date: date,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        sql = f"""
            SELECT
                UPPER(ticker) AS ticker,
                market,
                period_end,
                available_date,
                segment_type,
                segment_name,
                metric,
                value,
                accession,
                source,
                filing_date,
                accepted_at
            FROM segment_facts_quarterly
            WHERE market = ?
              AND period_end >= ?
              AND UPPER(ticker) IN {_quoted_literal_list(tickers)}
              AND metric IN ('revenue', 'operating_income')
        """
        frame = self.db.fetch_df(sql, [market.lower(), start_date])
        if frame.empty:
            return frame
        for col in ("period_end", "available_date", "filing_date"):
            frame[col] = pd.to_datetime(frame.get(col), errors="coerce").dt.normalize()
        frame["accepted_at"] = pd.to_datetime(frame.get("accepted_at"), errors="coerce", utc=True)
        frame["metric_name"] = frame["metric"].astype(str).str.strip().str.lower()
        frame["statement_type"] = "quarterly"
        frame["source_family"] = frame["source"].astype(str).eq(SEGMENT_COMPANYFACTS_SOURCE).map({True: "companyfacts", False: "filing"})
        frame["segment_type_normalized"] = frame.apply(
            lambda row: self._normalize_segment_type(row.get("segment_type"), row.get("segment_name")),
            axis=1,
        )
        frame["segment_name_normalized"] = frame["segment_name"].map(self._normalize_segment_name)
        frame["source_priority"] = frame["source"].astype(str).map(SEGMENT_FILING_SOURCE_PRIORITY).fillna(99).astype(int)
        frame["source_confidence"] = frame["source"].astype(str).map(SEGMENT_FILING_SOURCE_CONFIDENCE).fillna("n/a")
        return frame

    def _load_segment_extract_logs(
        self,
        *,
        tickers: list[str],
        market: str,
    ) -> pd.DataFrame:
        if not tickers or not self.db.table_exists("segment_extract_log"):
            return pd.DataFrame()
        sql = f"""
            SELECT
                UPPER(ticker) AS ticker,
                market,
                accession,
                method,
                status,
                reason,
                created_at
            FROM segment_extract_log
            WHERE market = ?
              AND UPPER(ticker) IN {_quoted_literal_list(tickers)}
        """
        frame = self.db.fetch_df(sql, [market.lower()])
        if frame.empty:
            return frame
        frame["created_at"] = pd.to_datetime(frame.get("created_at"), errors="coerce", utc=True)
        return frame

    def _load_recent_filing_meta(
        self,
        *,
        tickers: list[str],
        market: str,
        start_date: date,
    ) -> pd.DataFrame:
        if not tickers or not self.db.table_exists("filings"):
            return pd.DataFrame()
        sql = f"""
            SELECT
                UPPER(ticker) AS ticker,
                market,
                accession,
                form_type,
                report_date,
                filing_date,
                accepted_at
            FROM filings
            WHERE market = ?
              AND UPPER(ticker) IN {_quoted_literal_list(tickers)}
              AND COALESCE(report_date, filing_date) >= ?
        """
        frame = self.db.fetch_df(sql, [market.lower(), start_date])
        if frame.empty:
            return frame
        frame["report_date"] = pd.to_datetime(frame.get("report_date"), errors="coerce").dt.normalize()
        frame["filing_date"] = pd.to_datetime(frame.get("filing_date"), errors="coerce").dt.normalize()
        frame["accepted_at"] = pd.to_datetime(frame.get("accepted_at"), errors="coerce", utc=True)
        return frame

    def _build_result_frame(
        self,
        *,
        facts: pd.DataFrame,
        policies: pd.DataFrame,
        comparison_run_id: str,
        coverage: pd.DataFrame,
        cluster_policies: pd.DataFrame,
    ) -> pd.DataFrame:
        created_at = pd.Timestamp.utcnow()
        if facts.empty or policies.empty:
            return pd.DataFrame(columns=self._result_columns())
        coverage_map = {
            (self._coerce_str(row.get("ticker")), self._coerce_str(row.get("market"))): row
            for row in coverage.to_dict(orient="records")
        }
        cluster_policy_map = {
            self._coerce_str(row.get("issuer_cluster")): row
            for row in cluster_policies.to_dict(orient="records")
        }
        companyfacts_rows = self._collapse_segment_source_rows(facts.loc[facts["source_family"] == "companyfacts"].copy(), family="companyfacts")
        filing_rows = self._collapse_segment_source_rows(facts.loc[facts["source_family"] == "filing"].copy(), family="filing")

        join_keys = ["ticker", "market", "period_end", "statement_type", "metric_name", "segment_type_normalized", "segment_name_normalized"]
        merged = companyfacts_rows.add_prefix("cf__").merge(
            filing_rows.add_prefix("fil__"),
            left_on=[f"cf__{key}" for key in join_keys],
            right_on=[f"fil__{key}" for key in join_keys],
            how="outer",
        )
        for key in join_keys:
            merged[key] = self._coalesce_prefer_left(merged.get(f"cf__{key}"), merged.get(f"fil__{key}"))

        cf_metric_counts = (
            facts.loc[facts["source_family"] == "companyfacts"]
            .groupby(["ticker", "period_end", "metric_name"], dropna=False)
            .size()
            .to_dict()
        )
        filing_metric_counts = (
            facts.loc[facts["source_family"] == "filing"]
            .groupby(["ticker", "period_end", "metric_name"], dropna=False)
            .size()
            .to_dict()
        )
        cf_type_counts = (
            facts.loc[facts["source_family"] == "companyfacts"]
            .groupby(["ticker", "period_end", "metric_name", "segment_type_normalized"], dropna=False)
            .size()
            .to_dict()
        )
        filing_type_counts = (
            facts.loc[facts["source_family"] == "filing"]
            .groupby(["ticker", "period_end", "metric_name", "segment_type_normalized"], dropna=False)
            .size()
            .to_dict()
        )
        filing_name_pool = self._build_segment_value_pool(facts.loc[facts["source_family"] == "filing"].copy())
        companyfacts_name_pool = self._build_segment_value_pool(facts.loc[facts["source_family"] == "companyfacts"].copy())
        filing_consensus = self._build_filing_consensus_map(facts.loc[facts["source_family"] == "filing"].copy())
        filing_temporal_consistency = self._build_filing_temporal_consistency_map(facts.loc[facts["source_family"] == "filing"].copy())

        policy_map = {
            (str(row["metric_name"]), str(row["statement_type"])): row
            for row in policies.to_dict(orient="records")
        }
        rows: list[dict[str, Any]] = []
        for _, aligned in merged.iterrows():
            metric_name = self._coerce_str(aligned.get("metric_name"))
            statement_type = self._coerce_str(aligned.get("statement_type")) or "quarterly"
            policy = policy_map.get((str(metric_name), str(statement_type)))
            if not policy:
                continue
            ticker = self._coerce_str(aligned.get("ticker"))
            market = self._coerce_str(aligned.get("market"))
            period_end = pd.to_datetime(aligned.get("period_end"), errors="coerce")
            coverage_row = coverage_map.get((ticker, market), {})
            issuer_cluster = self._coerce_str(coverage_row.get("issuer_cluster"))
            issuer_cluster_detail = self._coerce_str(coverage_row.get("issuer_cluster_detail"))
            cluster_policy = cluster_policy_map.get(issuer_cluster, {})
            source_lane = self._coerce_str(coverage_row.get("source_lane")) or "companyfacts_vs_filing"
            coverage_reason = self._coerce_str(coverage_row.get("coverage_reason"))
            issuer_confidence_score = self._coerce_float(coverage_row.get("confidence_score"))
            issuer_confidence_band = self._coerce_str(coverage_row.get("confidence_band"))
            companyfacts_available = bool(coverage_row.get("companyfacts_available", False))
            filing_available = bool(coverage_row.get("filing_available", False))
            overlap_available = bool(coverage_row.get("overlap_available", False))
            cf_value = self._coerce_float(aligned.get("cf__value"))
            filing_value = self._coerce_float(aligned.get("fil__value"))
            filing_source_confidence = self._source_confidence(self._coerce_str(aligned.get("fil__source")))
            filing_source_rank = self._source_rank(self._coerce_str(aligned.get("fil__source")))
            missing_on_companyfacts = cf_value is None
            missing_on_filing = filing_value is None
            abs_diff = abs(cf_value - filing_value) if cf_value is not None and filing_value is not None else None
            pct_diff = self._pct_diff(abs_diff, cf_value) if abs_diff is not None else None
            tolerance_type = str(policy.get("tolerance_type") or "relative")
            tolerance_value = float(policy.get("tolerance_value") or 0.0)
            tolerance_breach = self._tolerance_breach(abs_diff, pct_diff, cf_value, filing_value, tolerance_type, tolerance_value)
            diagnostic_codes: list[str] = []
            if issuer_cluster:
                diagnostic_codes.append(f"issuer_cluster_{issuer_cluster}")
            if issuer_cluster_detail:
                diagnostic_codes.append(f"issuer_cluster_detail_{issuer_cluster_detail}")
            if source_lane:
                diagnostic_codes.append(f"source_lane_{source_lane}")
            if self._coerce_str(aligned.get("cf__segment_name")) != self._coerce_str(aligned.get("fil__segment_name")) and not missing_on_companyfacts and not missing_on_filing:
                diagnostic_codes.append("segment_name_normalized")
            if self._coerce_str(aligned.get("cf__segment_type")) != self._coerce_str(aligned.get("fil__segment_type")) and not missing_on_companyfacts and not missing_on_filing:
                diagnostic_codes.append("segment_type_normalized")
            if self._coerce_int(aligned.get("cf__row_count")) and int(aligned.get("cf__row_count")) > 1:
                diagnostic_codes.append("companyfacts_multiple_candidates")
            if self._coerce_int(aligned.get("fil__row_count")) and int(aligned.get("fil__row_count")) > 1:
                diagnostic_codes.append("filing_multiple_candidates")
            filing_source = self._coerce_str(aligned.get("fil__source"))
            if filing_source_confidence:
                diagnostic_codes.append(f"filing_source_confidence_{filing_source_confidence}")
            if issuer_confidence_band:
                diagnostic_codes.append(f"issuer_confidence_band_{issuer_confidence_band}")
            if filing_source == "html_table_mvp":
                diagnostic_codes.append("filing_html_table_fallback")

            lookup_key = (ticker, self._to_date(period_end), metric_name)
            type_key = (ticker, self._to_date(period_end), metric_name, self._coerce_str(aligned.get("segment_type_normalized")))
            consensus_key = (
                ticker,
                market,
                self._to_date(period_end),
                statement_type,
                metric_name,
                self._coerce_str(aligned.get("segment_type_normalized")),
                self._coerce_str(aligned.get("segment_name_normalized")),
            )
            if missing_on_filing:
                diagnostic_codes.append("missing_filing_segment_metric")
                if filing_metric_counts.get(lookup_key, 0) == 0:
                    diagnostic_codes.append("filing_segment_reference_missing")
                elif filing_type_counts.get(type_key, 0) > 0:
                    if self._has_similar_segment_value(filing_name_pool, type_key, cf_value, tolerance_value):
                        diagnostic_codes.append("segment_member_normalization_candidate")
                    else:
                        diagnostic_codes.append("segment_dimension_scope_gap_candidate")
            if missing_on_companyfacts:
                diagnostic_codes.append("missing_companyfacts_segment_metric")
                if cf_metric_counts.get(lookup_key, 0) == 0:
                    diagnostic_codes.append("companyfacts_segment_reference_missing")
                elif cf_type_counts.get(type_key, 0) > 0:
                    if self._has_similar_segment_value(companyfacts_name_pool, type_key, filing_value, tolerance_value):
                        diagnostic_codes.append("segment_member_normalization_candidate")
                    else:
                        diagnostic_codes.append("segment_dimension_scope_gap_candidate")
                filing_consensus_item = filing_consensus.get(consensus_key)
                if filing_consensus_item:
                    if filing_consensus_item["source_count"] >= 2 and filing_consensus_item["within_tolerance"]:
                        diagnostic_codes.append("filing_source_consensus")
                    elif filing_consensus_item["source_count"] >= 2:
                        diagnostic_codes.append("filing_source_conflict")
                if source_lane.startswith("filing_only"):
                    diagnostic_codes.append("companyfacts_source_unavailable_for_issuer")
                    if coverage_reason:
                        diagnostic_codes.extend(self._split_codes(coverage_reason))
                    if self._cluster_supports_metric(cluster_policy, metric_name):
                        temporal_key = (
                            ticker,
                            metric_name,
                            self._coerce_str(aligned.get("segment_type_normalized")),
                            self._coerce_str(aligned.get("segment_name_normalized")),
                        )
                        temporal_item = filing_temporal_consistency.get(temporal_key)
                        if temporal_item and temporal_item["period_count"] >= 2:
                            diagnostic_codes.append("filing_only_period_consistency")
                            if temporal_item["source_count"] >= 1:
                                diagnostic_codes.append("filing_only_candidate")
                    else:
                        diagnostic_codes.append("cluster_metric_not_supported")
            if tolerance_breach and not missing_on_companyfacts and not missing_on_filing:
                if "segment_name_normalized" in diagnostic_codes:
                    diagnostic_codes.append("segment_member_scope_gap")
                elif "segment_type_normalized" in diagnostic_codes:
                    diagnostic_codes.append("segment_dimension_scope_gap")
                else:
                    diagnostic_codes.append("segment_metric_semantics_gap")
            if self._is_intersegment_name(self._coerce_str(aligned.get("segment_name_normalized"))):
                diagnostic_codes.append("segment_intersegment_scope")

            comparison_status = self._comparison_status(
                missing_on_companyfacts,
                missing_on_filing,
                tolerance_breach,
                diagnostic_codes=diagnostic_codes,
            )
            semantic_gap_class = self._semantic_gap_class(diagnostic_codes)
            mismatch_class = self._mismatch_class(
                comparison_status=comparison_status,
                diagnostic_codes=diagnostic_codes,
            )
            rows.append(
                {
                    "result_id": self._result_id(
                        comparison_run_id=comparison_run_id,
                        ticker=ticker,
                        period_end=self._to_date(period_end),
                        metric_name=metric_name,
                        segment_type=self._coerce_str(aligned.get("segment_type_normalized")),
                        segment_name=self._coerce_str(aligned.get("segment_name_normalized")),
                    ),
                    "ticker": ticker,
                    "market": market,
                    "period_end": self._to_date(period_end),
                    "statement_type": statement_type,
                    "metric_name": metric_name,
                    "issuer_cluster": issuer_cluster,
                    "source_cluster": self._coerce_str(coverage_row.get("source_cluster")),
                    "metric_cluster": self._coerce_str(coverage_row.get("metric_cluster")),
                    "segment_type_profile": self._coerce_str(coverage_row.get("segment_type_profile")),
                    "source_lane": source_lane,
                    "issuer_cluster_detail": issuer_cluster_detail,
                    "coverage_reason": coverage_reason,
                    "companyfacts_available": companyfacts_available,
                    "filing_available": filing_available,
                    "overlap_available": overlap_available,
                    "segment_type_companyfacts": self._coerce_str(aligned.get("cf__segment_type")),
                    "segment_type_filing": self._coerce_str(aligned.get("fil__segment_type")),
                    "segment_type_normalized": self._coerce_str(aligned.get("segment_type_normalized")),
                    "segment_name_companyfacts": self._coerce_str(aligned.get("cf__segment_name")),
                    "segment_name_filing": self._coerce_str(aligned.get("fil__segment_name")),
                    "segment_name_normalized": self._coerce_str(aligned.get("segment_name_normalized")),
                    "companyfacts_value": cf_value,
                    "filing_value": filing_value,
                    "abs_diff": abs_diff,
                    "pct_diff": pct_diff,
                    "missing_on_companyfacts": missing_on_companyfacts,
                    "missing_on_filing": missing_on_filing,
                    "comparison_status": comparison_status,
                    "mismatch_class": mismatch_class,
                    "semantic_gap_class": semantic_gap_class,
                    "companyfacts_source": self._coerce_str(aligned.get("cf__source")),
                    "filing_source": filing_source,
                    "filing_source_confidence": filing_source_confidence,
                    "filing_source_rank": filing_source_rank,
                    "issuer_confidence_score": issuer_confidence_score,
                    "issuer_confidence_band": issuer_confidence_band,
                    "companyfacts_available_date": self._to_date(aligned.get("cf__available_date")),
                    "filing_available_date": self._to_date(aligned.get("fil__available_date")),
                    "filing_accession": self._coerce_str(aligned.get("fil__accession")),
                    "companyfacts_row_count": self._coerce_int(aligned.get("cf__row_count")),
                    "filing_row_count": self._coerce_int(aligned.get("fil__row_count")),
                    "compare_mode": COMPARE_MODE_DEFAULT,
                    "value_mode_used": VALUE_MODE_REPORTED,
                    "diagnostic_code": "|".join(dict.fromkeys(code for code in diagnostic_codes if code)),
                    "notes": self._build_notes(policy=policy, diagnostic_codes=diagnostic_codes),
                    "comparison_run_id": comparison_run_id,
                    "created_at": created_at,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=self._result_columns())
        for col in self._result_columns():
            if col not in frame.columns:
                frame[col] = None
        return frame[self._result_columns()].reset_index(drop=True)

    def _build_source_coverage_summary(
        self,
        *,
        facts: pd.DataFrame,
        logs: pd.DataFrame,
        filing_meta: pd.DataFrame,
        tickers: list[str],
        market: str,
        start_date: date,
        survey_run_id: str,
        cluster_policies: pd.DataFrame,
    ) -> pd.DataFrame:
        policy_map = {
            self._coerce_str(row.get("issuer_cluster")): row
            for row in cluster_policies.to_dict(orient="records")
        }
        rows: list[dict[str, Any]] = []
        created_at = pd.Timestamp.utcnow()
        all_facts = facts.copy() if facts is not None else pd.DataFrame()
        all_logs = logs.copy() if logs is not None else pd.DataFrame()
        all_filing_meta = filing_meta.copy() if filing_meta is not None else pd.DataFrame()
        for ticker in tickers:
            frame = all_facts.loc[all_facts["ticker"] == ticker].copy() if not all_facts.empty else pd.DataFrame()
            companyfacts = frame.loc[frame["source_family"] == "companyfacts"].copy() if not frame.empty else pd.DataFrame()
            filing = frame.loc[frame["source_family"] == "filing"].copy() if not frame.empty else pd.DataFrame()
            log_frame = all_logs.loc[all_logs["ticker"] == ticker].copy() if not all_logs.empty else pd.DataFrame()
            filing_meta_frame = (
                all_filing_meta.loc[all_filing_meta["ticker"] == ticker].copy() if not all_filing_meta.empty else pd.DataFrame()
            )
            companyfacts_rows = int(len(companyfacts))
            filing_rows = int(len(filing))
            cf_periods = int(companyfacts["period_end"].nunique()) if not companyfacts.empty else 0
            filing_periods = int(filing["period_end"].nunique()) if not filing.empty else 0
            overlap_periods = 0
            if not companyfacts.empty and not filing.empty:
                overlap_periods = len(set(companyfacts["period_end"].dropna().tolist()) & set(filing["period_end"].dropna().tolist()))
            cf_rev_periods = int(companyfacts.loc[companyfacts["metric_name"] == "revenue", "period_end"].nunique()) if not companyfacts.empty else 0
            cf_op_periods = int(companyfacts.loc[companyfacts["metric_name"] == "operating_income", "period_end"].nunique()) if not companyfacts.empty else 0
            filing_rev_periods = int(filing.loc[filing["metric_name"] == "revenue", "period_end"].nunique()) if not filing.empty else 0
            filing_op_periods = int(filing.loc[filing["metric_name"] == "operating_income", "period_end"].nunique()) if not filing.empty else 0
            business_rows = int(frame["segment_type_normalized"].eq("business").sum()) if not frame.empty else 0
            geography_rows = int(frame["segment_type_normalized"].eq("geography").sum()) if not frame.empty else 0
            product_rows = int(frame["segment_type_normalized"].eq("product").sum()) if not frame.empty else 0
            other_rows = int((~frame["segment_type_normalized"].isin(["business", "geography", "product"])).sum()) if not frame.empty else 0
            distinct_segment_keys = int(frame[["metric_name", "segment_type_normalized", "segment_name_normalized"]].drop_duplicates().shape[0]) if not frame.empty else 0
            filing_source_count = int(filing["source"].nunique()) if not filing.empty else 0
            ixbrl_rows = int(filing["source"].eq("ixbrl_dimension").sum()) if not filing.empty else 0
            xbrl_instance_rows = int(filing["source"].isin(["xbrl_instance", "xbrl_instance_dimension"]).sum()) if not filing.empty else 0
            html_table_rows = int(filing["source"].eq("html_table_mvp").sum()) if not filing.empty else 0
            ixbrl_revenue_periods = self._source_metric_periods(filing, "ixbrl_dimension", "revenue")
            ixbrl_operating_income_periods = self._source_metric_periods(filing, "ixbrl_dimension", "operating_income")
            xbrl_instance_revenue_periods = self._source_metric_periods(filing, "xbrl_instance", "revenue") + self._source_metric_periods(filing, "xbrl_instance_dimension", "revenue")
            xbrl_instance_operating_income_periods = self._source_metric_periods(filing, "xbrl_instance", "operating_income") + self._source_metric_periods(filing, "xbrl_instance_dimension", "operating_income")
            html_table_revenue_periods = self._source_metric_periods(filing, "html_table_mvp", "revenue")
            html_table_operating_income_periods = self._source_metric_periods(filing, "html_table_mvp", "operating_income")
            high_confidence_rows = int(filing["source_confidence"].eq("high").sum()) if not filing.empty else 0
            medium_confidence_rows = int(filing["source_confidence"].eq("medium").sum()) if not filing.empty else 0
            low_confidence_rows = int(filing["source_confidence"].eq("low").sum()) if not filing.empty else 0
            filing_meta_count = int(len(filing_meta_frame))
            filing_10q_count = int(filing_meta_frame["form_type"].astype(str).str.upper().isin(["10-Q", "10-Q/A"]).sum()) if not filing_meta_frame.empty else 0
            filing_10k_count = int(filing_meta_frame["form_type"].astype(str).str.upper().isin(["10-K", "10-K/A"]).sum()) if not filing_meta_frame.empty else 0
            log_signals = self._extract_log_signals(log_frame)
            source_cluster = self._classify_source_cluster(
                companyfacts_rows=companyfacts_rows,
                filing_rows=filing_rows,
                companyfacts_periods=cf_periods,
                filing_periods=filing_periods,
                overlap_periods=overlap_periods,
            )
            metric_cluster = self._classify_metric_cluster(
                filing_revenue_periods=filing_rev_periods,
                filing_operating_income_periods=filing_op_periods,
            )
            segment_type_profile = self._classify_segment_type_profile(
                business_rows=business_rows,
                geography_rows=geography_rows,
                product_rows=product_rows,
                other_rows=other_rows,
            )
            issuer_cluster = self._combine_issuer_cluster(
                source_cluster=source_cluster,
                metric_cluster=metric_cluster,
            )
            cluster_policy = policy_map.get(issuer_cluster, {})
            companyfacts_available = companyfacts_rows > 0
            filing_available = filing_rows > 0
            overlap_available = overlap_periods > 0
            filing_only_validated_candidate = (
                issuer_cluster in {"filing_only_full", "filing_only_revenue_only"}
                and filing_periods >= 2
            )
            source_poor_reason, source_poor_reason_detail, parser_candidate_flag = self._classify_source_poor_reason(
                issuer_cluster=issuer_cluster,
                filing_rows=filing_rows,
                filing_meta_count=int(len(filing_meta_frame)),
                log_frame=log_frame,
            )
            issuer_cluster_detail = self._classify_issuer_cluster_detail(
                issuer_cluster=issuer_cluster,
                filing_revenue_periods=filing_rev_periods,
                filing_operating_income_periods=filing_op_periods,
                ixbrl_rows=ixbrl_rows,
                xbrl_instance_rows=xbrl_instance_rows,
                html_table_rows=html_table_rows,
                source_poor_reason=source_poor_reason,
                parser_candidate_flag=parser_candidate_flag,
                filing=filing,
                log_signals=log_signals,
                distinct_segment_keys=distinct_segment_keys,
            )
            best_source_name = self._best_filing_source_name(
                ixbrl_rows=ixbrl_rows,
                xbrl_instance_rows=xbrl_instance_rows,
                html_table_rows=html_table_rows,
            )
            best_source_rank = self._source_rank(best_source_name)
            cache_presence_detail = self._classify_cache_presence_detail(
                ixbrl_rows=ixbrl_rows,
                xbrl_instance_rows=xbrl_instance_rows,
                html_table_rows=html_table_rows,
                filing_meta_count=filing_meta_count,
                source_poor_reason=source_poor_reason,
            )
            filing_meta_strength = self._classify_filing_meta_strength(
                filing_meta_count=filing_meta_count,
                filing_10q_count=filing_10q_count,
                filing_10k_count=filing_10k_count,
            )
            consistency_score = self._consistency_score(
                filing_periods=filing_periods,
                filing_revenue_periods=filing_rev_periods,
                filing_operating_income_periods=filing_op_periods,
                filing_source_count=filing_source_count,
            )
            source_signal_strength = self._classify_source_signal_strength(
                issuer_cluster=issuer_cluster,
                issuer_cluster_detail=issuer_cluster_detail,
                source_poor_reason=source_poor_reason,
                parser_candidate_flag=parser_candidate_flag,
                filing_rows=filing_rows,
                filing_revenue_periods=filing_rev_periods,
                filing_operating_income_periods=filing_op_periods,
                ixbrl_rows=ixbrl_rows,
                xbrl_instance_rows=xbrl_instance_rows,
                html_table_rows=html_table_rows,
            )
            confidence_score, confidence_band, confidence_reason = self._score_segment_source_confidence(
                issuer_cluster=issuer_cluster,
                issuer_cluster_detail=issuer_cluster_detail,
                best_source_rank=best_source_rank,
                filing_periods=filing_periods,
                filing_revenue_periods=filing_rev_periods,
                filing_operating_income_periods=filing_op_periods,
                filing_source_count=filing_source_count,
                distinct_segment_keys=distinct_segment_keys,
                consistency_score=consistency_score,
                high_confidence_rows=high_confidence_rows,
                medium_confidence_rows=medium_confidence_rows,
                low_confidence_rows=low_confidence_rows,
                filing_meta_strength=filing_meta_strength,
                cache_presence_detail=cache_presence_detail,
                source_signal_strength=source_signal_strength,
            )
            (
                recovery_bucket,
                recoverability_class,
                recovery_score,
                recovery_reason,
                recommended_action,
                fetch_required,
                target_source,
                expected_gain,
                priority_rank,
                recovery_confidence,
            ) = self._classify_recovery_candidate(
                issuer_cluster=issuer_cluster,
                issuer_cluster_detail=issuer_cluster_detail,
                source_poor_reason=source_poor_reason,
                filing_meta_count=filing_meta_count,
                filing_10q_count=filing_10q_count,
                filing_10k_count=filing_10k_count,
                parser_candidate_flag=parser_candidate_flag,
                confidence_score=confidence_score,
                cache_presence_detail=cache_presence_detail,
                filing_meta_strength=filing_meta_strength,
                source_signal_strength=source_signal_strength,
            )
            coverage_reason = "|".join(
                self._build_coverage_reason_codes(
                    companyfacts_rows=companyfacts_rows,
                    filing_rows=filing_rows,
                    overlap_periods=overlap_periods,
                    filing_revenue_periods=filing_rev_periods,
                    filing_operating_income_periods=filing_op_periods,
                    filing_source_count=filing_source_count,
                    segment_type_profile=segment_type_profile,
                    issuer_cluster_detail=issuer_cluster_detail,
                    source_poor_reason=source_poor_reason,
                    high_confidence_rows=high_confidence_rows,
                    medium_confidence_rows=medium_confidence_rows,
                    low_confidence_rows=low_confidence_rows,
                    confidence_band=confidence_band,
                    recovery_bucket=recovery_bucket,
                    recoverability_class=recoverability_class,
                    cache_presence_detail=cache_presence_detail,
                    filing_meta_strength=filing_meta_strength,
                    source_signal_strength=source_signal_strength,
                )
            )
            rows.append(
                {
                    "coverage_result_id": self._coverage_result_id(survey_run_id=survey_run_id, ticker=ticker, market=market),
                    "survey_run_id": survey_run_id,
                    "ticker": ticker,
                    "market": market,
                    "start_date": start_date,
                    "companyfacts_rows": companyfacts_rows,
                    "filing_rows": filing_rows,
                    "companyfacts_periods": cf_periods,
                    "filing_periods": filing_periods,
                    "overlap_periods": overlap_periods,
                    "companyfacts_revenue_periods": cf_rev_periods,
                    "companyfacts_operating_income_periods": cf_op_periods,
                    "filing_revenue_periods": filing_rev_periods,
                    "filing_operating_income_periods": filing_op_periods,
                    "business_rows": business_rows,
                    "geography_rows": geography_rows,
                    "product_rows": product_rows,
                    "other_rows": other_rows,
                    "distinct_segment_keys": distinct_segment_keys,
                    "filing_source_count": filing_source_count,
                    "ixbrl_rows": ixbrl_rows,
                    "xbrl_instance_rows": xbrl_instance_rows,
                    "html_table_rows": html_table_rows,
                    "ixbrl_revenue_periods": ixbrl_revenue_periods,
                    "ixbrl_operating_income_periods": ixbrl_operating_income_periods,
                    "xbrl_instance_revenue_periods": xbrl_instance_revenue_periods,
                    "xbrl_instance_operating_income_periods": xbrl_instance_operating_income_periods,
                    "html_table_revenue_periods": html_table_revenue_periods,
                    "html_table_operating_income_periods": html_table_operating_income_periods,
                    "high_confidence_rows": high_confidence_rows,
                    "medium_confidence_rows": medium_confidence_rows,
                    "low_confidence_rows": low_confidence_rows,
                    "filing_meta_count": filing_meta_count,
                    "filing_10q_count": filing_10q_count,
                    "filing_10k_count": filing_10k_count,
                    "best_source_name": best_source_name,
                    "best_source_rank": best_source_rank,
                    "consistency_score": consistency_score,
                    "confidence_score": confidence_score,
                    "confidence_band": confidence_band,
                    "confidence_reason": confidence_reason,
                    "source_cluster": source_cluster,
                    "metric_cluster": metric_cluster,
                    "segment_type_profile": segment_type_profile,
                    "issuer_cluster": issuer_cluster,
                    "issuer_cluster_detail": issuer_cluster_detail,
                    "source_lane": self._coerce_str(cluster_policy.get("source_lane")) or "source_poor",
                    "coverage_reason": coverage_reason,
                    "source_poor_reason": source_poor_reason,
                    "source_poor_reason_detail": source_poor_reason_detail,
                    "parser_candidate_flag": parser_candidate_flag,
                    "recovery_bucket": recovery_bucket,
                    "recoverability_class": recoverability_class,
                    "recovery_score": recovery_score,
                    "recovery_reason": recovery_reason,
                    "recommended_action": recommended_action,
                    "fetch_required": fetch_required,
                    "target_source": target_source,
                    "expected_gain": expected_gain,
                    "priority_rank": priority_rank,
                    "recovery_confidence": recovery_confidence,
                    "cache_presence_detail": cache_presence_detail,
                    "filing_meta_strength": filing_meta_strength,
                    "source_signal_strength": source_signal_strength,
                    "companyfacts_available": companyfacts_available,
                    "filing_available": filing_available,
                    "overlap_available": overlap_available,
                    "filing_only_validated_candidate": filing_only_validated_candidate,
                    "created_at": created_at,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=self._coverage_result_columns())
        return frame[self._coverage_result_columns()].reset_index(drop=True)

    @staticmethod
    def _classify_source_cluster(
        *,
        companyfacts_rows: int,
        filing_rows: int,
        companyfacts_periods: int,
        filing_periods: int,
        overlap_periods: int,
    ) -> str:
        if companyfacts_rows > 0 and filing_rows > 0 and overlap_periods > 0:
            return "companyfacts_overlap"
        if filing_rows > 0 and companyfacts_rows == 0:
            if filing_periods >= 4:
                return "filing_only"
            if filing_periods >= 2:
                return "filing_only_partial"
        if companyfacts_rows == 0 and filing_rows == 0:
            return "segment_source_poor"
        return "segment_source_poor"

    @staticmethod
    def _classify_metric_cluster(*, filing_revenue_periods: int, filing_operating_income_periods: int) -> str:
        if filing_revenue_periods >= 4 and filing_operating_income_periods >= 2:
            return "revenue_and_operating_income"
        if filing_revenue_periods >= 2 and filing_operating_income_periods == 0:
            return "revenue_only"
        if filing_revenue_periods >= 2:
            return "revenue_primary"
        return "metric_sparse"

    @staticmethod
    def _classify_segment_type_profile(*, business_rows: int, geography_rows: int, product_rows: int, other_rows: int) -> str:
        counts = {
            "business": business_rows,
            "geography": geography_rows,
            "product": product_rows,
            "other": other_rows,
        }
        total = sum(counts.values())
        if total <= 0:
            return "none"
        dominant, dominant_count = max(counts.items(), key=lambda item: item[1])
        if dominant_count / max(total, 1) >= 0.6:
            return f"{dominant}_dominant"
        return "mixed"

    @staticmethod
    def _combine_issuer_cluster(*, source_cluster: str, metric_cluster: str) -> str:
        if source_cluster == "companyfacts_overlap":
            return "companyfacts_overlap"
        if source_cluster == "filing_only":
            if metric_cluster == "revenue_only":
                return "filing_only_revenue_only"
            return "filing_only_full"
        if source_cluster == "filing_only_partial":
            return "filing_only_partial"
        return "segment_source_poor"

    @staticmethod
    def _build_coverage_reason_codes(
        *,
        companyfacts_rows: int,
        filing_rows: int,
        overlap_periods: int,
        filing_revenue_periods: int,
        filing_operating_income_periods: int,
        filing_source_count: int,
        segment_type_profile: str,
        issuer_cluster_detail: str,
        source_poor_reason: str | None,
        high_confidence_rows: int,
        medium_confidence_rows: int,
        low_confidence_rows: int,
        confidence_band: str | None,
        recovery_bucket: str | None,
        recoverability_class: str | None,
        cache_presence_detail: str | None,
        filing_meta_strength: str | None,
        source_signal_strength: str | None,
    ) -> list[str]:
        codes: list[str] = []
        if companyfacts_rows == 0 and filing_rows > 0:
            codes.append("companyfacts_missing_filing_present")
        if overlap_periods <= 0 and companyfacts_rows > 0 and filing_rows > 0:
            codes.append("no_companyfacts_filing_overlap")
        if filing_revenue_periods > 0 and filing_operating_income_periods == 0:
            codes.append("operating_income_sparse")
        if filing_source_count <= 1 and filing_rows > 0:
            codes.append("single_filing_source")
        if high_confidence_rows > 0:
            codes.append("high_confidence_filing_available")
        if medium_confidence_rows > 0:
            codes.append("medium_confidence_filing_available")
        if low_confidence_rows > 0:
            codes.append("low_confidence_filing_available")
        if confidence_band:
            codes.append(f"confidence_band_{confidence_band}")
        if recovery_bucket:
            codes.append(f"recovery_bucket_{recovery_bucket}")
        if recoverability_class:
            codes.append(f"recoverability_class_{recoverability_class}")
        if cache_presence_detail:
            codes.append(f"cache_presence_{cache_presence_detail}")
        if filing_meta_strength:
            codes.append(f"filing_meta_{filing_meta_strength}")
        if source_signal_strength:
            codes.append(f"source_signal_{source_signal_strength}")
        if segment_type_profile.endswith("_dominant"):
            codes.append(f"segment_type_{segment_type_profile}")
        if issuer_cluster_detail:
            codes.append(f"cluster_detail_{issuer_cluster_detail}")
        if source_poor_reason:
            codes.append(f"source_poor_{source_poor_reason}")
        if companyfacts_rows == 0 and filing_rows == 0:
            codes.append("segment_source_missing")
        return codes

    @staticmethod
    def _source_metric_periods(frame: pd.DataFrame, source_name: str, metric_name: str) -> int:
        if frame.empty:
            return 0
        mask = frame["source"].astype(str).eq(str(source_name)) & frame["metric_name"].astype(str).eq(str(metric_name))
        return int(frame.loc[mask, "period_end"].nunique())

    @staticmethod
    def _extract_log_signals(log_frame: pd.DataFrame) -> dict[str, bool]:
        if log_frame.empty:
            return {}
        reasons = [str(item).lower() for item in log_frame.get("reason", pd.Series(dtype="object")).dropna().astype(str).tolist()]
        methods = [str(item).lower() for item in log_frame.get("method", pd.Series(dtype="object")).dropna().astype(str).tolist()]
        statuses = [str(item).lower() for item in log_frame.get("status", pd.Series(dtype="object")).dropna().astype(str).tolist()]
        joined = " | ".join(reasons)
        return {
            "cache_missing": "filenotfounderror" in joined or "primary_doc_fetch_failed" in joined or "index_fetch_failed" in joined,
            "instance_missing": "instance_doc_not_found_in_index" in joined or "instance_fetch_or_parse_failed" in joined,
            "segment_not_found": "segment_not_found" in joined,
            "revenue_only_success": any("metrics=revenue" in reason for reason in reasons),
            "xbrl_instance_success": any(method.startswith("xbrl_instance") and status == "success" for method, status in zip(methods, statuses)),
            "html_success": any(method == "html_table_mvp" and status == "success" for method, status in zip(methods, statuses)),
            "html_fail": any(method == "html_table_mvp" and status == "fail" for method, status in zip(methods, statuses)),
        }

    @staticmethod
    def _classify_source_poor_reason(
        *,
        issuer_cluster: str,
        filing_rows: int,
        filing_meta_count: int,
        log_frame: pd.DataFrame,
    ) -> tuple[str | None, str | None, bool]:
        if issuer_cluster != "segment_source_poor":
            return None, None, False
        if filing_rows > 0:
            return None, None, False
        reasons = [str(item) for item in log_frame.get("reason", pd.Series(dtype="object")).dropna().astype(str).tolist()]
        methods = [str(item) for item in log_frame.get("method", pd.Series(dtype="object")).dropna().astype(str).tolist()]
        if any("FileNotFoundError" in reason for reason in reasons):
            return "filing_cache_missing", "|".join(sorted(set(reasons))[:5]), False
        if any("instance_doc_not_found_in_index" in reason or "instance_fetch_or_parse_failed" in reason for reason in reasons):
            return "filing_present_but_xbrl_instance_missing", "|".join(sorted(set(reasons))[:5]), False
        if any(method == "html_table_mvp" and "success" in str(status).lower() for method, status in zip(methods, log_frame.get("status", pd.Series(dtype='object')).tolist())):
            return "html_only_low_confidence_candidate", "|".join(sorted(set(reasons))[:5]) or "html_only_candidate", False
        if any("fetch_failed" in reason for reason in reasons):
            return "filing_cache_incomplete", "|".join(sorted(set(reasons))[:5]), False
        if any("segment_not_found" in reason for reason in reasons):
            return "filing_present_but_segment_not_disclosed", "|".join(sorted(set(reasons))[:5]), False
        if filing_meta_count > 0 or not log_frame.empty:
            return "parser_candidate_but_not_materialized", "|".join(sorted(set(reasons))[:5]) or "filings_present_without_segment_rows", True
        return "source_unavailable", "no_filing_meta_or_segment_logs", False

    @staticmethod
    def _classify_issuer_cluster_detail(
        *,
        issuer_cluster: str,
        filing_revenue_periods: int,
        filing_operating_income_periods: int,
        ixbrl_rows: int,
        xbrl_instance_rows: int,
        html_table_rows: int,
        source_poor_reason: str | None,
        parser_candidate_flag: bool,
        filing: pd.DataFrame,
        log_signals: dict[str, bool],
        distinct_segment_keys: int,
    ) -> str:
        if issuer_cluster == "filing_only_revenue_only":
            if filing_operating_income_periods > 0:
                return "operating_income_recovered"
            if parser_candidate_flag and (xbrl_instance_rows > 0 or html_table_rows > 0):
                return "operating_income_xbrl_present_but_parser_unmapped"
            if distinct_segment_keys <= max(filing_revenue_periods, 1):
                return "operating_income_total_only"
            if xbrl_instance_rows > 0 or html_table_rows > 0:
                if log_signals.get("revenue_only_success"):
                    return "operating_income_lower_confidence_available"
                return "operating_income_disclosure_sparse"
            if log_signals.get("instance_missing") or (ixbrl_rows > 0 and xbrl_instance_rows <= 0 and html_table_rows <= 0):
                return "operating_income_lower_confidence_unavailable"
            if distinct_segment_keys <= max(filing_revenue_periods, 1):
                return "operating_income_total_only"
            return "operating_income_source_sparse"
        if issuer_cluster == "filing_only_full":
            if filing_revenue_periods > 0 and filing_operating_income_periods > 0:
                return "revenue_and_operating_income_supported"
            return "revenue_primary_supported"
        if issuer_cluster == "segment_source_poor":
            if source_poor_reason == "filing_present_but_xbrl_instance_missing":
                return "filing_present_but_xbrl_instance_missing"
            if source_poor_reason == "html_only_low_confidence_candidate":
                return "html_only_low_confidence_candidate"
            if source_poor_reason == "filing_present_but_segment_not_disclosed":
                return "filing_present_but_segment_not_disclosed"
            if parser_candidate_flag:
                return "parser_candidate_but_not_materialized"
            return source_poor_reason or "source_poor_unknown"
        return issuer_cluster or "unclassified"

    @staticmethod
    def _best_filing_source_name(*, ixbrl_rows: int, xbrl_instance_rows: int, html_table_rows: int) -> str | None:
        if ixbrl_rows > 0:
            return "ixbrl_dimension"
        if xbrl_instance_rows > 0:
            return "xbrl_instance"
        if html_table_rows > 0:
            return "html_table_mvp"
        return None

    @staticmethod
    def _classify_cache_presence_detail(
        *,
        ixbrl_rows: int,
        xbrl_instance_rows: int,
        html_table_rows: int,
        filing_meta_count: int,
        source_poor_reason: str | None,
    ) -> str:
        if source_poor_reason == "filing_cache_missing":
            return "filing_cache_missing"
        if ixbrl_rows > 0 and xbrl_instance_rows > 0 and html_table_rows > 0:
            return "ixbrl_xbrl_html"
        if ixbrl_rows > 0 and xbrl_instance_rows > 0:
            return "ixbrl_and_xbrl_instance"
        if ixbrl_rows > 0 and html_table_rows > 0:
            return "ixbrl_and_html"
        if xbrl_instance_rows > 0 and html_table_rows > 0:
            return "xbrl_instance_and_html"
        if ixbrl_rows > 0:
            return "ixbrl_only"
        if xbrl_instance_rows > 0:
            return "xbrl_instance_only"
        if html_table_rows > 0:
            return "html_only"
        if filing_meta_count > 0:
            return "filing_meta_without_cached_source"
        return "no_filing_cache_signal"

    @staticmethod
    def _classify_filing_meta_strength(
        *,
        filing_meta_count: int,
        filing_10q_count: int,
        filing_10k_count: int,
    ) -> str:
        if filing_meta_count >= 4 and filing_10q_count >= 3 and filing_10k_count >= 1:
            return "strong"
        if filing_meta_count >= 2:
            return "moderate"
        if filing_meta_count >= 1:
            return "weak"
        return "none"

    @staticmethod
    def _classify_source_signal_strength(
        *,
        issuer_cluster: str,
        issuer_cluster_detail: str,
        source_poor_reason: str | None,
        parser_candidate_flag: bool,
        filing_rows: int,
        filing_revenue_periods: int,
        filing_operating_income_periods: int,
        ixbrl_rows: int,
        xbrl_instance_rows: int,
        html_table_rows: int,
    ) -> str:
        if source_poor_reason == "source_unavailable":
            return "none"
        if parser_candidate_flag or source_poor_reason == "filing_cache_missing":
            return "strong"
        if issuer_cluster_detail in {"operating_income_lower_confidence_unavailable", "operating_income_xbrl_present_but_parser_unmapped"}:
            return "moderate"
        if filing_operating_income_periods > 0:
            return "strong"
        if xbrl_instance_rows > 0 or html_table_rows > 0:
            return "moderate"
        if ixbrl_rows > 0 and filing_revenue_periods > 0:
            return "weak"
        if filing_rows > 0 or issuer_cluster == "filing_only_revenue_only":
            return "weak"
        return "none"

    @staticmethod
    def _consistency_score(
        *,
        filing_periods: int,
        filing_revenue_periods: int,
        filing_operating_income_periods: int,
        filing_source_count: int,
    ) -> float:
        score = min(float(filing_periods) * 5.0, 25.0)
        if filing_revenue_periods >= 4:
            score += 10.0
        if filing_operating_income_periods >= 2:
            score += 10.0
        if filing_source_count >= 2:
            score += 5.0
        return min(score, 50.0)

    @staticmethod
    def _score_segment_source_confidence(
        *,
        issuer_cluster: str,
        issuer_cluster_detail: str,
        best_source_rank: int | None,
        filing_periods: int,
        filing_revenue_periods: int,
        filing_operating_income_periods: int,
        filing_source_count: int,
        distinct_segment_keys: int,
        consistency_score: float,
        high_confidence_rows: int,
        medium_confidence_rows: int,
        low_confidence_rows: int,
        filing_meta_strength: str,
        cache_presence_detail: str,
        source_signal_strength: str,
    ) -> tuple[float, str, str]:
        base = 0.0
        if best_source_rank == 1:
            base = 55.0
        elif best_source_rank == 2:
            base = 38.0
        elif best_source_rank == 3:
            base = 20.0
        score = base + consistency_score
        if filing_revenue_periods > 0 and filing_operating_income_periods > 0:
            score += 12.0
        elif filing_revenue_periods > 0:
            score += 5.0
        if distinct_segment_keys >= max(filing_periods, 1):
            score += 5.0
        if filing_source_count >= 2:
            score += 5.0
        if filing_meta_strength == "strong":
            score += 5.0
        elif filing_meta_strength == "moderate":
            score += 2.0
        if source_signal_strength == "strong":
            score += 3.0
        elif source_signal_strength == "none":
            score -= 10.0
        if issuer_cluster == "segment_source_poor":
            score = min(score, 25.0)
        if issuer_cluster_detail in {"operating_income_lower_confidence_unavailable", "source_unavailable"}:
            score = min(score, 35.0)
        if cache_presence_detail == "filing_meta_without_cached_source":
            score = min(score, 30.0)
        score = max(0.0, min(score, 100.0))
        if score >= 90.0:
            band = "very_high"
        elif score >= 75.0:
            band = "high"
        elif score >= 50.0:
            band = "medium"
        elif score >= 25.0:
            band = "low"
        else:
            band = "unusable"
        reasons: list[str] = []
        if best_source_rank == 1:
            reasons.append("best_source_ixbrl_dimension")
        elif best_source_rank == 2:
            reasons.append("best_source_xbrl_instance")
        elif best_source_rank == 3:
            reasons.append("best_source_html_table")
        if high_confidence_rows > 0:
            reasons.append("high_confidence_rows_present")
        if medium_confidence_rows > 0:
            reasons.append("medium_confidence_rows_present")
        if low_confidence_rows > 0:
            reasons.append("low_confidence_rows_present")
        if filing_revenue_periods > 0 and filing_operating_income_periods > 0:
            reasons.append("dual_metric_supported")
        elif filing_revenue_periods > 0:
            reasons.append("revenue_only_supported")
        if filing_source_count >= 2:
            reasons.append("multi_source_supported")
        if filing_meta_strength != "none":
            reasons.append(f"filing_meta_strength_{filing_meta_strength}")
        if cache_presence_detail:
            reasons.append(f"cache_presence_{cache_presence_detail}")
        if source_signal_strength:
            reasons.append(f"source_signal_{source_signal_strength}")
        reasons.append(f"issuer_cluster_detail_{issuer_cluster_detail}")
        return score, band, "|".join(reasons)

    @staticmethod
    def _classify_recovery_candidate(
        *,
        issuer_cluster: str,
        issuer_cluster_detail: str,
        source_poor_reason: str | None,
        filing_meta_count: int,
        filing_10q_count: int,
        filing_10k_count: int,
        parser_candidate_flag: bool,
        confidence_score: float,
        cache_presence_detail: str,
        filing_meta_strength: str,
        source_signal_strength: str,
    ) -> tuple[str, str, float, str, str, bool, str, str, int, str]:
        score = float(confidence_score)
        if source_poor_reason == "filing_cache_missing":
            score += 35.0
        if issuer_cluster_detail == "operating_income_lower_confidence_unavailable":
            score += 30.0
        if parser_candidate_flag:
            score += 15.0
        if filing_meta_strength == "strong":
            score += 10.0
        elif filing_meta_strength == "moderate":
            score += 5.0
        if filing_10q_count >= 3:
            score += 5.0
        if filing_10k_count >= 1:
            score += 5.0
        score = max(0.0, min(score, 100.0))
        if score >= 80.0:
            recovery_confidence = "high"
        elif score >= 55.0:
            recovery_confidence = "medium"
        elif score >= 30.0:
            recovery_confidence = "low"
        else:
            recovery_confidence = "minimal"
        if source_poor_reason == "source_unavailable":
            return (
                "not_worth_fetching_yet",
                "structurally_unavailable",
                score,
                "source_unavailable",
                "wait for stronger source coverage",
                False,
                "none",
                "none",
                999,
                "minimal",
            )
        if source_poor_reason == "filing_cache_missing":
            return (
                "recover_now",
                "recover_now_high_priority" if filing_meta_strength == "strong" else "recover_now_medium_priority",
                score,
                "cache_missing_recoverable",
                "fetch filing cache and reparse",
                True,
                "filing_cache_and_instance",
                "high",
                10 if filing_meta_strength == "strong" else 20,
                recovery_confidence,
            )
        if issuer_cluster_detail == "operating_income_lower_confidence_unavailable":
            return (
                "recover_now",
                "recover_now_medium_priority",
                score,
                "lower_confidence_source_missing",
                "fetch instance/html source for operating income check",
                True,
                "xbrl_instance_or_html_table",
                "medium",
                25,
                recovery_confidence,
            )
        if source_poor_reason == "filing_present_but_xbrl_instance_missing":
            return (
                "fetch_optional",
                "recover_now_low_priority",
                score,
                "lower_confidence_source_missing",
                "fetch instance/html source if issuer remains high-value",
                True,
                "xbrl_instance_or_html_table",
                "medium",
                35,
                recovery_confidence,
            )
        if parser_candidate_flag:
            return (
                "fetch_optional",
                "fetch_optional",
                score,
                "parser_candidate_without_materialization",
                "inspect parser path before fetch",
                cache_presence_detail in {"filing_meta_without_cached_source", "html_only"},
                "parser_path_or_html_table",
                "medium" if source_signal_strength in {"strong", "moderate"} else "low",
                45,
                recovery_confidence,
            )
        if issuer_cluster_detail in {
            "operating_income_lower_confidence_available",
            "operating_income_disclosure_sparse",
            "operating_income_total_only",
            "operating_income_dimension_mismatch",
            "operating_income_label_variant",
        }:
            return (
                "not_worth_fetching_yet",
                "not_worth_fetching_yet",
                score,
                "lower_confidence_already_checked_or_sparse",
                "treat as disclosure/source reality",
                False,
                "none",
                "low",
                70,
                recovery_confidence,
            )
        if issuer_cluster == "segment_source_poor":
            return (
                "not_worth_fetching_yet",
                "not_worth_fetching_yet",
                score,
                source_poor_reason or "source_poor",
                "wait for stronger source coverage",
                False,
                "none",
                "minimal",
                80,
                recovery_confidence,
            )
        return (
            "not_worth_fetching_yet",
            "not_worth_fetching_yet",
            score,
            "no_recovery_signal",
            "no fetch recommended",
            False,
            "none",
            "minimal",
            90,
            recovery_confidence,
        )

    @staticmethod
    def _build_filing_temporal_consistency_map(
        frame: pd.DataFrame,
    ) -> dict[tuple[str | None, str | None, str | None, str | None], dict[str, Any]]:
        if frame.empty:
            return {}
        grouped: dict[tuple[str | None, str | None, str | None, str | None], dict[str, Any]] = {}
        for _, row in frame.iterrows():
            key = (
                SECSegmentValidationService._coerce_str(row.get("ticker")),
                SECSegmentValidationService._coerce_str(row.get("metric_name")),
                SECSegmentValidationService._coerce_str(row.get("segment_type_normalized")),
                SECSegmentValidationService._coerce_str(row.get("segment_name_normalized")),
            )
            item = grouped.setdefault(key, {"periods": set(), "sources": set()})
            period_end = SECSegmentValidationService._to_date(row.get("period_end"))
            if period_end is not None:
                item["periods"].add(period_end)
            source = SECSegmentValidationService._coerce_str(row.get("source"))
            if source:
                item["sources"].add(source)
        result: dict[tuple[str | None, str | None, str | None, str | None], dict[str, Any]] = {}
        for key, item in grouped.items():
            result[key] = {
                "period_count": len(item["periods"]),
                "source_count": len(item["sources"]),
            }
        return result

    def _summarize_source_coverage(
        self,
        *,
        coverage: pd.DataFrame,
        survey_run_id: str,
        tickers: list[str],
        market: str,
        start_date: date,
    ) -> dict[str, Any]:
        if coverage.empty:
            return {
                "survey_run_id": survey_run_id,
                "market": market,
                "start_date": start_date.isoformat(),
                "tickers": tickers,
                "issuer_count": 0,
                "issuer_cluster_counts": {},
                "metric_coverage": {},
                "segment_type_coverage": {},
            }
        return self._json_safe(
            {
                "survey_run_id": survey_run_id,
                "market": market,
                "start_date": start_date.isoformat(),
                "tickers": tickers,
                "issuer_count": int(len(coverage)),
                "issuer_cluster_counts": coverage["issuer_cluster"].value_counts(dropna=False).to_dict(),
                "issuer_cluster_detail_counts": coverage["issuer_cluster_detail"].value_counts(dropna=False).to_dict(),
                "source_lane_counts": coverage["source_lane"].value_counts(dropna=False).to_dict(),
                "segment_type_profile_counts": coverage["segment_type_profile"].value_counts(dropna=False).to_dict(),
                "source_poor_reason_counts": coverage["source_poor_reason"].dropna().value_counts(dropna=False).to_dict(),
                "companyfacts_available_issuers": int(coverage["companyfacts_available"].fillna(False).sum()),
                "filing_available_issuers": int(coverage["filing_available"].fillna(False).sum()),
                "overlap_issuers": int(coverage["overlap_available"].fillna(False).sum()),
                "filing_only_validated_candidates": int(coverage["filing_only_validated_candidate"].fillna(False).sum()),
                "revenue_supported_issuers": int((coverage["filing_revenue_periods"].fillna(0) > 0).sum()),
                "operating_income_supported_issuers": int((coverage["filing_operating_income_periods"].fillna(0) > 0).sum()),
                "high_confidence_rows": int(coverage["high_confidence_rows"].fillna(0).sum()),
                "medium_confidence_rows": int(coverage["medium_confidence_rows"].fillna(0).sum()),
                "low_confidence_rows": int(coverage["low_confidence_rows"].fillna(0).sum()),
                "confidence_band_counts": coverage["confidence_band"].value_counts(dropna=False).to_dict(),
                "recovery_bucket_counts": coverage["recovery_bucket"].value_counts(dropna=False).to_dict(),
                "confidence_score_summary": {
                    "min": float(coverage["confidence_score"].min()) if coverage["confidence_score"].notna().any() else None,
                    "median": float(coverage["confidence_score"].median()) if coverage["confidence_score"].notna().any() else None,
                    "max": float(coverage["confidence_score"].max()) if coverage["confidence_score"].notna().any() else None,
                },
                "sample_rows": coverage.head(10).to_dict(orient="records"),
            }
        )

    @staticmethod
    def _collapse_segment_source_rows(frame: pd.DataFrame, *, family: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        key_cols = ["ticker", "market", "period_end", "statement_type", "metric_name", "segment_type_normalized", "segment_name_normalized"]
        frame = frame.copy()
        frame["row_count"] = frame.groupby(key_cols, dropna=False)["ticker"].transform("size")
        sort_cols = ["available_date", "filing_date", "accepted_at"]
        ascending = [False, False, False]
        if family == "filing":
            sort_cols = ["source_priority", "available_date", "filing_date", "accepted_at"]
            ascending = [True, False, False, False]
        frame = frame.sort_values(key_cols + sort_cols, ascending=[True] * len(key_cols) + ascending, na_position="last")
        frame = frame.groupby(key_cols, dropna=False, as_index=False).head(1).reset_index(drop=True)
        return frame

    @staticmethod
    def _build_segment_value_pool(frame: pd.DataFrame) -> dict[tuple[str | None, date | None, str | None, str | None], list[float]]:
        if frame.empty:
            return {}
        pool: dict[tuple[str | None, date | None, str | None, str | None], list[float]] = {}
        for _, row in frame.iterrows():
            key = (
                SECSegmentValidationService._coerce_str(row.get("ticker")),
                SECSegmentValidationService._to_date(row.get("period_end")),
                SECSegmentValidationService._coerce_str(row.get("metric_name")),
                SECSegmentValidationService._coerce_str(row.get("segment_type_normalized")),
            )
            value = SECSegmentValidationService._coerce_float(row.get("value"))
            if value is None:
                continue
            pool.setdefault(key, []).append(value)
        return pool

    @staticmethod
    def _build_filing_consensus_map(
        frame: pd.DataFrame,
    ) -> dict[tuple[str | None, str | None, date | None, str | None, str | None, str | None, str | None], dict[str, Any]]:
        if frame.empty:
            return {}
        grouped: dict[tuple[str | None, str | None, date | None, str | None, str | None, str | None, str | None], dict[str, Any]] = {}
        for _, row in frame.iterrows():
            key = (
                SECSegmentValidationService._coerce_str(row.get("ticker")),
                SECSegmentValidationService._coerce_str(row.get("market")),
                SECSegmentValidationService._to_date(row.get("period_end")),
                SECSegmentValidationService._coerce_str(row.get("statement_type")),
                SECSegmentValidationService._coerce_str(row.get("metric_name")),
                SECSegmentValidationService._coerce_str(row.get("segment_type_normalized")),
                SECSegmentValidationService._coerce_str(row.get("segment_name_normalized")),
            )
            source = SECSegmentValidationService._coerce_str(row.get("source")) or ""
            value = SECSegmentValidationService._coerce_float(row.get("value"))
            item = grouped.setdefault(key, {"sources": set(), "values": []})
            item["sources"].add(source)
            if value is not None:
                item["values"].append(value)
        result: dict[tuple[str | None, str | None, date | None, str | None, str | None, str | None, str | None], dict[str, Any]] = {}
        for key, item in grouped.items():
            values = item["values"]
            sources = item["sources"]
            within_tolerance = False
            if len(values) >= 2:
                ref = max(max(abs(v) for v in values), 1.0)
                within_tolerance = (max(values) - min(values)) <= ref * 0.05
            result[key] = {
                "source_count": len(sources),
                "value_count": len(values),
                "within_tolerance": within_tolerance,
            }
        return result

    @staticmethod
    def _has_similar_segment_value(
        pool: dict[tuple[str | None, date | None, str | None, str | None], list[float]],
        key: tuple[str | None, date | None, str | None, str | None],
        target_value: float | None,
        tolerance_value: float,
    ) -> bool:
        if target_value is None:
            return False
        for candidate in pool.get(key, []):
            if abs(candidate - target_value) <= max(abs(candidate), abs(target_value), 1.0) * max(float(tolerance_value), 0.02):
                return True
        return False

    def _summarize_results(
        self,
        frame: pd.DataFrame,
        *,
        comparison_run_id: str,
        tickers: list[str],
        market: str,
        start_date: date,
        coverage: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        if frame.empty:
            return {
                "comparison_run_id": comparison_run_id,
                "market": market,
                "start_date": start_date.isoformat(),
                "tickers": tickers,
                "results_written": 0,
                "comparison_status_counts": {},
                "mismatch_class_counts": {},
                "issuer_cluster_counts": {},
                "source_lane_counts": {},
                "sample_results": [],
            }
        coverage_frame = coverage if coverage is not None else pd.DataFrame()
        return self._json_safe(
            {
                "comparison_run_id": comparison_run_id,
                "market": market,
                "start_date": start_date.isoformat(),
                "tickers": tickers,
                "results_written": int(len(frame)),
                "comparison_status_counts": frame["comparison_status"].value_counts(dropna=False).to_dict(),
                "mismatch_class_counts": frame["mismatch_class"].value_counts(dropna=False).to_dict(),
                "semantic_gap_counts": frame["semantic_gap_class"].dropna().value_counts(dropna=False).to_dict(),
                "issuer_cluster_counts": frame["issuer_cluster"].value_counts(dropna=False).to_dict(),
                "issuer_cluster_detail_counts": frame["issuer_cluster_detail"].value_counts(dropna=False).to_dict(),
                "source_lane_counts": frame["source_lane"].value_counts(dropna=False).to_dict(),
                "filing_source_confidence_counts": frame["filing_source_confidence"].value_counts(dropna=False).to_dict(),
                "coverage_summary": self._summarize_source_coverage(
                    coverage=coverage_frame,
                    survey_run_id=comparison_run_id,
                    tickers=tickers,
                    market=market,
                    start_date=start_date,
                )
                if not coverage_frame.empty
                else {},
                "sample_results": frame.head(10).to_dict(orient="records"),
            }
        )

    @staticmethod
    def _normalize_segment_type(raw_type: Any, raw_name: Any) -> str:
        base = str(raw_type or "").strip().lower()
        if base in SEGMENT_TYPE_CANONICAL:
            return SEGMENT_TYPE_CANONICAL[base]
        text = f"{base} {str(raw_name or '').strip().lower()}"
        if "geo" in text or "region" in text or "country" in text:
            return "geography"
        if "product" in text or "service" in text or "brand" in text:
            return "product"
        if "business" in text or "operating" in text:
            return "business"
        return base or "other"

    @staticmethod
    def _normalize_segment_name(raw_name: Any) -> str:
        text = str(raw_name or "").strip().lower()
        text = text.replace("&", " and ")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        tokens: list[str] = []
        for token in text.split():
            if token in SEGMENT_NAME_STOPWORDS:
                continue
            if token.endswith("ies") and len(token) > 4:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
                token = token[:-1]
            tokens.append(token)
        return " ".join(tokens)

    @staticmethod
    def _is_intersegment_name(text: str | None) -> bool:
        raw = str(text or "").lower()
        return "intersegment" in raw or "elimination" in raw

    @staticmethod
    def _source_confidence(source: str | None) -> str | None:
        if not source:
            return None
        return SEGMENT_FILING_SOURCE_CONFIDENCE.get(str(source), "n/a")

    @staticmethod
    def _source_rank(source: str | None) -> int | None:
        if not source:
            return None
        return int(SEGMENT_FILING_SOURCE_PRIORITY.get(str(source), 99))

    @staticmethod
    def _comparison_status(
        missing_on_companyfacts: bool,
        missing_on_filing: bool,
        tolerance_breach: bool,
        *,
        diagnostic_codes: list[str] | None = None,
    ) -> str:
        codes = set(diagnostic_codes or [])
        if missing_on_companyfacts and missing_on_filing:
            return "missing_both"
        if missing_on_companyfacts and (
            "filing_source_consensus" in codes or "filing_only_period_consistency" in codes
        ):
            return COMPARISON_STATUS_FILING_ONLY_VALIDATED
        if missing_on_companyfacts:
            return "missing_companyfacts"
        if missing_on_filing:
            return "missing_filing"
        if tolerance_breach:
            return "tolerance_breach"
        return "match"

    @staticmethod
    def _semantic_gap_class(diagnostic_codes: list[str]) -> str | None:
        code_set = set(diagnostic_codes)
        if "segment_intersegment_scope" in code_set:
            return "segment_intersegment_scope_gap"
        if "segment_dimension_scope_gap" in code_set or "segment_dimension_scope_gap_candidate" in code_set:
            return "segment_dimension_scope_gap"
        if "segment_member_scope_gap" in code_set:
            return "segment_member_scope_gap"
        if "segment_metric_semantics_gap" in code_set:
            return "segment_metric_semantics_gap"
        return None

    @staticmethod
    def _mismatch_class(*, comparison_status: str, diagnostic_codes: list[str]) -> str:
        if comparison_status in {"match", COMPARISON_STATUS_FILING_ONLY_VALIDATED}:
            return MISMATCH_CLASS_NONE
        code_set = set(diagnostic_codes)
        if comparison_status == "missing_both":
            return MISMATCH_CLASS_REFERENCE_GAP
        if "segment_source_missing" in code_set or "issuer_cluster_segment_source_poor" in code_set:
            return MISMATCH_CLASS_REFERENCE_GAP
        if comparison_status == "missing_companyfacts" and (
            "companyfacts_source_unavailable_for_issuer" in code_set
            or "companyfacts_missing_filing_present" in code_set
            or "cluster_metric_not_supported" in code_set
        ):
            return MISMATCH_CLASS_SOURCE_COVERAGE_GAP
        if "filing_segment_reference_missing" in code_set or "companyfacts_segment_reference_missing" in code_set:
            return MISMATCH_CLASS_REFERENCE_GAP
        if "segment_member_normalization_candidate" in code_set:
            return MISMATCH_CLASS_NORMALIZATION_GAP
        if {
            "segment_dimension_scope_gap",
            "segment_dimension_scope_gap_candidate",
            "segment_member_scope_gap",
            "segment_metric_semantics_gap",
            "segment_intersegment_scope",
        } & code_set:
            return MISMATCH_CLASS_SOURCE_SEMANTICS_GAP
        if comparison_status == "tolerance_breach" and "segment_name_normalized" in code_set:
            return MISMATCH_CLASS_NORMALIZATION_GAP
        return MISMATCH_CLASS_PARSER_BUG

    @staticmethod
    def _build_notes(*, policy: dict[str, Any], diagnostic_codes: list[str]) -> str:
        pieces = []
        note = str(policy.get("notes") or "").strip()
        if note:
            pieces.append(note)
        if diagnostic_codes:
            pieces.append("diagnostics=" + ",".join(dict.fromkeys(code for code in diagnostic_codes if code)))
        return "; ".join(pieces)

    def _latest_comparison_run_id(self) -> str | None:
        if not self.db.table_exists("validation_sec_segment_quality"):
            return None
        row = self.db.fetch_one(
            """
            SELECT comparison_run_id
            FROM validation_sec_segment_quality
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        return str(row[0]) if row and row[0] is not None else None

    @staticmethod
    def _pct_diff(abs_diff: float | None, reference: float | None) -> float | None:
        if abs_diff is None or reference in (None, 0.0):
            return None
        return abs_diff / abs(reference)

    @staticmethod
    def _tolerance_breach(
        abs_diff: float | None,
        pct_diff: float | None,
        left_value: float | None,
        right_value: float | None,
        tolerance_type: str,
        tolerance_value: float,
    ) -> bool:
        if left_value is None or right_value is None:
            return False
        if tolerance_type == "absolute":
            return (abs_diff or 0.0) > tolerance_value
        return (pct_diff or 0.0) > tolerance_value

    @staticmethod
    def _result_columns() -> list[str]:
        return [
            "result_id",
            "ticker",
            "market",
            "period_end",
            "statement_type",
            "metric_name",
            "issuer_cluster",
            "source_cluster",
            "metric_cluster",
            "segment_type_profile",
            "source_lane",
            "issuer_cluster_detail",
            "coverage_reason",
            "companyfacts_available",
            "filing_available",
            "overlap_available",
            "segment_type_companyfacts",
            "segment_type_filing",
            "segment_type_normalized",
            "segment_name_companyfacts",
            "segment_name_filing",
            "segment_name_normalized",
            "companyfacts_value",
            "filing_value",
            "abs_diff",
            "pct_diff",
            "missing_on_companyfacts",
            "missing_on_filing",
            "comparison_status",
            "mismatch_class",
            "semantic_gap_class",
            "companyfacts_source",
            "filing_source",
            "filing_source_confidence",
            "filing_source_rank",
            "issuer_confidence_score",
            "issuer_confidence_band",
            "companyfacts_available_date",
            "filing_available_date",
            "filing_accession",
            "companyfacts_row_count",
            "filing_row_count",
            "compare_mode",
            "value_mode_used",
            "diagnostic_code",
            "notes",
            "comparison_run_id",
            "created_at",
        ]

    @staticmethod
    def _coverage_result_columns() -> list[str]:
        return [
            "coverage_result_id",
            "survey_run_id",
            "ticker",
            "market",
            "start_date",
            "companyfacts_rows",
            "filing_rows",
            "companyfacts_periods",
            "filing_periods",
            "overlap_periods",
            "companyfacts_revenue_periods",
            "companyfacts_operating_income_periods",
            "filing_revenue_periods",
            "filing_operating_income_periods",
            "business_rows",
            "geography_rows",
            "product_rows",
            "other_rows",
            "distinct_segment_keys",
            "filing_source_count",
            "ixbrl_rows",
            "xbrl_instance_rows",
            "html_table_rows",
            "ixbrl_revenue_periods",
            "ixbrl_operating_income_periods",
            "xbrl_instance_revenue_periods",
            "xbrl_instance_operating_income_periods",
            "html_table_revenue_periods",
            "html_table_operating_income_periods",
            "high_confidence_rows",
            "medium_confidence_rows",
            "low_confidence_rows",
            "filing_meta_count",
            "filing_10q_count",
            "filing_10k_count",
            "best_source_name",
            "best_source_rank",
            "consistency_score",
            "confidence_score",
            "confidence_band",
            "confidence_reason",
            "source_cluster",
            "metric_cluster",
            "segment_type_profile",
            "issuer_cluster",
            "issuer_cluster_detail",
            "source_lane",
            "coverage_reason",
            "source_poor_reason",
            "source_poor_reason_detail",
            "parser_candidate_flag",
            "recovery_bucket",
            "recoverability_class",
            "recovery_score",
            "recovery_reason",
            "recommended_action",
            "fetch_required",
            "target_source",
            "expected_gain",
            "priority_rank",
            "recovery_confidence",
            "cache_presence_detail",
            "filing_meta_strength",
            "source_signal_strength",
            "companyfacts_available",
            "filing_available",
            "overlap_available",
            "filing_only_validated_candidate",
            "created_at",
        ]

    @staticmethod
    def _result_id(
        *,
        comparison_run_id: str,
        ticker: str | None,
        period_end: date | None,
        metric_name: str | None,
        segment_type: str | None,
        segment_name: str | None,
    ) -> str:
        payload = json.dumps(
            {
                "comparison_run_id": comparison_run_id,
                "ticker": ticker,
                "period_end": period_end.isoformat() if period_end else None,
                "metric_name": metric_name,
                "segment_type": segment_type,
                "segment_name": segment_name,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _coverage_result_id(*, survey_run_id: str, ticker: str | None, market: str | None) -> str:
        payload = json.dumps(
            {
                "survey_run_id": survey_run_id,
                "ticker": ticker,
                "market": market,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _recovery_result_columns() -> list[str]:
        return [
            "candidate_result_id",
            "survey_run_id",
            "ticker",
            "market",
            "issuer_cluster",
            "issuer_cluster_detail",
            "source_poor_reason",
            "confidence_score",
            "confidence_band",
            "recovery_bucket",
            "recoverability_class",
            "recovery_score",
            "recovery_reason",
            "recommended_action",
            "target_source",
            "fetch_required",
            "expected_gain",
            "priority_rank",
            "recovery_confidence",
            "cache_presence_detail",
            "filing_meta_strength",
            "source_signal_strength",
            "filing_meta_count",
            "filing_10q_count",
            "filing_10k_count",
            "parser_candidate_flag",
            "created_at",
        ]

    @staticmethod
    def _recovery_result_id(*, survey_run_id: str, ticker: str | None, market: str | None) -> str:
        payload = json.dumps(
            {
                "survey_run_id": survey_run_id,
                "ticker": ticker,
                "market": market,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _split_codes(value: str | None) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in str(value).split("|") if item and str(item).strip()]

    @staticmethod
    def _cluster_supports_metric(cluster_policy: dict[str, Any], metric_name: str) -> bool:
        metric = str(metric_name or "").strip().lower()
        if metric == "revenue":
            return bool(cluster_policy.get("revenue_supported", False))
        if metric == "operating_income":
            return bool(cluster_policy.get("operating_income_supported", False))
        return False

    def _latest_coverage_run_id(self) -> str | None:
        if not self.db.table_exists("segment_issuer_source_coverage"):
            return None
        row = self.db.fetch_one(
            """
            SELECT survey_run_id
            FROM segment_issuer_source_coverage
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        return str(row[0]) if row and row[0] is not None else None

    def _latest_recovery_run_id(self) -> str | None:
        if not self.db.table_exists("segment_recovery_candidate_registry"):
            return None
        row = self.db.fetch_one(
            """
            SELECT survey_run_id
            FROM segment_recovery_candidate_registry
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        return str(row[0]) if row and row[0] is not None else None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None or pd.isna(value):
            return None
        try:
            return int(value)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _coerce_str(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _to_date(value: Any) -> date | None:
        ts = pd.to_datetime(value, errors="coerce")
        return ts.date() if pd.notna(ts) else None

    @staticmethod
    def _coalesce_prefer_left(left: pd.Series | None, right: pd.Series | None) -> pd.Series:
        left_series = pd.Series(left if left is not None else pd.Series(dtype="object"), copy=False)
        right_series = pd.Series(right if right is not None else pd.Series(dtype="object"), copy=False).reindex(left_series.index)
        return left_series.where(left_series.notna(), right_series)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, pd.Timestamp):
            return None if pd.isna(value) else value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:  # noqa: BLE001
            pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:  # noqa: BLE001
                return value
        return value
