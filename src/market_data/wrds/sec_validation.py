from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from market_data.wrds.duckdb_io import DuckDBManager

SEC_VALIDATION_START_DATE = date(2012, 1, 1)
# Raised from 10→15: non-December fiscal-year-end issuers (e.g. retail Jan/Feb, tech Sep)
# can have period_end dates that differ by up to 15 days between Compustat and SEC XBRL
# due to statutory filing delays and weekend/holiday adjustments.
PERIOD_END_ALIGNMENT_TOLERANCE_DAYS = 15
SEQUENCE_ALIGNMENT_MIN_DELTA_DAYS = 80
SEQUENCE_ALIGNMENT_MAX_DELTA_DAYS = 100
SEC_FISCAL_CYCLE_MAX_DAYS = 400
QUARTERLY_SEC_FORMS = ("10-Q", "10-Q/A")
ANNUAL_SEC_FORMS = ("10-K", "10-K/A")
ALL_SEC_FORMS = QUARTERLY_SEC_FORMS + ANNUAL_SEC_FORMS
VALUE_MODE_REPORTED = "reported_sec"
VALUE_MODE_NORMALIZED = "wrds_aligned_normalized"
COMPARE_MODE_DEFAULT = "default"
COMPARE_MODE_REPORTED = "reported"
COMPARE_MODE_NORMALIZED = "normalized"
METRIC_CLASS_DIRECT = "direct_match"
METRIC_CLASS_RECONSTRUCTED = "reconstructed"
METRIC_CLASS_SCOPE_SENSITIVE = "scope_sensitive"
MISMATCH_CLASS_NONE = "no_mismatch"
MISMATCH_CLASS_PARSER_BUG = "parser_bug"
MISMATCH_CLASS_NORMALIZATION_GAP = "normalization_gap"
MISMATCH_CLASS_SOURCE_SEMANTICS_GAP = "source_semantics_gap"
MISMATCH_CLASS_COMPATIBILITY_GAP = "compatibility_gap"
TIME_REGIME_POST_2012 = "post_2012_high_confidence"
TIME_REGIME_PRE_2012 = "pre_2012_compatibility"
SEC_FINANCIALS_EXTRA_COLUMNS = {
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
}
ANNUAL_AGGREGATE_FLOW_METRICS = {
    "revenue",
    "gross_profit",
    "cogs",
    "sga",
    "operating_income",
    "pretax_income",
    "tax",
    "net_income",
    "net_income_common",
    "interest",
    "d_and_a",
    "r_and_d",
    "amortization",
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    "capex",
    "dividends_paid",
    "repurchases",
}
ANNUAL_AGGREGATE_SEC_COLUMNS = {"EPS", "Diluted EPS"}
SEC_ROW_FORM_PRIORITY = {
    "10-Q": 1,
    "10-Q/A": 2,
    "10-K": 3,
    "10-K/A": 4,
}
ANNUAL_RAW_SEC_FACT_NAMES = (
    "Diluted EPS",
    "EPS",
    "Diluted Shares",
    "Net Income",
    "Net Income Common",
)
RAW_QUARTER_MAX_DURATION_DAYS = 120
RAW_SOURCE_PERIOD_END_FALLBACK = "raw_fact_period_end_fallback"
RAW_SOURCE_EXTRA_FALLBACK = "extra_table_fallback"
RAW_SOURCE_NORMALIZED_PRETAX = "normalized_pretax_from_net_income_plus_tax"
RAW_SOURCE_NORMALIZED_DIVIDEND_ZERO = "normalized_zero_dividend_proxy"
RAW_SOURCE_NORMALIZED_INTANGIBLES = "normalized_goodwill_inclusive_intangibles_proxy"
RAW_SOURCE_NORMALIZED_AOCI = "normalized_aoci_component_sum"
RAW_SOURCE_NORMALIZED_AOCI_DIRECT = "normalized_aoci_direct_total"
RAW_SOURCE_NORMALIZED_NET_INCOME_COMMON = "normalized_common_income_from_net_income"
RAW_SOURCE_NORMALIZED_GROSS_PROFIT = "normalized_gross_profit_from_revenue_minus_cogs"


def _quote(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _quoted_literal_list(values: list[str]) -> str:
    payload = ", ".join("'" + value.replace("'", "''") + "'" for value in values)
    return f"({payload})" if payload else "('')"


@dataclass(frozen=True)
class MetricMapping:
    canonical_metric_name: str
    statement_type: str
    metric_class: str
    value_mode_default: str
    reported_value_policy: str
    normalized_value_policy: str
    scope_rule: str
    mismatch_policy: str
    wrds_table_name: str
    wrds_column_name: str
    sec_table_name: str
    sec_column_name: str
    sec_extra_column_name: str | None
    comparison_rule: str
    tolerance_type: str
    tolerance_value: float
    value_scale_rule: str
    sign_rule: str
    post_2012_mode: str
    pre_2012_mode: str
    pre_2012_supported_flag: bool
    pre_2012_value_mode_default: str
    pre_2012_tolerance_override: float | None
    pre_2012_known_gap_policy: str
    known_gap_flag: bool
    known_gap_reason: str
    recommended_compare_mode: str
    notes: str
    is_active: bool = True


def default_metric_mappings() -> list[MetricMapping]:
    rows: list[MetricMapping] = []

    def add_both(
        metric: str,
        *,
        wrds_quarterly: str,
        wrds_annual: str,
        sec_quarterly: str,
        sec_annual: str,
        sec_quarterly_extra: str | None = None,
        sec_annual_extra: str | None = None,
        metric_class: str = METRIC_CLASS_DIRECT,
        value_mode_default: str = VALUE_MODE_NORMALIZED,
        reported_value_policy: str = "base_sec_column_only",
        normalized_value_policy: str = "reported_passthrough",
        scope_rule: str = "generic_corporate",
        mismatch_policy: str = "strict_direct_match",
        tolerance_type: str = "relative",
        tolerance_value: float = 0.05,
        value_scale_rule: str = "wrds_millions_to_units",
        post_2012_mode: str = TIME_REGIME_POST_2012,
        pre_2012_mode: str = TIME_REGIME_PRE_2012,
        pre_2012_supported_flag: bool | None = None,
        pre_2012_value_mode_default: str = VALUE_MODE_REPORTED,
        pre_2012_tolerance_override: float | None = None,
        pre_2012_known_gap_policy: str | None = None,
        known_gap_flag: bool = False,
        known_gap_reason: str = "",
        recommended_compare_mode: str | None = None,
        notes: str = "",
        is_active: bool = True,
        wrds_quarterly_table: str = "financials_quarterly_canonical",
        wrds_annual_table: str = "financials_annual_canonical",
    ) -> None:
        resolved_pre_2012_supported = (
            bool(pre_2012_supported_flag)
            if pre_2012_supported_flag is not None
            else metric_class == METRIC_CLASS_DIRECT
        )
        resolved_pre_2012_tolerance = pre_2012_tolerance_override
        if resolved_pre_2012_tolerance is None:
            if tolerance_type == "absolute":
                resolved_pre_2012_tolerance = max(float(tolerance_value), 0.10)
            elif resolved_pre_2012_supported:
                resolved_pre_2012_tolerance = max(float(tolerance_value) * 2.0, 0.10)
            else:
                resolved_pre_2012_tolerance = max(float(tolerance_value) * 3.0, 0.20)
        resolved_pre_2012_gap_policy = pre_2012_known_gap_policy or (
            "reported_only_low_confidence" if resolved_pre_2012_supported else "missing_allowed_low_confidence"
        )
        resolved_compare_mode = recommended_compare_mode or (
            COMPARE_MODE_REPORTED if value_mode_default == VALUE_MODE_REPORTED else COMPARE_MODE_NORMALIZED
        )
        rows.extend(
            [
                MetricMapping(
                    canonical_metric_name=metric,
                    statement_type="quarterly",
                    metric_class=metric_class,
                    value_mode_default=value_mode_default,
                    reported_value_policy=reported_value_policy,
                    normalized_value_policy=normalized_value_policy,
                    scope_rule=scope_rule,
                    mismatch_policy=mismatch_policy,
                    wrds_table_name=wrds_quarterly_table,
                    wrds_column_name=wrds_quarterly,
                    sec_table_name="financials_quarterly",
                    sec_column_name=sec_quarterly,
                    sec_extra_column_name=sec_quarterly_extra,
                    comparison_rule="match_on_fiscal_period_key",
                    tolerance_type=tolerance_type,
                    tolerance_value=tolerance_value,
                    value_scale_rule=value_scale_rule,
                    sign_rule="as_is",
                    post_2012_mode=post_2012_mode,
                    pre_2012_mode=pre_2012_mode,
                    pre_2012_supported_flag=resolved_pre_2012_supported,
                    pre_2012_value_mode_default=pre_2012_value_mode_default,
                    pre_2012_tolerance_override=resolved_pre_2012_tolerance,
                    pre_2012_known_gap_policy=resolved_pre_2012_gap_policy,
                    known_gap_flag=known_gap_flag,
                    known_gap_reason=known_gap_reason,
                    recommended_compare_mode=resolved_compare_mode,
                    notes=notes,
                    is_active=is_active,
                ),
                MetricMapping(
                    canonical_metric_name=metric,
                    statement_type="annual",
                    metric_class=metric_class,
                    value_mode_default=value_mode_default,
                    reported_value_policy=reported_value_policy,
                    normalized_value_policy=normalized_value_policy,
                    scope_rule=scope_rule,
                    mismatch_policy=mismatch_policy,
                    wrds_table_name=wrds_annual_table,
                    wrds_column_name=wrds_annual,
                    sec_table_name="financials_quarterly",
                    sec_column_name=sec_annual,
                    sec_extra_column_name=sec_annual_extra,
                    comparison_rule="match_on_fiscal_year_key",
                    tolerance_type=tolerance_type,
                    tolerance_value=tolerance_value,
                    value_scale_rule=value_scale_rule,
                    sign_rule="as_is",
                    post_2012_mode=post_2012_mode,
                    pre_2012_mode=pre_2012_mode,
                    pre_2012_supported_flag=resolved_pre_2012_supported,
                    pre_2012_value_mode_default=pre_2012_value_mode_default,
                    pre_2012_tolerance_override=resolved_pre_2012_tolerance,
                    pre_2012_known_gap_policy=resolved_pre_2012_gap_policy,
                    known_gap_flag=known_gap_flag,
                    known_gap_reason=known_gap_reason,
                    recommended_compare_mode=resolved_compare_mode,
                    notes=notes,
                    is_active=is_active,
                ),
            ]
        )

    add_both("revenue", wrds_quarterly="revenue", wrds_annual="revenue", sec_quarterly="Revenue", sec_annual="Revenue")
    add_both(
        "gross_profit",
        wrds_quarterly="gross_profit",
        wrds_annual="gross_profit",
        sec_quarterly="Gross Profit",
        sec_annual="Gross Profit",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_gross_profit_only",
        normalized_value_policy="direct_then_revenue_minus_cogs",
        mismatch_policy="track_semantic_gap",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Gross profit can diverge whenever WRDS COGS semantics do not line up with SEC direct gross-profit or revenue-minus-COGS presentation.",
        notes="WRDS gross profit is revenue minus COGS; SEC gross profit may be direct or inferred from revenue and COGS. Residual gaps usually track the same COGS classification drift seen in operating metrics.",
    )
    add_both(
        "cogs",
        wrds_quarterly="cogs",
        wrds_annual="cogs",
        sec_quarterly="COGS",
        sec_annual="COGS",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_cogs_only",
        normalized_value_policy="direct_then_gross_profit_then_operating_expense_proxy",
        mismatch_policy="track_semantic_gap",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Compustat COGS and SEC direct cost tags can diverge for classification-sensitive reporters.",
        notes="Keep direct SEC COGS when reported; normalized mode may use deterministic proxy logic.",
    )
    add_both(
        "sga",
        wrds_quarterly="sga",
        wrds_annual="sga",
        sec_quarterly="SG&A",
        sec_annual="SG&A",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_sga_only",
        normalized_value_policy="direct_then_component_aggregation_with_rd",
        mismatch_policy="prefer_normalized_then_compare",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
    )
    add_both(
        "operating_income",
        wrds_quarterly="operating_income",
        wrds_annual="operating_income",
        sec_quarterly="Operating Income",
        sec_annual="Operating Income",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_operating_income_only",
        normalized_value_policy="direct_then_revenue_minus_operating_expenses",
        mismatch_policy="track_semantic_gap",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Operating income can differ because Compustat operating classification is not always identical to SEC direct presentation.",
        notes="Direct SEC OperatingIncomeLoss is preserved; normalized mode only reconstructs when direct fact is unavailable.",
    )
    add_both(
        "net_income",
        wrds_quarterly="net_income",
        wrds_annual="net_income",
        sec_quarterly="Net Income",
        sec_annual="Net Income",
        sec_quarterly_extra="net_income",
        sec_annual_extra="net_income",
    )
    add_both(
        "pretax_income",
        wrds_quarterly="pretax_income",
        wrds_annual="pretax_income",
        sec_quarterly="Pretax Income",
        sec_annual="Pretax Income",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_pretax_fact_only",
        normalized_value_policy="direct_then_net_income_plus_tax",
        mismatch_policy="prefer_normalized_then_compare",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        notes="Normalized mode reconstructs pretax income from net income plus tax only when a direct SEC pretax fact is unavailable.",
    )
    add_both(
        "tax",
        wrds_quarterly="tax",
        wrds_annual="tax",
        sec_quarterly="Tax",
        sec_annual="Tax",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
    )
    add_both(
        "net_income_common",
        wrds_quarterly="net_income_common",
        wrds_annual="net_income_common",
        sec_quarterly="Net Income Common",
        sec_annual="Net Income Common",
        sec_quarterly_extra="owner_net_income",
        sec_annual_extra="owner_net_income",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_common_income_only",
        normalized_value_policy="direct_then_extra_then_net_income_fallback",
        mismatch_policy="prefer_normalized_then_compare",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        notes="WRDS common net income falls back to NI/IB when ibcom is unavailable; SEC prefers direct common-income facts.",
    )
    add_both(
        "interest",
        wrds_quarterly="interest",
        wrds_annual="interest",
        sec_quarterly="Interest",
        sec_annual="Interest",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        scope_rule="generic_or_financing_interest",
        mismatch_policy="track_semantic_gap",
        tolerance_value=0.08,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Marketplace, platform, and financing-heavy issuers can disclose an interest scope that does not line up cleanly with Compustat xint/xintq.",
        notes="Interest mapping includes nonoperating, borrowing, and operating interest-expense variants when plain InterestExpense is absent.",
    )
    add_both(
        "d_and_a",
        wrds_quarterly="d_and_a",
        wrds_annual="d_and_a",
        sec_quarterly="D&A",
        sec_annual="D&A",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_d_and_a_only",
        normalized_value_policy="direct_annual_or_quarter_sum",
        mismatch_policy="prefer_normalized_then_compare",
        tolerance_value=0.08,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        notes="Annual WRDS D&A uses depreciation plus amortization when available; quarterly uses direct depreciation/amortization flow.",
    )
    add_both(
        "r_and_d",
        wrds_quarterly="xrdq",
        wrds_annual="xrd",
        sec_quarterly="R&D",
        sec_annual="R&D",
        sec_quarterly_extra="r_and_d",
        sec_annual_extra="r_and_d",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        value_mode_default=VALUE_MODE_REPORTED,
        reported_value_policy="direct_r_and_d_only",
        normalized_value_policy="direct_then_extra_r_and_d_fallback",
        mismatch_policy="track_semantic_gap",
        tolerance_value=0.08,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        recommended_compare_mode=COMPARE_MODE_REPORTED,
        known_gap_flag=True,
        known_gap_reason="R&D can diverge when issuers exclude acquired in-process costs or software-development variants from the direct SEC tag set.",
        notes="Pilot metric: compare direct SEC R&D expense against direct Compustat xrd/xrdq sourced from WRDS local source tables.",
        is_active=False,
        wrds_quarterly_table="wrds_compustat_quarterly",
        wrds_annual_table="wrds_compustat_annual",
    )
    add_both(
        "total_assets",
        wrds_quarterly="assets",
        wrds_annual="assets",
        sec_quarterly="Total Assets",
        sec_annual="Total Assets",
        sec_quarterly_extra="total_assets",
        sec_annual_extra="total_assets",
    )
    add_both(
        "total_liabilities",
        wrds_quarterly="liabilities",
        wrds_annual="liabilities",
        sec_quarterly="Total Liabilities",
        sec_annual="Total Liabilities",
    )
    add_both(
        "shareholders_equity",
        wrds_quarterly="equity",
        wrds_annual="equity",
        sec_quarterly="Shareholders Equity",
        sec_annual="Shareholders Equity",
        sec_quarterly_extra="owner_equity",
        sec_annual_extra="owner_equity",
    )
    add_both(
        "cash",
        wrds_quarterly="cash",
        wrds_annual="cash",
        sec_quarterly="Cash",
        sec_annual="Cash",
        sec_quarterly_extra="cash",
        sec_annual_extra="cash",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        reported_value_policy="direct_cash_only",
        normalized_value_policy="regulated_or_broad_cash_scope",
        scope_rule="financial_or_platform_cash",
        mismatch_policy="track_semantic_gap",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Cash scope can differ because WRDS che/cheq may include regulated or treasury-related balances not explicit in SEC cash tags.",
    )
    add_both(
        "current_assets",
        wrds_quarterly="current_assets",
        wrds_annual="current_assets",
        sec_quarterly="Current Assets",
        sec_annual="Current Assets",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
    )
    add_both(
        "current_liabilities",
        wrds_quarterly="current_liabilities",
        wrds_annual="current_liabilities",
        sec_quarterly="Current Liabilities",
        sec_annual="Current Liabilities",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
    )
    add_both(
        "current_fin_assets",
        wrds_quarterly="current_fin_assets",
        wrds_annual="current_fin_assets",
        sec_quarterly="Current Fin Assets",
        sec_annual="Current Fin Assets",
        sec_quarterly_extra="current_fin_assets",
        sec_annual_extra="current_fin_assets",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        reported_value_policy="direct_current_fin_asset_tags_or_extra",
        normalized_value_policy="direct_then_extra_current_fin_asset_fallback",
        scope_rule="short_term_investments_and_current_fin_assets",
        mismatch_policy="track_semantic_gap",
        tolerance_value=0.08,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="WRDS current financial assets rely on Compustat investment/current-financial-asset fields, while SEC issuers can report narrower marketable-securities or broader current-financial-asset totals.",
        notes="Reported SEC keeps direct current financial-asset tags when available; normalized mode falls back to extra-table current financial assets derived from those same direct tags.",
    )
    add_both(
        "non_current_fin_assets",
        wrds_quarterly="non_current_fin_assets",
        wrds_annual="non_current_fin_assets",
        sec_quarterly="Non Current Fin Assets",
        sec_annual="Non Current Fin Assets",
        sec_quarterly_extra="non_current_fin_assets",
        sec_annual_extra="non_current_fin_assets",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        reported_value_policy="direct_noncurrent_fin_asset_tags_or_extra",
        normalized_value_policy="direct_then_extra_noncurrent_fin_asset_fallback",
        scope_rule="noncurrent_investments_and_financial_assets",
        mismatch_policy="track_semantic_gap",
        tolerance_value=0.08,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="WRDS non-current financial assets use Compustat other-investment proxies; SEC filers can split investment, securities, and other financial assets differently.",
        notes="Use direct non-current investment or financial-asset tags when available; otherwise rely on extra-table values built from those direct SEC tags.",
    )
    add_both(
        "current_fin_liabilities",
        wrds_quarterly="current_fin_liabilities",
        wrds_annual="current_fin_liabilities",
        sec_quarterly="Current Fin Liabilities",
        sec_annual="Current Fin Liabilities",
        sec_quarterly_extra="current_fin_liabilities",
        sec_annual_extra="current_fin_liabilities",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        reported_value_policy="direct_current_fin_liability_tags_or_extra",
        normalized_value_policy="direct_then_extra_current_fin_liability_fallback",
        scope_rule="current_financial_liability_or_current_debt_proxy",
        mismatch_policy="track_semantic_gap",
        tolerance_value=0.08,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Direct current financial-liability totals are sparse; WRDS often falls back to current debt style fields and SEC issuers vary between debt-current and broader current-financial-liability presentation.",
        notes="Direct current-financial-liability tags are preferred; normalized mode may fall back to the extra-table current financial-liability proxy when stored SEC raw facts are sparse.",
    )
    add_both(
        "non_current_fin_liabilities",
        wrds_quarterly="non_current_fin_liabilities",
        wrds_annual="non_current_fin_liabilities",
        sec_quarterly="Non Current Fin Liabilities",
        sec_annual="Non Current Fin Liabilities",
        sec_quarterly_extra="non_current_fin_liabilities",
        sec_annual_extra="non_current_fin_liabilities",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        reported_value_policy="direct_noncurrent_fin_liability_tags_or_extra",
        normalized_value_policy="direct_then_extra_noncurrent_fin_liability_fallback",
        scope_rule="noncurrent_financial_liability_or_long_debt_proxy",
        mismatch_policy="track_semantic_gap",
        tolerance_value=0.08,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Direct non-current financial-liability totals are sparse; WRDS canonicalization often relies on long-term debt style fields and SEC disclosure scope can remain broader or narrower.",
        notes="Direct non-current-financial-liability tags are preferred; normalized mode falls back to the extra-table non-current financial-liability proxy when needed.",
    )
    add_both(
        "debt_short",
        wrds_quarterly="debt_short",
        wrds_annual="debt_short",
        sec_quarterly="Debt Short",
        sec_annual="Debt Short",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        scope_rule="debt_maturity_classification",
        mismatch_policy="track_semantic_gap",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Compustat dlcq can diverge from SEC short-debt tags when current maturities, short-term borrowings, or commercial paper are classified differently.",
    )
    add_both(
        "debt_long",
        wrds_quarterly="debt_long",
        wrds_annual="debt_long",
        sec_quarterly="Debt Long",
        sec_annual="Debt Long",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        scope_rule="debt_maturity_classification",
        mismatch_policy="track_semantic_gap",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Compustat dltt/dlttq can diverge from SEC long-debt tags when current portions or financing facilities are classified differently.",
    )
    add_both(
        "receivables",
        wrds_quarterly="accounts_receivable",
        wrds_annual="accounts_receivable",
        sec_quarterly="AR",
        sec_annual="AR",
        sec_quarterly_extra="ar",
        sec_annual_extra="ar",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        reported_value_policy="direct_receivables_only",
        normalized_value_policy="direct_then_extra_receivables_fallback",
        scope_rule="broader_receivables",
        mismatch_policy="track_semantic_gap",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Compustat receivables can represent a broader receivable scope than SEC trade receivable tags.",
    )
    add_both(
        "inventory",
        wrds_quarterly="inventory",
        wrds_annual="inventory",
        sec_quarterly="Inventory",
        sec_annual="Inventory",
        sec_quarterly_extra="inventory",
        sec_annual_extra="inventory",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
    )
    add_both(
        "accounts_payable",
        wrds_quarterly="accounts_payable",
        wrds_annual="accounts_payable",
        sec_quarterly="AP",
        sec_annual="AP",
        sec_quarterly_extra="ap",
        sec_annual_extra="ap",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
    )
    add_both(
        "deferred_revenue",
        wrds_quarterly="deferred_revenue",
        wrds_annual="deferred_revenue",
        sec_quarterly="Deferred Revenue",
        sec_annual="Deferred Revenue",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        reported_value_policy="direct_deferred_revenue_only",
        normalized_value_policy="prefer_total_contract_liability_then_current",
        scope_rule="contract_liability_total_preferred",
        mismatch_policy="track_semantic_gap",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="WRDS drc/drcq and SEC contract-liability disclosures can diverge between current-only and total contract-liability scope.",
        notes="Parser prefers total contract-liability/deferred-revenue facts before current-only tags; residual differences remain scope-sensitive.",
    )
    add_both(
        "goodwill",
        wrds_quarterly="goodwill",
        wrds_annual="goodwill",
        sec_quarterly="Goodwill",
        sec_annual="Goodwill",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
    )
    add_both(
        "amortization",
        wrds_quarterly="amortization",
        wrds_annual="amortization",
        sec_quarterly="Amortization",
        sec_annual="Amortization",
        sec_quarterly_extra="amortization",
        sec_annual_extra="amortization",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_amortization_only",
        normalized_value_policy="direct_then_extra_amortization_fallback",
        mismatch_policy="prefer_normalized_then_compare",
        tolerance_value=0.10,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        notes="Quarterly amortization uses co_ifndq.amq on WRDS and direct SEC amortization-specific facts only; no reverse-engineering from D&A is allowed.",
    )
    add_both(
        "intangibles",
        wrds_quarterly="intangibles",
        wrds_annual="intangibles",
        sec_quarterly="Intangibles",
        sec_annual="Intangibles",
        sec_quarterly_extra="intangibles",
        sec_annual_extra="intangibles",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        reported_value_policy="direct_intangibles_only",
        normalized_value_policy="goodwill_inclusive_wrds_proxy",
        scope_rule="goodwill_plus_intangibles_for_wrds",
        mismatch_policy="track_semantic_gap",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="WRDS intan/intanq often behaves like goodwill-inclusive intangible assets, while SEC direct intangibles facts are frequently goodwill-exclusive.",
        notes="Reported SEC intangibles stay direct; normalized mode adds goodwill back when available to align with WRDS calibration semantics.",
    )
    add_both(
        "common_stock",
        wrds_quarterly="common_stock",
        wrds_annual="common_stock",
        sec_quarterly="Common Stock",
        sec_annual="Common Stock",
        sec_quarterly_extra="common_stock",
        sec_annual_extra="common_stock",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        value_mode_default=VALUE_MODE_REPORTED,
        reported_value_policy="pure_common_stock_line_only",
        normalized_value_policy="reported_only_unless_extra_common_stock_exists",
        scope_rule="isolated_common_stock_only",
        mismatch_policy="track_semantic_gap",
        tolerance_value=0.05,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Common stock can be embedded in combined equity lines with APIC; only isolated common-stock values are treated as comparable.",
        recommended_compare_mode=COMPARE_MODE_REPORTED,
        notes="Combined common-stock-plus-APIC balance-sheet lines are not decomposed in compare mode.",
    )
    add_both(
        "retained_earnings",
        wrds_quarterly="retained_earnings",
        wrds_annual="retained_earnings",
        sec_quarterly="Retained Earnings",
        sec_annual="Retained Earnings",
        sec_quarterly_extra="retained_earnings",
        sec_annual_extra="retained_earnings",
        metric_class=METRIC_CLASS_DIRECT,
        reported_value_policy="direct_retained_earnings_only",
        normalized_value_policy="direct_then_extra_retained_earnings_fallback",
        mismatch_policy="strict_direct_match",
        tolerance_value=0.05,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        notes="Retained earnings preserves direct SEC retained-earnings or accumulated-deficit facts, with extra-table fallback when the base table is sparse.",
    )
    add_both(
        "aoci",
        wrds_quarterly="aoci",
        wrds_annual="aoci",
        sec_quarterly="AOCI",
        sec_annual="AOCI",
        sec_quarterly_extra="aoci",
        sec_annual_extra="aoci",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        value_mode_default=VALUE_MODE_NORMALIZED,
        reported_value_policy="direct_total_aoci_only",
        normalized_value_policy="direct_then_component_sum",
        scope_rule="aoci_total_or_component_sum",
        mismatch_policy="prefer_normalized_then_compare",
        tolerance_value=0.05,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        notes="Normalized AOCI can use a component sum when SEC discloses the pieces but not an explicit total-AOCI fact.",
    )
    add_both(
        "operating_cash_flow",
        wrds_quarterly="operating_cash_flow",
        wrds_annual="operating_cash_flow",
        sec_quarterly="Operating Cash Flow",
        sec_annual="Operating Cash Flow",
        sec_quarterly_extra="cfo",
        sec_annual_extra="cfo",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_cfo_only",
        normalized_value_policy="direct_then_extra_cfo_fallback",
        mismatch_policy="prefer_normalized_then_compare",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        tolerance_value=0.08,
    )
    add_both(
        "investing_cash_flow",
        wrds_quarterly="investing_cash_flow",
        wrds_annual="investing_cash_flow",
        sec_quarterly="Investing Cash Flow",
        sec_annual="Investing Cash Flow",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_cfi_only",
        normalized_value_policy="direct_annual_or_quarter_delta",
        mismatch_policy="prefer_normalized_then_compare",
        tolerance_value=0.08,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
    )
    add_both(
        "financing_cash_flow",
        wrds_quarterly="financing_cash_flow",
        wrds_annual="financing_cash_flow",
        sec_quarterly="Financing Cash Flow",
        sec_annual="Financing Cash Flow",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_cff_only",
        normalized_value_policy="direct_annual_or_quarter_delta",
        mismatch_policy="prefer_normalized_then_compare",
        tolerance_value=0.08,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
    )
    add_both(
        "capex",
        wrds_quarterly="capital_expenditure",
        wrds_annual="capital_expenditure",
        sec_quarterly="Capital Expenditure",
        sec_annual="Capital Expenditure",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_capex_only",
        normalized_value_policy="ppe_plus_software_plus_other_productive_assets",
        mismatch_policy="prefer_normalized_then_compare",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        tolerance_value=0.08,
    )
    add_both(
        "shares",
        wrds_quarterly="shares_diluted",
        wrds_annual="shares_outstanding",
        sec_quarterly="Diluted Shares",
        sec_annual="Shares",
        sec_quarterly_extra="Shares",
        sec_annual_extra="shares_outstanding",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_share_count_only",
        normalized_value_policy="prefer_diluted_then_reported_shares",
        mismatch_policy="prefer_normalized_then_compare",
        pre_2012_supported_flag=True,
        pre_2012_value_mode_default=VALUE_MODE_REPORTED,
        pre_2012_known_gap_policy="reported_only_low_confidence",
        tolerance_value=0.03,
        value_scale_rule="wrds_millions_to_units",
        notes="Quarterly prefers diluted shares; annual falls back to Shares from SEC base table.",
    )
    add_both(
        "dividends_paid",
        wrds_quarterly="dividends_paid",
        wrds_annual="dividends_paid",
        sec_quarterly="Dividends Paid",
        sec_annual="Dividends Paid",
        sec_quarterly_extra="dividends_paid",
        sec_annual_extra="dividends_paid",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_dividends_only",
        normalized_value_policy="direct_or_zero_when_no_dividend_line",
        mismatch_policy="prefer_normalized_then_compare",
        tolerance_value=0.08,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        notes="Normalized mode treats a missing dividend cash-flow line as zero when the filing otherwise exposes a usable statement-of-cash-flows row.",
    )
    add_both(
        "repurchases",
        wrds_quarterly="repurchases",
        wrds_annual="repurchases",
        sec_quarterly="Repurchases",
        sec_annual="Repurchases",
        sec_quarterly_extra="share_repurchases",
        sec_annual_extra="share_repurchases",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_repurchases_only",
        normalized_value_policy="direct_then_extra_repurchases_fallback",
        mismatch_policy="prefer_normalized_then_compare",
        tolerance_value=0.10,
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="reported_only_or_missing_allowed",
        known_gap_flag=True,
        known_gap_reason="Compustat prstkcy/prstkc and SEC treasury-stock repurchase cash-flow lines can diverge because of ASR timing, settlement conventions, or non-open-market treasury activity.",
        notes="Quarterly WRDS repurchases are derived from prstkcy YTD deltas; annual uses Compustat prstkc.",
    )
    add_both(
        "eps",
        wrds_quarterly="eps_diluted",
        wrds_annual="eps_diluted",
        sec_quarterly="Diluted EPS",
        sec_annual="Diluted EPS",
        sec_quarterly_extra="EPS",
        sec_annual_extra="EPS",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        reported_value_policy="direct_diluted_eps_only",
        normalized_value_policy="annual_raw_eps_override_then_quarter_sum_fallback",
        mismatch_policy="prefer_normalized_then_compare",
        pre_2012_supported_flag=True,
        pre_2012_value_mode_default=VALUE_MODE_REPORTED,
        pre_2012_known_gap_policy="reported_only_low_confidence",
        known_gap_flag=True,
        known_gap_reason="Annual diluted EPS can retain small denominator or rounding differences versus WRDS.",
        tolerance_type="absolute",
        tolerance_value=0.05,
        value_scale_rule="as_is",
        notes="Prefer diluted EPS; fall back to EPS when diluted EPS is unavailable on SEC rows.",
    )
    add_both(
        "sbc",
        wrds_quarterly="stkcoq",
        wrds_annual="stkco",
        sec_quarterly="SBC",
        sec_annual="SBC",
        sec_quarterly_extra="sbc",
        sec_annual_extra="sbc",
        metric_class=METRIC_CLASS_RECONSTRUCTED,
        value_mode_default=VALUE_MODE_REPORTED,
        reported_value_policy="sec_only_until_wrds_aux_source",
        normalized_value_policy="sec_only_until_wrds_aux_source",
        scope_rule="wrds_aux_source_required",
        mismatch_policy="reference_gap_until_wrds_aux_source",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="missing_allowed_low_confidence",
        known_gap_flag=True,
        known_gap_reason="No durable direct WRDS gold-reference source for SBC is onboarded yet; SEC extraction exists but compare coverage stays deferred.",
        recommended_compare_mode=COMPARE_MODE_REPORTED,
        notes="Pilot metric: use WRDS auxiliary stkcoq/stkco as a shadow reference while keeping SBC outside the active calibrated core.",
        is_active=False,
        wrds_quarterly_table="wrds_compustat_quarterly",
        wrds_annual_table="wrds_compustat_annual",
    )
    add_both(
        "apic",
        wrds_quarterly="ucapsq",
        wrds_annual="ucaps",
        sec_quarterly="APIC",
        sec_annual="APIC",
        sec_quarterly_extra="additional_paid_in_capital",
        sec_annual_extra="additional_paid_in_capital",
        metric_class=METRIC_CLASS_SCOPE_SENSITIVE,
        value_mode_default=VALUE_MODE_REPORTED,
        reported_value_policy="direct_apic_only",
        normalized_value_policy="reported_only_until_wrds_source_confirmed",
        scope_rule="common_stock_and_apic_can_be_combined",
        mismatch_policy="reference_gap_until_wrds_aux_source",
        pre_2012_supported_flag=False,
        pre_2012_known_gap_policy="missing_allowed_low_confidence",
        known_gap_flag=True,
        known_gap_reason="No durable direct WRDS APIC reference has been confirmed, and SEC APIC is often combined with common stock in equity-line presentation.",
        recommended_compare_mode=COMPARE_MODE_REPORTED,
        notes="Deferred metric: limited pilot compare can inspect ucaps/ucapsq overlap, but active onboarding is blocked by sparse WRDS coverage and combined equity-line semantics risk.",
        is_active=False,
        wrds_quarterly_table="wrds_compustat_quarterly",
        wrds_annual_table="wrds_compustat_annual",
    )
    return rows


def seed_default_metric_mapping_registry(db: DuckDBManager) -> int:
    frame = pd.DataFrame([asdict(item) for item in default_metric_mappings()])
    frame["updated_at"] = pd.Timestamp.utcnow()
    if frame.empty:
        return 0
    return db.merge_dataframe("metric_mapping_registry", frame, ("canonical_metric_name", "statement_type"))


class SECVsWRDSFinancialValidationService:
    def __init__(
        self,
        db: DuckDBManager,
        logger: logging.Logger | None = None,
        *,
        wrds_db_path: str | Path | None = None,
    ) -> None:
        self.db = db
        self.logger = logger or logging.getLogger(__name__)
        self.wrds_db_path = Path(wrds_db_path).expanduser() if wrds_db_path is not None else None
        self.wrds_attachments = {"wrds_lake": self.wrds_db_path} if self.wrds_db_path is not None else None
        self._raw_compare_lookup: dict[tuple[str, str, date | None, str], dict[str, Any]] = {}
        self._table_schema_cache: dict[str, set[str]] = {}

    def ensure_registry_seeded(self) -> int:
        return seed_default_metric_mapping_registry(self.db)

    def _table_columns(self, table_name: str, *, attachments: dict[str, Path] | None = None) -> set[str]:
        cache_key = table_name if attachments is None else f"attached::{table_name}"
        cached = self._table_schema_cache.get(cache_key)
        if cached is not None:
            return cached
        if not self.db.table_exists(table_name, attachments=attachments):
            self._table_schema_cache[cache_key] = set()
            return set()
        columns = {str(row["column_name"]) for row in self.db.schema_info(table_name, attachments=attachments)}
        self._table_schema_cache[cache_key] = columns
        return columns

    def _qualified_wrds_table_name(self, table_name: str) -> str:
        if self.wrds_db_path is None:
            return table_name
        if "." in str(table_name):
            return str(table_name)
        return f"wrds_lake.{table_name}"

    @staticmethod
    def _select_column_or_null(
        *,
        table_alias: str | None,
        source_name: str,
        alias_name: str | None = None,
        available_columns: set[str],
    ) -> str:
        alias = alias_name or source_name
        qualified = f"{table_alias}.{_quote(source_name)}" if table_alias else _quote(source_name)
        if source_name in available_columns:
            return f"{qualified} AS {_quote(alias)}"
        return f"NULL AS {_quote(alias)}"

    def compare(
        self,
        *,
        comparison_run_id: str,
        tickers: str | list[str] | None = None,
        market: str = "us",
        start_date: date | None = None,
        limit: int | None = None,
        statement_types: tuple[str, ...] = ("quarterly", "annual"),
        compare_mode: str = COMPARE_MODE_DEFAULT,
        metric_names: tuple[str, ...] | None = None,
        active_only: bool = True,
    ) -> dict[str, Any]:
        start = start_date or SEC_VALIDATION_START_DATE
        seeded = self.ensure_registry_seeded()
        selected_metric_names = (
            tuple(
                sorted(
                    {
                        str(item).strip().lower()
                        for item in (metric_names or ())
                        if str(item).strip()
                    }
                )
            )
            or None
        )
        mappings = self._load_mappings(
            statement_types,
            active_only=active_only,
            metric_names=selected_metric_names,
        )
        target_tickers = self._resolve_target_tickers(tickers=tickers, market=market, start_date=start, limit=limit)
        self._raw_compare_lookup = self._build_raw_compare_lookup(
            market=market.lower(),
            tickers=target_tickers,
            start_date=start,
            mappings=mappings,
        )
        created_at = pd.Timestamp.utcnow()

        results: list[dict[str, Any]] = []
        summary: dict[str, Any] = {
            "comparison_run_id": comparison_run_id,
            "market": market.lower(),
            "start_date": start.isoformat(),
            "tickers": target_tickers,
            "registry_rows_seeded": seeded,
            "statement_types": list(statement_types),
            "compare_mode": compare_mode,
            "active_only": active_only,
            "metric_names": list(selected_metric_names or ()),
            "statement_summaries": {},
        }

        for statement_type in statement_types:
            statement_mappings = mappings.loc[mappings["statement_type"] == statement_type].copy()
            if statement_mappings.empty:
                continue
            wrds_rows = self._load_wrds_rows(statement_type, target_tickers, start, statement_mappings)
            sec_rows = self._load_sec_rows(statement_type, market, target_tickers, start, statement_mappings)
            merged = self._align_rows(statement_type, wrds_rows, sec_rows)
            statement_results = self._build_result_rows(
                merged=merged,
                mappings=statement_mappings,
                market=market.lower(),
                comparison_run_id=comparison_run_id,
                created_at=created_at,
                compare_mode=compare_mode,
            )
            results.extend(statement_results)
            summary["statement_summaries"][statement_type] = {
                "wrds_rows": int(len(wrds_rows)),
                "sec_rows": int(len(sec_rows)),
                "aligned_rows": int(len(merged)),
                "result_rows": int(len(statement_results)),
            }

        frame = pd.DataFrame(results)
        if not frame.empty:
            status_rank = {
                "match": 0,
                "matched_with_ambiguity": 1,
                "tolerance_breach": 2,
                "missing_sec": 3,
                "missing_wrds": 4,
                "missing_both": 5,
            }
            frame["_status_rank"] = frame["comparison_status"].map(status_rank).fillna(9)
            frame["_abs_diff_rank"] = pd.to_numeric(frame.get("abs_diff"), errors="coerce").fillna(float("inf"))
            frame = (
                frame.sort_values(
                    ["result_id", "_status_rank", "_abs_diff_rank", "wrds_row_count", "sec_row_count"],
                    ascending=[True, True, True, True, True],
                    na_position="last",
                )
                .drop_duplicates(subset=["result_id"], keep="first")
                .drop(columns=["_status_rank", "_abs_diff_rank"])
                .reset_index(drop=True)
            )
            self.db.merge_dataframe("validation_sec_vs_wrds_financials", frame, ("result_id",))

        summary["results_written"] = int(len(frame))
        summary["comparison_status_counts"] = (
            frame["comparison_status"].value_counts(dropna=False).to_dict() if not frame.empty else {}
        )
        summary["mismatch_class_counts"] = (
            frame["mismatch_class"].value_counts(dropna=False).to_dict() if not frame.empty and "mismatch_class" in frame else {}
        )
        summary["value_mode_used_counts"] = (
            frame["value_mode_used"].value_counts(dropna=False).to_dict() if not frame.empty and "value_mode_used" in frame else {}
        )
        summary["sample_results"] = frame.head(10).to_dict(orient="records") if not frame.empty else []
        return self._json_safe(summary)

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
            FROM validation_sec_vs_wrds_financials
            WHERE comparison_run_id = ?
            """,
            [run_id],
        )
        if df.empty:
            return {"comparison_run_id": run_id, "group_by": group_by, "rows": [], "status_counts": {}}

        group_col_map = {
            "metric": "metric_name",
            "ticker": "ticker",
            "statement_type": "statement_type",
            "mismatch_class": "mismatch_class",
            "compare_mode": "compare_mode",
            "value_mode": "value_mode_used",
            "time_regime": "time_regime",
        }
        if group_by not in group_col_map:
            raise ValueError("group_by must be one of: metric, ticker, statement_type, mismatch_class, compare_mode, value_mode, time_regime")
        group_col = group_col_map[group_by]

        summary = (
            df.groupby(group_col, dropna=False)
            .agg(
                comparisons=("result_id", "count"),
                tolerance_breaches=("tolerance_breach", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                missing_on_sec=("missing_on_sec", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                missing_on_wrds=("missing_on_wrds", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                sign_mismatches=("sign_mismatch", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                avg_pct_diff=("pct_diff", "mean"),
                max_abs_diff=("abs_diff", "max"),
            )
            .reset_index()
            .sort_values(["tolerance_breaches", "missing_on_sec", "missing_on_wrds", "comparisons"], ascending=[False, False, False, False])
        )

        top_breaches = (
            df.loc[df["tolerance_breach"].fillna(False)]
            .sort_values(["pct_diff", "abs_diff"], ascending=[False, False], na_position="last")
            .head(max(1, int(limit)))
        )
        missing_sec_by_metric = (
            df.loc[df["missing_on_sec"].fillna(False)]
            .groupby("metric_name", dropna=False)["result_id"]
            .count()
            .sort_values(ascending=False)
            .head(max(1, int(limit)))
            .reset_index(name="missing_rows")
        )
        statement_patterns = (
            df.groupby("statement_type", dropna=False)
            .agg(
                comparisons=("result_id", "count"),
                breaches=("tolerance_breach", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                sign_mismatches=("sign_mismatch", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
            )
            .reset_index()
            .sort_values("statement_type")
        )
        return self._json_safe(
            {
            "comparison_run_id": run_id,
            "group_by": group_by,
            "status_counts": df["comparison_status"].value_counts(dropna=False).to_dict(),
            "rows": summary.head(max(1, int(limit))).to_dict(orient="records"),
            "missing_sec_by_metric": missing_sec_by_metric.to_dict(orient="records"),
            "top_tolerance_breaches": top_breaches[
                [
                    "ticker",
                    "statement_type",
                    "metric_name",
                    "compare_mode",
                    "value_mode_used",
                    "period_end",
                    "wrds_value",
                    "sec_value",
                    "abs_diff",
                    "pct_diff",
                    "comparison_status",
                    "mismatch_class",
                    "semantic_gap_class",
                    "notes",
                ]
            ].to_dict(orient="records"),
            "statement_type_summary": statement_patterns.to_dict(orient="records"),
            }
        )

    def policy_summary(
        self,
        *,
        comparison_run_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        run_id = comparison_run_id or self._latest_comparison_run_id()
        mappings = self._load_mappings(("quarterly", "annual"), active_only=False)
        policy_rows = mappings.rename(columns={"canonical_metric_name": "metric_name"})
        deferred_rows = (
            policy_rows.loc[~policy_rows["is_active"].fillna(False)]
            .sort_values(["metric_name", "statement_type"])
            .head(max(1, int(limit)))
        )
        deferred_payload = self._build_deferred_metric_payload(deferred_rows)
        if not run_id:
            return {
                "comparison_run_id": None,
                "rows": [],
                "mismatch_class_summary": [],
                "semantic_gap_summary": [],
                "time_regime_summary": [],
                "compare_mode_counts": {},
                "value_mode_used_counts": {},
                "deferred_metrics": deferred_payload,
            }

        df = self.db.fetch_df(
            """
            SELECT *
            FROM validation_sec_vs_wrds_financials
            WHERE comparison_run_id = ?
            """,
            [run_id],
        )
        if df.empty:
            return {
                "comparison_run_id": run_id,
                "rows": [],
                "mismatch_class_summary": [],
                "semantic_gap_summary": [],
                "time_regime_summary": [],
                "compare_mode_counts": {},
                "value_mode_used_counts": {},
                "deferred_metrics": deferred_payload,
            }
        summary = (
            df.groupby(["metric_name", "statement_type"], dropna=False)
            .agg(
                comparisons=("result_id", "count"),
                matches=("comparison_status", lambda s: int((pd.Series(s, copy=False) == "match").sum())),
                tolerance_breaches=("tolerance_breach", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                missing_on_sec=("missing_on_sec", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                parser_bug_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_PARSER_BUG).sum())),
                compatibility_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_COMPATIBILITY_GAP).sum())),
                normalization_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_NORMALIZATION_GAP).sum())),
                source_semantics_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_SOURCE_SEMANTICS_GAP).sum())),
            )
            .reset_index()
        )
        joined = summary.merge(
            policy_rows[
                [
                    "metric_name",
                    "statement_type",
                    "metric_class",
                    "value_mode_default",
                    "reported_value_policy",
                    "normalized_value_policy",
                    "scope_rule",
                    "mismatch_policy",
                    "post_2012_mode",
                    "pre_2012_mode",
                    "pre_2012_supported_flag",
                    "pre_2012_value_mode_default",
                    "pre_2012_tolerance_override",
                    "pre_2012_known_gap_policy",
                    "known_gap_flag",
                    "known_gap_reason",
                    "recommended_compare_mode",
                    "notes",
                ]
            ],
            on=["metric_name", "statement_type"],
            how="left",
        ).sort_values(
            ["source_semantics_gap_rows", "parser_bug_rows", "tolerance_breaches", "missing_on_sec", "metric_name", "statement_type"],
            ascending=[False, False, False, False, True, True],
        )

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
        regime_summary = (
            df.groupby("time_regime", dropna=False)
            .agg(
                rows=("result_id", "count"),
                missing_on_sec=("missing_on_sec", lambda s: int(pd.Series(s, copy=False).fillna(False).sum())),
                compatibility_gap_rows=("mismatch_class", lambda s: int((pd.Series(s, copy=False) == MISMATCH_CLASS_COMPATIBILITY_GAP).sum())),
            )
            .reset_index()
            .sort_values("rows", ascending=False)
        )
        return self._json_safe(
            {
                "comparison_run_id": run_id,
                "compare_mode_counts": df["compare_mode"].value_counts(dropna=False).to_dict(),
                "value_mode_used_counts": df["value_mode_used"].value_counts(dropna=False).to_dict(),
                "mismatch_class_summary": mismatch_summary.to_dict(orient="records"),
                "semantic_gap_summary": semantic_summary.to_dict(orient="records"),
                "time_regime_summary": regime_summary.to_dict(orient="records"),
                "deferred_metrics": deferred_payload,
                "rows": joined.head(max(1, int(limit))).to_dict(orient="records"),
            }
        )

    def _build_deferred_metric_payload(self, deferred_rows: pd.DataFrame) -> list[dict[str, Any]]:
        if deferred_rows.empty:
            return []
        rows = deferred_rows[
            [
                "metric_name",
                "statement_type",
                "metric_class",
                "scope_rule",
                "known_gap_flag",
                "known_gap_reason",
                "recommended_compare_mode",
                "notes",
                "wrds_table_name",
                "wrds_column_name",
                "sec_column_name",
                "sec_extra_column_name",
            ]
        ].to_dict(orient="records")
        for row in rows:
            row.update(self._deferred_metric_readiness(row))
        return rows

    def _deferred_metric_readiness(self, row: dict[str, Any]) -> dict[str, Any]:
        metric_name = str(row.get("metric_name") or "").strip()
        statement_type = str(row.get("statement_type") or "quarterly").strip().lower()
        wrds_table = str(
            row.get("wrds_table_name")
            or ("financials_quarterly_canonical" if statement_type == "quarterly" else "financials_annual_canonical")
        ).strip()
        sec_column = self._coerce_str(row.get("sec_column_name"))
        sec_extra_column = self._coerce_str(row.get("sec_extra_column_name"))
        wrds_column = self._coerce_str(row.get("wrds_column_name"))
        sec_base_non_null = self._safe_non_null_count("financials_quarterly", sec_column)
        sec_extra_non_null = self._safe_non_null_count("financials_quarterly_extra", sec_extra_column)
        wrds_non_null = self._safe_non_null_count(
            self._qualified_wrds_table_name(wrds_table),
            wrds_column,
            attachments=self.wrds_attachments,
        )
        raw_fact_rows = self._safe_raw_fact_count(sec_column)
        blockers: list[str] = []
        if wrds_non_null == 0:
            blockers.append("wrds_reference_missing")
        if sec_base_non_null == 0 and sec_extra_non_null == 0 and raw_fact_rows == 0:
            blockers.append("sec_materialization_missing")
        elif sec_base_non_null == 0 and sec_extra_non_null == 0:
            blockers.append("sec_materialization_not_promoted")
        blocker = ",".join(blockers) if blockers else "compare_ready_candidate"
        payload = {
            "sec_base_non_null_rows": sec_base_non_null,
            "sec_extra_non_null_rows": sec_extra_non_null,
            "sec_raw_fact_rows": raw_fact_rows,
            "wrds_reference_non_null_rows": wrds_non_null,
            "promotion_blocker": blocker,
        }
        aux_readiness = self._latest_aux_readiness(metric_name=metric_name, statement_type=statement_type)
        if aux_readiness:
            payload.update(aux_readiness)
        return payload

    def _latest_aux_readiness(self, *, metric_name: str, statement_type: str) -> dict[str, Any]:
        if not self.db.table_exists("wrds_aux_metric_readiness_summary"):
            return {}
        row = self.db.fetch_df(
            """
            SELECT *
            FROM wrds_aux_metric_readiness_summary
            WHERE metric_name = ?
              AND statement_type = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [metric_name, statement_type],
        )
        if row.empty:
            return {}
        record = row.iloc[0].to_dict()
        keep = {
            "wrds_reference_candidate",
            "wrds_reference_table",
            "wrds_reference_column",
            "candidate_non_null_rows",
            "candidate_non_null_issuers",
            "candidate_overlap_rows",
            "candidate_overlap_issuers",
            "blocker_type",
            "blocker_reason",
            "readiness_class",
            "pilot_compare_ready",
            "recommended_next_action",
        }
        return {key: record.get(key) for key in keep if key in record}

    def _safe_non_null_count(
        self,
        table_name: str,
        column_name: str | None,
        *,
        attachments: dict[str, Path] | None = None,
    ) -> int:
        if not column_name or not self.db.table_exists(table_name, attachments=attachments):
            return 0
        columns = {row["column_name"] for row in self.db.schema_info(table_name, attachments=attachments)}
        if column_name not in columns:
            return 0
        row = self.db.fetch_one(
            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{column_name}" IS NOT NULL'
            if "." not in table_name
            else f'SELECT COUNT(*) FROM {table_name} WHERE "{column_name}" IS NOT NULL',
            attachments=attachments,
        )
        return int(row[0]) if row and row[0] is not None else 0

    def _safe_raw_fact_count(self, fact_name: str | None) -> int:
        if not fact_name or not self.db.table_exists("sec_facts_raw_normalized"):
            return 0
        row = self.db.fetch_one(
            """
            SELECT COUNT(*)
            FROM sec_facts_raw_normalized
            WHERE fact_name = ?
            """,
            [fact_name],
        )
        return int(row[0]) if row and row[0] is not None else 0

    def _load_mappings(
        self,
        statement_types: tuple[str, ...],
        *,
        active_only: bool = True,
        metric_names: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        token_list = _quoted_literal_list(list(statement_types))
        where = [f"statement_type IN {token_list}"]
        if active_only:
            where.insert(0, "is_active")
        params: list[Any] = []
        if metric_names:
            where.append(f"LOWER(canonical_metric_name) IN {_quoted_literal_list([name.lower() for name in metric_names])}")
        return self.db.fetch_df(
            f"""
            SELECT *
            FROM metric_mapping_registry
            WHERE {" AND ".join(where)}
            ORDER BY statement_type, canonical_metric_name
            """,
            params,
        )

    def _load_active_mappings(self, statement_types: tuple[str, ...]) -> pd.DataFrame:
        return self._load_mappings(statement_types, active_only=True)

    def _resolve_target_tickers(
        self,
        *,
        tickers: str | list[str] | None,
        market: str,
        start_date: date,
        limit: int | None,
    ) -> list[str]:
        if tickers:
            if isinstance(tickers, str):
                tokens = [item.strip().upper() for item in tickers.split(",") if item.strip()]
            else:
                tokens = [str(item).strip().upper() for item in tickers if str(item).strip()]
            return tokens

        wrds_table = self._qualified_wrds_table_name("financials_quarterly_canonical")
        sql = f"""
            SELECT DISTINCT UPPER(w.ticker) AS ticker
            FROM {wrds_table} w
            INNER JOIN financials_quarterly s
                ON UPPER(w.ticker) = UPPER(s.ticker)
            WHERE w.period_end >= ?
              AND s.market = ?
              AND s."PeriodEnd" >= ?
            ORDER BY 1
        """
        if limit is not None:
            sql += f" LIMIT {max(1, int(limit))}"
        frame = self.db.fetch_df(sql, [start_date, market.lower(), start_date], attachments=self.wrds_attachments)
        return frame["ticker"].astype(str).str.upper().tolist() if not frame.empty else []

    def _load_wrds_rows(
        self,
        statement_type: str,
        tickers: list[str],
        start_date: date,
        mappings: pd.DataFrame,
    ) -> pd.DataFrame:
        if mappings.empty:
            return pd.DataFrame()
        wrds_tables = sorted({str(item) for item in mappings["wrds_table_name"].dropna().tolist() if str(item).strip()})
        if not wrds_tables:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []

        def _load_from_table(source_table_name: str, source_mappings: pd.DataFrame) -> pd.DataFrame:
            table_name = self._qualified_wrds_table_name(source_table_name)
            available_columns = self._table_columns(table_name, attachments=self.wrds_attachments)

            def _select_first_available(candidates: tuple[str, ...], alias_name: str) -> str:
                for candidate in candidates:
                    if candidate in available_columns:
                        return self._select_column_or_null(
                            table_alias=None,
                            source_name=candidate,
                            alias_name=alias_name,
                            available_columns=available_columns,
                        )
                return f"NULL AS {_quote(alias_name)}"

            metadata_columns = [
                _select_first_available(("ticker", "tic"), "ticker"),
                _select_first_available(("gvkey",), "gvkey"),
                _select_first_available(("cik",), "cik"),
                _select_first_available(("period_end", "datadate"), "period_end"),
                _select_first_available(("available_date", "rdq", "datadate", "period_end"), "available_date"),
                _select_first_available(("fiscal_year", "fyearq", "fyear"), "fiscal_year"),
                (
                    _select_first_available(("fiscal_quarter", "fqtr"), "fiscal_quarter")
                    if statement_type == "quarterly"
                    else "NULL AS \"fiscal_quarter\""
                ),
            ]
            metric_columns = []
            metric_aliases: list[str] = []
            for record in source_mappings[["canonical_metric_name", "wrds_column_name"]].drop_duplicates().to_dict(orient="records"):
                canonical_metric_name = str(record["canonical_metric_name"])
                wrds_column_name = str(record["wrds_column_name"])
                metric_aliases.append(canonical_metric_name)
                metric_columns.append(
                    self._select_column_or_null(
                        table_alias=None,
                        source_name=wrds_column_name,
                        alias_name=canonical_metric_name,
                        available_columns=available_columns,
                    )
                )
            cols = metadata_columns + metric_columns
            period_end_source = "period_end" if "period_end" in available_columns else "datadate"
            where = [f"{_quote(period_end_source)} >= DATE '{start_date.isoformat()}'"]
            if tickers:
                ticker_source = "ticker" if "ticker" in available_columns else "tic"
                if ticker_source in available_columns:
                    where.append(f"UPPER({_quote(ticker_source)}) IN {_quoted_literal_list(tickers)}")
            sql = f"""
                SELECT {", ".join(cols)}
                FROM {table_name}
                WHERE {" AND ".join(where)}
            """
            frame = self.db.fetch_df(sql, attachments=self.wrds_attachments)
            if frame.empty:
                return frame
            for column_name in ("ticker", "gvkey", "cik"):
                if column_name not in frame.columns:
                    frame[column_name] = pd.NA
            for column_name in ("period_end", "available_date"):
                if column_name not in frame.columns:
                    frame[column_name] = pd.NaT
            if "fiscal_year" not in frame.columns:
                frame["fiscal_year"] = pd.NA
            if "fiscal_quarter" not in frame.columns:
                frame["fiscal_quarter"] = pd.NA
            keep_cols = [
                "ticker",
                "gvkey",
                "cik",
                "period_end",
                "available_date",
                "fiscal_year",
                "fiscal_quarter",
                *metric_aliases,
            ]
            return frame[keep_cols]

        for source_table_name, source_mappings in mappings.groupby("wrds_table_name", dropna=True):
            source_name = str(source_table_name).strip()
            if not source_name:
                continue
            source_frame = _load_from_table(source_name, source_mappings)
            if not source_frame.empty:
                frames.append(source_frame)
        if not frames:
            return pd.DataFrame()

        frame = frames[0].copy()
        join_cols = ["gvkey", "period_end", "fiscal_year"]
        if statement_type == "quarterly":
            join_cols.append("fiscal_quarter")
        for next_frame in frames[1:]:
            frame = frame.merge(next_frame, on=join_cols, how="outer", suffixes=("", "__dup"))
            for meta_col in ("ticker", "cik", "available_date"):
                dup_col = f"{meta_col}__dup"
                if dup_col in frame.columns:
                    frame[meta_col] = frame[meta_col].combine_first(frame[dup_col])
                    frame = frame.drop(columns=[dup_col])

        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        for col in ("period_end", "available_date"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.normalize()
        frame["fiscal_year"] = pd.to_numeric(frame.get("fiscal_year"), errors="coerce").astype("Int64")
        frame["fiscal_quarter"] = pd.to_numeric(frame.get("fiscal_quarter"), errors="coerce").astype("Int64")
        frame["alignment_key"] = frame.apply(lambda row: self._alignment_key(statement_type, row), axis=1)
        frame["wrds_row_count"] = frame.groupby("alignment_key", dropna=False)["ticker"].transform("size")
        frame = frame.sort_values(
            ["alignment_key", "available_date", "period_end", "fiscal_year", "fiscal_quarter"],
            ascending=[True, True, True, True, True],
            na_position="last",
        )
        frame = frame.groupby("alignment_key", dropna=False, as_index=False).tail(1).reset_index(drop=True)
        frame["wrds_alignment_basis"] = frame.apply(lambda row: self._alignment_basis(statement_type, row), axis=1)
        return frame

    def _load_sec_rows(
        self,
        statement_type: str,
        market: str,
        tickers: list[str],
        start_date: date,
        mappings: pd.DataFrame,
    ) -> pd.DataFrame:
        if statement_type == "annual":
            source_rows = self._load_sec_source_rows(
                market=market,
                tickers=tickers,
                start_date=start_date,
                mappings=mappings,
                forms=ALL_SEC_FORMS,
            )
            return self._normalize_sec_annual_rows(source_rows, mappings)
        source_rows = self._load_sec_source_rows(
            market=market,
            tickers=tickers,
            start_date=start_date,
            mappings=mappings,
            forms=ALL_SEC_FORMS,
        )
        return self._collapse_sec_period_rows(source_rows, statement_type="quarterly")

    def _load_sec_source_rows(
        self,
        *,
        market: str,
        tickers: list[str],
        start_date: date,
        mappings: pd.DataFrame,
        forms: tuple[str, ...],
    ) -> pd.DataFrame:
        base_table_columns = self._table_columns("financials_quarterly")
        extra_table_columns = self._table_columns("financials_quarterly_extra")
        base_columns = [
            'q.ticker AS ticker',
            'q.market AS market',
            'q.term AS term',
            'q."PeriodEnd" AS period_end',
            'q."FormType" AS form_type',
            'q."FilingDate" AS filing_date',
            'q."AcceptedAt" AS accepted_at',
            'q."AvailableDate" AS available_date',
            'q.collected_at AS collected_at',
        ]
        sec_columns = sorted({str(col) for col in mappings["sec_column_name"].dropna().tolist()})
        fallback_columns = sorted({str(col) for col in mappings["sec_extra_column_name"].dropna().tolist()})
        metric_names = {str(name) for name in mappings["canonical_metric_name"].dropna().tolist()}
        if "common_stock" in metric_names:
            if "APIC" not in sec_columns:
                sec_columns.append("APIC")
            if "additional_paid_in_capital" not in fallback_columns:
                fallback_columns.append("additional_paid_in_capital")
        base_fallback_columns = [col for col in fallback_columns if col not in SEC_FINANCIALS_EXTRA_COLUMNS]
        extra_columns = [col for col in fallback_columns if col in SEC_FINANCIALS_EXTRA_COLUMNS]
        base_columns.extend(
            [
                self._select_column_or_null(
                    table_alias="q",
                    source_name=col,
                    alias_name=col,
                    available_columns=base_table_columns,
                )
                for col in sec_columns
            ]
        )
        base_columns.extend(
            [
                self._select_column_or_null(
                    table_alias="q",
                    source_name=col,
                    alias_name=col,
                    available_columns=base_table_columns,
                )
                for col in base_fallback_columns
                if col not in sec_columns
            ]
        )
        join_extra = bool(extra_columns) and self.db.table_exists("financials_quarterly_extra")
        if join_extra:
            base_columns.extend(
                [
                    self._select_column_or_null(
                        table_alias="e",
                        source_name=col,
                        alias_name="extra__" + col,
                        available_columns=extra_table_columns,
                    )
                    for col in extra_columns
                ]
            )

        where = [f"q.market = '{market.lower()}'", f'q."PeriodEnd" >= DATE \'{start_date.isoformat()}\'']
        where.append(f"COALESCE(q.\"FormType\", '') IN {_quoted_literal_list(list(forms))}")
        if tickers:
            where.append(f"UPPER(q.ticker) IN {_quoted_literal_list(tickers)}")

        join_clause = (
            'LEFT JOIN financials_quarterly_extra e ON q.ticker = e.ticker'
            ' AND q.market = e.market AND q."PeriodEnd" = e.period_end'
            ' AND q."FormType" = e.form_type'
            ' AND (q."FilingDate" = e.filing_date'
            ' OR (q."FilingDate" IS NULL AND e.filing_date IS NULL))'
        ) if join_extra else ""
        sql = f"""
            SELECT {", ".join(base_columns)}
            FROM financials_quarterly q
            {join_clause}
            WHERE {" AND ".join(where)}
        """
        frame = self.db.fetch_df(sql)
        if frame.empty:
            return frame

        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        for col in ("period_end", "filing_date", "accepted_at", "available_date"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")
                if col != "accepted_at":
                    frame[col] = frame[col].dt.normalize()
        frame["fiscal_year"], frame["fiscal_quarter"], frame["sec_term_fallback"] = self._derive_sec_fiscal_fields(
            frame.get("term"),
            frame.get("period_end"),
            "quarterly",
        )
        frame = self._refine_sec_fiscal_fields(frame, statement_type="quarterly")
        frame["form_priority"] = frame.get("form_type").astype("string").map(SEC_ROW_FORM_PRIORITY).fillna(0).astype("Int64")
        frame["is_amendment"] = frame.get("form_type").astype("string").str.endswith("/A", na=False)
        return frame

    def _collapse_sec_period_rows(self, frame: pd.DataFrame, *, statement_type: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        out["alignment_key"] = out.apply(lambda row: self._alignment_key(statement_type, row), axis=1)
        out["sec_row_count"] = out.groupby("alignment_key", dropna=False)["ticker"].transform("size")
        out["sec_amendment_chain_only"] = out.groupby("alignment_key", dropna=False)["form_type"].transform(
            lambda s: self._is_amendment_chain_only(s)
        )
        out = out.sort_values(
            ["alignment_key", "available_date", "filing_date", "accepted_at", "form_priority", "period_end"],
            ascending=[True, True, True, True, True, True],
            na_position="last",
        )
        out = out.groupby("alignment_key", dropna=False, as_index=False).tail(1).reset_index(drop=True)
        out["sec_row_count"] = out["sec_row_count"].where(~out["sec_amendment_chain_only"].fillna(False), 1)
        out["sec_alignment_basis"] = out.apply(lambda row: self._alignment_basis(statement_type, row), axis=1)
        return out

    def _normalize_sec_annual_rows(self, frame: pd.DataFrame, mappings: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame.copy()
        out = out.loc[out["fiscal_year"].notna()].copy()
        if out.empty:
            return out
        out["fiscal_year"] = pd.to_numeric(out["fiscal_year"], errors="coerce").astype("Int64")
        out["fiscal_quarter"] = pd.to_numeric(out["fiscal_quarter"], errors="coerce").astype("Int64")
        out["annual_key"] = out["ticker"].astype(str).str.upper() + "|annual|FY" + out["fiscal_year"].astype(str)
        out["quarter_key"] = out["ticker"].astype(str).str.upper() + "|quarterly|FY" + out["fiscal_year"].astype(str)
        out["quarter_key"] = out["quarter_key"] + "Q" + out["fiscal_quarter"].astype(str)

        quarter_rows = out.loc[out["fiscal_quarter"].notna()].copy()
        if quarter_rows.empty:
            return pd.DataFrame()
        quarter_rows["quarter_key"] = quarter_rows["ticker"].astype(str).str.upper() + "|quarterly|FY" + quarter_rows["fiscal_year"].astype(str) + "Q" + quarter_rows["fiscal_quarter"].astype(str)
        quarter_rows["quarter_candidate_count"] = quarter_rows.groupby("quarter_key", dropna=False)["ticker"].transform("size")
        quarter_rows = quarter_rows.sort_values(
            ["quarter_key", "available_date", "filing_date", "accepted_at", "form_priority", "period_end"],
            ascending=[True, True, True, True, True, True],
            na_position="last",
        )
        quarter_rows = quarter_rows.groupby("quarter_key", dropna=False, as_index=False).tail(1).reset_index(drop=True)

        annual_candidates = out.loc[out.get("form_type").astype("string").isin(ANNUAL_SEC_FORMS)].copy()
        if annual_candidates.empty:
            annual_candidates = quarter_rows.loc[quarter_rows["fiscal_quarter"] == 4].copy()
        annual_candidates["annual_candidate_count"] = annual_candidates.groupby("annual_key", dropna=False)["ticker"].transform("size")
        annual_candidates = annual_candidates.sort_values(
            ["annual_key", "available_date", "filing_date", "accepted_at", "form_priority", "period_end"],
            ascending=[True, True, True, True, True, True],
            na_position="last",
        )
        annual_anchor = annual_candidates.groupby("annual_key", dropna=False, as_index=False).tail(1).reset_index(drop=True)
        raw_annual_lookup = self._build_raw_annual_fact_lookup(annual_anchor)

        sec_columns = sorted({str(col) for col in mappings["sec_column_name"].dropna().tolist()})
        fallback_columns = sorted({str(col) for col in mappings["sec_extra_column_name"].dropna().tolist()})
        base_fallback_columns = [col for col in fallback_columns if col not in SEC_FINANCIALS_EXTRA_COLUMNS]
        extra_columns = [f"extra__{col}" for col in fallback_columns if col in SEC_FINANCIALS_EXTRA_COLUMNS]
        column_names = sec_columns + [col for col in base_fallback_columns if col not in sec_columns] + extra_columns
        aggregate_rules = self._annual_aggregate_rules(mappings)

        normalized_rows: list[dict[str, Any]] = []
        for annual_key, anchor in annual_anchor.groupby("annual_key", dropna=False):
            anchor_row = anchor.iloc[-1]
            quarters = quarter_rows.loc[quarter_rows["annual_key"] == annual_key].copy()
            if quarters.empty:
                quarters = out.loc[out["annual_key"] == annual_key].copy()
            quarter_coverage = int(quarters["fiscal_quarter"].dropna().astype(int).nunique()) if not quarters.empty else 0
            q4_rows = quarters.loc[quarters["fiscal_quarter"] == 4].copy()
            q4_row = q4_rows.iloc[-1] if not q4_rows.empty else quarters.sort_values(
                ["fiscal_quarter", "available_date", "filing_date", "accepted_at", "form_priority", "period_end"],
                ascending=[True, True, True, True, True, True],
                na_position="last",
            ).iloc[-1]
            row_payload = anchor_row.to_dict()
            row_payload["alignment_key"] = annual_key
            row_payload["sec_row_count"] = 1
            row_payload["annual_candidate_count"] = self._coerce_int(anchor_row.get("annual_candidate_count")) or 1
            row_payload["annual_quarter_coverage"] = quarter_coverage
            row_payload["sec_alignment_basis"] = "fiscal_period"
            row_payload["period_end"] = anchor_row.get("period_end") if pd.notna(anchor_row.get("period_end")) else q4_row.get("period_end")
            row_payload["filing_date"] = anchor_row.get("filing_date")
            row_payload["accepted_at"] = anchor_row.get("accepted_at")
            row_payload["available_date"] = anchor_row.get("available_date")
            row_payload["form_type"] = anchor_row.get("form_type")
            row_payload["fiscal_quarter"] = pd.NA
            for column_name in column_names:
                if column_name not in quarters.columns and column_name not in row_payload:
                    continue
                anchor_value = anchor_row.get(column_name)
                reported_value = anchor_value if pd.notna(anchor_value) else q4_row.get(column_name)
                row_payload[f"reported__{column_name}"] = reported_value
                rule = aggregate_rules.get(column_name, "last")
                if rule == "sum":
                    series = pd.to_numeric(quarters.get(column_name), errors="coerce")
                    value = series.sum(min_count=1) if series is not None else pd.NA
                else:
                    value = anchor_value if pd.notna(anchor_value) else q4_row.get(column_name)
                row_payload[column_name] = value
                row_payload[f"normalized__{column_name}"] = value
            self._apply_raw_annual_eps_override(row_payload, anchor_row, raw_annual_lookup)
            for column_name in ("Diluted EPS", "EPS"):
                if column_name in row_payload:
                    row_payload[f"normalized__{column_name}"] = row_payload.get(column_name)
            normalized_rows.append(row_payload)

        normalized = pd.DataFrame(normalized_rows)
        if normalized.empty:
            return normalized
        normalized["fiscal_year"] = pd.to_numeric(normalized.get("fiscal_year"), errors="coerce").astype("Int64")
        normalized["fiscal_quarter"] = pd.to_numeric(normalized.get("fiscal_quarter"), errors="coerce").astype("Int64")
        normalized["period_end"] = pd.to_datetime(normalized.get("period_end"), errors="coerce").dt.normalize()
        normalized["available_date"] = pd.to_datetime(normalized.get("available_date"), errors="coerce").dt.normalize()
        normalized["filing_date"] = pd.to_datetime(normalized.get("filing_date"), errors="coerce").dt.normalize()
        normalized["accepted_at"] = pd.to_datetime(normalized.get("accepted_at"), errors="coerce", utc=True)
        return normalized

    def _build_raw_annual_fact_lookup(self, annual_anchor: pd.DataFrame) -> dict[tuple[str, str, date | None, date | None], dict[str, Any]]:
        if annual_anchor.empty:
            return {}
        if not self.db.table_exists("sec_facts_raw_normalized"):
            return {}
        market_series = annual_anchor.get("market")
        market = "us"
        if market_series is not None:
            market_non_null = market_series.dropna()
            if not market_non_null.empty:
                market = str(market_non_null.iloc[0]).strip().lower() or "us"
        tickers = sorted(annual_anchor["ticker"].dropna().astype(str).str.upper().unique().tolist())
        if not tickers:
            return {}
        period_end_series = pd.to_datetime(annual_anchor.get("period_end"), errors="coerce")
        period_end_series = period_end_series.dropna()
        if period_end_series.empty:
            return {}
        min_period_end = period_end_series.min().date().isoformat()
        sql = f"""
            SELECT
                UPPER(ticker) AS ticker,
                UPPER(form_type) AS form_type,
                period_end,
                filing_date,
                available_date,
                accepted_at,
                fact_name,
                value
            FROM sec_facts_raw_normalized
            WHERE market = '{market}'
              AND UPPER(ticker) IN {_quoted_literal_list(tickers)}
              AND UPPER(form_type) IN {_quoted_literal_list(list(ANNUAL_SEC_FORMS))}
              AND fact_name IN {_quoted_literal_list(list(ANNUAL_RAW_SEC_FACT_NAMES))}
              AND period_end >= DATE '{min_period_end}'
        """
        raw = self.db.fetch_df(sql)
        if raw.empty:
            return {}
        for col in ("period_end", "filing_date", "available_date"):
            raw[col] = pd.to_datetime(raw[col], errors="coerce").dt.normalize()
        raw["accepted_at"] = pd.to_datetime(raw.get("accepted_at"), errors="coerce", utc=True)
        raw = raw.sort_values(
            ["ticker", "form_type", "filing_date", "period_end", "fact_name", "available_date", "accepted_at"],
            ascending=[True, True, True, True, True, True, True],
            na_position="last",
        )
        raw = raw.groupby(["ticker", "form_type", "filing_date", "period_end", "fact_name"], dropna=False, as_index=False).tail(1)
        wide = (
            raw.pivot_table(
                index=["ticker", "form_type", "filing_date", "period_end"],
                columns="fact_name",
                values="value",
                aggfunc="last",
            )
            .reset_index()
        )
        wide["form_priority"] = wide.get("form_type").astype("string").map(SEC_ROW_FORM_PRIORITY).fillna(0).astype("Int64")
        wide = wide.sort_values(
            ["ticker", "period_end", "filing_date", "form_priority"],
            ascending=[True, True, True, True],
            na_position="last",
        )
        lookup: dict[tuple[str, str, date | None, date | None], dict[str, Any]] = {}
        for record in wide.to_dict(orient="records"):
            key = (
                str(record.get("ticker") or "").upper(),
                str(record.get("form_type") or "").upper(),
                self._to_date(record.get("filing_date")),
                self._to_date(record.get("period_end")),
            )
            lookup[key] = record
        fallback_rows = wide.groupby(["ticker", "period_end"], dropna=False, as_index=False).tail(1)
        for record in fallback_rows.to_dict(orient="records"):
            fallback_key = (
                str(record.get("ticker") or "").upper(),
                "",
                None,
                self._to_date(record.get("period_end")),
            )
            lookup[fallback_key] = record
        return lookup

    def _build_raw_compare_lookup(
        self,
        *,
        market: str,
        tickers: list[str],
        start_date: date,
        mappings: pd.DataFrame,
    ) -> dict[tuple[str, str, date | None, str], dict[str, Any]]:
        if not tickers or mappings.empty:
            return {}
        if not self.db.table_exists("sec_facts_raw_normalized"):
            return {}
        metric_frame = (
            mappings.loc[mappings["sec_column_name"].notna(), ["canonical_metric_name", "statement_type", "sec_column_name"]]
            .drop_duplicates()
            .copy()
        )
        if metric_frame.empty:
            return {}
        fact_names = sorted(metric_frame["sec_column_name"].astype(str).str.strip().unique().tolist())
        sql = f"""
            SELECT
                UPPER(ticker) AS ticker,
                UPPER(form_type) AS form_type,
                fact_name,
                period_start,
                period_end,
                filing_date,
                available_date,
                accepted_at,
                value
            FROM sec_facts_raw_normalized
            WHERE market = '{market}'
              AND UPPER(ticker) IN {_quoted_literal_list(tickers)}
              AND fact_name IN {_quoted_literal_list(fact_names)}
              AND period_end >= DATE '{start_date.isoformat()}'
        """
        raw = self.db.fetch_df(sql)
        if raw.empty:
            return {}
        for col in ("period_start", "period_end", "filing_date", "available_date"):
            raw[col] = pd.to_datetime(raw[col], errors="coerce").dt.normalize()
        raw["accepted_at"] = pd.to_datetime(raw.get("accepted_at"), errors="coerce", utc=True)
        raw["form_priority"] = raw.get("form_type").astype("string").map(SEC_ROW_FORM_PRIORITY).fillna(0).astype("Int64")
        raw["duration_days"] = (
            pd.to_datetime(raw["period_end"], errors="coerce") - pd.to_datetime(raw["period_start"], errors="coerce")
        ).dt.days.astype("Float64")

        lookup: dict[tuple[str, str, date | None, str], dict[str, Any]] = {}
        for metric in metric_frame.to_dict(orient="records"):
            metric_name = str(metric["canonical_metric_name"])
            statement_type = str(metric["statement_type"])
            fact_name = str(metric["sec_column_name"])
            subset = raw.loc[raw["fact_name"] == fact_name].copy()
            if subset.empty:
                continue
            for (ticker, period_end), candidates in subset.groupby(["ticker", "period_end"], dropna=False):
                selected = self._select_raw_compare_candidate(
                    candidates,
                    metric_name=metric_name,
                    statement_type=statement_type,
                )
                if selected is None:
                    continue
                lookup[(statement_type, str(ticker).upper(), self._to_date(period_end), fact_name)] = selected
        return lookup

    @staticmethod
    def _select_raw_compare_candidate(
        frame: pd.DataFrame,
        *,
        metric_name: str,
        statement_type: str,
    ) -> dict[str, Any] | None:
        if frame.empty:
            return None
        candidates = frame.copy()
        flow_metric = metric_name in ANNUAL_AGGREGATE_FLOW_METRICS or metric_name == "eps"
        if statement_type == "annual":
            annual_form_candidates = candidates.loc[candidates["form_type"].isin(ANNUAL_SEC_FORMS)].copy()
            if not annual_form_candidates.empty:
                candidates = annual_form_candidates
            if flow_metric:
                duration_candidates = candidates.loc[candidates["duration_days"].notna()].copy()
                if not duration_candidates.empty:
                    candidates = duration_candidates
                    candidates = candidates.sort_values(
                        ["duration_days", "available_date", "accepted_at", "filing_date", "form_priority"],
                        ascending=[False, False, False, False, False],
                        na_position="last",
                    )
                    selected = candidates.iloc[0]
                    return {
                        "value": selected.get("value"),
                        "source_code": RAW_SOURCE_PERIOD_END_FALLBACK,
                        "selection_rule": "annual_longest_duration",
                    }
            candidates = candidates.sort_values(
                ["available_date", "accepted_at", "filing_date", "form_priority"],
                ascending=[False, False, False, False],
                na_position="last",
            )
            selected = candidates.iloc[0]
            return {
                "value": selected.get("value"),
                "source_code": RAW_SOURCE_PERIOD_END_FALLBACK,
                "selection_rule": "annual_latest_period_end",
            }

        if flow_metric:
            duration_candidates = candidates.loc[
                candidates["duration_days"].notna() & (candidates["duration_days"] <= RAW_QUARTER_MAX_DURATION_DAYS)
            ].copy()
            if duration_candidates.empty:
                return None
            duration_candidates = duration_candidates.sort_values(
                ["duration_days", "available_date", "accepted_at", "filing_date", "form_priority"],
                ascending=[True, False, False, False, False],
                na_position="last",
            )
            selected = duration_candidates.iloc[0]
            return {
                "value": selected.get("value"),
                "source_code": RAW_SOURCE_PERIOD_END_FALLBACK,
                "selection_rule": "quarter_shortest_duration",
            }

        candidates = candidates.sort_values(
            ["available_date", "accepted_at", "filing_date", "form_priority"],
            ascending=[False, False, False, False],
            na_position="last",
        )
        selected = candidates.iloc[0]
        return {
            "value": selected.get("value"),
            "source_code": RAW_SOURCE_PERIOD_END_FALLBACK,
            "selection_rule": "stock_latest_period_end",
        }

    def _apply_raw_annual_eps_override(
        self,
        row_payload: dict[str, Any],
        anchor_row: pd.Series,
        raw_lookup: dict[tuple[str, str, date | None, date | None], dict[str, Any]],
    ) -> None:
        key = (
            str(anchor_row.get("ticker") or "").upper(),
            str(anchor_row.get("form_type") or "").upper(),
            self._to_date(anchor_row.get("filing_date")),
            self._to_date(anchor_row.get("period_end")),
        )
        raw_row = raw_lookup.get(key)
        if not raw_row:
            raw_row = raw_lookup.get((key[0], "", None, key[3]))
        if not raw_row:
            return
        diluted_eps = self._coerce_float(raw_row.get("Diluted EPS"))
        basic_eps = self._coerce_float(raw_row.get("EPS"))
        diluted_shares = self._coerce_float(raw_row.get("Diluted Shares"))
        net_income = self._coerce_float(raw_row.get("Net Income"))
        net_income_common = self._coerce_float(raw_row.get("Net Income Common"))
        if diluted_eps is None and diluted_shares not in (None, 0.0):
            numerator = net_income_common if net_income_common is not None else net_income
            if numerator is not None:
                diluted_eps = numerator / diluted_shares
        if diluted_eps is not None:
            row_payload["Diluted EPS"] = diluted_eps
        if basic_eps is not None:
            row_payload["EPS"] = basic_eps
        elif diluted_eps is not None and pd.isna(row_payload.get("EPS")):
            row_payload["EPS"] = diluted_eps

    def _align_rows(self, statement_type: str, wrds_rows: pd.DataFrame, sec_rows: pd.DataFrame) -> pd.DataFrame:
        wrds = wrds_rows.add_prefix("wrds__") if not wrds_rows.empty else pd.DataFrame(columns=["wrds__alignment_key"])
        sec = sec_rows.add_prefix("sec__") if not sec_rows.empty else pd.DataFrame(columns=["sec__alignment_key"])
        merged = wrds.merge(
            sec,
            left_on="wrds__alignment_key",
            right_on="sec__alignment_key",
            how="outer",
            indicator=True,
        )
        merged["statement_type"] = statement_type
        merged["alignment_key"] = merged["wrds__alignment_key"].combine_first(merged["sec__alignment_key"])
        wrds_ticker = merged.get("wrds__ticker", pd.Series(pd.NA, index=merged.index, dtype="object"))
        sec_ticker = merged.get("sec__ticker", pd.Series(pd.NA, index=merged.index, dtype="object"))
        wrds_fy = merged.get("wrds__fiscal_year", pd.Series(pd.NA, index=merged.index, dtype="Int64"))
        sec_fy = merged.get("sec__fiscal_year", pd.Series(pd.NA, index=merged.index, dtype="Int64"))
        wrds_fq = merged.get("wrds__fiscal_quarter", pd.Series(pd.NA, index=merged.index, dtype="Int64"))
        sec_fq = merged.get("sec__fiscal_quarter", pd.Series(pd.NA, index=merged.index, dtype="Int64"))
        wrds_pe = merged.get("wrds__period_end", pd.Series(pd.NaT, index=merged.index, dtype="datetime64[ns]"))
        sec_pe = merged.get("sec__period_end", pd.Series(pd.NaT, index=merged.index, dtype="datetime64[ns]"))
        merged["ticker"] = self._coalesce_prefer_left(wrds_ticker, sec_ticker)
        merged["fiscal_year"] = self._coalesce_prefer_left(wrds_fy, sec_fy)
        merged["fiscal_quarter"] = self._coalesce_prefer_left(wrds_fq, sec_fq)
        merged["period_end"] = self._coalesce_prefer_left(wrds_pe, sec_pe)
        delta = (
            pd.to_datetime(wrds_pe, errors="coerce")
            - pd.to_datetime(sec_pe, errors="coerce")
        ).abs()
        merged["period_end_delta_days"] = delta.dt.days.astype("Int64")
        merged["sequence_alignment_candidate"] = False
        merged["sequence_alignment_fallback"] = False
        merged["fiscal_calendar_shift_alignment"] = False
        if statement_type == "quarterly":
            merged = self._apply_sequence_alignment_fallback(merged, wrds_rows, sec_rows)
        return merged

    def _apply_sequence_alignment_fallback(
        self,
        merged: pd.DataFrame,
        wrds_rows: pd.DataFrame,
        sec_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        if merged.empty or wrds_rows.empty or sec_rows.empty:
            return merged
        if "ticker" not in merged.columns:
            return merged

        ticker_frames: list[pd.DataFrame] = []
        for ticker, ticker_frame in merged.groupby("ticker", dropna=False, sort=False):
            if not isinstance(ticker, str) or not ticker.strip():
                ticker_frames.append(ticker_frame.copy())
                continue
            wrds_ticker = wrds_rows.loc[wrds_rows["ticker"].astype(str).str.upper() == ticker].copy()
            sec_ticker = sec_rows.loc[sec_rows["ticker"].astype(str).str.upper() == ticker].copy()
            candidate, apply_fallback, sec_offset = self._sequence_alignment_decision(
                ticker_frame=ticker_frame,
                wrds_rows=wrds_ticker,
                sec_rows=sec_ticker,
            )
            if not candidate:
                ticker_frames.append(ticker_frame.copy())
                continue
            if not apply_fallback:
                candidate_frame = ticker_frame.copy()
                candidate_frame["sequence_alignment_candidate"] = True
                ticker_frames.append(candidate_frame)
                continue
            ticker_frames.append(
                self._sequence_align_ticker_rows(
                    ticker=ticker,
                    wrds_rows=wrds_ticker,
                    sec_rows=sec_ticker,
                    sec_offset=sec_offset,
                )
            )
        if not ticker_frames:
            return merged
        return pd.concat(ticker_frames, ignore_index=True, sort=False)

    def _sequence_alignment_decision(
        self,
        *,
        ticker_frame: pd.DataFrame,
        wrds_rows: pd.DataFrame,
        sec_rows: pd.DataFrame,
    ) -> tuple[bool, bool, int]:
        if wrds_rows.empty or sec_rows.empty:
            return False, False, 0
        wrds_sorted = wrds_rows.sort_values("period_end").reset_index(drop=True)
        sec_sorted = sec_rows.sort_values("period_end").reset_index(drop=True)
        if len(wrds_sorted) < 3:
            return False, False, 0
        if not wrds_sorted["period_end"].is_monotonic_increasing or not sec_sorted["period_end"].is_monotonic_increasing:
            return False, False, 0

        shared = ticker_frame.loc[
            ticker_frame["wrds__period_end"].notna() & ticker_frame["sec__period_end"].notna()
        ].copy()
        if shared.empty:
            return False, False, 0
        large_delta_mask = shared["period_end_delta_days"].between(
            SEQUENCE_ALIGNMENT_MIN_DELTA_DAYS,
            SEQUENCE_ALIGNMENT_MAX_DELTA_DAYS,
            inclusive="both",
        )
        if not bool(large_delta_mask.any()):
            return False, False, 0

        candidate = float(large_delta_mask.mean()) >= 0.5
        if not candidate:
            return False, False, 0
        best_offset: int | None = None
        best_pair_count = -1
        for sec_offset in (0, 1, -1):
            pair_deltas = self._sequence_pair_deltas(wrds_sorted, sec_sorted, sec_offset=sec_offset)
            if pair_deltas.empty:
                continue
            if len(pair_deltas) < 3:
                continue
            if not bool((pair_deltas <= PERIOD_END_ALIGNMENT_TOLERANCE_DAYS).all()):
                continue
            if len(pair_deltas) > best_pair_count:
                best_pair_count = len(pair_deltas)
                best_offset = sec_offset
        apply_fallback = best_offset is not None
        return candidate, apply_fallback, int(best_offset or 0)

    @staticmethod
    def _sequence_pair_deltas(
        wrds_rows: pd.DataFrame,
        sec_rows: pd.DataFrame,
        *,
        sec_offset: int,
    ) -> pd.Series:
        deltas: list[int] = []
        for wrds_index in range(len(wrds_rows)):
            sec_index = wrds_index + sec_offset
            if sec_index < 0 or sec_index >= len(sec_rows):
                continue
            wrds_period_end = pd.to_datetime(wrds_rows.iloc[wrds_index].get("period_end"), errors="coerce")
            sec_period_end = pd.to_datetime(sec_rows.iloc[sec_index].get("period_end"), errors="coerce")
            if pd.isna(wrds_period_end) or pd.isna(sec_period_end):
                continue
            deltas.append(abs((wrds_period_end - sec_period_end).days))
        return pd.Series(deltas, dtype="Int64")

    def _sequence_align_ticker_rows(
        self,
        *,
        ticker: str,
        wrds_rows: pd.DataFrame,
        sec_rows: pd.DataFrame,
        sec_offset: int,
    ) -> pd.DataFrame:
        wrds_sorted = wrds_rows.sort_values("period_end").reset_index(drop=True)
        sec_sorted = sec_rows.sort_values("period_end").reset_index(drop=True)
        rows: list[dict[str, Any]] = []
        used_sec_indexes: set[int] = set()
        for wrds_index in range(len(wrds_sorted)):
            sec_index = wrds_index + sec_offset
            wrds_row = wrds_sorted.iloc[wrds_index].to_dict()
            sec_row = {}
            if 0 <= sec_index < len(sec_sorted):
                sec_row = sec_sorted.iloc[sec_index].to_dict()
                used_sec_indexes.add(sec_index)
            payload: dict[str, Any] = {}
            for key, value in wrds_row.items():
                payload[f"wrds__{key}"] = value
            for key, value in sec_row.items():
                payload[f"sec__{key}"] = value
            payload["statement_type"] = "quarterly"
            payload["alignment_key"] = wrds_row.get("alignment_key") or sec_row.get("alignment_key")
            payload["ticker"] = ticker
            payload["fiscal_year"] = wrds_row.get("fiscal_year", sec_row.get("fiscal_year"))
            payload["fiscal_quarter"] = wrds_row.get("fiscal_quarter", sec_row.get("fiscal_quarter"))
            payload["period_end"] = wrds_row.get("period_end", sec_row.get("period_end"))
            wrds_period_end = pd.to_datetime(wrds_row.get("period_end"), errors="coerce")
            sec_period_end = pd.to_datetime(sec_row.get("period_end"), errors="coerce")
            payload["period_end_delta_days"] = (
                abs((wrds_period_end - sec_period_end).days)
                if pd.notna(wrds_period_end) and pd.notna(sec_period_end)
                else pd.NA
            )
            exact_or_within_window = (
                pd.notna(payload["period_end_delta_days"])
                and int(payload["period_end_delta_days"]) <= PERIOD_END_ALIGNMENT_TOLERANCE_DAYS
            )
            fiscal_quarter_shift = (
                pd.notna(wrds_row.get("fiscal_quarter"))
                and pd.notna(sec_row.get("fiscal_quarter"))
                and int(wrds_row.get("fiscal_quarter")) != int(sec_row.get("fiscal_quarter"))
            )
            requires_shift = bool(sec_row) and (
                sec_offset != 0 or fiscal_quarter_shift or not exact_or_within_window
            )
            payload["sequence_alignment_candidate"] = True
            payload["sequence_alignment_fallback"] = requires_shift
            payload["fiscal_calendar_shift_alignment"] = requires_shift
            rows.append(payload)

        for sec_index in range(len(sec_sorted)):
            if sec_index in used_sec_indexes:
                continue
            sec_row = sec_sorted.iloc[sec_index].to_dict()
            payload: dict[str, Any] = {}
            for key, value in sec_row.items():
                payload[f"sec__{key}"] = value
            payload["statement_type"] = "quarterly"
            payload["alignment_key"] = sec_row.get("alignment_key")
            payload["ticker"] = ticker
            payload["fiscal_year"] = sec_row.get("fiscal_year")
            payload["fiscal_quarter"] = sec_row.get("fiscal_quarter")
            payload["period_end"] = sec_row.get("period_end")
            sec_period_end = pd.to_datetime(sec_row.get("period_end"), errors="coerce")
            payload["period_end_delta_days"] = pd.NA if pd.notna(sec_period_end) else pd.NA
            payload["sequence_alignment_candidate"] = True
            payload["sequence_alignment_fallback"] = sec_offset != 0
            payload["fiscal_calendar_shift_alignment"] = sec_offset != 0
            rows.append(payload)
        return pd.DataFrame(rows)

    def _build_result_rows(
        self,
        *,
        merged: pd.DataFrame,
        mappings: pd.DataFrame,
        market: str,
        comparison_run_id: str,
        created_at: pd.Timestamp,
        compare_mode: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _, aligned in merged.iterrows():
            wrds_row_count = self._coerce_int(aligned.get("wrds__wrds_row_count")) or 0
            sec_row_count = self._coerce_int(aligned.get("sec__sec_row_count")) or 0
            for _, mapping in mappings.iterrows():
                period_end = pd.to_datetime(aligned.get("period_end"), errors="coerce")
                time_regime = self._time_regime(period_end)
                effective_policy = self._effective_policy(mapping, time_regime=time_regime)
                canonical_metric_name = str(mapping.get("canonical_metric_name") or "")
                raw_wrds_value = self._coerce_float(aligned.get(f"wrds__{canonical_metric_name}"))
                if raw_wrds_value is None:
                    raw_wrds_value = self._coerce_float(aligned.get(f"wrds__{mapping['wrds_column_name']}"))
                reported_sec_value, reported_source = self._extract_reported_sec_value(aligned, mapping)
                normalized_sec_value, normalized_source = self._extract_normalized_sec_value(aligned, mapping)
                value_mode_used = self._resolve_value_mode(mapping, compare_mode, time_regime=time_regime)

                wrds_reported_value, reported_sec_value = self._apply_value_scale(
                    raw_wrds_value,
                    reported_sec_value,
                    str(mapping.get("value_scale_rule") or "as_is"),
                )
                wrds_reported_value, reported_sec_value = self._apply_sign_rule(
                    wrds_reported_value,
                    reported_sec_value,
                    str(mapping["sign_rule"]),
                )
                wrds_normalized_value, normalized_sec_value = self._apply_value_scale(
                    raw_wrds_value,
                    normalized_sec_value,
                    str(mapping.get("value_scale_rule") or "as_is"),
                )
                wrds_normalized_value, normalized_sec_value = self._apply_sign_rule(
                    wrds_normalized_value,
                    normalized_sec_value,
                    str(mapping["sign_rule"]),
                )
                if value_mode_used == VALUE_MODE_REPORTED:
                    wrds_value = wrds_reported_value
                    sec_value = reported_sec_value
                else:
                    wrds_value = wrds_normalized_value
                    sec_value = normalized_sec_value
                missing_on_wrds = wrds_value is None
                missing_on_sec = sec_value is None
                abs_diff = abs(wrds_value - sec_value) if not missing_on_wrds and not missing_on_sec else None
                pct_diff = self._pct_diff(abs_diff, wrds_value) if abs_diff is not None else None
                sign_mismatch = (
                    False
                    if missing_on_wrds or missing_on_sec
                    else (wrds_value != 0 and sec_value != 0 and (wrds_value > 0) != (sec_value > 0))
                )
                tolerance_breach = self._tolerance_breach(
                    abs_diff=abs_diff,
                    pct_diff=pct_diff,
                    wrds_value=wrds_value,
                    sec_value=sec_value,
                    tolerance_type=str(mapping["tolerance_type"]),
                    tolerance_value=float(effective_policy["tolerance_value"]),
                )
                diagnostic_codes: list[str] = []
                if time_regime == TIME_REGIME_PRE_2012:
                    diagnostic_codes.append("pre_2012_compatibility_mode")
                    if not bool(effective_policy["supported_in_time_regime"]):
                        diagnostic_codes.append("pre_2012_reduced_support")
                    if value_mode_used == VALUE_MODE_REPORTED:
                        diagnostic_codes.append("pre_2012_reported_only")
                if wrds_row_count > 1:
                    diagnostic_codes.append("duplicate_wrds_row")
                if sec_row_count > 1:
                    diagnostic_codes.append("duplicate_sec_row")
                if bool(aligned.get("sec__sec_term_fallback")):
                    diagnostic_codes.append("sec_term_missing_fallback")
                if bool(aligned.get("sec__sec_fiscal_anchor_based")):
                    diagnostic_codes.append("sec_fiscal_anchor_based")
                if bool(aligned.get("sec__sec_fiscal_anchor_shifted")):
                    diagnostic_codes.append("sec_fiscal_anchor_resequenced")
                delta_days = aligned.get("period_end_delta_days")
                if pd.notna(delta_days) and int(delta_days) > PERIOD_END_ALIGNMENT_TOLERANCE_DAYS:
                    diagnostic_codes.append("period_end_delta_exceeds_window")
                if bool(aligned.get("sequence_alignment_candidate")):
                    diagnostic_codes.append("sequence_alignment_candidate")
                if bool(aligned.get("sequence_alignment_fallback")):
                    diagnostic_codes.append("sequence_alignment_fallback")
                if bool(aligned.get("fiscal_calendar_shift_alignment")):
                    diagnostic_codes.append("fiscal_calendar_shift_alignment")
                if sign_mismatch:
                    diagnostic_codes.append("sign_mismatch")
                if missing_on_wrds:
                    diagnostic_codes.append("missing_wrds_metric")
                if missing_on_sec:
                    diagnostic_codes.append("missing_sec_metric")
                if reported_source == RAW_SOURCE_PERIOD_END_FALLBACK:
                    diagnostic_codes.append("reported_raw_fact_period_end_fallback")
                if normalized_source == RAW_SOURCE_PERIOD_END_FALLBACK:
                    diagnostic_codes.append("normalized_raw_fact_period_end_fallback")
                metric_name = str(mapping.get("canonical_metric_name") or "").strip()
                if (
                    metric_name == "pretax_income"
                    and normalized_source == RAW_SOURCE_NORMALIZED_PRETAX
                ):
                    diagnostic_codes.append(RAW_SOURCE_NORMALIZED_PRETAX)
                if (
                    metric_name == "dividends_paid"
                    and normalized_source == RAW_SOURCE_NORMALIZED_DIVIDEND_ZERO
                    and self._aligned_sec_has_row(aligned)
                ):
                    diagnostic_codes.append(RAW_SOURCE_NORMALIZED_DIVIDEND_ZERO)
                if (
                    metric_name == "intangibles"
                    and normalized_source == RAW_SOURCE_NORMALIZED_INTANGIBLES
                ):
                    diagnostic_codes.append(RAW_SOURCE_NORMALIZED_INTANGIBLES)
                if (
                    metric_name == "gross_profit"
                    and normalized_source == RAW_SOURCE_NORMALIZED_GROSS_PROFIT
                ):
                    diagnostic_codes.append(RAW_SOURCE_NORMALIZED_GROSS_PROFIT)
                if (
                    metric_name == "aoci"
                    and normalized_source == RAW_SOURCE_NORMALIZED_AOCI_DIRECT
                ):
                    diagnostic_codes.append(RAW_SOURCE_NORMALIZED_AOCI_DIRECT)
                    diagnostic_codes.append("aoci_direct_total")
                if (
                    metric_name == "aoci"
                    and normalized_source == RAW_SOURCE_NORMALIZED_AOCI
                ):
                    diagnostic_codes.append(RAW_SOURCE_NORMALIZED_AOCI)
                    diagnostic_codes.append("aoci_component_sum")
                if metric_name == "aoci" and sign_mismatch:
                    diagnostic_codes.append("aoci_sign_mismatch")
                if (
                    metric_name == "net_income_common"
                    and normalized_source == RAW_SOURCE_NORMALIZED_NET_INCOME_COMMON
                ):
                    diagnostic_codes.append(RAW_SOURCE_NORMALIZED_NET_INCOME_COMMON)
                diagnostic_codes.extend(
                    self._semantic_diagnostic_codes(
                        mapping=mapping,
                        aligned=aligned,
                        missing_on_wrds=missing_on_wrds,
                        missing_on_sec=missing_on_sec,
                        tolerance_breach=tolerance_breach,
                        sign_mismatch=sign_mismatch,
                    )
                )

                comparison_status = self._comparison_status(
                    missing_on_wrds=missing_on_wrds,
                    missing_on_sec=missing_on_sec,
                    tolerance_breach=tolerance_breach,
                    diagnostic_codes=diagnostic_codes,
                )
                semantic_gap_class = self._semantic_gap_class(
                    metric_name=str(mapping["canonical_metric_name"]),
                    statement_type=str(mapping["statement_type"]),
                    diagnostic_codes=diagnostic_codes,
                    tolerance_breach=tolerance_breach,
                )
                mismatch_class = self._mismatch_class(
                    mapping=mapping,
                    comparison_status=comparison_status,
                    diagnostic_codes=diagnostic_codes,
                    semantic_gap_class=semantic_gap_class,
                    time_regime=time_regime,
                    supported_in_time_regime=bool(effective_policy["supported_in_time_regime"]),
                )
                notes = self._build_notes(
                    aligned,
                    mapping,
                    diagnostic_codes,
                    time_regime=time_regime,
                    effective_policy=effective_policy,
                )
                ticker = str(aligned.get("ticker") or "").upper() or None
                fiscal_year = self._coerce_int(aligned.get("fiscal_year"))
                fiscal_quarter = self._coerce_int(aligned.get("fiscal_quarter"))
                wrds_period_end = pd.to_datetime(aligned.get("wrds__period_end"), errors="coerce")
                sec_period_end = pd.to_datetime(aligned.get("sec__period_end"), errors="coerce")
                payload = {
                    "result_id": self._result_id(
                        comparison_run_id=comparison_run_id,
                        ticker=ticker,
                        statement_type=str(mapping["statement_type"]),
                        metric_name=str(mapping["canonical_metric_name"]),
                        fiscal_year=fiscal_year,
                        fiscal_quarter=fiscal_quarter,
                    ),
                    "ticker": ticker,
                    "market": market,
                    "gvkey": self._coerce_str(aligned.get("wrds__gvkey")),
                    "cik": self._coerce_str(aligned.get("wrds__cik")) or self._coerce_str(aligned.get("sec__cik")),
                    "period_end": period_end.date() if pd.notna(period_end) else None,
                    "wrds_period_end": wrds_period_end.date() if pd.notna(wrds_period_end) else None,
                    "sec_period_end": sec_period_end.date() if pd.notna(sec_period_end) else None,
                    "period_end_delta_days": self._coerce_int(delta_days),
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": fiscal_quarter,
                    "statement_type": str(mapping["statement_type"]),
                    "metric_name": str(mapping["canonical_metric_name"]),
                    "compare_mode": compare_mode,
                    "value_mode_used": value_mode_used,
                    "metric_class": str(mapping.get("metric_class") or METRIC_CLASS_DIRECT),
                    "scope_rule": str(mapping.get("scope_rule") or "generic_corporate"),
                    "supported_in_time_regime": bool(effective_policy["supported_in_time_regime"]),
                    "effective_tolerance_value": float(effective_policy["tolerance_value"]),
                    "wrds_value": wrds_value,
                    "reported_sec_value": reported_sec_value,
                    "normalized_sec_value": normalized_sec_value,
                    "sec_value": sec_value,
                    "abs_diff": abs_diff,
                    "pct_diff": pct_diff,
                    "missing_on_wrds": missing_on_wrds,
                    "missing_on_sec": missing_on_sec,
                    "sign_mismatch": sign_mismatch,
                    "tolerance_breach": tolerance_breach,
                    "sec_filing_date": self._to_date(aligned.get("sec__filing_date")),
                    "sec_available_date": self._to_date(aligned.get("sec__available_date")),
                    "comparison_status": comparison_status,
                    "mismatch_class": mismatch_class,
                    "semantic_gap_class": semantic_gap_class,
                    "time_regime": time_regime,
                    "reference_db_path": str(self.wrds_db_path or self.db.db_path),
                    "reference_source_kind": "wrds_duckdb_lake" if self.wrds_db_path is not None else "local_wrds_tables",
                    "identifier_linkage_basis": "ticker_market_period_alignment",
                    "identifier_gap": bool("identifier_gap" in diagnostic_codes or not aligned.get("wrds__ticker")),
                    "pit_mismatch": bool(
                        "period_end_delta_exceeds_window" in diagnostic_codes
                        or "sequence_alignment_gap" in diagnostic_codes
                        or "sec_fiscal_anchor_shifted" in diagnostic_codes
                    ),
                    "diagnostic_code": "|".join(diagnostic_codes) if diagnostic_codes else None,
                    "notes": notes,
                    "wrds_row_count": wrds_row_count,
                    "sec_row_count": sec_row_count,
                    "comparison_run_id": comparison_run_id,
                    "created_at": created_at,
                }
                rows.append(payload)
        return rows

    def _latest_comparison_run_id(self) -> str | None:
        row = self.db.fetch_one(
            """
            SELECT comparison_run_id
            FROM validation_sec_vs_wrds_financials
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        return str(row[0]) if row and row[0] else None

    @staticmethod
    def _derive_sec_fiscal_fields(
        term_series: pd.Series | None,
        period_end_series: pd.Series | None,
        statement_type: str,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        term = pd.Series(term_series, copy=False) if term_series is not None else pd.Series(dtype="object")
        period_end = pd.to_datetime(pd.Series(period_end_series, copy=False), errors="coerce")
        extracted = term.astype("string").str.extract(r"(?P<fiscal_year>\d{4})Q(?P<fiscal_quarter>[1-4])")
        fiscal_year = pd.to_numeric(extracted.get("fiscal_year"), errors="coerce").astype("Int64")
        fiscal_quarter = pd.to_numeric(extracted.get("fiscal_quarter"), errors="coerce").astype("Int64")
        fallback_mask = fiscal_year.isna()
        if fallback_mask.any():
            fiscal_year = fiscal_year.where(~fallback_mask, period_end.dt.year.astype("Int64"))
            if statement_type == "quarterly":
                fiscal_quarter = fiscal_quarter.where(~fallback_mask, period_end.dt.quarter.astype("Int64"))
        return fiscal_year, fiscal_quarter, fallback_mask.fillna(False)

    @staticmethod
    def _quarter_like_spacing(dates: list[pd.Timestamp]) -> bool:
        if len(dates) <= 1:
            return True
        idx = pd.DatetimeIndex(sorted(pd.Timestamp(ts).normalize() for ts in dates))
        deltas = pd.Series(idx).diff().dt.days.dropna()
        if deltas.empty:
            return True
        return bool(deltas.between(70, 130, inclusive="both").all())

    @classmethod
    def _select_cycle_dates(
        cls,
        dates: list[pd.Timestamp],
        *,
        annual_end: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        annual_end = pd.Timestamp(annual_end).normalize()
        ordered = sorted(pd.Timestamp(ts).normalize() for ts in dates if pd.notna(ts))
        if annual_end not in ordered:
            return []
        selected: list[pd.Timestamp] = [annual_end]
        prior_dates = [ts for ts in ordered if ts < annual_end]
        while prior_dates and len(selected) < 4:
            candidate = prior_dates.pop()
            trial = [candidate, *selected]
            if cls._quarter_like_spacing(trial):
                selected = trial
        if not cls._quarter_like_spacing(selected):
            return []
        return selected

    @classmethod
    def _select_forward_cycle_dates(cls, dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
        ordered = sorted(pd.Timestamp(ts).normalize() for ts in dates if pd.notna(ts))
        selected: list[pd.Timestamp] = []
        for ts in ordered:
            trial = [*selected, ts]
            if cls._quarter_like_spacing(trial):
                selected = trial
            elif selected:
                break
            if len(selected) >= 3:
                break
        return selected

    @staticmethod
    def _annual_anchor_fiscal_year(annual_end: pd.Timestamp) -> int | None:
        ts = pd.Timestamp(annual_end) if pd.notna(annual_end) else pd.NaT
        if pd.isna(ts):
            return None
        ts = ts.normalize()
        # WRDS quarterly fiscal years use the prior calendar year for Jan-May
        # year-ends, plus early-June spillovers from 52/53-week May year-ends.
        # True June 30 year-ends (for example MSFT) keep the same fiscal year.
        prior_year_anchor = ts.month <= 5 or (ts.month == 6 and ts.day <= 7)
        return int(ts.year - 1 if prior_year_anchor else ts.year)

    def _refine_sec_fiscal_fields(self, frame: pd.DataFrame, *, statement_type: str) -> pd.DataFrame:
        if frame.empty or statement_type != "quarterly":
            out = frame.copy()
            out["sec_fiscal_anchor_based"] = False
            out["sec_fiscal_anchor_shifted"] = False
            return out

        out = frame.copy()
        out["sec_fiscal_anchor_based"] = False
        out["sec_fiscal_anchor_shifted"] = False
        out["ticker"] = out["ticker"].astype(str).str.upper()
        out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce").dt.normalize()

        latest_period_rows = out.sort_values(
            ["ticker", "period_end", "available_date", "filing_date", "accepted_at"],
            ascending=[True, True, True, True, True],
            na_position="last",
        ).drop_duplicates(subset=["ticker", "period_end"], keep="last")

        assignments: dict[tuple[str, date], tuple[int, int]] = {}
        for ticker, ticker_rows in latest_period_rows.groupby("ticker", dropna=False, sort=False):
            if not isinstance(ticker, str) or not ticker.strip():
                continue
            period_ends = [
                pd.Timestamp(ts).normalize()
                for ts in pd.DatetimeIndex(ticker_rows["period_end"]).dropna().sort_values().unique()
            ]
            if not period_ends:
                continue
            annual_ends = [
                pd.Timestamp(ts).normalize()
                for ts in pd.DatetimeIndex(
                    ticker_rows.loc[ticker_rows["form_type"].astype("string").isin(ANNUAL_SEC_FORMS), "period_end"]
                )
                .dropna()
                .sort_values()
                .unique()
            ]
            if not annual_ends:
                continue

            prev_annual_end: pd.Timestamp | None = None
            for annual_end in annual_ends:
                anchor_fiscal_year = self._annual_anchor_fiscal_year(annual_end)
                if anchor_fiscal_year is None:
                    prev_annual_end = annual_end
                    continue
                cycle_pool = [
                    ts
                    for ts in period_ends
                    if ts <= annual_end
                    and ts >= annual_end - pd.Timedelta(days=SEC_FISCAL_CYCLE_MAX_DAYS)
                    and (prev_annual_end is None or ts > prev_annual_end)
                ]
                cycle_dates = self._select_cycle_dates(cycle_pool, annual_end=annual_end)
                if cycle_dates:
                    start_quarter = 5 - len(cycle_dates)
                    for quarter_number, period_end in enumerate(cycle_dates, start=start_quarter):
                        assignments[(ticker, period_end.date())] = (anchor_fiscal_year, quarter_number)
                prev_annual_end = annual_end

            latest_annual_end = annual_ends[-1]
            latest_anchor_fiscal_year = self._annual_anchor_fiscal_year(latest_annual_end)
            if latest_anchor_fiscal_year is None:
                continue
            forward_pool = [
                ts
                for ts in period_ends
                if ts > latest_annual_end and ts <= latest_annual_end + pd.Timedelta(days=SEC_FISCAL_CYCLE_MAX_DAYS)
            ]
            forward_dates = self._select_forward_cycle_dates(forward_pool)
            for quarter_number, period_end in enumerate(forward_dates, start=1):
                assignments[(ticker, period_end.date())] = (latest_anchor_fiscal_year + 1, quarter_number)

        if not assignments:
            return out

        original_fiscal_year = pd.to_numeric(out.get("fiscal_year"), errors="coerce").astype("Int64")
        original_fiscal_quarter = pd.to_numeric(out.get("fiscal_quarter"), errors="coerce").astype("Int64")
        assigned_years: list[int | pd._libs.missing.NAType] = []
        assigned_quarters: list[int | pd._libs.missing.NAType] = []
        anchor_based: list[bool] = []
        anchor_shifted: list[bool] = []

        for _, row in out.iterrows():
            ticker = str(row.get("ticker") or "").upper().strip()
            period_end = pd.to_datetime(row.get("period_end"), errors="coerce")
            assigned = assignments.get((ticker, period_end.date())) if ticker and pd.notna(period_end) else None
            if assigned is None:
                assigned_years.append(pd.NA)
                assigned_quarters.append(pd.NA)
                anchor_based.append(False)
                anchor_shifted.append(False)
                continue
            assigned_year, assigned_quarter = assigned
            row_fiscal_year = self._coerce_int(row.get("fiscal_year"))
            row_fiscal_quarter = self._coerce_int(row.get("fiscal_quarter"))
            shifted = row_fiscal_year != assigned_year or row_fiscal_quarter != assigned_quarter
            assigned_years.append(assigned_year)
            assigned_quarters.append(assigned_quarter)
            anchor_based.append(True)
            anchor_shifted.append(bool(shifted))

        assigned_year_series = pd.Series(assigned_years, index=out.index, dtype="Int64")
        assigned_quarter_series = pd.Series(assigned_quarters, index=out.index, dtype="Int64")
        out["fiscal_year"] = assigned_year_series.where(assigned_year_series.notna(), original_fiscal_year)
        out["fiscal_quarter"] = assigned_quarter_series.where(assigned_quarter_series.notna(), original_fiscal_quarter)
        out["sec_fiscal_anchor_based"] = anchor_based
        out["sec_fiscal_anchor_shifted"] = anchor_shifted
        return out

    @staticmethod
    def _alignment_key(statement_type: str, row: pd.Series) -> str | None:
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            return None
        fiscal_year = row.get("fiscal_year")
        fiscal_quarter = row.get("fiscal_quarter")
        if pd.notna(fiscal_year):
            if statement_type == "quarterly" and pd.notna(fiscal_quarter):
                return f"{ticker}|{statement_type}|FY{int(fiscal_year)}Q{int(fiscal_quarter)}"
            return f"{ticker}|{statement_type}|FY{int(fiscal_year)}"
        period_end = pd.to_datetime(row.get("period_end"), errors="coerce")
        if pd.notna(period_end):
            return f"{ticker}|{statement_type}|PE{period_end.date().isoformat()}"
        return None

    @staticmethod
    def _alignment_basis(statement_type: str, row: pd.Series) -> str:
        if pd.notna(row.get("fiscal_year")) and (statement_type == "annual" or pd.notna(row.get("fiscal_quarter"))):
            return "fiscal_period"
        return "period_end_fallback"

    @staticmethod
    def _is_amendment_chain_only(form_series: pd.Series) -> bool:
        forms = pd.Series(form_series, copy=False).dropna().astype(str).str.upper().str.strip()
        if forms.empty or len(forms) <= 1:
            return False
        base_forms = forms.str.replace("/A", "", regex=False)
        return int(base_forms.nunique(dropna=True)) == 1

    def _lookup_raw_compare_value(self, aligned: pd.Series, fact_name: str) -> tuple[float | None, str | None]:
        if not fact_name:
            return None, None
        ticker = str(aligned.get("ticker") or aligned.get("sec__ticker") or "").upper().strip()
        period_end = self._to_date(aligned.get("period_end"))
        statement_type = str(aligned.get("statement_type") or "").strip().lower()
        if not ticker or not period_end or not statement_type:
            return None, None
        candidate_dates: list[date | None] = [
            period_end,
            self._to_date(aligned.get("sec__period_end")),
            self._to_date(aligned.get("wrds__period_end")),
        ]
        seen_dates: set[date | None] = set()
        for candidate_date in candidate_dates:
            if candidate_date in seen_dates or candidate_date is None:
                continue
            seen_dates.add(candidate_date)
            raw_row = self._raw_compare_lookup.get((statement_type, ticker, candidate_date, fact_name))
            if raw_row:
                return self._coerce_float(raw_row.get("value")), self._coerce_str(raw_row.get("source_code"))
        raw_row = None
        nearest_delta: int | None = None
        for (lookup_statement_type, lookup_ticker, lookup_period_end, lookup_fact_name), candidate in self._raw_compare_lookup.items():
            if lookup_statement_type != statement_type or lookup_ticker != ticker or lookup_fact_name != fact_name:
                continue
            if lookup_period_end is None:
                continue
            delta_days = abs((lookup_period_end - period_end).days)
            if delta_days > PERIOD_END_ALIGNMENT_TOLERANCE_DAYS:
                continue
            if nearest_delta is None or delta_days < nearest_delta:
                raw_row = candidate
                nearest_delta = delta_days
        if not raw_row:
            return None, None
        return self._coerce_float(raw_row.get("value")), self._coerce_str(raw_row.get("source_code"))

    def _extract_reported_sec_value(self, aligned: pd.Series, mapping: pd.Series) -> tuple[float | None, str | None]:
        column_name = str(mapping.get("sec_column_name") or "").strip()
        if not column_name:
            return None, None
        reported_annual = self._coerce_float(aligned.get(f"sec__reported__{column_name}"))
        if reported_annual is not None:
            return reported_annual, "reported_anchor"
        direct_value = self._coerce_float(aligned.get(f"sec__{column_name}"))
        if direct_value is not None:
            return direct_value, "reported_base"
        raw_value, raw_source = self._lookup_raw_compare_value(aligned, column_name)
        if raw_value is not None:
            return raw_value, raw_source
        reported_policy = str(mapping.get("reported_value_policy") or "").strip()
        extra_name = str(mapping.get("sec_extra_column_name") or "").strip()
        if extra_name and "or_extra" in reported_policy:
            extra_value, extra_source = self._aligned_sec_numeric(aligned, extra_name)
            if extra_value is not None:
                return extra_value, RAW_SOURCE_EXTRA_FALLBACK if extra_source == RAW_SOURCE_EXTRA_FALLBACK else extra_source
        return None, None

    def _aligned_sec_numeric(self, aligned: pd.Series, column_name: str) -> tuple[float | None, str | None]:
        if not column_name:
            return None, None
        for key in (
            f"sec__normalized__{column_name}",
            f"sec__reported__{column_name}",
            f"sec__{column_name}",
            f"sec__extra__{column_name}",
        ):
            value = self._coerce_float(aligned.get(key))
            if value is not None:
                return value, RAW_SOURCE_EXTRA_FALLBACK if key.startswith("sec__extra__") else "sec_row"
        return self._lookup_raw_compare_value(aligned, column_name)

    @staticmethod
    def _aligned_sec_has_row(aligned: pd.Series) -> bool:
        form_type = str(aligned.get("sec__form_type") or "").strip()
        if form_type:
            return True
        for key in ("sec__filing_date", "sec__accepted_at", "sec__available_date", "sec__period_end"):
            if pd.notna(aligned.get(key)):
                return True
        return False

    def _extract_normalized_sec_value(self, aligned: pd.Series, mapping: pd.Series) -> tuple[float | None, str | None]:
        metric_name = str(mapping.get("canonical_metric_name") or "").strip()
        column_name = str(mapping.get("sec_column_name") or "").strip()
        direct_value, direct_source = self._aligned_sec_numeric(aligned, column_name)

        if metric_name == "gross_profit":
            if direct_value is not None:
                return direct_value, direct_source
            revenue, _ = self._aligned_sec_numeric(aligned, "Revenue")
            cogs, _ = self._aligned_sec_numeric(aligned, "COGS")
            if revenue is not None and cogs is not None:
                return revenue - cogs, RAW_SOURCE_NORMALIZED_GROSS_PROFIT
            return None, None

        if metric_name == "pretax_income":
            if direct_value is not None:
                return direct_value, direct_source
            net_income, _ = self._aligned_sec_numeric(aligned, "Net Income")
            tax, _ = self._aligned_sec_numeric(aligned, "Tax")
            if net_income is not None and tax is not None:
                return net_income + tax, RAW_SOURCE_NORMALIZED_PRETAX
            return None, None

        if metric_name == "intangibles":
            goodwill, _ = self._aligned_sec_numeric(aligned, "Goodwill")
            if direct_value is not None and goodwill is not None:
                return direct_value + goodwill, RAW_SOURCE_NORMALIZED_INTANGIBLES
            return direct_value, direct_source

        if metric_name == "aoci":
            if direct_value is not None:
                return direct_value, RAW_SOURCE_NORMALIZED_AOCI_DIRECT
            extra_value = self._coerce_float(aligned.get("sec__extra__aoci"))
            if extra_value is not None:
                return extra_value, RAW_SOURCE_NORMALIZED_AOCI
            return None, None

        if metric_name == "dividends_paid":
            if direct_value is not None:
                return direct_value, direct_source
            if self._aligned_sec_has_row(aligned):
                cashflow_available = any(
                    self._aligned_sec_numeric(aligned, column)[0] is not None
                    for column in ("Financing Cash Flow", "Operating Cash Flow", "Net Income")
                )
                if cashflow_available:
                    return 0.0, RAW_SOURCE_NORMALIZED_DIVIDEND_ZERO
            return None, None

        if metric_name == "net_income_common":
            if direct_value is not None:
                return direct_value, direct_source
            extra_value = self._coerce_float(aligned.get("sec__extra__owner_net_income"))
            if extra_value is not None:
                return extra_value, RAW_SOURCE_EXTRA_FALLBACK
            net_income, _ = self._aligned_sec_numeric(aligned, "Net Income")
            if net_income is not None:
                return net_income, RAW_SOURCE_NORMALIZED_NET_INCOME_COMMON
            return None, None

        if direct_value is not None:
            return direct_value, direct_source

        extra_name = mapping.get("sec_extra_column_name")
        if extra_name and pd.notna(extra_name):
            return self._aligned_sec_numeric(aligned, str(extra_name))
        return None, None

    @staticmethod
    def _resolve_value_mode(mapping: pd.Series, compare_mode: str, *, time_regime: str) -> str:
        requested = str(compare_mode or COMPARE_MODE_DEFAULT).strip().lower()
        if requested == COMPARE_MODE_REPORTED:
            return VALUE_MODE_REPORTED
        if requested == COMPARE_MODE_NORMALIZED:
            return VALUE_MODE_NORMALIZED
        if time_regime == TIME_REGIME_PRE_2012:
            default_mode = str(mapping.get("pre_2012_value_mode_default") or VALUE_MODE_REPORTED).strip()
            return default_mode if default_mode in (VALUE_MODE_REPORTED, VALUE_MODE_NORMALIZED) else VALUE_MODE_REPORTED
        default_mode = str(mapping.get("value_mode_default") or VALUE_MODE_NORMALIZED).strip()
        return default_mode if default_mode in (VALUE_MODE_REPORTED, VALUE_MODE_NORMALIZED) else VALUE_MODE_NORMALIZED

    @staticmethod
    def _effective_policy(mapping: pd.Series, *, time_regime: str) -> dict[str, Any]:
        if time_regime == TIME_REGIME_PRE_2012:
            tolerance_override = SECVsWRDSFinancialValidationService._coerce_float(mapping.get("pre_2012_tolerance_override"))
            base_tolerance = SECVsWRDSFinancialValidationService._coerce_float(mapping.get("tolerance_value"))
            tolerance_value = tolerance_override if tolerance_override is not None else base_tolerance
            supported = SECVsWRDSFinancialValidationService._coerce_bool(mapping.get("pre_2012_supported_flag"))
            return {
                "supported_in_time_regime": bool(supported) if supported is not None else False,
                "tolerance_value": float(tolerance_value) if tolerance_value is not None else 0.0,
                "known_gap_policy": str(mapping.get("pre_2012_known_gap_policy") or "").strip(),
            }
        base_tolerance = SECVsWRDSFinancialValidationService._coerce_float(mapping.get("tolerance_value"))
        return {
            "supported_in_time_regime": True,
            "tolerance_value": float(base_tolerance) if base_tolerance is not None else 0.0,
            "known_gap_policy": str(mapping.get("mismatch_policy") or "").strip(),
        }

    @staticmethod
    def _annual_aggregate_rules(mappings: pd.DataFrame) -> dict[str, str]:
        rules: dict[str, str] = {}
        for _, mapping in mappings.iterrows():
            metric_name = str(mapping.get("canonical_metric_name") or "")
            aggregate = "sum" if metric_name in ANNUAL_AGGREGATE_FLOW_METRICS else "last"
            sec_column_name = str(mapping.get("sec_column_name") or "").strip()
            if sec_column_name:
                if sec_column_name in ANNUAL_AGGREGATE_SEC_COLUMNS:
                    aggregate = "sum"
                rules.setdefault(sec_column_name, aggregate)
            sec_extra_name = str(mapping.get("sec_extra_column_name") or "").strip()
            if sec_extra_name:
                if sec_extra_name in SEC_FINANCIALS_EXTRA_COLUMNS:
                    rules.setdefault(f"extra__{sec_extra_name}", aggregate)
                else:
                    rules.setdefault(sec_extra_name, aggregate)
        return rules

    def _semantic_diagnostic_codes(
        self,
        *,
        mapping: pd.Series,
        aligned: pd.Series,
        missing_on_wrds: bool,
        missing_on_sec: bool,
        tolerance_breach: bool,
        sign_mismatch: bool,
    ) -> list[str]:
        codes: list[str] = []
        metric_name = str(mapping.get("canonical_metric_name") or "")
        statement_type = str(mapping.get("statement_type") or "")
        sec_form_type = str(aligned.get("sec__form_type") or "").upper()
        if statement_type == "quarterly" and sec_form_type in ANNUAL_SEC_FORMS and not missing_on_sec:
            codes.append("quarterly_q4_from_10k")
        if bool(aligned.get("sequence_alignment_fallback")):
            codes.append("sequence_alignment_gap")
        if missing_on_wrds or missing_on_sec or not tolerance_breach:
            if metric_name == "goodwill" and missing_on_sec and self._aligned_sec_has_row(aligned):
                codes.append("goodwill_parser_gap")
            if metric_name == "common_stock" and missing_on_sec:
                apic_value = self._coerce_float(aligned.get("sec__APIC"))
                apic_extra = self._coerce_float(aligned.get("sec__extra__additional_paid_in_capital"))
                if apic_value is not None or apic_extra is not None:
                    codes.append("common_stock_combined_equity_line")
            if metric_name == "apic" and missing_on_sec:
                common_stock_value = self._coerce_float(aligned.get("sec__Common Stock"))
                common_stock_extra = self._coerce_float(aligned.get("sec__extra__common_stock"))
                if common_stock_value is not None or common_stock_extra is not None:
                    codes.append("apic_combined_equity_line")
            return codes
        if metric_name == "cash":
            codes.append("candidate_cash_scope_gap")
        elif metric_name == "gross_profit":
            revenue, _ = self._aligned_sec_numeric(aligned, "Revenue")
            cogs, _ = self._aligned_sec_numeric(aligned, "COGS")
            gross_profit, _ = self._aligned_sec_numeric(aligned, "Gross Profit")
            if revenue is not None and cogs is not None:
                derived_gross = revenue - cogs
                if gross_profit is None:
                    codes.append("gross_profit_revenue_minus_cogs_only")
                elif abs(gross_profit - derived_gross) > max(abs(gross_profit), abs(derived_gross), 1.0) * 0.02:
                    codes.append("gross_profit_direct_vs_revenue_minus_cogs_diverge")
            codes.append("candidate_gross_profit_semantic_gap")
        elif metric_name == "repurchases":
            codes.append("candidate_repurchases_semantic_gap")
        elif metric_name == "receivables":
            codes.append("candidate_receivables_scope_gap")
        elif metric_name == "deferred_revenue":
            codes.append("candidate_deferred_revenue_scope_gap")
        elif metric_name == "intangibles":
            codes.append("candidate_intangibles_scope_gap")
        elif metric_name == "current_fin_assets":
            codes.append("candidate_current_fin_assets_scope_gap")
        elif metric_name == "non_current_fin_assets":
            codes.append("candidate_non_current_fin_assets_scope_gap")
        elif metric_name == "current_fin_liabilities":
            fin_liab_value, _ = self._aligned_sec_numeric(aligned, "Current Fin Liabilities")
            debt_short_value, _ = self._aligned_sec_numeric(aligned, "Debt Short")
            if (
                fin_liab_value is not None
                and debt_short_value is not None
                and abs(fin_liab_value - debt_short_value) <= max(abs(fin_liab_value), abs(debt_short_value), 1.0) * 0.001
            ):
                codes.append("current_fin_liabilities_debt_proxy")
            codes.append("candidate_current_fin_liabilities_scope_gap")
        elif metric_name == "non_current_fin_liabilities":
            fin_liab_value, _ = self._aligned_sec_numeric(aligned, "Non Current Fin Liabilities")
            debt_long_value, _ = self._aligned_sec_numeric(aligned, "Debt Long")
            if (
                fin_liab_value is not None
                and debt_long_value is not None
                and abs(fin_liab_value - debt_long_value) <= max(abs(fin_liab_value), abs(debt_long_value), 1.0) * 0.001
            ):
                codes.append("non_current_fin_liabilities_debt_proxy")
            codes.append("candidate_non_current_fin_liabilities_scope_gap")
        elif metric_name in {"debt_short", "debt_long"}:
            codes.append("candidate_debt_maturity_gap")
        elif metric_name == "cogs":
            codes.append("candidate_cogs_semantic_gap")
        elif metric_name == "interest":
            debt_short, _ = self._aligned_sec_numeric(aligned, "Debt Short")
            debt_long, _ = self._aligned_sec_numeric(aligned, "Debt Long")
            if not sign_mismatch and (debt_short in (None, 0.0)) and (debt_long in (None, 0.0)):
                codes.append("candidate_interest_scope_gap")
        elif metric_name == "operating_income":
            codes.append(
                "candidate_operating_income_classification_gap"
                if sign_mismatch
                else "candidate_operating_income_semantic_gap"
            )
        elif metric_name == "goodwill":
            codes.append("goodwill_parser_gap")
        elif metric_name == "common_stock":
            apic_value = self._coerce_float(aligned.get("sec__APIC"))
            apic_extra = self._coerce_float(aligned.get("sec__extra__additional_paid_in_capital"))
            if apic_value is not None or apic_extra is not None:
                codes.append("common_stock_combined_equity_line")
            else:
                codes.append("common_stock_scope_gap")
        elif metric_name == "apic":
            common_stock_value = self._coerce_float(aligned.get("sec__Common Stock"))
            common_stock_extra = self._coerce_float(aligned.get("sec__extra__common_stock"))
            if common_stock_value is not None or common_stock_extra is not None:
                codes.append("apic_combined_equity_line")
            codes.append("candidate_apic_scope_gap")
        elif metric_name == "r_and_d":
            codes.append("candidate_r_and_d_scope_gap")
        return codes

    @staticmethod
    def _semantic_gap_class(
        *,
        metric_name: str,
        statement_type: str,
        diagnostic_codes: list[str],
        tolerance_breach: bool,
    ) -> str | None:
        code_set = set(diagnostic_codes)
        if "candidate_cash_scope_gap" in code_set:
            return "cash_scope"
        if "candidate_gross_profit_semantic_gap" in code_set:
            return "gross_profit_semantics"
        if "candidate_repurchases_semantic_gap" in code_set:
            return "repurchases_semantics"
        if "candidate_receivables_scope_gap" in code_set:
            return "receivables_scope"
        if "candidate_deferred_revenue_scope_gap" in code_set:
            return "deferred_revenue_scope"
        if "candidate_intangibles_scope_gap" in code_set:
            return "intangibles_scope"
        if "candidate_current_fin_assets_scope_gap" in code_set:
            return "current_fin_assets_scope"
        if "candidate_non_current_fin_assets_scope_gap" in code_set:
            return "non_current_fin_assets_scope"
        if "candidate_current_fin_liabilities_scope_gap" in code_set:
            if "current_fin_liabilities_debt_proxy" in code_set:
                return "current_fin_liabilities_debt_proxy"
            return "current_fin_liabilities_scope"
        if "candidate_non_current_fin_liabilities_scope_gap" in code_set:
            if "non_current_fin_liabilities_debt_proxy" in code_set:
                return "non_current_fin_liabilities_debt_proxy"
            return "non_current_fin_liabilities_scope"
        if "candidate_debt_maturity_gap" in code_set:
            return "debt_maturity"
        if "candidate_cogs_semantic_gap" in code_set:
            return "cogs_semantics"
        if "candidate_interest_scope_gap" in code_set:
            return "interest_scope"
        if "common_stock_combined_equity_line" in code_set:
            return "common_stock_combined_equity_line"
        if "common_stock_scope_gap" in code_set:
            return "common_stock_scope_gap"
        if "apic_combined_equity_line" in code_set:
            return "apic_combined_equity_line"
        if "candidate_apic_scope_gap" in code_set:
            return "apic_scope_gap"
        if "candidate_r_and_d_scope_gap" in code_set:
            return "r_and_d_scope_gap"
        if (
            "candidate_operating_income_semantic_gap" in code_set
            or "candidate_operating_income_classification_gap" in code_set
        ):
            return "operating_income_semantics"
        if metric_name == "eps" and statement_type == "annual" and tolerance_breach:
            return "annual_eps_residual"
        return None

    @staticmethod
    def _mismatch_class(
        *,
        mapping: pd.Series,
        comparison_status: str,
        diagnostic_codes: list[str],
        semantic_gap_class: str | None,
        time_regime: str,
        supported_in_time_regime: bool,
    ) -> str:
        if comparison_status in {"match", "matched_with_ambiguity"}:
            return MISMATCH_CLASS_NONE
        if semantic_gap_class is not None:
            return MISMATCH_CLASS_SOURCE_SEMANTICS_GAP
        if time_regime == TIME_REGIME_PRE_2012 and (
            not supported_in_time_regime
            or "pre_2012_reduced_support" in diagnostic_codes
            or "pre_2012_compatibility_mode" in diagnostic_codes
        ):
            return MISMATCH_CLASS_COMPATIBILITY_GAP
        if "sequence_alignment_gap" in diagnostic_codes and comparison_status == "matched_with_ambiguity":
            return MISMATCH_CLASS_NORMALIZATION_GAP
        if comparison_status in {"missing_sec", "period_alignment_issue"}:
            return MISMATCH_CLASS_PARSER_BUG
        if comparison_status in {"missing_wrds", "missing_both"}:
            return "reference_gap"
        if any(
            code in diagnostic_codes
            for code in ("duplicate_sec_row", "sec_term_missing_fallback", "goodwill_parser_gap")
        ):
            return MISMATCH_CLASS_PARSER_BUG
        metric_class = str(mapping.get("metric_class") or METRIC_CLASS_DIRECT)
        if metric_class in (METRIC_CLASS_RECONSTRUCTED, METRIC_CLASS_SCOPE_SENSITIVE):
            return MISMATCH_CLASS_NORMALIZATION_GAP
        return MISMATCH_CLASS_PARSER_BUG

    @staticmethod
    def _time_regime(period_end: pd.Timestamp) -> str:
        if pd.notna(period_end) and period_end.date() < SEC_VALIDATION_START_DATE:
            return TIME_REGIME_PRE_2012
        return TIME_REGIME_POST_2012

    @staticmethod
    def _apply_sign_rule(wrds_value: float | None, sec_value: float | None, sign_rule: str) -> tuple[float | None, float | None]:
        if sign_rule == "invert_sec" and sec_value is not None:
            return wrds_value, -sec_value
        if sign_rule == "absolute_value":
            return (abs(wrds_value) if wrds_value is not None else None, abs(sec_value) if sec_value is not None else None)
        return wrds_value, sec_value

    @staticmethod
    def _apply_value_scale(
        wrds_value: float | None,
        sec_value: float | None,
        value_scale_rule: str,
    ) -> tuple[float | None, float | None]:
        if value_scale_rule == "wrds_millions_to_units":
            return (
                wrds_value * 1_000_000.0 if wrds_value is not None else None,
                sec_value,
            )
        return wrds_value, sec_value

    @staticmethod
    def _pct_diff(abs_diff: float | None, wrds_value: float | None) -> float | None:
        if abs_diff is None or wrds_value is None:
            return None
        if wrds_value == 0:
            return 0.0 if abs_diff == 0 else None
        return abs_diff / abs(wrds_value)

    @staticmethod
    def _tolerance_breach(
        *,
        abs_diff: float | None,
        pct_diff: float | None,
        wrds_value: float | None,
        sec_value: float | None,
        tolerance_type: str,
        tolerance_value: float,
    ) -> bool:
        if wrds_value is None or sec_value is None:
            return False
        if tolerance_type == "absolute":
            return bool(abs_diff is not None and abs_diff > tolerance_value)
        if tolerance_type == "relative":
            if pct_diff is None:
                return abs(sec_value - wrds_value) > tolerance_value
            return pct_diff > tolerance_value
        return bool(abs_diff is not None and abs_diff > tolerance_value)

    @staticmethod
    def _comparison_status(
        *,
        missing_on_wrds: bool,
        missing_on_sec: bool,
        tolerance_breach: bool,
        diagnostic_codes: list[str],
    ) -> str:
        if missing_on_wrds and missing_on_sec:
            return "missing_both"
        if missing_on_wrds:
            return "missing_wrds"
        if missing_on_sec:
            return "missing_sec"
        if "period_end_delta_exceeds_window" in diagnostic_codes:
            return "period_alignment_issue"
        if tolerance_breach:
            return "tolerance_breach"
        if any(
            code in diagnostic_codes
            for code in (
                "duplicate_wrds_row",
                "duplicate_sec_row",
                "sec_term_missing_fallback",
                "sequence_alignment_fallback",
                "fiscal_calendar_shift_alignment",
            )
        ):
            return "matched_with_ambiguity"
        return "match"

    @staticmethod
    def _build_notes(
        aligned: pd.Series,
        mapping: pd.Series,
        diagnostic_codes: list[str],
        *,
        time_regime: str,
        effective_policy: dict[str, Any],
    ) -> str | None:
        notes: list[str] = []
        mapping_note = str(mapping.get("notes") or "").strip()
        if mapping_note:
            notes.append(mapping_note)
        notes.append(f"time_regime={time_regime}")
        if time_regime == TIME_REGIME_PRE_2012:
            notes.append(f"pre_2012_supported={bool(effective_policy.get('supported_in_time_regime'))}")
            gap_policy = str(effective_policy.get("known_gap_policy") or "").strip()
            if gap_policy:
                notes.append(f"pre_2012_policy={gap_policy}")
        wrds_row_count = SECVsWRDSFinancialValidationService._coerce_int(aligned.get("wrds__wrds_row_count")) or 0
        sec_row_count = SECVsWRDSFinancialValidationService._coerce_int(aligned.get("sec__sec_row_count")) or 0
        if wrds_row_count > 1:
            notes.append(f"wrds_rows={wrds_row_count}")
        if sec_row_count > 1:
            notes.append(f"sec_rows={sec_row_count}")
        annual_candidate_count = SECVsWRDSFinancialValidationService._coerce_int(aligned.get("sec__annual_candidate_count")) or 0
        if annual_candidate_count > 1:
            notes.append(f"annual_candidates={annual_candidate_count}")
        annual_quarter_coverage = SECVsWRDSFinancialValidationService._coerce_int(aligned.get("sec__annual_quarter_coverage")) or 0
        if annual_quarter_coverage and annual_quarter_coverage != 4:
            notes.append(f"annual_quarter_coverage={annual_quarter_coverage}")
        if bool(aligned.get("sec__sec_term_fallback")):
            notes.append("sec term missing; used period_end-based fiscal fallback")
        delta_days = aligned.get("period_end_delta_days")
        if pd.notna(delta_days):
            notes.append(f"period_end_delta_days={int(delta_days)}")
        if diagnostic_codes:
            notes.append("diagnostics=" + ",".join(diagnostic_codes))
        return "; ".join(notes) if notes else None

    @staticmethod
    def _result_id(
        *,
        comparison_run_id: str,
        ticker: str | None,
        statement_type: str,
        metric_name: str,
        fiscal_year: int | None,
        fiscal_quarter: int | None,
    ) -> str:
        payload = json.dumps(
            {
                "run_id": comparison_run_id,
                "ticker": ticker,
                "statement_type": statement_type,
                "metric_name": metric_name,
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

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
    def _coerce_bool(value: Any) -> bool | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return bool(value)

    @staticmethod
    def _to_date(value: Any) -> date | None:
        ts = pd.to_datetime(value, errors="coerce")
        return ts.date() if pd.notna(ts) else None

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

    @staticmethod
    def _coalesce_prefer_left(left: pd.Series, right: pd.Series) -> pd.Series:
        left_series = pd.Series(left, copy=False)
        right_series = pd.Series(right, copy=False).reindex(left_series.index)
        return left_series.where(left_series.notna(), right_series)
