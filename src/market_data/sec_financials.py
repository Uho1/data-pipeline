from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from market_data.config import DATA_DIR
from market_data.fiscal_periods import infer_fiscal_period_meta, is_annual_form
from market_data.utils import ensure_dir, now_utc_iso, retry_call, sanitize_ticker

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_TICKERS_MF_URL = "https://www.sec.gov/files/company_tickers_mf.json"
SEC_COMPANYFACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

SEC_TERM_CACHE_DIR = DATA_DIR / "sec_term_cache"
SEC_RAW_COMPANYFACTS_DIR = SEC_TERM_CACHE_DIR / "raw_companyfacts"
SEC_RAW_SUBMISSIONS_DIR = SEC_TERM_CACHE_DIR / "raw_submissions"
SEC_TICKER_QUARTERLY_DIR = SEC_TERM_CACHE_DIR / "ticker_quarterly"
SEC_TICKER_SEGMENT_DIR = SEC_TERM_CACHE_DIR / "ticker_segments"
SEC_FILINGS_CACHE_DIR = SEC_TERM_CACHE_DIR / "filings"
SEC_TICKER_MAP_CACHE = SEC_TERM_CACHE_DIR / "company_tickers.json"
SEC_TICKER_EXCHANGE_CACHE = SEC_TERM_CACHE_DIR / "company_tickers_exchange.json"
SEC_TICKER_MF_CACHE = SEC_TERM_CACHE_DIR / "company_tickers_mf.json"

# SEC XBRL/companyfacts coverage becomes materially more standardized from mid-2013.
# Keep SEC financial/cache defaults anchored there unless a caller explicitly narrows further.
SEC_DEFAULT_START_DATE = pd.Timestamp("2013-06-01")
SEC_EXTRACTOR_VERSION = 49  # bumped: _iter_unit_records JPY/CNY fallback 제거 — 외국 ADR JPY 오저장 수정
RAW_QUARTER_MAX_DURATION_DAYS = 120
FLOW_DERIVATION_LOOKBACK_DAYS = 400
SEC_USER_AGENT_ENV = "SEC_USER_AGENT"
SEC_DEFAULT_USER_AGENT = "market-data-lake/0.1 (local-tooling@example.com)"
SEC_XBRL_PHASEIN_START = pd.Timestamp("2009-06-15")
SEC_HIGH_CONFIDENCE_START_DATE = pd.Timestamp("2012-01-01")
SEC_PRE_2012_COMPATIBILITY_END = SEC_HIGH_CONFIDENCE_START_DATE - pd.Timedelta(days=1)
SEC_TIME_REGIME_POST_2012 = "post_2012_high_confidence"
SEC_TIME_REGIME_PRE_2012 = "pre_2012_compatibility"
SEC_PRE_XBRL_HTML_PILOT_TICKERS = {"AAPL"}
SEC_AAPL_HTML_FALLBACK_END = pd.Timestamp("2013-12-31")
SEC_ARCHIVES_MIN_REQUEST_INTERVAL_SECONDS = 0.4
SEC_DATA_API_MIN_REQUEST_INTERVAL_SECONDS = 0.25
SEC_OTHER_MIN_REQUEST_INTERVAL_SECONDS = 0.35
SEC_FAST_DATA_API_MIN_REQUEST_INTERVAL_SECONDS = 0.15
SEC_JSON_RESPONSE_CACHE_MAX_ENTRIES = 256
SEC_TEXT_RESPONSE_CACHE_MAX_ENTRIES = 96

_SEC_REQUEST_LOCK = threading.Lock()
_SEC_REQUEST_LAST_AT: dict[str, float] = {}
_SEC_TICKER_REFERENCE_CACHE: dict[str, dict[str, Any]] | None = None
_SEC_RESPONSE_CACHE_LOCK = threading.Lock()
_SEC_JSON_RESPONSE_CACHE: OrderedDict[str, dict[str, Any] | list[Any]] = OrderedDict()
_SEC_TEXT_RESPONSE_CACHE: OrderedDict[str, str] = OrderedDict()
_SEC_INFLIGHT_REQUESTS: dict[tuple[str, str], threading.Event] = {}
_SEC_REQUEST_INTERVAL_OVERRIDES: dict[str, float | None] = {
    "archives": None,
    "data_api": None,
    "sec_other": None,
}


def cleanup_sec_ticker_cache(
    ticker: str,
    *,
    raw_cache_dir: Path | None = SEC_RAW_COMPANYFACTS_DIR,
    submissions_cache_dir: Path | None = SEC_RAW_SUBMISSIONS_DIR,
    filings_cache_dir: Path | None = SEC_FILINGS_CACHE_DIR,
) -> None:
    symbol = sanitize_ticker(str(ticker).strip().upper())
    if not symbol:
        return

    if raw_cache_dir is not None:
        raw_companyfacts_path = raw_cache_dir / f"{symbol}.json"
        if raw_companyfacts_path.exists():
            raw_companyfacts_path.unlink(missing_ok=True)

    if submissions_cache_dir is not None:
        submissions_root = submissions_cache_dir / f"{symbol}.json"
        history_names: list[str] = []
        if submissions_root.exists():
            try:
                payload = json.loads(submissions_root.read_text(encoding="utf-8"))
                history_names = [
                    str((entry or {}).get("name", "")).strip()
                    for entry in list((payload or {}).get("filings", {}).get("files", []) or [])
                    if str((entry or {}).get("name", "")).strip()
                ]
            except Exception:
                history_names = []
            submissions_root.unlink(missing_ok=True)
        for name in history_names:
            hist_path = submissions_cache_dir / name
            if hist_path.exists():
                hist_path.unlink(missing_ok=True)

    if filings_cache_dir is not None:
        filings_path = filings_cache_dir / symbol
        if filings_path.exists():
            shutil.rmtree(filings_path, ignore_errors=True)


def _sec_request_bucket(url: str) -> str | None:
    parsed = urlparse(str(url))
    host = str(parsed.netloc or "").lower()
    path = str(parsed.path or "")

    if host == "www.sec.gov" and path.startswith("/Archives/"):
        return "archives"
    if host == "data.sec.gov":
        return "data_api"
    if host.endswith("sec.gov"):
        return "sec_other"
    return None


def _throttle_sec_request(url: str) -> None:
    bucket = _sec_request_bucket(url)
    if bucket is None:
        return

    defaults = {
        "archives": SEC_ARCHIVES_MIN_REQUEST_INTERVAL_SECONDS,
        "data_api": SEC_DATA_API_MIN_REQUEST_INTERVAL_SECONDS,
        "sec_other": SEC_OTHER_MIN_REQUEST_INTERVAL_SECONDS,
    }
    min_interval = _SEC_REQUEST_INTERVAL_OVERRIDES.get(bucket)
    if min_interval is None:
        min_interval = defaults.get(bucket, 0.0)
    if min_interval <= 0:
        return

    while True:
        with _SEC_REQUEST_LOCK:
            now = time.monotonic()
            last = _SEC_REQUEST_LAST_AT.get(bucket)
            wait_for = 0.0 if last is None else (last + min_interval - now)
            if wait_for <= 0:
                _SEC_REQUEST_LAST_AT[bucket] = now
                return
        time.sleep(wait_for)


def _clone_sec_response_for_caller(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    return value


def _sec_response_cache_lookup(
    cache: OrderedDict[str, Any],
    *,
    url: str,
) -> Any | None:
    value = cache.get(url)
    if value is None:
        return None
    cache.move_to_end(url)
    return _clone_sec_response_for_caller(value)


def _sec_response_cache_store(
    cache: OrderedDict[str, Any],
    *,
    url: str,
    value: Any,
    max_entries: int,
) -> None:
    cache[url] = value
    cache.move_to_end(url)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _load_or_fetch_sec_response(
    *,
    kind: str,
    url: str,
    fetcher: Callable[[], Any],
    max_entries: int,
) -> Any:
    cache = _SEC_JSON_RESPONSE_CACHE if kind == "json" else _SEC_TEXT_RESPONSE_CACHE
    cache_key = (kind, url)

    while True:
        owner = False
        with _SEC_RESPONSE_CACHE_LOCK:
            cached = _sec_response_cache_lookup(cache, url=url)
            if cached is not None:
                return cached
            wait_event = _SEC_INFLIGHT_REQUESTS.get(cache_key)
            if wait_event is None:
                wait_event = threading.Event()
                _SEC_INFLIGHT_REQUESTS[cache_key] = wait_event
                owner = True

        if owner:
            break
        wait_event.wait()

    try:
        value = fetcher()
    except Exception:
        with _SEC_RESPONSE_CACHE_LOCK:
            _SEC_INFLIGHT_REQUESTS.pop(cache_key, None)
            wait_event.set()
        raise

    with _SEC_RESPONSE_CACHE_LOCK:
        _sec_response_cache_store(cache, url=url, value=value, max_entries=max_entries)
        _SEC_INFLIGHT_REQUESTS.pop(cache_key, None)
        wait_event.set()
    return _clone_sec_response_for_caller(value)


def _reset_sec_response_memory_cache() -> None:
    with _SEC_RESPONSE_CACHE_LOCK:
        _SEC_JSON_RESPONSE_CACHE.clear()
        _SEC_TEXT_RESPONSE_CACHE.clear()
        _SEC_INFLIGHT_REQUESTS.clear()


def configure_sec_request_throttle(
    *,
    archives: float | None = None,
    data_api: float | None = None,
    sec_other: float | None = None,
) -> None:
    """Override SEC request throttle intervals for the current process.

    Pass ``None`` for any bucket to fall back to the module default.
    """
    with _SEC_REQUEST_LOCK:
        _SEC_REQUEST_INTERVAL_OVERRIDES["archives"] = archives
        _SEC_REQUEST_INTERVAL_OVERRIDES["data_api"] = data_api
        _SEC_REQUEST_INTERVAL_OVERRIDES["sec_other"] = sec_other

FLOW_COLUMNS = [
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
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Capital Expenditure",
    "Dividends Paid",
    "Repurchases",
]

STOCK_COLUMNS = [
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
    "Common Stock",
    "APIC",
    "Retained Earnings",
    "AOCI",
    "Current Fin Assets",
    "Non Current Fin Assets",
    "Current Fin Liabilities",
    "Non Current Fin Liabilities",
    "Shares",
    "Diluted Shares",
    "Basic Shares",
    "Price",
    "Price_M1",
    "Price_M2",
    "Price_M3",
]

META_COLUMNS = ["name", "name_kr", "sector", "industry", "avg_volume", "Source"]
EXTRA_COLUMNS = [
    "end_date",
    "PeriodStart",
    "PeriodEnd",
    "FormType",
    "FilingDate",
    "AcceptedAt",
    "AvailableDate",
    "AvailabilityMethod",
    "diluted_eps",
    "diluted_shares",
    "basic_shares",
    "net_income_common",
    "eps_source",
    "fiscal_year",
    "fiscal_quarter",
    "fiscal_label",
]


def _collapse_quarterly_period_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate SEC quarter rows to one canonical row per PeriodEnd."""
    if frame is None or frame.empty or "PeriodEnd" not in frame.columns:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    out = frame.copy()
    out["__period_key"] = pd.to_datetime(out.get("PeriodEnd"), errors="coerce").dt.normalize()
    out = out.loc[out["__period_key"].notna()].copy()
    if out.empty:
        return out.drop(columns=["__period_key"], errors="ignore")

    meta_cols = {
        "symbol",
        "term",
        "StatementDate",
        "PeriodEnd",
        "PeriodStart",
        "FormType",
        "FilingDate",
        "AcceptedAt",
        "AvailableDate",
        "AvailabilityMethod",
        "CollectedAt",
        "RequestedStart",
        "ExtractorVersion",
        "name",
        "name_kr",
        "sector",
        "industry",
        "avg_volume",
        "Source",
        "end_date",
        "fiscal_year",
        "fiscal_quarter",
        "fiscal_label",
    }
    score_cols = [col for col in out.columns if col not in meta_cols and not col.startswith("__")]
    out["__nonnull_score"] = out[score_cols].notna().sum(axis=1) if score_cols else 0
    out["__available_sort"] = pd.to_datetime(out.get("AvailableDate"), errors="coerce")
    out["__filing_sort"] = pd.to_datetime(out.get("FilingDate"), errors="coerce")
    out["__accepted_sort"] = pd.to_datetime(out.get("AcceptedAt"), errors="coerce")

    def _annual_penalty(row: pd.Series) -> int:
        if not is_annual_form(row.get("FormType")):
            return 0
        fiscal_quarter = row.get("fiscal_quarter")
        try:
            if pd.notna(fiscal_quarter) and int(float(fiscal_quarter)) == 4:
                return 0
        except Exception:
            pass
        return 1

    out["__annual_penalty"] = out.apply(_annual_penalty, axis=1)
    out = out.sort_values(
        [
            "__period_key",
            "__annual_penalty",
            "__nonnull_score",
            "__available_sort",
            "__filing_sort",
            "__accepted_sort",
        ],
        ascending=[True, True, False, True, True, True],
        kind="stable",
    )
    out = out.drop_duplicates(subset=["__period_key"], keep="first")
    return out.drop(columns=[col for col in out.columns if col.startswith("__")], errors="ignore").reset_index(drop=True)
SEGMENT_COLUMNS = [
    "ticker",
    "market",
    "period_end",
    "period_start",
    "form_type",
    "filing_date",
    "accepted_at",
    "available_date",
    "availability_method",
    "segment_type",
    "segment_name",
    "revenue",
    "op_income",
    "source",
    "collected_at",
]

SEC_ALLOWED_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}
SEC_NT_ALLOWED_FORMS = {"NT 10-Q", "NT 10-K", "NT 10-Q/A", "NT 10-K/A"}
SEC_FILING_META_FORMS = SEC_ALLOWED_FORMS | SEC_NT_ALLOWED_FORMS

SEGMENT_FACT_COLUMNS = [
    "ticker",
    "market",
    "period_end",
    "period_start",
    "form_type",
    "filing_date",
    "accepted_at",
    "available_date",
    "availability_method",
    "segment_type",
    "segment_name",
    "metric",
    "value",
    "currency",
    "accession",
    "source",
    "collected_at",
]

FILING_COLUMNS = [
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

SEC_ISSUER_COLUMNS = [
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

SEGMENT_EXTRACT_LOG_COLUMNS = [
    "ticker",
    "market",
    "accession",
    "method",
    "status",
    "reason",
    "created_at",
]

RAW_FACT_COLUMNS = [
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

FINANCIAL_EXTRA_COLUMNS = [
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

SEGMENT_MEMBER_GEO_TOKENS = (
    "geograph",
    "country",
    "region",
    "domestic",
    "international",
    "americas",
    "europe",
    "asia",
    "japan",
    "china",
    "apac",
    "emea",
    "latam",
)
SEGMENT_MEMBER_PRODUCT_TOKENS = (
    "product",
    "service",
    "platform",
    "iphone",
    "ipad",
    "mac",
    "wearable",
    "software",
    "hardware",
)
SEGMENT_MEMBER_EXCLUDE_TOKENS = (
    "fairvalue",
    "debtsecurities",
    "treasury",
    "derivative",
    "note",
    "availableforsale",
    "operatingsegmentsmember",
    "assetbacked",
    "certificatesofdeposit",
    "certificateofdeposit",
    "commercialpaper",
    "mortgage",
    "municipal",
    "government",
    "politicalsubdivision",
    "commercialcustomer",
    "intersegment",
    "elimination",
    "unallocated",
    "statementequitycomponentsaxis",
    "statementequitycomponentsmember",
    "retainedearningsmember",
    "classofstockaxis",
    "classesofcommonstockaxis",
    "legalentityaxis",
    "legalentitymember",
    "parentcompanymember",
    "commonstockmember",
)


@dataclass
class SecSegmentBundle:
    wide: pd.DataFrame
    facts: pd.DataFrame
    filings: pd.DataFrame
    extract_log: pd.DataFrame


@dataclass(frozen=True)
class MetricSpec:
    tags: tuple[str, ...]
    preferred_units: tuple[str, ...]
    is_flow: bool


METRIC_SPECS: dict[str, MetricSpec] = {
    "Revenue": MetricSpec(
        tags=(
            "us-gaap:Revenues",
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
            "us-gaap:SalesRevenueNet",
            "us-gaap:SalesRevenueGoodsNet",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Gross Profit": MetricSpec(
        tags=("us-gaap:GrossProfit",),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "COGS": MetricSpec(
        tags=(
            # For health-insurance issuers (e.g. UNH), policyholder medical-benefit
            # payments are the primary cost of providing coverage and map to Compustat
            # cogsq.  This tag is XBRL-specific to insurance companies so it is safe
            # to place at highest priority — non-insurance filers do not report it.
            "us-gaap:PolicyholderBenefitsAndClaimsIncurredNet",
            "us-gaap:CostOfGoodsAndServicesSold",
            "us-gaap:CostOfRevenue",
            "us-gaap:CostOfGoodsSold",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "SG&A": MetricSpec(
        tags=(
            "us-gaap:SellingGeneralAndAdministrativeExpense",
            "us-gaap:SellingAndMarketingExpense",
            "us-gaap:GeneralAndAdministrativeExpense",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "R&D": MetricSpec(
        tags=(
            "us-gaap:ResearchAndDevelopmentExpense",
            "us-gaap:OtherResearchAndDevelopmentExpense",
            "us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
            "us-gaap:ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
            "ifrs-full:ResearchAndDevelopmentExpense",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Operating Income": MetricSpec(
        tags=("us-gaap:OperatingIncomeLoss",),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "D&A": MetricSpec(
        tags=(
            # Broad consolidated D&A totals (highest priority - quarterly derivable from YTD cash-flow)
            "us-gaap:DepreciationDepletionAndAmortization",
            "us-gaap:DepreciationAndAmortization",
            "us-gaap:DepreciationAndAmortizationExpense",
            "us-gaap:DepreciationAmortizationAndAccretionNet",
            # Oil & gas: results-of-operations DD&A (NOG, DVN, MRO — sector-specific tag)
            "us-gaap:ResultsOfOperationsDepreciationDepletionAmortizationAndAccretion",
            # Rental/service companies embed equipment depreciation in cost-of-revenues (e.g. URI)
            "us-gaap:CostOfGoodsAndServicesSoldDepreciationAndAmortization",
            # Car rental fleet depreciation reported under cost-of-goods (CAR, HTZ)
            "us-gaap:CostOfGoodsAndServicesSoldDepreciation",
            # Depreciation-only tags (used when amortization is separately disclosed)
            "us-gaap:Depreciation",
            "us-gaap:CostDepreciationAmortizationAndDepletion",
            "us-gaap:OtherDepreciationAndAmortization",
            # Bank / financial-sector premises depreciation (TFC, USB, MTB — no combined DDA tag)
            "us-gaap:DepreciationNonproduction",
            # Last resort: amortization-only (broadens recall for software/IP-heavy issuers)
            "us-gaap:AmortizationOfIntangibleAssets",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Amortization": MetricSpec(
        tags=(
            "us-gaap:AmortizationOfIntangibleAssets",
            "us-gaap:AmortizationOfAcquiredIntangibleAssets",
            "us-gaap:FiniteLivedIntangibleAssetsAmortizationExpense",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "SBC": MetricSpec(
        tags=(
            # Cash-flow non-cash adjustment tags (most commonly reported quarterly)
            "us-gaap:ShareBasedCompensation",
            # Income-statement / notes disclosure tags
            "us-gaap:AllocatedShareBasedCompensationExpense",
            "us-gaap:ShareBasedCompensationArrangementByShareBasedPaymentAwardCompensationCost",
            "us-gaap:EmployeeBenefitsAndShareBasedCompensation",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Interest": MetricSpec(
        tags=(
            "us-gaap:InterestExpense",
            "us-gaap:InterestAndDebtExpense",
            "us-gaap:InterestExpenseDebt",
            "us-gaap:InterestExpenseNonoperating",
            "us-gaap:InterestExpenseBorrowings",
            "us-gaap:InterestExpenseOperating",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Pretax Income": MetricSpec(
        tags=(
            "us-gaap:IncomeBeforeTax",
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Tax": MetricSpec(
        tags=("us-gaap:IncomeTaxExpenseBenefit",),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Net Income": MetricSpec(
        tags=("us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Net Income Common": MetricSpec(
        tags=(
            "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic",
            "us-gaap:NetIncomeLossAvailableToCommonStockholdersDiluted",
            "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasicAndDiluted",
            "us-gaap:NetIncomeLossAttributableToCommonStockholders",
            "us-gaap:NetIncomeLossFromContinuingOperationsAvailableToCommonShareholdersBasic",
            "us-gaap:NetIncomeLossFromContinuingOperationsAvailableToCommonShareholdersDiluted",
            "us-gaap:NetIncomeLossAttributableToParentDiluted",
            "us-gaap:NetIncomeLossAttributableToParent",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Other Gain": MetricSpec(
        tags=(
            "us-gaap:OtherNonoperatingIncomeExpense",
            "us-gaap:NonoperatingIncomeExpense",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Financial Gain": MetricSpec(
        tags=(
            "us-gaap:InvestmentIncomeInterest",
            "us-gaap:InvestmentIncomeNet",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Equity Method Gain": MetricSpec(
        tags=(
            "us-gaap:IncomeLossFromEquityMethodInvestments",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Other Income": MetricSpec(
        tags=(
            "us-gaap:OtherIncome",
            "us-gaap:OtherNonoperatingIncome",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Other Expense": MetricSpec(
        tags=(
            "us-gaap:OtherExpense",
            "us-gaap:OtherNonoperatingExpense",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Financial Income": MetricSpec(
        tags=(
            "us-gaap:InterestIncome",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Financial Expense": MetricSpec(
        tags=(
            "us-gaap:InterestExpense",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "EPS": MetricSpec(
        tags=(
            "us-gaap:EarningsPerShareDiluted",
            "us-gaap:EarningsPerShareBasic",
        ),
        preferred_units=("USD/shares",),
        is_flow=True,
    ),
    "Diluted EPS": MetricSpec(
        tags=("us-gaap:EarningsPerShareDiluted",),
        preferred_units=("USD/shares",),
        is_flow=True,
    ),
    "Operating Cash Flow": MetricSpec(
        tags=(
            "us-gaap:NetCashProvidedByUsedInOperatingActivities",
            "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Investing Cash Flow": MetricSpec(
        tags=(
            "us-gaap:NetCashProvidedByUsedInInvestingActivities",
            "us-gaap:NetCashProvidedByUsedInInvestingActivitiesContinuingOperations",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Financing Cash Flow": MetricSpec(
        tags=(
            "us-gaap:NetCashProvidedByUsedInFinancingActivities",
            "us-gaap:NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Capital Expenditure": MetricSpec(
        tags=(
            "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
            "us-gaap:CapitalExpendituresIncurredButNotYetPaid",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Dividends Paid": MetricSpec(
        tags=(
            "us-gaap:PaymentsOfDividends",
            "us-gaap:PaymentsOfDividendsCommonStock",
            "us-gaap:Dividends",
            "us-gaap:DividendsCash",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Repurchases": MetricSpec(
        tags=(
            "us-gaap:PaymentsForRepurchaseOfCommonStock",
            "us-gaap:PaymentsForRepurchaseOfEquity",
            "us-gaap:StockRepurchasedDuringPeriodValue",
            "us-gaap:StockRepurchasedAndRetiredDuringPeriodValue",
            "us-gaap:TreasuryStockValueAcquiredCostMethod",
        ),
        preferred_units=("USD",),
        is_flow=True,
    ),
    "Total Assets": MetricSpec(
        tags=("us-gaap:Assets",),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Total Liabilities": MetricSpec(
        tags=("us-gaap:Liabilities",),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Shareholders Equity": MetricSpec(
        tags=(
            "us-gaap:StockholdersEquity",
            "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "us-gaap:CommonStockholdersEquity",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Current Assets": MetricSpec(
        tags=("us-gaap:AssetsCurrent",),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Current Liabilities": MetricSpec(
        tags=("us-gaap:LiabilitiesCurrent",),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "AR": MetricSpec(
        tags=(
            "us-gaap:AccountsReceivableNetCurrent",
            "us-gaap:ReceivablesNetCurrent",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "AP": MetricSpec(
        tags=("us-gaap:AccountsPayableCurrent",),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Inventory": MetricSpec(
        tags=(
            "us-gaap:InventoryNet",
            "us-gaap:InventoryFinishedGoods",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Cash": MetricSpec(
        tags=(
            "us-gaap:CashCashEquivalentsAndShortTermInvestments",
            "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "us-gaap:CashAndSecuritiesSegregatedUnderFederalAndOtherRegulations",
            "us-gaap:CashAndSecuritiesSegregatedUnderSecuritiesExchangeCommissionRegulation",
            "us-gaap:CashSegregatedUnderOtherRegulations",
            "us-gaap:CashAndCashEquivalentsAtCarryingValue",
            "us-gaap:Cash",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Debt Short": MetricSpec(
        tags=(
            "us-gaap:DebtCurrent",
            "us-gaap:ShortTermBorrowings",
            "us-gaap:LongTermDebtCurrent",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Debt Long": MetricSpec(
        tags=(
            # Noncurrent-only tags first — these correctly exclude the current portion
            "us-gaap:LongTermDebtNoncurrent",
            "us-gaap:LongTermDebtAndCapitalLeaseObligations",
            # LongTermDebt is a TOTAL (current + non-current) tag; keep as last-resort fallback only
            "us-gaap:LongTermDebt",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Deferred Revenue": MetricSpec(
        tags=(
            "us-gaap:ContractWithCustomerLiability",
            "us-gaap:DeferredRevenue",
            "us-gaap:DeferredRevenueAndCreditsCurrent",
            "us-gaap:ContractWithCustomerLiabilityCurrent",
            "us-gaap:DeferredRevenueCurrent",
            "us-gaap:DeferredRevenueNoncurrent",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Goodwill": MetricSpec(
        tags=("us-gaap:Goodwill",),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Intangibles": MetricSpec(
        tags=(
            "us-gaap:FiniteLivedIntangibleAssetsNet",
            "us-gaap:IntangibleAssetsNetExcludingGoodwill",
            "us-gaap:IntangibleAssetsNetIncludingGoodwill",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Common Stock": MetricSpec(
        tags=("us-gaap:CommonStockValue",),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "APIC": MetricSpec(
        tags=(
            "us-gaap:AdditionalPaidInCapital",
            "us-gaap:AdditionalPaidInCapitalCommonStock",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Retained Earnings": MetricSpec(
        tags=("us-gaap:RetainedEarningsAccumulatedDeficit",),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "AOCI": MetricSpec(
        tags=(
            "us-gaap:AccumulatedOtherComprehensiveIncomeLossNetOfTax",
            "us-gaap:AccumulatedOtherComprehensiveIncomeLossNetOfTaxPortionAttributableToParent",
            "us-gaap:AccumulatedOtherComprehensiveIncome",
            "ifrs-full:AccumulatedOtherComprehensiveIncome",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Current Fin Assets": MetricSpec(
        tags=(
            "us-gaap:ShortTermInvestments",
            "us-gaap:AvailableForSaleSecuritiesCurrent",
            "us-gaap:AvailableForSaleDebtSecuritiesCurrent",
            "us-gaap:MarketableSecuritiesCurrent",
            "us-gaap:TradingSecuritiesCurrent",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Non Current Fin Assets": MetricSpec(
        tags=(
            "us-gaap:OtherInvestments",
            "us-gaap:AvailableForSaleSecuritiesNoncurrent",
            "us-gaap:AvailableForSaleDebtSecuritiesNoncurrent",
            "us-gaap:MarketableSecuritiesNoncurrent",
            "us-gaap:TradingSecuritiesNoncurrent",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Current Fin Liabilities": MetricSpec(
        tags=(
            "us-gaap:DebtCurrent",
            "us-gaap:ShortTermBorrowings",
            "us-gaap:LongTermDebtCurrent",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Non Current Fin Liabilities": MetricSpec(
        tags=(
            # Noncurrent-only tags first — these correctly exclude the current portion
            "us-gaap:LongTermDebtNoncurrent",
            "us-gaap:LongTermDebtAndCapitalLeaseObligations",
            # LongTermDebt is a TOTAL (current + non-current) tag; keep as last-resort fallback only
            "us-gaap:LongTermDebt",
        ),
        preferred_units=("USD",),
        is_flow=False,
    ),
    "Shares": MetricSpec(
        tags=(
            "dei:EntityCommonStockSharesOutstanding",
            "us-gaap:CommonStockSharesOutstanding",
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
            "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
        ),
        preferred_units=("shares",),
        is_flow=False,
    ),
    "Diluted Shares": MetricSpec(
        tags=("us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",),
        preferred_units=("shares",),
        is_flow=True,
    ),
    "Basic Shares": MetricSpec(
        tags=("us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",),
        preferred_units=("shares",),
        is_flow=True,
    ),
}


def _sec_headers(user_agent: str | None = None) -> dict[str, str]:
    ua = (user_agent or os.getenv(SEC_USER_AGENT_ENV) or SEC_DEFAULT_USER_AGENT).strip()
    return {
        "User-Agent": ua,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }


def _split_tag(tag: str) -> tuple[str, str]:
    text = str(tag).strip()
    if ":" in text:
        ns, name = text.split(":", 1)
        return ns.strip(), name.strip()
    return "us-gaap", text


def _unit_suffix_multiplier(raw_suffix: str) -> float:
    text = re.sub(r"[^a-z0-9]", "", str(raw_suffix or "").lower())
    if text in {"", "u", "unit", "units"}:
        return 1.0
    if text in {"k", "thousand", "thousands", "000", "th", "ths"}:
        return 1_000.0
    if text in {"m", "mn", "mm", "million", "millions", "000000"}:
        return 1_000_000.0
    if text in {"b", "bn", "billion", "billions", "000000000"}:
        return 1_000_000_000.0
    if text.startswith("thousand"):
        return 1_000.0
    if text.startswith("million"):
        return 1_000_000.0
    if text.startswith("billion"):
        return 1_000_000_000.0
    if text.isdigit() and len(text) % 3 == 0:
        groups = len(text) // 3
        return float(1000 ** groups)
    return 1.0


def _unit_multiplier(unit: str, preferred_units: tuple[str, ...]) -> float:
    raw = str(unit or "").strip()
    if not raw:
        return 1.0
    if raw in preferred_units:
        return 1.0

    lower = raw.lower()
    left = lower.split("/", 1)[0]

    if left.startswith("usd"):
        return _unit_suffix_multiplier(left[3:])
    if left.startswith("shares"):
        return _unit_suffix_multiplier(left[6:])
    return 1.0


def _quarter_end(ts: pd.Timestamp) -> pd.Timestamp:
    # Use actual reported period-end date (fiscal-aware) instead of calendar quarter end.
    # Calendar coercion can shift non-December fiscal reporters by ~1-2 months.
    return pd.Timestamp(ts).normalize()


def _normalize_accession(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"[^0-9]", "", text)


def _infer_form_type(period_end: pd.Timestamp, current: str | None = None) -> str:
    form = str(current or "").strip().upper()
    if form in SEC_ALLOWED_FORMS:
        return form
    if pd.isna(period_end):
        return "10-Q"
    return "10-K" if int(pd.Timestamp(period_end).month) == 12 else "10-Q"


def _load_trading_days_for_market(market: str = "us") -> pd.DatetimeIndex:
    market_norm = str(market or "us").strip().lower()
    from market_data.db_router import db_available_for_market, get_prices_connection_for_market

    if not db_available_for_market(market_norm):
        return pd.DatetimeIndex([])

    try:
        con = get_prices_connection_for_market(market_norm)
        refs = ["005930", "000660", "035420"] if market_norm == "kr" else ["SPY", "IVV", "VOO", "AAPL"]
        for ref in refs:
            rows = con.execute(
                "SELECT DISTINCT date FROM prices WHERE ticker = ? AND market = ? ORDER BY date",
                [ref, market_norm],
            ).fetchall()
            if rows:
                return pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows]).dropna().sort_values().unique()
        rows = con.execute(
            "SELECT DISTINCT date FROM prices WHERE market = ? ORDER BY date",
            [market_norm],
        ).fetchall()
    except Exception:
        return pd.DatetimeIndex([])
    if not rows:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows]).dropna().sort_values().unique()


def _next_trading_day(ts: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    dt = pd.Timestamp(ts).normalize()
    if len(trading_days) == 0:
        return (dt + pd.offsets.BDay(1)).normalize()
    pos = int(np.searchsorted(trading_days.values, np.datetime64(dt), side="right"))
    if pos >= len(trading_days):
        return (dt + pd.offsets.BDay(1)).normalize()
    return pd.Timestamp(trading_days[pos]).normalize()


def _sec_time_regime(period_end: pd.Timestamp | pd.NaT) -> str:
    if pd.notna(period_end) and pd.Timestamp(period_end).normalize() < SEC_HIGH_CONFIDENCE_START_DATE:
        return SEC_TIME_REGIME_PRE_2012
    return SEC_TIME_REGIME_POST_2012


def _compatibility_lane_requested(min_date: pd.Timestamp | pd.NaT) -> bool:
    return bool(pd.notna(min_date) and pd.Timestamp(min_date).normalize() < SEC_HIGH_CONFIDENCE_START_DATE)


def _coerce_available_date(
    *,
    filing_date: pd.Timestamp | pd.NaT,
    accepted_at: pd.Timestamp | pd.NaT,
    period_end: pd.Timestamp | pd.NaT,
    form_type: str,
    use_next_trading_day: bool,
    trading_days: pd.DatetimeIndex,
    fallback_enabled: bool,
    fallback_q_days: int,
    fallback_k_days: int,
) -> tuple[pd.Timestamp | pd.NaT, str]:
    base = pd.NaT
    if pd.notna(accepted_at):
        _ts = pd.Timestamp(accepted_at)
        base = (_ts.tz_localize(None) if _ts.tz is not None else _ts).normalize()
    elif pd.notna(filing_date):
        _ts = pd.Timestamp(filing_date)
        base = (_ts.tz_localize(None) if _ts.tz is not None else _ts).normalize()

    if pd.notna(base):
        if use_next_trading_day:
            return _next_trading_day(pd.Timestamp(base), trading_days), "filed_next_trading_day"
        return pd.Timestamp(base), "filed"

    if not fallback_enabled or pd.isna(period_end):
        return pd.NaT, "missing"

    period_ts = pd.Timestamp(period_end).normalize()
    time_regime = _sec_time_regime(period_ts)
    form = _infer_form_type(period_ts, form_type)
    lag_days = int(fallback_k_days if "10-K" in form else fallback_q_days)
    fallback_date = (period_ts + pd.Timedelta(days=max(lag_days, 0))).normalize()
    if use_next_trading_day:
        fallback_date = _next_trading_day(fallback_date, trading_days)
        return (
            fallback_date,
            "compatibility_fallback_next_trading_day" if time_regime == SEC_TIME_REGIME_PRE_2012 else "fallback_next_trading_day",
        )
    return fallback_date, "compatibility_fallback" if time_regime == SEC_TIME_REGIME_PRE_2012 else "fallback"


def _coerce_numeric(val: Any) -> float:
    num = pd.to_numeric(val, errors="coerce")
    if pd.isna(num):
        return float("nan")
    return float(num)


_GENERIC_SEC_CONTEXT_RE = re.compile(r"^(?:CY|FY)\d{4}", re.IGNORECASE)


def _form_rank(form: str) -> int:
    text = str(form or "").strip().upper()
    if text == "10-Q":
        return 4
    if text == "10-Q/A":
        return 3
    if text == "10-K":
        return 2
    if text == "10-K/A":
        return 1
    return 0


def _normalize_context_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(value or "").strip()).upper()


def _context_specificity_rank(context_id: Any, accession: Any) -> int:
    ctx = str(context_id or "").strip()
    if not ctx:
        return 0
    ctx_norm = _normalize_context_token(ctx)
    acc_norm = _normalize_accession(accession)
    if acc_norm and acc_norm.upper() in ctx_norm:
        return 3
    if _GENERIC_SEC_CONTEXT_RE.match(ctx):
        return 1
    return 2


def _fact_records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(records):
        end = pd.to_datetime(item.get("end"), errors="coerce")
        if pd.isna(end):
            continue
        val = _coerce_numeric(item.get("val"))
        if not np.isfinite(val):
            continue
        start_raw = item.get("start")
        start = pd.to_datetime(start_raw, errors="coerce") if start_raw is not None else pd.NaT
        filed = pd.to_datetime(item.get("filed"), errors="coerce")
        accession = _normalize_accession(item.get("accn"))
        context_id = _raw_context_id(item, idx)
        rows.append(
            {
                "start": start,
                "end": end,
                "quarter_end": _quarter_end(pd.Timestamp(end)),
                "val": val,
                "fp": str(item.get("fp", "")).strip().upper(),
                "form": str(item.get("form", "")).strip().upper(),
                "filed": filed,
                "accession": accession,
                "context_id": context_id,
                "context_rank": _context_specificity_rank(context_id, accession),
                "is_generic_context": bool(_GENERIC_SEC_CONTEXT_RE.match(str(context_id or "").strip())),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "quarter_end",
                "val",
                "fp",
                "form",
                "filed",
                "accession",
                "context_id",
                "context_rank",
                "is_generic_context",
            ]
        )
    out = pd.DataFrame(rows)
    out["duration_days"] = (out["end"] - out["start"]).dt.days + 1
    out["duration_days"] = pd.to_numeric(out["duration_days"], errors="coerce")
    return out


def _normalize_expense_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.empty:
        return out
    return out.abs()


def _reconcile_signed_reconstruction(
    direct_series: pd.Series,
    reconstructed_series: pd.Series,
    *,
    rel_tol: float = 0.15,
) -> pd.Series:
    direct = pd.to_numeric(direct_series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    reconstructed = pd.to_numeric(reconstructed_series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    union_index = direct.index.union(reconstructed.index)
    if union_index.empty:
        return pd.Series(dtype=float)
    direct = direct.reindex(union_index)
    reconstructed = reconstructed.reindex(union_index)
    out = direct.copy()
    use_reconstructed = out.isna() & reconstructed.notna()
    mag_close = (out.abs() - reconstructed.abs()).abs() <= np.maximum(
        np.maximum(out.abs(), reconstructed.abs()) * float(rel_tol),
        1.0,
    )
    sign_conflict = (
        out.notna()
        & reconstructed.notna()
        & (np.sign(out) != np.sign(reconstructed))
        & mag_close
    )
    out = out.where(~(use_reconstructed | sign_conflict), reconstructed)
    return pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan).sort_index()


def _apply_negative_margin_gross_profit_proxy(
    *,
    revenue_series: pd.Series,
    cogs_series: pd.Series,
    gross_profit_series: pd.Series,
    sga_series: pd.Series,
    operating_income_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    revenue = pd.to_numeric(revenue_series, errors="coerce")
    cogs = _normalize_expense_series(cogs_series)
    gross_profit = pd.to_numeric(gross_profit_series, errors="coerce")
    sga = _normalize_expense_series(sga_series)
    operating_income = pd.to_numeric(operating_income_series, errors="coerce")
    union_index = revenue.index.union(cogs.index).union(gross_profit.index).union(sga.index).union(operating_income.index)
    if union_index.empty:
        return cogs.sort_index(), gross_profit.sort_index()

    revenue = revenue.reindex(union_index)
    cogs = cogs.reindex(union_index)
    gross_profit = gross_profit.reindex(union_index)
    sga = sga.reindex(union_index)
    operating_income = operating_income.reindex(union_index)

    tiny_cogs = cogs.isna() | (
        revenue.notna()
        & cogs.notna()
        & cogs.le(revenue.abs() * 0.10)
    )
    opex_gap = _normalize_expense_series(gross_profit - operating_income)
    # Guard: if SGA alone explains ≥70% of the opex gap, the operating loss is driven by
    # operating expenses (SG&A, R&D), not by missing COGS.  Applying the proxy in this case
    # would incorrectly inflate COGS for pre-revenue / R&D-intensive issuers.
    sga_explains_gap = sga.notna() & opex_gap.notna() & (sga >= opex_gap * 0.70)
    use_opex_gap_proxy = (
        revenue.notna()
        & gross_profit.notna()
        & operating_income.notna()
        & gross_profit.gt(0)
        & operating_income.lt(0)
        & opex_gap.gt(revenue.abs() * 1.02)
        & tiny_cogs
        & ~sga_explains_gap
    )
    use_sga_proxy = (
        revenue.notna()
        & sga.notna()
        & gross_profit.notna()
        & gross_profit.gt(0)
        & sga.gt(revenue.abs() * 1.02)
        & tiny_cogs
        & ~use_opex_gap_proxy
    )

    cogs = cogs.where(~use_opex_gap_proxy, opex_gap)
    cogs = cogs.where(~use_sga_proxy, sga)
    reconstructed_gross_profit = revenue - cogs
    gross_profit = _reconcile_signed_reconstruction(gross_profit, reconstructed_gross_profit)
    force_reconstructed = use_opex_gap_proxy | use_sga_proxy
    gross_profit = gross_profit.where(~force_reconstructed, reconstructed_gross_profit)
    return _normalize_expense_series(cogs).dropna().sort_index(), pd.to_numeric(gross_profit, errors="coerce").dropna().sort_index()


def _pick_flow_quarter_values(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    out = frame.copy()
    if "context_rank" not in out.columns:
        out["context_rank"] = 0
    if "is_generic_context" not in out.columns:
        out["is_generic_context"] = False
    out["is_quarter_duration"] = out["duration_days"].between(70, 130, inclusive="both")
    out["form_rank"] = out["form"].map(_form_rank)

    # 1) Direct quarter values (duration ~ one quarter)
    direct = out.loc[
        out["is_quarter_duration"]
        & ~(
            out["form"].isin({"10-K", "10-K/A"})
            & out["is_generic_context"].fillna(False)
        )
    ].copy()
    if direct.empty:
        direct = out.loc[out["is_quarter_duration"]].copy()
    direct_series = pd.Series(dtype=float)
    if not direct.empty:
        direct = direct.sort_values(
            ["quarter_end", "context_rank", "form_rank", "filed"],
            ascending=[True, False, False, False],
        )
        picked = direct.drop_duplicates(subset=["quarter_end"], keep="first")
        direct_series = pd.Series(picked["val"].to_numpy(dtype=float), index=pd.DatetimeIndex(picked["quarter_end"]))
        direct_series = direct_series[~direct_series.index.duplicated(keep="last")].sort_index()

    # 2) Derive quarter values from cumulative records grouped by same fiscal-year start.
    derived: dict[pd.Timestamp, float] = {}
    with_start = out.loc[out["start"].notna()].copy()
    if not with_start.empty:
        for _start, group in with_start.groupby("start", sort=False):
            grp = group.copy()
            grp["end"] = pd.to_datetime(grp["end"], errors="coerce")
            grp = grp.loc[~grp["end"].isna()].copy()
            if grp.empty:
                continue
            grp = grp.sort_values(
                ["end", "context_rank", "duration_days", "form_rank", "filed"],
                ascending=[True, False, False, False, False],
            )
            by_end: list[tuple[pd.Timestamp, pd.DataFrame]] = []
            for end_dt, end_group in grp.groupby("end", sort=True):
                ordered = end_group.sort_values(
                    ["context_rank", "duration_days", "form_rank", "filed"],
                    ascending=[False, False, False, False],
                )
                by_end.append((pd.Timestamp(end_dt), ordered))

            # Need at least 3 points and at least one 6M/9M cumulative point.
            if len(by_end) < 3:
                continue
            if not grp["duration_days"].between(131, 320, inclusive="both").any():
                continue

            prev_cum: float | None = None
            for _end_dt, end_group in by_end:
                pick = end_group.iloc[0]
                val = _coerce_numeric(pick.get("val"))
                dur = _coerce_numeric(pick.get("duration_days"))

                if prev_cum is not None and np.isfinite(prev_cum) and np.isfinite(val):
                    # Prefer a same-end alternative that keeps cumulative continuity.
                    # This avoids mixing amended records that drastically reset FY values.
                        continuity_floor = float(prev_cum) * 0.85
                        if val < continuity_floor:
                            alt = end_group.loc[pd.to_numeric(end_group["val"], errors="coerce") >= continuity_floor]
                            if not alt.empty:
                                pick = alt.sort_values(
                                    ["context_rank", "duration_days", "form_rank", "filed"],
                                    ascending=[False, False, False, False],
                                ).iloc[0]
                                val = _coerce_numeric(pick.get("val"))
                                dur = _coerce_numeric(pick.get("duration_days"))

                if not np.isfinite(val):
                    continue
                q_end = _quarter_end(pd.Timestamp(pick.get("end")))

                if prev_cum is None:
                    # First point can serve as Q1 only when it is quarter-length.
                    if np.isfinite(dur) and 70 <= dur <= 130:
                        derived[q_end] = val
                    prev_cum = val
                    continue

                q_val = val - prev_cum
                prev_cum = val
                if np.isfinite(q_val):
                    # keep latest if duplicate quarter end appears
                    derived[q_end] = q_val

    derived_series = pd.Series(dtype=float)
    if derived:
        derived_series = pd.Series(derived, dtype=float).sort_index()

    if direct_series.empty and derived_series.empty:
        return pd.Series(dtype=float)
    if direct_series.empty:
        return derived_series
    if derived_series.empty:
        return direct_series

    # Direct quarter values remain priority; cumulative-derived fills only gaps.
    return direct_series.combine_first(derived_series).sort_index()


def _pick_flow_annual_values(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    out = frame.copy()
    if "context_rank" not in out.columns:
        out["context_rank"] = 0
    # Some issuers (notably in older restated companyfacts histories) expose
    # quarter-duration rows relabeled with fp='FY' in later filings. Duration
    # needs to dominate here or Q4 derivation will treat those quarter rows as
    # full-year anchors and generate nonsensical negative deltas.
    out["is_annual"] = (out["duration_days"] >= 300) | (
        out["duration_days"].isna() & out["fp"].eq("FY")
    )
    out = out.loc[out["is_annual"]]
    if out.empty:
        return pd.Series(dtype=float)
    out["form_rank"] = out["form"].map(_form_rank)
    out = out.sort_values(
        ["quarter_end", "context_rank", "form_rank", "filed", "duration_days"],
        ascending=[True, False, False, False, False],
    )
    picked = out.drop_duplicates(subset=["quarter_end"], keep="first")
    return pd.Series(picked["val"].to_numpy(dtype=float), index=pd.DatetimeIndex(picked["quarter_end"]))


def _pick_direct_quarter_values(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    out = frame.copy()
    out["is_quarter_duration"] = out["duration_days"].between(70, 130, inclusive="both")
    out["form_rank"] = out["form"].map(_form_rank)
    out = out.loc[out["is_quarter_duration"]]
    if out.empty:
        return pd.Series(dtype=float)
    out = out.sort_values(["quarter_end", "form_rank", "filed"], ascending=[True, False, False])
    picked = out.drop_duplicates(subset=["quarter_end"], keep="first")
    series = pd.Series(picked["val"].to_numpy(dtype=float), index=pd.DatetimeIndex(picked["quarter_end"]))
    return series[~series.index.duplicated(keep="last")].sort_index()


def _fill_q4_from_annual(quarterly: pd.Series, annual: pd.Series) -> pd.Series:
    if quarterly.empty or annual.empty:
        return quarterly
    out = quarterly.copy()
    for annual_end, annual_val in annual.items():
        annual_ts = pd.Timestamp(annual_end)
        history = pd.DatetimeIndex(
            sorted(ts for ts in pd.DatetimeIndex(out.index) if pd.Timestamp(ts) < annual_ts)
        )
        if len(history) < 3:
            continue
        prior_quarters = history[-3:]
        all_quarters = prior_quarters.append(pd.DatetimeIndex([annual_ts]))
        deltas = pd.Series(all_quarters).diff().dt.days.dropna()
        if deltas.empty or not deltas.between(70, 130, inclusive="both").all():
            continue
        q_vals = pd.to_numeric(out.reindex(prior_quarters), errors="coerce")
        if q_vals.isna().any() or not np.isfinite(float(annual_val)):
            continue
        q4_idx = annual_ts
        q4_existing = out.get(q4_idx, np.nan)
        derived = float(annual_val) - float(q_vals.sum())
        if not np.isfinite(derived):
            continue

        if not np.isfinite(q4_existing):
            out.loc[q4_idx] = derived
            continue

        # If q4 looks like full-year (common in 10-K), replace with derived quarter value.
        if abs(float(q4_existing) - float(annual_val)) <= max(abs(float(annual_val)) * 0.03, 1.0):
            out.loc[q4_idx] = derived
    return out.sort_index()


def _pick_stock_quarter_values(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    out = frame.copy()
    if "context_rank" not in out.columns:
        out["context_rank"] = 0
    out["is_stock_like"] = out["start"].isna() | (out["duration_days"] <= 7)
    out["form_rank"] = out["form"].map(_form_rank)
    out = out.sort_values(
        ["quarter_end", "is_stock_like", "context_rank", "form_rank", "filed"],
        ascending=[True, False, False, False, False],
    )
    picked = out.drop_duplicates(subset=["quarter_end"], keep="first")
    series = pd.Series(picked["val"].to_numpy(dtype=float), index=pd.DatetimeIndex(picked["quarter_end"]))
    return series[~series.index.duplicated(keep="last")].sort_index()


def _iter_unit_records(
    companyfacts: dict[str, Any],
    namespace: str,
    name: str,
    preferred_units: tuple[str, ...],
) -> list[tuple[str, list[dict[str, Any]]]]:
    scope = companyfacts.get("facts", {}).get(namespace, {}).get(name, {})
    units = scope.get("units", {})
    if not isinstance(units, dict) or not units:
        return []

    ordered_units: list[str] = []
    for unit in preferred_units:
        if unit in units and unit not in ordered_units:
            ordered_units.append(unit)
    # Do NOT fall back to non-preferred units (e.g. JPY/CNY).
    # Foreign ADRs (MUFG, TM, NMR…) file XBRL in local currency only;
    # using those values as USD causes trillion-yen amounts to be stored
    # as dollar figures.  If no preferred unit is present, return empty.

    out: list[tuple[str, list[dict[str, Any]]]] = []
    for unit in ordered_units:
        rows = units.get(unit)
        if isinstance(rows, list) and rows:
            out.append((str(unit), rows))
    return out


def _normalize_share_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.dropna().size < 3:
        return out

    vals = out.to_numpy(dtype=float)
    n = len(vals)
    if n < 3:
        return out

    # Pass 1: fix unit-scale mixups (e.g. 351 vs 351,000,000 in "shares" unit).
    for i in range(n):
        v = vals[i]
        if not np.isfinite(v) or v <= 0:
            continue

        lo = max(0, i - 4)
        hi = min(n, i + 5)
        neigh = np.concatenate([vals[lo:i], vals[i + 1:hi]])
        neigh = neigh[np.isfinite(neigh) & (neigh > 0)]
        if neigh.size < 3:
            continue

        groups = np.floor(np.log10(neigh) / 3.0).astype(int)
        uniq, counts = np.unique(groups, return_counts=True)
        dom_idx = int(np.argmax(counts))
        dom_group = int(uniq[dom_idx])
        dom_share = float(counts[dom_idx]) / float(neigh.size)
        if dom_share < 0.6:
            continue

        dom_vals = neigh[groups == dom_group]
        if dom_vals.size < 2:
            continue
        base = float(np.median(dom_vals))
        if not np.isfinite(base) or base <= 0:
            continue

        v_group = int(np.floor(np.log10(v) / 3.0))
        diff_groups = dom_group - v_group
        if diff_groups == 0 or abs(diff_groups) > 3:
            continue

        if diff_groups < 0:
            # Avoid over-aggressive downscaling unless local evidence is very strong.
            if dom_share < 0.80 or dom_vals.size < 3:
                continue

        scaled = v * float(1000.0 ** diff_groups)
        ratio = scaled / base
        if np.isfinite(ratio) and 0.2 <= ratio <= 5.0:
            vals[i] = scaled

    # Endpoint cleanup for isolated unit mismatches.
    endpoint_factors = [1_000.0, 1_000_000.0, 1_000_000_000.0]
    for i in (0, n - 1):
        v = vals[i]
        if not np.isfinite(v) or v <= 0:
            continue
        neigh = vals[1:3] if i == 0 else vals[max(0, n - 3):n - 1]
        neigh = neigh[np.isfinite(neigh) & (neigh > 0)]
        if neigh.size < 2:
            continue
        p25, p75 = np.percentile(neigh, [25, 75])
        if not np.isfinite(p25) or p25 <= 0 or not np.isfinite(p75):
            continue
        if (p75 / p25) > 3.0:
            continue
        base = float(np.median(neigh))
        ratio = v / base
        for f in endpoint_factors:
            inv = 1.0 / f
            if abs(ratio - inv) / inv <= 0.2:
                vals[i] = v * f
                break

    # Pass 1.5: fix isolated x1000/x1,000,000 outliers when immediate neighbors agree.
    # This catches SEC rows where the "shares" unit is intermittently reported in thousands.
    scale_factors = [1_000.0, 1_000_000.0]
    scale_tol = 0.25
    for i in range(1, n - 1):
        v = vals[i]
        left = vals[i - 1]
        right = vals[i + 1]
        if not np.isfinite(v) or v <= 0 or not np.isfinite(left) or left <= 0 or not np.isfinite(right) or right <= 0:
            continue

        # If one adjacent quarter clearly indicates x1,000/x1,000,000 mismatch,
        # repair first (the opposite side may be on post-split basis).
        candidate = None
        for neigh, other in ((left, right), (right, left)):
            if not np.isfinite(neigh) or neigh <= 0:
                continue
            ratio_single = v / neigh
            for f in scale_factors:
                inv = 1.0 / f
                scaled = None
                if abs(ratio_single - inv) / inv <= scale_tol:
                    scaled = v * f
                elif abs(ratio_single - f) / f <= scale_tol:
                    scaled = v / f
                if scaled is None:
                    continue
                if np.isfinite(other) and other > 0:
                    rel = scaled / other
                    if not (0.05 <= rel <= 20.0):
                        continue
                candidate = scaled
                break
            if candidate is not None:
                break
        if candidate is not None:
            vals[i] = candidate
            continue

        # Only apply scale correction when neighboring quarters are locally stable.
        neigh_ratio = max(left, right) / min(left, right)
        if not np.isfinite(neigh_ratio) or neigh_ratio > 4.0:
            continue

        base = float(np.sqrt(left * right))
        if not np.isfinite(base) or base <= 0:
            continue

        ratio = v / base
        for f in scale_factors:
            inv = 1.0 / f
            if abs(ratio - inv) / inv <= scale_tol:
                vals[i] = v * f
                break
            if abs(ratio - f) / f <= scale_tol:
                vals[i] = v / f
                break

    # Pass 2: fix isolated split-like outliers after unit normalization.
    factors = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 28.0]
    tol = 0.12
    for i in range(1, n - 1):
        v = vals[i]
        if not np.isfinite(v) or v <= 0:
            continue

        left = vals[max(0, i - 2):i]
        right = vals[i + 1:min(n, i + 3)]
        neigh = np.concatenate([left, right])
        neigh = neigh[np.isfinite(neigh) & (neigh > 0)]
        if neigh.size < 2:
            continue
        p25, p75 = np.percentile(neigh, [25, 75])
        if not np.isfinite(p25) or not np.isfinite(p75) or p25 <= 0:
            continue
        if (p75 / p25) > 3.0:
            # Mixed-scale neighborhood; skip split-ratio correction.
            continue
        base = float(np.median(neigh))
        if not np.isfinite(base) or base <= 0:
            continue

        ratio = v / base
        adjusted = v
        for f in factors:
            if abs(ratio - f) / f <= tol:
                adjusted = v / f
                break
            inv = 1.0 / f
            if abs(ratio - inv) / inv <= tol:
                adjusted = v * f
                break
        vals[i] = adjusted

    return pd.Series(vals, index=out.index, dtype=float)


def _enforce_balance_identity(
    assets: pd.Series,
    liabilities: pd.Series,
    equity: pd.Series,
    residual_tol: float = 0.20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    a = pd.to_numeric(assets, errors="coerce").replace([np.inf, -np.inf], np.nan)
    l = pd.to_numeric(liabilities, errors="coerce").replace([np.inf, -np.inf], np.nan)
    e = pd.to_numeric(equity, errors="coerce").replace([np.inf, -np.inf], np.nan)

    implied_equity = a - l
    sign_flipped_equity = (
        a.notna()
        & l.notna()
        & e.notna()
        & (implied_equity.notna())
        & (e * implied_equity < 0)
        & ((e.abs() - implied_equity.abs()).abs() <= np.maximum(implied_equity.abs() * 0.05, 1.0))
    )
    e = e.where(~sign_flipped_equity, implied_equity)

    e = e.where(e.notna(), a - l)
    l = l.where(l.notna(), a - e)
    a = a.where(a.notna(), l + e)

    def _resid(sa: pd.Series, sl: pd.Series, se: pd.Series) -> pd.Series:
        denom = sa.abs().replace(0, np.nan)
        return (sa - (sl + se)).abs() / denom

    resid = _resid(a, l, e)
    bad = resid > float(residual_tol)
    if bad.any():
        # For SEC common-stock issuers, assets/liabilities are usually more stable than
        # equity when companyfacts mixes generic CY instant contexts with accession-
        # specific contexts on opposite signs. Repair equity first when A and L exist.
        can_fix_e = bad & a.notna() & l.notna()
        e = e.where(~can_fix_e, a - l)

        resid = _resid(a, l, e)
        bad = resid > float(residual_tol)
        can_fix_l = bad & a.notna() & e.notna()
        l = l.where(~can_fix_l, a - e)

        a = a.where(a.notna(), l + e)

    return a, l, e


def _extract_metric_series(
    companyfacts: dict[str, Any],
    metric_name: str,
    spec: MetricSpec,
    min_date: pd.Timestamp,
) -> pd.Series:
    candidates: list[tuple[pd.Series, int]] = []
    history_floor = pd.Timestamp(min_date)
    if spec.is_flow and metric_name not in {"EPS", "Diluted EPS", "Diluted Shares", "Basic Shares"}:
        history_floor = history_floor - pd.Timedelta(days=FLOW_DERIVATION_LOOKBACK_DAYS)
    for tag_rank, tag in enumerate(spec.tags):
        namespace, name = _split_tag(tag)
        unit_records = _iter_unit_records(companyfacts, namespace, name, spec.preferred_units)
        if not unit_records:
            continue
        for unit, rows in unit_records:
            frame = _fact_records_to_frame(rows)
            if frame.empty:
                continue
            frame = frame.loc[frame["quarter_end"] >= history_floor].copy()
            if frame.empty:
                continue

            if metric_name in {"EPS", "Diluted EPS", "Diluted Shares", "Basic Shares"}:
                # Do not derive per-share / weighted-average-share metrics from cumulative deltas.
                # Use direct quarter points only; fallback is handled downstream (NI/shares or Shares).
                candidate = _pick_direct_quarter_values(frame).sort_index()
                if candidate.empty:
                    continue
            elif spec.is_flow:
                quarter = _pick_flow_quarter_values(frame)
                if quarter.empty:
                    continue
                annual = _pick_flow_annual_values(frame)
                candidate = _fill_q4_from_annual(quarter, annual).sort_index()
            else:
                candidate = _pick_stock_quarter_values(frame).sort_index()
                if candidate.empty:
                    continue

            scale = _unit_multiplier(unit, spec.preferred_units)
            candidate = pd.to_numeric(candidate, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if np.isfinite(scale) and scale != 1.0:
                candidate = candidate * float(scale)

            candidate = candidate.loc[pd.DatetimeIndex(candidate.index) >= pd.Timestamp(min_date)]
            candidate = candidate[~candidate.index.duplicated(keep="last")].sort_index()
            if candidate.notna().sum() <= 0:
                continue
            candidates.append((candidate, tag_rank))

    if not candidates:
        return pd.Series(dtype=float)

    def _score(series: pd.Series) -> tuple[int, int, int]:
        valid = pd.to_numeric(series, errors="coerce").dropna()
        if valid.empty:
            return (0, 0, 0)
        idx = pd.DatetimeIndex(valid.index)
        return (int(valid.shape[0]), int(idx.max().value), -int(idx.min().value))

    if metric_name in {"Shares", "D&A", "Debt Long", "Non Current Fin Liabilities", "COGS"}:
        # Pick best candidate per tag, then merge in tag-priority order:
        # For Shares: outstanding shares first, weighted averages only as gap-fill.
        # For D&A: broad depreciation/amortization totals should win over amortization-only tags.
        # For Debt Long / Non Current Fin Liabilities: noncurrent-only tags must win over total-debt
        #   tags (e.g. LongTermDebt covers current+noncurrent and would overcount when used alone).
        # For COGS: insurance-specific tags (PolicyholderBenefitsAndClaimsIncurredNet) must win
        #   over generic CostOfRevenue when they are present (UNH etc.) even if they cover fewer
        #   periods.  Score-based priority would always prefer the wider generic tag.
        best_by_tag: dict[int, pd.Series] = {}
        for series, rank in candidates:
            cur = best_by_tag.get(int(rank))
            if cur is None or _score(series) > _score(cur):
                best_by_tag[int(rank)] = series
        merged: pd.Series | None = None
        for rank in sorted(best_by_tag.keys()):
            s = best_by_tag[rank].sort_index()
            merged = s if merged is None else merged.combine_first(s)
        return (merged if merged is not None else pd.Series(dtype=float)).sort_index()

    ranked = [series for series, _ in sorted(candidates, key=lambda x: (_score(x[0]), -x[1]), reverse=True)]
    union_index = pd.DatetimeIndex([])
    for series in ranked:
        union_index = union_index.union(pd.DatetimeIndex(series.index))
    merged = ranked[0].reindex(union_index)
    for series in ranked[1:]:
        merged = merged.combine_first(series.reindex(union_index))
    return merged.sort_index()


def _extract_custom_metric_series(
    companyfacts: dict[str, Any],
    *,
    tags: tuple[str, ...],
    preferred_units: tuple[str, ...],
    is_flow: bool,
    min_date: pd.Timestamp,
    metric_name: str,
) -> pd.Series:
    return _extract_metric_series(
        companyfacts,
        metric_name=metric_name,
        spec=MetricSpec(tags=tags, preferred_units=preferred_units, is_flow=is_flow),
        min_date=min_date,
    )


def _combine_series_sum(*series_list: pd.Series) -> pd.Series:
    frames = [pd.to_numeric(series, errors="coerce") for series in series_list if series is not None and not series.empty]
    if not frames:
        return pd.Series(dtype=float)
    aligned = pd.concat(frames, axis=1).sort_index()
    return aligned.sum(axis=1, min_count=1).dropna().sort_index()


def _combine_series_max(*series_list: pd.Series) -> pd.Series:
    frames = [pd.to_numeric(series, errors="coerce") for series in series_list if series is not None and not series.empty]
    if not frames:
        return pd.Series(dtype=float)
    aligned = pd.concat(frames, axis=1).sort_index()
    return aligned.max(axis=1, skipna=True).dropna().sort_index()


def _build_refined_debt_short_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
    base_series: pd.Series,
) -> pd.Series:
    """Build Debt Short (current obligations) matching WRDS dlcq definition.

    WRDS dlcq includes: short-term borrowings, commercial paper, current portion
    of LT debt, finance lease current, and operating lease current.

    Strategy:
      1. If DebtCurrent is available, use it (most comprehensive single tag).
      2. Otherwise, sum available components:
         traditional (ShortTermBorrowings + LongTermDebtCurrent + CommercialPaper)
         + lease current (FinanceLeaseLiabilityCurrent or CapitalLeaseObligationsCurrent)
         + OperatingLeaseLiabilityCurrent
    """
    def _ext(tags: tuple[str, ...]) -> pd.Series:
        return _extract_custom_metric_series(
            companyfacts, tags=tags, preferred_units=("USD",),
            is_flow=False, min_date=min_date, metric_name="Debt Short",
        )

    # Most comprehensive single-tag — use when present
    debt_current = _ext(("us-gaap:DebtCurrent",))

    # Individual current-debt components
    stb = _ext(("us-gaap:ShortTermBorrowings",))
    ltd_current = _ext(("us-gaap:LongTermDebtCurrent",))
    commercial_paper = _ext(("us-gaap:CommercialPaper",))
    notes_payable_current = _ext(("us-gaap:NotesPayableCurrent",))

    # Lease current obligations (ASC 842 finance + pre-ASC 842 capital)
    fin_lease_cur = _combine_series_max(
        _ext(("us-gaap:FinanceLeaseLiabilityCurrent",)),
        _ext(("us-gaap:CapitalLeaseObligationsCurrent",)),
    )
    op_lease_cur = _ext(("us-gaap:OperatingLeaseLiabilityCurrent",))

    # Build base traditional short-term debt (sum of components)
    trad_components = _combine_series_sum(stb, ltd_current, commercial_paper, notes_payable_current)

    # If DebtCurrent exists: use it and add operating lease current (if not yet included)
    # DebtCurrent already includes capital/finance leases but NOT ASC-842 operating leases
    if not debt_current.empty:
        # Add operating lease current on top of DebtCurrent for post-ASC-842 parity
        if not op_lease_cur.empty:
            combined = _combine_series_sum(debt_current, op_lease_cur)
        else:
            combined = debt_current
        refined = pd.to_numeric(combined, errors="coerce")
        refined = refined.combine_first(pd.to_numeric(base_series, errors="coerce"))
    else:
        # Sum all known components
        all_components = _combine_series_sum(trad_components, fin_lease_cur, op_lease_cur)
        refined = pd.to_numeric(all_components, errors="coerce")
        refined = refined.combine_first(pd.to_numeric(base_series, errors="coerce"))

    return pd.to_numeric(refined, errors="coerce").dropna().sort_index()


def _build_refined_debt_long_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
    base_series: pd.Series,
) -> pd.Series:
    """Build Debt Long (noncurrent obligations) matching WRDS dlttq definition.

    WRDS dlttq includes: long-term debt noncurrent, finance lease noncurrent,
    and operating lease noncurrent (post-ASC 842).

    Strategy:
      1. Get traditional LT debt noncurrent (LongTermDebtNoncurrent preferred over
         LongTermDebt total).
      2. Add FinanceLeaseLiabilityNoncurrent (or CapitalLeaseObligationsNoncurrent).
      3. Add OperatingLeaseLiabilityNoncurrent.
      4. Guard against double-counting: if LongTermDebtAndCapitalLeaseObligations
         is the source, don't also add CapitalLeaseObligationsNoncurrent.
    """
    def _ext(tags: tuple[str, ...]) -> pd.Series:
        return _extract_custom_metric_series(
            companyfacts, tags=tags, preferred_units=("USD",),
            is_flow=False, min_date=min_date, metric_name="Debt Long",
        )

    # Traditional LT debt (noncurrent-only preferred, total as fallback)
    ltd_noncurrent = _ext(("us-gaap:LongTermDebtNoncurrent",))
    # LongTermDebtAndCapitalLeaseObligations already bundles capital leases
    ltd_and_cap_lease = _ext(("us-gaap:LongTermDebtAndCapitalLeaseObligations",))
    ltd_total = _ext(("us-gaap:LongTermDebt",))  # current+noncurrent — last resort

    # Finance/capital lease noncurrent
    fin_lease_nc = _combine_series_max(
        _ext(("us-gaap:FinanceLeaseLiabilityNoncurrent",)),
        _ext(("us-gaap:CapitalLeaseObligationsNoncurrent",)),
    )
    op_lease_nc = _ext(("us-gaap:OperatingLeaseLiabilityNoncurrent",))

    # Build traditional-debt base using combine_first to cover all periods:
    # 1. LongTermDebtNoncurrent (preferred — noncurrent only, no double-count with Debt Short)
    # 2. LongTermDebtAndCapitalLeaseObligations (already bundles capital leases)
    # 3. LongTermDebt total (current+noncurrent — last resort for periods with no better tag)
    # combine_first ensures we always have the best available value per period
    trad_base = pd.to_numeric(ltd_noncurrent, errors="coerce")
    trad_base = trad_base.combine_first(pd.to_numeric(ltd_and_cap_lease, errors="coerce"))
    trad_base = trad_base.combine_first(pd.to_numeric(ltd_total, errors="coerce"))
    trad_base = trad_base.combine_first(pd.to_numeric(base_series, errors="coerce"))

    # Track whether capital leases are already bundled (to avoid double-counting)
    # LongTermDebtAndCapitalLeaseObligations bundles capital leases; add only op_lease then
    has_ltd_nc = not ltd_noncurrent.empty
    has_cap_lease_bundle = not ltd_and_cap_lease.empty
    # If we have noncurrent-only LT debt OR neither bundle nor noncurrent: add fin_lease
    if has_ltd_nc or not has_cap_lease_bundle:
        refined = _combine_series_sum(trad_base, fin_lease_nc, op_lease_nc)
    else:
        # ltd_and_cap_lease already includes capital leases — only add operating leases
        refined = _combine_series_sum(trad_base, op_lease_nc)

    refined = pd.to_numeric(refined, errors="coerce")
    refined = refined.combine_first(pd.to_numeric(base_series, errors="coerce"))
    return refined.dropna().sort_index()


def _build_refined_cash_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
    base_series: pd.Series,
) -> pd.Series:
    cash_equiv = _extract_custom_metric_series(
        companyfacts,
        tags=("us-gaap:CashAndCashEquivalentsAtCarryingValue",),
        preferred_units=("USD",),
        is_flow=False,
        min_date=min_date,
        metric_name="Cash",
    )
    restricted_bundle = _extract_custom_metric_series(
        companyfacts,
        tags=("us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",),
        preferred_units=("USD",),
        is_flow=False,
        min_date=min_date,
        metric_name="Cash",
    )
    direct_cash_plus_sti = _extract_custom_metric_series(
        companyfacts,
        tags=("us-gaap:CashCashEquivalentsAndShortTermInvestments",),
        preferred_units=("USD",),
        is_flow=False,
        min_date=min_date,
        metric_name="Cash",
    )
    short_term_investments = _extract_custom_metric_series(
        companyfacts,
        tags=("us-gaap:ShortTermInvestments",),
        preferred_units=("USD",),
        is_flow=False,
        min_date=min_date,
        metric_name="Cash",
    )
    segregated_cash = _combine_series_max(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:CashAndSecuritiesSegregatedUnderFederalAndOtherRegulations",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="Cash",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:CashAndSecuritiesSegregatedUnderSecuritiesExchangeCommissionRegulation",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="Cash",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:CashSegregatedUnderOtherRegulations",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="Cash",
        ),
    )
    restricted_current = _extract_custom_metric_series(
        companyfacts,
        tags=("us-gaap:RestrictedCashCurrent",),
        preferred_units=("USD",),
        is_flow=False,
        min_date=min_date,
        metric_name="Cash",
    )
    restricted_noncurrent = _extract_custom_metric_series(
        companyfacts,
        tags=("us-gaap:RestrictedCashNoncurrent",),
        preferred_units=("USD",),
        is_flow=False,
        min_date=min_date,
        metric_name="Cash",
    )
    regulated_addon = _combine_series_max(
        segregated_cash,
        _combine_series_sum(restricted_current, restricted_noncurrent),
    )
    plain_cash = _extract_custom_metric_series(
        companyfacts,
        tags=("us-gaap:Cash",),
        preferred_units=("USD",),
        is_flow=False,
        min_date=min_date,
        metric_name="Cash",
    )
    general_cash = _combine_series_max(
        pd.to_numeric(base_series, errors="coerce"),
        cash_equiv,
        restricted_bundle,
        direct_cash_plus_sti,
        _combine_series_sum(cash_equiv, short_term_investments),
        plain_cash,
    )
    regulated_cash = _combine_series_max(
        pd.to_numeric(base_series, errors="coerce"),
        restricted_bundle,
        direct_cash_plus_sti,
        _combine_series_sum(cash_equiv, short_term_investments),
        _combine_series_sum(cash_equiv, regulated_addon),
        plain_cash,
    )
    regulated_flag = pd.Series(dtype=bool)
    for candidate in (restricted_bundle, regulated_addon):
        if candidate is None or candidate.empty:
            continue
        if regulated_flag.empty:
            regulated_flag = candidate.notna()
        else:
            union_index = regulated_flag.index.union(candidate.index)
            regulated_flag = regulated_flag.reindex(union_index, fill_value=False) | candidate.reindex(union_index).notna()
    if regulated_flag.empty:
        return general_cash
    union_index = general_cash.index.union(regulated_cash.index).union(regulated_flag.index)
    refined = general_cash.reindex(union_index)
    refined = refined.where(~regulated_flag.reindex(union_index, fill_value=False), regulated_cash.reindex(union_index))
    return pd.to_numeric(refined, errors="coerce").dropna().sort_index()


def _extract_operating_expense_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
) -> pd.Series:
    return _normalize_expense_series(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:OperatingExpenses", "us-gaap:OperatingCostsAndExpenses"),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="Operating Expenses",
        )
    )


def _extract_total_cost_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
) -> pd.Series:
    """Total costs for Operating Income reconstruction only (nature-based reporters).

    Unlike _extract_operating_expense_series (used for COGS proxy too), this
    function includes CostsAndExpenses which captures ALL costs on the income
    statement (COGS+SGA+everything) for nature-based reporters like CVX, XOM.
    OperatingExpenses / OperatingCostsAndExpenses take priority so function-based
    reporters are unaffected.
    """
    return _normalize_expense_series(
        _extract_custom_metric_series(
            companyfacts,
            tags=(
                "us-gaap:OperatingExpenses",
                "us-gaap:CostsAndExpenses",
                "us-gaap:OperatingCostsAndExpenses",
            ),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="Total Costs",
        )
    )


def _build_refined_cogs_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
    base_series: pd.Series,
    gross_profit_series: pd.Series,
    revenue_series: pd.Series,
) -> pd.Series:
    direct_cogs = _normalize_expense_series(base_series)
    gross_profit = pd.to_numeric(gross_profit_series, errors="coerce")
    revenue = pd.to_numeric(revenue_series, errors="coerce")
    operating_expenses = _normalize_expense_series(_extract_operating_expense_series(companyfacts, min_date=min_date))
    if operating_expenses.empty:
        return direct_cogs.dropna().sort_index()
    union_index = direct_cogs.index.union(gross_profit.index).union(revenue.index).union(operating_expenses.index)
    direct_aligned = direct_cogs.reindex(union_index)
    gross_aligned = gross_profit.reindex(union_index)
    revenue_aligned = revenue.reindex(union_index)
    op_aligned = operating_expenses.reindex(union_index)
    use_operating_proxy = direct_aligned.isna() & gross_aligned.isna() & op_aligned.notna()
    tiny_direct_cogs = direct_aligned.isna() | (
        revenue_aligned.notna()
        & direct_aligned.notna()
        & direct_aligned.le(revenue_aligned.abs() * 0.10)
    )
    negative_margin_proxy = (
        op_aligned.notna()
        & revenue_aligned.notna()
        & gross_aligned.notna()
        & gross_aligned.gt(0)
        & op_aligned.gt(revenue_aligned.abs() * 1.02)
        & tiny_direct_cogs
    )
    refined = direct_aligned.where(~(use_operating_proxy | negative_margin_proxy), op_aligned)
    return _normalize_expense_series(refined).dropna().sort_index()


def _build_refined_operating_income_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
    base_series: pd.Series,
    revenue_series: pd.Series,
    gross_profit_series: pd.Series,
    sga_series: pd.Series,
    rd_series: pd.Series,
    finance_like: bool = False,
) -> pd.Series:
    direct_operating_income = pd.to_numeric(base_series, errors="coerce")
    revenue = pd.to_numeric(revenue_series, errors="coerce")
    gross_profit = pd.to_numeric(gross_profit_series, errors="coerce")
    sga = _normalize_expense_series(sga_series)
    rd = _normalize_expense_series(rd_series)
    # Pass 1 uses _extract_total_cost_series (includes CostsAndExpenses for nature-based
    # reporters like CVX/XOM).  Pass 2 must use the NARROW opex series (OperatingExpenses /
    # OperatingCostsAndExpenses only) so that GP − CostsAndExpenses never triggers a
    # spurious sign_conflict when CostsAndExpenses > GrossProfit (e.g. MMM pre-2024).
    total_costs = _normalize_expense_series(_extract_total_cost_series(companyfacts, min_date=min_date))
    narrow_opex = _normalize_expense_series(_extract_operating_expense_series(companyfacts, min_date=min_date))
    if revenue.empty and gross_profit.empty and sga.empty and rd.empty and total_costs.empty:
        return direct_operating_income.dropna().sort_index()
    union_index = (
        direct_operating_income.index
        .union(revenue.index)
        .union(gross_profit.index)
        .union(sga.index)
        .union(rd.index)
        .union(total_costs.index)
        .union(narrow_opex.index)
    )
    direct_aligned = direct_operating_income.reindex(union_index)
    revenue_aligned = revenue.reindex(union_index)
    gross_aligned = gross_profit.reindex(union_index)
    sga_aligned = sga.reindex(union_index)
    rd_aligned = rd.reindex(union_index)
    op_aligned = total_costs.reindex(union_index)
    narrow_op_aligned = narrow_opex.reindex(union_index)

    # Pass 1: Revenue − TotalCosts (best for nature-based reporters without OperatingIncomeLoss)
    reconstructed_from_total_opex = revenue_aligned - op_aligned
    reconstructed_from_total_opex = reconstructed_from_total_opex.where(revenue_aligned.notna() & op_aligned.notna())

    # Guard: Operating Income cannot exceed Gross Profit (physically impossible).
    # When _extract_total_cost_series returns a "narrow" OperatingExpenses tag (SGA-type
    # below-GP costs only), Pass 1 gives OI = Revenue − NarrowOpEx which is implausibly
    # large (e.g. GPC, COP where OperatingExpenses ≠ total costs).  Null out Pass 1 when
    # result exceeds GrossProfit by more than 2%.  Nature-based P&L reporters without
    # GrossProfit are unaffected (gross_aligned.isna()).
    _oi_exceeds_gp = (
        gross_aligned.notna()
        & reconstructed_from_total_opex.notna()
        & (reconstructed_from_total_opex > gross_aligned * 1.02)
    )
    reconstructed_from_total_opex = reconstructed_from_total_opex.where(~_oi_exceeds_gp)

    # Pass 2: GrossProfit − NarrowOpEx (SGA-type only; avoids GP−CostsAndExpenses < 0 bug)
    reconstructed_from_gross_opex = gross_aligned - narrow_op_aligned
    reconstructed_from_gross_opex = reconstructed_from_gross_opex.where(gross_aligned.notna() & narrow_op_aligned.notna())

    # Pre-revenue issuers often report only operating expenses; use a zero-revenue
    # proxy only when standalone SG&A is absent or broadly tracks the same scale.
    prerevenue_proxy_ok = (
        revenue_aligned.isna()
        & gross_aligned.isna()
        & direct_aligned.notna()
        & op_aligned.notna()
    )
    sga_scale_guard = sga_aligned.isna() | (
        op_aligned.abs() <= (sga_aligned.abs() * 2.5)
    )
    prerevenue_sign_proxy = (-op_aligned).where(
        prerevenue_proxy_ok & sga_scale_guard
    )
    reconstructed_from_total_opex = reconstructed_from_total_opex.combine_first(prerevenue_sign_proxy)

    # GP - SGA - R&D reconstruction.  When R&D is available (pharma, tech without direct
    # OperatingIncomeLoss tag, e.g. IBM, PFE, MRK), include it so the reconstruction
    # matches the Compustat oiadpq definition.  Fall back to GP - SGA when R&D is absent.
    reconstructed_from_gross_with_rd = (gross_aligned - sga_aligned - rd_aligned).where(
        gross_aligned.notna() & sga_aligned.notna() & rd_aligned.notna()
    )
    reconstructed_from_gross_no_rd = (gross_aligned - sga_aligned).where(
        gross_aligned.notna() & sga_aligned.notna() & rd_aligned.isna()
    )
    reconstructed_from_gross = reconstructed_from_gross_no_rd.combine_first(reconstructed_from_gross_with_rd)

    if finance_like:
        refined = direct_aligned.copy()
        use_total = refined.isna() & reconstructed_from_total_opex.notna()
        refined = refined.where(~use_total, reconstructed_from_total_opex)
        use_gross_opex = refined.isna() & reconstructed_from_gross_opex.notna()
        refined = refined.where(~use_gross_opex, reconstructed_from_gross_opex)
        use_gross = refined.isna() & reconstructed_from_gross.notna()
        refined = refined.where(~use_gross, reconstructed_from_gross)
        return pd.to_numeric(refined, errors="coerce").dropna().sort_index()

    refined = _reconcile_signed_reconstruction(direct_aligned, reconstructed_from_total_opex)
    refined = _reconcile_signed_reconstruction(refined, reconstructed_from_gross_opex)
    refined = _reconcile_signed_reconstruction(refined, reconstructed_from_gross)
    return pd.to_numeric(refined, errors="coerce").dropna().sort_index()


def _build_refined_pretax_income_series(
    *,
    base_series: pd.Series,
    net_income_series: pd.Series,
    tax_series: pd.Series,
) -> pd.Series:
    direct_pretax = pd.to_numeric(base_series, errors="coerce")
    net_income = pd.to_numeric(net_income_series, errors="coerce")
    tax = pd.to_numeric(tax_series, errors="coerce")
    if net_income.empty or tax.empty:
        return direct_pretax.dropna().sort_index()
    union_index = direct_pretax.index.union(net_income.index).union(tax.index)
    direct_aligned = direct_pretax.reindex(union_index)
    net_aligned = net_income.reindex(union_index)
    tax_aligned = tax.reindex(union_index)
    reconstructed = net_aligned + tax_aligned
    use_reconstructed = direct_aligned.isna() & net_aligned.notna() & tax_aligned.notna()
    refined = direct_aligned.where(~use_reconstructed, reconstructed)
    return pd.to_numeric(refined, errors="coerce").dropna().sort_index()


def _build_refined_net_income_series(
    *,
    base_series: pd.Series,
    pretax_series: pd.Series,
    tax_series: pd.Series,
    operating_income_series: pd.Series,
    revenue_series: pd.Series,
) -> pd.Series:
    direct_net_income = pd.to_numeric(base_series, errors="coerce")
    pretax = pd.to_numeric(pretax_series, errors="coerce")
    tax = pd.to_numeric(tax_series, errors="coerce")
    operating_income = pd.to_numeric(operating_income_series, errors="coerce")
    revenue = pd.to_numeric(revenue_series, errors="coerce")
    if pretax.empty and tax.empty and operating_income.empty:
        return direct_net_income.dropna().sort_index()

    union_index = (
        direct_net_income.index
        .union(pretax.index)
        .union(tax.index)
        .union(operating_income.index)
        .union(revenue.index)
    )
    direct_aligned = direct_net_income.reindex(union_index)
    pretax_aligned = pretax.reindex(union_index)
    tax_aligned = tax.reindex(union_index)
    operating_income_aligned = operating_income.reindex(union_index)
    revenue_aligned = revenue.reindex(union_index)

    reconstructed_from_pretax = (pretax_aligned - tax_aligned).where(
        pretax_aligned.notna() & tax_aligned.notna()
    )
    refined = _reconcile_signed_reconstruction(direct_aligned, reconstructed_from_pretax)

    # Pre-revenue biotech-style issuers sometimes expose NetIncomeLoss with the
    # wrong sign while OperatingIncomeLoss has the correct loss direction.
    same_scale_as_op = (
        direct_aligned.notna()
        & operating_income_aligned.notna()
        & (
            (direct_aligned.abs() - operating_income_aligned.abs()).abs()
            <= np.maximum(direct_aligned.abs(), operating_income_aligned.abs()) * 0.15
        )
    )
    prerevenue_op_proxy = (
        direct_aligned.notna()
        & operating_income_aligned.notna()
        & pretax_aligned.isna()
        & tax_aligned.isna()
        & (revenue_aligned.isna() | revenue_aligned.abs().le(1_000_000.0))
        & (np.sign(direct_aligned) != np.sign(operating_income_aligned))
        & same_scale_as_op
    )
    refined = refined.where(~prerevenue_op_proxy, operating_income_aligned)
    return pd.to_numeric(refined, errors="coerce").dropna().sort_index()


def _build_refined_deferred_revenue_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
    base_series: pd.Series,
) -> pd.Series:
    direct_total = _combine_series_max(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:ContractWithCustomerLiability",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="Deferred Revenue",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:DeferredRevenue",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="Deferred Revenue",
        ),
    )
    current_only = _combine_series_max(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:ContractWithCustomerLiabilityCurrent",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="Deferred Revenue",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:DeferredRevenueCurrent", "us-gaap:DeferredRevenueAndCreditsCurrent"),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="Deferred Revenue",
        ),
    )
    refined = pd.to_numeric(direct_total, errors="coerce")
    if refined.empty:
        refined = pd.to_numeric(base_series, errors="coerce")
    else:
        refined = refined.combine_first(pd.to_numeric(base_series, errors="coerce"))
    refined = refined.combine_first(pd.to_numeric(current_only, errors="coerce"))
    return pd.to_numeric(refined, errors="coerce").dropna().sort_index()


def _build_refined_aoci_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
    base_series: pd.Series,
) -> pd.Series:
    direct_total = pd.to_numeric(base_series, errors="coerce")
    derivative_component = _extract_custom_metric_series(
        companyfacts,
        tags=("us-gaap:AccumulatedOtherComprehensiveIncomeLossCumulativeChangesInNetGainLossFromCashFlowHedgesEffectNetOfTax",),
        preferred_units=("USD",),
        is_flow=False,
        min_date=min_date,
        metric_name="AOCI",
    )
    securities_component = _extract_custom_metric_series(
        companyfacts,
        tags=("us-gaap:AccumulatedOtherComprehensiveIncomeLossAvailableForSaleSecuritiesAdjustmentNetOfTax",),
        preferred_units=("USD",),
        is_flow=False,
        min_date=min_date,
        metric_name="AOCI",
    )
    pension_component = _combine_series_max(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:AccumulatedOtherComprehensiveIncomeLossDefinedBenefitPensionAndOtherPostretirementPlansNetOfTax",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="AOCI",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:AccumulatedOtherComprehensiveIncomeMinimumPensionLiabilityNetAdjustment",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="AOCI",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:DefinedBenefitPlanAccumulatedOtherComprehensiveIncomeNetGainsLossesAfterTax",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="AOCI",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:DefinedBenefitPlanAccumulatedOtherComprehensiveIncomeLossAfterTax",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="AOCI",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:DefinedBenefitPlanAccumulatedOtherComprehensiveIncomeNetPriorServiceCostCreditAfterTax",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="AOCI",
        ),
    )
    other_component = _combine_series_max(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:AccumulatedOtherComprehensiveIncomeLossForeignCurrencyTranslationAdjustmentNetOfTax",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="AOCI",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:AccumulatedOtherComprehensiveIncomeLossForeignCurrencyTransactionAndTranslationAdjustmentNetOfTax",),
            preferred_units=("USD",),
            is_flow=False,
            min_date=min_date,
            metric_name="AOCI",
        ),
    )
    component_sum = _combine_series_sum(
        derivative_component,
        securities_component,
        pension_component,
        other_component,
    )
    # Before using component_sum as a fallback, guard against sign conflicts.
    # AOCI can be either positive or negative; when direct_total has a consistent sign
    # in nearby periods, use that to validate (or reject) the component-derived value.
    # This prevents sign flips caused by missing or sign-reversed individual components
    # (most commonly pension adjustment components).
    direct = pd.to_numeric(direct_total, errors="coerce")
    comp = pd.to_numeric(component_sum, errors="coerce")
    if direct.notna().sum() >= 2 and comp.notna().any():
        # Determine dominant sign of direct_total
        direct_signs = np.sign(direct.dropna())
        dominant_sign = int(np.sign(float(direct_signs.mean()))) if direct_signs.size > 0 else 0
        if dominant_sign != 0:
            # Zero out component_sum values that conflict with dominant direct sign
            comp = comp.where(np.sign(comp) == dominant_sign, np.nan)
    refined = direct.combine_first(comp)
    return pd.to_numeric(refined, errors="coerce").dropna().sort_index()


def _build_refined_da_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
    base_series: pd.Series,
) -> pd.Series:
    """Refine D&A series with component-sum recovery and annual-to-quarterly approximation.

    Two-pass approach:
    1. Component sum: Many tech/acquisition-heavy issuers (IBM, CSCO, ORCL, …) report
       Depreciation (PP&E) and AmortizationOfIntangibleAssets separately rather than a
       combined DepreciationDepletionAndAmortization total.  In those cases the main
       extraction only captures PP&E depreciation and misses acquired-intangible
       amortization.  We sum the two components and use the result whenever it is
       meaningfully larger than the main extraction.

    2. Annual fallback: For issuers that report D&A only in annual 10-K filings, we
       distribute the annual value as annual/4 across the four fiscal quarters.

    Coverage threshold for annual fallback: < 40% of periods covered.
    """
    refined = _normalize_expense_series(base_series)

    # ── Pass 0: DDA vs DepAndAm — take the larger value per period ─────────────
    # DepreciationDepletionAndAmortization (DDA, rank 0 in MetricSpec) wins in tag-rank
    # merge and suppresses DepreciationAndAmortization (DepAndAm, rank 1) even when
    # DepAndAm is larger.  For segment-based reporters like PCAR, DDA only covers the
    # manufacturing segment while DepAndAm aggregates all segments (incl. financial
    # services).  When DepAndAm > DDA we override with DepAndAm.
    # This is safe for oil/gas companies where DDA includes depletion (DDA ≥ DepAndAm),
    # since the guard fires only when DepAndAm is *larger*.
    _dep_and_am_p0 = _normalize_expense_series(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:DepreciationAndAmortization",),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="D&A",
        )
    )
    if not _dep_and_am_p0.empty and not refined.empty:
        _u0 = refined.index.union(_dep_and_am_p0.index)
        _r0 = pd.to_numeric(refined.reindex(_u0), errors="coerce")
        _d0 = pd.to_numeric(_dep_and_am_p0.reindex(_u0), errors="coerce")
        _prefer_da = _d0.notna() & (_r0.isna() | (_d0 > _r0 * 1.10))
        _r0 = _r0.where(~_prefer_da, _d0)
        refined = _normalize_expense_series(_r0).dropna().sort_index()

    # ── Pass 0b: CostOfGoodsAndServicesSoldDepreciationAndAmortization ─────────
    # Equipment rental companies (e.g. URI) embed virtually ALL depreciation in
    # cost-of-revenues and do not file a top-level DDA tag.  Their
    # CostOfGoodsAndServicesSoldDepreciationAndAmortization captures the full fleet
    # depreciation ($650-800 M/Q for URI) which far exceeds the partial DDA ($310 M).
    # Use this tag whenever it is substantially larger than the current refined value.
    # For manufacturing companies where this tag captures only COGS-embedded D&A
    # (a subset of total D&A), DDA/DepAndAm is already larger — the guard is harmless.
    _cogs_da_p0b = _normalize_expense_series(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:CostOfGoodsAndServicesSoldDepreciationAndAmortization",),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="D&A",
        )
    )
    if not _cogs_da_p0b.empty:
        _u0b = refined.index.union(_cogs_da_p0b.index)
        _r0b = pd.to_numeric(refined.reindex(_u0b), errors="coerce")
        _c0b = pd.to_numeric(_cogs_da_p0b.reindex(_u0b), errors="coerce")
        # Use CostOfGoodsDA only when it is substantially larger (>2x) than the
        # current refined value.  Equipment rental companies (URI) have fleet D&A
        # exclusively in CostOfGoodsDA (ratio 5-7x), so they correctly switch.
        # Retailers/utilities (HD, LOW) have CostOfGoodsDA only ~20% above their DDA
        # because it includes operating-lease ROU amortization — keeping DDA is correct.
        _prefer_cogs_da = _c0b.notna() & (_r0b.isna() | (_c0b > _r0b * 2.0))
        _r0b = _r0b.where(~_prefer_cogs_da, _c0b)
        refined = _normalize_expense_series(_r0b).dropna().sort_index()

        # Pass 0b-ii: For equipment rental companies (e.g. URI), fleet depreciation is in
        # CostOfGoodsAndServicesSoldDepreciationAndAmortization while corporate/office D&A
        # is in DepreciationAndAmortization.  These are COMPLEMENTARY (not overlapping), so
        # add DepAndAm on top of CostOfGoodsDA when CostOfGoodsDA >> DepAndAm (>2x),
        # which signals the "fleet-dominant" pattern.  For manufacturers where
        # CostOfGoodsDA is a SUBSET of DepAndAm, CostOfGoodsDA ≤ DepAndAm, so the
        # guard prevents double-counting.
        if not _dep_and_am_p0.empty:
            _u0b2 = refined.index.union(_dep_and_am_p0.index)
            _r0b2 = pd.to_numeric(refined.reindex(_u0b2), errors="coerce")
            _d0b2 = pd.to_numeric(_dep_and_am_p0.reindex(_u0b2), errors="coerce")
            # Only sum where CostOfGoodsDA was dominant (>2x DepAndAm) and DepAndAm available
            _add_mask = (
                _r0b2.notna()
                & _d0b2.notna()
                & (_r0b2 > _d0b2 * 2.0)
            )
            _r0b2 = _r0b2.where(~_add_mask, _r0b2 + _d0b2)
            refined = _normalize_expense_series(_r0b2).dropna().sort_index()

    # ── Pass 1: component-sum recovery ────────────────────────────────────────
    # Extract PP&E-only Depreciation and AmortizationOfIntangibleAssets separately,
    # then combine.  When the sum is > 105% of the main extraction for any period,
    # prefer the sum — this indicates the main tag excludes intangible amortization.
    #
    # IMPORTANT: Only use tags that represent PP&E depreciation ONLY (not combined
    # DepreciationAndAmortization or DepreciationDepletionAndAmortization tags).
    # Those combined tags already include intangibles, so adding AmortizationOfIntangibles
    # on top would create double-counting.
    # - us-gaap:Depreciation = PP&E depreciation only
    # - us-gaap:OtherDepreciationAndAmortization = catch-all PP&E D&A (used by AMD, etc.)
    # - us-gaap:DepreciationNonproduction = bank/financial premises depreciation (TFC, USB, MTB)
    _depr_series = _normalize_expense_series(
        _extract_custom_metric_series(
            companyfacts,
            tags=(
                "us-gaap:Depreciation",
                "us-gaap:OtherDepreciationAndAmortization",
                "us-gaap:DepreciationNonproduction",
            ),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="D&A",
        )
    )
    _amort_series = _normalize_expense_series(
        _extract_custom_metric_series(
            companyfacts,
            tags=(
                "us-gaap:AmortizationOfIntangibleAssets",
                "us-gaap:AmortizationOfAcquiredIntangibleAssets",
            ),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="D&A",
        )
    )
    if not _depr_series.empty and not _amort_series.empty:
        component_sum = _combine_series_sum(_depr_series, _amort_series)
        if not component_sum.empty:
            union_idx = refined.index.union(component_sum.index)
            r_al = pd.to_numeric(refined.reindex(union_idx), errors="coerce")
            c_al = pd.to_numeric(component_sum.reindex(union_idx), errors="coerce")
            # Use component sum where it exceeds the main series by >5% (intangibles excluded)
            prefer_sum = c_al.notna() & (r_al.isna() | (c_al > r_al * 1.05))
            r_al = r_al.where(~prefer_sum, c_al)
            refined = _normalize_expense_series(r_al).dropna().sort_index()

    # ── Pass 1b: annual amortization augmentation ──────────────────────────────
    # For issuers that report Depreciation (PP&E) quarterly but only report
    # AmortizationOfIntangibleAssets annually (e.g. MRK post-large acquisition),
    # we distribute the annual amortization as annual/4 and add to quarterly
    # depreciation.  Only applied when quarterly amort is unavailable.
    elif not _depr_series.empty and _amort_series.empty:
        history_floor_cs = pd.Timestamp(min_date) - pd.Timedelta(days=FLOW_DERIVATION_LOOKBACK_DAYS)
        _annual_amort_map: pd.Series = pd.Series(dtype=float)
        for _cs_tag in (
            "us-gaap:AmortizationOfIntangibleAssets",
            "us-gaap:AmortizationOfAcquiredIntangibleAssets",
        ):
            _cs_ns, _cs_nm = _split_tag(_cs_tag)
            for _cs_unit, _cs_rows in _iter_unit_records(companyfacts, _cs_ns, _cs_nm, ("USD",)):
                _cs_frame = _fact_records_to_frame(_cs_rows)
                if _cs_frame.empty:
                    continue
                _cs_frame = _cs_frame.loc[_cs_frame["quarter_end"] >= history_floor_cs].copy()
                _cs_ann = _pick_flow_annual_values(_cs_frame)
                if _cs_ann.empty:
                    continue
                _cs_scale = _unit_multiplier(_cs_unit, ("USD",))
                _cs_ann = pd.to_numeric(_cs_ann, errors="coerce")
                if np.isfinite(_cs_scale) and _cs_scale != 1.0:
                    _cs_ann = _cs_ann * float(_cs_scale)
                _cs_ann = _cs_ann.loc[pd.DatetimeIndex(_cs_ann.index) >= pd.Timestamp(min_date)]
                _annual_amort_map = _annual_amort_map.combine_first(_cs_ann)

        if not _annual_amort_map.empty:
            _amort_proxy_q: dict[pd.Timestamp, float] = {}
            for _ye_raw, _av in _annual_amort_map.items():
                _ye = pd.Timestamp(_ye_raw)
                _av_f = float(_av)
                if not np.isfinite(_av_f) or _av_f <= 0:
                    continue
                _qest = _av_f / 4.0
                for _mb in (9, 6, 3, 0):
                    _approx = _ye - pd.DateOffset(months=_mb)
                    _snapped = _quarter_end(pd.Timestamp(_approx))
                    if _snapped < pd.Timestamp(min_date):
                        continue
                    if _snapped not in _amort_proxy_q:
                        _amort_proxy_q[_snapped] = _qest

            if _amort_proxy_q:
                _amort_proxy = _normalize_expense_series(
                    pd.Series(_amort_proxy_q, dtype=float)
                    .pipe(lambda s: s[~s.index.duplicated(keep="last")])
                    .sort_index()
                )
                if not _amort_proxy.empty:
                    # Require BOTH components present (depr AND amort proxy).
                    # Using min_count=2 ensures NaN+value stays NaN — avoids
                    # filling periods where only amort proxy is available.
                    _depr_al = pd.to_numeric(_depr_series, errors="coerce")
                    _amort_al = pd.to_numeric(_amort_proxy, errors="coerce")
                    _cs1b_df = pd.concat([_depr_al, _amort_al], axis=1).sort_index()
                    _cs1b = _cs1b_df.sum(axis=1, min_count=2).dropna().sort_index()
                    if not _cs1b.empty:
                        _u1b = refined.index.union(_cs1b.index)
                        _r1b = pd.to_numeric(refined.reindex(_u1b), errors="coerce")
                        _c1b = pd.to_numeric(_cs1b.reindex(_u1b), errors="coerce")
                        _pref1b = _c1b.notna() & (_r1b.isna() | (_c1b > _r1b * 1.05))
                        _r1b = _r1b.where(~_pref1b, _c1b)
                        refined = _normalize_expense_series(_r1b).dropna().sort_index()

    # ── Pass 1c: CapitalizedComputerSoftwareAmortization supplement ───────────
    # Some issuers (Mettler-Toledo MTD, Intuit INTU, etc.) capitalise internal-use
    # software and report its amortization as a *separate* cash-flow line item tagged
    # us-gaap:CapitalizedComputerSoftwareAmortization1.  In XBRL filings this tag
    # always represents an additive (not overlapping) component relative to DDA/
    # DepAndAm — a company would not separately tag something already included in DDA.
    # Add it to refined when the ratio is meaningful (>5% of current refined), which
    # filters out noise without capping on the high side (a pure-software company
    # could have large CapSoftAmort relative to PP&E depreciation).
    _cap_soft_p1c = _normalize_expense_series(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:CapitalizedComputerSoftwareAmortization1",),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="D&A",
        )
    )
    if not _cap_soft_p1c.empty and not refined.empty:
        _u1c = refined.index.union(_cap_soft_p1c.index)
        _r1c = pd.to_numeric(refined.reindex(_u1c), errors="coerce")
        _c1c = pd.to_numeric(_cap_soft_p1c.reindex(_u1c), errors="coerce")
        # Only add when CapSoftAmort is meaningful (>5%) but not dominant (≤70%).
        # If CapSoftAmort ≈ refined (>70%), it is likely already embedded in DDA or
        # AmortOfIntangibleAssets (e.g. TFC/Truist where CapSoftAmort ≈ component-sum).
        _add1c = _r1c.notna() & _c1c.notna() & (_c1c > _r1c * 0.05) & (_c1c <= _r1c * 0.70)
        _r1c = _r1c.where(~_add1c, _r1c + _c1c)
        refined = _normalize_expense_series(_r1c).dropna().sort_index()

    covered = int(refined.notna().sum())
    total = max(len(refined.index), 1)
    if covered / total >= 0.40:
        # Good quarterly coverage — no approximation needed
        return refined.dropna().sort_index()

    # Collect annual D&A values from each tag using the history floor (same as flow extraction)
    history_floor = pd.Timestamp(min_date) - pd.Timedelta(days=FLOW_DERIVATION_LOOKBACK_DAYS)
    annual_map: pd.Series = pd.Series(dtype=float)
    for tag in METRIC_SPECS["D&A"].tags:
        ns, nm = _split_tag(tag)
        for unit, rows in _iter_unit_records(companyfacts, ns, nm, ("USD",)):
            frame = _fact_records_to_frame(rows)
            if frame.empty:
                continue
            frame = frame.loc[frame["quarter_end"] >= history_floor].copy()
            ann = _pick_flow_annual_values(frame)
            if ann.empty:
                continue
            scale = _unit_multiplier(unit, ("USD",))
            ann = pd.to_numeric(ann, errors="coerce")
            if np.isfinite(scale) and scale != 1.0:
                ann = ann * float(scale)
            ann = ann.loc[pd.DatetimeIndex(ann.index) >= pd.Timestamp(min_date)]
            annual_map = annual_map.combine_first(ann)

    if annual_map.empty:
        return refined.dropna().sort_index()

    # For each fiscal year-end with an annual D&A value, estimate the four quarterly periods
    # as annual / 4.  We approximate fiscal quarter-end dates by stepping back 9, 6, and 3
    # months from the fiscal year-end; snap each to a true quarter-end via _quarter_end().
    quarterly_from_annual: dict[pd.Timestamp, float] = {}
    for year_end_raw, annual_val in annual_map.items():
        year_end = pd.Timestamp(year_end_raw)
        av = float(annual_val)
        if not np.isfinite(av) or av <= 0:
            continue
        quarterly_estimate = av / 4.0
        for months_back in (9, 6, 3, 0):
            # Subtract months to approximate each quarterly period-end, then snap
            approx_end = year_end - pd.DateOffset(months=months_back)
            snapped = _quarter_end(pd.Timestamp(approx_end))
            if snapped < pd.Timestamp(min_date):
                continue
            # Only fill genuinely missing slots — do not overwrite real quarterly data
            existing = refined.get(snapped, np.nan)
            if not np.isfinite(float(existing) if existing is not None else np.nan):
                quarterly_from_annual[snapped] = quarterly_estimate

    if not quarterly_from_annual:
        return refined.dropna().sort_index()

    approx_series = pd.Series(quarterly_from_annual, dtype=float)
    approx_series = approx_series[~approx_series.index.duplicated(keep="last")].sort_index()
    result = refined.combine_first(approx_series)
    return _normalize_expense_series(result).dropna().sort_index()


def _build_refined_sga_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
    base_series: pd.Series,
) -> pd.Series:
    direct_sga = _normalize_expense_series(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:SellingGeneralAndAdministrativeExpense",),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="SG&A",
        )
    )
    selling_marketing = _normalize_expense_series(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:SellingAndMarketingExpense",),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="SG&A",
        )
    )
    general_admin = _normalize_expense_series(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:GeneralAndAdministrativeExpense",),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="SG&A",
        )
    )
    split_sga = _combine_series_sum(selling_marketing, general_admin)
    # Note: R&D is intentionally NOT added to SGA fallback.
    # WRDS xsgaq includes R&D for many companies, but our "SG&A" metric represents
    # only selling, general & administrative expenses — R&D is tracked separately.
    refined = direct_sga.combine_first(_normalize_expense_series(base_series))
    refined = refined.combine_first(split_sga)

    return _normalize_expense_series(refined).dropna().sort_index()


def _build_refined_capex_series(
    companyfacts: dict[str, Any],
    *,
    min_date: pd.Timestamp,
    base_series: pd.Series,
) -> pd.Series:
    ppe_capex = _extract_custom_metric_series(
        companyfacts,
        tags=("us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",),
        preferred_units=("USD",),
        is_flow=True,
        min_date=min_date,
        metric_name="Capital Expenditure",
    )
    software_capex = _combine_series_max(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:PaymentsForSoftware",),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="Capital Expenditure",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:PaymentsToDevelopSoftware",),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="Capital Expenditure",
        ),
    )
    productive_assets = _combine_series_max(
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:PaymentsToAcquireOtherProductiveAssets",),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="Capital Expenditure",
        ),
        _extract_custom_metric_series(
            companyfacts,
            tags=("us-gaap:PaymentsToAcquireProductiveAssets",),
            preferred_units=("USD",),
            is_flow=True,
            min_date=min_date,
            metric_name="Capital Expenditure",
        ),
    )
    refined = _combine_series_sum(ppe_capex, software_capex, productive_assets)
    refined = refined.combine_first(pd.to_numeric(base_series, errors="coerce"))
    return pd.to_numeric(refined, errors="coerce").dropna().sort_index()


def _load_price_df(ticker: str, market: str = "us") -> pd.DataFrame:
    """Load price DataFrame using DuckDB-first reader."""
    try:
        from market_data.reader import load_price_dataframe
        df, _ = load_price_dataframe(ticker=ticker, market=market)
        return df
    except Exception:
        return pd.DataFrame()


def _load_price_series(ticker: str, market: str = "us") -> pd.Series:
    out = _load_price_df(ticker, market)
    if out.empty:
        return pd.Series(dtype=float)
    close_col = "Adj Close" if "Adj Close" in out.columns else ("Close" if "Close" in out.columns else None)
    if close_col is None:
        return pd.Series(dtype=float)
    series = pd.to_numeric(out[close_col], errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _load_price_splits(ticker: str, market: str = "us") -> pd.Series:
    out = _load_price_df(ticker, market)
    if out.empty or "Stock Splits" not in out.columns:
        return pd.Series(dtype=float)
    splits = pd.to_numeric(out["Stock Splits"], errors="coerce")
    splits = splits.replace([np.inf, -np.inf], np.nan).dropna()
    splits = splits[(splits > 0.0) & (~np.isclose(splits, 1.0))]
    return splits.sort_index()


def _align_quarter_prices(price_series: pd.Series, quarter_index: pd.DatetimeIndex) -> pd.Series:
    if price_series.empty or quarter_index.empty:
        return pd.Series(np.nan, index=quarter_index, dtype=float)
    idx = pd.DatetimeIndex(quarter_index)
    unique_idx = idx.unique().sort_values()
    union_idx = price_series.index.union(unique_idx)
    aligned_unique = price_series.reindex(union_idx).sort_index().ffill().reindex(unique_idx)
    aligned = pd.Series(aligned_unique.reindex(idx).to_numpy(), index=idx)
    return pd.to_numeric(aligned, errors="coerce")


def _adjust_shares_to_adj_basis(shares: pd.Series, split_series: pd.Series, quarter_index: pd.DatetimeIndex) -> pd.Series:
    out = pd.to_numeric(shares, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.empty or split_series.empty or quarter_index.empty:
        return out

    splits = pd.to_numeric(split_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    splits = splits[(splits > 0.0) & (~np.isclose(splits, 1.0))]
    if splits.empty:
        return out

    split_idx = pd.DatetimeIndex(splits.index).sort_values()
    split_vals = splits.reindex(split_idx).to_numpy(dtype=float)
    suffix = np.ones(len(split_vals) + 1, dtype=float)
    for i in range(len(split_vals) - 1, -1, -1):
        suffix[i] = suffix[i + 1] * split_vals[i]

    factors = np.ones(len(quarter_index), dtype=float)
    for i, qd in enumerate(pd.DatetimeIndex(quarter_index)):
        pos = int(np.searchsorted(split_idx.values, np.datetime64(qd), side="right"))
        factors[i] = suffix[pos]

    factor_series = pd.Series(factors, index=pd.DatetimeIndex(quarter_index), dtype=float)
    return out.reindex(pd.DatetimeIndex(quarter_index)) * factor_series


def _future_split_factor_series(split_series: pd.Series, quarter_index: pd.DatetimeIndex) -> pd.Series:
    if split_series is None or split_series.empty:
        return pd.Series(1.0, index=pd.DatetimeIndex(quarter_index), dtype=float)
    splits = pd.to_numeric(split_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    splits = splits[(splits > 0.0) & (~np.isclose(splits, 1.0))]
    if splits.empty:
        return pd.Series(1.0, index=pd.DatetimeIndex(quarter_index), dtype=float)

    split_idx = pd.DatetimeIndex(splits.index).sort_values()
    if split_idx.tz is not None:
        split_idx = split_idx.tz_convert(None)
    split_vals = splits.reindex(split_idx).to_numpy(dtype=float)

    suffix = np.ones(len(split_vals) + 1, dtype=float)
    for i in range(len(split_vals) - 1, -1, -1):
        suffix[i] = suffix[i + 1] * split_vals[i]

    factors = np.ones(len(quarter_index), dtype=float)
    q_idx = pd.DatetimeIndex(quarter_index)
    for i, qd in enumerate(q_idx):
        pos = int(np.searchsorted(split_idx.values, np.datetime64(pd.Timestamp(qd)), side="right"))
        factors[i] = suffix[pos]
    return pd.Series(factors, index=q_idx, dtype=float)


def _normalize_shares_to_price_basis(
    shares: pd.Series,
    quarter_index: pd.DatetimeIndex,
    split_series: pd.Series | None,
) -> pd.Series:
    out = pd.to_numeric(shares, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.reindex(pd.DatetimeIndex(quarter_index))
    if out.notna().sum() < 3 or split_series is None or split_series.empty:
        return out

    factors = _future_split_factor_series(split_series=split_series, quarter_index=pd.DatetimeIndex(out.index))
    opt_keep = out.copy()
    opt_scaled = out * factors

    valid_pos = [i for i, v in enumerate(out.to_numpy(dtype=float)) if np.isfinite(v) and v > 0]
    if len(valid_pos) < 3:
        return out

    m = len(valid_pos)
    vals = np.full((m, 2), np.nan, dtype=float)
    for m_idx, src_idx in enumerate(valid_pos):
        v0 = float(opt_keep.iloc[src_idx]) if pd.notna(opt_keep.iloc[src_idx]) else np.nan
        v1 = float(opt_scaled.iloc[src_idx]) if pd.notna(opt_scaled.iloc[src_idx]) else np.nan
        vals[m_idx, 0] = v0 if np.isfinite(v0) and v0 > 0 else np.nan
        vals[m_idx, 1] = v1 if np.isfinite(v1) and v1 > 0 else np.nan

    dp = np.full((m, 2), np.inf, dtype=float)
    prev = np.full((m, 2), -1, dtype=int)

    for k in (0, 1):
        if np.isfinite(vals[0, k]):
            dp[0, k] = 0.0

    for i in range(1, m):
        for k in (0, 1):
            v_cur = vals[i, k]
            if not np.isfinite(v_cur):
                continue
            best_cost = np.inf
            best_prev = -1
            for j in (0, 1):
                v_prev = vals[i - 1, j]
                if not np.isfinite(v_prev):
                    continue
                if not np.isfinite(dp[i - 1, j]):
                    continue
                trans = abs(float(np.log(v_cur / v_prev)))
                # Small preference to keep source value when both are similar.
                if k == 1:
                    trans += 1e-4
                cost = dp[i - 1, j] + trans
                if cost < best_cost:
                    best_cost = cost
                    best_prev = j
            dp[i, k] = best_cost
            prev[i, k] = best_prev

    end_state = 0 if dp[m - 1, 0] <= dp[m - 1, 1] else 1
    if not np.isfinite(dp[m - 1, end_state]):
        return out

    chosen = np.zeros(m, dtype=int)
    chosen[m - 1] = end_state
    for i in range(m - 1, 0, -1):
        chosen[i - 1] = prev[i, chosen[i]] if prev[i, chosen[i]] >= 0 else 0

    result = out.copy()
    for m_idx, src_idx in enumerate(valid_pos):
        result.iloc[src_idx] = vals[m_idx, chosen[m_idx]]

    return result


def _build_quarterly_eps_priority(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    diluted_eps = pd.to_numeric(frame.get("Diluted EPS"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    net_income_common_direct = pd.to_numeric(frame.get("Net Income Common"), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    net_income = pd.to_numeric(frame.get("Net Income"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    net_income_common = net_income_common_direct.where(net_income_common_direct.notna(), net_income)

    shares_outstanding = pd.to_numeric(frame.get("Shares"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    diluted_shares = pd.to_numeric(frame.get("Diluted Shares"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    basic_shares_raw = pd.to_numeric(frame.get("Basic Shares"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if shares_outstanding.notna().any():
        dil_ratio = (diluted_shares / shares_outstanding.replace(0, np.nan)).abs()
        diluted_shares = diluted_shares.mask((dil_ratio < 0.5) | (dil_ratio > 2.0))
        bas_ratio = (basic_shares_raw / shares_outstanding.replace(0, np.nan)).abs()
        basic_shares_raw = basic_shares_raw.mask((bas_ratio < 0.5) | (bas_ratio > 2.0))

    basic_shares = basic_shares_raw.where(basic_shares_raw.notna(), shares_outstanding)

    eps_from_diluted = net_income_common / diluted_shares.replace(0, np.nan)
    eps_from_basic = net_income_common / basic_shares.replace(0, np.nan)

    # SEC diluted EPS is preferred, but invalidate obvious scale mismatches when NI/share is available.
    ref_eps = eps_from_diluted.where(eps_from_diluted.notna(), eps_from_basic)
    eps_ratio = (diluted_eps / ref_eps.replace(0, np.nan)).abs()
    sec_eps_invalid = ref_eps.notna() & diluted_eps.notna() & ((eps_ratio > 2.5) | (eps_ratio < 0.4))
    diluted_eps = diluted_eps.mask(sec_eps_invalid)

    eps_priority = diluted_eps.copy()
    source = pd.Series("none", index=frame.index, dtype="object")
    source.loc[diluted_eps.notna()] = "sec_eps"

    use_diluted = eps_priority.isna() & eps_from_diluted.notna()
    eps_priority.loc[use_diluted] = eps_from_diluted.loc[use_diluted]
    source.loc[use_diluted] = "ni_over_shares_diluted"

    use_basic = eps_priority.isna() & eps_from_basic.notna()
    eps_priority.loc[use_basic] = eps_from_basic.loc[use_basic]
    source.loc[use_basic] = "ni_over_shares_basic"

    eps_priority = eps_priority.replace([np.inf, -np.inf], np.nan)
    return eps_priority, diluted_eps, diluted_shares, basic_shares, net_income_common, source


def _build_standard_quarterly_frame(
    ticker: str,
    companyfacts: dict[str, Any],
    min_date: pd.Timestamp,
    price_series: pd.Series,
    split_series: pd.Series | None = None,
    market: str = "us",
    submissions_accession_map: dict[str, dict[str, Any]] | None = None,
    issuer_company_name: str = "",
    issuer_sic: Any = None,
    issuer_sic_description: str = "",
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for metric, spec in METRIC_SPECS.items():
        series_map[metric] = _extract_metric_series(companyfacts, metric_name=metric, spec=spec, min_date=min_date)
    series_map["COGS"] = _build_refined_cogs_series(
        companyfacts,
        min_date=min_date,
        base_series=series_map.get("COGS", pd.Series(dtype=float)),
        gross_profit_series=series_map.get("Gross Profit", pd.Series(dtype=float)),
        revenue_series=series_map.get("Revenue", pd.Series(dtype=float)),
    )
    series_map["SG&A"] = _build_refined_sga_series(companyfacts, min_date=min_date, base_series=series_map.get("SG&A", pd.Series(dtype=float)))
    finance_like_issuer = _is_finance_like_issuer(
        company_name=issuer_company_name or str(companyfacts.get("entityName") or ""),
        sic=issuer_sic,
        sic_description=issuer_sic_description,
    )
    operating_income_revenue = pd.to_numeric(series_map.get("Revenue", pd.Series(dtype=float)), errors="coerce")
    operating_income_gross_profit = _reconcile_signed_reconstruction(
        pd.to_numeric(series_map.get("Gross Profit", pd.Series(dtype=float)), errors="coerce"),
        operating_income_revenue - _normalize_expense_series(series_map.get("COGS", pd.Series(dtype=float))),
    )
    series_map["Operating Income"] = _build_refined_operating_income_series(
        companyfacts,
        min_date=min_date,
        base_series=series_map.get("Operating Income", pd.Series(dtype=float)),
        revenue_series=operating_income_revenue,
        gross_profit_series=operating_income_gross_profit,
        sga_series=series_map.get("SG&A", pd.Series(dtype=float)),
        rd_series=series_map.get("R&D", pd.Series(dtype=float)),
        finance_like=finance_like_issuer,
    )
    series_map["Net Income"] = _build_refined_net_income_series(
        base_series=series_map.get("Net Income", pd.Series(dtype=float)),
        pretax_series=series_map.get("Pretax Income", pd.Series(dtype=float)),
        tax_series=series_map.get("Tax", pd.Series(dtype=float)),
        operating_income_series=series_map.get("Operating Income", pd.Series(dtype=float)),
        revenue_series=series_map.get("Revenue", pd.Series(dtype=float)),
    )
    series_map["Pretax Income"] = _build_refined_pretax_income_series(
        base_series=series_map.get("Pretax Income", pd.Series(dtype=float)),
        net_income_series=series_map.get("Net Income", pd.Series(dtype=float)),
        tax_series=series_map.get("Tax", pd.Series(dtype=float)),
    )
    series_map["Cash"] = _build_refined_cash_series(companyfacts, min_date=min_date, base_series=series_map.get("Cash", pd.Series(dtype=float)))
    series_map["Capital Expenditure"] = _build_refined_capex_series(
        companyfacts,
        min_date=min_date,
        base_series=series_map.get("Capital Expenditure", pd.Series(dtype=float)),
    )
    series_map["D&A"] = _build_refined_da_series(
        companyfacts,
        min_date=min_date,
        base_series=series_map.get("D&A", pd.Series(dtype=float)),
    )
    series_map["AOCI"] = _build_refined_aoci_series(
        companyfacts,
        min_date=min_date,
        base_series=series_map.get("AOCI", pd.Series(dtype=float)),
    )
    series_map["Deferred Revenue"] = _build_refined_deferred_revenue_series(
        companyfacts,
        min_date=min_date,
        base_series=series_map.get("Deferred Revenue", pd.Series(dtype=float)),
    )
    series_map["Debt Short"] = _build_refined_debt_short_series(
        companyfacts,
        min_date=min_date,
        base_series=series_map.get("Debt Short", pd.Series(dtype=float)),
    )
    series_map["Debt Long"] = _build_refined_debt_long_series(
        companyfacts,
        min_date=min_date,
        base_series=series_map.get("Debt Long", pd.Series(dtype=float)),
    )

    index_union = pd.DatetimeIndex([])
    for series in series_map.values():
        if series.empty:
            continue
        index_union = index_union.union(pd.DatetimeIndex(series.index))

    if index_union.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "term",
                "StatementDate",
                *FLOW_COLUMNS,
                *STOCK_COLUMNS,
                *EXTRA_COLUMNS,
                *META_COLUMNS,
                "CollectedAt",
                "RequestedStart",
                "ExtractorVersion",
            ]
        )

    idx = pd.DatetimeIndex(index_union.unique()).sort_values()
    idx = idx[idx >= min_date]
    if idx.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "term",
                "StatementDate",
                *FLOW_COLUMNS,
                *STOCK_COLUMNS,
                *EXTRA_COLUMNS,
                *META_COLUMNS,
                "CollectedAt",
                "RequestedStart",
                "ExtractorVersion",
            ]
        )

    out = pd.DataFrame(index=idx)
    for metric, series in series_map.items():
        out[metric] = pd.to_numeric(series.reindex(idx), errors="coerce")

    revenue = pd.to_numeric(out.get("Revenue"), errors="coerce")
    cogs = _normalize_expense_series(out.get("COGS"))
    gross_direct = pd.to_numeric(out.get("Gross Profit"), errors="coerce")
    sga = _normalize_expense_series(out.get("SG&A"))

    gross_reconstructed = revenue - cogs
    gross = _reconcile_signed_reconstruction(gross_direct, gross_reconstructed)
    cogs, gross = _apply_negative_margin_gross_profit_proxy(
        revenue_series=revenue,
        cogs_series=cogs,
        gross_profit_series=gross,
        sga_series=sga,
        operating_income_series=pd.to_numeric(out.get("Operating Income"), errors="coerce"),
    )
    revenue = revenue.where(revenue.notna(), gross + cogs)
    cogs = cogs.where(cogs.notna(), _normalize_expense_series(revenue - gross))
    gross = _reconcile_signed_reconstruction(gross, revenue - cogs)
    cogs, gross = _apply_negative_margin_gross_profit_proxy(
        revenue_series=revenue,
        cogs_series=cogs,
        gross_profit_series=gross,
        sga_series=sga,
        operating_income_series=pd.to_numeric(out.get("Operating Income"), errors="coerce"),
    )

    out["Revenue"] = revenue
    out["COGS"] = cogs
    out["Gross Profit"] = gross

    assets = pd.to_numeric(out.get("Total Assets"), errors="coerce")
    liabilities = pd.to_numeric(out.get("Total Liabilities"), errors="coerce")
    equity = pd.to_numeric(out.get("Shareholders Equity"), errors="coerce")

    assets, liabilities, equity = _enforce_balance_identity(
        assets=assets,
        liabilities=liabilities,
        equity=equity,
    )

    out["Total Assets"] = assets
    out["Total Liabilities"] = liabilities
    out["Shareholders Equity"] = equity

    for col in (
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
        "Common Stock",
        "APIC",
        "Retained Earnings",
        "AOCI",
        "Shares",
    ):
        series = pd.to_numeric(out.get(col), errors="coerce")
        if series.notna().any():
            out[col] = series.ffill().bfill()

    assets = pd.to_numeric(out.get("Total Assets"), errors="coerce")
    liabilities = pd.to_numeric(out.get("Total Liabilities"), errors="coerce")
    equity = pd.to_numeric(out.get("Shareholders Equity"), errors="coerce")
    assets, liabilities, equity = _enforce_balance_identity(
        assets=assets,
        liabilities=liabilities,
        equity=equity,
    )
    out["Total Assets"] = assets
    out["Total Liabilities"] = liabilities
    out["Shareholders Equity"] = equity

    for share_col in ("Diluted Shares", "Basic Shares", "Shares"):
        share_series = pd.to_numeric(out.get(share_col), errors="coerce")
        normalized = _normalize_share_series(share_series)
        if share_col == "Shares":
            normalized = _normalize_shares_to_price_basis(
                shares=normalized,
                quarter_index=pd.DatetimeIndex(out.index),
                split_series=split_series,
            )
        out[share_col] = normalized

    eps_priority, diluted_eps, diluted_shares, basic_shares, net_income_common, eps_source = _build_quarterly_eps_priority(
        out
    )
    direct_common = pd.to_numeric(out.get("Net Income Common"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["Net Income Common"] = direct_common.where(direct_common.notna(), net_income_common)
    out["EPS"] = eps_priority
    out["diluted_eps"] = diluted_eps
    out["diluted_shares"] = diluted_shares
    out["basic_shares"] = basic_shares
    out["net_income_common"] = net_income_common
    out["eps_source"] = eps_source

    out["Price"] = _align_quarter_prices(price_series, idx)
    out["Price_M1"] = np.nan
    out["Price_M2"] = np.nan
    out["Price_M3"] = np.nan

    # Remove stock-only quarters that have no flow statements at all.
    flow_present = out[FLOW_COLUMNS].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    if flow_present.any():
        out = out.loc[flow_present].copy()

    symbol = str(ticker).strip().upper()
    statement_idx = pd.DatetimeIndex(out.index)
    out["symbol"] = symbol
    out["StatementDate"] = statement_idx
    out["end_date"] = statement_idx
    out["PeriodEnd"] = statement_idx
    out["PeriodStart"] = pd.NaT
    out["FormType"] = pd.NA
    out["FilingDate"] = pd.NaT
    out["AcceptedAt"] = pd.NaT
    out["AvailableDate"] = pd.NaT
    out["AvailabilityMethod"] = "missing"

    out["name"] = companyfacts.get("entityName")
    out["name_kr"] = pd.NA
    out["sector"] = pd.NA
    out["industry"] = pd.NA
    out["avg_volume"] = pd.NA
    out["Source"] = "sec"
    out["CollectedAt"] = now_utc_iso()
    out["RequestedStart"] = min_date
    out["ExtractorVersion"] = SEC_EXTRACTOR_VERSION

    filing_events = _extract_period_filing_events(
        companyfacts=companyfacts,
        min_date=min_date,
        accession_map=submissions_accession_map,
    )
    trading_days = _load_trading_days_for_market(market=market)
    expanded_rows: list[pd.Series] = []
    if filing_events.empty:
        for _, base_row in out.iterrows():
            period_end = pd.to_datetime(base_row.get("PeriodEnd"), errors="coerce")
            inferred_form = _infer_form_type(period_end)
            available_date, method = _coerce_available_date(
                filing_date=pd.NaT,
                accepted_at=pd.NaT,
                period_end=period_end,
                form_type=inferred_form,
                use_next_trading_day=use_next_trading_day_availability,
                trading_days=trading_days,
                fallback_enabled=availability_fallback,
                fallback_q_days=fallback_q_days,
                fallback_k_days=fallback_k_days,
            )
            row = base_row.copy()
            row["FormType"] = inferred_form
            row["FilingDate"] = pd.NaT
            row["AcceptedAt"] = pd.NaT
            row["AvailableDate"] = available_date
            row["AvailabilityMethod"] = method
            expanded_rows.append(row)
    else:
        filing_events = filing_events.sort_values(["PeriodEnd", "FilingDate", "AcceptedAt"])
        events_by_period: dict[pd.Timestamp, pd.DataFrame] = {
            pd.Timestamp(pe): grp.copy()
            for pe, grp in filing_events.groupby("PeriodEnd", sort=False)
        }
        for _, base_row in out.iterrows():
            period_end = pd.to_datetime(base_row.get("PeriodEnd"), errors="coerce")
            if pd.isna(period_end):
                continue
            period_key = pd.Timestamp(period_end).normalize()
            events = events_by_period.get(period_key)
            if events is None or events.empty:
                inferred_form = _infer_form_type(period_end)
                available_date, method = _coerce_available_date(
                    filing_date=pd.NaT,
                    accepted_at=pd.NaT,
                    period_end=period_end,
                    form_type=inferred_form,
                    use_next_trading_day=use_next_trading_day_availability,
                    trading_days=trading_days,
                    fallback_enabled=availability_fallback,
                    fallback_q_days=fallback_q_days,
                    fallback_k_days=fallback_k_days,
                )
                row = base_row.copy()
                row["FormType"] = inferred_form
                row["FilingDate"] = pd.NaT
                row["AcceptedAt"] = pd.NaT
                row["AvailableDate"] = available_date
                row["AvailabilityMethod"] = method
                expanded_rows.append(row)
                continue

            for _, evt in events.iterrows():
                filing_date = pd.to_datetime(evt.get("FilingDate"), errors="coerce")
                accepted_at = pd.to_datetime(evt.get("AcceptedAt"), errors="coerce")
                form_type = _infer_form_type(period_end, str(evt.get("FormType", "")))
                available_date, method = _coerce_available_date(
                    filing_date=filing_date,
                    accepted_at=accepted_at,
                    period_end=period_end,
                    form_type=form_type,
                    use_next_trading_day=use_next_trading_day_availability,
                    trading_days=trading_days,
                    fallback_enabled=availability_fallback,
                    fallback_q_days=fallback_q_days,
                    fallback_k_days=fallback_k_days,
                )
                row = base_row.copy()
                row["PeriodStart"] = pd.to_datetime(evt.get("PeriodStart"), errors="coerce")
                row["FormType"] = form_type
                row["FilingDate"] = filing_date.normalize() if pd.notna(filing_date) else pd.NaT
                row["AcceptedAt"] = accepted_at
                row["AvailableDate"] = available_date
                row["AvailabilityMethod"] = method
                expanded_rows.append(row)

    out = pd.DataFrame(expanded_rows) if expanded_rows else out.iloc[0:0].copy()

    keep_cols = [
        "symbol",
        "term",
        "StatementDate",
        *FLOW_COLUMNS,
        *STOCK_COLUMNS,
        *EXTRA_COLUMNS,
        *META_COLUMNS,
        "CollectedAt",
        "RequestedStart",
        "ExtractorVersion",
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = np.nan

    out = out[keep_cols]
    out["PeriodEnd"] = pd.to_datetime(out["PeriodEnd"], errors="coerce")
    out["PeriodStart"] = pd.to_datetime(out["PeriodStart"], errors="coerce")
    out["FilingDate"] = pd.to_datetime(out["FilingDate"], errors="coerce")
    out["AcceptedAt"] = pd.to_datetime(out["AcceptedAt"], errors="coerce")
    out["AvailableDate"] = pd.to_datetime(out["AvailableDate"], errors="coerce")
    fiscal_meta = infer_fiscal_period_meta(out.get("PeriodEnd"), out.get("FormType"), out.get("PeriodStart"))
    if not fiscal_meta.empty:
        fiscal_meta = fiscal_meta.drop_duplicates(subset=["period_end"], keep="last").set_index("period_end")
        period_index = pd.to_datetime(out.get("PeriodEnd"), errors="coerce").dt.normalize()
        for col in ("fiscal_year", "fiscal_quarter", "fiscal_label"):
            out[col] = fiscal_meta[col].reindex(period_index).to_numpy()
    out["term"] = out.get("fiscal_label")
    out = _collapse_quarterly_period_rows(out)
    return out.sort_values(["StatementDate", "AvailableDate", "FilingDate", "AcceptedAt", "term"]).reset_index(drop=True)


def _humanize_segment_member(member: str) -> str:
    text = str(member or "").strip()
    if not text:
        return ""
    if ":" in text:
        text = text.split(":", 1)[1]
    text = re.sub(r"Member$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _segment_type_from_dimension(dimension: str, member_text: str) -> str:
    dim = str(dimension or "").lower()
    member = str(member_text or "").lower()

    # 1. Check explicit dimension names first
    if "businesssegment" in dim or "operatingsegment" in dim:
        return "business"
    if "geograph" in dim or "country" in dim or "region" in dim:
        return "geography"
    if "product" in dim or "service" in dim:
        return "product"

    # 2. Fallback to token matching on dimension and member
    if any(tok in dim for tok in SEGMENT_MEMBER_GEO_TOKENS) or any(tok in member for tok in SEGMENT_MEMBER_GEO_TOKENS):
        return "geography"
    if any(tok in dim for tok in SEGMENT_MEMBER_PRODUCT_TOKENS) or any(tok in member for tok in SEGMENT_MEMBER_PRODUCT_TOKENS):
        return "product"
    
    return "business"


def _extract_segment_identity(item: dict[str, Any]) -> tuple[str, str]:
    raw = item.get("segment")
    if raw is None:
        return "", ""

    if isinstance(raw, dict):
        dim = str(raw.get("dimension", "")).strip()
        member_val = raw.get("value")
        if isinstance(member_val, list):
            member = str(member_val[0]) if member_val else ""
        else:
            member = str(member_val or "")
        return dim, member.strip()

    if isinstance(raw, list):
        for elem in raw:
            if isinstance(elem, dict):
                dim = str(elem.get("dimension", "")).strip()
                member_val = elem.get("value")
                if isinstance(member_val, list):
                    member = str(member_val[0]) if member_val else ""
                else:
                    member = str(member_val or "")
                if member:
                    return dim, member.strip()
        return "", ""

    return "", str(raw).strip()


def _extract_segment_metric_events(
    *,
    companyfacts: dict[str, Any],
    metric_name: str,
    spec: MetricSpec,
    min_date: pd.Timestamp,
    accession_map: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    accession_meta = accession_map or {}

    for tag in spec.tags:
        namespace, name = _split_tag(tag)
        unit_records = _iter_unit_records(companyfacts, namespace, name, spec.preferred_units)
        if not unit_records:
            continue
        for unit, records in unit_records:
            scale = _unit_multiplier(unit, spec.preferred_units)
            for item in records:
                if not isinstance(item, dict):
                    continue
                form = str(item.get("form", "")).strip().upper()
                if form not in SEC_ALLOWED_FORMS:
                    continue
                end_dt = pd.to_datetime(item.get("end"), errors="coerce")
                if pd.isna(end_dt):
                    continue
                period_end = _quarter_end(pd.Timestamp(end_dt))
                if period_end < min_date:
                    continue

                start_raw = item.get("start")
                start_dt = pd.to_datetime(start_raw, errors="coerce") if start_raw is not None else pd.NaT
                duration = (end_dt - start_dt).days + 1 if pd.notna(start_dt) else np.nan
                # Segment disclosures are usually quarter-duration values; skip long cumulative rows.
                if np.isfinite(duration) and not (70 <= float(duration) <= 130):
                    continue

                dim, member = _extract_segment_identity(item)
                if not member:
                    continue
                seg_name = _humanize_segment_member(member)
                seg_type = _segment_type_from_dimension(dim, seg_name)
                value = _coerce_numeric(item.get("val"))
                if not np.isfinite(value):
                    continue
                value = float(value) * float(scale)

                filing_date = pd.to_datetime(item.get("filed"), errors="coerce")
                accn = _normalize_accession(item.get("accn"))
                meta = accession_meta.get(accn, {}) if accn else {}
                accepted_at = pd.to_datetime(meta.get("accepted_at"), errors="coerce")
                if pd.isna(filing_date):
                    filing_date = pd.to_datetime(meta.get("filing_date"), errors="coerce")
                report_dt = pd.to_datetime(meta.get("report_date"), errors="coerce")
                if pd.notna(report_dt):
                    period_end = _quarter_end(pd.Timestamp(report_dt))

                rows.append(
                    {
                        "PeriodEnd": period_end,
                        "PeriodStart": start_dt,
                        "FormType": _infer_form_type(period_end, form),
                        "FilingDate": filing_date.normalize() if pd.notna(filing_date) else pd.NaT,
                        "AcceptedAt": accepted_at,
                        "accession": accn,
                        "segment_type": seg_type,
                        "segment_name": seg_name or member,
                        "metric": metric_name,
                        "value": value,
                    }
                )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["PeriodEnd"] = pd.to_datetime(out["PeriodEnd"], errors="coerce")
    out["PeriodStart"] = pd.to_datetime(out["PeriodStart"], errors="coerce")
    out["FilingDate"] = pd.to_datetime(out["FilingDate"], errors="coerce")
    out["AcceptedAt"] = pd.to_datetime(out["AcceptedAt"], errors="coerce")
    out = out.dropna(subset=["PeriodEnd", "segment_name"])
    if out.empty:
        return pd.DataFrame()

    out = out.sort_values(["PeriodEnd", "FilingDate", "AcceptedAt", "metric"])
    dedup_keys = [
        "PeriodEnd",
        "FormType",
        "FilingDate",
        "AcceptedAt",
        "segment_type",
        "segment_name",
        "metric",
        "accession",
    ]
    out = out.drop_duplicates(subset=dedup_keys, keep="last")
    return out.reset_index(drop=True)


def _build_segment_quarterly_frame(
    *,
    ticker: str,
    market: str,
    companyfacts: dict[str, Any],
    min_date: pd.Timestamp,
    submissions_accession_map: dict[str, dict[str, Any]] | None = None,
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
) -> pd.DataFrame:
    revenue_events = _extract_segment_metric_events(
        companyfacts=companyfacts,
        metric_name="revenue",
        spec=METRIC_SPECS["Revenue"],
        min_date=min_date,
        accession_map=submissions_accession_map,
    )
    op_events = _extract_segment_metric_events(
        companyfacts=companyfacts,
        metric_name="op_income",
        spec=METRIC_SPECS["Operating Income"],
        min_date=min_date,
        accession_map=submissions_accession_map,
    )
    events = pd.concat([revenue_events, op_events], ignore_index=True, sort=False)
    if events.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    # CompanyFacts segment rows can lack accepted timestamps even when filing dates exist.
    # Keep those rows by falling back to filing_date before pivoting on the metadata keys.
    accepted_at = pd.to_datetime(events.get("AcceptedAt"), errors="coerce", utc=True)
    filing_date_for_accept = pd.to_datetime(events.get("FilingDate"), errors="coerce", utc=True)
    events["AcceptedAt"] = accepted_at.where(accepted_at.notna(), filing_date_for_accept)

    pivot_keys = [
        "PeriodEnd",
        "PeriodStart",
        "FormType",
        "FilingDate",
        "AcceptedAt",
        "segment_type",
        "segment_name",
    ]
    # Keep the latest value by key/metric and then pivot.
    events = events.sort_values(["PeriodEnd", "FilingDate", "AcceptedAt"])
    events = events.drop_duplicates(subset=pivot_keys + ["metric"], keep="last")
    pivot = (
        events.pivot_table(
            index=pivot_keys,
            columns="metric",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )
    pivot.columns = [str(c) for c in pivot.columns]
    if "revenue" not in pivot.columns:
        pivot["revenue"] = np.nan
    if "op_income" not in pivot.columns:
        pivot["op_income"] = np.nan

    trading_days = _load_trading_days_for_market(market=market)
    available_values: list[pd.Timestamp | pd.NaT] = []
    method_values: list[str] = []
    for _, row in pivot.iterrows():
        available_date, method = _coerce_available_date(
            filing_date=pd.to_datetime(row.get("FilingDate"), errors="coerce"),
            accepted_at=pd.to_datetime(row.get("AcceptedAt"), errors="coerce"),
            period_end=pd.to_datetime(row.get("PeriodEnd"), errors="coerce"),
            form_type=str(row.get("FormType", "")),
            use_next_trading_day=use_next_trading_day_availability,
            trading_days=trading_days,
            fallback_enabled=availability_fallback,
            fallback_q_days=fallback_q_days,
            fallback_k_days=fallback_k_days,
        )
        available_values.append(available_date)
        method_values.append(method)

    pivot["available_date"] = pd.to_datetime(available_values, errors="coerce")
    pivot["availability_method"] = method_values
    pivot["ticker"] = str(ticker).strip().upper()
    pivot["market"] = str(market).strip().lower()
    pivot["period_end"] = pd.to_datetime(pivot["PeriodEnd"], errors="coerce")
    pivot["period_start"] = pd.to_datetime(pivot["PeriodStart"], errors="coerce")
    pivot["form_type"] = pivot["FormType"].astype(str)
    pivot["filing_date"] = pd.to_datetime(pivot["FilingDate"], errors="coerce")
    pivot["accepted_at"] = pd.to_datetime(pivot["AcceptedAt"], errors="coerce")
    pivot["source"] = "sec_companyfacts_segment"
    pivot["collected_at"] = now_utc_iso()

    out = pivot[
        [
            "ticker",
            "market",
            "period_end",
            "period_start",
            "form_type",
            "filing_date",
            "accepted_at",
            "available_date",
            "availability_method",
            "segment_type",
            "segment_name",
            "revenue",
            "op_income",
            "source",
            "collected_at",
        ]
    ].copy()
    out = out.dropna(subset=["period_end", "segment_name"]).reset_index(drop=True)
    return out


def _request_json(url: str, user_agent: str | None, timeout: int = 30) -> dict[str, Any]:
    def _fetch() -> dict[str, Any]:
        try:
            _throttle_sec_request(url)
            response = requests.get(url, headers=_sec_headers(user_agent), timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise KeyError(f"SEC data not found at url={url}") from e
            raise

    return _load_or_fetch_sec_response(
        kind="json",
        url=url,
        fetcher=_fetch,
        max_entries=SEC_JSON_RESPONSE_CACHE_MAX_ENTRIES,
    )


def load_ticker_cik_map(
    user_agent: str | None = None,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    cache_path: Path = SEC_TICKER_MAP_CACHE,
) -> dict[str, int]:
    ensure_dir(cache_path.parent)

    raw: dict[str, Any] | list[Any] | None = None
    if cache_path.exists() and not force_refresh:
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            raw = None

    if raw is None:
        try:
            raw = retry_call(
                lambda: _request_json(SEC_TICKERS_URL, user_agent=user_agent),
                retries=retries,
                backoff_base=backoff,
                label="sec:ticker-map",
            )
            cache_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        except Exception:
            if cache_path.exists():
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                raise

    ticker_map: dict[str, int] = {}
    if isinstance(raw, dict):
        values = raw.values()
    elif isinstance(raw, list):
        values = raw
    else:
        values = []

    for item in values:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        cik = pd.to_numeric(item.get("cik_str"), errors="coerce")
        if not ticker or not np.isfinite(cik):
            continue
        ticker_map[ticker] = int(cik)
    return ticker_map


def _load_sec_json_with_cache(
    *,
    url: str,
    cache_path: Path,
    user_agent: str | None = None,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    label: str,
) -> dict[str, Any] | list[Any]:
    ensure_dir(cache_path.parent)

    raw: dict[str, Any] | list[Any] | None = None
    if cache_path.exists() and not force_refresh:
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            raw = None

    if raw is None:
        raw = retry_call(
            lambda: _request_json(url, user_agent=user_agent),
            retries=retries,
            backoff_base=backoff,
            label=label,
        )
        cache_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    return raw


def _load_sec_reference_frame(
    *,
    url: str,
    cache_path: Path,
    user_agent: str | None = None,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    label: str,
) -> pd.DataFrame:
    raw = _load_sec_json_with_cache(
        url=url,
        cache_path=cache_path,
        user_agent=user_agent,
        force_refresh=force_refresh,
        retries=retries,
        backoff=backoff,
        label=label,
    )
    if not isinstance(raw, dict):
        return pd.DataFrame()
    fields = [str(v).strip() for v in list(raw.get("fields", []) or []) if str(v).strip()]
    data = list(raw.get("data", []) or [])
    if not fields or not data:
        return pd.DataFrame(columns=fields)
    try:
        out = pd.DataFrame(data, columns=fields)
    except Exception:
        return pd.DataFrame(columns=fields)
    return out


def _normalize_sec_exchange_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    upper = text.upper()
    mapping = {
        "NASDAQ": "NASDAQ",
        "NASDQ": "NASDAQ",
        "NYSE": "NYSE",
        "NYSE ARCA": "NYSE ARCA",
        "NYSEAMERICAN": "NYSE AMERICAN",
        "NYSE AMERICAN": "NYSE AMERICAN",
        "NYSE MKT": "NYSE AMERICAN",
        "AMEX": "AMEX",
    }
    return mapping.get(upper, upper)


def _ticker_variant_base(ticker: str, suffix: str) -> str:
    symbol = str(ticker).strip().upper()
    suf = str(suffix).strip().upper()
    if not symbol or not suf:
        return ""
    if symbol.endswith(f"-{suf}"):
        return symbol[: -(len(suf) + 1)]
    if symbol.endswith(suf):
        return symbol[: -len(suf)]
    return ""


def _looks_like_ticker_variant(ticker: str, suffixes: tuple[str, ...], all_tickers: set[str]) -> bool:
    symbol = str(ticker).strip().upper()
    if not symbol:
        return False
    for suffix in suffixes:
        base = _ticker_variant_base(symbol, suffix)
        if len(base) >= 1 and base in all_tickers:
            return True
    return False


def _name_has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(str(word).strip().upper())}\b", str(text or "").strip().upper()) is not None


def _classify_sec_security_category(
    *,
    ticker: str,
    company_name: str,
    all_tickers: set[str] | None = None,
    mutual_fund_tickers: set[str] | None = None,
) -> str:
    symbol = str(ticker).strip().upper()
    name_upper = str(company_name or "").strip().upper()
    ticker_set = all_tickers or set()
    mf_set = mutual_fund_tickers or set()

    if symbol in mf_set:
        return "fund"

    if _name_has_word(name_upper, "ETF") or "EXCHANGE TRADED FUND" in name_upper:
        return "etf"
    if _name_has_word(name_upper, "ETN") or "EXCHANGE TRADED NOTE" in name_upper:
        return "etn"
    if _name_has_word(name_upper, "TRUST"):
        return "trust"
    if _name_has_word(name_upper, "FUND"):
        return "fund"
    if _name_has_word(name_upper, "PREFERRED") or _name_has_word(name_upper, "PREFERENCE"):
        return "preferred"
    if _name_has_word(name_upper, "WARRANT"):
        return "warrants"
    if _name_has_word(name_upper, "RIGHT") or _name_has_word(name_upper, "RIGHTS"):
        return "rights"
    if _name_has_word(name_upper, "UNIT") or _name_has_word(name_upper, "UNITS"):
        return "units"

    if "-" in symbol:
        suffix = symbol.split("-", 1)[1]
        if suffix.startswith("PR") or re.fullmatch(r"P[A-Z0-9]+", suffix):
            return "preferred"
        if suffix in {"W", "WS", "WT"}:
            return "warrants"
        if suffix in {"R", "RT"}:
            return "rights"
        if suffix in {"U", "UN"}:
            return "units"

    if ticker_set:
        if _looks_like_ticker_variant(symbol, ("WS", "WT", "W"), ticker_set):
            return "warrants"
        if _looks_like_ticker_variant(symbol, ("RT", "R"), ticker_set):
            return "rights"
        if _looks_like_ticker_variant(symbol, ("UN", "U"), ticker_set):
            return "units"

    return "common_stock"


_FINANCE_LIKE_SIC_PREFIXES = ("60", "61", "62", "63", "64", "65", "67")
_FINANCE_LIKE_SIC_CODES = {"6798", "6799"}
_FINANCE_LIKE_TEXT_TOKENS = (
    "BANK",
    "BANC",
    "CREDIT",
    "LENDING",
    "LOAN",
    "MORTGAGE",
    "INSURANCE",
    "REIT",
    "REAL ESTATE INVESTMENT TRUST",
    "INVESTMENT ADVICE",
    "ASSET MANAGEMENT",
    "BROKER",
    "DEALER",
    "BDC",
    "BUSINESS DEVELOPMENT",
)


def _is_finance_like_issuer(
    *,
    company_name: str = "",
    sic: Any = None,
    sic_description: str = "",
) -> bool:
    sic_digits = re.sub(r"\D", "", str(sic or "").strip())
    sic4 = sic_digits[:4] if len(sic_digits) >= 4 else ""
    if sic4 in _FINANCE_LIKE_SIC_CODES:
        return True
    if any(sic4.startswith(prefix) for prefix in _FINANCE_LIKE_SIC_PREFIXES):
        return True

    text = f"{company_name or ''} {sic_description or ''}".strip().upper()
    if not text:
        return False
    return any(token in text for token in _FINANCE_LIKE_TEXT_TOKENS)


def load_sec_ticker_reference(
    user_agent: str | None = None,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    exchange_cache_path: Path = SEC_TICKER_EXCHANGE_CACHE,
    mf_cache_path: Path = SEC_TICKER_MF_CACHE,
) -> pd.DataFrame:
    exchange_frame = _load_sec_reference_frame(
        url=SEC_TICKERS_EXCHANGE_URL,
        cache_path=exchange_cache_path,
        user_agent=user_agent,
        force_refresh=force_refresh,
        retries=retries,
        backoff=backoff,
        label="sec:ticker-exchange-map",
    )
    if exchange_frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "market",
                "cik",
                "company_name",
                "exchange",
                "security_category",
                "is_common_stock",
                "source",
            ]
        )

    mf_frame = _load_sec_reference_frame(
        url=SEC_TICKERS_MF_URL,
        cache_path=mf_cache_path,
        user_agent=user_agent,
        force_refresh=force_refresh,
        retries=retries,
        backoff=backoff,
        label="sec:ticker-mf-map",
    )

    out = exchange_frame.copy()
    out["ticker"] = out.get("ticker", pd.Series(dtype=object)).astype(str).str.strip().str.upper()
    out["cik"] = pd.to_numeric(out.get("cik"), errors="coerce").astype("Int64").astype(str)
    out["cik"] = out["cik"].replace("<NA>", "")
    out["company_name"] = out.get("name", pd.Series(dtype=object)).astype(str).str.strip()
    out["exchange"] = out.get("exchange", pd.Series(dtype=object)).map(_normalize_sec_exchange_name)
    out["market"] = "us"

    mf_tickers: set[str] = set()
    if not mf_frame.empty and "ticker" in mf_frame.columns:
        mf_tickers = {
            str(v).strip().upper()
            for v in mf_frame["ticker"].dropna().astype(str).tolist()
            if str(v).strip()
        }

    all_tickers = {
        str(v).strip().upper()
        for v in out["ticker"].dropna().astype(str).tolist()
        if str(v).strip()
    }
    out["security_category"] = [
        _classify_sec_security_category(
            ticker=ticker,
            company_name=company_name,
            all_tickers=all_tickers,
            mutual_fund_tickers=mf_tickers,
        )
        for ticker, company_name in zip(out["ticker"], out["company_name"], strict=False)
    ]
    out["is_common_stock"] = out["security_category"].eq("common_stock")
    out["source"] = "sec_company_tickers_exchange"
    out = out.drop_duplicates(subset=["ticker"], keep="last")
    return out[
        [
            "ticker",
            "market",
            "cik",
            "company_name",
            "exchange",
            "security_category",
            "is_common_stock",
            "source",
        ]
    ].reset_index(drop=True)


def load_sec_ticker_reference_map(
    user_agent: str | None = None,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
) -> dict[str, dict[str, Any]]:
    global _SEC_TICKER_REFERENCE_CACHE  # noqa: PLW0603

    if _SEC_TICKER_REFERENCE_CACHE is not None and not force_refresh:
        return _SEC_TICKER_REFERENCE_CACHE

    frame = load_sec_ticker_reference(
        user_agent=user_agent,
        force_refresh=force_refresh,
        retries=retries,
        backoff=backoff,
    )
    ref_map: dict[str, dict[str, Any]] = {}
    if not frame.empty:
        for row in frame.to_dict(orient="records"):
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            ref_map[ticker] = row
    _SEC_TICKER_REFERENCE_CACHE = ref_map
    return ref_map


def fetch_companyfacts(
    ticker: str,
    user_agent: str | None = None,
    force_refresh: bool = False,
    cache_only: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    map_force_refresh: bool = False,
    map_cache_path: Path = SEC_TICKER_MAP_CACHE,
    raw_cache_dir: Path | None = SEC_RAW_COMPANYFACTS_DIR,
) -> tuple[dict[str, Any], int]:
    symbol = str(ticker).strip().upper()
    if not symbol:
        raise ValueError("ticker is required")

    raw_path = None if raw_cache_dir is None else raw_cache_dir / f"{sanitize_ticker(symbol)}.json"
    if raw_cache_dir is not None:
        ensure_dir(raw_cache_dir)

    if raw_path is not None and raw_path.exists() and (cache_only or not force_refresh):
        try:
            raw_json = json.loads(raw_path.read_text(encoding="utf-8"))
            cached_cik = pd.to_numeric(raw_json.get("cik"), errors="coerce")
            if np.isfinite(cached_cik):
                return raw_json, int(cached_cik)
            if cache_only:
                raise RuntimeError(f"cached SEC companyfacts for {symbol} is missing cik metadata")
        except Exception:
            pass
    if cache_only:
        raise RuntimeError(f"cached SEC companyfacts JSON not found for ticker={symbol}")

    cik_map = load_ticker_cik_map(
        user_agent=user_agent,
        force_refresh=map_force_refresh,
        retries=retries,
        backoff=backoff,
        cache_path=map_cache_path,
    )
    cik = cik_map.get(symbol)
    if cik is None:
        raise KeyError(f"SEC CIK not found for ticker={symbol}")

    url = SEC_COMPANYFACTS_URL_TEMPLATE.format(cik=cik)
    raw_json = retry_call(
        lambda: _request_json(url, user_agent=user_agent),
        retries=retries,
        backoff_base=backoff,
        non_retriable_exceptions=(KeyError,),
        label=f"sec:companyfacts:{symbol}",
    )
    if not isinstance(raw_json, dict):
        raise RuntimeError(f"SEC companyfacts response is not a dict for {symbol}: got {type(raw_json)}")
    if raw_path is not None:
        raw_path.write_text(json.dumps(raw_json, ensure_ascii=False), encoding="utf-8")
    return raw_json, cik


def fetch_submissions(
    ticker: str,
    cik: int,
    user_agent: str | None = None,
    force_refresh: bool = False,
    cache_only: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    raw_cache_dir: Path | None = SEC_RAW_SUBMISSIONS_DIR,
) -> dict[str, Any]:
    symbol = str(ticker).strip().upper()
    raw_path = None if raw_cache_dir is None else raw_cache_dir / f"{sanitize_ticker(symbol)}.json"
    if raw_cache_dir is not None:
        ensure_dir(raw_cache_dir)

    if raw_path is not None and raw_path.exists() and (cache_only or not force_refresh):
        try:
            return json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if cache_only:
        raise RuntimeError(f"cached SEC submissions JSON not found for ticker={symbol}")

    url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=int(cik))
    payload = retry_call(
        lambda: _request_json(url, user_agent=user_agent),
        retries=retries,
        backoff_base=backoff,
        non_retriable_exceptions=(KeyError,),
        label=f"sec:submissions:{symbol}",
    )
    if raw_path is not None:
        raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _coerce_sec_cik(cik: Any) -> str:
    numeric = pd.to_numeric(cik, errors="coerce")
    if not np.isfinite(numeric):
        return ""
    return str(int(numeric))


def _resolve_submission_exchange(submissions: dict[str, Any] | None, ticker: str) -> str:
    if not isinstance(submissions, dict):
        return ""
    symbol = str(ticker).strip().upper()
    tickers = [str(v).strip().upper() for v in list(submissions.get("tickers", []) or []) if str(v).strip()]
    exchanges = [str(v).strip() for v in list(submissions.get("exchanges", []) or []) if str(v).strip()]
    if symbol and tickers:
        for idx, listed_ticker in enumerate(tickers):
            if listed_ticker != symbol:
                continue
            if idx < len(exchanges):
                return exchanges[idx]
            break
    return exchanges[0] if exchanges else ""


def build_sec_issuer_profile(
    *,
    ticker: str,
    market: str = "us",
    companyfacts: dict[str, Any] | None = None,
    submissions: dict[str, Any] | None = None,
    cik: int | None = None,
    user_agent: str | None = None,
) -> pd.DataFrame:
    symbol = str(ticker).strip().upper()
    market_norm = str(market).strip().lower()
    if not symbol:
        return pd.DataFrame(columns=SEC_ISSUER_COLUMNS)

    companyfacts_payload = companyfacts if isinstance(companyfacts, dict) else {}
    submissions_payload = submissions if isinstance(submissions, dict) else {}
    reference_map = load_sec_ticker_reference_map(user_agent=user_agent)
    reference_row = reference_map.get(symbol, {})
    cik_str = (
        _coerce_sec_cik(cik)
        or _coerce_sec_cik(submissions_payload.get("cik"))
        or _coerce_sec_cik(companyfacts_payload.get("cik"))
        or _coerce_sec_cik(reference_row.get("cik"))
    )
    company_name = str(
        submissions_payload.get("name")
        or companyfacts_payload.get("entityName")
        or reference_row.get("company_name")
        or ""
    ).strip()
    exchange = _normalize_sec_exchange_name(
        _resolve_submission_exchange(submissions_payload, symbol)
        or reference_row.get("exchange")
    )
    reference_tickers = set(reference_map.keys())
    mutual_fund_tickers = {
        key
        for key, value in reference_map.items()
        if str(value.get("security_category", "")).strip().lower() == "fund"
    }
    security_category = _classify_sec_security_category(
        ticker=symbol,
        company_name=company_name,
        all_tickers=reference_tickers,
        mutual_fund_tickers=mutual_fund_tickers,
    )
    if not (cik_str or company_name or exchange):
        return pd.DataFrame(columns=SEC_ISSUER_COLUMNS)
    source = (
        "sec_submissions"
        if submissions_payload
        else ("sec_companyfacts" if companyfacts_payload else str(reference_row.get("source") or "sec_unknown"))
    )

    row = {
        "ticker": symbol,
        "market": market_norm,
        "cik": cik_str or None,
        "company_name": company_name or None,
        "exchange": exchange or None,
        "security_category": security_category,
        "is_common_stock": security_category == "common_stock",
        "source": source,
        "collected_at": pd.Timestamp.now(tz="UTC"),
    }
    return pd.DataFrame([row], columns=SEC_ISSUER_COLUMNS)


def _build_submissions_accession_map(submissions: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(submissions, dict):
        return out

    filings = submissions.get("filings", {})
    if not isinstance(filings, dict):
        return out
    recent = filings.get("recent", {})
    if not isinstance(recent, dict) or not recent:
        return out

    forms = list(recent.get("form", []) or [])
    accns = list(recent.get("accessionNumber", []) or [])
    filed = list(recent.get("filingDate", []) or [])
    accepted = list(recent.get("acceptanceDateTime", []) or [])
    report_dates = list(recent.get("reportDate", []) or [])
    n = max(len(forms), len(accns), len(filed), len(accepted), len(report_dates))
    for i in range(n):
        form = str(forms[i] if i < len(forms) else "").strip().upper()
        if form not in SEC_ALLOWED_FORMS:
            continue
        accn = _normalize_accession(accns[i] if i < len(accns) else "")
        if not accn:
            continue
        filed_dt = pd.to_datetime(filed[i] if i < len(filed) else pd.NaT, errors="coerce")
        accepted_dt = pd.to_datetime(accepted[i] if i < len(accepted) else pd.NaT, errors="coerce")
        report_dt = pd.to_datetime(report_dates[i] if i < len(report_dates) else pd.NaT, errors="coerce")
        row = {
            "form": form,
            "filing_date": filed_dt,
            "accepted_at": accepted_dt,
            "report_date": report_dt,
        }
        cur = out.get(accn)
        if cur is None:
            out[accn] = row
            continue
        cur_filed = pd.to_datetime(cur.get("filing_date"), errors="coerce")
        if pd.notna(filed_dt) and (pd.isna(cur_filed) or filed_dt >= cur_filed):
            out[accn] = row
    return out


def _scale_label_from_multiplier(multiplier: float) -> str:
    if not np.isfinite(multiplier):
        return "unknown"
    if abs(multiplier - 1.0) < 1e-12:
        return "1"
    if abs(multiplier - 1_000.0) < 1e-6:
        return "1e3"
    if abs(multiplier - 1_000_000.0) < 1e-3:
        return "1e6"
    if abs(multiplier - 1_000_000_000.0) < 1e-1:
        return "1e9"
    return f"{multiplier:g}"


def _coerce_tz_aware_utc(ts: Any) -> pd.Timestamp | pd.NaT:
    out = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(out):
        return pd.NaT
    return pd.Timestamp(out)


def _raw_context_id(item: dict[str, Any], fallback_seq: int) -> str:
    frame = str(item.get("frame", "")).strip()
    if frame:
        return frame
    fp = str(item.get("fp", "")).strip().upper()
    fy = str(item.get("fy", "")).strip()
    end = str(item.get("end", "")).strip()
    accn = _normalize_accession(item.get("accn"))
    parts = [p for p in (accn, fy, fp, end) if p]
    if parts:
        return "|".join(parts)
    return f"row_{fallback_seq}"


def _extract_raw_normalized_facts(
    *,
    ticker: str,
    market: str,
    cik: int | None,
    companyfacts: dict[str, Any],
    min_date: pd.Timestamp,
    submissions_accession_map: dict[str, dict[str, Any]] | None,
    use_next_trading_day_availability: bool,
    availability_fallback: bool,
    fallback_q_days: int,
    fallback_k_days: int,
) -> pd.DataFrame:
    accession_meta = submissions_accession_map or {}
    trading_days = _load_trading_days_for_market(market=market)
    source_url = SEC_COMPANYFACTS_URL_TEMPLATE.format(cik=int(cik)) if cik is not None else ""
    now_ts = pd.Timestamp.now(tz="UTC")
    rows: list[dict[str, Any]] = []

    for metric_name, spec in METRIC_SPECS.items():
        for tag in spec.tags:
            namespace, fact_key = _split_tag(tag)
            unit_rows = _iter_unit_records(companyfacts, namespace, fact_key, spec.preferred_units)
            if not unit_rows:
                continue
            for unit_name, facts in unit_rows:
                multiplier = _unit_multiplier(unit_name, spec.preferred_units)
                scale_label = _scale_label_from_multiplier(multiplier)
                for idx, item in enumerate(facts):
                    if not isinstance(item, dict):
                        continue
                    raw_val = _coerce_numeric(item.get("val"))
                    if not np.isfinite(raw_val):
                        continue
                    value = float(raw_val) * float(multiplier)
                    end_dt = pd.to_datetime(item.get("end"), errors="coerce")
                    instant_dt = pd.to_datetime(item.get("instant"), errors="coerce")
                    period_end = pd.NaT
                    if pd.notna(end_dt):
                        period_end = _quarter_end(pd.Timestamp(end_dt))
                    elif pd.notna(instant_dt):
                        period_end = pd.Timestamp(instant_dt).normalize()
                    if pd.isna(period_end):
                        continue
                    if pd.Timestamp(period_end) < min_date:
                        continue

                    period_start = pd.to_datetime(item.get("start"), errors="coerce")
                    form_raw = str(item.get("form", "")).strip().upper()
                    accn = _normalize_accession(item.get("accn"))
                    meta = accession_meta.get(accn, {}) if accn else {}
                    filing_date = pd.to_datetime(item.get("filed"), errors="coerce")
                    if pd.isna(filing_date):
                        filing_date = pd.to_datetime(meta.get("filing_date"), errors="coerce")
                    accepted_at = pd.to_datetime(meta.get("accepted_at"), errors="coerce")
                    form_type = _infer_form_type(
                        pd.Timestamp(period_end),
                        form_raw or str(meta.get("form", "")),
                    )
                    available_date, availability_method = _coerce_available_date(
                        filing_date=filing_date,
                        accepted_at=accepted_at,
                        period_end=pd.Timestamp(period_end),
                        form_type=form_type,
                        use_next_trading_day=use_next_trading_day_availability,
                        trading_days=trading_days,
                        fallback_enabled=availability_fallback,
                        fallback_q_days=fallback_q_days,
                        fallback_k_days=fallback_k_days,
                    )
                    if not accn:
                        filed_key = (
                            pd.Timestamp(filing_date).strftime("%Y%m%d")
                            if pd.notna(filing_date)
                            else "nofile"
                        )
                        accn = f"noaccn-{filed_key}-{form_type}"

                    segment_payload = item.get("segment")
                    if segment_payload is None:
                        dim_json = ""
                    elif isinstance(segment_payload, (dict, list)):
                        dim_json = json.dumps(segment_payload, ensure_ascii=False, sort_keys=True)
                    else:
                        dim_json = json.dumps({"segment": str(segment_payload)}, ensure_ascii=False, sort_keys=True)

                    rows.append(
                        {
                            "ticker": str(ticker).strip().upper(),
                            "market": str(market).strip().lower(),
                            "accession": accn,
                            "form_type": form_type,
                            "fact_name": metric_name,
                            "taxonomy": namespace,
                            "unit": str(unit_name),
                            "scale": scale_label,
                            "period_start": pd.to_datetime(period_start, errors="coerce"),
                            "period_end": pd.to_datetime(period_end, errors="coerce"),
                            "instant_date": pd.to_datetime(instant_dt, errors="coerce"),
                            "value": value,
                            "context_id": _raw_context_id(item, idx),
                            "dimension_json": dim_json,
                            "filing_date": pd.to_datetime(filing_date, errors="coerce"),
                            "accepted_at": _coerce_tz_aware_utc(accepted_at),
                            "available_date": pd.to_datetime(available_date, errors="coerce"),
                            "availability_method": str(availability_method or ""),
                            "source": "sec_companyfacts",
                            "source_url": source_url,
                            "collected_at": now_ts,
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=RAW_FACT_COLUMNS)

    out = pd.DataFrame(rows)
    period_end_ts = pd.to_datetime(out.get("period_end"), errors="coerce")
    period_start_ts = pd.to_datetime(out.get("period_start"), errors="coerce")
    instant_ts = pd.to_datetime(out.get("instant_date"), errors="coerce")
    # Primary-key safety: period_start/instant_date are required in DB PK.
    period_start_ts = period_start_ts.where(period_start_ts.notna(), period_end_ts)
    instant_ts = instant_ts.where(instant_ts.notna(), period_end_ts)
    out["period_start"] = period_start_ts.dt.date
    out["period_end"] = period_end_ts.dt.date
    out["instant_date"] = instant_ts.dt.date
    out["filing_date"] = pd.to_datetime(out.get("filing_date"), errors="coerce").dt.date
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce").dt.date
    out["accepted_at"] = pd.to_datetime(out.get("accepted_at"), errors="coerce", utc=True)
    out["collected_at"] = pd.to_datetime(out.get("collected_at"), errors="coerce", utc=True)
    out["value"] = pd.to_numeric(out.get("value"), errors="coerce")
    out = out.dropna(subset=["period_end", "value"]).copy()
    out = out.loc[pd.to_datetime(out["period_end"], errors="coerce") >= min_date]
    out = out.sort_values(["period_end", "fact_name", "filing_date", "accepted_at"]).drop_duplicates(
        subset=["ticker", "accession", "fact_name", "unit", "period_start", "period_end", "instant_date", "context_id"],
        keep="last",
    )
    for col in RAW_FACT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[RAW_FACT_COLUMNS].reset_index(drop=True)


def _build_financials_extra_from_quarterly(
    *,
    ticker: str,
    market: str,
    quarterly: pd.DataFrame,
    raw_facts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if quarterly is None or quarterly.empty:
        return pd.DataFrame(columns=FINANCIAL_EXTRA_COLUMNS)

    out = pd.DataFrame()
    out["ticker"] = str(ticker).strip().upper()
    out["market"] = str(market).strip().lower()
    out["period_end"] = pd.to_datetime(quarterly.get("PeriodEnd"), errors="coerce").dt.date
    out["available_date"] = pd.to_datetime(quarterly.get("AvailableDate"), errors="coerce").dt.date
    out["filing_date"] = pd.to_datetime(quarterly.get("FilingDate"), errors="coerce").dt.date
    out["accepted_at"] = pd.to_datetime(quarterly.get("AcceptedAt"), errors="coerce", utc=True)
    out["form_type"] = quarterly.get("FormType")
    out["dividends_paid"] = pd.to_numeric(quarterly.get("Dividends Paid"), errors="coerce")
    out["share_repurchases"] = pd.to_numeric(quarterly.get("Repurchases"), errors="coerce")
    out["sbc"] = pd.to_numeric(quarterly.get("SBC"), errors="coerce")
    out["r_and_d"] = pd.to_numeric(quarterly.get("R&D"), errors="coerce")
    out["shares_outstanding"] = pd.to_numeric(quarterly.get("Shares"), errors="coerce")
    out["shares_eop"] = pd.to_numeric(quarterly.get("Shares"), errors="coerce")
    out["ar"] = pd.to_numeric(quarterly.get("AR"), errors="coerce")
    out["inventory"] = pd.to_numeric(quarterly.get("Inventory"), errors="coerce")
    out["ap"] = pd.to_numeric(quarterly.get("AP"), errors="coerce")
    out["cash"] = pd.to_numeric(quarterly.get("Cash"), errors="coerce")
    out["debt_total"] = (
        pd.to_numeric(quarterly.get("Debt Short"), errors="coerce").fillna(0.0)
        + pd.to_numeric(quarterly.get("Debt Long"), errors="coerce").fillna(0.0)
    )
    out["net_income"] = pd.to_numeric(quarterly.get("Net Income"), errors="coerce")
    out["cfo"] = pd.to_numeric(quarterly.get("Operating Cash Flow"), errors="coerce")
    out["total_assets"] = pd.to_numeric(quarterly.get("Total Assets"), errors="coerce")
    out["owner_equity"] = pd.to_numeric(quarterly.get("Shareholders Equity"), errors="coerce")
    out["owner_net_income"] = pd.to_numeric(quarterly.get("Net Income Common"), errors="coerce").where(
        pd.to_numeric(quarterly.get("Net Income Common"), errors="coerce").notna(),
        pd.to_numeric(quarterly.get("net_income_common"), errors="coerce"),
    )
    out["common_stock"] = pd.to_numeric(quarterly.get("Common Stock"), errors="coerce")
    out["additional_paid_in_capital"] = pd.to_numeric(quarterly.get("APIC"), errors="coerce")
    out["retained_earnings"] = pd.to_numeric(quarterly.get("Retained Earnings"), errors="coerce")
    out["aoci"] = pd.to_numeric(quarterly.get("AOCI"), errors="coerce")
    out["ppe"] = np.nan
    out["ppe_capex"] = np.nan
    out["intangibles"] = pd.to_numeric(quarterly.get("Intangibles"), errors="coerce")
    out["intangible_capex"] = np.nan
    out["amortization"] = pd.to_numeric(quarterly.get("Amortization"), errors="coerce")
    out["other_gain"] = pd.to_numeric(quarterly.get("Other Gain"), errors="coerce")
    out["financial_gain"] = pd.to_numeric(quarterly.get("Financial Gain"), errors="coerce")
    out["equity_method_gain"] = pd.to_numeric(quarterly.get("Equity Method Gain"), errors="coerce")
    out["other_income"] = pd.to_numeric(quarterly.get("Other Income"), errors="coerce")
    out["other_expense"] = pd.to_numeric(quarterly.get("Other Expense"), errors="coerce")
    out["financial_income"] = pd.to_numeric(quarterly.get("Financial Income"), errors="coerce")
    out["financial_expense"] = pd.to_numeric(quarterly.get("Financial Expense"), errors="coerce")
    out["current_fin_assets"] = pd.to_numeric(quarterly.get("Current Fin Assets"), errors="coerce")
    out["non_current_fin_assets"] = pd.to_numeric(quarterly.get("Non Current Fin Assets"), errors="coerce")
    out["current_fin_liabilities"] = pd.to_numeric(quarterly.get("Current Fin Liabilities"), errors="coerce")
    out["non_current_fin_liabilities"] = pd.to_numeric(quarterly.get("Non Current Fin Liabilities"), errors="coerce")
    out["source"] = "sec_companyfacts_derived"
    out["confidence"] = 0.65
    out["collected_at"] = pd.Timestamp.now(tz="UTC")

    raw_series_map = _build_financial_extra_raw_series_map(raw_facts)
    raw_fill_map = {
        "common_stock": "Common Stock",
        "additional_paid_in_capital": "APIC",
        "retained_earnings": "Retained Earnings",
        "aoci": "AOCI",
        "amortization": "Amortization",
        "r_and_d": "R&D",
        "current_fin_assets": "Current Fin Assets",
        "non_current_fin_assets": "Non Current Fin Assets",
        "current_fin_liabilities": "Current Fin Liabilities",
        "non_current_fin_liabilities": "Non Current Fin Liabilities",
    }
    out["period_end_ts"] = pd.to_datetime(out["period_end"], errors="coerce").dt.normalize()
    for target_column, raw_metric_name in raw_fill_map.items():
        if target_column not in out.columns:
            continue
        raw_series = raw_series_map.get(raw_metric_name)
        if raw_series is None or raw_series.empty:
            continue
        aligned_raw = out["period_end_ts"].map(raw_series)
        target = pd.to_numeric(out[target_column], errors="coerce")
        out[target_column] = target.where(target.notna(), aligned_raw)
    out = out.drop(columns=["period_end_ts"], errors="ignore")

    out = out.dropna(subset=["period_end"]).copy()
    out = out.sort_values(["period_end", "available_date", "filing_date", "accepted_at"]).drop_duplicates(
        subset=["ticker", "market", "period_end", "available_date", "form_type"],
        keep="last",
    )
    for col in FINANCIAL_EXTRA_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[FINANCIAL_EXTRA_COLUMNS].reset_index(drop=True)


def _build_financial_extra_raw_series_map(raw_facts: pd.DataFrame | None) -> dict[str, pd.Series]:
    if raw_facts is None or raw_facts.empty:
        return {}
    frame = raw_facts.copy()
    frame["period_end"] = pd.to_datetime(frame.get("period_end"), errors="coerce").dt.normalize()
    frame["period_start"] = pd.to_datetime(frame.get("period_start"), errors="coerce").dt.normalize()
    frame["available_date"] = pd.to_datetime(frame.get("available_date"), errors="coerce").dt.normalize()
    frame["filing_date"] = pd.to_datetime(frame.get("filing_date"), errors="coerce").dt.normalize()
    frame["accepted_at"] = pd.to_datetime(frame.get("accepted_at"), errors="coerce", utc=True)
    frame["value"] = pd.to_numeric(frame.get("value"), errors="coerce")
    frame = frame.dropna(subset=["period_end", "value"]).copy()
    if frame.empty:
        return {}
    frame["duration_days"] = (
        pd.to_datetime(frame["period_end"], errors="coerce") - pd.to_datetime(frame["period_start"], errors="coerce")
    ).dt.days

    out: dict[str, pd.Series] = {}
    flow_metric_names = {
        "Amortization",
        "R&D",
    }
    for fact_name, fact_frame in frame.groupby("fact_name", dropna=False):
        if not isinstance(fact_name, str) or not fact_name.strip():
            continue
        candidates = fact_frame.copy()
        if fact_name in flow_metric_names:
            candidates = candidates.loc[
                candidates["duration_days"].notna() & (candidates["duration_days"] <= RAW_QUARTER_MAX_DURATION_DAYS)
            ].copy()
            if candidates.empty:
                continue
            candidates = candidates.sort_values(
                ["duration_days", "available_date", "accepted_at", "filing_date"],
                ascending=[True, False, False, False],
                na_position="last",
            )
        else:
            candidates = candidates.sort_values(
                ["available_date", "accepted_at", "filing_date"],
                ascending=[False, False, False],
                na_position="last",
            )
        latest = candidates.drop_duplicates(subset=["period_end"], keep="first").copy()
        out[fact_name] = pd.Series(latest["value"].to_numpy(), index=latest["period_end"])
    return out


def _build_recent_filings_from_submissions(
    submissions: dict[str, Any] | None,
    *,
    ticker: str,
    market: str,
    cik: int,
    lookback_filings: int = 8,
) -> pd.DataFrame:
    out = _submissions_records_to_frame(
        submissions,
        ticker=ticker,
        market=market,
        cik=cik,
        allowed_forms=SEC_ALLOWED_FORMS,
    )
    if out.empty:
        return pd.DataFrame(columns=FILING_COLUMNS)
    out = out.sort_values(["filing_date", "accepted_at"], ascending=[False, False], na_position="last")
    out = out.drop_duplicates(subset=["accession"], keep="first")
    out = out.head(max(1, int(lookback_filings)))
    for col in FILING_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[FILING_COLUMNS].reset_index(drop=True)


def _request_text(url: str, user_agent: str | None, timeout: int = 30) -> str:
    def _fetch() -> str:
        _throttle_sec_request(url)
        response = requests.get(url, headers=_sec_headers(user_agent), timeout=timeout)
        response.raise_for_status()
        response.encoding = response.encoding or "utf-8"
        return response.text

    return _load_or_fetch_sec_response(
        kind="text",
        url=url,
        fetcher=_fetch,
        max_entries=SEC_TEXT_RESPONSE_CACHE_MAX_ENTRIES,
    )


def _cache_text(path: Path | None, text: str) -> None:
    if path is None:
        return
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _load_cached_or_fetch_text(
    *,
    url: str,
    cache_path: Path | None,
    user_agent: str | None,
    force_refresh: bool,
    cache_only: bool,
    retries: int,
    backoff: float,
    label: str,
) -> str:
    if cache_path is not None and cache_path.exists() and not force_refresh:
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            pass
    if cache_only:
        raise FileNotFoundError(f"SEC cache not found for {label}: {cache_path}")
    text = retry_call(
        lambda: _request_text(url, user_agent=user_agent),
        retries=retries,
        backoff_base=backoff,
        label=label,
    )
    _cache_text(cache_path, text)
    return text


def _write_json_cache(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _submissions_records_to_frame(
    payload: dict[str, Any] | None,
    *,
    ticker: str,
    market: str,
    cik: int,
    allowed_forms: set[str] | None = None,
) -> pd.DataFrame:
    recent = {}
    if isinstance(payload, dict):
        maybe_filings = payload.get("filings")
        if isinstance(maybe_filings, dict):
            recent = maybe_filings.get("recent", {})
        else:
            recent = payload
    if not isinstance(recent, dict):
        return pd.DataFrame()

    def _arr(name: str) -> list[Any]:
        return list(recent.get(name, []) or [])

    forms = _arr("form")
    accns = _arr("accessionNumber")
    report_dates = _arr("reportDate")
    filing_dates = _arr("filingDate")
    accepted = _arr("acceptanceDateTime")
    docs = _arr("primaryDocument")
    is_xbrl = _arr("isXBRL")
    n = max(len(forms), len(accns), len(report_dates), len(filing_dates), len(accepted), len(docs), len(is_xbrl))

    rows: list[dict[str, Any]] = []
    allowed = allowed_forms or SEC_ALLOWED_FORMS
    for i in range(n):
        form = str(forms[i] if i < len(forms) else "").strip().upper()
        if form not in allowed:
            continue
        accn_raw = accns[i] if i < len(accns) else ""
        accn = _normalize_accession(accn_raw)
        if not accn:
            continue
        accession_nodash = accn.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}"
        primary_doc = str(docs[i] if i < len(docs) else "").strip()
        primary_doc_url = f"{base}/{primary_doc}" if primary_doc else f"{base}/"
        report_date = pd.to_datetime(report_dates[i] if i < len(report_dates) else pd.NaT, errors="coerce")
        filing_date = pd.to_datetime(filing_dates[i] if i < len(filing_dates) else pd.NaT, errors="coerce")
        accepted_at = pd.to_datetime(accepted[i] if i < len(accepted) else pd.NaT, errors="coerce")
        available_date = accepted_at.normalize() if pd.notna(accepted_at) else pd.NaT
        if pd.isna(available_date):
            available_date = filing_date
        rows.append(
            {
                "ticker": str(ticker).strip().upper(),
                "market": str(market).strip().lower(),
                "accession": accn,
                "form_type": form,
                "period_end": report_date,
                "report_date": report_date,
                "available_date": available_date,
                "filing_date": filing_date,
                "accepted_at": accepted_at,
                "primary_doc_url": primary_doc_url,
                "index_url": f"{base}/index.json",
                "is_amendment": form.endswith("/A"),
                "is_nt": form.startswith("NT "),
                "is_xbrl": bool(int(is_xbrl[i])) if i < len(is_xbrl) and str(is_xbrl[i]).strip() else False,
                "collected_at": now_utc_iso(),
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(["filing_date", "accepted_at"], ascending=[False, False], na_position="last")
    out = out.drop_duplicates(subset=["accession"], keep="first")
    return out.reset_index(drop=True)


def _load_full_submissions_history(
    *,
    submissions: dict[str, Any] | None,
    ticker: str,
    market: str,
    cik: int,
    user_agent: str | None,
    force_refresh: bool,
    cache_only: bool,
    retries: int,
    backoff: float,
    raw_cache_dir: Path | None,
    allowed_forms: set[str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    root = _submissions_records_to_frame(
        submissions,
        ticker=ticker,
        market=market,
        cik=cik,
        allowed_forms=allowed_forms,
    )
    if not root.empty:
        frames.append(root)

    files = []
    if isinstance(submissions, dict):
        files = list(submissions.get("filings", {}).get("files", []) or [])
    for entry in files:
        name = str((entry or {}).get("name", "")).strip()
        if not name:
            continue
        cache_path = None if raw_cache_dir is None else raw_cache_dir / name
        try:
            if cache_path is not None and cache_path.exists() and not force_refresh:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            elif cache_only:
                raise FileNotFoundError(f"SEC submissions history cache not found: {cache_path}")
            else:
                payload = retry_call(
                    lambda: _request_json(f"https://data.sec.gov/submissions/{name}", user_agent=user_agent),
                    retries=retries,
                    backoff_base=backoff,
                    non_retriable_exceptions=(KeyError,),
                    label=f"sec:submissions-history:{ticker}:{name}",
                )
                _write_json_cache(cache_path, payload)
        except Exception:
            continue
        hist = _submissions_records_to_frame(
            payload,
            ticker=ticker,
            market=market,
            cik=cik,
            allowed_forms=allowed_forms,
        )
        if not hist.empty:
            frames.append(hist)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["filing_date", "accepted_at"], ascending=[False, False], na_position="last")
    out = out.drop_duplicates(subset=["accession"], keep="first")
    return out.reset_index(drop=True)


def fetch_sec_filing_history(
    *,
    ticker: str,
    market: str = "us",
    start: str | pd.Timestamp = SEC_DEFAULT_START_DATE,
    user_agent: str | None = None,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    map_force_refresh: bool = False,
    map_cache_path: Path = SEC_TICKER_MAP_CACHE,
    submissions_cache_dir: Path | None = SEC_RAW_SUBMISSIONS_DIR,
    offline_mode: bool = False,
    cache_only: bool = False,
    prefetched_submissions: dict[str, Any] | None = None,
    prefetched_cik: int | None = None,
) -> pd.DataFrame:
    if offline_mode:
        return pd.DataFrame(columns=FILING_COLUMNS)

    start_ts = pd.to_datetime(start, errors="coerce")
    if pd.isna(start_ts):
        start_ts = SEC_DEFAULT_START_DATE
    min_date = max(pd.Timestamp(start_ts).normalize(), SEC_DEFAULT_START_DATE)

    symbol = str(ticker).strip().upper()
    market_norm = str(market).strip().lower()
    try:
        if prefetched_submissions is not None:
            submissions = prefetched_submissions
            cik = int(prefetched_cik) if prefetched_cik is not None else int(pd.to_numeric((submissions or {}).get("cik"), errors="coerce"))
        elif cache_only:
            submissions = fetch_submissions(
                ticker=symbol,
                cik=0,
                user_agent=user_agent,
                force_refresh=False,
                cache_only=True,
                retries=retries,
                backoff=backoff,
                raw_cache_dir=submissions_cache_dir,
            )
            cik = int(pd.to_numeric((submissions or {}).get("cik"), errors="coerce"))
        else:
            cik_map = load_ticker_cik_map(
                user_agent=user_agent,
                force_refresh=map_force_refresh,
                retries=retries,
                backoff=backoff,
                cache_path=map_cache_path,
            )
            cik = cik_map.get(symbol)
            if cik is None:
                return pd.DataFrame(columns=FILING_COLUMNS)
            submissions = fetch_submissions(
                ticker=symbol,
                cik=cik,
                user_agent=user_agent,
                force_refresh=force_refresh,
                cache_only=False,
                retries=retries,
                backoff=backoff,
                raw_cache_dir=submissions_cache_dir,
            )
    except Exception:
        return pd.DataFrame(columns=FILING_COLUMNS)

    out = _load_full_submissions_history(
        submissions=submissions,
        ticker=symbol,
        market=market_norm,
        cik=cik,
        user_agent=user_agent,
        force_refresh=force_refresh and not cache_only,
        cache_only=cache_only,
        retries=retries,
        backoff=backoff,
        raw_cache_dir=submissions_cache_dir,
        allowed_forms=SEC_FILING_META_FORMS,
    )
    if out.empty:
        return pd.DataFrame(columns=FILING_COLUMNS)

    period_end = pd.to_datetime(out.get("period_end"), errors="coerce")
    out = out.loc[period_end.isna() | (period_end >= min_date)].copy()
    out = out.sort_values(["filing_date", "accepted_at"], ascending=[False, False], na_position="last")
    out = out.drop_duplicates(subset=["accession"], keep="first")
    for col in FILING_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[FILING_COLUMNS].reset_index(drop=True)


def _normalize_html_label(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\xa0", " ").replace("\u2019", "'").replace("\u2018", "'").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("", "'").replace("", '"').replace("", '"').replace("", "-")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"\s*\((?:\d+|[a-z]|[ivx]{1,4})\)$", "", text).strip()
    return text


def _parse_html_numeric(value: Any) -> float | None:
    text = _normalize_html_label(value)
    if text in {"", "nan", "-", "—", "n/a"}:
        return None
    text = text.replace("$", "").replace(",", "").replace("%", "")
    text = text.replace("(", "-").replace(")", "")
    text = text.strip()
    if text in {"", "-", "--"}:
        return None
    try:
        parsed = float(text)
        return parsed if np.isfinite(parsed) else None
    except Exception:
        return None


def _html_row_label_cells(row: pd.Series, *, max_cols: int = 4) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for idx, cell in enumerate(row.iloc[: min(max_cols, len(row))]):
        label = _normalize_html_label(cell)
        if label in {"", "nan", "$"}:
            continue
        if _parse_html_numeric(cell) is not None:
            continue
        out.append((idx, label))
    return out


def _html_row_label_text(row: pd.Series, *, max_cols: int = 4) -> str:
    labels = [label for _, label in _html_row_label_cells(row, max_cols=max_cols)]
    return " | ".join(labels)


def _extract_first_numeric_after_label(table: pd.DataFrame, aliases: tuple[str, ...]) -> float | None:
    if table is None or table.empty:
        return None
    want = tuple(_normalize_html_label(alias) for alias in aliases)
    exact_rows: list[tuple[pd.Series, int]] = []
    fallback_rows: list[tuple[pd.Series, int]] = []
    for _, row in table.iterrows():
        label_cells = _html_row_label_cells(row)
        if not label_cells:
            continue
        label_text = " | ".join(label for _, label in label_cells)
        if any(label == alias for _, label in label_cells for alias in want) or label_text in want:
            exact_rows.append((row, max(idx for idx, _ in label_cells) + 1))
            continue
        if any(alias in label_text for alias in want):
            fallback_rows.append((row, max(idx for idx, _ in label_cells) + 1))
    for row, start_idx in [*exact_rows, *fallback_rows]:
        for cell in row.iloc[start_idx:]:
            parsed = _parse_html_numeric(cell)
            if parsed is not None:
                return parsed
    return None


def _table_row_hit_score(table: pd.DataFrame, aliases: tuple[str, ...]) -> int:
    if table is None or table.empty:
        return 0
    labels = [_html_row_label_text(table.iloc[idx]) for idx in range(len(table))]
    labels = [label for label in labels if label]
    return sum(1 for alias in aliases if any(alias in label for label in labels))


def _find_html_table_by_row_aliases(
    tables: list[pd.DataFrame],
    *,
    required_aliases: tuple[str, ...],
    min_hits: int,
) -> pd.DataFrame | None:
    best_table: pd.DataFrame | None = None
    best_score = -1
    for table in tables:
        score = _table_row_hit_score(table, required_aliases)
        if score > best_score:
            best_table = table
            best_score = score
    if best_score < int(min_hits):
        return None
    return best_table


def _find_aapl_selected_quarterly_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    for table in tables:
        header_text = " ".join(_normalize_html_label(v) for v in table.columns)
        body_text = " ".join(_normalize_html_label(v) for v in table.to_numpy().flatten())
        label_text = " ".join(_normalize_html_label(v) for v in table.iloc[:, : min(3, table.shape[1])].to_numpy().flatten())
        if (
            "fourth quarter" in header_text
            and "net sales" in label_text
            and "net income" in label_text
            and ("gross margin" in label_text or "gross profit" in label_text)
        ):
            return table
        if "fourth quarter" in body_text and "net sales" in body_text and "gross margin" in body_text and "net income" in body_text:
            return table
    return None


def _html_cashflow_months(table: pd.DataFrame) -> int | None:
    if table is None or table.empty:
        return None
    text = " ".join(_normalize_html_label(v) for v in table.head(4).to_numpy().flatten())
    if "nine months ended" in text:
        return 9
    if "six months ended" in text:
        return 6
    if "three months ended" in text:
        return 3
    if "fiscal years ended" in text or "year ended" in text:
        return 12
    return None


def _extract_aapl_pre_xbrl_tables(html_text: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    try:
        tables = pd.read_html(StringIO(html_text), flavor="lxml")
    except Exception:
        return None, None, None, None, None

    selected_quarterly = _find_aapl_selected_quarterly_table(tables)
    statement_tables = [table for table in tables if table is not selected_quarterly]
    income = _find_html_table_by_row_aliases(
        statement_tables,
        required_aliases=("net sales", "cost of sales", "gross margin", "operating income", "operating loss", "net income"),
        min_hits=3,
    )
    balance = _find_html_table_by_row_aliases(
        statement_tables,
        required_aliases=("cash and cash equivalents", "total current assets", "total assets", "total liabilities", "shareholders' equity"),
        min_hits=3,
    )
    cashflow = _find_html_table_by_row_aliases(
        statement_tables,
        required_aliases=(
            "operating activities",
            "operating:",
            "investing activities",
            "investing:",
            "financing activities",
            "financing:",
            "purchase of property",
            "purchases of property",
            "payments for acquisition of property",
        ),
        min_hits=2,
    )
    eps = _find_html_table_by_row_aliases(
        statement_tables,
        required_aliases=("denominator for diluted earnings per share", "diluted earnings per share"),
        min_hits=1,
    )
    return income, balance, cashflow, eps, selected_quarterly


def _extract_aapl_statement_values(
    *,
    income_table: pd.DataFrame | None,
    balance_table: pd.DataFrame | None,
    cashflow_table: pd.DataFrame | None,
    eps_table: pd.DataFrame | None,
) -> dict[str, Any]:
    revenue = _extract_first_numeric_after_label(income_table, ("net sales", "revenue", "total net sales"))
    gross_profit = _extract_first_numeric_after_label(income_table, ("gross margin", "gross profit"))
    cogs = _extract_first_numeric_after_label(income_table, ("cost of sales", "cost of revenue"))
    sga = _extract_first_numeric_after_label(income_table, ("selling, general, and administrative",))
    operating_income = _extract_first_numeric_after_label(
        income_table,
        ("operating income", "operating income (loss)", "operating loss", "total operating income", "total operating loss"),
    )
    pretax_income = _extract_first_numeric_after_label(
        income_table,
        (
            "income before provision for income taxes",
            "income before income taxes",
            "income before taxes",
            "income (loss) before provision for (benefit from) income taxes",
            "income (loss) before provision for income taxes",
        ),
    )
    tax_expense = _extract_first_numeric_after_label(
        income_table,
        (
            "provision for income taxes",
            "income tax provision",
            "income taxes",
            "provision (benefit) for income taxes",
            "benefit from income taxes",
        ),
    )
    net_income = _extract_first_numeric_after_label(income_table, ("net income", "net income (loss)"))

    cash = _extract_first_numeric_after_label(balance_table, ("cash and cash equivalents",))
    ar = _extract_first_numeric_after_label(balance_table, ("accounts receivable",))
    inventory = _extract_first_numeric_after_label(balance_table, ("inventories", "inventory"))
    ap = _extract_first_numeric_after_label(balance_table, ("accounts payable",))
    current_assets = _extract_first_numeric_after_label(balance_table, ("total current assets",))
    current_liabilities = _extract_first_numeric_after_label(balance_table, ("total current liabilities",))
    total_assets = _extract_first_numeric_after_label(balance_table, ("total assets",))
    total_liabilities = _extract_first_numeric_after_label(balance_table, ("total liabilities",))
    equity = _extract_first_numeric_after_label(
        balance_table,
        ("total shareholders' equity", "shareholders' equity", "total stockholders' equity", "stockholders' equity"),
    )

    cfo_ytd = _extract_first_numeric_after_label(
        cashflow_table,
        (
            "cash generated by operating activities",
            "cash generated by (used for) operating activities",
            "cash provided by operating activities",
            "net cash provided by operating activities",
            "net cash provided by (used in) operating activities",
        ),
    )
    cfi_ytd = _extract_first_numeric_after_label(
        cashflow_table,
        (
            "cash used in investing activities",
            "cash generated by (used for) investing activities",
            "net cash used in investing activities",
            "net cash provided by (used in) investing activities",
        ),
    )
    cff_ytd = _extract_first_numeric_after_label(
        cashflow_table,
        (
            "cash generated by financing activities",
            "cash generated by (used for) financing activities",
            "net cash provided by financing activities",
            "net cash provided by (used in) financing activities",
        ),
    )
    capex_ytd = _extract_first_numeric_after_label(
        cashflow_table,
        (
            "payment for acquisition of property, plant, and equipment",
            "payments for acquisition of property, plant and equipment",
            "payment for acquisition of property, plant and equipment",
            "payment for acquisition of property plant and equipment",
            "purchase of property, plant, and equipment",
            "purchases of property, plant, and equipment",
            "purchase of property, plant and equipment",
            "purchases of property, plant and equipment",
        ),
    )

    basic_shares = _extract_first_numeric_after_label(eps_table, ("weighted-average shares outstanding",))
    diluted_shares = _extract_first_numeric_after_label(eps_table, ("denominator for diluted earnings per share",))
    basic_eps = _extract_first_numeric_after_label(eps_table, ("basic earnings per share",))
    diluted_eps = _extract_first_numeric_after_label(eps_table, ("diluted earnings per share",))
    shares = diluted_shares if diluted_shares is not None else basic_shares
    eps = diluted_eps if diluted_eps is not None else basic_eps

    return {
        "Revenue": revenue,
        "COGS": cogs,
        "Gross Profit": gross_profit,
        "SG&A": sga,
        "Operating Income": operating_income,
        "Pretax Income": pretax_income,
        "Tax": tax_expense,
        "Net Income": net_income,
        "Net Income Common": net_income,
        "Cash": cash,
        "AR": ar,
        "AP": ap,
        "Inventory": inventory,
        "Current Assets": current_assets,
        "Current Liabilities": current_liabilities,
        "Total Assets": total_assets,
        "Total Liabilities": total_liabilities,
        "Shareholders Equity": equity,
        "Operating Cash Flow": cfo_ytd,
        "Investing Cash Flow": cfi_ytd,
        "Financing Cash Flow": cff_ytd,
        "Capital Expenditure": abs(capex_ytd) if capex_ytd is not None else None,
        "Basic Shares": basic_shares,
        "Diluted Shares": diluted_shares,
        "Shares": shares,
        "EPS": eps,
        "Diluted EPS": diluted_eps,
        "__basic_eps": basic_eps,
        "__cf_months": _html_cashflow_months(cashflow_table),
        "__cfo_ytd": cfo_ytd,
        "__cfi_ytd": cfi_ytd,
        "__cff_ytd": cff_ytd,
        "__capex_ytd": abs(capex_ytd) if capex_ytd is not None else None,
    }


def _extract_aapl_selected_q4_values(selected_quarterly_table: pd.DataFrame | None) -> dict[str, Any]:
    if selected_quarterly_table is None or selected_quarterly_table.empty:
        return {}

    col_labels = [_normalize_html_label(col) for col in selected_quarterly_table.columns]
    q4_start = next((idx for idx, label in enumerate(col_labels) if "fourth quarter" in label), None)

    def _extract_q4_metric(*aliases: str) -> float | None:
        want = tuple(_normalize_html_label(alias) for alias in aliases)
        for _, row in selected_quarterly_table.iterrows():
            row_labels = [_normalize_html_label(v) for v in row.iloc[: min(3, len(row))]]
            if not any(label in want for label in row_labels if label):
                continue
            if q4_start is not None:
                for cell in row.iloc[q4_start : min(len(row), q4_start + 3)]:
                    parsed = _parse_html_numeric(cell)
                    if parsed is not None:
                        return parsed
            for cell in row.iloc[min(3, len(row)) :]:
                parsed = _parse_html_numeric(cell)
                if parsed is not None:
                    return parsed
        return None

    revenue = _extract_q4_metric("net sales")
    gross_profit = _extract_q4_metric("gross margin", "gross profit")
    sga = _extract_q4_metric("selling, general, and administrative")
    operating_income = _extract_q4_metric("operating income", "operating income (loss)")
    pretax_income = _extract_q4_metric(
        "income before provision for income taxes",
        "income before income taxes",
        "income before taxes",
    )
    tax_expense = _extract_q4_metric(
        "provision for income taxes",
        "income tax provision",
        "income taxes",
    )
    net_income = _extract_q4_metric("net income")
    basic_eps = _extract_q4_metric("basic")
    diluted_eps = _extract_q4_metric("diluted")
    basic_shares = None
    diluted_shares = None
    if net_income is not None and basic_eps not in (None, 0):
        basic_shares = (net_income * 1_000.0) / basic_eps
    if net_income is not None and diluted_eps not in (None, 0):
        diluted_shares = (net_income * 1_000.0) / diluted_eps
    return {
        "Revenue": revenue,
        "COGS": (revenue - gross_profit) if revenue is not None and gross_profit is not None else None,
        "Gross Profit": gross_profit,
        "SG&A": sga,
        "Operating Income": operating_income,
        "Pretax Income": pretax_income,
        "Tax": tax_expense,
        "Net Income": net_income,
        "Net Income Common": net_income,
        "Basic Shares": basic_shares,
        "Diluted Shares": diluted_shares,
        "Shares": diluted_shares if diluted_shares is not None else basic_shares,
        "EPS": diluted_eps if diluted_eps is not None else basic_eps,
        "Diluted EPS": diluted_eps,
        "__basic_eps": basic_eps,
    }


def _derive_aapl_pre_xbrl_cashflow_quarters(records: pd.DataFrame) -> pd.DataFrame:
    if records is None or records.empty:
        return records if isinstance(records, pd.DataFrame) else pd.DataFrame()
    out = records.copy()
    out = out.sort_values(["period_end", "filing_date", "accepted_at"], ascending=[True, True, True], na_position="last")
    quarter_metrics = [
        ("Operating Cash Flow", "__cfo_ytd"),
        ("Investing Cash Flow", "__cfi_ytd"),
        ("Financing Cash Flow", "__cff_ytd"),
        ("Capital Expenditure", "__capex_ytd"),
    ]
    for idx in out.index:
        months = pd.to_numeric(pd.Series([out.at[idx, "__cf_months"]]), errors="coerce").iloc[0]
        for target_col, ytd_col in quarter_metrics:
            current = pd.to_numeric(pd.Series([out.at[idx, ytd_col]]), errors="coerce").iloc[0]
            if not np.isfinite(current):
                out.at[idx, target_col] = np.nan
                continue
            if not np.isfinite(months) or int(months) <= 3:
                out.at[idx, target_col] = current
                continue
            prev_rows = out.loc[out.index < idx].copy()
            prev_rows = prev_rows.loc[pd.to_datetime(prev_rows.get("period_end"), errors="coerce") < pd.to_datetime(out.at[idx, "period_end"], errors="coerce")]
            prev_rows = prev_rows.loc[pd.to_numeric(prev_rows.get("__cf_months"), errors="coerce").isin([3, 6, 9])]
            prev_rows = prev_rows.sort_values(["period_end", "filing_date", "accepted_at"], ascending=[False, False, False], na_position="last")
            prev_ytd = np.nan
            if not prev_rows.empty:
                prev_ytd = pd.to_numeric(prev_rows.iloc[0].get(ytd_col), errors="coerce")
            out.at[idx, target_col] = current - prev_ytd if np.isfinite(prev_ytd) else current
    return out


def _fill_aapl_pre_xbrl_q4_from_annual(
    quarter_records: pd.DataFrame,
    annual_records: pd.DataFrame,
    q4_direct_records: pd.DataFrame,
) -> pd.DataFrame:
    out = quarter_records.copy() if quarter_records is not None and not quarter_records.empty else pd.DataFrame()
    if annual_records is None or annual_records.empty:
        return out

    annual_sorted = annual_records.sort_values(["period_end", "filing_date", "accepted_at"], ascending=[True, True, True], na_position="last")
    q4_direct_sorted = q4_direct_records.sort_values(["period_end", "filing_date", "accepted_at"], ascending=[True, True, True], na_position="last") if q4_direct_records is not None and not q4_direct_records.empty else pd.DataFrame()
    q4_rows: list[dict[str, Any]] = []

    flow_cols = [
        "Revenue",
        "COGS",
        "Gross Profit",
        "SG&A",
        "Operating Income",
        "Net Income",
        "Operating Cash Flow",
        "Investing Cash Flow",
        "Financing Cash Flow",
        "Capital Expenditure",
    ]
    stock_cols = [
        "Cash",
        "AR",
        "AP",
        "Inventory",
        "Current Assets",
        "Current Liabilities",
        "Total Assets",
        "Total Liabilities",
        "Shareholders Equity",
    ]

    for _, annual in annual_sorted.iterrows():
        period_end = pd.to_datetime(annual.get("period_end"), errors="coerce")
        if pd.isna(period_end):
            continue
        prior = out.loc[
            (pd.to_datetime(out.get("period_end"), errors="coerce") < period_end)
            & (pd.to_datetime(out.get("period_end"), errors="coerce") >= (period_end - pd.Timedelta(days=370)))
        ].copy()
        prior = prior.sort_values(["period_end", "filing_date", "accepted_at"], ascending=[True, True, True], na_position="last")
        prior = prior.drop_duplicates(subset=["period_end"], keep="last").tail(3)
        if prior.shape[0] < 3:
            continue

        row: dict[str, Any] = {
            "period_end": period_end,
            "form_type": str(annual.get("form_type", "10-K")),
            "filing_date": annual.get("filing_date"),
            "accepted_at": annual.get("accepted_at"),
            "available_date": annual.get("available_date"),
            "availability_method": annual.get("availability_method"),
        }
        direct = pd.DataFrame()
        if not q4_direct_sorted.empty:
            direct = q4_direct_sorted.loc[q4_direct_sorted["period_end"] == period_end].sort_values(
                ["filing_date", "accepted_at"], ascending=[True, True], na_position="last"
            )
        if not direct.empty:
            row.update({k: v for k, v in direct.iloc[-1].to_dict().items() if pd.notna(v)})

        for col in flow_cols:
            annual_val = pd.to_numeric(pd.Series([annual.get(col)]), errors="coerce").iloc[0]
            prior_sum = pd.to_numeric(prior.get(col), errors="coerce").sum(min_count=1)
            if np.isfinite(annual_val) and np.isfinite(prior_sum):
                if pd.isna(row.get(col)):
                    row[col] = annual_val - prior_sum
            elif np.isfinite(annual_val) and prior[col].notna().sum() == 0 and pd.isna(row.get(col)):
                row[col] = annual_val

        for col in stock_cols:
            if pd.isna(row.get(col)):
                row[col] = annual.get(col)

        if pd.isna(row.get("COGS")):
            revenue = pd.to_numeric(pd.Series([row.get("Revenue")]), errors="coerce").iloc[0]
            gross_profit = pd.to_numeric(pd.Series([row.get("Gross Profit")]), errors="coerce").iloc[0]
            if np.isfinite(revenue) and np.isfinite(gross_profit):
                row["COGS"] = revenue - gross_profit

        if pd.isna(row.get("Basic Shares")):
            net_income = pd.to_numeric(pd.Series([row.get("Net Income")]), errors="coerce").iloc[0]
            basic_eps = pd.to_numeric(pd.Series([row.get("__basic_eps")]), errors="coerce").iloc[0]
            if np.isfinite(net_income) and np.isfinite(basic_eps) and basic_eps != 0:
                row["Basic Shares"] = (net_income * 1_000.0) / basic_eps

        if pd.isna(row.get("Diluted Shares")):
            net_income = pd.to_numeric(pd.Series([row.get("Net Income")]), errors="coerce").iloc[0]
            diluted_eps = pd.to_numeric(pd.Series([row.get("Diluted EPS")]), errors="coerce").iloc[0]
            if np.isfinite(net_income) and np.isfinite(diluted_eps) and diluted_eps != 0:
                row["Diluted Shares"] = (net_income * 1_000.0) / diluted_eps

        if pd.isna(row.get("Shares")):
            row["Shares"] = row.get("Diluted Shares") if pd.notna(row.get("Diluted Shares")) else row.get("Basic Shares")

        q4_rows.append(row)

    if not q4_rows:
        return out
    q4_frame = pd.DataFrame(q4_rows)
    if out.empty:
        return q4_frame.reset_index(drop=True)
    return pd.concat([out, q4_frame], ignore_index=True, sort=False)


def _finalize_aapl_pre_xbrl_backfill_rows(
    rows: pd.DataFrame,
    *,
    ticker: str,
    min_date: pd.Timestamp,
    price_series: pd.Series,
    split_series: pd.Series | None,
) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame()

    out = rows.copy()
    out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce").dt.normalize()
    out["filing_date"] = pd.to_datetime(out.get("filing_date"), errors="coerce")
    out["accepted_at"] = pd.to_datetime(out.get("accepted_at"), errors="coerce")
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce")
    out = out.loc[(out["period_end"].notna()) & (out["period_end"] >= min_date)].copy()
    if out.empty:
        return pd.DataFrame()

    money_cols = [
        "Revenue",
        "COGS",
        "Gross Profit",
        "SG&A",
        "Operating Income",
        "Pretax Income",
        "Tax",
        "Net Income",
        "Net Income Common",
        "Operating Cash Flow",
        "Investing Cash Flow",
        "Financing Cash Flow",
        "Capital Expenditure",
        "Total Assets",
        "Total Liabilities",
        "Shareholders Equity",
        "Current Assets",
        "Current Liabilities",
        "AR",
        "AP",
        "Inventory",
        "Cash",
    ]
    for col in money_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * 1_000_000.0
    for col in ("Shares", "Diluted Shares", "Basic Shares"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * 1_000.0

    quarter_index = pd.DatetimeIndex(out["period_end"])
    out["Shares"] = _normalize_shares_to_price_basis(
        shares=_normalize_share_series(pd.to_numeric(out.get("Shares"), errors="coerce")),
        quarter_index=quarter_index,
        split_series=split_series,
    ).to_numpy()
    out["Diluted Shares"] = _normalize_share_series(pd.to_numeric(out.get("Diluted Shares"), errors="coerce")).to_numpy()
    out["Basic Shares"] = _normalize_share_series(pd.to_numeric(out.get("Basic Shares"), errors="coerce")).to_numpy()
    out["Price"] = _align_quarter_prices(price_series, quarter_index).to_numpy()
    out["Price_M1"] = np.nan
    out["Price_M2"] = np.nan
    out["Price_M3"] = np.nan

    assets = pd.to_numeric(out.get("Total Assets"), errors="coerce")
    liabilities = pd.to_numeric(out.get("Total Liabilities"), errors="coerce")
    equity = pd.to_numeric(out.get("Shareholders Equity"), errors="coerce")
    assets, liabilities, equity = _enforce_balance_identity(assets=assets, liabilities=liabilities, equity=equity)
    out["Total Assets"] = assets
    out["Total Liabilities"] = liabilities
    out["Shareholders Equity"] = equity

    out["symbol"] = str(ticker).strip().upper()
    out["term"] = [_to_term(ts) for ts in quarter_index]
    out["StatementDate"] = quarter_index
    out["end_date"] = quarter_index
    out["PeriodEnd"] = quarter_index
    out["PeriodStart"] = pd.NaT
    out["FormType"] = out.get("form_type")
    out["FilingDate"] = out.get("filing_date")
    out["AcceptedAt"] = out.get("accepted_at")
    out["AvailableDate"] = out.get("available_date")
    out["AvailabilityMethod"] = out.get("availability_method", "filing_date")
    out["name"] = pd.NA
    out["name_kr"] = pd.NA
    out["sector"] = pd.NA
    out["industry"] = pd.NA
    out["avg_volume"] = pd.NA
    out["Source"] = "sec_html_pre_xbrl_compatibility"
    out["CollectedAt"] = now_utc_iso()
    out["RequestedStart"] = min_date
    out["ExtractorVersion"] = SEC_EXTRACTOR_VERSION
    out["diluted_eps"] = pd.to_numeric(out.get("Diluted EPS"), errors="coerce")
    out["diluted_shares"] = pd.to_numeric(out.get("Diluted Shares"), errors="coerce")
    out["basic_shares"] = pd.to_numeric(out.get("Basic Shares"), errors="coerce")
    out["net_income_common"] = pd.to_numeric(out.get("Net Income Common"), errors="coerce")
    out["eps_source"] = "pre_xbrl_html_compatibility"

    keep_cols = [
        "symbol",
        "term",
        "StatementDate",
        *FLOW_COLUMNS,
        *STOCK_COLUMNS,
        *EXTRA_COLUMNS,
        *META_COLUMNS,
        "CollectedAt",
        "RequestedStart",
        "ExtractorVersion",
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = np.nan
    out = out[keep_cols]
    out = _collapse_quarterly_period_rows(out)
    return out.sort_values(["StatementDate", "AvailableDate", "FilingDate", "AcceptedAt", "term"]).reset_index(drop=True)


def _build_aapl_pre_xbrl_html_backfill_frame(
    *,
    ticker: str,
    market: str,
    cik: int,
    min_date: pd.Timestamp,
    submissions: dict[str, Any] | None,
    user_agent: str | None,
    force_refresh: bool,
    cache_only: bool,
    retries: int,
    backoff: float,
    price_series: pd.Series,
    split_series: pd.Series | None,
    raw_cache_dir: Path | None,
    filings_cache_dir: Path | None,
    use_next_trading_day_availability: bool,
    availability_fallback: bool,
    fallback_q_days: int,
    fallback_k_days: int,
) -> pd.DataFrame:
    symbol = str(ticker).strip().upper()
    if symbol not in SEC_PRE_XBRL_HTML_PILOT_TICKERS:
        return pd.DataFrame()

    filings = _load_full_submissions_history(
        submissions=submissions,
        ticker=symbol,
        market=market,
        cik=cik,
        user_agent=user_agent,
        force_refresh=force_refresh,
        cache_only=cache_only,
        retries=retries,
        backoff=backoff,
        raw_cache_dir=raw_cache_dir,
    )
    if filings.empty:
        return pd.DataFrame()

    report_dates = pd.to_datetime(filings.get("report_date"), errors="coerce")
    filings = filings.loc[
        (report_dates >= min_date)
        & (report_dates <= min(SEC_AAPL_HTML_FALLBACK_END, SEC_PRE_2012_COMPATIBILITY_END))
    ].copy()
    filings = filings.sort_values(["report_date", "filing_date", "accepted_at"], ascending=[True, True, True], na_position="last")
    if filings.empty:
        return pd.DataFrame()

    trading_days = _load_trading_days_for_market(market=str(market).strip().lower())
    quarter_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    q4_direct_rows: list[dict[str, Any]] = []

    for _, filing in filings.iterrows():
        accession = str(filing.get("accession", "")).strip()
        if not accession:
            continue
        primary_doc_url = str(filing.get("primary_doc_url", "")).strip()
        if not primary_doc_url:
            continue

        filing_cache_dir = (
            filings_cache_dir / sanitize_ticker(symbol) / accession.replace("-", "")
            if filings_cache_dir is not None
            else None
        )
        primary_doc_name = Path(primary_doc_url).name or "primary_doc.html"
        try:
            html_text = _load_cached_or_fetch_text(
                url=primary_doc_url,
                cache_path=(filing_cache_dir / primary_doc_name) if filing_cache_dir is not None else None,
                user_agent=user_agent,
                force_refresh=force_refresh,
                cache_only=cache_only,
                retries=retries,
                backoff=backoff,
                label=f"sec:html-fallback-doc:{symbol}:{accession}",
            )
        except Exception:
            continue

        income_table, balance_table, cashflow_table, eps_table, selected_quarterly_table = _extract_aapl_pre_xbrl_tables(html_text)
        if income_table is None:
            continue

        report_date = pd.to_datetime(filing.get("report_date"), errors="coerce")
        filing_date = pd.to_datetime(filing.get("filing_date"), errors="coerce")
        accepted_at = pd.to_datetime(filing.get("accepted_at"), errors="coerce")
        form_type = str(filing.get("form_type", "")).strip().upper() or _infer_form_type(report_date)
        available_date, availability_method = _coerce_available_date(
            filing_date=filing_date,
            accepted_at=accepted_at,
            period_end=report_date,
            form_type=form_type,
            use_next_trading_day=use_next_trading_day_availability,
            trading_days=trading_days,
            fallback_enabled=availability_fallback,
            fallback_q_days=fallback_q_days,
            fallback_k_days=fallback_k_days,
        )

        base = {
            "period_end": pd.Timestamp(report_date).normalize() if pd.notna(report_date) else pd.NaT,
            "form_type": form_type,
            "filing_date": filing_date.normalize() if pd.notna(filing_date) else pd.NaT,
            "accepted_at": accepted_at,
            "available_date": available_date,
            "availability_method": availability_method,
        }
        parsed = _extract_aapl_statement_values(
            income_table=income_table,
            balance_table=balance_table,
            cashflow_table=cashflow_table,
            eps_table=eps_table,
        )
        if form_type.startswith("10-Q"):
            quarter_rows.append({**base, **parsed})
            continue

        annual_rows.append({**base, **parsed})
        q4_direct = _extract_aapl_selected_q4_values(selected_quarterly_table)
        if q4_direct:
            q4_direct_rows.append({**base, **q4_direct})

    quarter_df = pd.DataFrame(quarter_rows)
    annual_df = pd.DataFrame(annual_rows)
    q4_df = pd.DataFrame(q4_direct_rows)

    if not quarter_df.empty:
        quarter_df = _derive_aapl_pre_xbrl_cashflow_quarters(quarter_df)
    combined = _fill_aapl_pre_xbrl_q4_from_annual(quarter_df, annual_df, q4_df)
    return _finalize_aapl_pre_xbrl_backfill_rows(
        combined,
        ticker=symbol,
        min_date=min_date,
        price_series=price_series,
        split_series=split_series,
    )


def _merge_quarterly_html_fallback_rows(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if primary is None or primary.empty:
        return fallback.copy() if isinstance(fallback, pd.DataFrame) else pd.DataFrame()
    if fallback is None or fallback.empty:
        return primary.copy()

    merged_input = pd.concat(
        [
            primary.assign(__fallback_rank=0),
            fallback.assign(__fallback_rank=1),
        ],
        ignore_index=True,
        sort=False,
    )
    for col in ("StatementDate", "PeriodEnd", "PeriodStart", "FilingDate", "AcceptedAt", "AvailableDate"):
        if col in merged_input.columns:
            merged_input[col] = pd.to_datetime(merged_input[col], errors="coerce")

    key_cols = [col for col in ["term", "PeriodEnd", "AvailableDate", "FilingDate", "FormType"] if col in merged_input.columns]
    if not key_cols:
        out = merged_input.drop(columns="__fallback_rank", errors="ignore")
        return out.sort_values(["StatementDate", "AvailableDate", "FilingDate", "AcceptedAt", "term"]).reset_index(drop=True)

    rows: list[pd.Series] = []
    for _, chunk in merged_input.groupby(key_cols, dropna=False, sort=False):
        ordered = chunk.sort_values("__fallback_rank", kind="stable")
        row = ordered.iloc[0].copy()
        for col in ordered.columns:
            if col == "__fallback_rank":
                continue
            if pd.notna(row.get(col)):
                continue
            vals = ordered[col].dropna()
            if not vals.empty:
                row[col] = vals.iloc[0]
        rows.append(row.drop(labels="__fallback_rank"))

    out = pd.DataFrame(rows)
    return out.sort_values(["StatementDate", "AvailableDate", "FilingDate", "AcceptedAt", "term"]).reset_index(drop=True)


def _extract_instance_doc_name_from_index_json(index_text: str, *, primary_doc_name: str = "") -> str | None:
    if not index_text:
        return None
    try:
        payload = json.loads(index_text)
    except Exception:
        return None

    directory = payload.get("directory", {})
    items = directory.get("item", []) if isinstance(directory, dict) else []
    if not isinstance(items, list):
        return None

    names: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    if not names:
        return None

    xml_names = []
    for name in names:
        lower = name.lower()
        if not lower.endswith(".xml") or lower.endswith(".xsd"):
            continue
        if re.search(r"_(cal|def|lab|pre)\.xml$", lower):
            continue
        xml_names.append(name)
    if not xml_names:
        return None

    primary_stem = Path(primary_doc_name).stem.lower() if primary_doc_name else ""
    preferred = [n for n in xml_names if n.lower().endswith("_htm.xml")]
    if primary_stem:
        preferred.extend([n for n in xml_names if primary_stem and primary_stem in n.lower()])
    if preferred:
        return preferred[0]
    return xml_names[0]


def _resolve_cached_instance_path(filing_cache_dir: Path, instance_doc_name: str | None) -> Path:
    if instance_doc_name:
        named_path = filing_cache_dir / str(instance_doc_name).strip()
        if named_path.exists():
            return named_path
        return named_path
    generic_path = filing_cache_dir / "instance.xml"
    return generic_path


def _local_tag_name(tag_name: str | None) -> str:
    if not tag_name:
        return ""
    name = str(tag_name)
    if ":" in name:
        name = name.split(":", 1)[1]
    return name.lower()


def _parse_numeric_text(raw_text: Any, *, sign_attr: str | None = None, scale_attr: Any = None) -> float | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    text = text.replace("\u00a0", "").replace(",", "").replace("$", "").replace("¥", "").replace("₩", "")
    text = text.replace("%", "").strip()
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        val = float(text)
    except Exception:
        return None
    if sign_attr == "-":
        val *= -1.0
    try:
        if scale_attr is not None and str(scale_attr).strip() != "":
            val *= float(10 ** int(scale_attr))
    except Exception:
        pass
    if not np.isfinite(val):
        return None
    return float(val)


def _segment_metric_from_concept(concept_name: str) -> str | None:
    local = str(concept_name or "").lower()
    if ":" in local:
        local = local.split(":", 1)[1]
    if (
        "operatingincomeloss" in local
        or "operatingprofitloss" in local
        or "incomefromoperations" in local
        or "segmentoperatingincomeloss" in local
        or "segmentoperatingprofitloss" in local
    ):
        return "operating_income"
    if "revenue" in local or "sales" in local:
        # Exclude non-segment or unrelated revenue-like concepts.
        if any(tok in local for tok in ("deferred", "liability", "recognized")):
            return None
        return "revenue"
    return None


def _segment_metric_from_table_label(label: str) -> str | None:
    local = re.sub(r"[^a-z0-9]+", " ", str(label or "").strip().lower())
    if not local:
        return None
    if any(
        token in local
        for token in (
            "operating income",
            "income from operations",
            "operating profit",
            "operating earnings",
            "segment operating income",
        )
    ):
        return "operating_income"
    if any(token in local for token in ("revenue", "net sales", "sales")):
        if any(token in local for token in ("deferred", "recognized", "liability")):
            return None
        return "revenue"
    return None


def _looks_like_periodish_label(label: str) -> bool:
    text = str(label or "").strip().lower()
    if not text:
        return False
    if re.search(r"\b(20\d{2}|19\d{2}|q[1-4]|quarter|three months|six months|nine months|twelve months|month ended|months ended|year ended|years ended)\b", text):
        return True
    return False


def _segment_member_score(dimension: str, member: str) -> int:
    dim = str(dimension or "").lower()
    mem = str(member or "").lower()
    text = f"{dim} {mem}"
    text_nospace = text.replace(" ", "").replace("_", "").replace("-", "")
    if any(tok in text_nospace for tok in SEGMENT_MEMBER_EXCLUDE_TOKENS):
        return -100
    score = 0
    if any(tok in text for tok in SEGMENT_MEMBER_GEO_TOKENS):
        score += 20
    if any(tok in text for tok in SEGMENT_MEMBER_PRODUCT_TOKENS):
        score += 20
    if "segment" in text or "business" in text:
        score += 10
    if "axis" in dim and any(tok in dim for tok in ("segment", "business", "product", "geograph", "region")):
        score += 8
    if mem.startswith("country:"):
        score += 18
    return score


def _pick_segment_candidate(members: list[tuple[str, str]]) -> tuple[str, str] | None:
    best: tuple[int, tuple[str, str] | None] = (-10_000, None)
    for dim, mem in members:
        score = _segment_member_score(dim, mem)
        if score > best[0]:
            best = (score, (dim, mem))
    if best[0] <= 0:
        return None
    return best[1]


def _extract_ixbrl_segment_facts_from_html(
    *,
    html_text: str,
    ticker: str,
    market: str,
    accession: str,
    form_type: str,
    filing_date: pd.Timestamp,
    accepted_at: pd.Timestamp,
    report_date: pd.Timestamp,
    use_next_trading_day_availability: bool,
    availability_fallback: bool,
    fallback_q_days: int,
    fallback_k_days: int,
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not html_text:
        return pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
    parser = "xml" if "<xbrl" in html_text[:2000].lower() or html_text.lstrip().startswith("<?xml") else "lxml"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html_text, parser)
    if parser == "xml":
        # Some inline docs contain undeclared prefixes in HTML-like wrappers;
        # fallback to lxml HTML parser when XML parsing yields no ix facts.
        has_nonfraction = bool(
            soup.find(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "nonfraction")
        )
        if not has_nonfraction:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                soup = BeautifulSoup(html_text, "lxml")

    def _tag_attr(tag: Any, name: str) -> Any:
        if tag is None:
            return None
        val = tag.get(name)
        if val is not None:
            return val
        lname = str(name).lower()
        for k, v in getattr(tag, "attrs", {}).items():
            if str(k).lower() == lname:
                return v
        return None

    context_map: dict[str, dict[str, Any]] = {}
    for ctx in soup.find_all(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "context"):
        ctx_id = str(_tag_attr(ctx, "id") or "").strip()
        if not ctx_id:
            continue
        period_end = pd.NaT
        period_start = pd.NaT
        period = ctx.find(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "period")
        if period is not None:
            instant = period.find(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "instant")
            if instant is not None:
                period_end = pd.to_datetime(instant.get_text(" ", strip=True), errors="coerce")
            else:
                end_dt = period.find(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "enddate")
                start_dt = period.find(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "startdate")
                if end_dt is not None:
                    period_end = pd.to_datetime(end_dt.get_text(" ", strip=True), errors="coerce")
                if start_dt is not None:
                    period_start = pd.to_datetime(start_dt.get_text(" ", strip=True), errors="coerce")
        members: list[tuple[str, str]] = []
        for em in ctx.find_all(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "explicitmember"):
            members.append((str(_tag_attr(em, "dimension") or "").strip(), em.get_text(" ", strip=True)))
        context_map[ctx_id] = {"period_end": period_end, "period_start": period_start, "members": members}

    rows: list[dict[str, Any]] = []
    for fact in soup.find_all(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "nonfraction"):
        concept = str(_tag_attr(fact, "name") or "").strip()
        metric = _segment_metric_from_concept(concept)
        if metric is None:
            continue
        context_ref = str(_tag_attr(fact, "contextref") or "").strip()
        context = context_map.get(context_ref, {})
        members = context.get("members", []) or []
        if not members:
            continue
        candidate = _pick_segment_candidate(members)
        if candidate is None:
            continue
        dim, member = candidate
        member_human = _humanize_segment_member(member)
        segment_type = _segment_type_from_dimension(dim, member_human or member)
        value = _parse_numeric_text(
            fact.get_text(" ", strip=True),
            sign_attr=str(_tag_attr(fact, "sign") or "").strip(),
            scale_attr=_tag_attr(fact, "scale"),
        )
        if value is None:
            continue
        period_end = pd.to_datetime(context.get("period_end"), errors="coerce")
        period_start = pd.to_datetime(context.get("period_start"), errors="coerce")
        if pd.isna(period_end):
            period_end = pd.to_datetime(report_date, errors="coerce")
        if pd.isna(period_end):
            continue
        available_date, availability_method = _coerce_available_date(
            filing_date=pd.to_datetime(filing_date, errors="coerce"),
            accepted_at=pd.to_datetime(accepted_at, errors="coerce"),
            period_end=pd.to_datetime(period_end, errors="coerce"),
            form_type=str(form_type or ""),
            use_next_trading_day=use_next_trading_day_availability,
            trading_days=trading_days,
            fallback_enabled=availability_fallback,
            fallback_q_days=fallback_q_days,
            fallback_k_days=fallback_k_days,
        )
        rows.append(
            {
                "ticker": str(ticker).strip().upper(),
                "market": str(market).strip().lower(),
                "period_end": pd.to_datetime(period_end, errors="coerce"),
                "period_start": pd.to_datetime(period_start, errors="coerce"),
                "form_type": str(form_type or ""),
                "filing_date": pd.to_datetime(filing_date, errors="coerce"),
                "accepted_at": pd.to_datetime(accepted_at, errors="coerce"),
                "available_date": pd.to_datetime(available_date, errors="coerce"),
                "availability_method": availability_method,
                "segment_type": segment_type,
                "segment_name": member_human or str(member),
                "metric": metric,
                "value": float(value),
                "currency": str(_tag_attr(fact, "unitref") or "").split("/", 1)[0] or "USD",
                "accession": str(accession),
                "source": "ixbrl_dimension",
                "collected_at": now_utc_iso(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["period_end", "segment_type", "segment_name", "metric", "value"])
    if out.empty:
        return pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
    out = out.sort_values(["period_end", "available_date", "segment_type", "segment_name", "metric"])
    out = out.drop_duplicates(
        subset=["ticker", "market", "period_end", "segment_type", "segment_name", "metric", "accession"],
        keep="last",
    )
    for col in SEGMENT_FACT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[SEGMENT_FACT_COLUMNS].reset_index(drop=True)


def _extract_xbrl_instance_segment_facts_from_xml(
    *,
    xml_text: str,
    ticker: str,
    market: str,
    accession: str,
    form_type: str,
    filing_date: pd.Timestamp,
    accepted_at: pd.Timestamp,
    report_date: pd.Timestamp,
    use_next_trading_day_availability: bool,
    availability_fallback: bool,
    fallback_q_days: int,
    fallback_k_days: int,
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not xml_text:
        return pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(xml_text, "xml")

    def _tag_attr(tag: Any, name: str) -> Any:
        if tag is None:
            return None
        val = tag.get(name)
        if val is not None:
            return val
        lname = str(name).lower()
        for k, v in getattr(tag, "attrs", {}).items():
            if str(k).lower() == lname:
                return v
        return None

    context_map: dict[str, dict[str, Any]] = {}
    for ctx in soup.find_all(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "context"):
        ctx_id = str(_tag_attr(ctx, "id") or "").strip()
        if not ctx_id:
            continue
        period_end = pd.NaT
        period_start = pd.NaT
        period = ctx.find(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "period")
        if period is not None:
            instant = period.find(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "instant")
            if instant is not None:
                period_end = pd.to_datetime(instant.get_text(" ", strip=True), errors="coerce")
            else:
                end_dt = period.find(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "enddate")
                start_dt = period.find(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "startdate")
                if end_dt is not None:
                    period_end = pd.to_datetime(end_dt.get_text(" ", strip=True), errors="coerce")
                if start_dt is not None:
                    period_start = pd.to_datetime(start_dt.get_text(" ", strip=True), errors="coerce")
        members: list[tuple[str, str]] = []
        for em in ctx.find_all(lambda tag: _local_tag_name(getattr(tag, "name", "")) == "explicitmember"):
            members.append((str(_tag_attr(em, "dimension") or "").strip(), em.get_text(" ", strip=True)))
        context_map[ctx_id] = {"period_end": period_end, "period_start": period_start, "members": members}

    rows: list[dict[str, Any]] = []
    for fact in soup.find_all(True):
        context_ref = str(_tag_attr(fact, "contextRef") or _tag_attr(fact, "contextref") or "").strip()
        if not context_ref:
            continue
        context = context_map.get(context_ref, {})
        members = context.get("members", []) or []
        if not members:
            continue

        concept = str(getattr(fact, "name", "") or "").strip()
        metric = _segment_metric_from_concept(concept)
        if metric is None:
            continue

        candidate = _pick_segment_candidate(members)
        if candidate is None:
            continue
        dim, member = candidate
        member_human = _humanize_segment_member(member)
        segment_type = _segment_type_from_dimension(dim, member_human or member)

        value = _parse_numeric_text(
            fact.get_text(" ", strip=True),
            sign_attr=str(_tag_attr(fact, "sign") or "").strip(),
            scale_attr=_tag_attr(fact, "scale"),
        )
        if value is None:
            continue

        period_end = pd.to_datetime(context.get("period_end"), errors="coerce")
        period_start = pd.to_datetime(context.get("period_start"), errors="coerce")
        if pd.isna(period_end):
            period_end = pd.to_datetime(report_date, errors="coerce")
        if pd.isna(period_end):
            continue

        available_date, availability_method = _coerce_available_date(
            filing_date=pd.to_datetime(filing_date, errors="coerce"),
            accepted_at=pd.to_datetime(accepted_at, errors="coerce"),
            period_end=pd.to_datetime(period_end, errors="coerce"),
            form_type=str(form_type or ""),
            use_next_trading_day=use_next_trading_day_availability,
            trading_days=trading_days,
            fallback_enabled=availability_fallback,
            fallback_q_days=fallback_q_days,
            fallback_k_days=fallback_k_days,
        )
        rows.append(
            {
                "ticker": str(ticker).strip().upper(),
                "market": str(market).strip().lower(),
                "period_end": pd.to_datetime(period_end, errors="coerce"),
                "period_start": pd.to_datetime(period_start, errors="coerce"),
                "form_type": str(form_type or ""),
                "filing_date": pd.to_datetime(filing_date, errors="coerce"),
                "accepted_at": pd.to_datetime(accepted_at, errors="coerce"),
                "available_date": pd.to_datetime(available_date, errors="coerce"),
                "availability_method": availability_method,
                "segment_type": segment_type,
                "segment_name": member_human or str(member),
                "metric": metric,
                "value": float(value),
                "currency": str(_tag_attr(fact, "unitRef") or _tag_attr(fact, "unitref") or "").split("/", 1)[0] or "USD",
                "accession": str(accession),
                "source": "xbrl_instance",
                "collected_at": now_utc_iso(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["period_end", "segment_type", "segment_name", "metric", "value"])
    if out.empty:
        return pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
    out = out.sort_values(["period_end", "available_date", "segment_type", "segment_name", "metric"])
    out = out.drop_duplicates(
        subset=["ticker", "market", "period_end", "segment_type", "segment_name", "metric", "accession"],
        keep="last",
    )
    for col in SEGMENT_FACT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[SEGMENT_FACT_COLUMNS].reset_index(drop=True)


def _extract_html_table_segment_facts_mvp(
    *,
    html_text: str,
    ticker: str,
    market: str,
    accession: str,
    form_type: str,
    filing_date: pd.Timestamp,
    accepted_at: pd.Timestamp,
    report_date: pd.Timestamp,
    use_next_trading_day_availability: bool,
    availability_fallback: bool,
    fallback_q_days: int,
    fallback_k_days: int,
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not html_text:
        return pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
    try:
        tables = pd.read_html(StringIO(html_text), flavor="lxml")
    except Exception:
        return pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)

    report_end = pd.to_datetime(report_date, errors="coerce")
    available_date, availability_method = _coerce_available_date(
        filing_date=pd.to_datetime(filing_date, errors="coerce"),
        accepted_at=pd.to_datetime(accepted_at, errors="coerce"),
        period_end=report_end,
        form_type=str(form_type or ""),
        use_next_trading_day=use_next_trading_day_availability,
        trading_days=trading_days,
        fallback_enabled=availability_fallback,
        fallback_q_days=fallback_q_days,
        fallback_k_days=fallback_k_days,
    )

    rows: list[dict[str, Any]] = []

    def _numeric_series(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace(r"^\((.*)\)$", r"-\1", regex=True),
            errors="coerce",
        )

    def _append_row(
        *,
        segment_name: str,
        metric: str,
        value: float,
        source: str = "html_table_mvp",
    ) -> None:
        rows.append(
            {
                "ticker": str(ticker).strip().upper(),
                "market": str(market).strip().lower(),
                "period_end": report_end,
                "period_start": pd.NaT,
                "form_type": str(form_type or ""),
                "filing_date": pd.to_datetime(filing_date, errors="coerce"),
                "accepted_at": pd.to_datetime(accepted_at, errors="coerce"),
                "available_date": pd.to_datetime(available_date, errors="coerce"),
                "availability_method": availability_method,
                "segment_type": _segment_type_from_dimension("html_table", segment_name),
                "segment_name": segment_name,
                "metric": metric,
                "value": float(value),
                "currency": "USD",
                "accession": str(accession),
                "source": source,
                "collected_at": now_utc_iso(),
            }
        )

    for tbl in tables:
        if tbl is None or tbl.empty or tbl.shape[1] < 2:
            continue
        df = tbl.copy()
        df.columns = [str(c).strip() for c in df.columns]
        first_col = df.columns[0]
        label_series = df[first_col].astype(str).str.strip()
        # Heuristic: try only tables that include likely segment labels.
        joined = " ".join(label_series.tolist()).lower()
        joined_headers = " ".join(str(col).strip().lower() for col in df.columns[1:])
        metric_rows = {
            idx: metric
            for idx, label in label_series.items()
            for metric in [_segment_metric_from_table_label(label)]
            if metric is not None
        }
        header_segment_candidates = [
            col
            for col in df.columns[1:]
            if str(col).strip()
            and not _segment_metric_from_table_label(str(col))
            and not _looks_like_periodish_label(str(col))
            and str(col).strip().lower() not in {"total", "consolidated"}
        ]
        if not any(tok in f"{joined} {joined_headers}" for tok in ("americas", "europe", "japan", "china", "iphone", "services", "segment")):
            if not (metric_rows and len(header_segment_candidates) >= 2):
                continue
        before_len = len(rows)
        if metric_rows:
            for col in header_segment_candidates:
                numeric = _numeric_series(df[col])
                if numeric.notna().sum() == 0:
                    continue
                for idx, metric in metric_rows.items():
                    value = pd.to_numeric(numeric.get(idx), errors="coerce")
                    if pd.isna(value):
                        continue
                    _append_row(segment_name=str(col).strip(), metric=metric, value=float(value))

        metric_cols = {
            col: metric
            for col in df.columns[1:]
            for metric in [_segment_metric_from_table_label(str(col))]
            if metric is not None
        }
        if metric_cols:
            for _, row in df.iterrows():
                segment_name = str(row.get(first_col, "")).strip()
                if (
                    not segment_name
                    or _segment_metric_from_table_label(segment_name) is not None
                    or _looks_like_periodish_label(segment_name)
                    or segment_name.lower() in {"total", "consolidated"}
                ):
                    continue
                for col, metric in metric_cols.items():
                    value = pd.to_numeric(_numeric_series(pd.Series([row.get(col)])).iloc[0], errors="coerce")
                    if pd.isna(value):
                        continue
                    _append_row(segment_name=segment_name, metric=metric, value=float(value))

        if len(rows) > before_len:
            continue
        # choose first mostly-numeric column as value
        value_col: str | None = None
        for c in df.columns[1:]:
            parsed = _numeric_series(df[c])
            if parsed.notna().sum() >= max(2, len(df) // 3):
                value_col = c
                df[c] = parsed
                break
        if value_col is None:
            continue
        for _, r in df.iterrows():
            label = str(r.get(first_col, "")).strip()
            val = pd.to_numeric(r.get(value_col), errors="coerce")
            if not label or pd.isna(val):
                continue
            _append_row(segment_name=label, metric="revenue", value=float(val))
    if not rows:
        return pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(
        subset=["ticker", "market", "period_end", "segment_type", "segment_name", "metric", "accession"],
        keep="last",
    )
    for col in SEGMENT_FACT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[SEGMENT_FACT_COLUMNS].reset_index(drop=True)


def _segment_facts_to_wide(facts: pd.DataFrame) -> pd.DataFrame:
    if facts is None or facts.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    f = facts.copy()
    f["metric"] = f["metric"].astype(str).str.lower()
    f["period_end"] = pd.to_datetime(f.get("period_end"), errors="coerce")
    f["period_start"] = pd.to_datetime(f.get("period_start"), errors="coerce")
    f["filing_date"] = pd.to_datetime(f.get("filing_date"), errors="coerce")
    f["accepted_at"] = pd.to_datetime(f.get("accepted_at"), errors="coerce")
    f["available_date"] = pd.to_datetime(f.get("available_date"), errors="coerce")
    f = f.dropna(subset=["period_end", "segment_type", "segment_name", "metric", "value"])
    if f.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    key_cols = [
        "ticker",
        "market",
        "period_end",
        "period_start",
        "form_type",
        "filing_date",
        "accepted_at",
        "available_date",
        "availability_method",
        "segment_type",
        "segment_name",
        "source",
        "collected_at",
    ]
    piv = (
        f.pivot_table(index=key_cols, columns="metric", values="value", aggfunc="last")
        .reset_index()
    )
    piv.columns = [str(c) for c in piv.columns]
    if "revenue" not in piv.columns:
        piv["revenue"] = np.nan
    if "operating_income" not in piv.columns:
        piv["operating_income"] = np.nan
    out = piv.rename(columns={"operating_income": "op_income"})
    out["op_income"] = pd.to_numeric(out.get("op_income"), errors="coerce")
    for col in SEGMENT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[SEGMENT_COLUMNS].reset_index(drop=True)


def _extract_period_filing_events(
    companyfacts: dict[str, Any],
    min_date: pd.Timestamp,
    accession_map: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    accession_meta = accession_map or {}
    rows: list[dict[str, Any]] = []
    facts = companyfacts.get("facts", {})
    if not isinstance(facts, dict):
        return pd.DataFrame()

    for _ns, ns_payload in facts.items():
        if not isinstance(ns_payload, dict):
            continue
        for _tag, metric_payload in ns_payload.items():
            if not isinstance(metric_payload, dict):
                continue
            units = metric_payload.get("units", {})
            if not isinstance(units, dict):
                continue
            for _unit, unit_rows in units.items():
                if not isinstance(unit_rows, list):
                    continue
                for item in unit_rows:
                    if not isinstance(item, dict):
                        continue
                    form = str(item.get("form", "")).strip().upper()
                    if form not in SEC_ALLOWED_FORMS:
                        continue
                    end_dt = pd.to_datetime(item.get("end"), errors="coerce")
                    if pd.isna(end_dt):
                        continue
                    period_end = _quarter_end(pd.Timestamp(end_dt))
                    if period_end < min_date:
                        continue
                    period_start = pd.to_datetime(item.get("start"), errors="coerce")
                    filed_dt = pd.to_datetime(item.get("filed"), errors="coerce")
                    accn = _normalize_accession(item.get("accn"))
                    meta = accession_meta.get(accn, {}) if accn else {}
                    accepted_dt = pd.to_datetime(meta.get("accepted_at"), errors="coerce")
                    if pd.isna(filed_dt):
                        filed_dt = pd.to_datetime(meta.get("filing_date"), errors="coerce")
                    report_dt = pd.to_datetime(meta.get("report_date"), errors="coerce")
                    if pd.notna(report_dt):
                        period_end = _quarter_end(pd.Timestamp(report_dt))
                    rows.append(
                        {
                            "PeriodEnd": period_end,
                            "PeriodStart": period_start,
                            "FormType": _infer_form_type(period_end, form),
                            "FilingDate": filed_dt.normalize() if pd.notna(filed_dt) else pd.NaT,
                            "AcceptedAt": accepted_dt,
                            "accession": accn,
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=["PeriodEnd", "PeriodStart", "FormType", "FilingDate", "AcceptedAt", "accession"])

    events = pd.DataFrame(rows)
    events["PeriodEnd"] = pd.to_datetime(events["PeriodEnd"], errors="coerce").dt.normalize()
    events["PeriodStart"] = pd.to_datetime(events["PeriodStart"], errors="coerce").dt.normalize()
    events["FilingDate"] = pd.to_datetime(events["FilingDate"], errors="coerce").dt.normalize()
    events["AcceptedAt"] = pd.to_datetime(events["AcceptedAt"], errors="coerce")
    events = events.dropna(subset=["PeriodEnd"]).copy()

    events["_sort_filed"] = pd.to_datetime(events["FilingDate"], errors="coerce")
    events["_sort_accepted"] = pd.to_datetime(events["AcceptedAt"], errors="coerce")
    events = events.sort_values(
        ["PeriodEnd", "_sort_filed", "_sort_accepted", "FormType", "accession"],
        ascending=[True, True, True, True, True],
    )
    events = events.drop(columns=["_sort_filed", "_sort_accepted"])

    dedup_keys = ["PeriodEnd", "FormType", "FilingDate", "AcceptedAt", "accession"]
    events = events.drop_duplicates(subset=dedup_keys, keep="last")
    return events.reset_index(drop=True)


def fetch_sec_quarterly_history(
    ticker: str,
    market: str = "us",
    start: str | pd.Timestamp = SEC_DEFAULT_START_DATE,
    user_agent: str | None = None,
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    map_force_refresh: bool = False,
    map_cache_path: Path = SEC_TICKER_MAP_CACHE,
    raw_cache_dir: Path | None = SEC_RAW_COMPANYFACTS_DIR,
    submissions_cache_dir: Path | None = SEC_RAW_SUBMISSIONS_DIR,
    ticker_cache_dir: Path = SEC_TICKER_QUARTERLY_DIR,
    filings_cache_dir: Path | None = SEC_FILINGS_CACHE_DIR,
    offline_mode: bool = False,
    reparse_from_cache: bool = False,
    prefetched_companyfacts: dict[str, Any] | None = None,
    prefetched_submissions: dict[str, Any] | None = None,
    prefetched_cik: int | None = None,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start, errors="coerce")
    if pd.isna(start_ts):
        start_ts = SEC_DEFAULT_START_DATE
    min_date = max(pd.Timestamp(start_ts).normalize(), SEC_DEFAULT_START_DATE)

    symbol = str(ticker).strip().upper()
    ensure_dir(ticker_cache_dir)
    cache_path = ticker_cache_dir / f"{sanitize_ticker(symbol)}.parquet"

    if cache_path.exists() and not force_refresh and not reparse_from_cache:
        try:
            cached = pd.read_parquet(cache_path)
            if not cached.empty:
                cached["StatementDate"] = pd.to_datetime(cached["StatementDate"], errors="coerce")
                cached = cached.loc[~cached["StatementDate"].isna()]
                cached_dates = pd.to_datetime(cached["StatementDate"], errors="coerce").dropna()
                cached_requested_start = pd.NaT
                if "RequestedStart" in cached.columns:
                    requested = pd.to_datetime(cached["RequestedStart"], errors="coerce")
                    requested = requested.dropna()
                    if not requested.empty:
                        cached_requested_start = pd.Timestamp(requested.min()).normalize()
                version_ok = False
                if "ExtractorVersion" in cached.columns:
                    versions = pd.to_numeric(cached["ExtractorVersion"], errors="coerce").dropna()
                    if not versions.empty:
                        version_ok = int(versions.max()) >= SEC_EXTRACTOR_VERSION

                has_min_coverage = False
                if pd.notna(cached_requested_start):
                    has_min_coverage = cached_requested_start <= min_date
                elif not cached_dates.empty:
                    # Legacy cache without RequestedStart: assume coverage only if earliest row reaches min_date.
                    has_min_coverage = pd.Timestamp(cached_dates.min()).normalize() <= min_date

                # Check for mandatory PIT columns to handle legacy cache
                mandatory_cols = {"AcceptedAt", "AvailableDate", "AvailabilityMethod", "FilingDate"}
                has_mandatory_cols = mandatory_cols.issubset(cached.columns)

                if has_min_coverage and version_ok and has_mandatory_cols:
                    cached = cached.loc[cached["StatementDate"] >= min_date]
                    if not cached.empty:
                        return cached.sort_values("StatementDate").reset_index(drop=True)
        except Exception:
            pass

    if offline_mode:
        return pd.DataFrame()

    try:
        if prefetched_companyfacts is not None and prefetched_cik is not None:
            companyfacts, cik = prefetched_companyfacts, int(prefetched_cik)
        else:
            companyfacts, cik = fetch_companyfacts(
                ticker=symbol,
                user_agent=user_agent,
                force_refresh=force_refresh and not reparse_from_cache,
                cache_only=reparse_from_cache,
                retries=retries,
                backoff=backoff,
                map_force_refresh=map_force_refresh,
                map_cache_path=map_cache_path,
                raw_cache_dir=raw_cache_dir,
            )
    except RuntimeError as e:
        if isinstance(e.__cause__, KeyError):
            print(f"[INFO] Skipping {symbol}: {e.__cause__}")
            built = _build_standard_quarterly_frame(
                ticker=symbol,
                companyfacts={},
                min_date=min_date,
                price_series=_load_price_series(symbol, market=market),
                split_series=_load_price_splits(symbol, market=market),
                issuer_company_name="",
                issuer_sic=None,
                issuer_sic_description="",
            )
            built.to_parquet(cache_path, index=False)
            return built
        raise

    submissions: dict[str, Any] = {}
    try:
        if prefetched_submissions is not None:
            submissions = prefetched_submissions
        else:
            submissions = fetch_submissions(
                ticker=symbol,
                cik=cik,
                user_agent=user_agent,
                force_refresh=force_refresh and not reparse_from_cache,
                cache_only=reparse_from_cache,
                retries=retries,
                backoff=backoff,
                raw_cache_dir=submissions_cache_dir,
            )
    except Exception:
        submissions = {}
    accession_map = _build_submissions_accession_map(submissions)
    price_series = _load_price_series(symbol, market=market)
    split_series = _load_price_splits(symbol, market=market)
    built = _build_standard_quarterly_frame(
        ticker=symbol,
        companyfacts=companyfacts,
        min_date=min_date,
        price_series=price_series,
        split_series=split_series,
        market=market,
        submissions_accession_map=accession_map,
        issuer_company_name=str(submissions.get("name") or companyfacts.get("entityName") or ""),
        issuer_sic=submissions.get("sic"),
        issuer_sic_description=str(submissions.get("sicDescription") or ""),
        use_next_trading_day_availability=use_next_trading_day_availability,
        availability_fallback=availability_fallback,
        fallback_q_days=fallback_q_days,
        fallback_k_days=fallback_k_days,
    )
    html_backfill = pd.DataFrame()
    if _compatibility_lane_requested(min_date):
        html_backfill = _build_aapl_pre_xbrl_html_backfill_frame(
            ticker=symbol,
            market=market,
            cik=cik,
            min_date=min_date,
            submissions=submissions,
            user_agent=user_agent,
            force_refresh=force_refresh,
            cache_only=reparse_from_cache,
            retries=retries,
            backoff=backoff,
            price_series=price_series,
            split_series=split_series,
            raw_cache_dir=submissions_cache_dir,
            filings_cache_dir=filings_cache_dir,
            use_next_trading_day_availability=use_next_trading_day_availability,
            availability_fallback=availability_fallback,
            fallback_q_days=fallback_q_days,
            fallback_k_days=fallback_k_days,
        )
    if not html_backfill.empty:
        built = _merge_quarterly_html_fallback_rows(built, html_backfill)
        built = _collapse_quarterly_period_rows(built)
    if "RequestedStart" not in built.columns:
        built["RequestedStart"] = min_date
    built["ExtractorVersion"] = SEC_EXTRACTOR_VERSION
    built.to_parquet(cache_path, index=False)
    return built


def build_sec_enrichment_frames(
    *,
    ticker: str,
    market: str = "us",
    start: str | pd.Timestamp = SEC_DEFAULT_START_DATE,
    quarterly_frame: pd.DataFrame | None = None,
    user_agent: str | None = None,
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    map_force_refresh: bool = False,
    map_cache_path: Path = SEC_TICKER_MAP_CACHE,
    raw_cache_dir: Path | None = SEC_RAW_COMPANYFACTS_DIR,
    submissions_cache_dir: Path | None = SEC_RAW_SUBMISSIONS_DIR,
    offline_mode: bool = False,
    reparse_from_cache: bool = False,
    prefetched_companyfacts: dict[str, Any] | None = None,
    prefetched_submissions: dict[str, Any] | None = None,
    prefetched_cik: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_ts = pd.to_datetime(start, errors="coerce")
    if pd.isna(start_ts):
        start_ts = SEC_DEFAULT_START_DATE
    min_date = max(pd.Timestamp(start_ts).normalize(), SEC_DEFAULT_START_DATE)
    symbol = str(ticker).strip().upper()
    market_norm = str(market).strip().lower()
    if quarterly_frame is None:
        quarterly = pd.DataFrame()
    else:
        quarterly = quarterly_frame.copy()

    if offline_mode:
        return pd.DataFrame(columns=RAW_FACT_COLUMNS), _build_financials_extra_from_quarterly(
            ticker=symbol,
            market=market_norm,
            quarterly=quarterly,
            raw_facts=None,
        )

    try:
        if prefetched_companyfacts is not None and prefetched_cik is not None:
            companyfacts, cik = prefetched_companyfacts, int(prefetched_cik)
        else:
            companyfacts, cik = fetch_companyfacts(
                ticker=symbol,
                user_agent=user_agent,
                force_refresh=force_refresh and not reparse_from_cache,
                cache_only=reparse_from_cache,
                retries=retries,
                backoff=backoff,
                map_force_refresh=map_force_refresh,
                map_cache_path=map_cache_path,
                raw_cache_dir=raw_cache_dir,
            )
    except Exception:
        return pd.DataFrame(columns=RAW_FACT_COLUMNS), _build_financials_extra_from_quarterly(
            ticker=symbol,
            market=market_norm,
            quarterly=quarterly,
            raw_facts=None,
        )

    submissions: dict[str, Any] = {}
    try:
        if prefetched_submissions is not None:
            submissions = prefetched_submissions
        else:
            submissions = fetch_submissions(
                ticker=symbol,
                cik=cik,
                user_agent=user_agent,
                force_refresh=force_refresh and not reparse_from_cache,
                cache_only=reparse_from_cache,
                retries=retries,
                backoff=backoff,
                raw_cache_dir=submissions_cache_dir,
            )
    except Exception:
        submissions = {}
    accession_map = _build_submissions_accession_map(submissions)
    raw_facts = _extract_raw_normalized_facts(
        ticker=symbol,
        market=market_norm,
        cik=cik,
        companyfacts=companyfacts,
        min_date=min_date,
        submissions_accession_map=accession_map,
        use_next_trading_day_availability=use_next_trading_day_availability,
        availability_fallback=availability_fallback,
        fallback_q_days=fallback_q_days,
        fallback_k_days=fallback_k_days,
    )
    extra = _build_financials_extra_from_quarterly(
        ticker=symbol,
        market=market_norm,
        quarterly=quarterly,
        raw_facts=raw_facts,
    )
    return raw_facts, extra


def fetch_sec_segment_bundle(
    ticker: str,
    market: str = "us",
    start: str | pd.Timestamp = SEC_DEFAULT_START_DATE,
    user_agent: str | None = None,
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    map_force_refresh: bool = False,
    map_cache_path: Path = SEC_TICKER_MAP_CACHE,
    raw_cache_dir: Path | None = SEC_RAW_COMPANYFACTS_DIR,
    submissions_cache_dir: Path | None = SEC_RAW_SUBMISSIONS_DIR,
    lookback_filings: int = 8,
    filings_cache_dir: Path | None = SEC_FILINGS_CACHE_DIR,
    offline_mode: bool = False,
    cache_only: bool = False,
    prefetched_companyfacts: dict[str, Any] | None = None,
    prefetched_submissions: dict[str, Any] | None = None,
    prefetched_cik: int | None = None,
) -> SecSegmentBundle:
    start_ts = pd.to_datetime(start, errors="coerce")
    if pd.isna(start_ts):
        start_ts = SEC_DEFAULT_START_DATE
    min_date = max(pd.Timestamp(start_ts).normalize(), SEC_DEFAULT_START_DATE)

    symbol = str(ticker).strip().upper()
    if offline_mode:
        return SecSegmentBundle(
            wide=pd.DataFrame(columns=SEGMENT_COLUMNS),
            facts=pd.DataFrame(columns=SEGMENT_FACT_COLUMNS),
            filings=pd.DataFrame(columns=FILING_COLUMNS),
            extract_log=pd.DataFrame(columns=SEGMENT_EXTRACT_LOG_COLUMNS),
        )

    if prefetched_companyfacts is not None and prefetched_cik is not None:
        companyfacts, cik = prefetched_companyfacts, int(prefetched_cik)
    else:
        companyfacts, cik = fetch_companyfacts(
            ticker=symbol,
            user_agent=user_agent,
            force_refresh=force_refresh and not cache_only,
            cache_only=cache_only,
            retries=retries,
            backoff=backoff,
            map_force_refresh=map_force_refresh,
            map_cache_path=map_cache_path,
            raw_cache_dir=raw_cache_dir,
        )
    submissions: dict[str, Any] = {}
    try:
        if prefetched_submissions is not None:
            submissions = prefetched_submissions
        else:
            submissions = fetch_submissions(
                ticker=symbol,
                cik=cik,
                user_agent=user_agent,
                force_refresh=force_refresh and not cache_only,
                cache_only=cache_only,
                retries=retries,
                backoff=backoff,
                raw_cache_dir=submissions_cache_dir,
            )
    except Exception:
        submissions = {}
    accession_map = _build_submissions_accession_map(submissions)

    base_wide = _build_segment_quarterly_frame(
        ticker=symbol,
        market=market,
        companyfacts=companyfacts,
        min_date=min_date,
        submissions_accession_map=accession_map,
        use_next_trading_day_availability=use_next_trading_day_availability,
        availability_fallback=availability_fallback,
        fallback_q_days=fallback_q_days,
        fallback_k_days=fallback_k_days,
    )
    base_wide = base_wide.loc[pd.to_datetime(base_wide.get("period_end"), errors="coerce") >= min_date] if not base_wide.empty else base_wide

    filings_df = _build_recent_filings_from_submissions(
        submissions,
        ticker=symbol,
        market=market,
        cik=cik,
        lookback_filings=max(1, int(lookback_filings)),
    )
    if not filings_df.empty:
        filings_df["report_date"] = pd.to_datetime(filings_df.get("report_date"), errors="coerce")
        filings_df = filings_df.loc[(filings_df["report_date"].isna()) | (filings_df["report_date"] >= min_date)].copy()

    trading_days = _load_trading_days_for_market(market=str(market).strip().lower())
    fact_frames: list[pd.DataFrame] = []
    log_rows: list[dict[str, Any]] = []

    for _, filing in filings_df.iterrows():
        accession = str(filing.get("accession", "")).strip()
        if not accession:
            continue
        accession_nodash = accession.replace("-", "")
        filing_cache_dir = (
            filings_cache_dir / sanitize_ticker(symbol) / accession_nodash
            if filings_cache_dir is not None
            else None
        )
        if filing_cache_dir is not None:
            ensure_dir(filing_cache_dir)
        parse_result: dict[str, Any] = {
            "ticker": symbol,
            "market": str(market).strip().lower(),
            "accession": accession,
            "methods": [],
            "rows": {},
            "status": "fail",
            "reason": "unknown",
            "updated_at": now_utc_iso(),
        }
        index_text = ""
        index_url = str(filing.get("index_url", "")).strip()
        if index_url:
            try:
                index_text = _load_cached_or_fetch_text(
                    url=index_url,
                    cache_path=(filing_cache_dir / "index.json") if filing_cache_dir is not None else None,
                    user_agent=user_agent,
                    force_refresh=force_refresh and not cache_only,
                    cache_only=cache_only,
                    retries=retries,
                    backoff=backoff,
                    label=f"sec:filing-index:{symbol}:{accession}",
                )
            except Exception as exc:
                log_rows.append(
                    {
                        "ticker": symbol,
                        "market": str(market).strip().lower(),
                        "accession": accession,
                        "method": "index_json",
                        "status": "fail",
                        "reason": f"index_fetch_failed:{exc.__class__.__name__}",
                        "created_at": now_utc_iso(),
                    }
                )
        primary_doc_url = str(filing.get("primary_doc_url", "")).strip()
        if not primary_doc_url:
            log_rows.append(
                {
                    "ticker": symbol,
                    "market": str(market).strip().lower(),
                    "accession": accession,
                    "method": "ixbrl_dimension",
                    "status": "fail",
                    "reason": "primary_doc_url_missing",
                    "created_at": now_utc_iso(),
                }
            )
            parse_result["reason"] = "primary_doc_url_missing"
            _write_json_cache((filing_cache_dir / "parse_result.json") if filing_cache_dir is not None else None, parse_result)
            continue
        primary_doc_name = Path(primary_doc_url).name or "primary_doc.html"
        try:
            html_text = _load_cached_or_fetch_text(
                url=primary_doc_url,
                cache_path=(filing_cache_dir / "primary_doc.html") if filing_cache_dir is not None else None,
                user_agent=user_agent,
                force_refresh=force_refresh and not cache_only,
                cache_only=cache_only,
                retries=retries,
                backoff=backoff,
                label=f"sec:filing-doc:{symbol}:{accession}",
            )
        except Exception as exc:
            log_rows.append(
                {
                    "ticker": symbol,
                    "market": str(market).strip().lower(),
                    "accession": accession,
                    "method": "ixbrl_dimension",
                    "status": "fail",
                    "reason": f"primary_doc_fetch_failed:{exc.__class__.__name__}",
                    "created_at": now_utc_iso(),
                }
            )
            parse_result["reason"] = f"primary_doc_fetch_failed:{exc.__class__.__name__}"
            _write_json_cache((filing_cache_dir / "parse_result.json") if filing_cache_dir is not None else None, parse_result)
            continue

        ix = _extract_ixbrl_segment_facts_from_html(
            html_text=html_text,
            ticker=symbol,
            market=market,
            accession=accession,
            form_type=str(filing.get("form_type", "")),
            filing_date=pd.to_datetime(filing.get("filing_date"), errors="coerce"),
            accepted_at=pd.to_datetime(filing.get("accepted_at"), errors="coerce"),
            report_date=pd.to_datetime(filing.get("report_date"), errors="coerce"),
            use_next_trading_day_availability=use_next_trading_day_availability,
            availability_fallback=availability_fallback,
            fallback_q_days=fallback_q_days,
            fallback_k_days=fallback_k_days,
            trading_days=trading_days,
        )
        ix_metrics = set(ix["metric"].dropna().astype(str).tolist()) if not ix.empty else set()
        if not ix.empty:
            fact_frames.append(ix)
            parse_result["methods"].append("ixbrl_dimension")
            parse_result["rows"]["ixbrl_dimension"] = int(len(ix))
            log_rows.append(
                {
                    "ticker": symbol,
                    "market": str(market).strip().lower(),
                    "accession": accession,
                    "method": "ixbrl_dimension",
                    "status": "success",
                    "reason": f"rows={len(ix)};metrics={','.join(sorted(ix_metrics))}",
                    "created_at": now_utc_iso(),
                }
            )
            parse_result["status"] = "success"
            parse_result["reason"] = f"ixbrl_rows={len(ix)}"

        instance_doc_name = _extract_instance_doc_name_from_index_json(
            index_text,
            primary_doc_name=primary_doc_name,
        )
        instance_facts = pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
        need_lower_confidence = (not ix_metrics) or ("operating_income" not in ix_metrics)
        if instance_doc_name and need_lower_confidence:
            archive_base = primary_doc_url.rsplit("/", 1)[0]
            instance_url = f"{archive_base}/{instance_doc_name}"
            instance_cache_path = (
                _resolve_cached_instance_path(filing_cache_dir, instance_doc_name)
                if filing_cache_dir is not None
                else None
            )
            try:
                instance_text = _load_cached_or_fetch_text(
                    url=instance_url,
                    cache_path=instance_cache_path,
                    user_agent=user_agent,
                    force_refresh=force_refresh and not cache_only,
                    cache_only=cache_only,
                    retries=retries,
                    backoff=backoff,
                    label=f"sec:filing-instance:{symbol}:{accession}",
                )
                instance_facts = _extract_xbrl_instance_segment_facts_from_xml(
                    xml_text=instance_text,
                    ticker=symbol,
                    market=market,
                    accession=accession,
                    form_type=str(filing.get("form_type", "")),
                    filing_date=pd.to_datetime(filing.get("filing_date"), errors="coerce"),
                    accepted_at=pd.to_datetime(filing.get("accepted_at"), errors="coerce"),
                    report_date=pd.to_datetime(filing.get("report_date"), errors="coerce"),
                    use_next_trading_day_availability=use_next_trading_day_availability,
                    availability_fallback=availability_fallback,
                    fallback_q_days=fallback_q_days,
                    fallback_k_days=fallback_k_days,
                    trading_days=trading_days,
                )
            except Exception as exc:
                log_rows.append(
                    {
                        "ticker": symbol,
                        "market": str(market).strip().lower(),
                        "accession": accession,
                        "method": "xbrl_instance_dimension",
                        "status": "fail",
                        "reason": f"instance_fetch_or_parse_failed:{exc.__class__.__name__}",
                        "created_at": now_utc_iso(),
                    }
                )
        else:
            log_rows.append(
                {
                    "ticker": symbol,
                    "market": str(market).strip().lower(),
                    "accession": accession,
                    "method": "xbrl_instance_dimension",
                    "status": "fail",
                    "reason": "instance_doc_not_found_in_index" if need_lower_confidence else "not_needed_after_ixbrl",
                    "created_at": now_utc_iso(),
                }
            )

        instance_metrics = set(instance_facts["metric"].dropna().astype(str).tolist()) if not instance_facts.empty else set()
        if not instance_facts.empty:
            fact_frames.append(instance_facts)
            parse_result["methods"].append("xbrl_instance_dimension")
            parse_result["rows"]["xbrl_instance_dimension"] = int(len(instance_facts))
            log_rows.append(
                {
                    "ticker": symbol,
                    "market": str(market).strip().lower(),
                    "accession": accession,
                    "method": "xbrl_instance_dimension",
                    "status": "success",
                    "reason": f"rows={len(instance_facts)};metrics={','.join(sorted(instance_metrics))}",
                    "created_at": now_utc_iso(),
                }
            )
            parse_result["status"] = "success"
            parse_result["reason"] = f"xbrl_instance_rows={len(instance_facts)}"

        combined_metrics = ix_metrics | instance_metrics
        need_html = (not combined_metrics) or ("operating_income" not in combined_metrics)
        html_tbl = (
            _extract_html_table_segment_facts_mvp(
                html_text=html_text,
                ticker=symbol,
                market=market,
                accession=accession,
                form_type=str(filing.get("form_type", "")),
                filing_date=pd.to_datetime(filing.get("filing_date"), errors="coerce"),
                accepted_at=pd.to_datetime(filing.get("accepted_at"), errors="coerce"),
                report_date=pd.to_datetime(filing.get("report_date"), errors="coerce"),
                use_next_trading_day_availability=use_next_trading_day_availability,
                availability_fallback=availability_fallback,
                fallback_q_days=fallback_q_days,
                fallback_k_days=fallback_k_days,
                trading_days=trading_days,
            )
            if need_html
            else pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
        )
        html_metrics = set(html_tbl["metric"].dropna().astype(str).tolist()) if not html_tbl.empty else set()
        if not html_tbl.empty:
            fact_frames.append(html_tbl)
            parse_result["methods"].append("html_table_mvp")
            parse_result["rows"]["html_table_mvp"] = int(len(html_tbl))
            log_rows.append(
                {
                    "ticker": symbol,
                    "market": str(market).strip().lower(),
                    "accession": accession,
                    "method": "html_table_mvp",
                    "status": "partial",
                    "reason": f"rows={len(html_tbl)};metrics={','.join(sorted(html_metrics))}",
                    "created_at": now_utc_iso(),
                }
            )
            parse_result["status"] = "partial"
            parse_result["reason"] = f"html_table_rows={len(html_tbl)}"
        else:
            log_rows.append(
                {
                    "ticker": symbol,
                    "market": str(market).strip().lower(),
                    "accession": accession,
                    "method": "html_table_mvp",
                    "status": "fail",
                    "reason": "segment_not_found_in_ixbrl_or_html" if need_html else "not_needed_after_higher_confidence_sources",
                    "created_at": now_utc_iso(),
                }
            )
            if not parse_result["methods"]:
                parse_result["status"] = "fail"
                parse_result["reason"] = "segment_not_found_in_ixbrl_xbrl_instance_or_html"
        _write_json_cache((filing_cache_dir / "parse_result.json") if filing_cache_dir is not None else None, parse_result)

    filing_facts = (
        pd.concat(fact_frames, ignore_index=True, sort=False)
        if fact_frames
        else pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
    )
    if not filing_facts.empty:
        filing_facts["period_end"] = pd.to_datetime(filing_facts.get("period_end"), errors="coerce")
        filing_facts = filing_facts.loc[(filing_facts["period_end"].isna()) | (filing_facts["period_end"] >= min_date)].copy()

    base_long_rows: list[dict[str, Any]] = []
    if base_wide is not None and not base_wide.empty:
        for _, row in base_wide.iterrows():
            for metric, source_col in (("revenue", "revenue"), ("operating_income", "op_income")):
                val = pd.to_numeric(row.get(source_col), errors="coerce")
                if pd.isna(val):
                    continue
                base_long_rows.append(
                    {
                        "ticker": str(row.get("ticker", symbol)).strip().upper(),
                        "market": str(row.get("market", market)).strip().lower(),
                        "period_end": pd.to_datetime(row.get("period_end"), errors="coerce"),
                        "period_start": pd.to_datetime(row.get("period_start"), errors="coerce"),
                        "form_type": str(row.get("form_type", "")),
                        "filing_date": pd.to_datetime(row.get("filing_date"), errors="coerce"),
                        "accepted_at": pd.to_datetime(row.get("accepted_at"), errors="coerce"),
                        "available_date": pd.to_datetime(row.get("available_date"), errors="coerce"),
                        "availability_method": str(row.get("availability_method", "")),
                        "segment_type": str(row.get("segment_type", "")),
                        "segment_name": str(row.get("segment_name", "")),
                        "metric": metric,
                        "value": float(val),
                        "currency": "USD",
                        "accession": "",
                        "source": str(row.get("source", "sec_companyfacts_segment") or "sec_companyfacts_segment"),
                        "collected_at": str(row.get("collected_at", now_utc_iso())),
                    }
                )
    base_long = pd.DataFrame(base_long_rows, columns=SEGMENT_FACT_COLUMNS) if base_long_rows else pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)

    fact_parts = [part for part in (base_long, filing_facts) if part is not None and not part.empty]
    facts = pd.concat(fact_parts, ignore_index=True, sort=False) if fact_parts else pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)
    if not facts.empty:
        facts["period_end"] = pd.to_datetime(facts.get("period_end"), errors="coerce", utc=True).dt.tz_localize(None)
        facts["period_start"] = pd.to_datetime(facts.get("period_start"), errors="coerce", utc=True).dt.tz_localize(None)
        facts["filing_date"] = pd.to_datetime(facts.get("filing_date"), errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        facts["available_date"] = pd.to_datetime(facts.get("available_date"), errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        facts["accepted_at"] = pd.to_datetime(facts.get("accepted_at"), errors="coerce", utc=True)
        facts = facts.dropna(subset=["period_end", "segment_type", "segment_name", "metric", "value"])
        facts = facts.loc[facts["period_end"] >= min_date]
        facts = facts.sort_values(["period_end", "available_date", "filing_date", "accepted_at", "source"])
        facts = facts.drop_duplicates(
            subset=["ticker", "market", "period_end", "segment_type", "segment_name", "metric", "accession"],
            keep="last",
        )
    else:
        facts = pd.DataFrame(columns=SEGMENT_FACT_COLUMNS)

    wide = _segment_facts_to_wide(facts)
    if wide.empty and base_wide is not None and not base_wide.empty:
        wide = base_wide.copy()
    if not wide.empty:
        wide["period_end"] = pd.to_datetime(wide.get("period_end"), errors="coerce")
        wide = wide.loc[(wide["period_end"].isna()) | (wide["period_end"] >= min_date)].copy()
        wide = wide.sort_values(["period_end", "available_date", "segment_type", "segment_name"]).reset_index(drop=True)
    else:
        wide = pd.DataFrame(columns=SEGMENT_COLUMNS)

    logs = pd.DataFrame(log_rows, columns=SEGMENT_EXTRACT_LOG_COLUMNS) if log_rows else pd.DataFrame(columns=SEGMENT_EXTRACT_LOG_COLUMNS)
    filings_out = filings_df.copy() if filings_df is not None else pd.DataFrame(columns=FILING_COLUMNS)
    for col in FILING_COLUMNS:
        if col not in filings_out.columns:
            filings_out[col] = None

    return SecSegmentBundle(
        wide=wide,
        facts=facts if not facts.empty else pd.DataFrame(columns=SEGMENT_FACT_COLUMNS),
        filings=filings_out[FILING_COLUMNS].reset_index(drop=True),
        extract_log=logs.reset_index(drop=True),
    )


def fetch_sec_segment_history(
    ticker: str,
    market: str = "us",
    start: str | pd.Timestamp = SEC_DEFAULT_START_DATE,
    user_agent: str | None = None,
    use_next_trading_day_availability: bool = False,
    availability_fallback: bool = True,
    fallback_q_days: int = 45,
    fallback_k_days: int = 90,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    map_force_refresh: bool = False,
    map_cache_path: Path = SEC_TICKER_MAP_CACHE,
    raw_cache_dir: Path | None = SEC_RAW_COMPANYFACTS_DIR,
    submissions_cache_dir: Path | None = SEC_RAW_SUBMISSIONS_DIR,
    ticker_cache_dir: Path = SEC_TICKER_SEGMENT_DIR,
    lookback_filings: int = 8,
    filings_cache_dir: Path | None = SEC_FILINGS_CACHE_DIR,
    offline_mode: bool = False,
    cache_only: bool = False,
    prefetched_companyfacts: dict[str, Any] | None = None,
    prefetched_submissions: dict[str, Any] | None = None,
    prefetched_cik: int | None = None,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start, errors="coerce")
    if pd.isna(start_ts):
        start_ts = SEC_DEFAULT_START_DATE
    min_date = max(pd.Timestamp(start_ts).normalize(), SEC_DEFAULT_START_DATE)
    years_span = max(1, pd.Timestamp.utcnow().tz_localize(None).year - min_date.year + 1)
    effective_lookback = max(int(lookback_filings), min(96, years_span * 5 + 4))

    symbol = str(ticker).strip().upper()
    ensure_dir(ticker_cache_dir)
    cache_path = ticker_cache_dir / f"{sanitize_ticker(symbol)}.parquet"

    if cache_path.exists() and not force_refresh and not cache_only:
        try:
            cached = pd.read_parquet(cache_path)
            if not cached.empty:
                cached["period_end"] = pd.to_datetime(cached.get("period_end"), errors="coerce")
                cached_dates = cached["period_end"].dropna()
                cached_requested_start = pd.NaT
                if "RequestedStart" in cached.columns:
                    requested = pd.to_datetime(cached["RequestedStart"], errors="coerce").dropna()
                    if not requested.empty:
                        cached_requested_start = pd.Timestamp(requested.min()).normalize()
                version_ok = False
                if "ExtractorVersion" in cached.columns:
                    versions = pd.to_numeric(cached["ExtractorVersion"], errors="coerce").dropna()
                    if not versions.empty:
                        version_ok = int(versions.max()) >= SEC_EXTRACTOR_VERSION
                cached_lookback = 0
                if "LookbackFilings" in cached.columns:
                    lookbacks = pd.to_numeric(cached["LookbackFilings"], errors="coerce").dropna()
                    if not lookbacks.empty:
                        cached_lookback = int(lookbacks.max())

                has_min_coverage = False
                if pd.notna(cached_requested_start):
                    has_min_coverage = cached_requested_start <= min_date
                elif not cached_dates.empty:
                    has_min_coverage = pd.Timestamp(cached_dates.min()).normalize() <= min_date

                if version_ok and has_min_coverage and cached_lookback >= effective_lookback:
                    cached = cached.loc[cached["period_end"] >= min_date]
                    if not cached.empty:
                        return cached.sort_values(["period_end", "available_date", "segment_type", "segment_name"]).reset_index(drop=True)
        except Exception:
            pass

    bundle = fetch_sec_segment_bundle(
        ticker=ticker,
        market=market,
        start=start,
        user_agent=user_agent,
        use_next_trading_day_availability=use_next_trading_day_availability,
        availability_fallback=availability_fallback,
        fallback_q_days=fallback_q_days,
        fallback_k_days=fallback_k_days,
        force_refresh=force_refresh,
        retries=retries,
        backoff=backoff,
        map_force_refresh=map_force_refresh,
        map_cache_path=map_cache_path,
        raw_cache_dir=raw_cache_dir,
        submissions_cache_dir=submissions_cache_dir,
        lookback_filings=effective_lookback,
        filings_cache_dir=filings_cache_dir,
        offline_mode=offline_mode,
        cache_only=cache_only,
        prefetched_companyfacts=prefetched_companyfacts,
        prefetched_submissions=prefetched_submissions,
        prefetched_cik=prefetched_cik,
    )
    built = bundle.wide.copy()
    if built.empty:
        return built
    built["RequestedStart"] = min_date
    built["LookbackFilings"] = effective_lookback
    built["ExtractorVersion"] = SEC_EXTRACTOR_VERSION
    built.to_parquet(cache_path, index=False)
    return built


def load_sec_ticker_cache(
    ticker: str,
    ticker_cache_dir: Path = SEC_TICKER_QUARTERLY_DIR,
) -> pd.DataFrame:
    path = ticker_cache_dir / f"{sanitize_ticker(str(ticker).strip().upper())}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        out = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if out is None or out.empty:
        return pd.DataFrame()
    out["StatementDate"] = pd.to_datetime(out.get("StatementDate"), errors="coerce")
    out = out.loc[~out["StatementDate"].isna()].sort_values("StatementDate")
    return out.reset_index(drop=True)


def price_tickers(market: str = "us") -> list[str]:
    """Return tickers that have price data (DB-first, parquet fallback)."""
    from market_data.reader import available_tickers
    return available_tickers(market=market)


def build_sec_cache_for_price_tickers(
    market: str = "us",
    start: str | pd.Timestamp = SEC_DEFAULT_START_DATE,
    user_agent: str | None = None,
    force_refresh: bool = False,
    retries: int = 3,
    backoff: float = 1.0,
    progress_cb: Callable[[str, str], None] | None = None,
) -> tuple[int, int, int]:
    symbols = price_tickers(market=market)
    ok = 0
    failed = 0
    skipped = 0

    for ticker in symbols:
        try:
            cached = load_sec_ticker_cache(ticker)
            if not force_refresh and not cached.empty:
                min_date = pd.to_datetime(start, errors="coerce")
                if pd.isna(min_date):
                    min_date = SEC_DEFAULT_START_DATE
                min_date = max(pd.Timestamp(min_date).normalize(), SEC_DEFAULT_START_DATE)
                cached_dates = pd.to_datetime(cached["StatementDate"], errors="coerce").dropna()
                cached_requested_start = pd.NaT
                if "RequestedStart" in cached.columns:
                    requested = pd.to_datetime(cached["RequestedStart"], errors="coerce").dropna()
                    if not requested.empty:
                        cached_requested_start = pd.Timestamp(requested.min()).normalize()
                version_ok = False
                if "ExtractorVersion" in cached.columns:
                    versions = pd.to_numeric(cached["ExtractorVersion"], errors="coerce").dropna()
                    if not versions.empty:
                        version_ok = int(versions.max()) >= SEC_EXTRACTOR_VERSION

                has_min_coverage = False
                if pd.notna(cached_requested_start):
                    has_min_coverage = cached_requested_start <= min_date
                elif not cached_dates.empty:
                    has_min_coverage = pd.Timestamp(cached_dates.min()).normalize() <= min_date

                if version_ok and has_min_coverage and cached_dates.ge(min_date).any():
                    skipped += 1
                    if progress_cb is not None:
                        progress_cb(ticker, "SKIP")
                    continue

            built = fetch_sec_quarterly_history(
                ticker=ticker,
                market=market,
                start=start,
                user_agent=user_agent,
                force_refresh=force_refresh,
                retries=retries,
                backoff=backoff,
            )
            if built.empty:
                failed += 1
                if progress_cb is not None:
                    progress_cb(ticker, "EMPTY")
            else:
                ok += 1
                if progress_cb is not None:
                    progress_cb(ticker, "OK")
        except Exception:
            failed += 1
            if progress_cb is not None:
                progress_cb(ticker, "FAIL")

    return ok, skipped, failed
