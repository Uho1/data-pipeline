from __future__ import annotations

import argparse
import os
import json
from datetime import date
from pathlib import Path

from market_data.config import KOSPI_EXTERNAL_DEFAULT_URL, KOSPI_TOP_N_DEFAULT
from market_data.backtest.cli import (
    run_backtest_screen_cli,
    run_backtest_strategy_cli,
    run_rank_test_cli,
)
from market_data.backtest.factors import available_price_symbols
from market_data.ingest import DEFAULT_FINANCIAL_WORKERS, IngestOptions, ingest_data
from market_data.ingest_kr import KRXIngestOptions, ingest_krx_data
from market_data.kr_dart.cli import KrDartOptions, run_kr_dart_command
from market_data.sec_cache import SecCacheOptions, run_sec_cache
from market_data.sec_financials import SEC_DEFAULT_START_DATE
from market_data.sec_sector_proxy import (
    build_sector_proxy_cache_for_universe,
    build_sector_proxy_validation_report,
    ensure_sector_proxy_reference_files,
)
from market_data.sp500_pit import (
    build_sp500_pit,
    create_sp500_manual_template,
    ingest_sp500_manual_events,
    import_sp500_github_secondary,
    report_sp500_pit,
    sp500_pit_diff,
    validate_sp500_pit_cache,
)
from market_data.sample import run_sample
from market_data.universe import build_universe
from market_data.validate import run_validate
try:
    from market_data.wrds.cli import add_wrds_subparser, run_wrds_command
except ImportError:
    add_wrds_subparser = None  # type: ignore[assignment]
    run_wrds_command = None  # type: ignore[assignment]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market_data", description="Local market data lake CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build universe and ingest price/financial datasets")
    p_ingest.add_argument("--universe", choices=["sp500", "kospi", "custom"], required=True)
    p_ingest.add_argument("--tickers-file", default=None, help="CSV file for custom/kospi file mode")
    p_ingest.add_argument("--start", default="2000-01-01", help="Start date for price download")
    p_ingest.add_argument(
        "--price-start",
        default=None,
        help="Override start date for price download only (defaults to --start)",
    )
    p_ingest.add_argument(
        "--financial-start",
        default=None,
        help="Override start date for financial/SEC ingest only (defaults to --start)",
    )
    p_ingest.add_argument("--end", default=date.today().isoformat(), help="End date for price download")
    p_ingest.add_argument("--interval", default="1d", help="yfinance interval")
    p_ingest.add_argument("--kospi-external-url", default=KOSPI_EXTERNAL_DEFAULT_URL)
    p_ingest.add_argument(
        "--kospi-top-n",
        type=int,
        default=KOSPI_TOP_N_DEFAULT,
        help="Number of KOSPI tickers to keep (default: 500, <=0 means all)",
    )
    p_ingest.add_argument("--fresh-days", type=int, default=7, help="Skip file if parquet age <= N days")
    p_ingest.add_argument("--retries", type=int, default=3, help="Retry count on network/rate-limit errors")
    p_ingest.add_argument("--backoff-base", type=float, default=1.0, help="Initial backoff seconds")
    p_ingest.add_argument(
        "--financial-workers",
        type=int,
        default=DEFAULT_FINANCIAL_WORKERS,
        help="Concurrent workers for annual/quarterly financial statement pulls per ticker",
    )
    p_ingest.add_argument(
        "--financial-source",
        choices=["yfinance", "sec"],
        default="sec",
        help="Financial statement source",
    )
    p_ingest.add_argument(
        "--sec-user-agent",
        default=os.getenv("SEC_USER_AGENT"),
        help="SEC API User-Agent header (or env SEC_USER_AGENT)",
    )
    p_ingest.add_argument(
        "--use-next-trading-day-availability",
        action="store_true",
        help="When SEC filing timestamp is available, shift fundamentals AvailableDate to next trading day",
    )
    p_ingest.add_argument(
        "--no-fundamentals-availability-fallback",
        action="store_true",
        help="Disable SEC fallback lag (10-Q +45d / 10-K +90d) when filing date is missing",
    )
    p_ingest.add_argument(
        "--fundamentals-fallback-q-days",
        type=int,
        default=45,
        help="Fallback lag days for 10-Q when filing date is unavailable (default: 45)",
    )
    p_ingest.add_argument(
        "--fundamentals-fallback-k-days",
        type=int,
        default=90,
        help="Fallback lag days for 10-K when filing date is unavailable (default: 90)",
    )
    p_ingest.add_argument("--workers", type=int, default=1, help="Thread workers (default 1 = sequential)")
    p_ingest.add_argument(
        "--price-batch-size",
        type=int,
        default=50,
        help="Number of tickers per price batch request when workers=1",
    )
    p_ingest.add_argument(
        "--disable-price-batch",
        action="store_true",
        help="Disable price batch prefetch and download prices one ticker at a time",
    )
    p_ingest.add_argument("--force", action="store_true", help="Ignore freshness/checkpoints and redownload")
    p_ingest.add_argument(
        "--skip-sector-cache",
        action="store_true",
        help="Skip automatic SEC sector proxy cache build after ingest (default: sector cache is built)",
    )
    p_ingest.add_argument(
        "--backfill-financials-extra",
        action="store_true",
        help="Run SEC companyfacts expansion backfill and write sec_facts_raw_normalized/financials_quarterly_extra",
    )
    p_ingest.add_argument(
        "--sec-financials-only",
        action="store_true",
        help="SEC fast path: refresh financials_quarterly/derived only, without filings/raw fact/enrichment fetches",
    )
    p_ingest.add_argument(
        "--reparse-sec-from-cache",
        action="store_true",
        help="Rebuild SEC financial metrics from cached raw JSON without SEC network fetch",
    )

    p_kr_ingest = sub.add_parser("krx-ingest", help="Build KRX universe and ingest KR prices/investor flows/DART")
    p_kr_ingest.add_argument("--start", default="2000-01-01", help="Start date for KR ingest")
    p_kr_ingest.add_argument("--end", default=date.today().isoformat(), help="End date for KR ingest")
    p_kr_ingest.add_argument("--tickers", default=None, help="Comma-separated 6-digit KR tickers")
    p_kr_ingest.add_argument("--tickers-file", default=None, help="CSV file with KR tickers")
    p_kr_ingest.add_argument("--include-dart", action="store_true", help="Also ingest DART raw metadata and raw financials")
    p_kr_ingest.add_argument("--skip-dart-materialize", action="store_true", help="Skip best-effort DART canonical materialization")
    p_kr_ingest.add_argument("--skip-master", action="store_true", help="Reuse existing ticker_master instead of rebuilding it")
    p_kr_ingest.add_argument("--skip-prices", action="store_true", help="Skip KR price ingest")
    p_kr_ingest.add_argument("--skip-investors", action="store_true", help="Skip investor flow ingest")
    p_kr_ingest.add_argument("--skip-indices", action="store_true", help="Skip representative index ingest")
    p_kr_ingest.add_argument("--fresh-days", type=int, default=None, help="Skip recently completed KR tickers")
    p_kr_ingest.add_argument("--force", action="store_true", help="Ignore KR checkpoints and rebuild")
    p_kr_ingest.add_argument("--use-universe", action="store_true", default=True, help="Only ingest tickers in universe list (default: True)")
    p_kr_ingest.add_argument("--no-universe", action="store_true", help="Ingest ALL tickers, not just universe")
    p_kr_ingest.add_argument("--workers", type=int, default=1, help="Concurrent ticker workers for KR ingest")
    p_kr_ingest.add_argument("--skip-dart-company-enrich", action="store_true", help="Skip per-company OpenDART company.json enrichment when building ticker master")

    p_kr_dart = sub.add_parser("kr-dart", help="Run Korea DART raw/materialization workflows")
    kr_dart_sub = p_kr_dart.add_subparsers(dest="kr_dart_command", required=True)
    p_kr_dart_corp = kr_dart_sub.add_parser("corp-master", help="Refresh DART corp master")
    p_kr_dart_filings = kr_dart_sub.add_parser("filings", help="Fetch DART filing metadata into KR DB")
    p_kr_dart_filings.add_argument("--tickers", default=None, help="Comma-separated 6-digit KR tickers")
    p_kr_dart_filings.add_argument("--start-date", default="20130601", help="Inclusive start date YYYYMMDD")
    p_kr_dart_filings.add_argument("--end-date", default=date.today().strftime("%Y%m%d"), help="Inclusive end date YYYYMMDD")
    p_kr_dart_financials = kr_dart_sub.add_parser("financials", help="Fetch DART single-account raw financials")
    p_kr_dart_financials.add_argument("--tickers", default=None, help="Comma-separated 6-digit KR tickers")
    p_kr_dart_financials.add_argument("--start-year", type=int, default=2013)
    p_kr_dart_financials.add_argument("--end-year", type=int, default=date.today().year)
    p_kr_dart_financials.add_argument("--fs-div", choices=["CFS", "OFS"], default="CFS")
    p_kr_dart_materialize = kr_dart_sub.add_parser("materialize", help="Materialize KR DART raw data into financials_quarterly")
    p_kr_dart_materialize.add_argument("--tickers", default=None, help="Optional comma-separated 6-digit KR tickers")
    p_kr_dart_materialize.add_argument("--start-year", type=int, default=2013)
    p_kr_dart_materialize.add_argument("--end-year", type=int, default=date.today().year)
    p_export_json = sub.add_parser("export-json", help="Export DuckDB data to per-ticker JSON files for web serving")
    p_export_json.add_argument("--market", default="kr", choices=["kr", "us"], help="Market to export")
    p_export_json.add_argument("--tickers", default=None, help="Comma-separated tickers (default: all)")
    p_export_json.add_argument("--start-date", default="2013-06-01", help="Only include data from this date (default: 2013-06-01)")
    p_export_json.add_argument("--top-n", type=int, default=0, help="Only export top N tickers by market cap (US, default: all)")
    p_export_json.add_argument("--use-universe", action="store_true", default=True, help="Only export tickers in universe list (default: True)")
    p_export_json.add_argument("--no-universe", action="store_true", help="Export ALL tickers, not just universe")

    p_build_universe = sub.add_parser("build-universe", help="Build universe: common stocks with financials, ranked by market cap")
    p_build_universe.add_argument("--market", default="kr", choices=["kr", "us"], help="Market to build universe for")
    p_build_universe.add_argument("--top-n", type=int, default=2000, help="Max tickers by market cap (0=all, default: 2000)")
    p_build_universe.add_argument("--min-market-cap", type=float, default=0, help="Minimum market cap filter (default: 0)")
    p_build_universe.add_argument("--no-require-financials", action="store_true", help="Include tickers without financial data")
    p_build_universe.add_argument("--no-common-stock-filter", action="store_true", help="Include preferred stocks etc.")

    p_export_parquet = sub.add_parser("export-parquet", help="Export DuckDB tables to Parquet for raw data preservation")
    p_export_parquet.add_argument("--market", default="all", choices=["kr", "us", "all"], help="Market to export")
    p_export_parquet.add_argument("--start-date", default="2013-06-01", help="Only include data from this date (default: 2013-06-01)")

    p_validate = sub.add_parser("validate", help="Validate local parquet files and output reports")
    p_validate.set_defaults(command="validate")

    p_sample = sub.add_parser("sample", help="Show summary of saved ticker data")
    p_sample.add_argument("--ticker", required=True)
    p_sample.add_argument("--market", default=None, help="Optional market folder (us/kr)")

    p_chart = sub.add_parser("chart", help="Plot saved ticker price data")
    p_chart.add_argument("--ticker", required=True)
    p_chart.add_argument("--market", default=None, help="Optional market folder (us/kr)")
    p_chart.add_argument(
        "--chart-type",
        choices=["candles", "line"],
        default="candles",
        help="Chart style",
    )
    p_chart.add_argument("--price-col", default="Adj Close", help="Price column to plot")
    p_chart.add_argument("--save-path", default=None, help="Optional output PNG path")
    p_chart.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window (save-only or headless mode)",
    )

    sub.add_parser("recent-data-gui", help="Open Recent Data Updater GUI (prices + SEC financials + sector)")

    p_gui = sub.add_parser("gui", help="Open ticker chart GUI")
    p_gui.add_argument("--ticker", default="AAPL", help="Initial ticker")
    p_gui.add_argument("--market", default="auto", help="Initial market: auto/us/kr")
    p_gui.add_argument(
        "--price-field",
        choices=["adjclose", "close"],
        default="adjclose",
        help="Price field for valuation calculations (default: adjclose)",
    )
    p_gui.add_argument(
        "--per-negative",
        choices=["nan", "allow"],
        default="nan",
        help="How to handle EPS_TTM<=0 in PER (default: nan)",
    )
    p_gui.add_argument(
        "--band-window",
        choices=["all", "10y", "5y"],
        default="all",
        help="Window for automatic PER band quantiles (default: all)",
    )
    p_gui.add_argument(
        "--band-quantiles",
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated five quantiles for PER bands (default: 0.1,0.3,0.5,0.7,0.9)",
    )
    p_gui.add_argument(
        "--outlier",
        choices=["none", "winsorize-1-99"],
        default="none",
        help="Outlier handling for band level estimation",
    )

    p_sec = sub.add_parser(
        "sec-cache",
        help="Fetch SEC CompanyFacts quarterly financial cache for US tickers with local price files",
    )
    p_sec.add_argument("--market", default="us", help="Target market folder (default: us)")
    p_sec.add_argument(
        "--start",
        default=str(SEC_DEFAULT_START_DATE.date()),
        help=f"Minimum statement date (default: {SEC_DEFAULT_START_DATE.date()})",
    )
    p_sec.add_argument("--retries", type=int, default=3, help="Retry count on network errors")
    p_sec.add_argument("--backoff-base", type=float, default=1.0, help="Initial backoff seconds")
    p_sec.add_argument(
        "--sec-user-agent",
        default=os.getenv("SEC_USER_AGENT"),
        help="SEC API User-Agent header (or env SEC_USER_AGENT)",
    )
    p_sec.add_argument("--force", action="store_true", help="Ignore existing cache files and refetch")

    p_sec_sector = sub.add_parser(
        "sec-sector-cache",
        help="Build SEC SIC->sector_proxy PIT cache for a ticker universe",
    )
    p_sec_sector.add_argument("--market", default="us", help="Target market folder (default: us)")
    p_sec_sector.add_argument(
        "--universe",
        choices=["local", "sp500", "kospi", "custom"],
        default="local",
        help="Universe source for ticker list",
    )
    p_sec_sector.add_argument("--tickers-file", default=None, help="Ticker CSV (for custom universe)")
    p_sec_sector.add_argument("--kospi-external-url", default=KOSPI_EXTERNAL_DEFAULT_URL)
    p_sec_sector.add_argument("--retries", type=int, default=3, help="Retry count on SEC network errors")
    p_sec_sector.add_argument("--backoff-base", type=float, default=1.0, help="Initial backoff seconds")
    p_sec_sector.add_argument(
        "--sec-user-agent",
        default=os.getenv("SEC_USER_AGENT"),
        help="SEC API User-Agent header (or env SEC_USER_AGENT)",
    )
    p_sec_sector.add_argument("--workers", type=int, default=4, help="Parallel workers for SEC API requests (default: 4)")
    p_sec_sector.add_argument("--force", action="store_true", help="Ignore existing submissions cache and refetch")
    p_sec_sector.add_argument("--fail-closed", action="store_true", help="Stop on first ticker-level failure")
    p_sec_sector.add_argument("--unclassified-top-n", type=int, default=20, help="Top-N unclassified report rows")
    p_sec_sector.add_argument(
        "--debug-sic-rules",
        action="store_true",
        help="Log SIC rule normalization samples (start/end raw vs normalized)",
    )

    p_sector_validate = sub.add_parser(
        "sector-validate",
        help="Validate sector proxy evidence for an AI review bundle",
    )
    p_sector_validate.add_argument("--bundle", required=True, help="Path to ai_review bundle root")
    p_sector_validate.add_argument("--market", default="us", help="Market used for sector PIT lookup")
    p_sector_validate.add_argument("--fail-closed", action="store_true", help="Return non-zero on FAIL status")

    p_sp500_pit_build = sub.add_parser(
        "sp500-pit-build",
        help="Build S&P 500 point-in-time (PIT) constituents table from configured providers",
    )
    p_sp500_pit_build.add_argument("--start", default="2000-01-01", help="Coverage start date (YYYY-MM-DD)")
    p_sp500_pit_build.add_argument("--end", default=None, help="Coverage end date (YYYY-MM-DD)")
    p_sp500_pit_build.add_argument(
        "--provider-order",
        default="wrds,spdji,manual,secondary",
        help="Provider order, comma-separated: wrds,spdji,manual,github,secondary",
    )
    p_sp500_pit_build.add_argument(
        "--seed-policy",
        choices=["allow", "official_only", "manual_only", "disable_secondary"],
        default="allow",
        help="How to use secondary snapshot seed data",
    )
    p_sp500_pit_build.add_argument(
        "--manual-events-path",
        default=None,
        help="Optional path to manual events CSV (default: config/sp500_manual_events.csv)",
    )
    p_sp500_pit_build.add_argument(
        "--manual-override-path",
        default=None,
        help="Optional legacy manual override CSV path",
    )
    p_sp500_pit_build.add_argument(
        "--pit-dir",
        default=None,
        help="Optional PIT cache directory override",
    )
    p_sp500_pit_build.add_argument("--strict", action="store_true", help="Treat data quality warnings as errors")
    p_sp500_pit_build.add_argument("--force-refresh", action="store_true", help="Force rebuild even when cache exists")
    p_sp500_pit_build.add_argument("--fail-closed", action="store_true", help="Abort on validation failure")
    p_sp500_pit_build.add_argument("--min-confidence", type=float, default=0.7, help="Minimum confidence threshold")

    p_sp500_pit_validate = sub.add_parser(
        "sp500-pit-validate",
        help="Validate cached S&P 500 PIT intervals and write coverage report",
    )
    p_sp500_pit_validate.add_argument("--start", default="2000-01-01", help="Coverage start date (YYYY-MM-DD)")
    p_sp500_pit_validate.add_argument("--end", default=None, help="Coverage end date (YYYY-MM-DD)")
    p_sp500_pit_validate.add_argument("--strict", action="store_true", help="Treat quality thresholds as hard errors")
    p_sp500_pit_validate.add_argument("--min-confidence", type=float, default=0.7, help="Minimum confidence threshold")
    p_sp500_pit_validate.add_argument("--fail-closed", action="store_true", help="Return non-zero on FAIL status")
    p_sp500_pit_validate.add_argument("--pit-dir", default=None, help="Optional PIT cache directory override")

    p_sp500_pit_report = sub.add_parser(
        "sp500-pit-report",
        help="Print latest S&P 500 PIT coverage report summary",
    )
    p_sp500_pit_report.add_argument("--pit-dir", default=None, help="Optional PIT cache directory override")
    p_sp500_pit_template = sub.add_parser(
        "sp500-pit-template",
        help="Create a manual S&P500 PIT events CSV template",
    )
    p_sp500_pit_template.add_argument(
        "--out",
        default=None,
        help="Template output path (default: data/reference/sp500_manual_events.template.csv)",
    )

    p_sp500_pit_ingest_manual = sub.add_parser(
        "sp500-pit-ingest-manual",
        help="Ingest manual S&P500 PIT event CSV into config cache",
    )
    p_sp500_pit_ingest_manual.add_argument("--events-file", required=True, help="Manual events CSV path")
    p_sp500_pit_ingest_manual.add_argument(
        "--target-path",
        default=None,
        help="Target events cache CSV (default: config/sp500_manual_events.csv)",
    )
    mode_group = p_sp500_pit_ingest_manual.add_mutually_exclusive_group()
    mode_group.add_argument("--append", action="store_true", help="Append to existing events (default)")
    mode_group.add_argument("--replace", action="store_true", help="Replace existing events with this file")
    p_sp500_pit_ingest_manual.add_argument("--strict", action="store_true", help="Fail on schema validation errors")
    p_sp500_pit_ingest_manual.add_argument("--require-source-ref", action="store_true", help="Require source_ref")
    p_sp500_pit_ingest_manual.add_argument("--require-source-doc-id", action="store_true", help="Require source_doc_id")
    p_sp500_pit_ingest_manual.add_argument("--require-provenance-text", action="store_true", help="Require provenance_text")
    p_sp500_pit_ingest_manual.add_argument("--require-confidence", action="store_true", help="Require confidence")
    p_sp500_pit_ingest_manual.add_argument("--no-normalize-actions", action="store_true", help="Do not normalize replace/rename to remove+add")
    p_sp500_pit_ingest_manual.add_argument("--rebuild", action="store_true", help="Rebuild PIT cache right after ingest")
    p_sp500_pit_ingest_manual.add_argument("--rebuild-start", default="2000-01-01")
    p_sp500_pit_ingest_manual.add_argument("--rebuild-end", default=None)
    p_sp500_pit_ingest_manual.add_argument(
        "--rebuild-provider-order",
        default="manual,secondary",
        help="Provider order used for rebuild",
    )
    p_sp500_pit_ingest_manual.add_argument(
        "--rebuild-seed-policy",
        choices=["allow", "official_only", "manual_only", "disable_secondary"],
        default="allow",
    )
    p_sp500_pit_ingest_manual.add_argument("--rebuild-fail-closed", action="store_true")
    p_sp500_pit_ingest_manual.add_argument("--rebuild-min-confidence", type=float, default=0.7)
    p_sp500_pit_ingest_manual.add_argument("--pit-dir", default=None, help="Optional PIT cache directory override")

    p_sp500_pit_import_github = sub.add_parser(
        "sp500-pit-import-github-secondary",
        help="Import GitHub secondary S&P500 historical components into manual events cache",
    )
    p_sp500_pit_import_github.add_argument(
        "--repo-url",
        default="fja05680/sp500",
        help="GitHub repo path (default: fja05680/sp500)",
    )
    p_sp500_pit_import_github.add_argument(
        "--raw-url",
        default=None,
        help="Optional raw CSV URL override (or file:///path/to/file.csv)",
    )
    p_sp500_pit_import_github.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory for GitHub raw/source metadata",
    )
    p_sp500_pit_import_github.add_argument(
        "--out",
        default="config/sp500_manual_events.github_seed.csv",
        help="Output CSV path for normalized manual-style events",
    )
    p_sp500_pit_import_github.add_argument(
        "--target-path",
        default=None,
        help="Target manual cache CSV for ingest (default: config/sp500_manual_events.csv)",
    )
    p_sp500_pit_import_github.add_argument("--confidence-default", type=float, default=0.7)
    p_sp500_pit_import_github.add_argument("--strict", action="store_true", help="Fail on schema/quality threshold violations")
    p_sp500_pit_import_github.add_argument("--fail-closed", action="store_true", help="Alias for strict import blocking")
    p_sp500_pit_import_github.add_argument("--dry-run", action="store_true", help="Do not write files; print summary only")
    p_sp500_pit_import_github.add_argument("--force-refresh", action="store_true", help="Ignore cache and force re-download")
    mode_group2 = p_sp500_pit_import_github.add_mutually_exclusive_group()
    mode_group2.add_argument("--append", action="store_true", help="Append imported rows into target manual cache (default)")
    mode_group2.add_argument("--replace", action="store_true", help="Replace target manual cache with imported rows")
    p_sp500_pit_import_github.add_argument("--since-date", default=None, help="Optional lower bound date filter (YYYY-MM-DD)")
    p_sp500_pit_import_github.add_argument("--until-date", default=None, help="Optional upper bound date filter (YYYY-MM-DD)")
    p_sp500_pit_import_github.add_argument("--parse-fail-rate-threshold", type=float, default=0.2)
    p_sp500_pit_import_github.add_argument("--missing-ticker-rate-threshold", type=float, default=0.05)
    p_sp500_pit_import_github.add_argument("--rebuild", action="store_true", help="Rebuild PIT right after ingest")
    p_sp500_pit_import_github.add_argument("--rebuild-start", default="2000-01-01")
    p_sp500_pit_import_github.add_argument("--rebuild-end", default=None)
    p_sp500_pit_import_github.add_argument(
        "--rebuild-provider-order",
        default="manual,secondary",
        help="Provider order used for rebuild",
    )
    p_sp500_pit_import_github.add_argument(
        "--rebuild-seed-policy",
        choices=["allow", "official_only", "manual_only", "disable_secondary"],
        default="allow",
    )
    p_sp500_pit_import_github.add_argument("--rebuild-fail-closed", action="store_true")
    p_sp500_pit_import_github.add_argument("--rebuild-min-confidence", type=float, default=0.7)
    p_sp500_pit_import_github.add_argument("--pit-dir", default=None, help="Optional PIT cache directory override")

    p_sp500_pit_diff = sub.add_parser(
        "sp500-pit-diff",
        help="Compare S&P500 PIT constituents between two dates",
    )
    p_sp500_pit_diff.add_argument("--date-a", required=True, help="Date A (YYYY-MM-DD)")
    p_sp500_pit_diff.add_argument("--date-b", required=True, help="Date B (YYYY-MM-DD)")
    p_sp500_pit_diff.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence filter")
    p_sp500_pit_diff.add_argument("--pit-dir", default=None, help="Optional PIT cache directory override")

    p_bts = sub.add_parser(
        "backtest-screen",
        help="Run screen-style backtest (full replacement each rebalance)",
    )
    p_bts.add_argument("--screen", required=True, help='Screen expression, e.g. "pe<=10 & ps<=3"')
    p_bts.add_argument("--universe", choices=["sp500", "kospi", "custom", "sp500_pit"], default="sp500", help="Universe source (default: sp500)")
    p_bts.add_argument("--freq", choices=["W", "M", "Q"], default="Q", help="Rebalance frequency")
    p_bts.add_argument("--start", default="2000-01-01", help="Start date")
    p_bts.add_argument("--end", default=None, help="Optional end date")
    p_bts.add_argument("--holdings", type=int, default=3, help="Number of holdings")
    p_bts.add_argument("--sizing", choices=["equal", "rank_weight"], default="equal", help="Position sizing")
    p_bts.add_argument("--market", default="us", help="Market folder (default: us)")
    p_bts.add_argument("--ranking", default=None, help="Optional ranking YAML/JSON")
    p_bts.add_argument("--benchmark", default="SPY", help="Benchmark ticker (default: SPY)")
    p_bts.add_argument(
        "--execution-timing",
        choices=["next_open", "same_close"],
        default="next_open",
        help="Order execution timing (default: next_open)",
    )
    p_bts.add_argument(
        "--use-fundamentals-pit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable fundamentals PIT as-of AvailableDate (default: False)",
    )
    p_bts.add_argument(
        "--use-sp500-pit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable S&P 500 PIT membership (default: False)",
    )
    p_bts.add_argument(
        "--use-sector-pit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable sector proxy PIT mapping (default: False)",
    )
    p_bts.add_argument(
        "--strict",
        action="store_true",
        help="Enable fail-closed strict policy for PIT missing data (default: False)",
    )
    p_bts.add_argument("--asof-lag-trading-days", type=int, default=0, help="Lag signal date by N trading days for fundamentals as-of")
    p_bts.add_argument(
        "--use-next-trading-day-availability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Shift filing availability to next trading day when building factors",
    )
    p_bts.add_argument(
        "--fundamentals-availability-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fallback lag when filing dates are unavailable",
    )
    p_bts.add_argument(
        "--fundamentals-missing-policy",
        choices=["exclude", "error"],
        default="exclude",
        help="When PIT enabled and as-of fundamentals missing: exclude ticker or fail",
    )
    p_bts.add_argument("--fundamentals-fallback-q-days", type=int, default=45)
    p_bts.add_argument("--fundamentals-fallback-k-days", type=int, default=90)
    p_bts.add_argument("--out-dir", default="logs/backtests", help="Output root directory")

    p_bst = sub.add_parser(
        "backtest-strategy",
        help="Run strategy simulation from YAML/JSON config",
    )
    p_bst.add_argument("--config", required=True, help="Path to strategy config file")
    p_bst.add_argument(
        "--use-fundamentals-pit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override strategy config PIT on/off",
    )
    p_bst.add_argument("--asof-lag-trading-days", type=int, default=None, help="Override fundamentals as-of lag")
    p_bst.add_argument(
        "--use-next-trading-day-availability",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override availability next-trading-day shift",
    )
    p_bst.add_argument(
        "--fundamentals-availability-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override fallback lag usage",
    )
    p_bst.add_argument(
        "--fundamentals-missing-policy",
        choices=["exclude", "error"],
        default=None,
        help="Override missing fundamentals policy",
    )
    p_bst.add_argument("--fundamentals-fallback-q-days", type=int, default=None)
    p_bst.add_argument("--fundamentals-fallback-k-days", type=int, default=None)
    p_bst.add_argument("--out-dir", default="logs/backtests", help="Output root directory")

    p_rank = sub.add_parser(
        "rank-test",
        help="Run ranking snapshots over rebalance dates",
    )
    p_rank.add_argument("--ranking", required=True, help="Ranking YAML/JSON config file")
    p_rank.add_argument("--universe", choices=["local", "sp500", "kospi", "custom"], default="local")
    p_rank.add_argument("--tickers-file", default=None, help="Ticker CSV (for custom universe)")
    p_rank.add_argument("--market", default="us", help="Market folder for local mode")
    p_rank.add_argument("--freq", choices=["W", "M", "Q"], default="Q")
    p_rank.add_argument("--start", default="2000-01-01")
    p_rank.add_argument("--end", default=None)
    p_rank.add_argument("--out-dir", default="logs/backtests", help="Output root directory")

    if add_wrds_subparser is not None:
        add_wrds_subparser(sub)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        opts = IngestOptions(
            universe=args.universe,
            start=args.start,
            end=args.end,
            interval=args.interval,
            tickers_file=args.tickers_file,
            kospi_external_url=args.kospi_external_url,
            kospi_top_n=args.kospi_top_n,
            fresh_days=args.fresh_days,
            retries=args.retries,
            backoff_base=args.backoff_base,
            financial_workers=max(1, args.financial_workers),
            workers=max(1, args.workers),
            force=args.force,
            price_batch_size=max(1, args.price_batch_size),
            disable_price_batch=args.disable_price_batch,
            financial_source=args.financial_source,
            sec_user_agent=args.sec_user_agent,
            use_next_trading_day_availability=bool(args.use_next_trading_day_availability),
            fundamentals_availability_fallback=not bool(args.no_fundamentals_availability_fallback),
            fundamentals_fallback_q_days=max(0, int(args.fundamentals_fallback_q_days)),
            fundamentals_fallback_k_days=max(0, int(args.fundamentals_fallback_k_days)),
            reparse_sec_from_cache=bool(args.reparse_sec_from_cache),
            backfill_financials_extra=bool(args.backfill_financials_extra),
            include_sector_cache=not bool(args.skip_sector_cache),
            sec_financials_only=bool(args.sec_financials_only),
            price_start=args.price_start,
            financial_start=args.financial_start,
        )
        return ingest_data(opts)

    if args.command == "krx-ingest":
        tickers = [item.strip() for item in str(args.tickers or "").split(",") if item.strip()]
        return ingest_krx_data(
            KRXIngestOptions(
                start=str(args.start),
                end=str(args.end),
                tickers=tickers,
                tickers_file=args.tickers_file,
                include_dart=bool(args.include_dart),
                materialize_dart=not bool(args.skip_dart_materialize),
                skip_master=bool(args.skip_master),
                skip_prices=bool(args.skip_prices),
                skip_investors=bool(args.skip_investors),
                skip_indices=bool(args.skip_indices),
                fresh_days=args.fresh_days,
                force=bool(args.force),
                use_universe=not bool(getattr(args, "no_universe", False)),
                workers=max(1, int(getattr(args, "workers", 1))),
                skip_dart_company_enrich=bool(getattr(args, "skip_dart_company_enrich", False)),
            )
        )

    if args.command == "kr-dart":
        tickers = [item.strip() for item in str(getattr(args, "tickers", "") or "").split(",") if item.strip()]
        return run_kr_dart_command(
            KrDartOptions(
                command=str(args.kr_dart_command),
                tickers=tickers,
                start_date=str(getattr(args, "start_date", "20200101")),
                end_date=getattr(args, "end_date", None),
                start_year=int(getattr(args, "start_year", 2020)),
                end_year=int(getattr(args, "end_year", date.today().year)),
                fs_div=str(getattr(args, "fs_div", "CFS")),
                mode=str(getattr(args, "mode", "segments")),
                workers=int(getattr(args, "workers", 1)),
            )
        )

    if args.command == "export-json":
        from market_data.export_json import run_export
        tickers = [t.strip() for t in str(args.tickers or "").split(",") if t.strip()] or None
        # If --use-universe is set and no explicit tickers, load universe list
        use_uni = not bool(getattr(args, "no_universe", False))
        if use_uni and not tickers:
            from market_data.universe_builder import load_universe
            universe_tickers = load_universe(market=args.market)
            if universe_tickers:
                tickers = universe_tickers
                print(f"[export-json] Using universe: {len(tickers)} tickers")
            else:
                print("[export-json] WARNING: No universe file found, exporting all tickers")
        run_export(
            market=args.market,
            tickers=tickers,
            start_date=args.start_date,
            top_n=args.top_n,
        )
        return 0

    if args.command == "build-universe":
        from market_data.universe_builder import run_build_universe
        run_build_universe(
            market=args.market,
            top_n=args.top_n,
            min_market_cap=args.min_market_cap,
        )
        return 0

    if args.command == "export-parquet":
        from market_data import export_parquet
        export_parquet._START_DATE = args.start_date
        if args.market in ("kr", "all"):
            export_parquet.export_market("kr")
        if args.market in ("us", "all"):
            export_parquet.export_market("us")
        return 0

    if args.command == "validate":
        return run_validate()

    if args.command == "sample":
        return run_sample(args.ticker, args.market)

    if args.command == "chart":
        from market_data.chart import run_chart

        return run_chart(
            ticker=args.ticker,
            market=args.market,
            price_col=args.price_col,
            chart_type=args.chart_type,
            save_path=args.save_path,
            show=not args.no_show,
        )

    if args.command == "recent-data-gui":
        from market_data.finterstellar_raw_gui import run_recent_data_gui

        return run_recent_data_gui()

    if args.command == "gui":
        from market_data.gui import run_gui

        return run_gui(
            initial_ticker=args.ticker,
            market=args.market,
            valuation_price_field=args.price_field,
            valuation_per_negative=args.per_negative,
            valuation_band_window=args.band_window,
            valuation_band_quantiles=args.band_quantiles,
            valuation_outlier=args.outlier,
        )

    if args.command == "sec-cache":
        return run_sec_cache(
            SecCacheOptions(
                market=args.market,
                start=args.start,
                user_agent=args.sec_user_agent,
                retries=max(0, int(args.retries)),
                backoff_base=max(float(args.backoff_base), 0.1),
                force=args.force,
            )
        )

    if args.command == "sec-sector-cache":
        ensure_sector_proxy_reference_files()
        market = str(args.market or "us").strip().lower()
        symbols: list[str] = []
        if args.universe == "local":
            symbols = available_price_symbols(market=market)
        else:
            try:
                symbols, inferred_market, _ = build_universe(
                    universe=args.universe,
                    tickers_file=args.tickers_file,
                    kospi_external_url=args.kospi_external_url,
                    kospi_top_n=None,
                )
                market = inferred_market or market
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Universe build failed: {exc}")
                symbols = available_price_symbols(market=market)

        symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        if not symbols:
            print("[ERROR] No symbols found for sec-sector-cache")
            return 2

        summary = build_sector_proxy_cache_for_universe(
            symbols=symbols,
            market=market,
            user_agent=args.sec_user_agent,
            retries=max(0, int(args.retries)),
            backoff_base=max(float(args.backoff_base), 0.1),
            force_refresh=bool(args.force),
            fail_closed=bool(args.fail_closed),
            unclassified_top_n=max(1, int(args.unclassified_top_n)),
            debug_rule_samples=bool(args.debug_sic_rules),
            workers=max(1, int(args.workers)),
        )
        print(
            f"[DONE] sec-sector-cache market={market} total={summary.get('total')} "
            f"ok={summary.get('ok')} failed={summary.get('failed')} "
            f"unclassified={summary.get('unclassified')}"
        )
        print(f"[REPORT] {summary.get('unclassified_report')}")
        if summary.get("errors"):
            print("[ERRORS]")
            for item in summary.get("errors", [])[:20]:
                print(f"  - {item.get('ticker')}: {item.get('error')}")
        return 0 if int(summary.get("failed", 0)) == 0 else 2

    if args.command == "sector-validate":
        bundle_root = args.bundle
        report, issues = build_sector_proxy_validation_report(
            bundle_root,
            market=str(args.market or "us").strip().lower(),
            fail_closed=bool(args.fail_closed),
        )
        print(
            f"[DONE] sector-validate status={report.get('status')} "
            f"mode={report.get('mode')} issues={report.get('issue_count')}"
        )
        print(f"[REPORT] {os.path.join(bundle_root, 'sector_proxy_validation_report.json')}")
        print(f"[ISSUES] {os.path.join(bundle_root, 'sector_proxy_validation_issues.csv')}")
        if not issues.empty:
            print(issues.head(10).to_string(index=False))
        if bool(args.fail_closed) and str(report.get("status", "")).lower() == "fail":
            return 2
        return 0

    if args.command == "sp500-pit-build":
        summary = build_sp500_pit(
            start=str(args.start),
            end=args.end,
            provider_order=str(args.provider_order),
            seed_policy=str(args.seed_policy),
            strict=bool(args.strict),
            fail_closed=bool(args.fail_closed),
            min_confidence=float(args.min_confidence),
            force_refresh=bool(args.force_refresh),
            manual_events_path=args.manual_events_path,
            manual_override_path=args.manual_override_path,
            pit_dir=args.pit_dir,
        )
        print(
            f"[DONE] sp500-pit-build status={summary.get('status')} "
            f"interval_rows={summary.get('interval_rows')} ticker_count={summary.get('ticker_count')}"
        )
        paths = summary.get("paths", {})
        print(f"[REPORT] {paths.get('coverage_report', '')}")
        print(f"[ISSUES] {paths.get('issues_path', '')}")
        print(f"[INTERVALS] {paths.get('intervals_path', '')}")
        if str(summary.get("status", "")).lower() == "fail":
            return 2
        return 0

    if args.command == "sp500-pit-validate":
        report = validate_sp500_pit_cache(
            start=str(args.start),
            end=args.end,
            min_confidence=float(args.min_confidence),
            strict=bool(args.strict),
            fail_closed=bool(args.fail_closed),
            pit_dir=args.pit_dir,
        )
        print(f"[DONE] sp500-pit-validate status={report.get('status')}")
        paths = report.get("paths", {})
        print(f"[REPORT] {paths.get('coverage_report', '')}")
        print(f"[ISSUES] {paths.get('issues_path', '')}")
        if bool(args.fail_closed) and str(report.get("status", "")).lower() == "fail":
            return 2
        return 0

    if args.command == "sp500-pit-report":
        report = report_sp500_pit(pit_dir=args.pit_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if str(report.get("status", "")).lower() == "fail":
            return 2
        return 0

    if args.command == "sp500-pit-template":
        out = create_sp500_manual_template(args.out)
        print(f"[DONE] sp500-pit-template out={out}")
        return 0

    if args.command == "sp500-pit-ingest-manual":
        summary = ingest_sp500_manual_events(
            args.events_file,
            append=not bool(args.replace),
            replace=bool(args.replace),
            target_path=args.target_path,
            strict=bool(args.strict),
            require_source_ref=bool(args.require_source_ref),
            require_source_doc_id=bool(args.require_source_doc_id),
            require_provenance_text=bool(args.require_provenance_text),
            require_confidence=bool(args.require_confidence),
            normalize_complex_actions=not bool(args.no_normalize_actions),
            rebuild_after_ingest=bool(args.rebuild),
            rebuild_start=str(args.rebuild_start),
            rebuild_end=args.rebuild_end,
            rebuild_provider_order=str(args.rebuild_provider_order),
            rebuild_seed_policy=str(args.rebuild_seed_policy),
            rebuild_fail_closed=bool(args.rebuild_fail_closed),
            rebuild_min_confidence=float(args.rebuild_min_confidence),
            rebuild_pit_dir=args.pit_dir,
        )
        print(
            f"[DONE] sp500-pit-ingest-manual status={summary.get('status')} "
            f"target={summary.get('target_path')} ingested_rows={summary.get('ingested_rows')} "
            f"rows_written={summary.get('rows_written')} issues={summary.get('issue_count')}"
        )
        return 0

    if args.command == "sp500-pit-import-github-secondary":
        summary = import_sp500_github_secondary(
            out_path=args.out,
            repo_url=str(args.repo_url),
            raw_url=args.raw_url,
            cache_dir=args.cache_dir,
            confidence_default=float(args.confidence_default),
            append=not bool(args.replace),
            replace=bool(args.replace),
            target_path=args.target_path,
            strict=bool(args.strict),
            fail_closed=bool(args.fail_closed),
            dry_run=bool(args.dry_run),
            force_refresh=bool(args.force_refresh),
            since_date=args.since_date,
            until_date=args.until_date,
            parse_fail_rate_threshold=float(args.parse_fail_rate_threshold),
            missing_ticker_rate_threshold=float(args.missing_ticker_rate_threshold),
            rebuild_after_import=bool(args.rebuild),
            rebuild_start=str(args.rebuild_start),
            rebuild_end=args.rebuild_end,
            rebuild_provider_order=str(args.rebuild_provider_order),
            rebuild_seed_policy=str(args.rebuild_seed_policy),
            rebuild_fail_closed=bool(args.rebuild_fail_closed),
            rebuild_min_confidence=float(args.rebuild_min_confidence),
            pit_dir=args.pit_dir,
        )
        print(
            f"[DONE] sp500-pit-import-github-secondary status={summary.get('status')} "
            f"events_rows={summary.get('events_rows')} dry_run={summary.get('dry_run')} "
            f"out={summary.get('out_path')}"
        )
        stats = summary.get("stats", {}) or {}
        print(
            "[STATS] "
            f"schema_drift_detected={stats.get('schema_drift_detected')} "
            f"parse_fail_rate={stats.get('parse_fail_rate', 0.0):.2%} "
            f"missing_ticker_rate={stats.get('missing_ticker_rate', 0.0):.2%} "
            f"import_count={stats.get('github_secondary_import_count', 0)}"
        )
        if summary.get("fail_reasons"):
            print(f"[WARN] fail_reasons={summary.get('fail_reasons')}")
        return 0

    if args.command == "sp500-pit-diff":
        payload = sp500_pit_diff(
            date_a=str(args.date_a),
            date_b=str(args.date_b),
            min_confidence=float(args.min_confidence),
            pit_path=(Path(args.pit_dir).expanduser() / "sp500_constituents_pit_intervals.parquet") if args.pit_dir else None,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "backtest-screen":
        return run_backtest_screen_cli(args)

    if args.command == "backtest-strategy":
        return run_backtest_strategy_cli(args)

    if args.command == "rank-test":
        return run_rank_test_cli(args)

    if args.command == "wrds":
        if run_wrds_command is None:
            print("[error] wrds package not installed. Run: pip install wrds")
            return 1
        return run_wrds_command(args)

    parser.print_help()
    return 1
