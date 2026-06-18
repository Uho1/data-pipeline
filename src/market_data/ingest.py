from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf

from market_data.config import FINANCIAL_FILE_MAP, LOGS_DIR, PRICE_REQUIRED_COLUMNS
from market_data import db_writer
from market_data.derived_factors import build_derived_factors_quarterly
from market_data.sec_financials import (
    SEC_DEFAULT_START_DATE,
    SEC_FILINGS_CACHE_DIR,
    SEC_FAST_DATA_API_MIN_REQUEST_INTERVAL_SECONDS,
    SEC_RAW_COMPANYFACTS_DIR,
    SEC_RAW_SUBMISSIONS_DIR,
    build_sec_issuer_profile,
    build_sec_enrichment_frames,
    cleanup_sec_ticker_cache,
    configure_sec_request_throttle,
    fetch_companyfacts,
    fetch_sec_filing_history,
    fetch_sec_quarterly_history,
    fetch_submissions,
)
from market_data.sec_sector_proxy import build_sector_proxy_cache_for_universe, ensure_sector_proxy_reference_files
from market_data.universe import build_universe
from market_data.utils import append_csv_row, coerce_datetime_index, ensure_dir, now_utc_iso, retry_call

DEFAULT_FINANCIAL_WORKERS = 3
SEC_FAST_DERIVED_PRICE_PREFETCH_CHUNK_SIZE = 250


@dataclass
class IngestOptions:
    universe: str
    start: str
    end: str | None
    interval: str
    tickers_file: str | None
    kospi_external_url: str
    kospi_top_n: int | None
    fresh_days: int | None
    retries: int
    backoff_base: float
    financial_workers: int
    workers: int
    force: bool
    price_batch_size: int
    disable_price_batch: bool
    financial_source: str
    sec_user_agent: str | None
    use_next_trading_day_availability: bool
    fundamentals_availability_fallback: bool
    fundamentals_fallback_q_days: int
    fundamentals_fallback_k_days: int
    reparse_sec_from_cache: bool = False
    backfill_financials_extra: bool = False
    include_sector_cache: bool = True
    skip_price: bool = False
    sec_financials_only: bool = False
    persist_sec_raw_cache: bool = False
    persist_sec_filing_cache: bool = False
    price_start: str | None = None
    financial_start: str | None = None


INCOME_COLUMNS = [
    "Revenue",
    "COGS",
    "Gross Profit",
    "SG&A",
    "Operating Income",
    "Net Income",
    "Net Income Common",
    "EPS",
    "Diluted EPS",
    "D&A",
    "SBC",
    "Interest",
    "Pretax Income",
    "Tax",
    "Shares",
]

BALANCE_COLUMNS = [
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
]

CASHFLOW_COLUMNS = [
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Capital Expenditure",
    "Dividends Paid",
    "Repurchases",
]

META_COLUMNS = [
    "Price",
    "Price_M1",
    "Price_M2",
    "Price_M3",
    "name",
    "name_kr",
    "sector",
    "industry",
    "avg_volume",
    "PeriodStart",
    "PeriodEnd",
    "FormType",
    "FilingDate",
    "AcceptedAt",
    "AvailableDate",
    "AvailabilityMethod",
    "fiscal_year",
    "fiscal_quarter",
    "fiscal_label",
]

BACKFILL_TARGET_COLUMNS = [
    "Current Assets",
    "Current Liabilities",
    "AR",
    "AP",
    "Inventory",
    "Cash",
    "Debt Short",
    "Debt Long",
    "Pretax Income",
    "Tax",
    "D&A",
    "SBC",
    "Dividends Paid",
    "Repurchases",
]


def _extract_single_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.get_level_values(-1)
        if ticker in levels:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _prepare_price_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = _extract_single_ticker_frame(raw, ticker)
    df = coerce_datetime_index(df)
    for col in PRICE_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col in {"Dividends", "Stock Splits"} else pd.NA
    out = df[PRICE_REQUIRED_COLUMNS].copy()
    out.index.name = "Date"
    out["Ticker"] = ticker
    out["CollectedAt"] = now_utc_iso()
    return out


def _normalize_statement(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["StatementDate", "CollectedAt"])

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(x) for x in c if str(x) != ""]) for c in out.columns]

    out.columns = pd.to_datetime(out.columns, errors="coerce")
    out = out.loc[:, ~out.columns.isna()]
    out = out.T
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()]
    out = out.sort_index()
    out.index.name = "StatementDate"
    out = out.reset_index()
    out["PeriodEnd"] = pd.to_datetime(out["StatementDate"], errors="coerce")
    out["PeriodStart"] = pd.NaT
    out["FormType"] = pd.NA
    out["FilingDate"] = pd.NaT
    out["AcceptedAt"] = pd.NaT
    out["AvailableDate"] = pd.to_datetime(out["StatementDate"], errors="coerce")
    out["AvailabilityMethod"] = "statement_date"
    out["CollectedAt"] = now_utc_iso()
    out["Source"] = "yfinance"
    return out


def _select_columns(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    base_candidates = ["StatementDate", "term", "symbol"]
    cols = [c for c in base_candidates if c in df.columns] + [c for c in value_columns if c in df.columns]
    out = df[cols].copy() if cols else pd.DataFrame()
    if "term" in out.columns:
        out = out.rename(columns={"term": "Term"})
    if "StatementDate" in out.columns:
        out["StatementDate"] = pd.to_datetime(out["StatementDate"], errors="coerce")
    for meta_col in META_COLUMNS:
        if meta_col not in out.columns and meta_col in df.columns:
            out[meta_col] = df[meta_col]
    if "PeriodEnd" not in out.columns:
        out["PeriodEnd"] = pd.to_datetime(out.get("StatementDate"), errors="coerce")
    if "AvailableDate" not in out.columns:
        out["AvailableDate"] = pd.to_datetime(out.get("StatementDate"), errors="coerce")
    if "Source" not in out.columns:
        out["Source"] = "sec"
    out["CollectedAt"] = now_utc_iso()
    return out


def _quarterly_to_annual_last(quarterly_df: pd.DataFrame) -> pd.DataFrame:
    if quarterly_df.empty:
        return quarterly_df.copy()
    out = quarterly_df.copy()
    out["StatementDate"] = pd.to_datetime(out["StatementDate"], errors="coerce")
    out = out.loc[~out["StatementDate"].isna()].sort_values("StatementDate")
    if out.empty:
        return quarterly_df.iloc[0:0].copy()
    out["Year"] = out["StatementDate"].dt.year
    idx = out.groupby("Year")["StatementDate"].idxmax()
    out = out.loc[idx].sort_values("StatementDate").drop(columns=["Year"])
    return out.reset_index(drop=True)


def _build_statement_frames_from_history(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if history is None or history.empty:
        empty = pd.DataFrame(columns=["StatementDate", "Term", "CollectedAt", "Source"])
        return {
            "income_quarterly": empty.copy(),
            "balance_quarterly": empty.copy(),
            "cashflow_quarterly": empty.copy(),
            "income_annual": empty.copy(),
            "balance_annual": empty.copy(),
            "cashflow_annual": empty.copy(),
        }
    quarter_income = _select_columns(history, INCOME_COLUMNS + META_COLUMNS)
    quarter_balance = _select_columns(history, BALANCE_COLUMNS + META_COLUMNS)
    quarter_cash = _select_columns(history, CASHFLOW_COLUMNS + META_COLUMNS)
    annual_income = _quarterly_to_annual_last(quarter_income)
    annual_balance = _quarterly_to_annual_last(quarter_balance)
    annual_cash = _quarterly_to_annual_last(quarter_cash)
    return {
        "income_quarterly": quarter_income,
        "balance_quarterly": quarter_balance,
        "cashflow_quarterly": quarter_cash,
        "income_annual": annual_income,
        "balance_annual": annual_balance,
        "cashflow_annual": annual_cash,
    }


def _merge_yf_quarterly(fin_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge yfinance income/balance/cashflow quarterly frames into one row per period."""
    frames = [
        fin_map.get("income_quarterly", pd.DataFrame()),
        fin_map.get("balance_quarterly", pd.DataFrame()),
        fin_map.get("cashflow_quarterly", pd.DataFrame()),
    ]
    non_empty = [f for f in frames if f is not None and not f.empty]
    if not non_empty:
        return pd.DataFrame()
    if len(non_empty) == 1:
        return non_empty[0].copy()

    key_cols = [c for c in ["StatementDate", "PeriodEnd", "AvailableDate", "FilingDate", "AcceptedAt", "FormType"] if c in non_empty[0].columns]
    merged = non_empty[0].copy()
    for nxt in non_empty[1:]:
        value_cols = [c for c in nxt.columns if c not in key_cols and c not in merged.columns]
        if value_cols:
            merged = merged.merge(nxt[key_cols + value_cols], on=key_cols, how="outer")
    return merged.reset_index(drop=True)


def _download_price(ticker: str, start: str, end: str | None, interval: str, retries: int, backoff: float) -> pd.DataFrame:
    def _op() -> pd.DataFrame:
        return yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
            group_by="column",
        )

    raw = retry_call(_op, retries=retries, backoff_base=backoff, label=f"price:{ticker}")
    if raw is None or raw.empty:
        raise RuntimeError(f"No price data returned for {ticker}")
    return _prepare_price_frame(raw, ticker)


def _download_price_batch(
    tickers: list[str],
    start: str,
    end: str | None,
    interval: str,
    retries: int,
    backoff: float,
) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}

    def _op() -> pd.DataFrame:
        return yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
            group_by="column",
        )

    raw = retry_call(
        _op,
        retries=retries,
        backoff_base=backoff,
        label=f"price-batch:{len(tickers)}",
    )
    if raw is None or raw.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            prepared = _prepare_price_frame(raw, ticker)
            if not prepared.empty and prepared.get("Close", pd.Series()).notna().any():
                out[ticker] = prepared
        except Exception:
            continue
    return out


def _download_financial_statement(
    ticker: str,
    attr: str,
    retries: int,
    backoff: float,
) -> pd.DataFrame:
    def _op() -> pd.DataFrame:
        tkr = yf.Ticker(ticker)
        return getattr(tkr, attr)

    return retry_call(_op, retries=retries, backoff_base=backoff, label=f"{attr}:{ticker}")


def _download_financials_yfinance(
    ticker: str,
    retries: int,
    backoff: float,
    financial_workers: int,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    items = list(FINANCIAL_FILE_MAP.items())

    if financial_workers <= 1:
        for stem, attr in items:
            df = _download_financial_statement(ticker, attr, retries, backoff)
            out[stem] = _normalize_statement(df)
        return out

    max_workers = min(financial_workers, len(items))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_financial_statement, ticker, attr, retries, backoff): stem
            for stem, attr in items
        }
        for future in as_completed(futures):
            stem = futures[future]
            out[stem] = _normalize_statement(future.result())

    return {stem: out[stem] for stem in FINANCIAL_FILE_MAP.keys()}


def _download_financials_sec(
    ticker: str,
    market: str,
    opts: IngestOptions,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if str(market).strip().lower() != "us":
        raise RuntimeError("SEC financial source supports only market=us")

    financial_start = getattr(opts, "financial_start", None) or opts.start
    start_ts = pd.to_datetime(financial_start, errors="coerce")
    if pd.isna(start_ts):
        start_ts = SEC_DEFAULT_START_DATE
    min_start = max(pd.Timestamp(start_ts).normalize(), SEC_DEFAULT_START_DATE)
    raw_cache_dir = (
        SEC_RAW_COMPANYFACTS_DIR
        if (bool(opts.persist_sec_raw_cache) or bool(opts.reparse_sec_from_cache))
        else None
    )
    submissions_cache_dir = (
        SEC_RAW_SUBMISSIONS_DIR
        if (bool(opts.persist_sec_raw_cache) or bool(opts.reparse_sec_from_cache))
        else None
    )
    filings_cache_dir = (
        SEC_FILINGS_CACHE_DIR
        if (bool(opts.persist_sec_filing_cache) or bool(opts.reparse_sec_from_cache))
        else None
    )
    prefetched_companyfacts: dict[str, Any] | None = None
    prefetched_submissions: dict[str, Any] | None = None
    prefetched_cik: int | None = None

    try:
        prefetched_companyfacts, prefetched_cik = fetch_companyfacts(
            ticker=ticker,
            user_agent=opts.sec_user_agent,
            force_refresh=opts.force and not bool(opts.reparse_sec_from_cache),
            cache_only=bool(opts.reparse_sec_from_cache),
            retries=opts.retries,
            backoff=opts.backoff_base,
            raw_cache_dir=raw_cache_dir,
        )
    except Exception:
        prefetched_companyfacts = None
        prefetched_cik = None

    if prefetched_cik is not None:
        try:
            prefetched_submissions = fetch_submissions(
                ticker=ticker,
                cik=prefetched_cik,
                user_agent=opts.sec_user_agent,
                force_refresh=opts.force and not bool(opts.reparse_sec_from_cache),
                cache_only=bool(opts.reparse_sec_from_cache),
                retries=opts.retries,
                backoff=opts.backoff_base,
                raw_cache_dir=submissions_cache_dir,
            )
        except Exception:
            prefetched_submissions = None

    history = fetch_sec_quarterly_history(
        ticker=ticker,
        market=market,
        start=min_start,
        user_agent=opts.sec_user_agent,
        force_refresh=opts.force,
        reparse_from_cache=bool(opts.reparse_sec_from_cache),
        retries=opts.retries,
        backoff=opts.backoff_base,
        use_next_trading_day_availability=bool(opts.use_next_trading_day_availability),
        availability_fallback=bool(opts.fundamentals_availability_fallback),
        fallback_q_days=max(0, int(opts.fundamentals_fallback_q_days)),
        fallback_k_days=max(0, int(opts.fundamentals_fallback_k_days)),
        raw_cache_dir=raw_cache_dir,
        submissions_cache_dir=submissions_cache_dir,
        filings_cache_dir=filings_cache_dir,
        prefetched_companyfacts=prefetched_companyfacts,
        prefetched_submissions=prefetched_submissions,
        prefetched_cik=prefetched_cik,
    )
    filings = pd.DataFrame()
    raw_facts = pd.DataFrame()
    financials_extra = pd.DataFrame()
    if not bool(getattr(opts, "sec_financials_only", False)):
        filings = fetch_sec_filing_history(
            ticker=ticker,
            market=market,
            start=min_start,
            user_agent=opts.sec_user_agent,
            force_refresh=opts.force,
            cache_only=bool(opts.reparse_sec_from_cache),
            retries=opts.retries,
            backoff=opts.backoff_base,
            submissions_cache_dir=submissions_cache_dir,
            prefetched_submissions=prefetched_submissions,
            prefetched_cik=prefetched_cik,
        )
        if not history.empty:
            raw_facts, financials_extra = build_sec_enrichment_frames(
                ticker=ticker,
                market=market,
                start=min_start,
                quarterly_frame=history,
                user_agent=opts.sec_user_agent,
                force_refresh=opts.force,
                reparse_from_cache=bool(opts.reparse_sec_from_cache),
                retries=opts.retries,
                backoff=opts.backoff_base,
                use_next_trading_day_availability=bool(opts.use_next_trading_day_availability),
                availability_fallback=bool(opts.fundamentals_availability_fallback),
                fallback_q_days=max(0, int(opts.fundamentals_fallback_q_days)),
                fallback_k_days=max(0, int(opts.fundamentals_fallback_k_days)),
                raw_cache_dir=raw_cache_dir,
                submissions_cache_dir=submissions_cache_dir,
                prefetched_companyfacts=prefetched_companyfacts,
                prefetched_submissions=prefetched_submissions,
                prefetched_cik=prefetched_cik,
            )
    fin_map = _build_statement_frames_from_history(history)
    ordered = {stem: fin_map.get(stem, pd.DataFrame()) for stem in FINANCIAL_FILE_MAP.keys()}
    issuer_profile = build_sec_issuer_profile(
        ticker=ticker,
        market=market,
        companyfacts=prefetched_companyfacts,
        submissions=prefetched_submissions,
        cik=prefetched_cik,
        user_agent=opts.sec_user_agent,
    )
    return ordered, history, filings, raw_facts, financials_extra, issuer_profile


def _checkpoint_satisfies_request(payload: dict[str, Any], opts: IngestOptions) -> bool:
    requested_source = str(getattr(opts, "financial_source", "") or "").strip().lower()
    checkpoint_source = str(payload.get("financial_source", "") or "").strip().lower()

    # Legacy checkpoints without capability metadata are not enough to skip
    # a richer SEC backfill run.
    if requested_source == "sec":
        if checkpoint_source != "sec":
            return False
        if not bool(getattr(opts, "sec_financials_only", False)) and bool(payload.get("sec_financials_only", False)):
            return False
        if bool(getattr(opts, "backfill_financials_extra", False)) and not bool(payload.get("backfill_financials_extra", False)):
            return False

    if not bool(getattr(opts, "skip_price", False)) and bool(payload.get("skip_price", False)):
        return False

    return True


def _should_skip_ticker(market: str, ticker: str, opts: IngestOptions) -> bool:
    if opts.force:
        return False
    payload = db_writer.get_checkpoint(ticker, market)
    if not payload:
        return False
    if not _checkpoint_satisfies_request(payload, opts):
        return False
    if opts.fresh_days is None:
        return True
    completed_str = payload.get("completed_at", "")
    if not completed_str:
        return False
    try:
        completed = pd.to_datetime(completed_str, utc=True)
        age_days = (pd.Timestamp.now(tz="UTC") - completed).days
        return age_days < opts.fresh_days
    except Exception:
        return False


def _process_ticker(
    ticker: str,
    market: str,
    opts: IngestOptions,
    run_id: str,
    prefetched_price: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if not opts.backfill_financials_extra and _should_skip_ticker(market, ticker, opts):
        return {"ticker": ticker, "status": "skipped", "reason": "fresh checkpoint"}

    try:
        price_for_derived: pd.DataFrame | None = None
        skip_price_download = bool(opts.skip_price or opts.backfill_financials_extra)
        if skip_price_download and prefetched_price is not None and not prefetched_price.empty:
            price_for_derived = prefetched_price.copy()
        if not skip_price_download:
            prefetch_valid = (
                prefetched_price is not None
                and not prefetched_price.empty
                and prefetched_price.get("Close", pd.Series()).notna().any()
            )
            if prefetch_valid:
                price_df = prefetched_price
            else:
                price_df = _download_price(
                    ticker,
                    getattr(opts, "price_start", None) or opts.start,
                    opts.end,
                    opts.interval,
                    opts.retries,
                    opts.backoff_base,
                )
            rows_written = db_writer.upsert_prices(price_df, ticker, market)
            price_for_derived = price_df.copy()
            print(f"[SAVE] price {ticker} -> DB ({rows_written} rows)")

        if opts.financial_source == "sec":
            (
                _fin_map,
                raw_history,
                filing_history,
                raw_facts,
                financials_extra,
                issuer_profile,
            ) = _download_financials_sec(ticker, market, opts)
            rows_written = db_writer.upsert_financials(raw_history, ticker, market)
            filing_written = db_writer.upsert_filings(filing_history, ticker, market)
            issuer_written = db_writer.upsert_sec_issuer_registry(issuer_profile, ticker, market)
            raw_fact_written = db_writer.upsert_sec_facts_raw_normalized(raw_facts, ticker, market)
            extra_written = db_writer.upsert_financials_extra(financials_extra, ticker, market)
            derived_source = "materialized_sec"
        else:
            fin_map = _download_financials_yfinance(
                ticker,
                opts.retries,
                opts.backoff_base,
                opts.financial_workers,
            )
            merged = _merge_yf_quarterly(fin_map)
            rows_written = db_writer.upsert_financials(merged, ticker, market)
            filing_written = 0
            issuer_written = 0
            raw_fact_written = 0
            extra_written = 0
            raw_history = merged
            derived_source = "materialized_yfinance"

        if price_for_derived is None:
            try:
                from market_data.db_reader import load_price_from_db

                loaded = load_price_from_db(ticker=ticker, market=market)
                if loaded is not None:
                    price_for_derived = loaded[0]
            except Exception:
                price_for_derived = None

        derived_frame = build_derived_factors_quarterly(
            financials=raw_history,
            prices=price_for_derived,
            filings=filing_history,
            source=derived_source,
        )
        derived_written = int(len(derived_frame)) if derived_frame is not None else 0
        checkpoint_payload = {
            "run_id": run_id,
            "ticker": ticker,
            "market": market,
            "completed_at": now_utc_iso(),
            "fresh_days": opts.fresh_days,
            "financial_source": opts.financial_source,
            "backfill_financials_extra": bool(opts.backfill_financials_extra),
            "skip_price": bool(opts.skip_price),
            "sec_financials_only": bool(getattr(opts, "sec_financials_only", False)),
        }
        derived_written = db_writer.upsert_derived_factors(derived_frame, ticker, market)
        print(
            f"[SAVE] financials {ticker} → DB ({rows_written} rows), "
            f"filings={filing_written}, issuer={issuer_written}, "
            f"raw_facts={raw_fact_written}, extra={extra_written}, derived={derived_written}"
        )

        if opts.financial_source == "sec" and not bool(opts.reparse_sec_from_cache):
            try:
                cleanup_sec_ticker_cache(
                    ticker=ticker,
                    raw_cache_dir=SEC_RAW_COMPANYFACTS_DIR if not bool(opts.persist_sec_raw_cache) else None,
                    submissions_cache_dir=SEC_RAW_SUBMISSIONS_DIR if not bool(opts.persist_sec_raw_cache) else None,
                    filings_cache_dir=SEC_FILINGS_CACHE_DIR if not bool(opts.persist_sec_filing_cache) else None,
                )
            except Exception:
                pass

        db_writer.save_checkpoint(ticker, market, checkpoint_payload)
        return {"ticker": ticker, "status": "ok"}
    except Exception as exc:  # noqa: BLE001
        if opts.financial_source == "sec" and not bool(opts.reparse_sec_from_cache):
            try:
                cleanup_sec_ticker_cache(
                    ticker=ticker,
                    raw_cache_dir=SEC_RAW_COMPANYFACTS_DIR if not bool(opts.persist_sec_raw_cache) else None,
                    submissions_cache_dir=SEC_RAW_SUBMISSIONS_DIR if not bool(opts.persist_sec_raw_cache) else None,
                    filings_cache_dir=SEC_FILINGS_CACHE_DIR if not bool(opts.persist_sec_filing_cache) else None,
                )
            except Exception:
                pass
        return {"ticker": ticker, "status": "failed", "error": str(exc)}


def _prefetch_prices(
    tickers: list[str],
    opts: IngestOptions,
) -> dict[str, pd.DataFrame]:
    if not tickers or opts.disable_price_batch:
        return {}

    chunk_size = max(1, opts.price_batch_size)
    prefetched: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            batch_map = _download_price_batch(
                tickers=chunk,
                start=getattr(opts, "price_start", None) or opts.start,
                end=opts.end,
                interval=opts.interval,
                retries=opts.retries,
                backoff=opts.backoff_base,
            )
            prefetched.update(batch_map)
            print(
                f"[BATCH] prices chunk={i // chunk_size + 1} size={len(chunk)} "
                f"resolved={len(batch_map)}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[BATCH WARN] prices chunk failed size={len(chunk)} reason={exc}")
    return prefetched


def _progress_status_line(ticker: str, status: str, current: int, total: int) -> str:
    return f"{ticker}...{status} ({current}/{total})"


def _prefetch_existing_prices_for_derived(
    tickers: list[str],
    *,
    market: str,
    start: str | None,
    end: str | None,
    chunk_size: int = SEC_FAST_DERIVED_PRICE_PREFETCH_CHUNK_SIZE,
) -> dict[str, pd.DataFrame]:
    from market_data.db_reader import bulk_load_price_close_frames

    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    step = max(1, int(chunk_size))
    for offset in range(0, len(tickers), step):
        subset = tickers[offset : offset + step]
        out.update(
            bulk_load_price_close_frames(
                subset,
                market=market,
                start=start,
                end=end,
            )
        )
    return out


def ingest_data(opts: IngestOptions) -> int:
    ensure_dir(LOGS_DIR)

    db_writer.init_schema()

    fast_mode = bool(opts.financial_source == "sec" and getattr(opts, "sec_financials_only", False))
    configure_sec_request_throttle(
        archives=None,
        data_api=SEC_FAST_DATA_API_MIN_REQUEST_INTERVAL_SECONDS if fast_mode else None,
        sec_other=None,
    )

    symbols, market, universe_path = build_universe(
        universe=opts.universe,
        tickers_file=opts.tickers_file,
        kospi_external_url=opts.kospi_external_url,
        kospi_top_n=opts.kospi_top_n,
    )
    symbols = list(dict.fromkeys([s.strip() for s in symbols if s.strip()]))
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    effective_workers = opts.workers
    if opts.financial_source == "sec" and opts.workers > 1:
        print(
            "[INFO] SEC ingest uses concurrent fetch/parse workers with immediate per-ticker writes"
        )
    if fast_mode:
        print(
            f"[INFO] SEC fast mode lowers data.sec.gov min interval to "
            f"{SEC_FAST_DATA_API_MIN_REQUEST_INTERVAL_SECONDS:.2f}s"
        )

    print(f"[UNIVERSE] {opts.universe} symbols={len(symbols)} source={universe_path}")
    print(
        f"[INGEST] market={market} workers={effective_workers} fresh_days={opts.fresh_days} "
        f"force={opts.force} financial_source={opts.financial_source} "
        "source_mode=parquet_direct"
    )

    if opts.financial_source == "sec" and str(market).strip().lower() != "us":
        print("[ERROR] SEC financial source supports only market=us")
        return 2
    if opts.backfill_financials_extra and opts.financial_source != "sec":
        print("[ERROR] --backfill-financials-extra requires --financial-source sec")
        return 2

    before_null_summary: dict[str, dict[str, float]] | None = None
    if opts.backfill_financials_extra:
        before_null_summary = db_writer.get_financial_null_rate_summary(
            market=market,
            columns=BACKFILL_TARGET_COLUMNS,
        )
        print("[BACKFILL] financials extra mode enabled (SEC companyfacts expansion)")

    failures_path = LOGS_DIR / "failures.csv"
    failure_fields = [
        "run_id",
        "timestamp",
        "universe",
        "market",
        "ticker",
        "step",
        "error",
    ]

    ok = 0
    skipped = 0
    failed = 0
    total_symbols = len(symbols)
    prefetched_price_map: dict[str, pd.DataFrame] = {}

    if fast_mode and bool(opts.skip_price):
        prefetched_price_map = _prefetch_existing_prices_for_derived(
            symbols,
            market=market,
            start=getattr(opts, "price_start", None) or getattr(opts, "financial_start", None) or opts.start,
            end=opts.end,
        )
        print(f"[INFO] Prefetched minimal derived-price series for {len(prefetched_price_map)} tickers")
    elif effective_workers == 1 and not opts.disable_price_batch:
        fresh_set = db_writer.get_fresh_tickers(market, opts.fresh_days) if not opts.force else set()
        need_price_tickers = [t for t in symbols if opts.force or t not in fresh_set]
        prefetched_price_map = _prefetch_prices(need_price_tickers, opts)

    if effective_workers > 1:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_map = {
                executor.submit(
                    _process_ticker,
                    ticker,
                    market,
                    opts,
                    run_id,
                    prefetched_price_map.get(ticker),
                ): ticker
                for ticker in symbols
            }
            processed = 0
            for future in as_completed(future_map):
                result = future.result()
                processed += 1
                status = result["status"]
                if status == "ok":
                    ok += 1
                    print(_progress_status_line(result["ticker"], "OK", processed, total_symbols))
                elif status == "skipped":
                    skipped += 1
                    print(_progress_status_line(result["ticker"], "SKIP", processed, total_symbols))
                else:
                    failed += 1
                    append_csv_row(
                        failures_path,
                        {
                            "run_id": run_id,
                            "timestamp": now_utc_iso(),
                            "universe": opts.universe,
                            "market": market,
                            "ticker": result["ticker"],
                            "step": "ingest",
                            "error": result.get("error", "unknown"),
                        },
                        failure_fields,
                    )
                    print(_progress_status_line(result["ticker"], "FAIL", processed, total_symbols))
                    print(f"[FAIL] {result['ticker']} -> {result.get('error')}")
    else:
        for processed, ticker in enumerate(symbols, start=1):
            result = _process_ticker(
                ticker,
                market,
                opts,
                run_id,
                prefetched_price=prefetched_price_map.get(ticker),
            )
            status = result["status"]
            if status == "ok":
                ok += 1
                print(_progress_status_line(result["ticker"], "OK", processed, total_symbols))
            elif status == "skipped":
                skipped += 1
                print(_progress_status_line(result["ticker"], "SKIP", processed, total_symbols))
            else:
                failed += 1
                append_csv_row(
                    failures_path,
                    {
                        "run_id": run_id,
                        "timestamp": now_utc_iso(),
                        "universe": opts.universe,
                        "market": market,
                        "ticker": result["ticker"],
                        "step": "ingest",
                        "error": result.get("error", "unknown"),
                    },
                    failure_fields,
                )
                print(_progress_status_line(result["ticker"], "FAIL", processed, total_symbols))
                print(f"[FAIL] {result['ticker']} -> {result.get('error')}")

    print(f"[DONE] ok={ok} skipped={skipped} failed={failed}")
    if opts.backfill_financials_extra:
        after_null_summary = db_writer.get_financial_null_rate_summary(
            market=market,
            columns=BACKFILL_TARGET_COLUMNS,
        )
        print("[BACKFILL] null-rate summary (before -> after):")
        for col in BACKFILL_TARGET_COLUMNS:
            before = (before_null_summary or {}).get(col, {})
            after = after_null_summary.get(col, {})
            b_rate = before.get("null_rate")
            a_rate = after.get("null_rate")
            if b_rate is None or a_rate is None:
                continue
            print(
                f"  - {col}: {b_rate:.3f} -> {a_rate:.3f} "
                f"(nonnull {int(before.get('non_null', 0))} -> {int(after.get('non_null', 0))})"
            )

    if opts.include_sector_cache and opts.financial_source == "sec":
        print(f"[SECTOR] Building SEC sector proxy cache for {len(symbols)} symbols (workers={opts.workers}) ...")
        ensure_sector_proxy_reference_files()
        sector_summary = build_sector_proxy_cache_for_universe(
            symbols=symbols,
            market=market,
            user_agent=opts.sec_user_agent,
            workers=opts.workers,
        )
        print(
            f"[SECTOR] done total={sector_summary.get('total')} "
            f"ok={sector_summary.get('ok')} failed={sector_summary.get('failed')} "
            f"unclassified={sector_summary.get('unclassified')}"
        )

    try:
        from market_data.export_json import export_ticker_to_file, export_ticker_master, _update_last_updated

        json_ok = 0
        for t in symbols:
            try:
                path = export_ticker_to_file(t, market=market)
                if path:
                    json_ok += 1
            except Exception:
                pass
        export_ticker_master(market=market)
        _update_last_updated(market, json_ok)
        print(f"[JSON] exported {json_ok}/{len(symbols)} ticker JSON files")
    except Exception as exc:
        print(f"[JSON WARN] JSON export failed: {exc}")

    return 0 if failed == 0 else 2
