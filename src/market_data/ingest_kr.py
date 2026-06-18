from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from market_data import db_reader_kr, db_writer_kr
from market_data import parquet_store
from market_data.kr_ksic import fetch_ksic_dim
from market_data.kr_dart.corp_master import fetch_corp_master, merge_with_ticker_master
from market_data.kr_dart.filings import fetch_filings_for_corp
from market_data.kr_dart.financials import REPORT_CODES, fetch_single_account_financials
from market_data.kr_dart.materialize import materialize_financials_quarterly
from market_data.krx.indices import fetch_index_price_frame, resolve_representative_indices
from market_data.krx.investors import fetch_investor_flow_frame
from market_data.krx.normalize import normalize_kr_tickers
from market_data.krx.prices import fetch_price_frame
from market_data.krx.universe import build_ticker_master
from market_data.utils import append_csv_row, ensure_dir, now_utc_iso


@dataclass
class KRXIngestOptions:
    start: str
    end: str
    tickers: list[str] = field(default_factory=list)
    tickers_file: str | None = None
    include_dart: bool = False
    materialize_dart: bool = True
    skip_master: bool = False
    skip_prices: bool = False
    skip_investors: bool = False
    skip_indices: bool = False
    fresh_days: int | None = None
    force: bool = False
    use_universe: bool = True
    workers: int = 1
    skip_dart_company_enrich: bool = False


def _match_filing_metadata(filing_frame: pd.DataFrame, *, year: int, report_code: str) -> dict[str, object]:
    if filing_frame is None or filing_frame.empty:
        return {}
    matched = filing_frame.copy()
    if "report_code" in matched.columns:
        matched = matched.loc[matched["report_code"].astype(str) == str(report_code)]
    if "period_end" in matched.columns:
        period_end = pd.to_datetime(matched["period_end"], errors="coerce")
        matched = matched.loc[period_end.dt.year == int(year)]
    if matched.empty:
        return {}
    if "filing_date" in matched.columns:
        matched = matched.sort_values("filing_date")
    row = matched.iloc[-1]
    return {
        "receipt_no": row.get("accession") or row.get("receipt_no"),
        "filing_date": row.get("filing_date"),
        "period_end": row.get("period_end"),
    }


def _load_requested_tickers(opts: KRXIngestOptions) -> list[str]:
    requested = list(opts.tickers)
    if opts.tickers_file:
        path = Path(opts.tickers_file)
        df = pd.read_csv(path, dtype=str)
        if not df.empty:
            column = df.columns[0]
            requested.extend(df[column].dropna().astype(str).tolist())
    return normalize_kr_tickers(requested)


def _should_skip_ticker(ticker: str, opts: KRXIngestOptions) -> bool:
    if opts.force:
        return False
    payload = db_writer_kr.get_checkpoint(ticker, "kr")
    if not payload:
        return False
    if opts.include_dart and not bool(payload.get("include_dart")):
        return False
    if opts.fresh_days is None:
        return True
    fresh = db_writer_kr.get_fresh_tickers("kr", opts.fresh_days)
    return ticker in fresh


def ingest_krx_data(opts: KRXIngestOptions) -> int:
    ensure_dir(Path("logs"))
    db_writer_kr.init_schema()
    try:
        run_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        failures_path = Path("logs") / "failures_kr.csv"
        failure_fields = ["run_id", "timestamp", "market", "ticker", "step", "error"]
        requested = _load_requested_tickers(opts)

        if opts.skip_master:
            ticker_master = db_reader_kr.load_ticker_master_all()
            if requested:
                ticker_master = ticker_master.loc[ticker_master["ticker"].astype(str).isin(requested)].copy()
        else:
            ticker_master = build_ticker_master(tickers=requested or None)
            if opts.include_dart:
                ksic_dim = fetch_ksic_dim()
                db_writer_kr.replace_ksic_dim(ksic_dim)
                corp_master = fetch_corp_master(
                    ticker_master=ticker_master,
                    enrich_company=not bool(getattr(opts, "skip_dart_company_enrich", False)),
                )
                db_writer_kr.replace_dart_corp_master(corp_master)
                ticker_master = merge_with_ticker_master(ticker_master, corp_master, ksic_dim)
            db_writer_kr.replace_ticker_master(ticker_master)

        common_universe = ticker_master.loc[ticker_master["is_common_stock"].fillna(False)].copy()
        if requested:
            common_universe = common_universe.loc[common_universe["ticker"].astype(str).isin(set(requested))].copy()

        # Apply universe filter if --use-universe is set
        if opts.use_universe and not requested:
            from market_data.universe_builder import load_universe
            universe_tickers = load_universe(market="kr")
            if universe_tickers:
                common_universe = common_universe.loc[
                    common_universe["ticker"].astype(str).isin(set(universe_tickers))
                ].copy()
                print(f"[KRX] Universe filter applied: {len(common_universe)} tickers (from {len(universe_tickers)} in universe)")
            else:
                print("[KRX WARN] --use-universe set but no universe file found, using all common stocks")

        common_universe = common_universe.sort_values(["market_tier", "ticker"]).reset_index(drop=True)
        if common_universe.empty:
            print("[ERROR] No KRX common-stock universe rows found.")
            return 2

        print(
            f"[KRX] tickers={len(common_universe)} start={opts.start} end={opts.end} "
            f"include_dart={opts.include_dart} force={opts.force} workers={opts.workers}"
        )

        ok = 0
        skipped = 0
        failed = 0

        total = len(common_universe)

        def _process_item(index: int, item: pd.Series) -> dict[str, object]:
            ticker = str(item["ticker"])
            try:
                if _should_skip_ticker(ticker, opts):
                    return {"ticker": ticker, "status": "skipped", "index": index}

                if not opts.skip_prices:
                    price_frame = fetch_price_frame(
                        ticker=ticker,
                        start=opts.start,
                        end=opts.end,
                        ticker_name=str(item.get("ticker_name") or ""),
                        market_tier=str(item.get("market_tier") or ""),
                    )
                    db_writer_kr.upsert_prices(price_frame, ticker=ticker, market="kr")

                if not opts.skip_investors:
                    investor_frame = fetch_investor_flow_frame(
                        ticker=ticker,
                        start=opts.start,
                        end=opts.end,
                        ticker_name=str(item.get("ticker_name") or ""),
                        market_tier=str(item.get("market_tier") or ""),
                    )
                    db_writer_kr.upsert_investor_flows(investor_frame, ticker=ticker, market="kr")

                if opts.include_dart:
                    corp_code = str(item.get("dart_corp_code") or "").strip()
                    if corp_code:
                        filing_frame = fetch_filings_for_corp(
                            corp_code=corp_code,
                            ticker=ticker,
                            ticker_name=str(item.get("ticker_name") or ""),
                            start_date=opts.start.replace("-", ""),
                            end_date=opts.end.replace("-", ""),
                        )
                        db_writer_kr.upsert_filings(filing_frame, ticker=ticker, market="kr")
                        parts: list[pd.DataFrame] = []
                        start_year = int(str(opts.start)[:4])
                        end_year = int(str(opts.end)[:4])
                        for year in range(start_year, end_year + 1):
                            for report_code in REPORT_CODES:
                                filing_meta = _match_filing_metadata(
                                    filing_frame,
                                    year=year,
                                    report_code=report_code,
                                )
                                raw = fetch_single_account_financials(
                                    corp_code=corp_code,
                                    ticker=ticker,
                                    bsns_year=year,
                                    reprt_code=report_code,
                                    receipt_no=filing_meta.get("receipt_no"),
                                    filing_date=filing_meta.get("filing_date"),
                                    period_end=filing_meta.get("period_end"),
                                )
                                if not raw.empty:
                                    parts.append(raw)
                        parts = [
                            part.dropna(axis=1, how="all")
                            for part in parts
                            if part is not None and not part.empty and not part.dropna(how="all").empty
                        ]
                        if parts:
                            raw_frame = pd.concat(parts, ignore_index=True, sort=False)
                            db_writer_kr.upsert_dart_financials_raw(raw_frame, corp_code=corp_code)

                db_writer_kr.save_checkpoint(
                    ticker=ticker,
                    market="kr",
                    payload={
                        "run_id": run_id,
                        "completed_at": now_utc_iso(),
                        "fresh_days": opts.fresh_days,
                        "include_dart": opts.include_dart,
                    },
                )
                return {"ticker": ticker, "status": "ok", "index": index}
            except Exception as exc:  # noqa: BLE001
                return {"ticker": ticker, "status": "failed", "index": index, "error": str(exc)}

        if int(opts.workers) > 1:
            with ThreadPoolExecutor(max_workers=int(opts.workers)) as executor:
                future_map = {
                    executor.submit(_process_item, index, item): (index, str(item["ticker"]))
                    for index, item in common_universe.iterrows()
                }
                for future in as_completed(future_map):
                    result = future.result()
                    ticker = str(result["ticker"])
                    idx = int(result["index"]) + 1
                    status = str(result["status"])
                    if status == "ok":
                        ok += 1
                        print(f"{ticker}...OK ({idx}/{total})")
                    elif status == "skipped":
                        skipped += 1
                        print(f"{ticker}...SKIP ({idx}/{total})")
                    else:
                        failed += 1
                        append_csv_row(
                            failures_path,
                            {
                                "run_id": run_id,
                                "timestamp": now_utc_iso(),
                                "market": "kr",
                                "ticker": ticker,
                                "step": "krx_ingest",
                                "error": str(result.get("error", "unknown")),
                            },
                            failure_fields,
                        )
                        print(f"{ticker}...FAIL ({idx}/{total}) {result.get('error')}")
        else:
            for index, item in common_universe.iterrows():
                result = _process_item(index, item)
                ticker = str(result["ticker"])
                idx = int(result["index"]) + 1
                status = str(result["status"])
                if status == "ok":
                    ok += 1
                    print(f"{ticker}...OK ({idx}/{total})")
                elif status == "skipped":
                    skipped += 1
                    print(f"{ticker}...SKIP ({idx}/{total})")
                else:
                    failed += 1
                    append_csv_row(
                        failures_path,
                        {
                            "run_id": run_id,
                            "timestamp": now_utc_iso(),
                            "market": "kr",
                            "ticker": ticker,
                            "step": "krx_ingest",
                            "error": str(result.get("error", "unknown")),
                        },
                        failure_fields,
                    )
                    print(f"{ticker}...FAIL ({idx}/{total}) {result.get('error')}")

        if not opts.skip_indices:
            try:
                index_targets = resolve_representative_indices()
                for _, (index_code, index_name) in index_targets.items():
                    frame = fetch_index_price_frame(
                        index_code=index_code,
                        index_name=index_name,
                        start=opts.start,
                        end=opts.end,
                    )
                    db_writer_kr.upsert_index_prices(frame, index_code=index_code)
                print(f"[KRX] index_targets={len(index_targets)}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[KRX WARN] index ingest failed: {exc}")

        if opts.include_dart and opts.materialize_dart:
            # Read raw financials from Parquet first, DuckDB fallback
            raw = parquet_store.load_dart_financials_raw_all("kr")
            if raw.empty:
                try:
                    raw = db_reader_kr.load_dart_financials_raw_all()
                except Exception:
                    raw = pd.DataFrame()

            filings = parquet_store.load_filings_all("kr")
            if filings.empty:
                try:
                    filings = db_reader_kr.load_filings_all()
                except Exception:
                    filings = pd.DataFrame()

            if not raw.empty:
                universe_tickers = set(common_universe["ticker"].astype(str))
                raw = raw.loc[raw["ticker"].astype(str).isin(universe_tickers)].copy()
                if not filings.empty:
                    filings = filings.loc[filings["ticker"].astype(str).isin(universe_tickers)].copy()

                materialized = materialize_financials_quarterly(raw, filings=filings, ticker_master=common_universe)
                if materialized is not None and not materialized.empty and "ticker" in materialized.columns:
                    for ticker, chunk in materialized.groupby("ticker", sort=False):
                        db_writer_kr.upsert_financials(chunk.reset_index(drop=True), ticker=ticker, market="kr")
                print(f"[KRX] materialized_financials={len(materialized)}")
            else:
                print("[KRX] No raw financials to materialize")

        # ── Export JSON ticker files for web serving ────────────────────
        try:
            from market_data.export_json import export_ticker_to_file, export_ticker_master, _update_last_updated

            exported_tickers = [str(row["ticker"]) for _, row in common_universe.iterrows()]
            json_ok = 0
            for t in exported_tickers:
                try:
                    path = export_ticker_to_file(t, market="kr")
                    if path:
                        json_ok += 1
                except Exception:
                    pass
            export_ticker_master(market="kr")
            _update_last_updated("kr", json_ok)
            print(f"[JSON] exported {json_ok}/{len(exported_tickers)} ticker JSON files")
        except Exception as exc:
            print(f"[JSON WARN] JSON export failed: {exc}")

        print(f"[DONE] ok={ok} skipped={skipped} failed={failed}")
        return 0 if failed == 0 else 2
    finally:
        db_writer_kr.close()
