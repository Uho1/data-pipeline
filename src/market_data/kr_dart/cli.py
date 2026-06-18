from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_data import db_reader_kr, db_writer_kr
from market_data.db_router import normalize_kr_ticker
from market_data.kr_ksic import fetch_ksic_dim
from market_data.kr_dart.client import DartClient, DartRateLimitError
from market_data.kr_dart.corp_master import fetch_corp_master, merge_with_ticker_master
from market_data.kr_dart.filings import fetch_filings_for_corp
from market_data.kr_dart.financials import REPORT_CODES, fetch_single_account_financials
from market_data.kr_dart.materialize import materialize_financials_quarterly
# Segment modules removed — segment feature is retired


@dataclass
class KrDartOptions:
    command: str
    tickers: list[str]
    start_date: str
    end_date: str | None
    start_year: int
    end_year: int
    fs_div: str = "CFS"
    mode: str = "segments"
    workers: int = 20


def _listed_year(value: object) -> int | None:
    listed = pd.to_datetime(value, errors="coerce")
    if pd.isna(listed):
        return None
    return int(listed.year)


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


def _load_ticker_master_subset(tickers: list[str]) -> pd.DataFrame:
    master = db_reader_kr.load_ticker_master_all()
    if master.empty:
        return master
    if not tickers:
        return master
    normalized = {normalize_kr_ticker(value) for value in tickers}
    return master.loc[master["ticker"].astype(str).isin(normalized)].copy()


def _existing_raw_request_keys(existing_raw: pd.DataFrame, *, fs_div: str) -> set[tuple[int, str]]:
    if existing_raw is None or existing_raw.empty:
        return set()
    frame = existing_raw.copy()
    frame["bsns_year"] = pd.to_numeric(frame.get("bsns_year"), errors="coerce")
    frame["reprt_code"] = frame.get("reprt_code").astype(str)
    frame["fs_div"] = frame.get("fs_div").astype(str).str.upper()
    frame = frame.loc[
        frame["bsns_year"].notna()
        & frame["reprt_code"].isin(REPORT_CODES)
        & frame["fs_div"].eq(str(fs_div).strip().upper())
    ]
    return {
        (int(year), str(report_code))
        for year, report_code in zip(frame["bsns_year"], frame["reprt_code"], strict=False)
    }


def _planned_financial_requests(
    filing_frame: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    listed_date: object = None,
    existing_raw: pd.DataFrame | None = None,
    fs_div: str = "CFS",
) -> list[dict[str, object]]:
    effective_start_year = int(start_year)
    listed_year = _listed_year(listed_date)
    if listed_year is not None:
        effective_start_year = max(effective_start_year, listed_year)
    if filing_frame is None or filing_frame.empty:
        return []
    required_columns = {"report_code", "period_end", "filing_date"}
    if not required_columns.issubset(set(filing_frame.columns)):
        return []

    planned = filing_frame.copy()
    planned["report_code"] = planned["report_code"].astype(str)
    planned["period_end"] = pd.to_datetime(planned["period_end"], errors="coerce")
    planned["filing_date"] = pd.to_datetime(planned["filing_date"], errors="coerce")
    planned = planned.loc[planned["report_code"].isin(REPORT_CODES)].copy()
    planned = planned.loc[planned["period_end"].notna()].copy()
    if planned.empty:
        return []

    planned["period_year"] = planned["period_end"].dt.year.astype(int)
    planned = planned.loc[
        planned["period_year"].between(int(effective_start_year), int(end_year), inclusive="both")
    ].copy()
    if planned.empty:
        return []

    if "accession" not in planned.columns:
        planned["accession"] = None
    if "receipt_no" not in planned.columns:
        planned["receipt_no"] = None
    planned = planned.sort_values(["period_year", "report_code", "filing_date", "accession"])
    planned = planned.drop_duplicates(subset=["period_year", "report_code"], keep="last")
    existing_keys = _existing_raw_request_keys(
        existing_raw if existing_raw is not None else pd.DataFrame(),
        fs_div=fs_div,
    )

    requests: list[dict[str, object]] = []
    for row in planned.itertuples(index=False):
        period_year = int(getattr(row, "period_year"))
        report_code = str(getattr(row, "report_code"))
        if (period_year, report_code) in existing_keys:
            continue
        requests.append(
            {
                "year": period_year,
                "report_code": report_code,
                "receipt_no": getattr(row, "accession", None) or getattr(row, "receipt_no", None),
                "filing_date": getattr(row, "filing_date", None),
                "period_end": getattr(row, "period_end", None),
            }
        )
    return requests


def run_kr_dart_command(opts: KrDartOptions) -> int:
    db_writer_kr.init_schema()
    try:
        if opts.command == "corp-master":
            ticker_master = db_reader_kr.load_ticker_master_all()
            corp_master = fetch_corp_master(
                ticker_master=ticker_master if ticker_master is not None and not ticker_master.empty else None,
                enrich_company=bool(ticker_master is not None and not ticker_master.empty),
            )
            ksic_dim = fetch_ksic_dim()
            db_writer_kr.replace_dart_corp_master(corp_master)
            db_writer_kr.replace_ksic_dim(ksic_dim)
            if ticker_master is not None and not ticker_master.empty:
                ticker_master = merge_with_ticker_master(ticker_master, corp_master, ksic_dim)
                db_writer_kr.replace_ticker_master(ticker_master)
            print(f"[DONE] kr-dart corp-master rows={len(corp_master)} ksic_rows={len(ksic_dim)}")
            return 0

        ticker_master = _load_ticker_master_subset(opts.tickers)
        if ticker_master.empty:
            print("[ERROR] ticker_master is empty. Run krx-ingest or kr-dart corp-master first.")
            return 2

        corp_master = db_reader_kr.load_dart_corp_master_all()
        if opts.command == "filings":
            rows_written = 0
            total = len(ticker_master)
            for index, item in enumerate(ticker_master.itertuples(index=False), start=1):
                ticker = str(getattr(item, "ticker"))
                corp_code = str(getattr(item, "dart_corp_code", "") or "").strip()
                if not corp_code and not corp_master.empty:
                    matches = corp_master.loc[corp_master["ticker"].astype(str) == ticker]
                    corp_code = str(matches["corp_code"].iloc[0]) if not matches.empty else ""
                if not corp_code:
                    print(f"[kr-dart filings] {ticker} ({index}/{total}) SKIP no corp_code", flush=True)
                    continue
                try:
                    frame = fetch_filings_for_corp(
                        corp_code=corp_code,
                        ticker=ticker,
                        ticker_name=str(getattr(item, "ticker_name", "") or ""),
                        start_date=opts.start_date,
                        end_date=opts.end_date,
                    )
                except DartRateLimitError as exc:
                    print(
                        f"[STOP] kr-dart filings rate_limit ticker={ticker} progress={index}/{total} reason={exc}",
                        flush=True,
                    )
                    return 75
                written = db_writer_kr.upsert_filings(frame, ticker, "kr")
                rows_written += written
                print(f"[kr-dart filings] {ticker} ({index}/{total}) rows={written}", flush=True)
            print(f"[DONE] kr-dart filings rows={rows_written}")
            return 0

        if opts.command == "financials":
            rows_written = 0
            filings_all = db_reader_kr.load_filings_all()
            total = len(ticker_master)
            for index, item in enumerate(ticker_master.itertuples(index=False), start=1):
                ticker = str(getattr(item, "ticker"))
                corp_code = str(getattr(item, "dart_corp_code", "") or "").strip()
                if not corp_code and not corp_master.empty:
                    matches = corp_master.loc[corp_master["ticker"].astype(str) == ticker]
                    corp_code = str(matches["corp_code"].iloc[0]) if not matches.empty else ""
                if not corp_code:
                    print(f"[kr-dart financials] {ticker} ({index}/{total}) SKIP no corp_code", flush=True)
                    continue
                filing_frame = filings_all.loc[filings_all["ticker"].astype(str) == ticker].copy()
                existing_raw = db_reader_kr.load_dart_financials_raw_for_ticker(ticker)
                request_plan = _planned_financial_requests(
                    filing_frame,
                    start_year=int(opts.start_year),
                    end_year=int(opts.end_year),
                    listed_date=getattr(item, "listed_date", None),
                    existing_raw=existing_raw,
                    fs_div=opts.fs_div,
                )
                if not request_plan:
                    cached_rows = 0 if existing_raw is None or existing_raw.empty else len(existing_raw)
                    print(
                        f"[kr-dart financials] {ticker} ({index}/{total}) cached_rows={cached_rows} fetch_slots=0",
                        flush=True,
                    )
                    continue
                parts: list[pd.DataFrame] = []
                rate_limit_exc: DartRateLimitError | None = None
                try:
                    for request in request_plan:
                        frame = fetch_single_account_financials(
                            corp_code=corp_code,
                            ticker=ticker,
                            bsns_year=int(request["year"]),
                            reprt_code=str(request["report_code"]),
                            fs_div=opts.fs_div,
                            receipt_no=request.get("receipt_no"),
                            filing_date=request.get("filing_date"),
                            period_end=request.get("period_end"),
                        )
                        if not frame.empty:
                            parts.append(frame)
                except DartRateLimitError as exc:
                    rate_limit_exc = exc
                parts = [
                    part.dropna(axis=1, how="all")
                    for part in parts
                    if part is not None and not part.empty and not part.dropna(how="all").empty
                ]
                if parts:
                    combine_parts: list[pd.DataFrame] = []
                    if existing_raw is not None and not existing_raw.empty:
                        combine_parts.append(existing_raw)
                    combine_parts.extend(parts)
                    raw = pd.concat(combine_parts, ignore_index=True, sort=False)
                    written = db_writer_kr.upsert_dart_financials_raw(raw, corp_code=corp_code)
                    rows_written += written
                    print(
                        f"[kr-dart financials] {ticker} ({index}/{total}) rows={written} fetch_slots={len(request_plan)}",
                        flush=True,
                    )
                else:
                    cached_rows = 0 if existing_raw is None or existing_raw.empty else len(existing_raw)
                    print(
                        f"[kr-dart financials] {ticker} ({index}/{total}) rows=0 cached_rows={cached_rows} fetch_slots={len(request_plan)}",
                        flush=True,
                    )
                if rate_limit_exc is not None:
                    print(
                        f"[STOP] kr-dart financials rate_limit ticker={ticker} progress={index}/{total} reason={rate_limit_exc}",
                        flush=True,
                    )
                    return 75
            print(f"[DONE] kr-dart financials rows={rows_written}")
            return 0

        if opts.command == "materialize":
            if opts.tickers:
                normalized = {normalize_kr_ticker(value) for value in opts.tickers}
                raw = db_reader_kr.load_dart_financials_raw_for_tickers(sorted(normalized))
                filings = db_reader_kr.load_filings_for_tickers(sorted(normalized))
            else:
                raw = db_reader_kr.load_dart_financials_raw_all()
                filings = db_reader_kr.load_filings_all()
            frame = materialize_financials_quarterly(raw, filings=filings, ticker_master=ticker_master)
            rows_written = 0
            for ticker, chunk in frame.groupby("ticker", sort=False):
                rows_written += db_writer_kr.upsert_financials(chunk.reset_index(drop=True), str(ticker), "kr")
            print(f"[DONE] kr-dart materialize rows={rows_written}")
            return 0

        def _write_rd_to_financials(ticker: str, market: str, rd_map: dict, writer) -> None:
            """Write R&D values (already in won) from XML notes to financials_quarterly."""
            con = writer._get_con()
            for pe_str, rd_won in rd_map.items():
                pe = pd.Timestamp(pe_str)
                year = pe.year
                rd_q = rd_won / 4.0
                con.execute(f"""UPDATE financials_quarterly SET "R&D"={rd_won}
                    WHERE ticker='{ticker}' AND market='{market}' AND "PeriodEnd"='{pe_str}'""")
                for q_pe in [f'{year}-03-31', f'{year}-06-30', f'{year}-09-30']:
                    con.execute(f"""UPDATE financials_quarterly SET "R&D"={rd_q}
                        WHERE ticker='{ticker}' AND market='{market}' AND "PeriodEnd"='{q_pe}' AND "R&D" IS NULL""")

        if opts.command == "segments":
            print("[kr-dart segments] Segment feature is retired. Skipping.", flush=True)
            return
            mode = opts.mode
            dart_client = DartClient()
            filings_all = db_reader_kr.load_filings_all()
            total = len(ticker_master)

            # ── Segment extraction ──
            if mode in ("segments", "all"):
                rows_written = 0
                from market_data.kr_dart import segment_store
                done_tickers = segment_store.load_segment_done_tickers("kr")
                skipped = 0
                print(f"[kr-dart segments] universe={total} already_done={len(done_tickers)}", flush=True)
                for index, item in enumerate(ticker_master.itertuples(index=False), start=1):
                    ticker = str(getattr(item, "ticker"))

                    # Resume: skip already-processed tickers
                    if ticker in done_tickers:
                        skipped += 1
                        continue

                    filing_frame = filings_all.loc[filings_all["ticker"].astype(str) == ticker].copy()
                    if filing_frame.empty:
                        segment_store.mark_segment_done(ticker, "kr", 0)
                        print(f"[kr-dart segments] {ticker} ({index}/{total}) SKIP no_filings", flush=True)
                        continue

                    # Filter to target report codes within year range
                    filing_frame["period_end_ts"] = pd.to_datetime(filing_frame["period_end"], errors="coerce")
                    filing_frame = filing_frame.loc[
                        filing_frame["period_end_ts"].dt.year.between(opts.start_year, opts.end_year)
                        & filing_frame["report_code"].astype(str).isin(REPORT_CODES)
                    ].copy()
                    if filing_frame.empty:
                        segment_store.mark_segment_done(ticker, "kr", 0)
                        print(f"[kr-dart segments] {ticker} ({index}/{total}) SKIP no_filings_in_range", flush=True)
                        continue

                    # De-duplicate: one filing per (year, report_code), latest filing wins
                    filing_frame["_year"] = filing_frame["period_end_ts"].dt.year
                    filing_frame = (
                        filing_frame.sort_values("filing_date")
                        .drop_duplicates(subset=["_year", "report_code"], keep="last")
                    )

                    all_facts: list[pd.DataFrame] = []
                    all_customer_facts: list[pd.DataFrame] = []
                    all_rd: dict[str, float] = {}  # period_end → rd_expense (won)
                    rate_limit_exc: DartRateLimitError | None = None

                    def _fetch_one_filing(row):
                        """Fetch segments for a single filing (thread-safe: no DB writes)."""
                        receipt_no = str(getattr(row, "accession", "") or getattr(row, "receipt_no", "") or "").strip()
                        report_code = str(getattr(row, "report_code") or "").strip()
                        if not receipt_no or not report_code:
                            return None
                        facts, extras = fetch_dart_segments(
                            ticker=ticker,
                            market="kr",
                            receipt_no=receipt_no,
                            reprt_code=report_code,
                            filing_date=getattr(row, "filing_date", None),
                            period_end=getattr(row, "period_end", None),
                            available_date=getattr(row, "available_date", None),
                            client=dart_client,
                        )
                        return (row, facts, extras)

                    try:
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        filing_rows = list(filing_frame.itertuples(index=False))
                        n_workers = min(opts.workers, len(filing_rows)) if hasattr(opts, "workers") and opts.workers > 1 else 1
                        if n_workers > 1:
                            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                                futures = {pool.submit(_fetch_one_filing, row): row for row in filing_rows}
                                for future in as_completed(futures):
                                    result = future.result()
                                    if result is None:
                                        continue
                                    row, facts, extras = result
                                    if not facts.empty:
                                        all_facts.append(facts)
                                    customer_facts = extras.get("customer_facts")
                                    if isinstance(customer_facts, pd.DataFrame) and not customer_facts.empty:
                                        all_customer_facts.append(customer_facts)
                                    rd = extras.get("rd_expense_won")
                                    pe = str(getattr(row, "period_end", ""))
                                    if rd is not None and pe:
                                        all_rd[pe] = rd
                        else:
                            for row in filing_rows:
                                result = _fetch_one_filing(row)
                                if result is None:
                                    continue
                                row, facts, extras = result
                                if not facts.empty:
                                    all_facts.append(facts)
                                customer_facts = extras.get("customer_facts")
                                if isinstance(customer_facts, pd.DataFrame) and not customer_facts.empty:
                                    all_customer_facts.append(customer_facts)
                                rd = extras.get("rd_expense_won")
                                pe = str(getattr(row, "period_end", ""))
                                if rd is not None and pe:
                                    all_rd[pe] = rd
                    except DartRateLimitError as exc:
                        rate_limit_exc = exc

                    if rate_limit_exc is not None:
                        # Don't mark done — will retry with next API key
                        print(
                            f"[STOP] kr-dart segments rate_limit ticker={ticker} progress={index}/{total} done={len(done_tickers)+index-skipped-1}",
                            flush=True,
                        )
                        return 75

                    if all_facts:
                        combined_facts = pd.concat(all_facts, ignore_index=True)
                        combined_facts = combined_facts.drop_duplicates(
                            subset=["ticker", "market", "period_end", "segment_type", "segment_name", "metric"],
                            keep="last",
                        )
                        written_facts = segment_store.save_segment_facts(combined_facts, ticker, "kr")
                        written_customer_facts = 0
                        if all_customer_facts:
                            combined_customer_facts = pd.concat(all_customer_facts, ignore_index=True)
                            combined_customer_facts = combined_customer_facts.drop_duplicates(
                                subset=["ticker", "market", "period_end", "segment_type", "segment_name", "metric"],
                                keep="last",
                            )
                            written_customer_facts = segment_store.save_customer_segment_facts(
                                combined_customer_facts, ticker, "kr"
                            )
                        segment_store.mark_segment_done(ticker, "kr", written_facts)
                        rows_written += written_facts
                        print(
                            f"[kr-dart segments] {ticker} ({index}/{total}) "
                            f"facts={written_facts} customer={written_customer_facts} rd={len(all_rd)}",
                            flush=True,
                        )
                    else:
                        segment_store.mark_segment_done(ticker, "kr", 0)
                        print(
                            f"[kr-dart segments] {ticker} ({index}/{total}) no_segments filings={len(filing_frame)} rd={len(all_rd)}",
                            flush=True,
                        )

                    # Write R&D values to Parquet (replaces DuckDB _write_rd_to_financials)
                    if all_rd:
                        segment_store.save_rd_expense(all_rd, ticker, "kr")

                print(f"[DONE] kr-dart segments rows={rows_written}")

            # ── Capacity extraction ──
            if mode in ("capacity", "all"):
                cap_rows_written = 0
                done_cap = segment_store.load_capacity_done_tickers("kr")
                cap_skipped = 0
                print(f"[kr-dart capacity] universe={total} already_done={len(done_cap)}", flush=True)
                for index, item in enumerate(ticker_master.itertuples(index=False), start=1):
                    ticker = str(getattr(item, "ticker"))

                    if ticker in done_cap:
                        cap_skipped += 1
                        continue

                    filing_frame = filings_all.loc[filings_all["ticker"].astype(str) == ticker].copy()
                    if filing_frame.empty:
                        segment_store.mark_capacity_done(ticker, "kr", 0)
                        print(f"[kr-dart capacity] {ticker} ({index}/{total}) SKIP no_filings", flush=True)
                        continue

                    filing_frame["period_end_ts"] = pd.to_datetime(filing_frame["period_end"], errors="coerce")
                    filing_frame = filing_frame.loc[
                        filing_frame["period_end_ts"].dt.year.between(opts.start_year, opts.end_year)
                        & filing_frame["report_code"].astype(str).isin(REPORT_CODES)
                    ].copy()
                    if filing_frame.empty:
                        segment_store.mark_capacity_done(ticker, "kr", 0)
                        print(f"[kr-dart capacity] {ticker} ({index}/{total}) SKIP no_filings_in_range", flush=True)
                        continue

                    filing_frame["_year"] = filing_frame["period_end_ts"].dt.year
                    filing_frame = (
                        filing_frame.sort_values("filing_date")
                        .drop_duplicates(subset=["_year", "report_code"], keep="last")
                    )

                    all_cap: list[pd.DataFrame] = []
                    rate_limit_exc_cap: DartRateLimitError | None = None
                    try:
                        for row in filing_frame.itertuples(index=False):
                            receipt_no = str(getattr(row, "accession", "") or getattr(row, "receipt_no", "") or "").strip()
                            report_code = str(getattr(row, "report_code") or "").strip()
                            if not receipt_no or not report_code:
                                continue
                            cap_df = fetch_dart_capacity_data(
                                ticker=ticker,
                                market="kr",
                                receipt_no=receipt_no,
                                reprt_code=report_code,
                                filing_date=getattr(row, "filing_date", None),
                                period_end=getattr(row, "period_end", None),
                                available_date=getattr(row, "available_date", None),
                            )
                            if not cap_df.empty:
                                all_cap.append(cap_df)
                    except DartRateLimitError as exc:
                        rate_limit_exc_cap = exc

                    if rate_limit_exc_cap is not None:
                        print(
                            f"[STOP] kr-dart capacity rate_limit ticker={ticker} progress={index}/{total}",
                            flush=True,
                        )
                        return 75

                    if all_cap:
                        combined_cap = pd.concat(all_cap, ignore_index=True)
                        combined_cap = combined_cap.drop_duplicates(
                            subset=["ticker", "market", "period_end", "section", "product_name"],
                            keep="last",
                        )
                        # Deduplicate near-identical product names that differ
                        # only in whitespace (e.g., "도 료" vs "도료").
                        import re as _re
                        _dedup_drop = []
                        for (period, section), grp in combined_cap.groupby(["period_end", "section"]):
                            if len(grp) < 2:
                                continue
                            rows = list(grp.itertuples())
                            for i in range(len(rows)):
                                for j in range(i + 1, len(rows)):
                                    ri, rj = rows[i], rows[j]
                                    ni = _re.sub(r"\s+", "", ri.product_name)
                                    nj = _re.sub(r"\s+", "", rj.product_name)
                                    if ni == nj:
                                        if len(ri.product_name) <= len(rj.product_name):
                                            _dedup_drop.append(rj.Index)
                                        else:
                                            _dedup_drop.append(ri.Index)
                        if _dedup_drop:
                            combined_cap = combined_cap.drop(
                                index=list(set(_dedup_drop))
                            )
                        # Deduplicate "등" products: if two product names
                        # ending with "등" share the same value for the same
                        # (period, section) and one is a subset of the other
                        # (ignoring "등"), keep the shorter one.
                        # e.g., "VS 텔레매틱스, AV, AVN 등" vs
                        #       "VS 텔레매틱스, AV, AVN, 전기차부품 등"
                        _etc_drop = []
                        _etc_prods = combined_cap[
                            combined_cap["product_name"].str.contains(r"등\s*$", regex=True)
                        ]
                        if not _etc_prods.empty:
                            for (period, section), grp in _etc_prods.groupby(
                                ["period_end", "section"]
                            ):
                                if len(grp) < 2:
                                    continue
                                rows = list(grp.itertuples())
                                for i in range(len(rows)):
                                    for j in range(i + 1, len(rows)):
                                        ri, rj = rows[i], rows[j]
                                        # Same value check
                                        if ri.value != rj.value:
                                            continue
                                        # Strip "등" and whitespace for comparison
                                        ni = _re.sub(r"등\s*$", "", _re.sub(r"\s+", "", ri.product_name))
                                        nj = _re.sub(r"등\s*$", "", _re.sub(r"\s+", "", rj.product_name))
                                        if ni in nj and ni != nj:
                                            _etc_drop.append(rj.Index)  # drop longer
                                        elif nj in ni and ni != nj:
                                            _etc_drop.append(ri.Index)  # drop longer
                            if _etc_drop:
                                combined_cap = combined_cap.drop(
                                    index=list(set(_etc_drop))
                                )
                        # Smart merge: when both bare "X" and location-prefixed "국내 X"
                        # exist for the same (period, section), drop the bare entry.
                        import re as _re
                        _LOC_PFX = ('국내', '해외', '베트남', '한국', '미국', '중국',
                                    '일본', '인도', '인도네시아', '멕시코', '독일', '터키')
                        _loc_prods = [p for p in combined_cap['product_name'].unique()
                                      if any(p.startswith(pf + ' ') for pf in _LOC_PFX)]
                        _drop_idx = []
                        for _lp in _loc_prods:
                            _bare = _re.sub(r'^(' + '|'.join(_LOC_PFX) + r')\s+', '', _lp)
                            if _bare not in combined_cap['product_name'].values:
                                continue
                            _lk = set(zip(
                                combined_cap[combined_cap['product_name']==_lp]['period_end'],
                                combined_cap[combined_cap['product_name']==_lp]['section']))
                            for idx, row in combined_cap[combined_cap['product_name']==_bare].iterrows():
                                if (row['period_end'], row['section']) in _lk:
                                    _drop_idx.append(idx)
                        if _drop_idx:
                            combined_cap = combined_cap.drop(_drop_idx)
                        # Deaccumulate YTD cumulative values
                        from market_data.kr_dart.segments_xml import deaccumulate_ytd_capacity
                        combined_cap = deaccumulate_ytd_capacity(combined_cap)
                        written_cap = segment_store.save_capacity(combined_cap, ticker, "kr")
                        segment_store.mark_capacity_done(ticker, "kr", written_cap)
                        cap_rows_written += written_cap
                        print(
                            f"[kr-dart capacity] {ticker} ({index}/{total}) rows={written_cap}",
                            flush=True,
                        )
                    else:
                        segment_store.mark_capacity_done(ticker, "kr", 0)
                        print(
                            f"[kr-dart capacity] {ticker} ({index}/{total}) no_capacity filings={len(filing_frame)}",
                            flush=True,
                        )

                print(f"[DONE] kr-dart capacity rows={cap_rows_written}")

            return 0

        print(f"[ERROR] Unsupported kr-dart command: {opts.command}")
        return 2
    finally:
        db_writer_kr.close()
