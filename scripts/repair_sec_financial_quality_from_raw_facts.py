#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from market_data import db_writer, sec_financials
from market_data.db import DB_PATH
from market_data.db_reader import load_price_from_db
from market_data.derived_factors import build_derived_factors_quarterly
from market_data.utils import now_utc_iso


TARGET_COLUMNS = [
    "COGS",
    "Gross Profit",
    "SG&A",
    "Operating Income",
    "Shareholders Equity",
]

NUMERIC_TOL = 1e-9


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repair SEC quarterly financial quality from existing sec_facts_raw_normalized rows."
    )
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--market", default="us")
    p.add_argument("--start", default="2013-06-01")
    p.add_argument("--exchanges", default="NASDAQ,NYSE")
    p.add_argument("--tickers", default=None, help="Comma-separated explicit ticker subset")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--write", action="store_true")
    p.add_argument("--skip-derived", action="store_true")
    p.add_argument("--backup-parquet", default=None)
    return p.parse_args()


def _ticker_cache_path(ticker: str) -> Path:
    return sec_financials.SEC_TICKER_QUARTERLY_DIR / f"{str(ticker).strip().upper()}.parquet"


def _existing_period_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns or frame.empty:
        return pd.Series(dtype=float)
    out = frame.loc[:, ["PeriodEnd", column]].copy()
    out["PeriodEnd"] = pd.to_datetime(out["PeriodEnd"], errors="coerce").dt.normalize()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["PeriodEnd"])
    if out.empty:
        return pd.Series(dtype=float)
    out = out.sort_values("PeriodEnd")
    out = out.drop_duplicates(subset=["PeriodEnd"], keep="last")
    return pd.Series(out[column].to_numpy(dtype=float), index=pd.DatetimeIndex(out["PeriodEnd"]))


def _safe_replace(
    current: pd.Series,
    candidate: pd.Series,
    *,
    rel_tol: float,
    fill_missing: bool = True,
    positive_only: bool = False,
    update_close: bool = False,
    allow_sign_conflict: bool = True,
) -> pd.Series:
    union_index = current.index.union(candidate.index)
    cur = pd.to_numeric(current.reindex(union_index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    cand = pd.to_numeric(candidate.reindex(union_index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = cur.copy()
    mag_close = (cur.abs() - cand.abs()).abs() <= np.maximum(
        np.maximum(cur.abs(), cand.abs()) * float(rel_tol),
        1.0,
    )
    replace_mask = pd.Series(False, index=union_index, dtype=bool)
    if fill_missing:
        replace_mask |= cur.isna() & cand.notna()
    if positive_only:
        replace_mask |= cur.notna() & cand.notna() & (cur < 0) & (cand >= 0)
    if allow_sign_conflict:
        replace_mask |= cur.notna() & cand.notna() & (cur * cand < 0) & mag_close
    if update_close:
        replace_mask |= cur.notna() & cand.notna() & mag_close
    out = out.where(~replace_mask, cand)
    return out.dropna().sort_index()


def _build_raw_records(raw_facts: pd.DataFrame, metric_name: str) -> list[dict[str, Any]]:
    if raw_facts.empty:
        return []
    rows = raw_facts.loc[raw_facts["fact_name"] == metric_name].copy()
    if rows.empty:
        return []
    rows["period_start"] = pd.to_datetime(rows["period_start"], errors="coerce")
    rows["period_end"] = pd.to_datetime(rows["period_end"], errors="coerce")
    rows["instant_date"] = pd.to_datetime(rows["instant_date"], errors="coerce")
    rows["filing_date"] = pd.to_datetime(rows["filing_date"], errors="coerce")
    rows["accepted_at"] = pd.to_datetime(rows["accepted_at"], errors="coerce", utc=True)
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows = rows.dropna(subset=["period_end", "value"]).sort_values(
        ["period_end", "filing_date", "accepted_at", "accession", "context_id"],
        ascending=[True, True, True, True, True],
    )
    out: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        rec: dict[str, Any] = {
            "end": pd.Timestamp(row["period_end"]).date().isoformat(),
            "val": float(row["value"]),
            "form": str(row.get("form_type") or ""),
            "filed": pd.Timestamp(row["filing_date"]).date().isoformat()
            if pd.notna(row.get("filing_date"))
            else "",
            "accn": str(row.get("accession") or ""),
            "frame": str(row.get("context_id") or ""),
        }
        if pd.notna(row.get("period_start")):
            rec["start"] = pd.Timestamp(row["period_start"]).date().isoformat()
        if pd.notna(row.get("instant_date")):
            rec["instant"] = pd.Timestamp(row["instant_date"]).date().isoformat()
        out.append(rec)
    return out


def _metric_series_from_raw_facts(
    raw_facts: pd.DataFrame,
    metric_name: str,
    min_date: pd.Timestamp,
) -> pd.Series:
    if metric_name not in sec_financials.METRIC_SPECS:
        return pd.Series(dtype=float)
    records = _build_raw_records(raw_facts, metric_name)
    if not records:
        return pd.Series(dtype=float)
    frame = sec_financials._fact_records_to_frame(records)
    if frame.empty:
        return pd.Series(dtype=float)
    frame = frame.loc[frame["quarter_end"] >= pd.Timestamp(min_date)].copy()
    if frame.empty:
        return pd.Series(dtype=float)
    spec = sec_financials.METRIC_SPECS[metric_name]
    if spec.is_flow:
        quarter = sec_financials._pick_flow_quarter_values(frame)
        annual = sec_financials._pick_flow_annual_values(frame)
        out = sec_financials._fill_q4_from_annual(quarter, annual).sort_index()
    else:
        out = sec_financials._pick_stock_quarter_values(frame).sort_index()
    return pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _load_target_tickers(
    con: duckdb.DuckDBPyConnection,
    *,
    market: str,
    min_date: pd.Timestamp,
    exchanges: list[str],
    explicit_tickers: list[str] | None,
    limit: int | None,
) -> list[str]:
    if explicit_tickers:
        return explicit_tickers[: limit or None]

    exchange_list = ", ".join(f"'{x}'" for x in exchanges)
    limit_sql = f" LIMIT {int(limit)}" if limit and limit > 0 else ""
    sql = f"""
        SELECT DISTINCT q.ticker
        FROM financials_quarterly q
        INNER JOIN sec_issuer_registry i
            ON q.ticker = i.ticker
           AND q.market = i.market
        WHERE q.market = ?
          AND i.is_common_stock
          AND upper(coalesce(i.exchange, '')) IN ({exchange_list})
          AND q."PeriodEnd" >= ?
        ORDER BY q.ticker
        {limit_sql}
    """
    rows = con.execute(sql, [market, pd.Timestamp(min_date).date()]).fetchall()
    return [str(r[0]).strip().upper() for r in rows if str(r[0]).strip()]


def _load_ticker_financials(con: duckdb.DuckDBPyConnection, ticker: str, market: str) -> pd.DataFrame:
    cache_path = _ticker_cache_path(ticker)
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if not cached.empty:
                return cached.copy()
        except Exception:
            pass
    return con.execute(
        """
        SELECT *
        FROM financials_quarterly
        WHERE ticker = ? AND market = ?
        ORDER BY "PeriodEnd", "AvailableDate", "FilingDate", "AcceptedAt"
        """,
        [ticker, market],
    ).fetchdf()


def _load_ticker_raw_facts(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    market: str,
    min_date: pd.Timestamp,
) -> pd.DataFrame:
    return con.execute(
        """
        SELECT *
        FROM sec_facts_raw_normalized
        WHERE ticker = ?
          AND market = ?
          AND coalesce(period_end, instant_date) >= ?
        ORDER BY period_end, filing_date, accepted_at, accession, context_id
        """,
        [ticker, market, pd.Timestamp(min_date).date()],
    ).fetchdf()


def _load_ticker_filings(con: duckdb.DuckDBPyConnection, ticker: str, market: str) -> pd.DataFrame:
    return con.execute(
        """
        SELECT *
        FROM filings
        WHERE ticker = ? AND market = ?
        ORDER BY period_end, available_date, filing_date, accepted_at
        """,
        [ticker, market],
    ).fetchdf()


def _load_ticker_company_name(con: duckdb.DuckDBPyConnection, ticker: str, market: str) -> str | None:
    row = con.execute(
        """
        SELECT company_name
        FROM sec_issuer_registry
        WHERE ticker = ? AND market = ?
        LIMIT 1
        """,
        [ticker, market],
    ).fetchone()
    if not row:
        return None
    name = str(row[0] or "").strip()
    return name or None


def _apply_series_update(
    out: pd.DataFrame,
    *,
    min_date: pd.Timestamp,
    column: str,
    values: pd.Series,
) -> int:
    if out.empty or values is None or values.empty:
        return 0
    period_end = pd.to_datetime(out["PeriodEnd"], errors="coerce").dt.normalize()
    aligned = pd.Series(values).reindex(pd.DatetimeIndex(period_end))
    aligned = pd.Series(pd.to_numeric(aligned.to_numpy(), errors="coerce"), index=out.index, dtype=float)
    current = pd.to_numeric(out.get(column), errors="coerce")
    period_mask = pd.Series(period_end.ge(pd.Timestamp(min_date)).fillna(False).to_numpy(), index=out.index)
    update_mask = (
        period_mask
        & aligned.notna()
        & (
            current.isna()
            | ((current - pd.to_numeric(aligned, errors="coerce")).abs() > NUMERIC_TOL)
        )
    )
    if not update_mask.any():
        return 0
    out.loc[update_mask, column] = pd.to_numeric(aligned[update_mask], errors="coerce").to_numpy()
    return int(update_mask.sum())


def _repair_financial_frame(
    frame: pd.DataFrame,
    raw_facts: pd.DataFrame,
    *,
    ticker: str,
    company_name: str | None,
    min_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = frame.copy()
    if out.empty:
        return out, {}

    current_revenue = _existing_period_series(out, "Revenue")
    current_cogs = _existing_period_series(out, "COGS")
    current_gross = _existing_period_series(out, "Gross Profit")
    current_sga = _existing_period_series(out, "SG&A")
    current_operating_income = _existing_period_series(out, "Operating Income")
    current_equity = _existing_period_series(out, "Shareholders Equity")
    current_assets = _existing_period_series(out, "Total Assets")
    current_liabilities = _existing_period_series(out, "Total Liabilities")

    revenue = _metric_series_from_raw_facts(raw_facts, "Revenue", min_date).combine_first(current_revenue)
    raw_cogs = sec_financials._normalize_expense_series(
        _metric_series_from_raw_facts(raw_facts, "COGS", min_date)
    )
    cogs = _safe_replace(
        current_cogs,
        raw_cogs,
        rel_tol=0.20,
        positive_only=True,
        update_close=True,
        allow_sign_conflict=False,
    )
    raw_gross = _metric_series_from_raw_facts(raw_facts, "Gross Profit", min_date)
    gross_candidate = sec_financials._reconcile_signed_reconstruction(raw_gross, revenue - cogs)
    gross = _safe_replace(
        current_gross,
        gross_candidate,
        rel_tol=0.15,
        positive_only=False,
        update_close=False,
        allow_sign_conflict=True,
    )

    raw_sga = sec_financials._normalize_expense_series(
        _metric_series_from_raw_facts(raw_facts, "SG&A", min_date)
    )
    sga = _safe_replace(
        current_sga,
        raw_sga,
        rel_tol=0.15,
        positive_only=True,
        update_close=False,
        allow_sign_conflict=False,
    )
    raw_operating_income = _metric_series_from_raw_facts(raw_facts, "Operating Income", min_date)
    operating_income = _safe_replace(
        current_operating_income,
        raw_operating_income,
        rel_tol=0.15,
        positive_only=False,
        update_close=False,
        allow_sign_conflict=True,
    )

    raw_assets = _metric_series_from_raw_facts(raw_facts, "Total Assets", min_date)
    raw_liabilities = _metric_series_from_raw_facts(raw_facts, "Total Liabilities", min_date)
    raw_equity = _metric_series_from_raw_facts(raw_facts, "Shareholders Equity", min_date).combine_first(current_equity)
    assets = raw_assets.combine_first(current_assets)
    liabilities = raw_liabilities.combine_first(current_liabilities)
    balance_index = assets.index.union(liabilities.index).union(raw_equity.index)
    assets = assets.reindex(balance_index)
    liabilities = liabilities.reindex(balance_index)
    raw_equity = raw_equity.reindex(balance_index)
    _, _, equity = sec_financials._enforce_balance_identity(
        assets=assets,
        liabilities=liabilities,
        equity=raw_equity,
    )
    equity = _safe_replace(
        current_equity,
        equity,
        rel_tol=0.05,
        positive_only=False,
        update_close=True,
        allow_sign_conflict=True,
    )

    change_counts: dict[str, int] = {}
    change_counts["COGS"] = _apply_series_update(out, min_date=min_date, column="COGS", values=cogs)
    change_counts["Gross Profit"] = _apply_series_update(
        out, min_date=min_date, column="Gross Profit", values=gross
    )
    change_counts["SG&A"] = _apply_series_update(out, min_date=min_date, column="SG&A", values=sga)
    change_counts["Operating Income"] = _apply_series_update(
        out, min_date=min_date, column="Operating Income", values=operating_income
    )
    change_counts["Shareholders Equity"] = _apply_series_update(
        out, min_date=min_date, column="Shareholders Equity", values=equity
    )

    if any(change_counts.values()):
        if company_name:
            out["name"] = company_name
        if "Source" in out.columns:
            out["Source"] = out["Source"].fillna("sec")
        if "collected_at" in out.columns:
            out["collected_at"] = now_utc_iso()

    return out, change_counts


def _build_derived_frame(
    *,
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    market: str,
    financials: pd.DataFrame,
) -> pd.DataFrame:
    price_payload = load_price_from_db(ticker=ticker, market=market)
    prices = price_payload[0] if price_payload is not None else None
    filings = _load_ticker_filings(con, ticker=ticker, market=market)
    return build_derived_factors_quarterly(
        financials=financials,
        prices=prices,
        filings=filings,
        source="materialized_sec_raw_repair",
    )


def _write_backup(
    con: duckdb.DuckDBPyConnection,
    *,
    market: str,
    min_date: pd.Timestamp,
    exchanges: list[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exchange_list = ", ".join(f"'{x}'" for x in exchanges)
    sql = f"""
        COPY (
            SELECT q.*
            FROM financials_quarterly q
            INNER JOIN sec_issuer_registry i
                ON q.ticker = i.ticker
               AND q.market = i.market
            WHERE q.market = '{market}'
              AND i.is_common_stock
              AND upper(coalesce(i.exchange, '')) IN ({exchange_list})
              AND q."PeriodEnd" >= DATE '{pd.Timestamp(min_date).date()}'
        ) TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(sql)


def main() -> None:
    args = _parse_args()
    market = str(args.market).strip().lower()
    min_date = pd.Timestamp(pd.to_datetime(args.start, errors="coerce")).normalize()
    exchanges = [x.strip().upper() for x in str(args.exchanges).split(",") if x.strip()]
    explicit_tickers = (
        [x.strip().upper() for x in str(args.tickers).split(",") if x.strip()]
        if args.tickers
        else None
    )

    if args.write:
        db_writer.init_schema()
    read_con = duckdb.connect(str(args.db))

    if args.backup_parquet:
        backup_path = Path(args.backup_parquet)
        print(f"[BACKUP] writing {backup_path}", flush=True)
        _write_backup(
            read_con,
            market=market,
            min_date=min_date,
            exchanges=exchanges,
            path=backup_path,
        )

    tickers = _load_target_tickers(
        read_con,
        market=market,
        min_date=min_date,
        exchanges=exchanges,
        explicit_tickers=explicit_tickers,
        limit=args.limit,
    )
    print(
        f"[INFO] target_tickers={len(tickers)} market={market} start={min_date.date()} exchanges={','.join(exchanges)}",
        flush=True,
    )

    changed_tickers = 0
    changed_rows = 0
    metric_change_totals = {col: 0 for col in TARGET_COLUMNS}
    skipped_no_raw = 0
    skipped_no_change = 0

    for idx, ticker in enumerate(tickers, start=1):
        base = _load_ticker_financials(read_con, ticker=ticker, market=market)
        if base.empty:
            print(f"[SKIP] {ticker} no financial rows ({idx}/{len(tickers)})", flush=True)
            continue
        raw = _load_ticker_raw_facts(read_con, ticker=ticker, market=market, min_date=min_date)
        if raw.empty:
            skipped_no_raw += 1
            print(f"[SKIP] {ticker} no raw facts ({idx}/{len(tickers)})", flush=True)
            continue
        company_name = _load_ticker_company_name(read_con, ticker=ticker, market=market)
        repaired, counts = _repair_financial_frame(
            base,
            raw,
            ticker=ticker,
            company_name=company_name,
            min_date=min_date,
        )
        total_changes = sum(int(v) for v in counts.values())
        if total_changes <= 0:
            skipped_no_change += 1
            print(f"[SKIP] {ticker} no material change ({idx}/{len(tickers)})", flush=True)
            continue

        changed_tickers += 1
        changed_rows += total_changes
        for col, val in counts.items():
            metric_change_totals[col] += int(val)

        if args.write:
            fin_rows = db_writer.upsert_financials(repaired, ticker, market)
            derived_rows = 0
            if not args.skip_derived:
                derived = _build_derived_frame(
                    con=read_con,
                    ticker=ticker,
                    market=market,
                    financials=repaired,
                )
                derived_rows = db_writer.upsert_derived_factors(derived, ticker, market)
            print(
                f"[WRITE] {ticker} financials={fin_rows} derived={derived_rows} "
                f"changes={total_changes} ({idx}/{len(tickers)})",
                flush=True,
            )
        else:
            print(f"[DRYRUN] {ticker} changes={total_changes} ({idx}/{len(tickers)})", flush=True)

    print(
        "[DONE] "
        f"changed_tickers={changed_tickers} "
        f"changed_cells={changed_rows} "
        f"skipped_no_raw={skipped_no_raw} "
        f"skipped_no_change={skipped_no_change}",
        flush=True,
    )
    print("[DONE] metric_changes=" + ", ".join(f"{k}={v}" for k, v in metric_change_totals.items()), flush=True)


if __name__ == "__main__":
    main()
