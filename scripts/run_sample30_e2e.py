#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market_data.duckdb"
DATA_REPORT_DIR = ROOT / "data" / "test_reports"
DOC_REPORT_DIR = ROOT / "docs" / "test_reports"
SAMPLE_TICKERS_PATH = DATA_REPORT_DIR / "sample30_tickers.txt"
RESULTS_JSON_PATH = DATA_REPORT_DIR / "sample30_results.json"
REPORT_MD_PATH = DOC_REPORT_DIR / "sample30_report.md"
START_DATE_FIXED = "2000-01-01"
DEFAULT_SEED = 20260306
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_ENDPOINT_TIMEOUT_SEC = 40
DEFAULT_INGEST_TIMEOUT_SEC = 1200
DEFAULT_SLEEP_SEC = 0.2
DEFAULT_API_BASE = "http://127.0.0.1:8000"


@dataclass
class ApiServerHandle:
    base_url: str
    mode: str  # existing | spawned | unavailable
    process: subprocess.Popen[str] | None = None
    error: str | None = None


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _pick_python_bin() -> str:
    cand = ROOT / ".venv" / "bin" / "python"
    if cand.exists():
        return str(cand)
    return sys.executable


def _safe_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _find_ticker_column(df: pd.DataFrame) -> str | None:
    candidates = ["Symbol", "symbol", "Ticker", "ticker", "Code", "code"]
    for col in candidates:
        if col in df.columns:
            return col
    if len(df.columns) > 0:
        return str(df.columns[0])
    return None


def _find_sector_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if "sector" in str(col).strip().lower():
            return str(col)
    return None


def _load_candidates_from_csv(path: Path) -> list[dict[str, str | None]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    ticker_col = _find_ticker_column(df)
    if ticker_col is None:
        return []
    sector_col = _find_sector_column(df)
    out: list[dict[str, str | None]] = []
    for _, row in df.iterrows():
        tkr = _safe_ticker(row.get(ticker_col))
        if not tkr:
            continue
        sector = None
        if sector_col is not None:
            val = row.get(sector_col)
            if pd.notna(val):
                sector = str(val).strip() or None
        out.append({"ticker": tkr, "sector": sector})
    return out


def _load_candidates_from_db() -> list[dict[str, str | None]]:
    if not DB_PATH.exists():
        return []
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
    except Exception:
        return []
    try:
        rows = con.execute(
            """
            SELECT DISTINCT upper(trim(ticker)) AS ticker
            FROM prices
            WHERE ticker IS NOT NULL AND trim(ticker) <> '' AND lower(coalesce(market, 'us')) = 'us'
            ORDER BY 1
            """
        ).fetchall()
    except Exception:
        rows = []
    finally:
        con.close()
    return [{"ticker": _safe_ticker(r[0]), "sector": None} for r in rows if _safe_ticker(r[0])]


def _load_universe_candidates(sample_size: int) -> tuple[list[dict[str, str | None]], str]:
    sources = [
        (ROOT / "data" / "universe" / "symbols_sp500.csv", "symbols_sp500.csv"),
        (ROOT / "data" / "universe" / "symbols_nasdaq_stock_only.csv", "symbols_nasdaq_stock_only.csv"),
    ]
    candidates: list[dict[str, str | None]] = []
    source_note = ""
    for path, note in sources:
        loaded = _load_candidates_from_csv(path)
        if loaded:
            candidates = loaded
            source_note = note
            if len(candidates) >= sample_size:
                break
    if len(candidates) < sample_size:
        db_loaded = _load_candidates_from_db()
        if db_loaded:
            if candidates:
                seen = {c["ticker"] for c in candidates}
                candidates.extend([row for row in db_loaded if row["ticker"] not in seen])
                source_note = f"{source_note}+duckdb_prices"
            else:
                candidates = db_loaded
                source_note = "duckdb_prices"
    return candidates, source_note or "unknown"


def _dedupe_candidates(candidates: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    seen: set[str] = set()
    out: list[dict[str, str | None]] = []
    for row in candidates:
        tkr = _safe_ticker(row.get("ticker"))
        if not tkr or tkr in seen:
            continue
        seen.add(tkr)
        out.append({"ticker": tkr, "sector": row.get("sector")})
    return out


def _sample_tickers(
    candidates: list[dict[str, str | None]],
    sample_size: int,
    seed: int,
) -> list[dict[str, str | None]]:
    deduped = _dedupe_candidates(candidates)
    if len(deduped) < sample_size:
        raise RuntimeError(f"Not enough candidate tickers: need={sample_size}, available={len(deduped)}")

    rng = random.Random(seed)
    with_sector = [r for r in deduped if r.get("sector")]
    sector_values = {str(r.get("sector")).strip() for r in with_sector if str(r.get("sector")).strip()}
    if len(sector_values) <= 1:
        picked = rng.sample(deduped, sample_size)
        return sorted(picked, key=lambda x: x["ticker"])

    by_sector: dict[str, list[dict[str, str | None]]] = defaultdict(list)
    for row in deduped:
        sector = str(row.get("sector") or "UNKNOWN").strip() or "UNKNOWN"
        by_sector[sector].append(row)
    sector_order = list(by_sector.keys())
    rng.shuffle(sector_order)
    for key in sector_order:
        rng.shuffle(by_sector[key])

    selected: list[dict[str, str | None]] = []
    while len(selected) < sample_size:
        progressed = False
        for sector in sector_order:
            bucket = by_sector.get(sector, [])
            if not bucket:
                continue
            selected.append(bucket.pop())
            progressed = True
            if len(selected) >= sample_size:
                break
        if not progressed:
            break

    if len(selected) < sample_size:
        selected_set = {r["ticker"] for r in selected}
        remainder = [r for r in deduped if r["ticker"] not in selected_set]
        rng.shuffle(remainder)
        selected.extend(remainder[: sample_size - len(selected)])

    return sorted(selected[:sample_size], key=lambda x: x["ticker"])


def _write_sample_tickers_file(tickers: list[str]) -> None:
    DATA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_TICKERS_PATH.write_text("\n".join(tickers) + "\n", encoding="utf-8")


def _run_cmd(cmd: list[str], timeout_sec: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, cwd=str(ROOT))
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "duration_ms": duration_ms,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": False,
            "returncode": 124,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else "",
            "duration_ms": duration_ms,
            "timeout": True,
        }


def _parse_ingest_status(run_result: dict[str, Any]) -> tuple[str, str]:
    text = "\n".join([run_result.get("stdout", ""), run_result.get("stderr", "")]).strip()
    if run_result.get("ok", False):
        if "skipped=1" in text and "ok=0" in text:
            return "skip", "fresh checkpoint or no update required"
        return "success", ""
    if run_result.get("timeout"):
        return "fail", "timeout"

    reason = ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        reason = lines[-1]
    if not reason:
        reason = f"returncode={run_result.get('returncode')}"
    return "fail", reason[:500]


def _build_ticker_csv(path: Path, ticker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Symbol\n{ticker}\n", encoding="utf-8")


def _run_ingest_for_ticker(
    *,
    python_bin: str,
    ticker: str,
    sec_user_agent: str | None,
    lookback_filings: int,
    ingest_timeout_sec: int,
) -> dict[str, Any]:
    tmp_csv = DATA_REPORT_DIR / "_tmp_ingest" / f"{ticker}.csv"
    _build_ticker_csv(tmp_csv, ticker)
    cmd = [
        python_bin,
        "-m",
        "market_data",
        "ingest",
        "--universe",
        "custom",
        "--tickers-file",
        str(tmp_csv),
        "--start",
        START_DATE_FIXED,
        "--financial-source",
        "sec",
        "--enable-segmentation",
        "--segment-lookback-filings",
        str(max(1, int(lookback_filings))),
        "--backfill-financials-extra",
        "--workers",
        "1",
        "--financial-workers",
        "1",
        "--force",
        "--skip-sector-cache",
    ]
    if sec_user_agent:
        cmd.extend(["--sec-user-agent", sec_user_agent])

    run = _run_cmd(cmd, timeout_sec=ingest_timeout_sec)
    status, reason = _parse_ingest_status(run)
    return {
        "ticker": ticker,
        "status": status,
        "reason": reason,
        "returncode": run.get("returncode"),
        "timeout": bool(run.get("timeout")),
        "duration_ms": int(run.get("duration_ms", 0)),
        "stdout_tail": "\n".join((run.get("stdout", "") or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((run.get("stderr", "") or "").splitlines()[-20:]),
    }


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _find_free_port(host: str = "127.0.0.1", start_port: int = 8001, end_port: int = 8099) -> int:
    for port in range(start_port, end_port + 1):
        if not _port_in_use(host, port):
            return port
    raise RuntimeError("No free port found for temporary API server")


def _check_health(base_url: str, timeout_sec: int = 3) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/api/health"
    try:
        res = requests.get(url, timeout=timeout_sec)
        if res.status_code == 200:
            return True, ""
        return False, f"status={res.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc.__class__.__name__}: {exc}"


def _ensure_api_server(python_bin: str, preferred_base: str) -> ApiServerHandle:
    ok, err = _check_health(preferred_base)
    if ok:
        return ApiServerHandle(base_url=preferred_base.rstrip("/"), mode="existing")

    host = "127.0.0.1"
    port = _find_free_port(host=host, start_port=8001, end_port=8099)
    base = f"http://{host}:{port}"
    cmd = [
        python_bin,
        "-m",
        "uvicorn",
        "web.backend.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    deadline = time.time() + 45
    while time.time() < deadline:
        ok, _ = _check_health(base)
        if ok:
            return ApiServerHandle(base_url=base, mode="spawned", process=proc)
        time.sleep(0.5)

    try:
        proc.terminate()
    except Exception:
        pass
    return ApiServerHandle(base_url=base, mode="unavailable", process=None, error=f"health check failed (preferred={err})")


def _stop_api_server(handle: ApiServerHandle) -> None:
    if handle.mode != "spawned" or handle.process is None:
        return
    try:
        handle.process.terminate()
        handle.process.wait(timeout=8)
    except Exception:
        try:
            handle.process.kill()
        except Exception:
            pass


def _count_series_points(series_list: list[dict[str, Any]]) -> tuple[int, int]:
    total = 0
    non_null = 0
    for s in series_list:
        data = s.get("data", [])
        if not isinstance(data, list):
            continue
        for p in data:
            if not isinstance(p, dict):
                continue
            total += 1
            y = p.get("y")
            if y is None and "value" in p:
                y = p.get("value")
            if y is not None:
                non_null += 1
    return total, non_null


def _iter_nested_chart_objects(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        if "series" in obj and isinstance(obj.get("series"), list):
            yield prefix, obj
            return
        for k, v in obj.items():
            if k == "missing_reason" and not isinstance(v, dict):
                continue
            child = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_nested_chart_objects(v, child)


def _extract_chart_stats(endpoint: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []

    if endpoint == "business":
        series_map = payload.get("series", {})
        if not isinstance(series_map, dict):
            return stats
        for key, points in series_map.items():
            if not isinstance(points, list):
                continue
            total = 0
            non_null = 0
            for point in points:
                if not isinstance(point, dict):
                    continue
                total += 1
                if point.get("value") is not None:
                    non_null += 1
            stats.append(
                {
                    "chart_id": str(key),
                    "series_count": 1,
                    "total_points": total,
                    "non_null_points": non_null,
                    "missing_reason": None,
                }
            )
        return stats

    charts = payload.get("charts", {})
    if not isinstance(charts, dict):
        return stats

    for chart_id, chart in _iter_nested_chart_objects(charts):
        if not isinstance(chart, dict):
            continue
        series = chart.get("series", [])
        if not isinstance(series, list):
            series = []
        total, non_null = _count_series_points(series)
        stats.append(
            {
                "chart_id": str(chart_id),
                "series_count": len(series),
                "total_points": total,
                "non_null_points": non_null,
                "missing_reason": chart.get("missing_reason"),
            }
        )
    return stats


def _call_endpoint(base_url: str, path: str, timeout_sec: int) -> tuple[int, int, dict[str, Any] | None, str]:
    url = base_url.rstrip("/") + path
    started = time.perf_counter()
    try:
        res = requests.get(url, timeout=timeout_sec)
        latency_ms = int((time.perf_counter() - started) * 1000)
        payload: dict[str, Any] | None = None
        err = ""
        if res.status_code == 200:
            try:
                payload = res.json()
            except Exception as exc:  # noqa: BLE001
                err = f"json_parse_failed:{exc.__class__.__name__}"
        else:
            err = (res.text or "")[:500]
        return res.status_code, latency_ms, payload, err
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return 0, latency_ms, None, f"{exc.__class__.__name__}: {exc}"


def _run_api_smoke(base_url: str, tickers: list[str], endpoint_timeout_sec: int) -> list[dict[str, Any]]:
    endpoint_paths = [
        ("summary_dashboard", "/api/ticker/{ticker}/summary_dashboard?market=auto&quarters=20"),
        ("financials_income", "/api/ticker/{ticker}/financials/income?market=auto&window=5y&basis=ttm"),
        ("financials_balance", "/api/ticker/{ticker}/financials/balance?market=auto&window=5y&basis=ttm"),
        ("financials_cashflow", "/api/ticker/{ticker}/financials/cashflow?market=auto&window=5y&basis=ttm"),
        ("fundamentals", "/api/ticker/{ticker}/fundamentals?market=auto&window=5y&basis=ttm"),
        ("valuation", "/api/ticker/{ticker}/valuation?market=auto&window=5y&basis=ttm"),
        ("business", "/api/ticker/{ticker}/business?market=auto&window=5y&basis=ttm&subtab=segment"),
    ]

    out: list[dict[str, Any]] = []
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[API] {idx}/{len(tickers)} {ticker}")
        for endpoint_name, path_tmpl in endpoint_paths:
            path = path_tmpl.format(ticker=ticker)
            status, latency, payload, err = _call_endpoint(base_url, path, timeout_sec=endpoint_timeout_sec)
            charts = _extract_chart_stats(endpoint_name, payload) if isinstance(payload, dict) else []
            out.append(
                {
                    "ticker": ticker,
                    "endpoint": endpoint_name,
                    "path": path,
                    "status_code": status,
                    "latency_ms": latency,
                    "charts_found": len(charts),
                    "missing_reason_count": sum(1 for c in charts if c.get("missing_reason")),
                    "chart_stats": charts,
                    "error": err,
                }
            )
    return out


def _aggregate_api_results(api_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total_calls": 0,
            "status_200": 0,
            "status_404": 0,
            "status_500": 0,
            "status_other": 0,
            "latency_ms_sum": 0,
            "latency_ms_avg": 0.0,
            "allowed_ok_calls": 0,
        }
    )
    chart_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "total_points": 0,
            "non_null_points": 0,
            "missing_reason_hits": 0,
        }
    )

    for row in api_results:
        ep = str(row.get("endpoint"))
        code = int(row.get("status_code") or 0)
        latency = int(row.get("latency_ms") or 0)
        ep_row = endpoint_acc[ep]
        ep_row["total_calls"] += 1
        ep_row["latency_ms_sum"] += latency
        if code == 200:
            ep_row["status_200"] += 1
        elif code == 404:
            ep_row["status_404"] += 1
        elif code == 500:
            ep_row["status_500"] += 1
        else:
            ep_row["status_other"] += 1

        allowed_ok = code == 200 or (ep == "business" and code == 404)
        if allowed_ok:
            ep_row["allowed_ok_calls"] += 1

        for chart in row.get("chart_stats", []):
            cid = str(chart.get("chart_id") or "")
            if not cid:
                continue
            key = f"{ep}:{cid}"
            c = chart_acc[key]
            c["calls"] += 1
            c["total_points"] += int(chart.get("total_points") or 0)
            c["non_null_points"] += int(chart.get("non_null_points") or 0)
            if chart.get("missing_reason"):
                c["missing_reason_hits"] += 1

    endpoint_summary: list[dict[str, Any]] = []
    for ep, vals in sorted(endpoint_acc.items()):
        total_calls = vals["total_calls"] or 1
        vals["latency_ms_avg"] = round(vals["latency_ms_sum"] / total_calls, 2)
        vals["ratio_200"] = round(vals["status_200"] / total_calls, 4)
        vals["ratio_allowed_ok"] = round(vals["allowed_ok_calls"] / total_calls, 4)
        endpoint_summary.append({"endpoint": ep, **vals})

    chart_summary: list[dict[str, Any]] = []
    for key, vals in chart_acc.items():
        total_points = vals["total_points"]
        non_null = vals["non_null_points"]
        null_rate = 1.0 if total_points <= 0 else max(0.0, 1.0 - (non_null / total_points))
        chart_summary.append(
            {
                "chart_key": key,
                "calls": vals["calls"],
                "total_points": total_points,
                "non_null_points": non_null,
                "null_rate": round(null_rate, 6),
                "missing_reason_hits": vals["missing_reason_hits"],
            }
        )
    chart_summary.sort(key=lambda x: (-x["null_rate"], -x["missing_reason_hits"], x["chart_key"]))
    return endpoint_summary, chart_summary


def _compute_db_coverage(tickers: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    if not DB_PATH.exists():
        return [], [], "duckdb_missing"
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
    except Exception as exc:  # noqa: BLE001
        return [], [], f"duckdb_open_failed:{exc}"

    sample_df = pd.DataFrame({"ticker": tickers})
    con.register("sample_tickers", sample_df)

    coverage_targets = [
        ("CurrentAssets", "Current Assets"),
        ("CurrentLiabilities", "Current Liabilities"),
        ("AR", "AR"),
        ("AP", "AP"),
        ("Inventory", "Inventory"),
        ("OperatingCashFlow", "Operating Cash Flow"),
        ("CapitalExpenditure", "Capital Expenditure"),
        ("Revenue", "Revenue"),
        ("OperatingIncome", "Operating Income"),
        ("NetIncome", "Net Income"),
        ("TotalAssets", "Total Assets"),
        ("TotalLiabilities", "Total Liabilities"),
        ("Equity", "Shareholders Equity"),
    ]
    coverage_rows: list[dict[str, Any]] = []
    for metric_name, col in coverage_targets:
        col_escaped = col.replace('"', '""')
        sql = f"""
            SELECT
              s.ticker,
              COALESCE(MAX(CASE WHEN f."{col_escaped}" IS NOT NULL THEN 1 ELSE 0 END), 0) AS has_value
            FROM sample_tickers s
            LEFT JOIN financials_quarterly f
              ON upper(f.ticker) = upper(s.ticker) AND lower(coalesce(f.market, 'us')) = 'us'
            GROUP BY s.ticker
            ORDER BY s.ticker
        """
        rows = con.execute(sql).fetchall()
        has_values = [int(r[1]) for r in rows]
        ratio = (sum(has_values) / len(has_values)) if has_values else 0.0
        coverage_rows.append(
            {
                "metric": metric_name,
                "column": col,
                "ticker_coverage_ratio": round(ratio, 6),
                "tickers_with_value": int(sum(has_values)),
                "sample_size": int(len(has_values)),
            }
        )

    earliest_sql = """
        SELECT
          s.ticker,
          MIN(p.date) AS prices_earliest_date,
          MIN(f."PeriodEnd") AS financials_earliest_period_end,
          MIN(f."AvailableDate") AS financials_earliest_available_date
        FROM sample_tickers s
        LEFT JOIN prices p
          ON upper(p.ticker) = upper(s.ticker) AND lower(coalesce(p.market, 'us')) = 'us'
        LEFT JOIN financials_quarterly f
          ON upper(f.ticker) = upper(s.ticker) AND lower(coalesce(f.market, 'us')) = 'us'
        GROUP BY s.ticker
        ORDER BY s.ticker
    """
    earliest_df = con.execute(earliest_sql).df()
    con.close()

    earliest_rows: list[dict[str, Any]] = []
    start_ts = pd.Timestamp(START_DATE_FIXED)
    for _, row in earliest_df.iterrows():
        prices_earliest = pd.to_datetime(row.get("prices_earliest_date"), errors="coerce")
        fin_period_earliest = pd.to_datetime(row.get("financials_earliest_period_end"), errors="coerce")
        fin_avail_earliest = pd.to_datetime(row.get("financials_earliest_available_date"), errors="coerce")
        note = ""
        if pd.isna(prices_earliest) and pd.isna(fin_period_earliest):
            note = "no-data"
        elif pd.isna(prices_earliest) and pd.notna(fin_period_earliest):
            note = "price no-data (pre-IPO/provider gap)"
        elif pd.notna(prices_earliest) and prices_earliest > start_ts:
            note = "pre-IPO/late-listing likely"
        earliest_rows.append(
            {
                "ticker": _safe_ticker(row.get("ticker")),
                "prices_earliest_date": prices_earliest.date().isoformat() if pd.notna(prices_earliest) else None,
                "financials_earliest_period_end": fin_period_earliest.date().isoformat() if pd.notna(fin_period_earliest) else None,
                "financials_earliest_available_date": fin_avail_earliest.date().isoformat() if pd.notna(fin_avail_earliest) else None,
                "note": note,
            }
        )
    return coverage_rows, earliest_rows, None


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_no rows_\n"
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([head, sep, *body]) + "\n"


def _write_report_md(
    *,
    seed: int,
    sample_source: str,
    tickers: list[str],
    ingest_results: list[dict[str, Any]],
    endpoint_summary: list[dict[str, Any]],
    chart_summary: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    earliest_rows: list[dict[str, Any]],
    api_server: ApiServerHandle,
) -> None:
    DOC_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ingest_counter = Counter(r.get("status", "unknown") for r in ingest_results)
    fail_reasons = Counter((r.get("reason") or "unknown")[:160] for r in ingest_results if r.get("status") == "fail")
    top_fail = fail_reasons.most_common(5)

    endpoint_table_rows = []
    for r in endpoint_summary:
        endpoint_table_rows.append(
            [
                r["endpoint"],
                r["total_calls"],
                r["status_200"],
                r["status_404"],
                r["status_500"],
                r["status_other"],
                f"{float(r['ratio_200']) * 100:.1f}%",
                f"{float(r['ratio_allowed_ok']) * 100:.1f}%",
                f"{float(r['latency_ms_avg']):.1f}",
            ]
        )

    top_chart_rows = []
    for r in chart_summary[:20]:
        top_chart_rows.append(
            [
                r["chart_key"],
                r["calls"],
                r["non_null_points"],
                r["total_points"],
                f"{float(r['null_rate']) * 100:.1f}%",
                r["missing_reason_hits"],
            ]
        )

    coverage_table_rows = []
    for r in coverage_rows:
        coverage_table_rows.append(
            [
                r["metric"],
                r["column"],
                r["tickers_with_value"],
                r["sample_size"],
                f"{float(r['ticker_coverage_ratio']) * 100:.1f}%",
            ]
        )

    earliest_table_rows = []
    for r in earliest_rows:
        earliest_table_rows.append(
            [
                r["ticker"],
                r["prices_earliest_date"] or "-",
                r["financials_earliest_period_end"] or "-",
                r["financials_earliest_available_date"] or "-",
                r["note"] or "-",
            ]
        )

    low_cov = [r for r in coverage_rows if float(r.get("ticker_coverage_ratio", 0.0)) < 0.6]
    severe_charts = [r for r in chart_summary if float(r.get("null_rate", 0.0)) >= 0.9]

    recommendations: list[str] = []
    if low_cov:
        recommendations.append(
            f"CompanyFacts tag 매핑 보강 우선: {', '.join(r['metric'] for r in low_cov[:6])}"
        )
    if any("business" in str(r.get("chart_key", "")) for r in severe_charts):
        recommendations.append("사업정보는 filings XBRL dimension 커버리지 확대 및 HTML fallback rule 보강 필요")
    if any(
        key in str(r.get("chart_key", ""))
        for r in severe_charts
        for key in ("growth", "turnover", "cycle", "peg", "por", "pfcfr")
    ):
        recommendations.append("파생지표 물질화(derived_factors_quarterly) 입력 컬럼 누락/분모 조건을 점검 필요")
    if not recommendations:
        recommendations.append("현재 샘플에서 치명적 누락 패턴이 크지 않음. low null-rate chart 위주로 우선순위 재조정 권장")

    md = []
    md.append("# Sample 30 E2E Report\n")
    md.append(f"- generated_at: `{_now_iso()}`")
    md.append(f"- seed: `{seed}`")
    md.append(f"- sample_source: `{sample_source}`")
    md.append(f"- sample_size: `{len(tickers)}`")
    md.append(f"- fixed_price_start_date: `{START_DATE_FIXED}`")
    md.append(f"- api_server_mode: `{api_server.mode}`")
    md.append(f"- api_base_url: `{api_server.base_url}`")
    if api_server.error:
        md.append(f"- api_server_error: `{api_server.error}`")
    md.append("")
    md.append("## Sample Tickers")
    md.append("")
    md.append(", ".join(f"`{t}`" for t in tickers))
    md.append("")
    md.append("## Ingest Summary")
    md.append("")
    md.append(f"- success: `{ingest_counter.get('success', 0)}`")
    md.append(f"- fail: `{ingest_counter.get('fail', 0)}`")
    md.append(f"- skip: `{ingest_counter.get('skip', 0)}`")
    md.append("")
    md.append("### Top Fail Reasons (Top 5)")
    md.append("")
    if top_fail:
        md.append(_md_table(["reason", "count"], [[reason, count] for reason, count in top_fail]))
    else:
        md.append("_no failures_\n")

    md.append("## Endpoint Health")
    md.append("")
    md.append(
        _md_table(
            ["endpoint", "calls", "200", "404", "500", "other", "200_ratio", "allowed_ok_ratio", "avg_latency_ms"],
            endpoint_table_rows,
        )
    )

    md.append("## Chart Null-rate Top 20")
    md.append("")
    md.append(
        _md_table(
            ["chart_key", "calls", "non_null_points", "total_points", "null_rate", "missing_reason_hits"],
            top_chart_rows,
        )
    )

    md.append("## Source Column Coverage (financials_quarterly)")
    md.append("")
    md.append(
        _md_table(
            ["metric", "column", "tickers_with_value", "sample_size", "coverage_ratio"],
            coverage_table_rows,
        )
    )

    md.append("## Earliest Coverage by Ticker")
    md.append("")
    md.append(
        _md_table(
            ["ticker", "prices_earliest_date", "financials_earliest_period_end", "financials_earliest_available_date", "note"],
            earliest_table_rows,
        )
    )

    md.append("## Next Actions")
    md.append("")
    for rec in recommendations:
        md.append(f"1. {rec}")
    md.append("")

    REPORT_MD_PATH.write_text("\n".join(md), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sample-30 E2E validation for data missing-rate improvements")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--sec-user-agent", default=os.getenv("SEC_USER_AGENT", "market-data-lake/0.1 (local-tooling@example.com)"))
    parser.add_argument("--segment-lookback-filings", type=int, default=8)
    parser.add_argument("--ingest-timeout-sec", type=int, default=DEFAULT_INGEST_TIMEOUT_SEC)
    parser.add_argument("--endpoint-timeout-sec", type=int, default=DEFAULT_ENDPOINT_TIMEOUT_SEC)
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC)
    args = parser.parse_args()

    DATA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    overall: dict[str, Any] = {
        "generated_at": _now_iso(),
        "seed": int(args.seed),
        "sample_size": int(args.sample_size),
        "fixed_price_start_date": START_DATE_FIXED,
        "sample_source": None,
        "tickers": [],
        "ingest": {"results": [], "summary": {}},
        "api": {"base_url": None, "server_mode": None, "server_error": None, "results": [], "endpoint_summary": []},
        "chart_null_rate": [],
        "db_coverage": {"columns": [], "earliest_by_ticker": [], "error": None},
        "errors": [],
    }

    if not DB_PATH.exists():
        msg = f"DuckDB not found: {DB_PATH}. Run ingest first."
        overall["errors"].append(msg)
        RESULTS_JSON_PATH.write_text(json.dumps(overall, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        REPORT_MD_PATH.write_text(f"# Sample 30 E2E Report\n\n- error: `{msg}`\n", encoding="utf-8")
        print(msg)
        return 2

    # 1) sample ticker selection
    try:
        candidates, source_note = _load_universe_candidates(sample_size=int(args.sample_size))
        selected = _sample_tickers(candidates, sample_size=int(args.sample_size), seed=int(args.seed))
        tickers = [_safe_ticker(x["ticker"]) for x in selected]
        _write_sample_tickers_file(tickers)
        overall["sample_source"] = source_note
        overall["tickers"] = tickers
        print(f"[SAMPLE] source={source_note} count={len(tickers)} seed={args.seed}")
    except Exception as exc:  # noqa: BLE001
        msg = f"sample_selection_failed: {exc}"
        overall["errors"].append(msg)
        tickers = []
        print(f"[ERROR] {msg}")

    python_bin = _pick_python_bin()

    # 2) ingest/backfill per ticker
    ingest_results: list[dict[str, Any]] = []
    if tickers:
        print(f"[INGEST] starting {len(tickers)} tickers (start={START_DATE_FIXED})")
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[INGEST] {idx}/{len(tickers)} {ticker}")
        result = _run_ingest_for_ticker(
            python_bin=python_bin,
            ticker=ticker,
            sec_user_agent=str(args.sec_user_agent or "").strip() or None,
            lookback_filings=int(args.segment_lookback_filings),
            ingest_timeout_sec=int(args.ingest_timeout_sec),
        )
        ingest_results.append(result)
        if result["status"] != "success":
            print(f"[INGEST WARN] {ticker} status={result['status']} reason={result['reason']}")
        if float(args.sleep_sec) > 0:
            time.sleep(float(args.sleep_sec))

    ingest_counter = Counter(r.get("status", "unknown") for r in ingest_results)
    overall["ingest"]["results"] = ingest_results
    overall["ingest"]["summary"] = {
        "success": int(ingest_counter.get("success", 0)),
        "fail": int(ingest_counter.get("fail", 0)),
        "skip": int(ingest_counter.get("skip", 0)),
        "total": int(len(ingest_results)),
    }

    # 3) API smoke test
    api_handle = _ensure_api_server(python_bin, str(args.api_base))
    overall["api"]["base_url"] = api_handle.base_url
    overall["api"]["server_mode"] = api_handle.mode
    overall["api"]["server_error"] = api_handle.error

    api_results: list[dict[str, Any]] = []
    endpoint_summary: list[dict[str, Any]] = []
    chart_summary: list[dict[str, Any]] = []
    if tickers and api_handle.mode != "unavailable":
        try:
            api_results = _run_api_smoke(api_handle.base_url, tickers, endpoint_timeout_sec=int(args.endpoint_timeout_sec))
            endpoint_summary, chart_summary = _aggregate_api_results(api_results)
        except Exception as exc:  # noqa: BLE001
            overall["errors"].append(f"api_smoke_failed:{exc}")
    else:
        overall["errors"].append("api_server_unavailable")
    overall["api"]["results"] = api_results
    overall["api"]["endpoint_summary"] = endpoint_summary
    overall["chart_null_rate"] = chart_summary

    _stop_api_server(api_handle)

    # 4) DB coverage
    coverage_rows, earliest_rows, coverage_error = _compute_db_coverage(tickers)
    overall["db_coverage"]["columns"] = coverage_rows
    overall["db_coverage"]["earliest_by_ticker"] = earliest_rows
    overall["db_coverage"]["error"] = coverage_error
    if coverage_error:
        overall["errors"].append(coverage_error)

    # 5) write outputs
    RESULTS_JSON_PATH.write_text(
        json.dumps(overall, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_report_md(
        seed=int(args.seed),
        sample_source=str(overall.get("sample_source") or "unknown"),
        tickers=tickers,
        ingest_results=ingest_results,
        endpoint_summary=endpoint_summary,
        chart_summary=chart_summary,
        coverage_rows=coverage_rows,
        earliest_rows=earliest_rows,
        api_server=api_handle,
    )

    print(f"[DONE] sample tickers: {SAMPLE_TICKERS_PATH}")
    print(f"[DONE] json report: {RESULTS_JSON_PATH}")
    print(f"[DONE] markdown report: {REPORT_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
