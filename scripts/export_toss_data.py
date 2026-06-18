"""Export pre-computed finance view JSON files for the Toss mini-app.

The toss-app frontend has no backend — it fetches static JSON files directly
from CDN/R2 (or ``toss-app/public/toss-data`` during development). This script
calls the existing FastAPI payload functions once per ticker and serializes the
results to disk in the shape the frontend expects.

Output layout::

    {output_dir}/
    ├── index/
    │   ├── kr.json                # [{ticker, name, name_kr, sector, industry}, ...]
    │   └── us.json
    └── tickers/
        ├── kr/
        │   └── {ticker}/
        │       ├── main.json      # summary_dashboard + valuation + info
        │       └── detail/
        │           ├── income.json
        │           ├── balance.json
        │           └── cashflow.json
        └── us/ (same)

Usage::

    # Single ticker (validation)
    python scripts/export_toss_data.py --ticker 005930

    # All KR tickers (first 50 for smoke test)
    python scripts/export_toss_data.py --market kr --limit 50

    # Full export
    python scripts/export_toss_data.py --market kr --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Ensure user site-packages is available (matches run_backend.py pattern)
_USER_SITE = r"C:\Users\yh\AppData\Roaming\Python\Python313\site-packages"
if _USER_SITE not in sys.path:
    sys.path.insert(0, _USER_SITE)

# Repo root for relative imports — both repo root (for web.backend.*) and
# src/ (for market_data.*, see pyproject.toml package-dir = {"" = "src"})
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Default output directory — frontend dev location
_DEFAULT_OUTPUT = _REPO_ROOT / "toss-app" / "public" / "toss-data"

# Export configuration — defaults that match what the FinanceView renders
# Single-basis export keeps data size down; the frontend will hide the basis
# toggle in the Toss build.
_SUMMARY_QUARTERS = 40       # 10y of quarters
_WINDOW = "10y"
_INCOME_BASIS = "ttm"
_BALANCE_BASIS = "quarter"   # matches FinancialDetail.tsx balanceBasis mapping
_CASHFLOW_BASIS = "ttm"
_VALUATION_BASIS = "ttm"
_SUMMARY_BASIS = "ttm"


def _import_backend():
    """Import backend payload functions lazily so workers pay the cost once."""
    from web.backend.services.ticker_analysis_service import (
        get_financials_balance_payload,
        get_financials_cashflow_payload,
        get_financials_income_payload,
        get_summary_dashboard_payload,
        get_valuation_payload,
    )
    from web.backend.services.json_data_service import (
        get_ticker_info,
        load_ticker_master,
    )
    from web.backend.schemas.financials_balance import FinancialsBalanceResponse
    from web.backend.schemas.financials_cashflow import FinancialsCashflowResponse
    from web.backend.schemas.financials_income import FinancialsIncomeResponse
    from web.backend.schemas.summary_dashboard import SummaryDashboardResponse
    from web.backend.schemas.valuation_tab import ValuationTabResponse

    return {
        "get_summary_dashboard_payload": get_summary_dashboard_payload,
        "get_valuation_payload": get_valuation_payload,
        "get_financials_income_payload": get_financials_income_payload,
        "get_financials_balance_payload": get_financials_balance_payload,
        "get_financials_cashflow_payload": get_financials_cashflow_payload,
        "get_ticker_info": get_ticker_info,
        "load_ticker_master": load_ticker_master,
        "SummaryDashboardResponse": SummaryDashboardResponse,
        "ValuationTabResponse": ValuationTabResponse,
        "FinancialsIncomeResponse": FinancialsIncomeResponse,
        "FinancialsBalanceResponse": FinancialsBalanceResponse,
        "FinancialsCashflowResponse": FinancialsCashflowResponse,
    }


def _serialize(model_cls: Any, payload: dict) -> dict:
    """Validate with Pydantic, then dump to a JSON-ready dict.

    Using the response model ensures the on-disk shape exactly matches what the
    live API returned, so the frontend can be written against a single schema.
    """
    return model_cls(**payload).model_dump(mode="json", exclude_none=False)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


# Per-process master cache — avoids re-reading ticker_master_{market}.json for
# every ticker within a worker. Keyed by market.
_MASTER_INDEX: dict[str, dict[str, dict]] = {}


def _get_master_index(market: str, backend: dict) -> dict[str, dict]:
    """Return a {ticker: master_entry} lookup, cached per process."""
    if market in _MASTER_INDEX:
        return _MASTER_INDEX[market]
    try:
        master = backend["load_ticker_master"](market) or []
    except Exception:  # noqa: BLE001
        master = []
    index: dict[str, dict] = {}
    for item in master:
        ticker = (item.get("ticker") or "").strip()
        if ticker:
            index[ticker] = item
    _MASTER_INDEX[market] = index
    return index


def _build_info(ticker: str, market: str, backend: dict) -> dict:
    """Build a complete ``info`` block, filling empties from the master list.

    backend.get_ticker_info() reads per-ticker JSON which often lacks fields
    like ``market_tier`` (KOSPI/NYSE) or the Korean translation of US names.
    The master file has those — we overlay them so KR and US JSON output
    shapes are consistent.
    """
    try:
        info = backend["get_ticker_info"](ticker, market) or {}
    except Exception:  # noqa: BLE001
        info = {"ticker": ticker, "market": market}

    # Ensure required keys exist with sensible defaults
    info.setdefault("ticker", ticker)
    info.setdefault("market", market)
    for field in ("company_name", "name_kr", "sector", "industry", "market_tier"):
        info.setdefault(field, "")

    master_entry = _get_master_index(market, backend).get(ticker) or {}
    # Only overlay fields that are currently empty — don't clobber richer data
    # from the per-ticker JSON.
    if not info["company_name"]:
        info["company_name"] = master_entry.get("name", "") or ""
    if not info["name_kr"]:
        info["name_kr"] = master_entry.get("name_kr", "") or ""
    if not info["sector"]:
        info["sector"] = master_entry.get("sector", "") or ""
    if not info["industry"]:
        info["industry"] = master_entry.get("industry", "") or ""
    if not info["market_tier"]:
        info["market_tier"] = master_entry.get("market_tier", "") or ""

    return info


def export_ticker(ticker: str, market: str, output_dir: Path) -> dict:
    """Export all finance-view JSON files for one ticker.

    Returns a result dict with status info. Failures do not raise — they are
    recorded and the batch continues.
    """
    backend = _import_backend()
    ticker_dir = output_dir / "tickers" / market / ticker
    detail_dir = ticker_dir / "detail"

    result = {
        "ticker": ticker,
        "market": market,
        "ok": False,
        "files": [],
        "errors": [],
    }

    # ── 1. main.json: summary_dashboard + valuation + info ──────────────────
    try:
        summary_payload = backend["get_summary_dashboard_payload"](
            ticker=ticker,
            market=market,
            asof=None,
            quarters=_SUMMARY_QUARTERS,
            basis=_SUMMARY_BASIS,
        )
        summary = _serialize(backend["SummaryDashboardResponse"], summary_payload)
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"summary_dashboard: {e}")
        return result

    try:
        valuation_payload = backend["get_valuation_payload"](
            ticker=ticker,
            market=market,
            window=_WINDOW,
            basis=_VALUATION_BASIS,
        )
        valuation = _serialize(backend["ValuationTabResponse"], valuation_payload)
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"valuation: {e}")
        valuation = None

    try:
        info = _build_info(ticker, market, backend)
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"ticker_info: {e}")
        info = {"ticker": ticker, "market": market}

    main_doc = {
        "ticker": ticker,
        "market": market,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "info": info,
        "summary_dashboard": summary,
        "valuation": valuation,
    }
    _write_json(ticker_dir / "main.json", main_doc)
    result["files"].append("main.json")

    # ── 2. detail/income.json ───────────────────────────────────────────────
    try:
        income_payload = backend["get_financials_income_payload"](
            ticker=ticker,
            market=market,
            window=_WINDOW,
            basis=_INCOME_BASIS,
        )
        income = _serialize(backend["FinancialsIncomeResponse"], income_payload)
        _write_json(detail_dir / "income.json", income)
        result["files"].append("detail/income.json")
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"financials_income: {e}")

    # ── 3. detail/balance.json (basis=quarter to match frontend mapping) ───
    try:
        balance_payload = backend["get_financials_balance_payload"](
            ticker=ticker,
            market=market,
            window=_WINDOW,
            basis=_BALANCE_BASIS,
        )
        balance = _serialize(backend["FinancialsBalanceResponse"], balance_payload)
        _write_json(detail_dir / "balance.json", balance)
        result["files"].append("detail/balance.json")
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"financials_balance: {e}")

    # ── 4. detail/cashflow.json ─────────────────────────────────────────────
    try:
        cashflow_payload = backend["get_financials_cashflow_payload"](
            ticker=ticker,
            market=market,
            window=_WINDOW,
            basis=_CASHFLOW_BASIS,
        )
        cashflow = _serialize(backend["FinancialsCashflowResponse"], cashflow_payload)
        _write_json(detail_dir / "cashflow.json", cashflow)
        result["files"].append("detail/cashflow.json")
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"financials_cashflow: {e}")

    result["ok"] = len(result["files"]) > 0
    return result


def build_index(market: str, output_dir: Path) -> int:
    """Build the per-market search index from ticker_master."""
    backend = _import_backend()
    try:
        master = backend["load_ticker_master"](market)
    except Exception as e:  # noqa: BLE001
        print(f"[index] failed to load ticker_master_{market}: {e}", file=sys.stderr)
        return 0

    items = []
    for item in master:
        ticker = (item.get("ticker") or "").strip()
        if not ticker:
            continue
        items.append({
            "ticker": ticker,
            "name": item.get("name", "") or "",
            "name_kr": item.get("name_kr", "") or "",
            "sector": item.get("sector", "") or "",
            "industry": item.get("industry", "") or "",
            "market_tier": item.get("market_tier", "") or "",
        })

    index_doc = {
        "market": market,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(items),
        "items": items,
    }
    _write_json(output_dir / "index" / f"{market}.json", index_doc)
    return len(items)


def _worker(args: tuple[str, str, str]) -> dict:
    """Multiprocessing worker — isolates backend state per ticker."""
    ticker, market, output_dir_str = args
    try:
        return export_ticker(ticker, market, Path(output_dir_str))
    except Exception as e:  # noqa: BLE001
        return {
            "ticker": ticker,
            "market": market,
            "ok": False,
            "files": [],
            "errors": [f"worker_crash: {e}\n{traceback.format_exc()}"],
        }


def resolve_tickers(
    market: str,
    explicit: list[str] | None,
    limit: int | None,
) -> list[str]:
    """Determine which tickers to export.

    Priority: --ticker flag > ticker_master order, truncated by --limit.
    """
    if explicit:
        return [t.strip() for t in explicit if t.strip()]

    backend = _import_backend()
    master = backend["load_ticker_master"](market)
    tickers = [item.get("ticker", "") for item in master if item.get("ticker")]
    if limit is not None and limit > 0:
        tickers = tickers[:limit]
    return tickers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--market", default="kr", choices=["kr", "us"])
    parser.add_argument("--ticker", action="append", help="Specific ticker(s) to export (repeatable)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers (for smoke testing)")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker count (1 = sequential)")
    parser.add_argument("--skip-index", action="store_true", help="Skip building index/{market}.json")
    parser.add_argument("--index-only", action="store_true", help="Only build index, skip ticker export")
    args = parser.parse_args()

    output_dir: Path = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[toss-export] output={output_dir}")
    print(f"[toss-export] market={args.market}")

    # Build index first — it's cheap and fail-fast tests the master file
    if not args.skip_index:
        count = build_index(args.market, output_dir)
        print(f"[toss-export] index/{args.market}.json written ({count} tickers)")

    if args.index_only:
        return 0

    tickers = resolve_tickers(args.market, args.ticker, args.limit)
    if not tickers:
        print("[toss-export] no tickers to export", file=sys.stderr)
        return 1
    print(f"[toss-export] exporting {len(tickers)} tickers, workers={args.workers}")

    start = time.time()
    successes = 0
    failures = 0
    failure_log: list[dict] = []

    work_items = [(t, args.market, str(output_dir)) for t in tickers]

    if args.workers <= 1:
        # Sequential — easier to debug, uses a single backend import
        for i, item in enumerate(work_items, 1):
            result = _worker(item)
            if result["ok"]:
                successes += 1
            else:
                failures += 1
                failure_log.append(result)
            if i % 10 == 0 or i == len(work_items):
                elapsed = time.time() - start
                print(f"[toss-export] {i}/{len(work_items)} "
                      f"({successes} ok, {failures} fail, {elapsed:.1f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, item): item[0] for item in work_items}
            for i, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                if result["ok"]:
                    successes += 1
                else:
                    failures += 1
                    failure_log.append(result)
                if i % 10 == 0 or i == len(work_items):
                    elapsed = time.time() - start
                    print(f"[toss-export] {i}/{len(work_items)} "
                          f"({successes} ok, {failures} fail, {elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"[toss-export] done: {successes} ok, {failures} failed in {elapsed:.1f}s")

    if failure_log:
        log_path = output_dir / f"_export_failures_{args.market}.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(failure_log, f, ensure_ascii=False, indent=2)
        print(f"[toss-export] failure log → {log_path}")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
