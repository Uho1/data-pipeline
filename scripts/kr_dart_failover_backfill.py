from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEYCHAIN_SERVICES = (
    "market-data-dart-api",
    "market-data-dart-api-2",
    "market-data-dart-api-3",
)
FILINGS_SCRIPT = ROOT / "scripts" / "kr_dart_resume_filings.py"
FINANCIALS_SCRIPT = ROOT / "scripts" / "kr_dart_resume_financials.py"
DB_PATH = ROOT / "data" / "market_data_kr.duckdb"
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"
RATE_LIMIT_EXIT_CODE = 75


def _normalize_service_names(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_KEYCHAIN_SERVICES)
    out: list[str] = []
    for value in values:
        for token in str(value).split(","):
            name = token.strip()
            if name and name not in out:
                out.append(name)
    return out or list(DEFAULT_KEYCHAIN_SERVICES)


def _get_keychain_secret(service_name: str) -> str:
    return subprocess.check_output(
        [
            "security",
            "find-generic-password",
            "-a",
            os.getenv("USER", ""),
            "-s",
            service_name,
            "-w",
        ],
        text=True,
    ).strip()


def _remaining_filings() -> int:
    con = duckdb.connect(str(DB_PATH))
    try:
        query = """
        WITH eligible AS (
            SELECT ticker
            FROM ticker_master
            WHERE market = 'kr'
              AND COALESCE(NULLIF(TRIM(dart_corp_code), ''), '') <> ''
        ),
        filing_cov AS (
            SELECT ticker, COUNT(*) AS filing_count
            FROM filings
            GROUP BY 1
        )
        SELECT COUNT(*)
        FROM eligible e
        LEFT JOIN filing_cov f USING (ticker)
        WHERE COALESCE(f.filing_count, 0) = 0
        """
        return int(con.execute(query).fetchone()[0])
    finally:
        con.close()


def _remaining_financials() -> int:
    con = duckdb.connect(str(DB_PATH))
    try:
        query = """
        WITH raw_cov AS (
            SELECT
                ticker,
                MIN(bsns_year) AS min_year,
                MAX(bsns_year) AS max_year
            FROM dart_financials_raw
            WHERE ticker IS NOT NULL
            GROUP BY 1
        ),
        tm AS (
            SELECT
                ticker,
                YEAR(COALESCE(listed_date, DATE '2013-06-01')) AS listed_year
            FROM ticker_master
            WHERE market = 'kr'
        )
        SELECT COUNT(*)
        FROM tm
        LEFT JOIN raw_cov r USING (ticker)
        WHERE r.ticker IS NULL
           OR r.min_year > GREATEST(2013, tm.listed_year)
           OR r.max_year < 2025
        """
        return int(con.execute(query).fetchone()[0])
    finally:
        con.close()


def _run_stage(script_path: Path, *, service_name: str, api_key: str) -> int:
    env = os.environ.copy()
    env["dart_api"] = api_key
    cmd = [str(PYTHON_BIN), "-u", str(script_path)]
    return subprocess.run(cmd, cwd=str(ROOT), env=env, check=False).returncode


def _run_stage_with_failover(
    *,
    stage_name: str,
    script_path: Path,
    remaining_fn,
    services: list[str],
) -> int:
    remaining = int(remaining_fn())
    print(f"[failover] stage={stage_name} remaining={remaining}")
    if remaining <= 0:
        return 0

    for service_name in services:
        before = int(remaining_fn())
        if before <= 0:
            print(f"[failover] stage={stage_name} already complete")
            return 0
        try:
            api_key = _get_keychain_secret(service_name)
        except subprocess.CalledProcessError:
            print(f"[failover] stage={stage_name} service={service_name} SKIP missing_key")
            continue

        print(f"[failover] stage={stage_name} service={service_name} before={before}")
        return_code = _run_stage(script_path, service_name=service_name, api_key=api_key)
        after = int(remaining_fn())
        print(
            f"[failover] stage={stage_name} service={service_name} rc={return_code} before={before} after={after}"
        )

        if return_code == 0:
            if after == 0:
                return 0
            if after < before:
                continue
            print(f"[failover] stage={stage_name} no_progress_after_success service={service_name}")
            return 1

        if return_code == RATE_LIMIT_EXIT_CODE:
            if after < before:
                continue
            print(f"[failover] stage={stage_name} rate_limit_without_progress service={service_name}")
            continue

        return return_code

    final_remaining = int(remaining_fn())
    if final_remaining <= 0:
        return 0
    print(f"[failover] stage={stage_name} exhausted_services remaining={final_remaining}")
    return RATE_LIMIT_EXIT_CODE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run KR DART backfill with automatic key failover.")
    parser.add_argument(
        "--services",
        action="append",
        help="Keychain service names, comma-separated or repeated. Defaults to market-data-dart-api, market-data-dart-api-2, market-data-dart-api-3.",
    )
    parser.add_argument(
        "--skip-filings",
        action="store_true",
        help="Skip filings resume stage and run financials only.",
    )
    args = parser.parse_args(argv)

    services = _normalize_service_names(args.services)
    print(f"[failover] services={','.join(services)}")

    if not args.skip_filings:
        code = _run_stage_with_failover(
            stage_name="filings",
            script_path=FILINGS_SCRIPT,
            remaining_fn=_remaining_filings,
            services=services,
        )
        if code != 0:
            return code

    return _run_stage_with_failover(
        stage_name="financials",
        script_path=FINANCIALS_SCRIPT,
        remaining_fn=_remaining_financials,
        services=services,
    )


if __name__ == "__main__":
    raise SystemExit(main())
