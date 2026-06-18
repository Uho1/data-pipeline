from __future__ import annotations

import importlib.util
import io
import site
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from market_data.utils import ensure_dir, now_utc_iso, retry_call

FINTERSTELLAR_CONSOLIDATED_URL = "https://api.finterstellar.com/api/consolidated"

INCOME_COLUMNS = [
    "Revenue",
    "COGS",
    "Gross Profit",
    "SG&A",
    "Operating Income",
    "Net Income",
    "EPS",
    "EBITDA",
    "EBIT",
    "Shares",
]

BALANCE_COLUMNS = [
    "Cash & Equivalents",
    "Receivables",
    "Inventory",
    "Current Assets",
    "Long Term Assets",
    "Total Assets",
    "Current Debt",
    "Current Liabilities",
    "Long Term Debt",
    "Long Term Liabilities",
    "Total Liabilities",
    "Shareholders Equity",
    "Shares",
]

CASHFLOW_COLUMNS = [
    "Depreciation",
    "Operating Cash Flow",
    "Capital Expenditure",
    "Investing Cash Flow",
    "Dividend",
    "Financing Cash Flow",
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
]


def _site_package_roots() -> list[Path]:
    paths: list[Path] = []
    try:
        paths.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        user = site.getusersitepackages()
        if user:
            paths.append(Path(user))
    except Exception:
        pass
    return paths


@lru_cache(maxsize=1)
def _load_fn_consolidated():
    for root in _site_package_roots():
        mod_path = root / "finterstellar" / "financials.py"
        if not mod_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("_fs_financials_mod", mod_path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "fn_consolidated", None)
        if callable(fn):
            return fn
    raise ImportError("finterstellar financials module not found")


def _call_consolidated_via_module(
    otp: str,
    symbol: str,
    term: str,
    vol: int,
    study: str,
) -> pd.DataFrame:
    fn = _load_fn_consolidated()
    df = fn(otp, symbol=symbol, term=term, vol=vol, study=study)
    if isinstance(df, str):
        raise RuntimeError(df)
    if df is None:
        raise RuntimeError("Empty response from finterstellar.fn_consolidated")
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"Unexpected finterstellar response type: {type(df)}")
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "symbol" in out.columns:
        out = out.set_index("symbol", drop=True)
    out.index = out.index.astype(str)
    return out


def _call_consolidated_via_http(
    otp: str,
    symbol: str,
    term: str,
    vol: int,
    study: str,
) -> pd.DataFrame:
    url = (
        f"{FINTERSTELLAR_CONSOLIDATED_URL}"
        f"?otp={otp}&symbol={symbol}&term={term}&vol={vol}&study={study}"
    )
    response = requests.get(url, timeout=20)
    text = response.text.strip()
    if response.status_code >= 500:
        # Finterstellar may return HTML 500 for out-of-history terms.
        if text.lower().startswith("<!doctype html") or "finterstellar" in text.lower():
            return pd.DataFrame()
        response.raise_for_status()
    if response.status_code >= 400:
        response.raise_for_status()
    if not text:
        return pd.DataFrame()
    if "Invalid OTP" in text:
        raise RuntimeError("Invalid OTP")
    if text.lower().startswith("<!doctype html"):
        return pd.DataFrame()

    try:
        df = pd.read_json(io.StringIO(text), orient="index")
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if "symbol" in df.columns:
        df = df.set_index("symbol", drop=True)
    df.index = df.index.astype(str)
    return df


def _call_consolidated(
    otp: str,
    symbol: str,
    term: str,
    vol: int,
    study: str,
) -> pd.DataFrame:
    try:
        return _call_consolidated_via_http(
            otp=otp,
            symbol=symbol,
            term=term,
            vol=vol,
            study=study,
        )
    except Exception as exc:
        if "Invalid OTP" in str(exc):
            raise
        return _call_consolidated_via_module(
            otp=otp,
            symbol=symbol,
            term=term,
            vol=vol,
            study=study,
        )


def _to_term(date_like: pd.Timestamp) -> str:
    p = pd.Period(date_like, freq="Q")
    return f"{p.year}Q{p.quarter}"


def _parse_term_to_statement_date(term: str) -> pd.Timestamp:
    text = str(term).strip().upper().replace("-", "").replace("_", "")
    text = text.replace(" ", "")
    if "Q" not in text:
        return pd.NaT
    try:
        p = pd.Period(text, freq="Q")
        return p.to_timestamp(how="end").normalize()
    except Exception:
        return pd.NaT


def _build_terms(
    start: str,
    end: str | None,
    history_years: int,
) -> list[str]:
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(end_ts):
        end_ts = pd.Timestamp.today().normalize()

    start_ts = pd.to_datetime(start, errors="coerce")
    if pd.isna(start_ts):
        start_ts = end_ts - pd.DateOffset(years=history_years)

    if history_years > 0:
        floor = end_ts - pd.DateOffset(years=history_years)
        start_ts = max(start_ts, floor)

    if start_ts > end_ts:
        start_ts = end_ts

    terms = pd.period_range(start=start_ts, end=end_ts, freq="Q")
    return [_to_term(t.to_timestamp()) for t in terms]


def fetch_consolidated_history(
    ticker: str,
    otp: str,
    start: str,
    end: str | None,
    history_years: int,
    retries: int,
    backoff: float,
    vol: int,
    study: str,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(end_ts):
        end_ts = pd.Timestamp.today().normalize()
    if pd.isna(start_ts):
        start_ts = end_ts - pd.DateOffset(years=history_years)

    # Fast path: single request can return whole history for a symbol.
    try:
        full_df = retry_call(
            lambda: _call_consolidated(
                otp=otp,
                symbol=ticker,
                term="",
                vol=vol,
                study=study,
            ),
            retries=retries,
            backoff_base=backoff,
            label=f"finterstellar:{ticker}:all",
        )
        if full_df is not None and not full_df.empty:
            full = full_df.copy()
            if ticker in full.index:
                full = full.loc[[ticker]]
            elif "symbol" in full.columns:
                full = full.loc[full["symbol"].astype(str) == ticker]
            if not full.empty:
                full["symbol"] = ticker
                if "term" not in full.columns:
                    full["term"] = ""
                full["term"] = full["term"].astype(str)
                full["StatementDate"] = full["term"].map(_parse_term_to_statement_date)
                full = full.loc[~full["StatementDate"].isna()]
                full = full.loc[(full["StatementDate"] >= start_ts) & (full["StatementDate"] <= end_ts)]
                full = full.sort_values(["StatementDate", "term"]).drop_duplicates(subset=["term"], keep="last")
                full["CollectedAt"] = now_utc_iso()
                if not full.empty:
                    return full.reset_index(drop=True)
    except Exception:
        # Fall back to term-by-term mode below.
        pass

    terms = _build_terms(start=start, end=end, history_years=history_years)
    rows: list[pd.DataFrame] = []
    errors: list[str] = []

    for term in terms:
        try:
            df = retry_call(
                lambda: _call_consolidated(
                    otp=otp,
                    symbol=ticker,
                    term=term,
                    vol=vol,
                    study=study,
                ),
                retries=retries,
                backoff_base=backoff,
                label=f"finterstellar:{ticker}:{term}",
            )
        except Exception as exc:  # noqa: BLE001
            cause = getattr(exc, "__cause__", None)
            cause_msg = str(cause) if cause is not None else ""
            msg = f"{exc}; cause={cause_msg}" if cause_msg else str(exc)
            errors.append(f"{term}:{msg}")
            if "Invalid OTP" in msg:
                raise RuntimeError("Invalid OTP for finterstellar API") from exc
            continue

        if df is None or df.empty:
            continue

        frame = df.copy()
        if ticker in frame.index:
            frame = frame.loc[[ticker]]
        elif "symbol" in frame.columns:
            frame = frame.loc[frame["symbol"].astype(str) == ticker]
        else:
            # symbol filter was applied in request, but guard anyway.
            frame = frame.head(1)

        if frame.empty:
            continue

        frame = frame.copy()
        frame["symbol"] = ticker
        frame["term"] = frame.get("term", term)
        rows.append(frame.reset_index(drop=True))

    if not rows:
        if errors:
            raise RuntimeError(f"No finterstellar rows for {ticker}; last_error={errors[-1]}")
        return pd.DataFrame(columns=["symbol", "term", "StatementDate", "CollectedAt"])

    out = pd.concat(rows, ignore_index=True, sort=False)
    out["term"] = out["term"].astype(str)
    out["StatementDate"] = out["term"].map(_parse_term_to_statement_date)
    out = out.loc[~out["StatementDate"].isna()]
    out = out.sort_values(["StatementDate", "term"]).drop_duplicates(subset=["term"], keep="last")
    out["CollectedAt"] = now_utc_iso()
    return out.reset_index(drop=True)


def fetch_consolidated_universe_history(
    otp: str,
    symbols: list[str],
    start: str,
    end: str | None,
    history_years: int,
    retries: int,
    backoff: float,
    vol: int,
    study: str,
    cache_dir: Path | None = None,
    force: bool = False,
    progress_cb: Callable[[str, str], None] | None = None,
) -> pd.DataFrame:
    terms = _build_terms(start=start, end=end, history_years=history_years)
    symbol_set = {str(s).strip().upper() for s in symbols if str(s).strip()}
    rows: list[pd.DataFrame] = []
    errors: list[str] = []

    if cache_dir is not None:
        ensure_dir(cache_dir)

    for term in terms:
        cache_path = cache_dir / f"{term}.parquet" if cache_dir is not None else None
        term_df: pd.DataFrame

        if cache_path is not None and cache_path.exists() and not force:
            try:
                term_df = pd.read_parquet(cache_path)
                if progress_cb is not None:
                    progress_cb(term, "OK")
            except Exception:
                term_df = pd.DataFrame()
                if progress_cb is not None:
                    progress_cb(term, "FAIL")
        else:
            try:
                term_df = retry_call(
                    lambda: _call_consolidated(
                        otp=otp,
                        symbol="",
                        term=term,
                        vol=vol,
                        study=study,
                    ),
                    retries=retries,
                    backoff_base=backoff,
                    label=f"finterstellar:all:{term}",
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{term}:{exc}")
                if "Invalid OTP" in str(exc):
                    raise RuntimeError("Invalid OTP for finterstellar API") from exc
                if progress_cb is not None:
                    progress_cb(term, "FAIL")
                continue

            if cache_path is not None and term_df is not None and not term_df.empty:
                saved = term_df.copy()
                if "symbol" not in saved.columns:
                    saved = saved.reset_index().rename(columns={"index": "symbol"})
                saved.to_parquet(cache_path, index=False)
            if progress_cb is not None:
                progress_cb(term, "OK")

        if term_df is None or term_df.empty:
            continue

        frame = term_df.copy()
        if "symbol" not in frame.columns:
            frame = frame.reset_index().rename(columns={"index": "symbol"})
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        if symbol_set:
            frame = frame.loc[frame["symbol"].isin(symbol_set)]
        if frame.empty:
            continue
        if "term" not in frame.columns:
            frame["term"] = term
        frame["term"] = frame["term"].astype(str)
        rows.append(frame.reset_index(drop=True))

    if not rows:
        if errors:
            raise RuntimeError(f"No finterstellar bulk rows; last_error={errors[-1]}")
        return pd.DataFrame(columns=["symbol", "term", "StatementDate", "CollectedAt"])

    out = pd.concat(rows, ignore_index=True, sort=False)
    out["term"] = out["term"].astype(str)
    out["StatementDate"] = out["term"].map(_parse_term_to_statement_date)
    out = out.loc[~out["StatementDate"].isna()]
    out = out.sort_values(["symbol", "StatementDate", "term"]).drop_duplicates(
        subset=["symbol", "term"], keep="last"
    )
    out["CollectedAt"] = now_utc_iso()
    return out.reset_index(drop=True)


def _select_columns(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    base = ["StatementDate", "term", "symbol"]
    cols = base + [c for c in value_columns if c in df.columns]
    out = df[cols].copy()
    out = out.rename(columns={"term": "Term"})
    out["CollectedAt"] = now_utc_iso()
    out["Source"] = "finterstellar_consolidated"
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


def build_statement_frames(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if history.empty:
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
