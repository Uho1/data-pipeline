from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from pyarrow import dataset as ds

from market_data.config import PARQUET_DIR
from market_data.db_router import normalize_kr_ticker


_US_FINANCIALS_EXTRA_MAP = {
    "owner_equity": "Owner Equity",
    "owner_net_income": "Owner Net Income",
    "common_stock": "Common Stock",
    "additional_paid_in_capital": "Additional Paid In Capital",
    "retained_earnings": "Retained Earnings",
    "aoci": "AOCI",
    "ppe": "PPE",
    "ppe_capex": "PPE Capex",
    "intangibles": "Intangibles",
    "intangible_capex": "Intangible Capex",
    "amortization": "Amortization",
    "other_gain": "Other Gain",
    "financial_gain": "Financial Gain",
    "equity_method_gain": "Equity Method Gain",
    "other_income": "Other Income",
    "other_expense": "Other Expense",
    "financial_income": "Financial Income",
    "financial_expense": "Financial Expense",
    "current_fin_assets": "Current Fin Assets",
    "non_current_fin_assets": "Non Current Fin Assets",
    "current_fin_liabilities": "Current Fin Liabilities",
    "non_current_fin_liabilities": "Non Current Fin Liabilities",
}

_TICKER_PARTITIONED_TABLES = {
    "prices",
    "financials_quarterly",
    "financials_quarterly_extra",
    "derived_factors_quarterly",
    "filings",
    "sec_facts_raw_normalized",
}
_CORP_CODE_PARTITIONED_TABLES = {
    "dart_financials_raw",
}
_DIRECT_TICKER_FILE_TABLES = _TICKER_PARTITIONED_TABLES - {"prices"}


def _normalize_market(market: str) -> str:
    return str(market or "").strip().lower()


def _normalize_ticker(ticker: str, market: str) -> str:
    if _normalize_market(market) == "kr":
        return normalize_kr_ticker(ticker)
    return str(ticker or "").strip().upper()


def _normalize_tickers(tickers: list[str], market: str) -> list[str]:
    out: list[str] = []
    for ticker in tickers:
        text = str(ticker or "").strip()
        if not text:
            continue
        out.append(_normalize_ticker(text, market))
    return out


def _table_path(market: str, table: str) -> Path | None:
    base = PARQUET_DIR / _normalize_market(market)
    dir_path = base / table
    if dir_path.exists():
        return dir_path
    file_path = base / f"{table}.parquet"
    if file_path.exists():
        return file_path
    return None


def _ticker_partition_file_path(market: str, table: str, ticker: str) -> Path:
    return PARQUET_DIR / _normalize_market(market) / table / f"ticker={_normalize_ticker(ticker, market)}" / "data.parquet"


def _read_exact_ticker_partition_table(
    market: str,
    table: str,
    ticker: str,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    path = _ticker_partition_file_path(market, table, ticker)
    if path.exists():
        selected = None if columns is None else [column for column in columns if column != "ticker"]
        df = pd.read_parquet(path, columns=selected)
        df["ticker"] = _normalize_ticker(ticker, market)
    else:
        legacy_path = _table_path(market, table)
        if legacy_path is None or not legacy_path.exists() or legacy_path.is_dir():
            return pd.DataFrame(columns=columns or [])
        df = pd.read_parquet(legacy_path, columns=columns)
        if "ticker" not in df.columns:
            df["ticker"] = _normalize_ticker(ticker, market)
        df = df[df["ticker"].astype(str) == _normalize_ticker(ticker, market)].copy()
    if columns is not None:
        for column in columns:
            if column not in df.columns:
                df[column] = pd.NA
        df = df[columns]
    return df


def _open_dataset(market: str, table: str):
    path = _table_path(market, table)
    if path is None:
        return None
    kwargs: dict[str, Any] = {"format": "parquet"}
    if path.is_dir():
        if table == "prices":
            kwargs["partitioning"] = ds.partitioning(
                pa.schema(
                    [
                        ("ticker", pa.string()),
                        ("year", pa.int32()),
                    ]
                ),
                flavor="hive",
            )
        elif table in _TICKER_PARTITIONED_TABLES:
            kwargs["partitioning"] = ds.partitioning(
                pa.schema(
                    [
                        ("ticker", pa.string()),
                    ]
                ),
                flavor="hive",
            )
        elif table in _CORP_CODE_PARTITIONED_TABLES:
            kwargs["partitioning"] = ds.partitioning(
                pa.schema(
                    [
                        ("corp_code", pa.string()),
                    ]
                ),
                flavor="hive",
            )
        else:
            kwargs["partitioning"] = "hive"
    return ds.dataset(str(path), **kwargs)


def _safe_columns(dataset, columns: list[str] | None) -> list[str] | None:
    if columns is None:
        return None
    available = set(dataset.schema.names)
    selected = [column for column in columns if column in available]
    return selected


def _normalize_filter_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0 and value.nanosecond == 0:
            return value.date()
        return value.to_pydatetime()
    if isinstance(value, list):
        return [_normalize_filter_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_filter_value(item) for item in value)
    return value


def _build_filter_expression(filters: list[tuple[str, str, Any]] | None):
    if not filters:
        return None
    expression = None
    for column, operator, raw_value in filters:
        value = _normalize_filter_value(raw_value)
        field = ds.field(column)
        if operator == "==":
            current = field == value
        elif operator == "!=":
            current = field != value
        elif operator == ">=":
            current = field >= value
        elif operator == "<=":
            current = field <= value
        elif operator == ">":
            current = field > value
        elif operator == "<":
            current = field < value
        elif operator == "in":
            current = field.isin(list(value))
        else:
            raise ValueError(f"Unsupported parquet filter operator: {operator}")
        expression = current if expression is None else expression & current
    return expression


def _extract_exact_ticker_filter(filters: list[tuple[str, str, Any]] | None) -> str | None:
    if not filters:
        return None
    for column, operator, raw_value in filters:
        if column == "ticker" and operator == "==":
            return str(raw_value)
    return None


def read_table(
    market: str,
    table: str,
    *,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    exact_ticker = _extract_exact_ticker_filter(filters)
    if exact_ticker and table in _DIRECT_TICKER_FILE_TABLES:
        df = _read_exact_ticker_partition_table(market, table, exact_ticker, columns=columns)
        if filters:
            for column, operator, raw_value in filters:
                value = _normalize_filter_value(raw_value)
                if operator == "==" and column in df.columns:
                    df = df[df[column].astype(str) == str(value)]
        return df.reset_index(drop=True)

    dataset = _open_dataset(market, table)
    if dataset is None:
        return pd.DataFrame(columns=columns or [])

    selected = _safe_columns(dataset, columns)
    if columns is not None and not selected:
        return pd.DataFrame(columns=columns)

    loaded = dataset.to_table(columns=selected, filter=_build_filter_expression(filters))
    return loaded.to_pandas()


def _coerce_datetime_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def _format_price_group(group: pd.DataFrame, *, close_only: bool = False) -> pd.DataFrame:
    rename_map = {"date": "Date", "close": "Close", "adj_close": "Adj Close"}
    if not close_only:
        rename_map.update(
            {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "volume": "Volume",
                "dividends": "Dividends",
                "stock_splits": "Stock Splits",
            }
        )
    if "market_cap" in group.columns:
        rename_map["market_cap"] = "MarketCap"
    if "shares_outstanding" in group.columns:
        rename_map["shares_outstanding"] = "SharesOutstanding"

    out = group.drop(columns=[column for column in ("ticker", "market") if column in group.columns]).rename(columns=rename_map)
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).set_index("Date").sort_index()
    return out


def bulk_load_prices(
    tickers: list[str],
    *,
    market: str,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    normalized = _normalize_tickers(tickers, market)
    if not normalized:
        return {}

    filters: list[tuple[str, str, Any]] = [("ticker", "in", normalized), ("market", "==", _normalize_market(market))]
    if start:
        filters.append(("date", ">=", pd.Timestamp(str(start)[:10])))
    if end:
        filters.append(("date", "<=", pd.Timestamp(str(end)[:10])))

    df = read_table(
        market,
        "prices",
        columns=["date", "ticker", "market", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits", "market_cap"],
        filters=filters,
    )
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"])

    result: dict[str, pd.DataFrame] = {}
    for ticker_value, group in df.groupby("ticker", sort=False):
        result[str(ticker_value)] = _format_price_group(group)
    return result


def bulk_load_price_close_frames(
    tickers: list[str],
    *,
    market: str,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    normalized = _normalize_tickers(tickers, market)
    if not normalized:
        return {}

    filters: list[tuple[str, str, Any]] = [("ticker", "in", normalized), ("market", "==", _normalize_market(market))]
    if start:
        filters.append(("date", ">=", pd.Timestamp(str(start)[:10])))
    if end:
        filters.append(("date", "<=", pd.Timestamp(str(end)[:10])))

    df = read_table(
        market,
        "prices",
        columns=["date", "ticker", "market", "close", "adj_close"],
        filters=filters,
    )
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"])

    result: dict[str, pd.DataFrame] = {}
    for ticker_value, group in df.groupby("ticker", sort=False):
        out = _format_price_group(group, close_only=True)
        keep_cols = [column for column in ("Adj Close", "Close") if column in out.columns and out[column].notna().any()]
        if keep_cols:
            result[str(ticker_value)] = out[keep_cols]
    return result


def load_price(
    ticker: str,
    *,
    market: str,
) -> tuple[pd.DataFrame, str] | None:
    normalized_ticker = _normalize_ticker(ticker, market)
    df = read_table(
        market,
        "prices",
        columns=["date", "ticker", "market", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits", "market_cap"],
        filters=[("ticker", "==", normalized_ticker), ("market", "==", _normalize_market(market))],
    )
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    effective_market = str(df["market"].iloc[0]) if "market" in df.columns and not df.empty else _normalize_market(market)
    return _format_price_group(df), effective_market


def bulk_load_financials_quarterly(
    tickers: list[str],
    *,
    market: str,
) -> dict[str, pd.DataFrame]:
    normalized = _normalize_tickers(tickers, market)
    if not normalized:
        return {}

    df = read_table(
        market,
        "financials_quarterly",
        filters=[("ticker", "in", normalized), ("market", "==", _normalize_market(market))],
    )
    if df.empty:
        return {}

    df = _coerce_datetime_columns(df, ("StatementDate", "PeriodEnd", "PeriodStart", "FilingDate", "AvailableDate", "AcceptedAt"))
    if "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    df = df.sort_values(["symbol", "PeriodEnd"])

    result: dict[str, pd.DataFrame] = {}
    for ticker_value, group in df.groupby("symbol", sort=False):
        result[str(ticker_value)] = group.reset_index(drop=True)
    return result


def _date_key(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce")
    return values.dt.strftime("%Y-%m-%d").fillna("<NULL>")


def _merge_us_financials_extra(financials: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if financials.empty or extra.empty:
        return financials

    base = financials.copy()
    right = extra.copy()

    base = _coerce_datetime_columns(base, ("PeriodEnd", "FilingDate"))
    right = _coerce_datetime_columns(right, ("period_end", "filing_date"))
    right = right.rename(columns={"period_end": "PeriodEnd", "form_type": "FormType", "filing_date": "FilingDate"})

    merge_keys = ["ticker", "market", "PeriodEnd", "FormType"]
    if "FilingDate" in base.columns and "FilingDate" in right.columns:
        base["__filing_key"] = _date_key(base["FilingDate"])
        right["__filing_key"] = _date_key(right["FilingDate"])
        merge_keys = [*merge_keys, "__filing_key"]

    extra_value_columns = [column for column in _US_FINANCIALS_EXTRA_MAP if column in right.columns]
    if not extra_value_columns:
        return financials

    right = right[[*merge_keys, *extra_value_columns]].drop_duplicates(subset=merge_keys, keep="last")
    merged = base.merge(right, on=merge_keys, how="left")
    if "__filing_key" in merged.columns:
        merged = merged.drop(columns=["__filing_key"])
    return merged.rename(columns={column: _US_FINANCIALS_EXTRA_MAP[column] for column in extra_value_columns})


def load_financials(
    ticker: str,
    *,
    market: str,
    include_extra: bool = False,
) -> pd.DataFrame | None:
    normalized_ticker = _normalize_ticker(ticker, market)
    df = read_table(
        market,
        "financials_quarterly",
        filters=[("ticker", "==", normalized_ticker), ("market", "==", _normalize_market(market))],
    )
    if df.empty:
        return None

    df = _coerce_datetime_columns(df, ("StatementDate", "PeriodEnd", "PeriodStart", "FilingDate", "AvailableDate", "AcceptedAt"))
    if include_extra and _normalize_market(market) == "us":
        extra = read_table(
            market,
            "financials_quarterly_extra",
            filters=[("ticker", "==", normalized_ticker), ("market", "==", _normalize_market(market))],
        )
        if not extra.empty:
            df = _merge_us_financials_extra(df, extra)

    if "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    if "PeriodEnd" in df.columns:
        df = df.sort_values("PeriodEnd")
    return df.reset_index(drop=True)


def load_derived_factors(
    ticker: str,
    *,
    market: str,
    basis: str | None = None,
) -> pd.DataFrame | None:
    filters: list[tuple[str, str, Any]] = [("ticker", "==", _normalize_ticker(ticker, market)), ("market", "==", _normalize_market(market))]
    if basis:
        filters.append(("basis", "==", str(basis).strip().lower()))

    df = read_table(market, "derived_factors_quarterly", filters=filters)
    if df.empty:
        return None
    df = _coerce_datetime_columns(df, ("period_end", "available_date", "collected_at"))
    if "basis" in df.columns:
        df["basis"] = df["basis"].astype(str).str.strip().str.lower()
    sort_columns = [column for column in ("basis", "period_end", "available_date") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df.reset_index(drop=True)


def load_filings(
    ticker: str,
    *,
    market: str,
) -> pd.DataFrame | None:
    df = read_table(
        market,
        "filings",
        filters=[("ticker", "==", _normalize_ticker(ticker, market)), ("market", "==", _normalize_market(market))],
    )
    if df.empty:
        return None
    df = _coerce_datetime_columns(df, ("period_end", "report_date", "available_date", "filing_date", "accepted_at"))
    sort_columns = [column for column in ("filing_date", "accepted_at", "accession") if column in df.columns]
    ascending = [False] * len(sort_columns)
    if sort_columns:
        df = df.sort_values(sort_columns, ascending=ascending)
    return df.reset_index(drop=True)


def load_filings_all(*, market: str) -> pd.DataFrame:
    df = read_table(market, "filings")
    if df.empty:
        return pd.DataFrame()
    df = _coerce_datetime_columns(df, ("period_end", "report_date", "available_date", "filing_date", "accepted_at"))
    sort_columns = [column for column in ("ticker", "filing_date", "accession") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df.reset_index(drop=True)


def load_filings_for_tickers(tickers: list[str], *, market: str) -> pd.DataFrame:
    normalized = _normalize_tickers(tickers, market)
    if not normalized:
        return pd.DataFrame()
    df = read_table(
        market,
        "filings",
        filters=[("ticker", "in", normalized), ("market", "==", _normalize_market(market))],
    )
    if df.empty:
        return pd.DataFrame()
    df = _coerce_datetime_columns(df, ("period_end", "report_date", "available_date", "filing_date", "accepted_at"))
    sort_columns = [column for column in ("ticker", "filing_date", "accession") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df.reset_index(drop=True)


def load_ticker_master(ticker: str, *, market: str) -> pd.DataFrame | None:
    df = read_table(
        market,
        "ticker_master",
        filters=[("ticker", "==", _normalize_ticker(ticker, market))],
    )
    if df.empty:
        return None
    return df.head(1).reset_index(drop=True)


def load_ticker_master_all(*, market: str) -> pd.DataFrame:
    df = read_table(market, "ticker_master")
    if df.empty:
        return pd.DataFrame()
    sort_columns = [column for column in ("market_tier", "ticker") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df.reset_index(drop=True)


def load_sec_issuer_registry(ticker: str, *, market: str = "us") -> pd.DataFrame | None:
    df = read_table(
        market,
        "sec_issuer_registry",
        filters=[("ticker", "==", _normalize_ticker(ticker, market)), ("market", "==", _normalize_market(market))],
    )
    if df.empty:
        return None
    df = _coerce_datetime_columns(df, ("collected_at",))
    return df.reset_index(drop=True)


def load_sec_issuer_registry_all(*, market: str = "us") -> pd.DataFrame:
    df = read_table(market, "sec_issuer_registry")
    if df.empty:
        return pd.DataFrame()
    df = _coerce_datetime_columns(df, ("collected_at",))
    sort_columns = [column for column in ("ticker", "market") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df.reset_index(drop=True)


def load_investor_flows(ticker: str, *, market: str = "kr") -> pd.DataFrame | None:
    df = read_table(
        market,
        "investor_flows",
        filters=[("ticker", "==", _normalize_ticker(ticker, market)), ("market", "==", _normalize_market(market))],
    )
    if df.empty:
        return None
    df = _coerce_datetime_columns(df, ("date",))
    sort_columns = [column for column in ("date", "investor_type") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df.reset_index(drop=True)


def load_index_prices(index_code: str, *, market: str = "kr") -> pd.DataFrame | None:
    df = read_table(market, "index_prices", filters=[("index_code", "==", str(index_code).strip())])
    if df.empty:
        return None
    df = _coerce_datetime_columns(df, ("date",))
    if "date" in df.columns:
        df = df.sort_values("date")
    return df.reset_index(drop=True)


def load_dart_corp_master_all(*, market: str = "kr") -> pd.DataFrame:
    df = read_table(market, "dart_corp_master")
    if df.empty:
        return pd.DataFrame()
    if "corp_code" in df.columns:
        df = df.sort_values("corp_code")
    return df.reset_index(drop=True)


def load_ksic_dim_all(*, market: str = "kr") -> pd.DataFrame:
    df = read_table(market, "ksic_dim")
    if df.empty:
        return pd.DataFrame()
    sort_columns = [column for column in ("depth", "ksic_code") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df.reset_index(drop=True)


def load_dart_financials_raw_all(*, market: str = "kr") -> pd.DataFrame:
    df = read_table(market, "dart_financials_raw")
    if df.empty:
        return pd.DataFrame()
    sort_columns = [column for column in ("ticker", "bsns_year", "reprt_code", "account_key") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df.reset_index(drop=True)


def load_dart_financials_raw_for_ticker(ticker: str, *, market: str = "kr") -> pd.DataFrame:
    df = read_table(
        market,
        "dart_financials_raw",
        filters=[("ticker", "==", _normalize_ticker(ticker, market))],
    )
    if df.empty:
        return pd.DataFrame()
    sort_columns = [column for column in ("bsns_year", "reprt_code", "account_key") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df.reset_index(drop=True)


def load_dart_financials_raw_for_tickers(tickers: list[str], *, market: str = "kr") -> pd.DataFrame:
    normalized = _normalize_tickers(tickers, market)
    if not normalized:
        return pd.DataFrame()
    df = read_table(market, "dart_financials_raw", filters=[("ticker", "in", normalized)])
    if df.empty:
        return pd.DataFrame()
    sort_columns = [column for column in ("ticker", "bsns_year", "reprt_code", "account_key") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    return df.reset_index(drop=True)
