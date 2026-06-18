from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from market_data.db_router import normalize_kr_ticker
from market_data.krx.normalize import normalize_investor_type
from market_data.utils import now_utc_iso, retry_call


def _require_pykrx():
    try:
        from pykrx import stock
    except ImportError as exc:
        raise RuntimeError(
            "pykrx is required for KRX ingest. Install it first: pip install pykrx"
        ) from exc
    return stock


def _fetch_metric(
    fetcher,
    start: str,
    end: str,
    ticker: str,
    metric_kind: str,
    *,
    detail: bool,
) -> pd.DataFrame:
    try:
        return fetcher(start, end, ticker, on=metric_kind, detail=detail)
    except TypeError:
        return fetcher(start, end, ticker, on=metric_kind)


def _melt(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "investor_type_label", value_name])
    out = df.copy().reset_index()
    date_col = out.columns[0]
    out = out.rename(columns={date_col: "date"})
    out = out.melt(id_vars=["date"], var_name="investor_type_label", value_name=value_name)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"])
    return out


def _fetch_pykrx_flow_frame(
    *,
    ticker_code: str,
    start: str,
    end: str,
    ticker_name: str | None,
    market_tier: str | None,
    detail: bool,
) -> pd.DataFrame:
    stock = _require_pykrx()

    buy_vol = _melt(_fetch_metric(stock.get_market_trading_volume_by_date, start, end, ticker_code, "매수", detail=detail), "buy_volume")
    sell_vol = _melt(_fetch_metric(stock.get_market_trading_volume_by_date, start, end, ticker_code, "매도", detail=detail), "sell_volume")
    net_vol = _melt(_fetch_metric(stock.get_market_trading_volume_by_date, start, end, ticker_code, "순매수", detail=detail), "net_volume")
    buy_val = _melt(_fetch_metric(stock.get_market_trading_value_by_date, start, end, ticker_code, "매수", detail=detail), "buy_value")
    sell_val = _melt(_fetch_metric(stock.get_market_trading_value_by_date, start, end, ticker_code, "매도", detail=detail), "sell_value")
    net_val = _melt(_fetch_metric(stock.get_market_trading_value_by_date, start, end, ticker_code, "순매수", detail=detail), "net_value")

    out = buy_vol.merge(sell_vol, on=["date", "investor_type_label"], how="outer")
    out = out.merge(net_vol, on=["date", "investor_type_label"], how="outer")
    out = out.merge(buy_val, on=["date", "investor_type_label"], how="outer")
    out = out.merge(sell_val, on=["date", "investor_type_label"], how="outer")
    out = out.merge(net_val, on=["date", "investor_type_label"], how="outer")
    if out.empty:
        return out

    out["ticker"] = ticker_code
    out["market"] = "kr"
    out["ticker_name"] = ticker_name
    out["market_tier"] = market_tier
    out["investor_type"] = out["investor_type_label"].map(normalize_investor_type)
    out["collected_at"] = pd.Timestamp(now_utc_iso())
    return out[
        [
            "date",
            "ticker",
            "market",
            "ticker_name",
            "market_tier",
            "investor_type",
            "investor_type_label",
            "buy_volume",
            "sell_volume",
            "net_volume",
            "buy_value",
            "sell_value",
            "net_value",
            "collected_at",
        ]
    ].sort_values(["date", "investor_type"]).reset_index(drop=True)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        columns = []
        for col in out.columns.to_flat_index():
            parts = [str(part).strip() for part in col if str(part).strip() and "Unnamed" not in str(part)]
            columns.append("|".join(parts) if parts else "")
        out.columns = columns
    else:
        out.columns = [str(col).strip() for col in out.columns]
    return out


def _find_naver_investor_table(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(html), flavor="lxml")
    except ValueError:
        return pd.DataFrame()
    for table in tables:
        flattened = _flatten_columns(table)
        joined = " ".join(flattened.columns)
        if "날짜" in joined and "기관" in joined and "외국인" in joined and "순매매량" in joined:
            return flattened
    return pd.DataFrame()


def _to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )


def _fetch_naver_flow_frame(
    *,
    ticker_code: str,
    start: str,
    end: str,
    ticker_name: str | None,
    market_tier: str | None,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return pd.DataFrame()
    start_date = start_ts.date()
    end_date = end_ts.date()

    rows: list[pd.DataFrame] = []
    seen_dates: set[object] = set()
    for page in range(1, 401):
        response = retry_call(
            lambda: requests.get(
                "https://finance.naver.com/item/frgn.naver",
                params={"code": ticker_code, "page": page},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            ),
            retries=2,
            backoff_base=0.5,
            label=f"naver-investor-{ticker_code}-p{page}",
        )
        response.raise_for_status()
        page_frame = _find_naver_investor_table(response.text)
        if page_frame.empty:
            break

        rename_map: dict[str, str] = {}
        for column in page_frame.columns:
            if "날짜" in column:
                rename_map[column] = "date"
            elif column.startswith("종가"):
                rename_map[column] = "close"
            elif column.startswith("거래량"):
                rename_map[column] = "volume"
            elif "기관" in column and "순매매량" in column:
                rename_map[column] = "institution_total_net_volume"
            elif "외국인" in column and "순매매량" in column:
                rename_map[column] = "foreign_total_net_volume"
        page_frame = page_frame.rename(columns=rename_map)
        required = {"date", "close", "volume", "institution_total_net_volume", "foreign_total_net_volume"}
        if not required.issubset(set(page_frame.columns)):
            break

        page_frame = page_frame[["date", "close", "volume", "institution_total_net_volume", "foreign_total_net_volume"]].copy()
        page_frame["date"] = pd.to_datetime(page_frame["date"], errors="coerce", format="%Y.%m.%d").dt.date
        page_frame["close"] = _to_number(page_frame["close"])
        page_frame["volume"] = _to_number(page_frame["volume"])
        page_frame["institution_total_net_volume"] = _to_number(page_frame["institution_total_net_volume"])
        page_frame["foreign_total_net_volume"] = _to_number(page_frame["foreign_total_net_volume"])
        page_frame = page_frame.dropna(subset=["date"])
        page_frame = page_frame.loc[(page_frame["date"] >= start_date) & (page_frame["date"] <= end_date)].copy()
        page_frame = page_frame.loc[~page_frame["date"].isin(seen_dates)].copy()
        if not page_frame.empty:
            seen_dates.update(page_frame["date"].tolist())
            rows.append(page_frame)

        full_page = _find_naver_investor_table(response.text)
        if full_page.empty:
            break
        full_page = full_page.rename(columns=rename_map)
        if "date" not in full_page.columns:
            break
        full_page["date"] = pd.to_datetime(full_page["date"], errors="coerce", format="%Y.%m.%d").dt.date
        full_page = full_page.dropna(subset=["date"])
        if full_page.empty or full_page["date"].min() < start_date:
            break

    if not rows:
        return pd.DataFrame()

    base = pd.concat(rows, ignore_index=True, sort=False)
    base = base.drop_duplicates(subset=["date"], keep="first").sort_values("date")

    rows: list[dict[str, object]] = []
    for item in base.to_dict("records"):
        rows.append(
            {
                "date": item["date"],
                "investor_type_label": "기관합계",
                "buy_volume": pd.NA,
                "sell_volume": pd.NA,
                "net_volume": item["institution_total_net_volume"],
                "buy_value": pd.NA,
                "sell_value": pd.NA,
                "net_value": pd.NA,
            }
        )
        rows.append(
            {
                "date": item["date"],
                "investor_type_label": "외국인합계",
                "buy_volume": pd.NA,
                "sell_volume": pd.NA,
                "net_volume": item["foreign_total_net_volume"],
                "buy_value": pd.NA,
                "sell_value": pd.NA,
                "net_value": pd.NA,
            }
        )
        rows.append(
            {
                "date": item["date"],
                "investor_type_label": "전체",
                "buy_volume": item["volume"],
                "sell_volume": item["volume"],
                "net_volume": 0.0,
                "buy_value": pd.NA,
                "sell_value": pd.NA,
                "net_value": 0.0,
            }
        )

    out = pd.DataFrame(rows)
    out["ticker"] = ticker_code
    out["market"] = "kr"
    out["ticker_name"] = ticker_name
    out["market_tier"] = market_tier
    out["investor_type"] = out["investor_type_label"].map(normalize_investor_type)
    out["collected_at"] = pd.Timestamp(now_utc_iso())
    return out[
        [
            "date",
            "ticker",
            "market",
            "ticker_name",
            "market_tier",
            "investor_type",
            "investor_type_label",
            "buy_volume",
            "sell_volume",
            "net_volume",
            "buy_value",
            "sell_value",
            "net_value",
            "collected_at",
        ]
    ].sort_values(["date", "investor_type"]).reset_index(drop=True)


def fetch_investor_flow_frame(
    *,
    ticker: str,
    start: str,
    end: str,
    ticker_name: str | None = None,
    market_tier: str | None = None,
    detail: bool = True,
) -> pd.DataFrame:
    ticker_code = normalize_kr_ticker(ticker)
    try:
        out = _fetch_pykrx_flow_frame(
            ticker_code=ticker_code,
            start=start,
            end=end,
            ticker_name=ticker_name,
            market_tier=market_tier,
            detail=detail,
        )
        if out is not None and not out.empty:
            return out
    except Exception:
        pass

    return _fetch_naver_flow_frame(
        ticker_code=ticker_code,
        start=start,
        end=end,
        ticker_name=ticker_name,
        market_tier=market_tier,
    )
