from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from market_data.config import KRX_CORP_LIST_URL
from market_data.krx.normalize import is_common_stock_name, normalize_kr_tickers
from market_data.utils import now_utc_iso, retry_call

_MARKET_TIER_MAP = {
    "유가": "KOSPI",
    "코스피": "KOSPI",
    "KOSPI": "KOSPI",
    "코스닥": "KOSDAQ",
    "KOSDAQ": "KOSDAQ",
    "코넥스": "KONEX",
    "KONEX": "KONEX",
}


def _clean_company_name(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "nat"}:
        return None
    if "Empty DataFrame" in text and "Columns:" in text and "Index:" in text:
        return None
    return text


def _require_pykrx():
    try:
        from pykrx import stock
    except ImportError as exc:
        raise RuntimeError(
            "pykrx is required for KRX ingest. Install it first: pip install pykrx"
        ) from exc
    return stock


def _today_krx() -> str:
    return pd.Timestamp.today().strftime("%Y%m%d")


def fetch_krx_corp_list() -> pd.DataFrame:
    response = retry_call(
        lambda: requests.get(KRX_CORP_LIST_URL, timeout=20),
        retries=3,
        backoff_base=1.0,
        label="krx-corp-list",
    )
    response.raise_for_status()
    response.encoding = "euc-kr"
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError("KRX corp list table not found")

    frame = tables[0].copy()
    rename_map = {
        "회사명": "ticker_name",
        "종목코드": "ticker",
        "업종": "industry_name",
        "주요제품": "product_name",
        "상장일": "listed_date",
        "결산월": "fiscal_month",
        "대표자명": "ceo_name",
        "홈페이지": "homepage",
        "지역": "region",
        "액면가": "par_value",
        "상장주식수": "shares_outstanding",
        "시장구분": "market_tier",
    }
    out = frame.rename(columns=rename_map).copy()
    if "ticker" not in out.columns:
        raise RuntimeError("KRX corp list missing 종목코드 column")
    out["ticker"] = out["ticker"].astype(str).str.extract(r"(\d{6})", expand=False)
    out = out.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="first")
    for column in (
        "ticker_name",
        "industry_name",
        "product_name",
        "listed_date",
        "fiscal_month",
        "ceo_name",
        "homepage",
        "region",
        "par_value",
        "shares_outstanding",
        "market_tier",
    ):
        if column not in out.columns:
            out[column] = pd.NA
    out["listed_date"] = pd.to_datetime(out["listed_date"], errors="coerce").dt.date
    out["par_value"] = pd.to_numeric(out["par_value"], errors="coerce")
    out["shares_outstanding"] = pd.to_numeric(out["shares_outstanding"], errors="coerce")
    out["market_tier"] = out["market_tier"].astype(str).str.strip().map(lambda x: _MARKET_TIER_MAP.get(x.upper(), _MARKET_TIER_MAP.get(x, x)))
    return out[
        [
            "ticker",
            "ticker_name",
            "industry_name",
            "product_name",
            "listed_date",
            "fiscal_month",
            "ceo_name",
            "homepage",
            "region",
            "par_value",
            "shares_outstanding",
            "market_tier",
        ]
    ].reset_index(drop=True)


def build_ticker_master(
    *,
    as_of: str | None = None,
    tickers: list[str] | None = None,
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
) -> pd.DataFrame:
    stock = _require_pykrx()
    requested = set(normalize_kr_tickers(tickers or []))
    corp_list = fetch_krx_corp_list()
    corp_list["market_tier"] = corp_list.get("market_tier", pd.Series(dtype=object)).astype(str).str.upper()
    out = corp_list.loc[corp_list["market_tier"].isin([str(m).upper() for m in markets])].copy()
    if requested:
        out = out.loc[out["ticker"].astype(str).isin(requested)].copy()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "market",
                "market_tier",
                "ticker_name",
                "short_name",
                "is_common_stock",
                "common_stock_filter_reason",
                "listed_date",
                "delisted_date",
                "shares_outstanding",
                "par_value",
                "industry_name",
                "sector_name",
                "subsector_name",
                "krx_industry_name",
                "induty_code",
                "ksic_name_ko",
                "ksic_name_en",
                "sector_code",
                "subsector_code",
                "classification_source",
                "kind_code",
                "kind_name",
                "dart_corp_code",
                "dart_corp_name",
                "representative_index",
                "source",
                "collected_at",
            ]
        )

    out = out.drop_duplicates(subset=["ticker"], keep="first")
    name_values: list[str | None] = []
    for ticker, fallback_name in zip(out["ticker"], out["ticker_name"], strict=False):
        fallback_text = _clean_company_name(fallback_name)
        try:
            resolved = stock.get_market_ticker_name(str(ticker).strip())
        except Exception:
            resolved = fallback_text
        resolved_text = _clean_company_name(resolved)
        name_values.append(resolved_text or fallback_text)

    out["market"] = "kr"
    out["ticker_name"] = pd.Series(name_values, index=out.index).fillna(out["ticker_name"])
    out["krx_industry_name"] = out.get("industry_name")
    out["industry_name"] = out.get("industry_name")
    out["sector_name"] = out.get("industry_name")
    out["subsector_name"] = pd.NA
    out["induty_code"] = pd.NA
    out["ksic_name_ko"] = pd.NA
    out["ksic_name_en"] = pd.NA
    out["sector_code"] = pd.NA
    out["subsector_code"] = pd.NA
    out["short_name"] = out["ticker_name"]
    out["delisted_date"] = pd.NaT
    out["kind_code"] = pd.NA
    out["kind_name"] = pd.NA
    out["dart_corp_code"] = pd.NA
    out["dart_corp_name"] = pd.NA
    out["representative_index"] = pd.NA
    common_rows = out["ticker_name"].map(is_common_stock_name)
    out["is_common_stock"] = common_rows.map(lambda item: bool(item[0]))
    out["common_stock_filter_reason"] = common_rows.map(lambda item: str(item[1] or ""))
    out["source"] = "pykrx+krx_corp_list"
    out["classification_source"] = "krx_corp_list"
    out["collected_at"] = pd.Timestamp(now_utc_iso())
    return out[
        [
            "ticker",
            "market",
            "market_tier",
            "ticker_name",
            "short_name",
            "is_common_stock",
            "common_stock_filter_reason",
            "listed_date",
            "delisted_date",
            "shares_outstanding",
            "par_value",
            "industry_name",
            "sector_name",
            "subsector_name",
            "krx_industry_name",
            "induty_code",
            "ksic_name_ko",
            "ksic_name_en",
            "sector_code",
            "subsector_code",
            "classification_source",
            "kind_code",
            "kind_name",
            "dart_corp_code",
            "dart_corp_name",
            "representative_index",
            "source",
            "collected_at",
        ]
    ].sort_values(["market_tier", "ticker"]).reset_index(drop=True)
