from __future__ import annotations

import pandas as pd

from market_data.utils import now_utc_iso

_INDEX_ALIASES = {
    "KOSPI": ("코스피", "KOSPI"),
    "KOSDAQ": ("코스닥", "KOSDAQ"),
    "KOSPI200": ("코스피 200", "코스피200", "KOSPI 200", "KOSPI200", "KPI200"),
}

_INDEX_FALLBACKS = {
    "KOSPI": {"symbol": "^KS11", "code_aliases": {"KOSPI", "1001"}},
    "KOSDAQ": {"symbol": "^KQ11", "code_aliases": {"KOSDAQ", "2001"}},
    "KOSPI200": {"symbol": "^KS200", "code_aliases": {"KOSPI200", "1028", "KPI200"}},
}


def _require_pykrx():
    try:
        from pykrx import stock
    except ImportError as exc:
        raise RuntimeError(
            "pykrx is required for KRX ingest. Install it first: pip install pykrx"
        ) from exc
    return stock


def _require_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for KR index fallback. Install it first: pip install yfinance"
        ) from exc
    return yf


def _flatten_download(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(levels[0]) for levels in out.columns.to_flat_index()]
    return out


def _fallback_key(index_code: str, index_name: str | None = None) -> str | None:
    candidates = {str(index_code).strip(), str(index_name or "").strip()}
    for key, meta in _INDEX_FALLBACKS.items():
        if candidates & meta["code_aliases"]:
            return key
    return None


def resolve_representative_indices(as_of: str | None = None) -> dict[str, tuple[str, str]]:
    discovered: dict[str, tuple[str, str]] = {}
    try:
        stock = _require_pykrx()
        date_text = str(as_of or pd.Timestamp.today().strftime("%Y%m%d"))
        for market_tier in ("KOSPI", "KOSDAQ"):
            try:
                codes = stock.get_index_ticker_list(date_text, market=market_tier)
            except Exception:
                continue
            for index_code in codes:
                name = str(stock.get_index_ticker_name(index_code)).strip()
                for target, aliases in _INDEX_ALIASES.items():
                    if target in discovered:
                        continue
                    if any(alias.lower() in name.lower() for alias in aliases):
                        discovered[target] = (str(index_code), name)
    except Exception:
        pass

    for target in _INDEX_FALLBACKS:
        discovered.setdefault(target, (target, target))
    return discovered


def _fetch_pykrx_frame(
    *,
    index_code: str,
    start: str,
    end: str,
    index_name: str | None,
) -> pd.DataFrame:
    stock = _require_pykrx()
    raw = stock.get_index_ohlcv_by_date(start, end, index_code)
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = raw.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    out.index.name = "date"
    out = out.rename(
        columns={
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
            "거래대금": "traded_value",
            "등락률": "pct_change",
        }
    )
    out["index_code"] = str(index_code)
    out["index_name"] = index_name
    out["market"] = "kr"
    out["price_change"] = pd.to_numeric(out.get("close"), errors="coerce").diff()
    out["pct_change"] = pd.to_numeric(out.get("pct_change"), errors="coerce")
    out["collected_at"] = pd.Timestamp(now_utc_iso())
    return out.reset_index()


def _fetch_yfinance_frame(
    *,
    fallback_key: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    yf = _require_yfinance()
    symbol = _INDEX_FALLBACKS[fallback_key]["symbol"]
    end_dt = pd.to_datetime(end, errors="coerce")
    end_next = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if pd.notna(end_dt) else end
    raw = yf.download(
        symbol,
        start=str(start),
        end=end_next,
        progress=False,
        auto_adjust=False,
    )
    raw = _flatten_download(raw)
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = raw.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    out.index.name = "date"
    out = out.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    out["index_code"] = fallback_key
    out["index_name"] = fallback_key
    out["market"] = "kr"
    out["traded_value"] = pd.NA
    out["price_change"] = pd.to_numeric(out.get("close"), errors="coerce").diff()
    out["pct_change"] = pd.to_numeric(out.get("close"), errors="coerce").pct_change() * 100.0
    out["collected_at"] = pd.Timestamp(now_utc_iso())
    return out.reset_index()


def fetch_index_price_frame(
    *,
    index_code: str,
    start: str,
    end: str,
    index_name: str | None = None,
) -> pd.DataFrame:
    try:
        out = _fetch_pykrx_frame(index_code=index_code, start=start, end=end, index_name=index_name)
        if out is not None and not out.empty:
            return out
    except Exception:
        pass

    fallback_key = _fallback_key(index_code=index_code, index_name=index_name)
    if fallback_key is None:
        return pd.DataFrame()
    return _fetch_yfinance_frame(fallback_key=fallback_key, start=start, end=end)
