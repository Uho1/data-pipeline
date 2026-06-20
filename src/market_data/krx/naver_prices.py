"""네이버 종가 + DART 발행주식수로 KR 시총 계산.

pykrx가 KRX 로그인(KRX_ID/KRX_PW) 필수화로 막혀서, 시총을
네이버 금융 일별 종가 × DART 주식총수(stockTotqySttus) 발행주식수로 계산한다.

- 종가: 네이버 fchart (10년+ 일별 OHLCV)
- 발행주식수: DART stockTotqySttus 사업보고서(11011) 연도별 보통주 발행총수(istc_totqy)
  (분기보고서엔 발행수가 비어 있어 연도값을 해당 연도 전체에 적용; 당해는 직전 연도)
"""
from __future__ import annotations

import re

import pandas as pd
import requests

from market_data.kr_dart.client import _load_all_api_keys

_NAVER_CHART = "https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count={count}&requestType=0"
_DART_SHARES = "https://opendart.fss.or.kr/api/stockTotqySttus.json"
_UA = {"User-Agent": "Mozilla/5.0"}
_SHARES_REPORT_CODES = ("11011",)  # 사업보고서만 발행수 제공 (분기보고서는 '-')


def fetch_naver_daily(ticker_code: str, count: int = 2700) -> pd.DataFrame | None:
    """네이버 fchart 일별 OHLCV. index=date(naive), cols=[Open,High,Low,Close,Volume]."""
    url = _NAVER_CHART.format(code=str(ticker_code).zfill(6), count=count)
    try:
        r = requests.get(url, headers=_UA, timeout=20)
        r.raise_for_status()
    except Exception:
        return None
    rows: list[tuple] = []
    for m in re.findall(r'<item data="([^"]+)"', r.text):
        p = m.split("|")
        if len(p) >= 6:
            try:
                rows.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
            except ValueError:
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "Open", "High", "Low", "Close", "Volume"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").sort_index()


def fetch_shares_by_year(corp_code: str, years: list[int], *, key: str | None = None) -> dict[int, int]:
    """DART stockTotqySttus 연도별 보통주 발행총수(istc_totqy). 반환 {year: shares}."""
    if key is None:
        keys = _load_all_api_keys()
        key = keys[0] if keys else None
    if not key:
        return {}
    out: dict[int, int] = {}
    for y in years:
        for rc in _SHARES_REPORT_CODES:
            try:
                j = requests.get(_DART_SHARES, params={
                    "crtfc_key": key, "corp_code": corp_code,
                    "bsns_year": str(y), "reprt_code": rc,
                }, timeout=15).json()
            except Exception:
                continue
            if str(j.get("status")) != "000":
                continue
            for it in j.get("list", []):
                if str(it.get("se", "")).strip() == "보통주":
                    raw = str(it.get("istc_totqy", "")).replace(",", "").strip()
                    if raw and raw != "-":
                        try:
                            out[y] = int(raw)
                        except ValueError:
                            pass
                    break
            if y in out:
                break
    return out


def market_cap_frame(ticker_code: str, corp_code: str, *, count: int = 2700, key: str | None = None) -> pd.DataFrame | None:
    """일별 종가 + 시총. cols=[Open,High,Low,Close,Volume,SharesOutstanding,MarketCap]."""
    daily = fetch_naver_daily(ticker_code, count=count)
    if daily is None or daily.empty:
        return None
    years = sorted({int(d.year) for d in daily.index})
    # 네이버 종가는 수정주가(액면분할 반영, 현재 기준)이라 발행주식수도 최신 기준으로 통일.
    # rate limit 고려: 전체 연도가 아니라 최신 사업보고서 발행수만 조회(종목당 1~2회).
    if key is None:
        keys = _load_all_api_keys()
        key = keys[0] if keys else None
    latest_shares = None
    for y in range(max(years), max(years) - 6, -1):
        s = fetch_shares_by_year(corp_code, [y], key=key)
        if y in s:
            latest_shares = s[y]
            break
    if not latest_shares:
        return None

    out = daily.copy()
    out["SharesOutstanding"] = latest_shares
    out["MarketCap"] = out["Close"] * latest_shares
    return out
