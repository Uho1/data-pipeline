from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from market_data.db_router import normalize_kr_ticker

_PREFERRED_RE = re.compile(r"(?:우|[0-9]우)(?:[A-Z])?$")
_PREFERRED_ALT_RE = re.compile(r"(우선주|신형우선주|전환우선주)")
_NON_COMMON_TOKENS = (
    "스팩",
    "리츠",
    "REIT",
    "ETF",
    "ETN",
    "ELW",
)
_INVESTOR_TYPE_MAP = {
    "기관합계": "institution_total",
    "기타법인": "other_corporation",
    "개인": "individual",
    "외국인합계": "foreign_total",
    "전체": "all",
    "금융투자": "financial_investment",
    "보험": "insurance",
    "투신": "investment_trust",
    "사모": "private_equity",
    "은행": "bank",
    "연기금": "pension",
    "기타금융": "other_financial",
    "기타외국인": "other_foreign",
}


def normalize_kr_tickers(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        out.append(normalize_kr_ticker(text))
    return list(dict.fromkeys(out))


def is_common_stock_name(name: str | None) -> tuple[bool, str]:
    text = str(name or "").strip()
    upper = text.upper()
    if not text:
        return False, "missing_name"
    if any(token in upper for token in _NON_COMMON_TOKENS):
        return False, "explicit_non_common_token"
    if "스팩" in text:
        return False, "spac_name"
    if "리츠" in text:
        return False, "reit_name"
    if _PREFERRED_ALT_RE.search(text):
        return False, "preferred_name"
    if _PREFERRED_RE.search(text):
        return False, "preferred_suffix"
    return True, ""


def normalize_investor_type(label: str) -> str:
    text = str(label or "").strip()
    if text in _INVESTOR_TYPE_MAP:
        return _INVESTOR_TYPE_MAP[text]
    ascii_key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return ascii_key or "unknown"


def to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date
