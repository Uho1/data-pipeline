from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from market_data.config import DATA_DIR, PRICES_DIR
from market_data.sec_financials import SEC_DEFAULT_USER_AGENT, SEC_TICKERS_URL, load_ticker_cik_map
from market_data.utils import ensure_dir, now_utc_iso, retry_call, sanitize_ticker

LOGGER = logging.getLogger(__name__)

SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

SEC_IDENTITY_CACHE_DIR = DATA_DIR / "sec_identity_cache"
SEC_SUBMISSIONS_CACHE_DIR = SEC_IDENTITY_CACHE_DIR / "submissions"
SEC_TICKER_SECTOR_PROXY_DIR = SEC_IDENTITY_CACHE_DIR / "ticker_sector_proxy"
SEC_UNCLASSIFIED_REPORT_PATH = SEC_IDENTITY_CACHE_DIR / "sector_proxy_unclassified_top.csv"

REFERENCE_DIR = DATA_DIR / "reference"
SEC_SIC_MASTER_PATH = REFERENCE_DIR / "sec_sic_master.csv"
SIC_RULES_PATH = REFERENCE_DIR / "sic_to_sector_proxy_rules.csv"
SIC_RULES_TEMPLATE_PATH = REFERENCE_DIR / "sic_to_sector_proxy_rules.template.csv"

DEFAULT_SYMBOL_IDENTITY_OVERRIDES = Path("config") / "symbol_identity_overrides.csv"

DEFAULT_MAPPING_VERSION = "sec_sector_proxy_v1"
DEFAULT_UNCLASSIFIED_L1 = "미분류"
DEFAULT_UNCLASSIFIED_L2 = "미분류"

_SIC_MASTER_LOCK = threading.Lock()


class _SecRateLimiter:
    """SEC EDGAR는 10 req/s 제한. 안전하게 8 req/s로 설정."""

    def __init__(self, max_per_second: float = 8.0) -> None:
        self._min_interval = 1.0 / max(max_per_second, 0.1)
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


_SEC_RATE_LIMITER = _SecRateLimiter(max_per_second=8.0)

PIT_COLUMNS = [
    "ticker",
    "cik",
    "valid_from",
    "valid_to",
    "sec_sic",
    "sec_sic_description",
    "sector_l1_kr",
    "sector_l2_kr",
    "sector_l1_en",
    "sector_l2_en",
    "mapping_source",
    "mapping_confidence",
    "rule_id",
    "mapping_version",
    "source_filing_date",
    "source_acceptance_datetime",
    "updated_at",
    "note",
]

SECTOR_SNAPSHOT_REQUIRED_COLUMNS = [
    "ticker",
    "sector_l1_kr",
    "sector_l2_kr",
    "mapping_source",
    "sec_sic",
    "sec_sic_description",
]


def _coerce_ts(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _coerce_dt(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _ts_to_iso_date(value: Any) -> str:
    ts = _coerce_ts(value)
    return "" if ts is None else ts.date().isoformat()


def _safe_float(value: Any, default: float = np.nan) -> float:
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return float(default)
    return float(num)


def _sec_headers(user_agent: str | None = None) -> dict[str, str]:
    ua = str(user_agent or SEC_DEFAULT_USER_AGENT).strip() or SEC_DEFAULT_USER_AGENT
    return {
        "User-Agent": ua,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }


def _normalize_tickers(symbols: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return sorted({str(s).strip().upper() for s in symbols if str(s).strip()})


def _normalize_sic_code(value: Any) -> str:
    if value is None:
        return ""

    # Numeric inputs from pandas/numpy should keep numeric semantics.
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        num = int(value)
        return f"{num:04d}" if num >= 0 else ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        num = int(value)
        return f"{num:04d}" if num >= 0 else ""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""

    # Decimal-like strings such as "3670.0" / "2834.00" should map to 3670/2834.
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        try:
            num = int(float(text))
            return f"{num:04d}" if num >= 0 else ""
        except Exception:
            return ""

    # If the string contains a decimal token inside text, parse token first to avoid
    # "3670.0" -> "36700" style digit-concatenation errors.
    token = re.search(r"[+-]?\d+\.\d+", text)
    if token is not None:
        try:
            num = int(float(token.group(0)))
            return f"{num:04d}" if num >= 0 else ""
        except Exception:
            pass

    # Fallback for mixed strings: extract digits conservatively.
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return ""
    try:
        num = int(digits)
        return f"{num:04d}" if num >= 0 else ""
    except Exception:
        return ""


def _estimate_market_cap_proxy(ticker: str, market: str = "us", offline_mode: bool = False) -> float:
    """Best-effort estimate for unclassified ranking report.

    market_cap ~= latest_adj_close * latest_shares_from_sec_quarterly_cache
    """

    from market_data.sec_term_reader import load_ticker_quarterly_cache

    sym = str(ticker).strip().upper()
    try:
        from market_data.reader import load_price_dataframe
        px, _ = load_price_dataframe(ticker=sym, market=market)
    except Exception:
        return float("nan")

    if px.empty:
        return float("nan")

    close_col = "Adj Close" if "Adj Close" in px.columns else ("Close" if "Close" in px.columns else None)
    if close_col is None:
        return float("nan")

    latest_close = pd.to_numeric(px[close_col], errors="coerce").dropna()
    if latest_close.empty:
        return float("nan")
    close_val = float(latest_close.iloc[-1])

    q = load_ticker_quarterly_cache(sym, rebuild_if_stale=not offline_mode)
    if q is None or q.empty:
        return float("nan")

    share_cols = [
        "Shares",
        "diluted_shares",
        "basic_shares",
        "Diluted Shares",
        "Basic Shares",
    ]
    shares = pd.Series(dtype=float)
    for col in share_cols:
        if col in q.columns:
            cand = pd.to_numeric(q[col], errors="coerce").dropna()
            if not cand.empty:
                shares = cand
                break
    if shares.empty:
        return float("nan")

    return float(close_val * float(shares.iloc[-1]))


def _default_rule_template_rows() -> list[dict[str, Any]]:
    """Conservative seed rules; ambiguous SICs intentionally left for fallback(unclassified)."""

    rows: list[dict[str, Any]] = [
        {
            "rule_id": "sic_exact_1311",
            "priority": 10,
            "enabled": 1,
            "match_type": "exact",
            "sic_code_start": 1311,
            "sic_code_end": 1311,
            "sic_keyword": "",
            "sector_l1_kr": "에너지",
            "sector_l2_kr": "에너지",
            "confidence": 0.96,
            "note": "Crude Petroleum & Natural Gas",
        },
        {
            "rule_id": "sic_range_1380_1389",
            "priority": 11,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 1380,
            "sic_code_end": 1389,
            "sic_keyword": "",
            "sector_l1_kr": "에너지",
            "sector_l2_kr": "에너지",
            "confidence": 0.90,
            "note": "Oil/Gas field services",
        },
        {
            "rule_id": "sic_range_2800_2899",
            "priority": 20,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 2800,
            "sic_code_end": 2899,
            "sic_keyword": "",
            "sector_l1_kr": "소재",
            "sector_l2_kr": "화학 소재",
            "confidence": 0.88,
            "note": "Chemicals",
        },
        {
            "rule_id": "sic_range_3200_3299",
            "priority": 21,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 3200,
            "sic_code_end": 3299,
            "sic_keyword": "",
            "sector_l1_kr": "소재",
            "sector_l2_kr": "건축 소재",
            "confidence": 0.84,
            "note": "Stone/Clay/Glass/Concrete",
        },
        {
            "rule_id": "sic_range_3300_3499",
            "priority": 22,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 3300,
            "sic_code_end": 3499,
            "sic_keyword": "",
            "sector_l1_kr": "소재",
            "sector_l2_kr": "기초 소재",
            "confidence": 0.82,
            "note": "Primary & fabricated metals",
        },
        {
            "rule_id": "sic_range_2600_2699",
            "priority": 23,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 2600,
            "sic_code_end": 2699,
            "sic_keyword": "",
            "sector_l1_kr": "소재",
            "sector_l2_kr": "제지 및 임산물",
            "confidence": 0.82,
            "note": "Paper & forest",
        },
        {
            "rule_id": "sic_range_3500_3599",
            "priority": 30,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 3500,
            "sic_code_end": 3599,
            "sic_keyword": "",
            "sector_l1_kr": "산업재",
            "sector_l2_kr": "자본재",
            "confidence": 0.86,
            "note": "Industrial machinery",
        },
        {
            "rule_id": "sic_range_3700_3799",
            "priority": 31,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 3700,
            "sic_code_end": 3799,
            "sic_keyword": "",
            "sector_l1_kr": "경기 소비재",
            "sector_l2_kr": "자동차 및 부품",
            "confidence": 0.84,
            "note": "Transportation equipment",
        },
        {
            "rule_id": "sic_range_4000_4799",
            "priority": 32,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 4000,
            "sic_code_end": 4799,
            "sic_keyword": "",
            "sector_l1_kr": "산업재",
            "sector_l2_kr": "운송",
            "confidence": 0.84,
            "note": "Transportation services",
        },
        {
            "rule_id": "sic_range_7300_7399",
            "priority": 33,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 7300,
            "sic_code_end": 7399,
            "sic_keyword": "",
            "sector_l1_kr": "산업재",
            "sector_l2_kr": "전문 서비스",
            "confidence": 0.78,
            "note": "Business services",
        },
        {
            "rule_id": "sic_range_3570_3579",
            "priority": 40,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 3570,
            "sic_code_end": 3579,
            "sic_keyword": "",
            "sector_l1_kr": "IT",
            "sector_l2_kr": "하드웨어 및 장비",
            "confidence": 0.90,
            "note": "Computer hardware",
        },
        {
            "rule_id": "sic_range_3660_3669",
            "priority": 41,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 3660,
            "sic_code_end": 3669,
            "sic_keyword": "",
            "sector_l1_kr": "IT",
            "sector_l2_kr": "하드웨어 및 장비",
            "confidence": 0.88,
            "note": "Communications equipment",
        },
        {
            "rule_id": "sic_range_3670_3679",
            "priority": 42,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 3670,
            "sic_code_end": 3679,
            "sic_keyword": "",
            "sector_l1_kr": "IT",
            "sector_l2_kr": "반도체 및 디스플레이",
            "confidence": 0.94,
            "note": "Semiconductor devices",
        },
        {
            "rule_id": "sic_range_7370_7379",
            "priority": 43,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 7370,
            "sic_code_end": 7379,
            "sic_keyword": "",
            "sector_l1_kr": "IT",
            "sector_l2_kr": "소프트웨어",
            "confidence": 0.92,
            "note": "Software/services",
        },
        {
            "rule_id": "sic_range_5000_5199",
            "priority": 50,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 5000,
            "sic_code_end": 5199,
            "sic_keyword": "",
            "sector_l1_kr": "필수 소비재",
            "sector_l2_kr": "필수 소비재 유통 및 소매",
            "confidence": 0.74,
            "note": "Durable wholesale",
        },
        {
            "rule_id": "sic_range_5200_5999",
            "priority": 51,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 5200,
            "sic_code_end": 5999,
            "sic_keyword": "",
            "sector_l1_kr": "경기 소비재",
            "sector_l2_kr": "경기 소비재 유통 및 소매",
            "confidence": 0.76,
            "note": "Retail",
        },
        {
            "rule_id": "sic_range_2000_2199",
            "priority": 52,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 2000,
            "sic_code_end": 2199,
            "sic_keyword": "",
            "sector_l1_kr": "필수 소비재",
            "sector_l2_kr": "음식료 및 담배",
            "confidence": 0.90,
            "note": "Food/Tobacco",
        },
        {
            "rule_id": "sic_range_2200_2399",
            "priority": 53,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 2200,
            "sic_code_end": 2399,
            "sic_keyword": "",
            "sector_l1_kr": "경기 소비재",
            "sector_l2_kr": "의류 사치품 레저용품",
            "confidence": 0.84,
            "note": "Textile/Apparel",
        },
        {
            "rule_id": "sic_range_2830_2839",
            "priority": 60,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 2830,
            "sic_code_end": 2839,
            "sic_keyword": "",
            "sector_l1_kr": "헬스케어",
            "sector_l2_kr": "제약 및 생명공학",
            "confidence": 0.95,
            "note": "Drugs",
        },
        {
            "rule_id": "sic_range_3840_3859",
            "priority": 61,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 3840,
            "sic_code_end": 3859,
            "sic_keyword": "",
            "sector_l1_kr": "헬스케어",
            "sector_l2_kr": "헬스케어 장비 및 서비스",
            "confidence": 0.90,
            "note": "Medical devices",
        },
        {
            "rule_id": "sic_range_8000_8099",
            "priority": 62,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 8000,
            "sic_code_end": 8099,
            "sic_keyword": "",
            "sector_l1_kr": "헬스케어",
            "sector_l2_kr": "헬스케어 장비 및 서비스",
            "confidence": 0.86,
            "note": "Health services",
        },
        {
            "rule_id": "sic_range_6000_6099",
            "priority": 70,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 6000,
            "sic_code_end": 6099,
            "sic_keyword": "",
            "sector_l1_kr": "금융",
            "sector_l2_kr": "은행",
            "confidence": 0.94,
            "note": "Banks",
        },
        {
            "rule_id": "sic_range_6100_6199",
            "priority": 71,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 6100,
            "sic_code_end": 6199,
            "sic_keyword": "",
            "sector_l1_kr": "금융",
            "sector_l2_kr": "자본시장 서비스",
            "confidence": 0.88,
            "note": "Credit agencies",
        },
        {
            "rule_id": "sic_range_6200_6299",
            "priority": 72,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 6200,
            "sic_code_end": 6299,
            "sic_keyword": "",
            "sector_l1_kr": "금융",
            "sector_l2_kr": "자본시장 서비스",
            "confidence": 0.91,
            "note": "Security brokers",
        },
        {
            "rule_id": "sic_range_6300_6399",
            "priority": 73,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 6300,
            "sic_code_end": 6399,
            "sic_keyword": "",
            "sector_l1_kr": "금융",
            "sector_l2_kr": "보험",
            "confidence": 0.92,
            "note": "Insurance",
        },
        {
            "rule_id": "sic_range_6400_6411",
            "priority": 74,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 6400,
            "sic_code_end": 6411,
            "sic_keyword": "",
            "sector_l1_kr": "금융",
            "sector_l2_kr": "보험",
            "confidence": 0.90,
            "note": "Insurance agents",
        },
        {
            "rule_id": "sic_range_6500_6599",
            "priority": 80,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 6500,
            "sic_code_end": 6599,
            "sic_keyword": "",
            "sector_l1_kr": "부동산",
            "sector_l2_kr": "부동산 개발 및 운용",
            "confidence": 0.86,
            "note": "Real estate",
        },
        {
            "rule_id": "sic_keyword_reit",
            "priority": 81,
            "enabled": 1,
            "match_type": "keyword",
            "sic_code_start": "",
            "sic_code_end": "",
            "sic_keyword": "reit|real estate investment trust",
            "sector_l1_kr": "부동산",
            "sector_l2_kr": "리츠",
            "confidence": 0.90,
            "note": "REIT keyword",
        },
        {
            "rule_id": "sic_range_4800_4899",
            "priority": 90,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 4800,
            "sic_code_end": 4899,
            "sic_keyword": "",
            "sector_l1_kr": "통신 및 유틸리티",
            "sector_l2_kr": "통신 서비스",
            "confidence": 0.90,
            "note": "Communication",
        },
        {
            "rule_id": "sic_range_4900_4999",
            "priority": 91,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 4900,
            "sic_code_end": 4999,
            "sic_keyword": "",
            "sector_l1_kr": "통신 및 유틸리티",
            "sector_l2_kr": "유틸리티",
            "confidence": 0.92,
            "note": "Utilities",
        },
        {
            "rule_id": "sic_range_9100_9721",
            "priority": 100,
            "enabled": 1,
            "match_type": "range",
            "sic_code_start": 9100,
            "sic_code_end": 9721,
            "sic_keyword": "",
            "sector_l1_kr": "기타",
            "sector_l2_kr": "정보기관 및 공공서비스",
            "confidence": 0.70,
            "note": "Public administration",
        },
    ]
    return rows


def ensure_sector_proxy_reference_files(
    *,
    template_path: str | Path = SIC_RULES_TEMPLATE_PATH,
    rules_path: str | Path = SIC_RULES_PATH,
    sic_master_path: str | Path = SEC_SIC_MASTER_PATH,
) -> dict[str, str]:
    template = Path(template_path).expanduser()
    rules = Path(rules_path).expanduser()
    master = Path(sic_master_path).expanduser()

    ensure_dir(template.parent)
    ensure_dir(rules.parent)
    ensure_dir(master.parent)

    rows = _default_rule_template_rows()
    template_created = False
    rules_created = False
    master_created = False

    if not template.exists():
        pd.DataFrame(rows).to_csv(template, index=False, encoding="utf-8")
        template_created = True

    if not rules.exists():
        if template.exists():
            shutil.copy2(template, rules)
        else:
            pd.DataFrame(rows).to_csv(rules, index=False, encoding="utf-8")
        rules_created = True

    if not master.exists():
        pd.DataFrame(columns=["sec_sic", "sec_sic_description", "sample_ticker", "sample_cik", "updated_at"]).to_csv(
            master,
            index=False,
            encoding="utf-8",
        )
        master_created = True

    return {
        "template_path": str(template),
        "rules_path": str(rules),
        "sic_master_path": str(master),
        "template_created": str(template_created).lower(),
        "rules_created": str(rules_created).lower(),
        "sic_master_created": str(master_created).lower(),
    }


def load_sector_proxy_rules(
    rules_path: str | Path = SIC_RULES_PATH,
    *,
    debug_sample: bool = False,
    debug_sample_rows: int = 5,
) -> pd.DataFrame:
    ensure_sector_proxy_reference_files(rules_path=rules_path)
    p = Path(rules_path).expanduser()
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(
            p,
            dtype={
                "rule_id": "string",
                "match_type": "string",
                "sic_code_start": "string",
                "sic_code_end": "string",
                "sic_keyword": "string",
                "sector_l1_kr": "string",
                "sector_l2_kr": "string",
                "note": "string",
            },
            keep_default_na=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read SIC mapping rules: {p}") from exc

    if df.empty:
        return df

    out = df.copy()
    required = [
        "rule_id",
        "priority",
        "enabled",
        "match_type",
        "sic_code_start",
        "sic_code_end",
        "sic_keyword",
        "sector_l1_kr",
        "sector_l2_kr",
        "confidence",
        "note",
    ]
    for col in required:
        if col not in out.columns:
            out[col] = np.nan

    out["rule_id"] = out["rule_id"].astype(str).str.strip()
    out["priority"] = pd.to_numeric(out["priority"], errors="coerce").fillna(9999).astype(int)
    out["enabled"] = out["enabled"].map(lambda x: str(x).strip().lower() in {"1", "true", "yes", "y", "on"})
    out["match_type"] = out["match_type"].astype(str).str.strip().str.lower()
    out["sic_keyword"] = out["sic_keyword"].fillna("").astype(str)
    out["sector_l1_kr"] = out["sector_l1_kr"].fillna(DEFAULT_UNCLASSIFIED_L1).astype(str)
    out["sector_l2_kr"] = out["sector_l2_kr"].fillna(DEFAULT_UNCLASSIFIED_L2).astype(str)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)

    out["sic_code_start_norm"] = out["sic_code_start"].map(_normalize_sic_code)
    out["sic_code_end_norm"] = out["sic_code_end"].map(_normalize_sic_code)
    if debug_sample and not out.empty:
        preview_cols = [
            "rule_id",
            "match_type",
            "sic_code_start",
            "sic_code_start_norm",
            "sic_code_end",
            "sic_code_end_norm",
        ]
        sample_n = max(int(debug_sample_rows), 1)
        preview = out[preview_cols].head(sample_n)
        LOGGER.info(
            "SIC rule normalization sample(top=%d):\n%s",
            len(preview),
            preview.to_string(index=False),
        )
    out = out.loc[out["enabled"]].sort_values(["priority", "rule_id"]).reset_index(drop=True)
    return out


def _load_override_rows(overrides_path: str | Path | None = None, market: str | None = None) -> pd.DataFrame:
    path = Path(overrides_path).expanduser() if overrides_path is not None else DEFAULT_SYMBOL_IDENTITY_OVERRIDES
    if not path.exists():
        return pd.DataFrame()

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".json", ".jsn"}:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                df = pd.DataFrame(raw)
            elif isinstance(raw, dict):
                rows = raw.get("rows") if isinstance(raw.get("rows"), list) else list(raw.values())
                df = pd.DataFrame(rows)
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    if market and "market" in out.columns:
        m = out["market"].astype(str).str.lower().str.strip()
        out = out.loc[(m == str(market).lower()) | (m == "")]

    return out.reset_index(drop=True)


def _find_override_rows_for_ticker(
    ticker: str,
    *,
    market: str,
    overrides_path: str | Path | None,
) -> pd.DataFrame:
    df = _load_override_rows(overrides_path=overrides_path, market=market)
    if df.empty:
        return pd.DataFrame()
    sym = str(ticker).strip().upper()
    out = df.loc[df["ticker"] == sym].copy()
    if out.empty:
        return out
    out["first_valid_date"] = pd.to_datetime(out.get("first_valid_date"), errors="coerce").dt.normalize()
    out["last_valid_date"] = pd.to_datetime(out.get("last_valid_date"), errors="coerce").dt.normalize()
    return out.sort_values(["first_valid_date", "last_valid_date"], na_position="last").reset_index(drop=True)


def _resolve_validity_window(
    ticker: str,
    *,
    market: str,
    overrides_path: str | Path | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    from market_data.backtest.validation_symbol_time import load_ticker_validity_ranges

    ranges = load_ticker_validity_ranges(
        [ticker],
        market=market,
        price_root=Path("data") / "prices",
        overrides_path=overrides_path,
    )
    item = ranges.get(str(ticker).upper(), {})
    first_valid = _coerce_ts(item.get("first_valid_date"))
    last_valid = _coerce_ts(item.get("last_valid_date"))
    source = str(item.get("source_used", "missing") or "missing")
    return first_valid, last_valid, source


def fetch_submissions_for_ticker(
    ticker: str,
    *,
    user_agent: str | None = None,
    retries: int = 3,
    backoff_base: float = 1.0,
    force_refresh: bool = False,
    cache_dir: str | Path = SEC_SUBMISSIONS_CACHE_DIR,
) -> tuple[dict[str, Any], int]:
    sym = str(ticker).strip().upper()
    if not sym:
        raise ValueError("ticker is required")

    ensure_dir(Path(cache_dir).expanduser())
    raw_path = Path(cache_dir).expanduser() / f"{sanitize_ticker(sym)}.json"

    cik_map = load_ticker_cik_map(user_agent=user_agent, retries=max(int(retries), 0), backoff=max(float(backoff_base), 0.1))
    cik = cik_map.get(sym)
    if cik is None:
        raise KeyError(f"SEC CIK not found for ticker={sym}")

    if raw_path.exists() and not force_refresh:
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            return payload, int(cik)
        except Exception:
            LOGGER.warning("Failed to parse cached submissions JSON: %s", raw_path)

    url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=int(cik))

    def _fetch() -> dict[str, Any]:
        _SEC_RATE_LIMITER.acquire()
        resp = requests.get(url, headers=_sec_headers(user_agent=user_agent), timeout=30)
        resp.raise_for_status()
        return resp.json()

    payload = retry_call(
        _fetch,
        retries=max(int(retries), 0),
        backoff_base=max(float(backoff_base), 0.1),
        label=f"sec:submissions:{sym}",
    )

    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, int(cik)


def _extract_submission_identity(payload: dict[str, Any], ticker: str, cik: int | None = None) -> dict[str, Any]:
    sym = str(ticker).strip().upper()
    cik_int: int | None = None
    if cik is not None:
        cik_int = int(cik)
    else:
        try:
            raw_cik = payload.get("cik")
            cik_digits = "".join(ch for ch in str(raw_cik) if ch.isdigit())
            if cik_digits:
                cik_int = int(cik_digits)
        except Exception:
            cik_int = None

    sic = _normalize_sic_code(payload.get("sic"))
    sic_desc = str(payload.get("sicDescription") or "").strip()
    entity_name = str(payload.get("name") or payload.get("entityName") or "").strip()

    recent = payload.get("filings", {}).get("recent", {})
    acceptance_dates = pd.to_datetime(pd.Series(recent.get("acceptanceDateTime", [])), errors="coerce").dropna()
    filing_dates = pd.to_datetime(pd.Series(recent.get("filingDate", [])), errors="coerce").dropna()

    source_acceptance = pd.Timestamp(acceptance_dates.min()) if not acceptance_dates.empty else pd.NaT
    source_filing = pd.Timestamp(filing_dates.min()).normalize() if not filing_dates.empty else pd.NaT

    return {
        "ticker": sym,
        "cik": cik_int,
        "sec_sic": sic,
        "sec_sic_description": sic_desc,
        "entity_name": entity_name,
        "source_acceptance_datetime": source_acceptance,
        "source_filing_date": source_filing,
    }


def map_sic_to_sector_proxy(
    sec_sic: str | int | None,
    sic_description: str | None,
    *,
    rules_df: pd.DataFrame | None = None,
    mapping_version: str = DEFAULT_MAPPING_VERSION,
) -> dict[str, Any]:
    rules = rules_df if isinstance(rules_df, pd.DataFrame) else load_sector_proxy_rules()

    sic_norm = _normalize_sic_code(sec_sic)
    sic_int = int(sic_norm) if sic_norm else None
    desc = str(sic_description or "")
    desc_l = desc.lower()

    if not rules.empty:
        for _, row in rules.iterrows():
            match_type = str(row.get("match_type", "")).strip().lower()
            rid = str(row.get("rule_id", "")).strip() or "rule_unknown"

            matched = False
            if match_type == "exact":
                start = _normalize_sic_code(row.get("sic_code_start"))
                if sic_norm and start and sic_norm == start:
                    matched = True
            elif match_type == "range":
                start_s = _normalize_sic_code(row.get("sic_code_start"))
                end_s = _normalize_sic_code(row.get("sic_code_end"))
                if sic_int is not None and start_s and end_s:
                    lo = int(start_s)
                    hi = int(end_s)
                    if lo <= sic_int <= hi:
                        matched = True
            elif match_type == "keyword":
                kw = str(row.get("sic_keyword", "") or "").strip().lower()
                if kw and desc_l:
                    tokens = [t.strip() for t in kw.split("|") if t.strip()]
                    if any(tok in desc_l for tok in tokens):
                        matched = True

            if matched:
                return {
                    "sector_l1_kr": str(row.get("sector_l1_kr") or DEFAULT_UNCLASSIFIED_L1),
                    "sector_l2_kr": str(row.get("sector_l2_kr") or DEFAULT_UNCLASSIFIED_L2),
                    "sector_l1_en": "",
                    "sector_l2_en": "",
                    "mapping_source": f"sic_{match_type}",
                    "mapping_confidence": float(pd.to_numeric(pd.Series([row.get("confidence")]), errors="coerce").fillna(0.5).iloc[0]),
                    "rule_id": rid,
                    "mapping_version": mapping_version,
                    "note": str(row.get("note") or "").strip(),
                }

    return {
        "sector_l1_kr": DEFAULT_UNCLASSIFIED_L1,
        "sector_l2_kr": DEFAULT_UNCLASSIFIED_L2,
        "sector_l1_en": "",
        "sector_l2_en": "",
        "mapping_source": "fallback_unclassified",
        "mapping_confidence": 0.0,
        "rule_id": "fallback_unclassified",
        "mapping_version": mapping_version,
        "note": "No SIC mapping rule matched",
    }


def _interval_to_finite_end(value: Any) -> pd.Timestamp:
    ts = _coerce_ts(value)
    if ts is None:
        return pd.Timestamp("2262-04-11")
    return ts


def _split_rows_with_override(
    rows: list[dict[str, Any]],
    override_row: dict[str, Any],
) -> list[dict[str, Any]]:
    if not rows:
        return [override_row]

    ov_s = _coerce_ts(override_row.get("valid_from"))
    ov_e = _coerce_ts(override_row.get("valid_to"))
    if ov_s is None:
        ov_s = pd.Timestamp("1900-01-01")
    ov_e_f = _interval_to_finite_end(ov_e)

    new_rows: list[dict[str, Any]] = []
    for row in rows:
        s = _coerce_ts(row.get("valid_from")) or pd.Timestamp("1900-01-01")
        e = _coerce_ts(row.get("valid_to"))
        e_f = _interval_to_finite_end(e)

        # no overlap
        if e_f < ov_s or s > ov_e_f:
            new_rows.append(row)
            continue

        # left remainder
        if s < ov_s:
            left = dict(row)
            left["valid_from"] = s
            left["valid_to"] = ov_s - pd.Timedelta(days=1)
            new_rows.append(left)

        # right remainder
        if e_f > ov_e_f:
            right = dict(row)
            right["valid_from"] = ov_e_f + pd.Timedelta(days=1)
            right["valid_to"] = e if e is not None else pd.NaT
            if _coerce_ts(right["valid_from"]) is not None:
                new_rows.append(right)

    new_rows.append(override_row)
    new_rows.sort(key=lambda r: (_coerce_ts(r.get("valid_from")) or pd.Timestamp("1900-01-01"), str(r.get("rule_id", ""))))
    return new_rows


def validate_sector_proxy_pit_intervals(
    pit_df: pd.DataFrame,
    *,
    fail_closed: bool = False,
) -> dict[str, Any]:
    if pit_df is None or pit_df.empty:
        out = {"status": "warn", "issues": [{"issue": "empty_pit"}]}
        if fail_closed:
            raise RuntimeError("Sector PIT is empty")
        return out

    df = pit_df.copy()
    df["valid_from"] = pd.to_datetime(df.get("valid_from"), errors="coerce").dt.normalize()
    df["valid_to"] = pd.to_datetime(df.get("valid_to"), errors="coerce").dt.normalize()

    issues: list[dict[str, Any]] = []

    invalid = df.loc[df["valid_from"].isna()]
    for _, row in invalid.iterrows():
        issues.append({"issue": "missing_valid_from", "row": row.to_dict()})

    reversed_rows = df.loc[df["valid_from"].notna() & df["valid_to"].notna() & (df["valid_from"] > df["valid_to"])]
    for _, row in reversed_rows.iterrows():
        issues.append({"issue": "reversed_interval", "row": row.to_dict()})

    valid = df.loc[df["valid_from"].notna()].sort_values(["valid_from", "valid_to"])
    if len(valid) > 1:
        prev_end = _interval_to_finite_end(valid.iloc[0]["valid_to"])
        prev_idx = 0
        for i in range(1, len(valid)):
            s = pd.Timestamp(valid.iloc[i]["valid_from"])  # non-na
            if s <= prev_end:
                issues.append(
                    {
                        "issue": "overlapping_interval",
                        "left_row": valid.iloc[prev_idx].to_dict(),
                        "right_row": valid.iloc[i].to_dict(),
                    }
                )
            e = _interval_to_finite_end(valid.iloc[i]["valid_to"])
            if e >= prev_end:
                prev_end = e
                prev_idx = i

    status = "pass" if not issues else "warn"
    if fail_closed and issues:
        raise RuntimeError(f"Sector PIT interval validation failed: {len(issues)} issues")

    return {"status": status, "issues": issues}


def _update_sec_sic_master(
    sec_sic: str,
    sec_sic_description: str,
    *,
    ticker: str,
    cik: int | None,
    master_path: str | Path = SEC_SIC_MASTER_PATH,
) -> None:
    ensure_sector_proxy_reference_files(sic_master_path=master_path)
    p = Path(master_path).expanduser()

    row = {
        "sec_sic": str(sec_sic or ""),
        "sec_sic_description": str(sec_sic_description or ""),
        "sample_ticker": str(ticker).upper(),
        "sample_cik": int(cik) if cik is not None else np.nan,
        "updated_at": now_utc_iso(),
    }

    with _SIC_MASTER_LOCK:
        try:
            old = pd.read_csv(p)
        except Exception:
            old = pd.DataFrame(columns=["sec_sic", "sec_sic_description", "sample_ticker", "sample_cik", "updated_at"])
        merged = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        merged["sec_sic"] = merged["sec_sic"].astype(str).str.strip()
        merged["sec_sic_description"] = merged["sec_sic_description"].astype(str).str.strip()
        merged = merged.drop_duplicates(subset=["sec_sic", "sec_sic_description"], keep="last")
        merged = merged.sort_values(["sec_sic", "sec_sic_description"]).reset_index(drop=True)
        merged.to_csv(p, index=False, encoding="utf-8")


def build_ticker_sector_proxy_cache(
    ticker: str,
    *,
    market: str = "us",
    user_agent: str | None = None,
    retries: int = 3,
    backoff_base: float = 1.0,
    force_refresh: bool = False,
    rules_path: str | Path = SIC_RULES_PATH,
    overrides_path: str | Path | None = None,
    mapping_version: str = DEFAULT_MAPPING_VERSION,
    fail_closed: bool = False,
    rules_df: pd.DataFrame | None = None,
    debug_rule_samples: bool = False,
) -> pd.DataFrame:
    ensure_sector_proxy_reference_files(rules_path=rules_path)

    sym = str(ticker).strip().upper()
    if not sym:
        raise ValueError("ticker is required")

    rules = (
        rules_df.copy()
        if isinstance(rules_df, pd.DataFrame)
        else load_sector_proxy_rules(rules_path=rules_path, debug_sample=debug_rule_samples)
    )
    payload, cik = fetch_submissions_for_ticker(
        sym,
        user_agent=user_agent,
        retries=retries,
        backoff_base=backoff_base,
        force_refresh=force_refresh,
        cache_dir=SEC_SUBMISSIONS_CACHE_DIR,
    )
    identity = _extract_submission_identity(payload, sym, cik=cik)

    mapped = map_sic_to_sector_proxy(
        identity.get("sec_sic"),
        identity.get("sec_sic_description"),
        rules_df=rules,
        mapping_version=mapping_version,
    )

    first_valid, last_valid, validity_source = _resolve_validity_window(
        sym,
        market=market,
        overrides_path=overrides_path,
    )

    base_row: dict[str, Any] = {
        "ticker": sym,
        "cik": int(cik),
        "valid_from": first_valid if first_valid is not None else pd.Timestamp("1900-01-01"),
        "valid_to": last_valid if last_valid is not None else pd.NaT,
        "sec_sic": identity.get("sec_sic", ""),
        "sec_sic_description": identity.get("sec_sic_description", ""),
        "sector_l1_kr": mapped["sector_l1_kr"],
        "sector_l2_kr": mapped["sector_l2_kr"],
        "sector_l1_en": mapped.get("sector_l1_en", ""),
        "sector_l2_en": mapped.get("sector_l2_en", ""),
        "mapping_source": mapped["mapping_source"],
        "mapping_confidence": float(mapped["mapping_confidence"]),
        "rule_id": mapped["rule_id"],
        "mapping_version": mapped["mapping_version"],
        "source_filing_date": identity.get("source_filing_date", pd.NaT),
        "source_acceptance_datetime": identity.get("source_acceptance_datetime", pd.NaT),
        "updated_at": now_utc_iso(),
        "note": (f"validity_source={validity_source}; {mapped.get('note', '')}").strip("; "),
    }

    rows: list[dict[str, Any]] = [base_row]

    ovr_rows = _find_override_rows_for_ticker(sym, market=market, overrides_path=overrides_path)
    if not ovr_rows.empty:
        sector_cols = ["sector_l1_kr", "sector_l2_kr", "sector_l1_en", "sector_l2_en", "mapping_confidence", "rule_id", "note"]
        for _, ovr in ovr_rows.iterrows():
            has_sector_override = any(str(ovr.get(c, "")).strip() for c in ["sector_l1_kr", "sector_l2_kr", "sector_l1_en", "sector_l2_en"])
            if not has_sector_override:
                continue

            ov_start = _coerce_ts(ovr.get("first_valid_date"))
            ov_end = _coerce_ts(ovr.get("last_valid_date"))
            if ov_start is None:
                ov_start = _coerce_ts(base_row.get("valid_from")) or pd.Timestamp("1900-01-01")

            override_row = dict(base_row)
            override_row["valid_from"] = ov_start
            override_row["valid_to"] = ov_end if ov_end is not None else pd.NaT
            for c in sector_cols:
                if c in ovr and pd.notna(ovr.get(c)) and str(ovr.get(c)).strip() != "":
                    override_row[c] = ovr.get(c)

            override_row["mapping_source"] = "override"
            override_row["rule_id"] = str(override_row.get("rule_id") or "override_manual")
            override_row["mapping_confidence"] = float(_safe_float(override_row.get("mapping_confidence"), default=1.0))
            override_row["mapping_version"] = mapping_version
            override_row["note"] = str(ovr.get("note") or ovr.get("replacement_or_note") or "manual override")
            rows = _split_rows_with_override(rows, override_row)

    pit = pd.DataFrame(rows)
    for c in PIT_COLUMNS:
        if c not in pit.columns:
            pit[c] = np.nan

    pit = pit[PIT_COLUMNS].copy()
    pit["ticker"] = pit["ticker"].astype(str).str.upper().str.strip()
    pit["cik"] = pd.to_numeric(pit["cik"], errors="coerce").astype("Int64")
    pit["valid_from"] = pd.to_datetime(pit["valid_from"], errors="coerce").dt.normalize()
    pit["valid_to"] = pd.to_datetime(pit["valid_to"], errors="coerce").dt.normalize()
    pit["source_filing_date"] = pd.to_datetime(pit["source_filing_date"], errors="coerce").dt.normalize()
    pit["source_acceptance_datetime"] = pd.to_datetime(pit["source_acceptance_datetime"], errors="coerce")
    pit["mapping_confidence"] = pd.to_numeric(pit["mapping_confidence"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    pit = pit.sort_values(["valid_from", "valid_to", "mapping_source"]).reset_index(drop=True)

    validation = validate_sector_proxy_pit_intervals(pit, fail_closed=fail_closed)
    if validation.get("status") != "pass":
        LOGGER.warning("Sector PIT interval warnings for %s: %d", sym, len(validation.get("issues", [])))

    ensure_dir(SEC_TICKER_SECTOR_PROXY_DIR)
    out_path = SEC_TICKER_SECTOR_PROXY_DIR / f"{sanitize_ticker(sym)}.parquet"
    pit.to_parquet(out_path, index=False)

    _update_sec_sic_master(
        str(identity.get("sec_sic") or ""),
        str(identity.get("sec_sic_description") or ""),
        ticker=sym,
        cik=cik,
        master_path=SEC_SIC_MASTER_PATH,
    )

    return pit


def build_sector_proxy_cache_for_universe(
    symbols: list[str] | tuple[str, ...] | set[str],
    *,
    market: str = "us",
    user_agent: str | None = None,
    retries: int = 3,
    backoff_base: float = 1.0,
    force_refresh: bool = False,
    rules_path: str | Path = SIC_RULES_PATH,
    overrides_path: str | Path | None = None,
    mapping_version: str = DEFAULT_MAPPING_VERSION,
    fail_closed: bool = False,
    unclassified_top_n: int = 20,
    debug_rule_samples: bool = False,
    offline_mode: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    syms = _normalize_tickers(symbols)
    if not syms:
        return {
            "ok": 0,
            "failed": 0,
            "unclassified": 0,
            "total": 0,
            "unclassified_report": "",
            "errors": [],
        }

    ensure_sector_proxy_reference_files(rules_path=rules_path)

    ok = 0
    failed = 0
    errors: list[dict[str, str]] = []
    unclassified_rows: list[dict[str, Any]] = []
    _counter_lock = threading.Lock()

    rules = load_sector_proxy_rules(rules_path=rules_path, debug_sample=debug_rule_samples)

    def _process_one(sym: str) -> tuple[str, str | None, dict[str, Any] | None]:
        """Process a single ticker. Returns (sym, error_msg_or_None, unclassified_row_or_None)."""
        try:
            if offline_mode:
                pit = load_sector_proxy_pit(sym, market=market)
                if pit.empty:
                    return sym, "no_cache_in_offline_mode", None
            else:
                pit = build_ticker_sector_proxy_cache(
                    sym,
                    market=market,
                    user_agent=user_agent,
                    retries=retries,
                    backoff_base=backoff_base,
                    force_refresh=force_refresh,
                    rules_path=rules_path,
                    overrides_path=overrides_path,
                    mapping_version=mapping_version,
                    fail_closed=fail_closed,
                    rules_df=rules,
                )

            latest = pit.sort_values(["valid_from", "updated_at"], ascending=[True, True]).iloc[-1]
            unclassified_row = None
            if str(latest.get("sector_l1_kr", "")) == DEFAULT_UNCLASSIFIED_L1:
                unclassified_row = {
                    "ticker": sym,
                    "sector_l1_kr": latest.get("sector_l1_kr"),
                    "sector_l2_kr": latest.get("sector_l2_kr"),
                    "sec_sic": latest.get("sec_sic"),
                    "sec_sic_description": latest.get("sec_sic_description"),
                    "mapping_source": latest.get("mapping_source"),
                    "market_cap_proxy": _estimate_market_cap_proxy(sym, market=market, offline_mode=offline_mode),
                }
            return sym, None, unclassified_row
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("SEC sector proxy cache failed for %s: %s", sym, exc)
            return sym, str(exc), None

    n_workers = max(1, int(workers))
    total = len(syms)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one, sym): sym for sym in syms}
        done_count = 0
        for future in as_completed(futures):
            sym, err, unclassified_row = future.result()
            done_count += 1
            if done_count % 200 == 0:
                LOGGER.info("sec-sector-cache progress: %d/%d", done_count, total)
            with _counter_lock:
                if err is not None:
                    failed += 1
                    errors.append({"ticker": sym, "error": err})
                    if fail_closed:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise RuntimeError(f"fail_closed: {sym}: {err}")
                else:
                    ok += 1
                    if unclassified_row is not None:
                        unclassified_rows.append(unclassified_row)

    report_path = SEC_UNCLASSIFIED_REPORT_PATH
    ensure_dir(report_path.parent)
    unclassified_df = pd.DataFrame(unclassified_rows)
    if not unclassified_df.empty:
        unclassified_df["market_cap_proxy"] = pd.to_numeric(unclassified_df["market_cap_proxy"], errors="coerce")
        unclassified_df = unclassified_df.sort_values("market_cap_proxy", ascending=False, na_position="last")
        unclassified_df.head(max(int(unclassified_top_n), 1)).to_csv(report_path, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["ticker", "market_cap_proxy", "sec_sic", "sec_sic_description"]).to_csv(
            report_path,
            index=False,
            encoding="utf-8",
        )

    return {
        "ok": ok,
        "failed": failed,
        "unclassified": int(len(unclassified_rows)),
        "total": int(len(syms)),
        "unclassified_report": str(report_path),
        "errors": errors,
    }


def load_sector_proxy_pit(
    ticker: str,
    *,
    market: str = "us",
    pit_dir: str | Path = SEC_TICKER_SECTOR_PROXY_DIR,
) -> pd.DataFrame:
    sym = str(ticker).strip().upper()
    if not sym:
        return pd.DataFrame(columns=PIT_COLUMNS)

    p = Path(pit_dir).expanduser() / f"{sanitize_ticker(sym)}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=PIT_COLUMNS)

    try:
        df = pd.read_parquet(p)
    except Exception:
        return pd.DataFrame(columns=PIT_COLUMNS)

    if df is None or df.empty:
        return pd.DataFrame(columns=PIT_COLUMNS)

    out = df.copy()
    for col in PIT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["valid_from"] = pd.to_datetime(out["valid_from"], errors="coerce").dt.normalize()
    out["valid_to"] = pd.to_datetime(out["valid_to"], errors="coerce").dt.normalize()
    out["source_filing_date"] = pd.to_datetime(out["source_filing_date"], errors="coerce").dt.normalize()
    out["source_acceptance_datetime"] = pd.to_datetime(out["source_acceptance_datetime"], errors="coerce")
    out = out.sort_values(["valid_from", "valid_to"]).reset_index(drop=True)
    return out[PIT_COLUMNS]


def get_sector_proxy_asof_from_pit(pit: pd.DataFrame, asof_date: str | pd.Timestamp) -> dict[str, Any] | None:
    if pit is None or pit.empty:
        return None
    asof = _coerce_ts(asof_date)
    if asof is None:
        return None

    df = pit.copy()
    df["valid_from"] = pd.to_datetime(df["valid_from"], errors="coerce").dt.normalize()
    df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce").dt.normalize()
    mask = df["valid_from"].notna() & (df["valid_from"] <= asof)
    mask &= df["valid_to"].isna() | (df["valid_to"] >= asof)
    cand = df.loc[mask].copy()
    if cand.empty:
        return None

    cand = cand.sort_values(["valid_from", "mapping_confidence"], ascending=[False, False])
    row = cand.iloc[0].to_dict()
    row = {k: (v.isoformat() if isinstance(v, pd.Timestamp) and pd.notna(v) else v) for k, v in row.items()}
    return row


def get_sector_proxy_asof(
    ticker: str,
    asof_date: str | pd.Timestamp,
    *,
    market: str = "us",
    pit_dir: str | Path = SEC_TICKER_SECTOR_PROXY_DIR,
) -> dict[str, Any] | None:
    pit = load_sector_proxy_pit(ticker, market=market, pit_dir=pit_dir)
    return get_sector_proxy_asof_from_pit(pit, asof_date)


def align_sector_proxy_to_dates(
    ticker: str,
    dates: pd.DatetimeIndex,
    *,
    market: str = "us",
    pit_dir: str | Path = SEC_TICKER_SECTOR_PROXY_DIR,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce")).dropna().sort_values().unique()
    if len(idx) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    pit = load_sector_proxy_pit(ticker, market=market, pit_dir=pit_dir)
    base = pd.DataFrame({"date": idx}, index=idx)

    if pit.empty:
        out = base.copy()
        out["sec_sic"] = ""
        out["sec_sic_description"] = ""
        out["sector_l1_kr"] = DEFAULT_UNCLASSIFIED_L1
        out["sector_l2_kr"] = DEFAULT_UNCLASSIFIED_L2
        out["sector_l1_en"] = ""
        out["sector_l2_en"] = ""
        out["mapping_source"] = "fallback_unclassified"
        out["mapping_confidence"] = 0.0
        out["rule_id"] = "fallback_unclassified"
        out["mapping_version"] = DEFAULT_MAPPING_VERSION
        out["source_filing_date"] = pd.NaT
        out["source_acceptance_datetime"] = pd.NaT
        out["sector_valid_from"] = pd.NaT
        out["sector_valid_to"] = pd.NaT
        out["updated_at"] = now_utc_iso()
        return out.drop(columns=["date"])

    right = pit.copy().sort_values("valid_from")
    left = base.reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="valid_from",
        direction="backward",
    )

    valid_to = pd.to_datetime(merged.get("valid_to"), errors="coerce").dt.normalize()
    valid = valid_to.isna() | (merged["date"] <= valid_to)

    # For rows outside PIT window, fallback to unclassified.
    merged.loc[~valid, [
        "sec_sic",
        "sec_sic_description",
        "sector_l1_kr",
        "sector_l2_kr",
        "sector_l1_en",
        "sector_l2_en",
        "mapping_source",
        "mapping_confidence",
        "rule_id",
        "mapping_version",
        "source_filing_date",
        "source_acceptance_datetime",
        "valid_from",
        "valid_to",
        "updated_at",
        "note",
    ]] = np.nan

    merged["sec_sic"] = merged["sec_sic"].fillna("")
    merged["sec_sic_description"] = merged["sec_sic_description"].fillna("")
    merged["sector_l1_kr"] = merged["sector_l1_kr"].fillna(DEFAULT_UNCLASSIFIED_L1)
    merged["sector_l2_kr"] = merged["sector_l2_kr"].fillna(DEFAULT_UNCLASSIFIED_L2)
    merged["sector_l1_en"] = merged["sector_l1_en"].fillna("")
    merged["sector_l2_en"] = merged["sector_l2_en"].fillna("")
    merged["mapping_source"] = merged["mapping_source"].fillna("fallback_unclassified")
    merged["mapping_confidence"] = pd.to_numeric(merged["mapping_confidence"], errors="coerce").fillna(0.0)
    merged["rule_id"] = merged["rule_id"].fillna("fallback_unclassified")
    merged["mapping_version"] = merged["mapping_version"].fillna(DEFAULT_MAPPING_VERSION)
    merged["source_filing_date"] = pd.to_datetime(merged["source_filing_date"], errors="coerce").dt.normalize()
    merged["source_acceptance_datetime"] = pd.to_datetime(merged["source_acceptance_datetime"], errors="coerce")
    merged["sector_valid_from"] = pd.to_datetime(merged["valid_from"], errors="coerce").dt.normalize()
    merged["sector_valid_to"] = pd.to_datetime(merged["valid_to"], errors="coerce").dt.normalize()
    merged["updated_at"] = merged["updated_at"].fillna(now_utc_iso())

    out = merged.set_index("date")
    out = out.reindex(idx)
    return out[
        [
            "sec_sic",
            "sec_sic_description",
            "sector_l1_kr",
            "sector_l2_kr",
            "sector_l1_en",
            "sector_l2_en",
            "mapping_source",
            "mapping_confidence",
            "rule_id",
            "mapping_version",
            "source_filing_date",
            "source_acceptance_datetime",
            "sector_valid_from",
            "sector_valid_to",
            "updated_at",
            "note",
        ]
    ]


def detect_sector_pit_time_inconsistency(
    trades_df: pd.DataFrame,
    *,
    market: str = "us",
    pit_dir: str | Path = SEC_TICKER_SECTOR_PROXY_DIR,
    tolerance_days: int = 7,
    warn_days: int = 30,
    fail_days: int = 180,
    mode_label: str | None = None,
) -> dict[str, Any]:
    cols = [
        "mode_label",
        "ticker",
        "trade_date",
        "first_valid_date",
        "last_valid_date",
        "delta_days_from_first",
        "delta_days_from_last",
        "check_result",
        "check_type",
        "source_used",
        "note",
    ]

    if trades_df is None or trades_df.empty:
        return {
            "status": "pass",
            "issues": pd.DataFrame(columns=cols),
            "summary": {"total_issues": 0, "warn_count": 0, "fail_count": 0},
        }

    trade_date_col = "exec_date" if "exec_date" in trades_df.columns else ("trade_date" if "trade_date" in trades_df.columns else "date")
    if trade_date_col not in trades_df.columns or "ticker" not in trades_df.columns:
        return {
            "status": "warn",
            "issues": pd.DataFrame(columns=cols),
            "summary": {"total_issues": 0, "warn_count": 1, "fail_count": 0, "note": "trades schema missing ticker/date"},
        }

    tr = trades_df.copy()
    tr["ticker"] = tr["ticker"].astype(str).str.upper().str.strip()
    tr["trade_date"] = pd.to_datetime(tr[trade_date_col], errors="coerce").dt.normalize()
    tr = tr.dropna(subset=["ticker", "trade_date"])
    if tr.empty:
        return {
            "status": "pass",
            "issues": pd.DataFrame(columns=cols),
            "summary": {"total_issues": 0, "warn_count": 0, "fail_count": 0},
        }

    tol = max(int(tolerance_days), 0)
    warn_d = max(int(warn_days), 0)
    fail_d = max(int(fail_days), warn_d)

    issues: list[dict[str, Any]] = []

    for ticker in sorted(tr["ticker"].unique()):
        pit = load_sector_proxy_pit(ticker, market=market, pit_dir=pit_dir)
        if pit.empty:
            sub = tr.loc[tr["ticker"] == ticker]
            for _, row in sub.iterrows():
                issues.append(
                    {
                        "mode_label": mode_label or "",
                        "ticker": ticker,
                        "trade_date": row["trade_date"],
                        "first_valid_date": pd.NaT,
                        "last_valid_date": pd.NaT,
                        "delta_days_from_first": np.nan,
                        "delta_days_from_last": np.nan,
                        "check_result": "warn",
                        "check_type": "missing_sector_pit",
                        "source_used": "sector_pit",
                        "note": "sector PIT not found",
                    }
                )
            continue

        first_valid = pd.to_datetime(pit["valid_from"], errors="coerce").dropna()
        last_valid = pd.to_datetime(pit["valid_to"], errors="coerce").dropna()
        first = pd.Timestamp(first_valid.min()).normalize() if not first_valid.empty else None
        last = pd.Timestamp(last_valid.max()).normalize() if not last_valid.empty else None

        sub = tr.loc[tr["ticker"] == ticker]
        for _, row in sub.iterrows():
            td = pd.Timestamp(row["trade_date"]).normalize()
            asof = get_sector_proxy_asof(ticker, td, market=market, pit_dir=pit_dir)
            if asof is not None:
                continue

            if first is not None:
                delta_first = int((first - td).days)
                if delta_first > tol:
                    result = "fail" if delta_first > fail_d else "warn"
                    issues.append(
                        {
                            "mode_label": mode_label or "",
                            "ticker": ticker,
                            "trade_date": td,
                            "first_valid_date": first,
                            "last_valid_date": last if last is not None else pd.NaT,
                            "delta_days_from_first": delta_first,
                            "delta_days_from_last": np.nan,
                            "check_result": result,
                            "check_type": "before_first_valid",
                            "source_used": "sector_pit",
                            "note": "trade date earlier than sector PIT validity",
                        }
                    )
                    continue

            if last is not None:
                delta_last = int((td - last).days)
                if delta_last > tol:
                    result = "fail" if delta_last > fail_d else ("warn" if delta_last > warn_d else "warn")
                    issues.append(
                        {
                            "mode_label": mode_label or "",
                            "ticker": ticker,
                            "trade_date": td,
                            "first_valid_date": first if first is not None else pd.NaT,
                            "last_valid_date": last,
                            "delta_days_from_first": np.nan,
                            "delta_days_from_last": delta_last,
                            "check_result": result,
                            "check_type": "after_last_valid",
                            "source_used": "sector_pit",
                            "note": "trade date later than sector PIT validity",
                        }
                    )
                continue

            issues.append(
                {
                    "mode_label": mode_label or "",
                    "ticker": ticker,
                    "trade_date": td,
                    "first_valid_date": first if first is not None else pd.NaT,
                    "last_valid_date": last if last is not None else pd.NaT,
                    "delta_days_from_first": np.nan,
                    "delta_days_from_last": np.nan,
                    "check_result": "warn",
                    "check_type": "missing_asof_interval",
                    "source_used": "sector_pit",
                    "note": "no matching PIT interval for trade date",
                }
            )

    issues_df = pd.DataFrame(issues, columns=cols)
    warn_count = int((issues_df.get("check_result", pd.Series(dtype=str)) == "warn").sum()) if not issues_df.empty else 0
    fail_count = int((issues_df.get("check_result", pd.Series(dtype=str)) == "fail").sum()) if not issues_df.empty else 0
    status = "fail" if fail_count > 0 else ("warn" if warn_count > 0 else "pass")

    return {
        "status": status,
        "issues": issues_df,
        "summary": {
            "total_issues": int(len(issues_df)),
            "warn_count": warn_count,
            "fail_count": fail_count,
            "tolerance_days": tol,
            "warn_days": warn_d,
            "fail_days": fail_d,
        },
    }


def build_sector_proxy_validation_report(
    bundle_path: str | Path,
    *,
    market: str = "us",
    fail_closed: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    from market_data.backtest.validation_snapshots import validate_rebalance_snapshots

    root = Path(bundle_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"bundle path not found: {root}")

    summary_path = root / "ai_review_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"ai_review_summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mode = str(summary.get("mode", "single"))

    run_specs: list[tuple[str, str, Path]] = []
    if mode == "single":
        run_specs.append(("single", "single", root / "result"))
    else:
        runs_dir = root / "runs"
        for sub in ["lump_sum", "dca", "va", "screen", "strategy"]:
            p = runs_dir / sub
            if p.exists() and p.is_dir():
                run_specs.append((sub, sub, p))

    if not run_specs:
        raise RuntimeError("No run directories found under bundle")

    run_reports: list[dict[str, Any]] = []
    all_issues: list[pd.DataFrame] = []

    for mode_key, mode_label, run_dir in run_specs:
        trades_path = run_dir / "trades.csv"
        funding_path = run_dir / "funding_flows.csv"
        snapshot_index_path = run_dir / "rebalance_snapshots_index.csv"
        rebalance_log_path = run_dir / "rebalance_log.csv"

        trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
        funding = pd.read_csv(funding_path) if funding_path.exists() else pd.DataFrame()
        rebalance_log = pd.read_csv(rebalance_log_path) if rebalance_log_path.exists() else pd.DataFrame()

        detected = detect_sector_pit_time_inconsistency(
            trades,
            market=market,
            mode_label=mode_label,
        )
        issues = detected.get("issues") if isinstance(detected.get("issues"), pd.DataFrame) else pd.DataFrame()
        if not issues.empty:
            issues = issues.copy()
            issues["run_mode"] = mode_key
            all_issues.append(issues)

        # Missing/unclassified/conflicting rates
        total_trades = int(len(trades))
        missing_count = int((issues.get("check_type", pd.Series(dtype=str)).isin(["missing_sector_pit", "missing_asof_interval"]).sum()) if not issues.empty else 0)

        unclassified_count = 0
        conflicting_count = 0
        if not trades.empty and {"ticker", "exec_date"}.issubset(set(trades.columns)):
            trades_tmp = trades.copy()
            trades_tmp["exec_date"] = pd.to_datetime(trades_tmp["exec_date"], errors="coerce").dt.normalize()
            for _, row in trades_tmp.dropna(subset=["ticker", "exec_date"]).iterrows():
                sym = str(row["ticker"]).upper().strip()
                dt = pd.Timestamp(row["exec_date"])  # normalized
                pit = load_sector_proxy_pit(sym)
                if pit.empty:
                    continue
                pit2 = pit.copy()
                pit2["valid_from"] = pd.to_datetime(pit2["valid_from"], errors="coerce").dt.normalize()
                pit2["valid_to"] = pd.to_datetime(pit2["valid_to"], errors="coerce").dt.normalize()
                mask = pit2["valid_from"].notna() & (pit2["valid_from"] <= dt)
                mask &= pit2["valid_to"].isna() | (pit2["valid_to"] >= dt)
                sub = pit2.loc[mask]
                if len(sub) > 1:
                    conflicting_count += 1
                    all_issues.append(
                        pd.DataFrame(
                            [
                                {
                                    "mode_label": mode_label,
                                    "ticker": sym,
                                    "trade_date": dt,
                                    "check_result": "warn",
                                    "check_type": "conflicting_mapping",
                                    "source_used": "sector_pit",
                                    "note": f"{len(sub)} overlapping PIT rows",
                                    "run_mode": mode_key,
                                }
                            ]
                        )
                    )
                if not sub.empty:
                    picked = sub.sort_values(["valid_from", "mapping_confidence"], ascending=[False, False]).iloc[0]
                    if str(picked.get("sector_l1_kr", "")) == DEFAULT_UNCLASSIFIED_L1:
                        unclassified_count += 1

        snapshot_check = validate_rebalance_snapshots(
            snapshot_index_path if snapshot_index_path.exists() else pd.DataFrame(),
            rebalance_log_df=rebalance_log,
            base_dir=run_dir,
            required_columns=SECTOR_SNAPSHOT_REQUIRED_COLUMNS,
            validation_mode="warn",
        )

        missing_rate = float(missing_count / total_trades) if total_trades > 0 else 0.0
        unclassified_rate = float(unclassified_count / total_trades) if total_trades > 0 else 0.0
        conflicting_rate = float(conflicting_count / total_trades) if total_trades > 0 else 0.0

        run_statuses = [detected.get("status", "pass"), snapshot_check.get("status", "pass")]
        status = "fail" if "fail" in run_statuses else ("warn" if "warn" in run_statuses else "pass")

        run_reports.append(
            {
                "mode_key": mode_key,
                "mode_label": mode_label,
                "status": status,
                "total_trades": total_trades,
                "missing_sector_count": missing_count,
                "missing_sector_rate": missing_rate,
                "unclassified_count": unclassified_count,
                "unclassified_rate": unclassified_rate,
                "conflicting_mapping_count": conflicting_count,
                "conflicting_mapping_rate": conflicting_rate,
                "ticker_time_summary": detected.get("summary", {}),
                "snapshot_sector_evidence": {
                    "status": snapshot_check.get("status", "warn"),
                    "counts": snapshot_check.get("counts", {}),
                    "summary": snapshot_check.get("summary", {}),
                },
            }
        )

    issues_df = pd.concat(all_issues, ignore_index=True, sort=False) if all_issues else pd.DataFrame()

    overall_status = "pass"
    if any(str(r.get("status")) == "fail" for r in run_reports):
        overall_status = "fail"
    elif any(str(r.get("status")) == "warn" for r in run_reports):
        overall_status = "warn"

    report = {
        "bundle_path": str(root),
        "generated_at": now_utc_iso(),
        "mode": mode,
        "status": overall_status,
        "run_reports": run_reports,
        "issue_count": int(len(issues_df)),
    }

    report_path = root / "sector_proxy_validation_report.json"
    issues_path = root / "sector_proxy_validation_issues.csv"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    issues_df.to_csv(issues_path, index=False, encoding="utf-8")

    if fail_closed and overall_status == "fail":
        raise RuntimeError(f"sector proxy validation failed: {root}")

    return report, issues_df


__all__ = [
    "DEFAULT_MAPPING_VERSION",
    "DEFAULT_UNCLASSIFIED_L1",
    "DEFAULT_UNCLASSIFIED_L2",
    "PIT_COLUMNS",
    "SECTOR_SNAPSHOT_REQUIRED_COLUMNS",
    "build_sector_proxy_cache_for_universe",
    "build_sector_proxy_validation_report",
    "build_ticker_sector_proxy_cache",
    "ensure_sector_proxy_reference_files",
    "fetch_submissions_for_ticker",
    "get_sector_proxy_asof",
    "get_sector_proxy_asof_from_pit",
    "align_sector_proxy_to_dates",
    "load_sector_proxy_pit",
    "load_sector_proxy_rules",
    "map_sic_to_sector_proxy",
    "detect_sector_pit_time_inconsistency",
    "validate_sector_proxy_pit_intervals",
]
