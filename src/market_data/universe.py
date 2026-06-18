from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from market_data.config import (
    KOSPI_EXTERNAL_DEFAULT_URL,
    KOSPI_TOP_N_DEFAULT,
    KRX_CORP_LIST_URL,
    SP500_WIKI_URL,
    UNIVERSE_DIR,
)
from market_data.utils import ensure_dir, retry_call

NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def _ticker_to_code(symbol: str) -> str | None:
    match = re.search(r"(\d{6})", str(symbol))
    if not match:
        return None
    return match.group(1)


def _fetch_krx_sector_table() -> pd.DataFrame:
    response = retry_call(
        lambda: requests.get(KRX_CORP_LIST_URL, headers=WIKI_HEADERS, timeout=20),
        retries=3,
        backoff_base=1.0,
        label="krx-corp-list",
    )
    response.raise_for_status()
    response.encoding = "euc-kr"
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError("KRX corpList table not found")

    df = tables[0].copy()
    required_cols = {"종목코드", "시장구분", "업종", "회사명"}
    if not required_cols.issubset(set(df.columns)):
        raise RuntimeError(f"KRX corpList missing columns: {required_cols - set(df.columns)}")

    out = df.loc[:, ["종목코드", "시장구분", "업종", "회사명"]].copy()
    out["종목코드"] = out["종목코드"].astype(str).str.extract(r"(\d{6})", expand=False)
    out = out.dropna(subset=["종목코드"]).drop_duplicates(subset=["종목코드"], keep="first")
    out = out.rename(
        columns={
            "종목코드": "Code",
            "시장구분": "KRXMarket",
            "업종": "Sector",
            "회사명": "CompanyName",
        }
    )
    return out


def _attach_kr_sector_info(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Code"] = out["Symbol"].map(_ticker_to_code)
    try:
        sector_df = _fetch_krx_sector_table()
        out = out.merge(sector_df, on="Code", how="left")
    except Exception as exc:  # noqa: BLE001
        print(f"[UNIVERSE WARN] KRX sector join failed: {exc}")
        out["Sector"] = pd.NA
        out["KRXMarket"] = pd.NA
        out["CompanyName"] = pd.NA
    return out


def _normalize_sp500_symbol(symbol: str) -> str:
    return str(symbol).strip().replace(".", "-")


def _normalize_kr_symbol(symbol: str) -> str:
    raw = str(symbol).strip().upper()
    if raw.endswith(".KS") or raw.endswith(".KQ"):
        return raw
    if raw.isdigit() and len(raw) == 6:
        return f"{raw}.KS"
    return raw


def _dedupe_keep_order(symbols: list[str]) -> list[str]:
    return list(dict.fromkeys(symbols))


def _apply_top_n(symbols: list[str], top_n: int | None) -> list[str]:
    deduped = _dedupe_keep_order(symbols)
    if top_n is None or top_n <= 0:
        return deduped
    return deduped[:top_n]


def _replace_page_in_url(url: str, page: int) -> str:
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    q["page"] = [str(page)]
    query = urlencode(q, doseq=True)
    return urlunparse(parsed._replace(query=query))


def _extract_naver_last_page(soup: BeautifulSoup) -> int:
    pages: list[int] = []
    for a in soup.select("td.pgRR a, td.pgR a, td.pgRr a, td.pgRRr a"):
        href = a.get("href", "")
        match = re.search(r"[?&]page=(\d+)", href)
        if match:
            pages.append(int(match.group(1)))
    return max(pages) if pages else 1


def _extract_naver_codes(soup: BeautifulSoup) -> list[str]:
    codes: list[str] = []
    for a in soup.select("a[href*='item/main.naver?code=']"):
        href = a.get("href", "")
        match = re.search(r"code=(\d{6})", href)
        if match:
            codes.append(match.group(1))
    return codes


def _fetch_kospi_from_naver(url: str, top_n: int | None) -> pd.DataFrame:
    def _get_page(u: str) -> requests.Response:
        return requests.get(u, headers=NAVER_HEADERS, timeout=20)

    first_resp = retry_call(
        lambda: _get_page(_replace_page_in_url(url, 1)),
        retries=3,
        backoff_base=1.0,
        label="kospi-naver-page1",
    )
    first_resp.raise_for_status()
    first_resp.encoding = "euc-kr"
    first_soup = BeautifulSoup(first_resp.text, "html.parser")

    max_page = _extract_naver_last_page(first_soup)
    all_codes = _extract_naver_codes(first_soup)

    for page in range(2, max_page + 1):
        if top_n is not None and top_n > 0 and len(_dedupe_keep_order(all_codes)) >= top_n:
            break
        resp = retry_call(
            lambda page_no=page: _get_page(_replace_page_in_url(url, page_no)),
            retries=3,
            backoff_base=1.0,
            label=f"kospi-naver-page{page}",
        )
        resp.raise_for_status()
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")
        all_codes.extend(_extract_naver_codes(soup))

    symbols = [f"{code}.KS" for code in _apply_top_n(all_codes, top_n)]
    out = pd.DataFrame({"Symbol": symbols})
    out["Market"] = "kr"
    out["Source"] = "naver"
    return out


def _fetch_kospi_from_csv(url: str, top_n: int | None) -> pd.DataFrame:
    response = retry_call(
        lambda: requests.get(url, timeout=20),
        retries=3,
        backoff_base=1.0,
        label="kospi-csv",
    )
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))

    symbol_col = next((c for c in ["Symbol", "symbol", "Code", "code"] if c in df.columns), None)
    if symbol_col is None:
        raise RuntimeError("External KOSPI source missing Symbol/Code column")

    market_col = next((c for c in ["Market", "market"] if c in df.columns), None)
    if market_col is not None:
        df = df[df[market_col].astype(str).str.upper().str.contains("KOSPI", na=False)]

    symbols_series = df[symbol_col].astype(str).str.extract(r"(\d{6})", expand=False).dropna()
    symbols = [f"{code}.KS" for code in _apply_top_n(symbols_series.tolist(), top_n)]
    out = pd.DataFrame({"Symbol": symbols})
    out["Market"] = "kr"
    out["Source"] = "csv"
    return out


def fetch_sp500_universe() -> pd.DataFrame:
    response = retry_call(
        lambda: requests.get(SP500_WIKI_URL, headers=WIKI_HEADERS, timeout=20),
        retries=3,
        backoff_base=1.0,
        label="sp500-wiki",
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError("No tables found on S&P 500 wiki page")
    df = tables[0].copy()
    if "Symbol" not in df.columns:
        raise RuntimeError("Wikipedia table does not contain Symbol column")
    df["Symbol"] = df["Symbol"].map(_normalize_sp500_symbol)
    df["Market"] = "us"
    return df


def save_universe_csv(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)


def load_tickers_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return []
    candidates = ["ticker", "symbol", "code", "Ticker", "Symbol", "Code"]
    col = next((c for c in candidates if c in df.columns), df.columns[0])
    return _dedupe_keep_order([_normalize_kr_symbol(v) for v in df[col].dropna().astype(str).tolist()])


def fetch_kospi_universe_external(
    url: str = KOSPI_EXTERNAL_DEFAULT_URL,
    top_n: int | None = KOSPI_TOP_N_DEFAULT,
) -> pd.DataFrame:
    if url.lower().endswith(".csv"):
        return _fetch_kospi_from_csv(url, top_n=top_n)
    return _fetch_kospi_from_naver(url, top_n=top_n)


def build_universe(
    universe: str,
    tickers_file: str | None,
    kospi_external_url: str,
    kospi_top_n: int | None = KOSPI_TOP_N_DEFAULT,
) -> tuple[list[str], str, Path]:
    ensure_dir(UNIVERSE_DIR)

    if universe == "sp500":
        df = fetch_sp500_universe()
        out_path = UNIVERSE_DIR / "symbols_sp500.csv"
        save_universe_csv(df, out_path)
        return df["Symbol"].tolist(), "us", out_path

    if universe == "kospi":
        if tickers_file:
            symbols = load_tickers_from_file(Path(tickers_file))
            limited_symbols = _apply_top_n(symbols, kospi_top_n)
            df = pd.DataFrame({"Symbol": limited_symbols, "Market": "kr", "Source": "file"})
            df = _attach_kr_sector_info(df)
            out_path = UNIVERSE_DIR / "symbols_kospi_from_file.csv"
            save_universe_csv(df, out_path)
            return limited_symbols, "kr", out_path

        df = fetch_kospi_universe_external(kospi_external_url, top_n=kospi_top_n)
        df = _attach_kr_sector_info(df)
        out_path = UNIVERSE_DIR / "symbols_kospi_external.csv"
        save_universe_csv(df, out_path)
        return df["Symbol"].tolist(), "kr", out_path

    if universe == "custom":
        if not tickers_file:
            raise ValueError("--tickers-file is required when --universe custom")
        symbols = load_tickers_from_file(Path(tickers_file))
        inferred_market = "kr" if all(s.endswith((".KS", ".KQ")) or s[:6].isdigit() for s in symbols) else "us"
        df = pd.DataFrame({"Symbol": symbols, "Market": inferred_market, "Source": "file"})
        out_path = UNIVERSE_DIR / "symbols_custom.csv"
        save_universe_csv(df, out_path)
        return symbols, inferred_market, out_path

    raise ValueError(f"Unsupported universe: {universe}")
