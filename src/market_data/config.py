from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(os.environ.get("MDL_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("MDL_DATA_DIR", ROOT_DIR / "data"))
PARQUET_DIR = DATA_DIR / "parquet"

# Storage backend: "duckdb" (file-based, default) or "parquet" (Parquet files + in-memory DuckDB)
STORAGE_BACKEND = os.environ.get("MDL_STORAGE", "parquet")
UNIVERSE_DIR = DATA_DIR / "universe"
PRICES_DIR = DATA_DIR / "prices"
FINANCIALS_DIR = DATA_DIR / "financials"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
KOSPI_EXTERNAL_DEFAULT_URL = "https://finance.naver.com/sise/sise_market_sum.nhn?sosok=0&page=1"
KRX_CORP_LIST_URL = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
KSSC_KSIC_BROWSE_URL = "https://kssc.mods.go.kr:8443/ksscNew_web/kssc/common/selectTscsList.do"
KSSC_KSIC_TREE_URL = "https://kssc.mods.go.kr:8443/ksscNew_web/kssc/common/selectTscsListTree.do"
KSSC_KSIC_REVISION = 11
KOSPI_TOP_N_DEFAULT = 500

PRICE_REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "MarketCap",
]

FINANCIAL_FILE_MAP = {
    "income_annual": "income_stmt",
    "income_quarterly": "quarterly_income_stmt",
    "balance_annual": "balance_sheet",
    "balance_quarterly": "quarterly_balance_sheet",
    "cashflow_annual": "cashflow",
    "cashflow_quarterly": "quarterly_cashflow",
}
