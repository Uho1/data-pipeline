"""Settings loaded from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve repo root from this file: web/backend/core/config.py → 3 levels up
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_data_dir() -> Path:
    """Find the best data directory: env var → repo root → cwd."""
    env_val = os.environ.get("MDL_DATA_DIR", "").strip()
    if env_val:
        p = Path(env_val)
        # Validate: must have meta/ticker_master_kr.json or tickers/ dir
        if (p / "meta" / "ticker_master_kr.json").exists() or (p / "tickers").exists():
            return p
    # Default: repo root data/
    return _REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[
            str(_REPO_ROOT / ".env"),  # project root .env
            ".env",                     # cwd .env
        ],
        env_prefix="MDL_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Market Data Lake API"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    # Paths — will be validated in model_post_init
    data_dir: Path = _resolve_data_dir()
    db_path: Path = _resolve_data_dir() / "market_data.duckdb"

    def model_post_init(self, __context: object) -> None:
        """Validate data_dir after pydantic loads env vars."""
        # Check if current data_dir has actual ticker JSON files
        # (not just an empty directory)
        has_ticker_json = (self.data_dir / "tickers" / "kr" / "005930.json").exists()
        has_meta = (self.data_dir / "meta" / "ticker_master_kr.json").exists()
        if has_ticker_json and has_meta:
            return  # current data_dir is good

        # Fall back to repo root data/
        fallback = _REPO_ROOT / "data"
        if (fallback / "tickers" / "kr" / "005930.json").exists():
            object.__setattr__(self, "data_dir", fallback)
            object.__setattr__(self, "db_path", fallback / "market_data.duckdb")

    # Cloudflare R2 public base URL (e.g. https://pub-xxxx.r2.dev)
    # When set, json_data_service fetches JSON from R2 instead of local disk.
    r2_public_url: str = ""  # set via MDL_R2_PUBLIC_URL env var

    # iTick real-time quote API (https://docs.itick.org)
    itick_api_token: str = ""  # set via MDL_ITICK_API_TOKEN env var

    # News APIs
    naver_client_id: str = ""
    naver_client_secret: str = ""
    finnhub_api_key: str = ""

    # Supabase — read from MDL_SUPABASE_URL / MDL_SUPABASE_SERVICE_ROLE_KEY
    # or fall back to SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY (no prefix)
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # Upstash Redis — read from MDL_UPSTASH_REDIS_URL or fall back to unprefixed
    upstash_redis_url: str = ""

    @property
    def redis_url(self) -> str:
        return self.upstash_redis_url or os.environ.get("UPSTASH_REDIS_URL", "")

    # OpenAI (AI summaries) — no MDL_ prefix; read directly from env
    @property
    def openai_api_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY", "")

    @property
    def openai_finance_analysis_model(self) -> str:
        return os.environ.get("OPENAI_FINANCE_ANALYSIS_MODEL", "gpt-4o-mini")

    @property
    def supabase_url_resolved(self) -> str:
        return self.supabase_url or os.environ.get("SUPABASE_URL", "")

    @property
    def supabase_service_role_key_resolved(self) -> str:
        return self.supabase_service_role_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


settings = Settings()
