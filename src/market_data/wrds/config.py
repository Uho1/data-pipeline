from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from market_data.config import DATA_DIR, LOGS_DIR
from market_data.db import DB_PATH

DEFAULT_SAMPLE_YEARS = 2
WRDS_LAKE_DB_PATH = DATA_DIR / "wrds_market_data.duckdb"
WRDS_KR_LAKE_DB_PATH = DATA_DIR / "wrds_kr_market_data.duckdb"


@dataclass(frozen=True)
class WRDSRelationConfig:
    """Configurable WRDS relation names used by the default SQL templates."""

    crsp_daily: str = "crsp.dsf"
    crsp_delist: str = "crsp.dsedelist"
    security_master: str = "crsp.stocknames"
    company_master: str = "comp.company"
    compustat_quarterly: str = "comp.fundq"
    compustat_annual: str = "comp.funda"
    compustat_segments_merged: str = "comp.wrds_segmerged"
    compustat_segments_geo: str = "comp.seg_geo"
    compustat_segments_product: str = "comp.seg_product"
    compustat_segments_customer: str = "comp.seg_customer"
    compustat_segment_names: str = "comp.names_seg"
    compustat_quarterly_ytd: str = "comp.co_ifndytd"
    compustat_quarterly_semi: str = "comp.co_ifndsa"
    compustat_quarterly_flags: str = "comp.co_ifntq"
    compustat_security_quarterly: str = "comp.sec_ifnd"
    compustat_security_quarterly_flags: str = "comp.sec_ifnt"
    compustat_quarterly_fncd: str = "comp.fundq_fncd"
    compustat_annual_fncd: str = "comp.funda_fncd"
    ibes_actuals_epsus: str = "ibes.act_epsus"
    ibes_summary_epsus: str = "ibes.statsum_epsus"
    ibes_guidance: str = "ibes.det_guidance"
    ccm_link: str = "crsp.ccmxpf_linktable"
    index_membership: str = "comp.idxcst_his"


@dataclass(frozen=True)
class WRDSIndexConfig:
    """Configurable identifiers for index membership history adapters."""

    sp500_index_ids: tuple[str, ...] = ("000003", "030824")
    nasdaq100_index_ids: tuple[str, ...] = ()
    enable_sp500: bool = True
    enable_nasdaq100: bool = False


@dataclass(frozen=True)
class WRDSChunkConfig:
    """Chunk sizes for the large historical source pulls."""

    crsp_daily_months: int = 12
    compustat_quarterly_months: int = 24
    compustat_annual_months: int = 60
    compustat_segments_months: int = 24
    compustat_quarterly_variant_months: int = 24
    ibes_actuals_months: int = 60
    ibes_summary_months: int = 60


@dataclass(frozen=True)
class WRDSSettings:
    """Runtime settings for WRDS ingestion and canonical table builds."""

    db_path: Path = DB_PATH
    log_dir: Path = LOGS_DIR / "wrds"
    manifest_dir: Path = DATA_DIR / "wrds_manifests"
    wrds_username: str | None = field(default_factory=lambda: os.getenv("WRDS_USERNAME") or os.getenv("WRDS_USER"))
    wrds_password: str | None = field(default_factory=lambda: os.getenv("WRDS_PASSWORD") or os.getenv("PGPASSWORD"))
    allow_interactive_password_prompt: bool = True
    sample_mode: bool = False
    sample_years: int = DEFAULT_SAMPLE_YEARS
    dry_run: bool = False
    force: bool = False
    start_date: date | None = None
    end_date: date | None = None
    max_retries: int = 3
    backoff_seconds: float = 2.0
    sample_row_limit: int = 5000
    relations: WRDSRelationConfig = field(default_factory=WRDSRelationConfig)
    indexes: WRDSIndexConfig = field(default_factory=WRDSIndexConfig)
    chunking: WRDSChunkConfig = field(default_factory=WRDSChunkConfig)
    sql_overrides: dict[str, str] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def resolved_start_date(self) -> date | None:
        """Return the effective start date after sample-mode narrowing."""
        start = self.start_date
        if not self.sample_mode:
            return start
        cutoff = date.today() - timedelta(days=365 * max(1, self.sample_years))
        if start is None:
            return cutoff
        return max(start, cutoff)

    def resolved_end_date(self) -> date:
        """Return the effective inclusive end date."""
        return self.end_date or date.today()

    def sql_override(self, dataset_name: str) -> str | None:
        """Return a configured SQL override for a dataset, if present."""
        text = self.sql_overrides.get(dataset_name)
        if text is None:
            return None
        stripped = str(text).strip()
        return stripped or None


def _parse_date(value: Any) -> date | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value)).date()


def _coerce_tuple(values: Any) -> tuple[str, ...]:
    if values in (None, "", []):
        return ()
    if isinstance(values, (list, tuple)):
        return tuple(str(v).strip() for v in values if str(v).strip())
    return (str(values).strip(),)


def load_wrds_settings(
    *,
    config_path: str | Path | None = None,
    db_path: str | Path | None = None,
    wrds_username: str | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    sample_mode: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> WRDSSettings:
    """Load WRDS settings from defaults plus an optional JSON override file."""

    settings = WRDSSettings(
        db_path=Path(db_path).expanduser() if db_path is not None else DB_PATH,
        wrds_username=wrds_username or (os.getenv("WRDS_USERNAME") or os.getenv("WRDS_USER")),
        wrds_password=os.getenv("WRDS_PASSWORD") or os.getenv("PGPASSWORD"),
        sample_mode=bool(sample_mode),
        dry_run=bool(dry_run),
        force=bool(force),
        start_date=_parse_date(start_date),
        end_date=_parse_date(end_date),
    )

    if config_path is None:
        return settings

    payload = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    relations = replace(settings.relations, **payload.get("relations", {}))
    indexes = replace(
        settings.indexes,
        sp500_index_ids=_coerce_tuple(payload.get("indexes", {}).get("sp500_index_ids", settings.indexes.sp500_index_ids)),
        nasdaq100_index_ids=_coerce_tuple(
            payload.get("indexes", {}).get("nasdaq100_index_ids", settings.indexes.nasdaq100_index_ids)
        ),
        enable_sp500=bool(payload.get("indexes", {}).get("enable_sp500", settings.indexes.enable_sp500)),
        enable_nasdaq100=bool(payload.get("indexes", {}).get("enable_nasdaq100", settings.indexes.enable_nasdaq100)),
    )
    chunking = replace(settings.chunking, **payload.get("chunking", {}))
    notes = dict(settings.notes)
    notes.update(payload.get("notes", {}))

    return replace(
        settings,
        db_path=Path(payload.get("db_path", settings.db_path)).expanduser(),
        log_dir=Path(payload.get("log_dir", settings.log_dir)).expanduser(),
        manifest_dir=Path(payload.get("manifest_dir", settings.manifest_dir)).expanduser(),
        wrds_username=payload.get("wrds_username", settings.wrds_username),
        wrds_password=settings.wrds_password,
        allow_interactive_password_prompt=bool(
            payload.get("allow_interactive_password_prompt", settings.allow_interactive_password_prompt)
        ),
        sample_mode=bool(payload.get("sample_mode", settings.sample_mode)),
        sample_years=int(payload.get("sample_years", settings.sample_years)),
        dry_run=bool(payload.get("dry_run", settings.dry_run)),
        force=bool(payload.get("force", settings.force)),
        start_date=_parse_date(payload.get("start_date", settings.start_date)),
        end_date=_parse_date(payload.get("end_date", settings.end_date)),
        max_retries=int(payload.get("max_retries", settings.max_retries)),
        backoff_seconds=float(payload.get("backoff_seconds", settings.backoff_seconds)),
        sample_row_limit=int(payload.get("sample_row_limit", settings.sample_row_limit)),
        relations=relations,
        indexes=indexes,
        chunking=chunking,
        sql_overrides=dict(payload.get("sql_overrides", settings.sql_overrides)),
        notes=notes,
    )
