from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from market_data.providers.sp500_constituents_provider import BaseSP500ConstituentsProvider

LOGGER = logging.getLogger(__name__)


class WRDSSP500Provider(BaseSP500ConstituentsProvider):
    """Adapter placeholder for licensed WRDS/Compustat constituent history.

    In this repository we default to local cache file loading (if present), so the
    PIT pipeline remains reproducible without direct WRDS network access.
    """

    def __init__(
        self,
        local_csv: str | Path | None = None,
        *,
        enable_live_fetch: bool = True,
    ) -> None:
        super().__init__(name="wrds_compustat", source_tier="licensed")
        self.local_csv = Path(local_csv).expanduser() if local_csv is not None else None
        self.enable_live_fetch = bool(enable_live_fetch)

    @staticmethod
    def _norm_col(df: pd.DataFrame, *candidates: str, default: Any = "") -> pd.Series:
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series([default] * len(df))

    def _convert_wrds_rows_to_events(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        out = df.copy()
        # Common column aliases across WRDS exports / ad-hoc dumps.
        ticker = (
            self._norm_col(out, "ticker", "tic", "symbol", "security")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        add_date = pd.to_datetime(self._norm_col(out, "effective_date", "from_date", "fromdt", "from"), errors="coerce")
        remove_date = pd.to_datetime(self._norm_col(out, "remove_date", "thru_date", "thrudt", "thru"), errors="coerce")
        company = self._norm_col(out, "company_name", "conm", "security_name").astype(str).fillna("")

        rows: list[dict[str, Any]] = []
        for t, a, r, nm in zip(ticker.tolist(), add_date.tolist(), remove_date.tolist(), company.tolist()):
            if not t or pd.isna(a):
                continue
            rows.append(
                {
                    "index_code": "SP500",
                    "ticker": t,
                    "company_name": nm,
                    "action": "add",
                    "effective_date": pd.Timestamp(a).date().isoformat(),
                    "source_ref": "wrds_live_query",
                    "provenance_text": "WRDS index constituents query (add)",
                    "confidence": 0.95,
                }
            )
            if pd.notna(r):
                rows.append(
                    {
                        "index_code": "SP500",
                        "ticker": t,
                        "company_name": nm,
                        "action": "remove",
                        "effective_date": pd.Timestamp(r).date().isoformat(),
                        "source_ref": "wrds_live_query",
                        "provenance_text": "WRDS index constituents query (remove)",
                        "confidence": 0.95,
                    }
                )
        return pd.DataFrame(rows)

    def fetch_events(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        if self.local_csv is not None and self.local_csv.exists():
            return self._read_csv_if_exists(self.local_csv)
        if not self.enable_live_fetch:
            return pd.DataFrame()

        wrds_user = os.getenv("WRDS_USERNAME") or os.getenv("WRDS_USER")
        if not wrds_user:
            LOGGER.info("WRDS provider skipped: WRDS_USERNAME not set")
            return pd.DataFrame()

        try:
            import wrds  # type: ignore
        except Exception:
            LOGGER.warning("WRDS provider unavailable: python package `wrds` not installed")
            return pd.DataFrame()

        sql = os.getenv("WRDS_SP500_EVENTS_SQL", "").strip()
        if not sql:
            # Conservative default: users can override with WRDS_SP500_EVENTS_SQL.
            sql = """
                SELECT tic AS ticker, conm AS company_name, from AS from_date, thru AS thru_date
                FROM comp.idxcst_his
                WHERE gvkeyx IN ('000003','030824')
                  AND from <= %(end)s
                  AND (thru IS NULL OR thru >= %(start)s)
            """
        try:
            db = wrds.Connection(wrds_username=wrds_user)
            raw = db.raw_sql(sql, params={"start": str(start), "end": str(end)})
            db.close()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("WRDS fetch failed: %s", exc)
            return pd.DataFrame()
        if raw is None or raw.empty:
            return pd.DataFrame()
        return self._convert_wrds_rows_to_events(raw)
