from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_data.config import UNIVERSE_DIR
from market_data.providers.sp500_constituents_provider import (
    BaseSP500ConstituentsProvider,
    normalize_sp500_event_frame,
)


class SnapshotSeedSP500Provider(BaseSP500ConstituentsProvider):
    """Fallback provider that seeds PIT from local current S&P 500 snapshot.

    This is intentionally `secondary` tier with conservative confidence.
    """

    def __init__(
        self,
        snapshot_csv: str | Path | None = None,
        *,
        seed_effective_date: str = "1900-01-01",
    ) -> None:
        super().__init__(name="snapshot_seed_secondary", source_tier="secondary")
        self.snapshot_csv = (
            Path(snapshot_csv).expanduser()
            if snapshot_csv is not None
            else (UNIVERSE_DIR / "symbols_sp500.csv")
        )
        self.seed_effective_date = str(seed_effective_date)

    def fetch_events(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        if not self.snapshot_csv.exists():
            return pd.DataFrame()
        try:
            raw = pd.read_csv(self.snapshot_csv)
        except Exception:
            return pd.DataFrame()
        if raw.empty:
            return pd.DataFrame()
        symbol_col = "Symbol" if "Symbol" in raw.columns else raw.columns[0]
        name_col = "Security" if "Security" in raw.columns else ("Name" if "Name" in raw.columns else None)
        rows = pd.DataFrame(
            {
                "index_code": "SP500",
                "ticker": raw[symbol_col].astype(str).str.strip().str.upper(),
                "security_name": raw[name_col].astype(str).fillna("") if name_col else "",
                "identifier_type": "ticker",
                "identifier_value": raw[symbol_col].astype(str).str.strip().str.upper(),
                "event_type": "seed",
                "effective_date": self.seed_effective_date,
                "announcement_date": pd.NaT,
                "source_tier": "secondary",
                "source_name": "current_snapshot_seed",
                "source_url": "",
                "source_doc_id": "",
                "evidence_text": f"seed from local snapshot {self.snapshot_csv}",
                "confidence": 0.35,
                "mapping_note": "fallback seed from current snapshot; survivorship risk",
            }
        )
        return normalize_sp500_event_frame(
            rows,
            source_tier=self.source_tier,
            source_name=self.name,
            default_event_type="seed",
        )

