from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_data.providers.sp500_constituents_provider import BaseSP500ConstituentsProvider

LOGGER = logging.getLogger(__name__)

DEFAULT_SPDJI_ANNOUNCEMENTS_URL = "https://www.spglobal.com/spdji/en/index-announcements/"


def _fetch_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": os.getenv(
                "SPDJI_USER_AGENT",
                "market_data_lake/1.0 (+https://example.local/sp500-pit)",
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - controlled URL
        payload = resp.read()
    return payload.decode("utf-8", errors="ignore")


def _parse_date_text(text: str) -> str | None:
    s = str(text or "").strip()
    if not s:
        return None
    for fmt in [
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
    ]:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            continue
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).date().isoformat()


def _extract_first_date(text: str) -> str | None:
    s = str(text or "")
    candidates = re.findall(r"([A-Z][a-z]+ \d{1,2}, \d{4})", s)
    if not candidates:
        candidates = re.findall(r"(\d{4}-\d{2}-\d{2})", s)
    if not candidates:
        candidates = re.findall(r"(\d{1,2}/\d{1,2}/\d{4})", s)
    for c in candidates:
        parsed = _parse_date_text(c)
        if parsed:
            return parsed
    return None


@dataclass
class AnnouncementEntry:
    title: str
    url: str
    announcement_date: str | None


def parse_spdji_listing_html(html: str, base_url: str) -> list[AnnouncementEntry]:
    text = str(html or "")
    entries: list[AnnouncementEntry] = []
    anchor_pat = re.compile(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
    seen: set[str] = set()
    for href, raw_title in anchor_pat.findall(text):
        title = re.sub(r"<[^>]+>", " ", raw_title)
        title = re.sub(r"\s+", " ", title).strip()
        if not title:
            continue
        if "s&p 500" not in title.lower() and "s&p dow jones" not in title.lower():
            continue
        url = urllib.parse.urljoin(base_url, href.strip())
        if not url or url in seen:
            continue
        seen.add(url)
        ann_date = _extract_first_date(title)
        entries.append(AnnouncementEntry(title=title, url=url, announcement_date=ann_date))
    return entries


def parse_spdji_announcement_events(
    *,
    title: str,
    body_text: str,
    source_url: str,
    source_doc_id: str = "",
) -> pd.DataFrame:
    txt = f"{title}\n{body_text}"
    if "s&p 500" not in txt.lower():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []

    # Pattern 1: explicit exchange ticker form.
    exch_replace = re.search(
        r"\((?:NYSE|NASDAQ|NYSEARCA|NYSE American):\s*([A-Z\.\-]{1,10})\)\s+will replace\s+.*?\((?:NYSE|NASDAQ|NYSEARCA|NYSE American):\s*([A-Z\.\-]{1,10})\)\s+in the S&P 500",
        txt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if exch_replace:
        add_t = exch_replace.group(1).upper()
        rm_t = exch_replace.group(2).upper()
        eff = _extract_first_date(re.search(r"effective[^.]*", txt, flags=re.IGNORECASE | re.DOTALL).group(0) if re.search(r"effective[^.]*", txt, flags=re.IGNORECASE | re.DOTALL) else txt)
        if eff:
            rows.append({"action": "add", "ticker": add_t, "effective_date": eff})
            rows.append({"action": "remove", "ticker": rm_t, "effective_date": eff})

    # Pattern 2: bare ticker sentence.
    bare_replace = re.search(
        r"\b([A-Z]{1,6})\b\s+will replace\s+\b([A-Z]{1,6})\b\s+in the S&P 500",
        txt,
        flags=re.IGNORECASE,
    )
    if bare_replace:
        add_t = bare_replace.group(1).upper()
        rm_t = bare_replace.group(2).upper()
        eff = _extract_first_date(re.search(r"effective[^.]*", txt, flags=re.IGNORECASE | re.DOTALL).group(0) if re.search(r"effective[^.]*", txt, flags=re.IGNORECASE | re.DOTALL) else txt)
        if eff:
            rows.append({"action": "add", "ticker": add_t, "effective_date": eff})
            rows.append({"action": "remove", "ticker": rm_t, "effective_date": eff})

    # Pattern 3: simple add/remove wording.
    for pat, action in [
        (r"\b([A-Z]{1,6})\b\s+(?:will be|to be|is)\s+added to the S&P 500", "add"),
        (r"\b([A-Z]{1,6})\b\s+(?:will be|to be|is)\s+removed from the S&P 500", "remove"),
    ]:
        for m in re.finditer(pat, txt, flags=re.IGNORECASE):
            sym = m.group(1).upper()
            eff = _extract_first_date(txt)
            if eff:
                rows.append({"action": action, "ticker": sym, "effective_date": eff})

    if not rows:
        return pd.DataFrame()

    ann = _extract_first_date(title) or _extract_first_date(txt)
    out = pd.DataFrame(rows).drop_duplicates(subset=["action", "ticker", "effective_date"]).reset_index(drop=True)
    out["index_code"] = "SP500"
    out["announcement_date"] = ann
    out["source_name"] = "spdji_announcements"
    out["source_type"] = "official"
    out["source_ref"] = source_url
    out["source_url"] = source_url
    out["source_doc_id"] = source_doc_id
    out["provenance_text"] = title
    out["evidence_text"] = title
    # If parsed via weak regex path, confidence is lower.
    out["confidence"] = 0.9 if exch_replace or bare_replace else 0.75
    out["note"] = "parsed_from_spdji_announcement"
    return out


class SPDJIAnnouncementsProvider(BaseSP500ConstituentsProvider):
    """Provider for S&P DJI announcement based PIT events.

    Priority:
    1) local normalized CSV cache
    2) incremental fetch/parse from announcement page
    """

    def __init__(
        self,
        local_csv: str | Path | None = None,
        *,
        feed_url: str | None = None,
        enable_live_fetch: bool = True,
        fetch_limit: int | None = None,
    ) -> None:
        super().__init__(name="spdji_announcements", source_tier="official")
        self.local_csv = Path(local_csv).expanduser() if local_csv is not None else None
        self.feed_url = str(feed_url or os.getenv("SPDJI_ANNOUNCEMENTS_URL", DEFAULT_SPDJI_ANNOUNCEMENTS_URL))
        self.enable_live_fetch = bool(enable_live_fetch)
        self.fetch_limit = int(fetch_limit or int(os.getenv("SPDJI_FETCH_MAX_PAGES", "20") or "20"))

    def _cache_dir(self) -> Path | None:
        if self.local_csv is None:
            return None
        return self.local_csv.parent

    def _load_existing(self) -> pd.DataFrame:
        if self.local_csv is None or not self.local_csv.exists():
            return pd.DataFrame()
        return self._read_csv_if_exists(self.local_csv)

    def _write_existing(self, df: pd.DataFrame) -> None:
        if self.local_csv is None:
            return
        try:
            self.local_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.local_csv, index=False, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("failed to write spdji cache csv: %s", exc)

    def fetch_events(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        existing = self._load_existing()
        if not self.enable_live_fetch:
            return existing

        try:
            listing_html = _fetch_text(self.feed_url)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("SPDJI fetch failed (listing): %s", exc)
            return existing

        cache_dir = self._cache_dir()
        if cache_dir is not None:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / "spdji_announcements_index.html").write_text(listing_html, encoding="utf-8")
            except Exception:
                pass

        entries = parse_spdji_listing_html(listing_html, self.feed_url)
        if not entries:
            return existing

        known_refs = set(existing.get("source_ref", pd.Series(dtype=str)).astype(str).tolist()) if not existing.empty else set()
        new_rows: list[pd.DataFrame] = []

        for ent in entries[: max(self.fetch_limit, 1)]:
            if ent.url in known_refs:
                continue
            try:
                page_html = _fetch_text(ent.url)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("SPDJI fetch failed (announcement): %s", exc)
                continue

            if cache_dir is not None:
                try:
                    doc_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", ent.url)[-120:]
                    (cache_dir / f"spdji_{doc_id}.html").write_text(page_html, encoding="utf-8")
                except Exception:
                    pass

            body_text = re.sub(r"<[^>]+>", " ", page_html)
            body_text = re.sub(r"\s+", " ", body_text).strip()
            parsed = parse_spdji_announcement_events(
                title=ent.title,
                body_text=body_text,
                source_url=ent.url,
                source_doc_id="",
            )
            if parsed is not None and not parsed.empty:
                if ent.announcement_date:
                    mask = parsed["announcement_date"].astype(str).str.strip().eq("")
                    parsed.loc[mask, "announcement_date"] = ent.announcement_date
                new_rows.append(parsed)

        if not new_rows:
            return existing

        new_df = pd.concat(new_rows, ignore_index=True, sort=False)
        merged = pd.concat([existing, new_df], ignore_index=True, sort=False) if not existing.empty else new_df
        # Keep latest parse for same source_ref/action/ticker/effective_date.
        for col in ["source_ref", "action", "ticker", "effective_date"]:
            if col not in merged.columns:
                merged[col] = ""
        merged = merged.drop_duplicates(
            subset=["source_ref", "action", "ticker", "effective_date"],
            keep="last",
        ).reset_index(drop=True)
        self._write_existing(merged)
        return merged


__all__ = [
    "SPDJIAnnouncementsProvider",
    "AnnouncementEntry",
    "parse_spdji_listing_html",
    "parse_spdji_announcement_events",
]

