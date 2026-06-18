from __future__ import annotations

import pandas as pd

from market_data.kr_dart.client import DartClient, dumps_json
from market_data.utils import now_utc_iso


def _infer_report_code(report_name: object) -> str | None:
    text = str(report_name or "").strip()
    if not text:
        return None
    if "사업보고서" in text:
        return "11011"
    if "반기보고서" in text:
        return "11012"
    if "분기보고서" in text and ".03" in text:
        return "11013"
    if "분기보고서" in text and ".09" in text:
        return "11014"
    return None


def _infer_period_end(report_name: object) -> pd.Timestamp | pd.NaT:
    text = str(report_name or "").strip()
    match = pd.Series([text]).str.extract(r"(\d{4})\.(\d{2})", expand=True).iloc[0]
    year = pd.to_numeric(match.iloc[0], errors="coerce")
    month = pd.to_numeric(match.iloc[1], errors="coerce")
    if pd.isna(year) or pd.isna(month):
        return pd.NaT
    try:
        return pd.Timestamp(year=int(year), month=int(month), day=1) + pd.offsets.MonthEnd(0)
    except ValueError:
        return pd.NaT


def fetch_filings_for_corp(
    *,
    corp_code: str,
    ticker: str,
    ticker_name: str | None = None,
    start_date: str,
    end_date: str | None = None,
    client: DartClient | None = None,
) -> pd.DataFrame:
    dart = client or DartClient()
    rows: list[dict[str, object]] = []
    page_no = 1
    total_pages = 1
    while page_no <= total_pages:
        payload = dart.list_filings(
            corp_code=corp_code,
            bgn_de=start_date,
            end_de=end_date,
            page_no=page_no,
            page_count=100,
        )
        listing = payload.get("list") or []
        if isinstance(listing, list):
            rows.extend(listing)
        total_count = int(payload.get("total_count", 0) or 0)
        page_count = int(payload.get("page_count", 100) or 100)
        total_pages = max(1, (total_count + page_count - 1) // page_count)
        page_no += 1

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "market",
                "accession",
                "corp_code",
                "corp_name",
                "stock_code",
                "report_name",
                "report_code",
                "period_end",
                "filing_date",
                "available_date",
                "accepted_at",
                "receipt_no",
                "filer_name",
                "remarks",
                "source_url",
                "raw_payload",
                "collected_at",
            ]
        )

    out = pd.DataFrame(rows)
    out["ticker"] = ticker
    out["market"] = "kr"
    out["corp_code"] = corp_code
    if ticker_name is not None:
        out["corp_name"] = out.get("corp_name").fillna(ticker_name)
    else:
        out["corp_name"] = out.get("corp_name").fillna("")
    out["stock_code"] = out.get("stock_code", pd.Series(dtype=object)).astype(str).str.extract(r"(\d{6})", expand=False)
    out["accession"] = out.get("rcept_no", pd.Series(dtype=object)).astype(str).str.strip()
    out["receipt_no"] = out["accession"]
    out["report_name"] = out.get("report_nm")
    out["report_code"] = out["report_name"].map(_infer_report_code)
    out["period_end"] = out["report_name"].map(_infer_period_end)
    out["filing_date"] = pd.to_datetime(out.get("rcept_dt"), errors="coerce").dt.date
    out["available_date"] = out["filing_date"]
    out["accepted_at"] = pd.NaT
    out["filer_name"] = out.get("flr_nm")
    out["remarks"] = out.get("rm")
    out["source_url"] = out["accession"].map(
        lambda value: f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={value}" if str(value).strip() else None
    )
    out["raw_payload"] = out.apply(lambda row: dumps_json(row.to_dict()), axis=1)
    out["collected_at"] = pd.Timestamp(now_utc_iso())
    return out[
        [
            "ticker",
            "market",
            "accession",
            "corp_code",
            "corp_name",
            "stock_code",
            "report_name",
            "report_code",
            "period_end",
            "filing_date",
            "available_date",
            "accepted_at",
            "receipt_no",
            "filer_name",
            "remarks",
            "source_url",
            "raw_payload",
            "collected_at",
        ]
    ].drop_duplicates(subset=["ticker", "market", "accession"], keep="last").reset_index(drop=True)
