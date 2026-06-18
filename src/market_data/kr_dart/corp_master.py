from __future__ import annotations

import re

import pandas as pd

from market_data.db_router import normalize_kr_ticker
from market_data.kr_dart.client import DartClient, dumps_json
from market_data.kr_ksic import normalize_ksic_code
from market_data.utils import now_utc_iso

_SECTION_RANGE_RE = re.compile(r"\s*\(\d{2}\s*[~\-]\s*\d{2}\)\s*$")
_HANGUL_SYLLABLE_RE = re.compile(r"^[가-힣]$")


def _empty_corp_master() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "corp_code",
            "corp_name",
            "stock_code",
            "stock_name",
            "corp_name_eng",
            "corp_cls",
            "induty_code",
            "ceo_name",
            "accounting_month",
            "established_date",
            "homepage_url",
            "address",
            "modify_date",
            "market_tier",
            "ticker",
            "is_common_stock",
            "raw_payload",
            "collected_at",
        ]
    )


def _normalize_company_payload(payload: dict[str, object]) -> dict[str, object]:
    established = pd.to_datetime(payload.get("est_dt"), errors="coerce")
    return {
        "corp_code": str(payload.get("corp_code") or "").strip(),
        "stock_code": str(payload.get("stock_code") or "").strip(),
        "stock_name": str(payload.get("stock_name") or "").strip() or pd.NA,
        "corp_name": str(payload.get("corp_name") or "").strip() or pd.NA,
        "corp_name_eng": str(payload.get("corp_name_eng") or "").strip() or pd.NA,
        "corp_cls": str(payload.get("corp_cls") or "").strip() or pd.NA,
        "induty_code": normalize_ksic_code(payload.get("induty_code")) or pd.NA,
        "ceo_name": str(payload.get("ceo_nm") or "").strip() or pd.NA,
        "accounting_month": str(payload.get("acc_mt") or "").strip() or pd.NA,
        "established_date": established.date() if pd.notna(established) else pd.NaT,
        "homepage_url": str(payload.get("hm_url") or "").strip() or pd.NA,
        "address": str(payload.get("adres") or "").strip() or pd.NA,
        "_company_payload": payload,
    }


def _fetch_company_details(
    corp_master: pd.DataFrame,
    *,
    client: DartClient,
    tickers: list[str],
) -> pd.DataFrame:
    if corp_master.empty or not tickers:
        return _empty_corp_master().drop(columns=["modify_date", "market_tier", "is_common_stock", "raw_payload", "collected_at"]).iloc[0:0]

    normalized_tickers = {
        normalize_kr_ticker(value)
        for value in tickers
        if str(value or "").strip()
    }
    targets = corp_master.loc[corp_master["ticker"].astype(str).isin(normalized_tickers)].copy()
    targets = targets.dropna(subset=["corp_code"]).drop_duplicates(subset=["corp_code"], keep="last")
    if targets.empty:
        return _empty_corp_master().drop(columns=["modify_date", "market_tier", "is_common_stock", "raw_payload", "collected_at"]).iloc[0:0]

    rows: list[dict[str, object]] = []
    for _, item in targets.iterrows():
        corp_code = str(item.get("corp_code") or "").strip()
        if not corp_code:
            continue
        try:
            payload = client.company(corp_code)
        except Exception:
            continue
        rows.append(_normalize_company_payload(payload))

    if not rows:
        return _empty_corp_master().drop(columns=["modify_date", "market_tier", "is_common_stock", "raw_payload", "collected_at"]).iloc[0:0]
    return pd.DataFrame(rows).drop_duplicates(subset=["corp_code"], keep="last").reset_index(drop=True)


def fetch_corp_master(
    client: DartClient | None = None,
    *,
    ticker_master: pd.DataFrame | None = None,
    enrich_company: bool = False,
) -> pd.DataFrame:
    dart = client or DartClient()
    rows = dart.get_corp_codes()
    if not rows:
        return _empty_corp_master()

    out = pd.DataFrame(rows)
    out["stock_code"] = (
        out.get("stock_code", pd.Series(dtype=object))
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
    )
    out["ticker"] = out["stock_code"].map(
        lambda value: normalize_kr_ticker(value)
        if pd.notna(value) and str(value).strip()
        else pd.NA
    )
    out["modify_date"] = pd.to_datetime(out.get("modify_date"), errors="coerce").dt.date
    out["stock_name"] = pd.NA
    out["corp_name_eng"] = pd.NA
    out["corp_cls"] = pd.NA
    out["induty_code"] = pd.NA
    out["ceo_name"] = pd.NA
    out["accounting_month"] = pd.NA
    out["established_date"] = pd.NaT
    out["homepage_url"] = pd.NA
    out["address"] = pd.NA

    if ticker_master is not None and not ticker_master.empty:
        meta = ticker_master[
            ["ticker", "market_tier", "is_common_stock"]
        ].drop_duplicates(subset=["ticker"])
        out = out.merge(meta, on="ticker", how="left")
    else:
        out["market_tier"] = pd.NA
        out["is_common_stock"] = pd.NA

    if enrich_company and ticker_master is not None and not ticker_master.empty:
        details = _fetch_company_details(
            out[["corp_code", "corp_name", "stock_code", "modify_date", "market_tier", "ticker", "is_common_stock"]].copy(),
            client=dart,
            tickers=ticker_master["ticker"].astype(str).tolist(),
        )
        if not details.empty:
            out = out.merge(
                details[
                    [
                        "corp_code",
                        "stock_name",
                        "corp_name",
                        "corp_name_eng",
                        "corp_cls",
                        "induty_code",
                        "ceo_name",
                        "accounting_month",
                        "established_date",
                        "homepage_url",
                        "address",
                    ]
                ],
                on="corp_code",
                how="left",
                suffixes=("", "_detail"),
            )
            for column in (
                "stock_name",
                "corp_name",
                "corp_name_eng",
                "corp_cls",
                "induty_code",
                "ceo_name",
                "accounting_month",
                "established_date",
                "homepage_url",
                "address",
            ):
                detail_column = f"{column}_detail"
                if detail_column in out.columns:
                    out[column] = out[detail_column].where(out[detail_column].notna(), out.get(column))
                    out = out.drop(columns=[detail_column])

    out["raw_payload"] = out.apply(lambda row: dumps_json(row.to_dict()), axis=1)
    out["collected_at"] = pd.Timestamp(now_utc_iso())
    return out[
        [
            "corp_code",
            "corp_name",
            "stock_code",
            "stock_name",
            "corp_name_eng",
            "corp_cls",
            "induty_code",
            "ceo_name",
            "accounting_month",
            "established_date",
            "homepage_url",
            "address",
            "modify_date",
            "market_tier",
            "ticker",
            "is_common_stock",
            "raw_payload",
            "collected_at",
        ]
    ].drop_duplicates(subset=["corp_code"], keep="last").reset_index(drop=True)


def _clean_section_name(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    text = _SECTION_RANGE_RE.sub("", text).strip() or text
    tokens = [token for token in text.split() if token]
    if tokens and all(_HANGUL_SYLLABLE_RE.fullmatch(token) for token in tokens):
        return "".join(tokens)
    return text


def merge_with_ticker_master(
    ticker_master: pd.DataFrame,
    corp_master: pd.DataFrame,
    ksic_dim: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if ticker_master is None or ticker_master.empty:
        return ticker_master if ticker_master is not None else pd.DataFrame()
    base = ticker_master.copy()
    for column in (
        "krx_industry_name",
        "induty_code",
        "ksic_name_ko",
        "ksic_name_en",
        "sector_code",
        "subsector_code",
        "subsector_name",
        "classification_source",
        "dart_corp_code",
        "dart_corp_name",
    ):
        if column not in base.columns:
            base[column] = pd.NA

    corp_frame = corp_master.copy() if corp_master is not None else pd.DataFrame()
    if corp_frame.empty:
        return base
    for column in ("corp_code", "corp_name", "stock_code", "induty_code"):
        if column not in corp_frame.columns:
            corp_frame[column] = pd.NA

    merged = base.merge(
        corp_frame[["corp_code", "corp_name", "stock_code", "induty_code"]],
        left_on="ticker",
        right_on="stock_code",
        how="left",
    )
    merged["dart_corp_code"] = merged["corp_code"].fillna(merged.get("dart_corp_code"))
    merged["dart_corp_name"] = merged["corp_name"].fillna(merged.get("dart_corp_name"))
    left_induty = (
        merged["induty_code_x"]
        if "induty_code_x" in merged.columns
        else pd.Series(pd.NA, index=merged.index, dtype=object)
    )
    if "induty_code_y" in merged.columns:
        right_induty = merged["induty_code_y"]
    elif "induty_code" in merged.columns:
        right_induty = merged["induty_code"]
    else:
        right_induty = pd.Series(pd.NA, index=merged.index, dtype=object)
    merged["induty_code"] = right_induty.fillna(left_induty)
    merged["krx_industry_name"] = merged.get("krx_industry_name").fillna(merged.get("industry_name"))
    merged["classification_source"] = merged.get("classification_source").fillna("krx_corp_list")

    if ksic_dim is not None and not ksic_dim.empty:
        ksic_cols = [
            "ksic_code",
            "name_ko",
            "name_en",
            "section_code",
            "section_name_ko",
            "division_code",
            "division_name_ko",
            "revision",
        ]
        merged = merged.merge(
            ksic_dim[ksic_cols].drop_duplicates(subset=["ksic_code"]),
            left_on="induty_code",
            right_on="ksic_code",
            how="left",
        )
        merged["ksic_name_ko"] = merged["name_ko"].fillna(merged.get("ksic_name_ko"))
        merged["ksic_name_en"] = merged["name_en"].fillna(merged.get("ksic_name_en"))
        merged["sector_code"] = merged["section_code"].fillna(merged.get("sector_code"))
        merged["subsector_code"] = merged["division_code"].fillna(merged.get("subsector_code"))
        merged["sector_name"] = merged["section_name_ko"].map(_clean_section_name).fillna(merged.get("sector_name"))
        merged["subsector_name"] = merged["division_name_ko"].fillna(merged.get("subsector_name"))
        merged["industry_name"] = merged["name_ko"].fillna(merged.get("industry_name"))
        revision_series = merged.get("revision", pd.Series(dtype=object))
        has_ksic = merged["ksic_name_ko"].notna()
        merged.loc[has_ksic, "classification_source"] = [
            f"dart_company+ksic_{int(value)}" if pd.notna(value) else "dart_company+ksic"
            for value in revision_series.loc[has_ksic]
        ]
        has_induty = merged["induty_code"].notna() & ~has_ksic
        merged.loc[has_induty, "classification_source"] = "dart_company"

    return merged.drop(
        columns=[
            col
            for col in (
                "corp_code",
                "corp_name",
                "stock_code",
                "induty_code_x",
                "induty_code_y",
                "ksic_code",
                "name_ko",
                "name_en",
                "section_code",
                "section_name_ko",
                "division_code",
                "division_name_ko",
                "revision",
            )
            if col in merged.columns
        ]
    )
