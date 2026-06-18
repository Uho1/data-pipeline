from __future__ import annotations

from typing import Any

import pandas as pd


ANNUAL_FORM_MARKERS: tuple[str, ...] = (
    "10-K",
    "10-K/A",
    "20-F",
    "20-F/A",
    "40-F",
    "40-F/A",
    "FY",
    "ANNUAL",
)


def is_annual_form(form_type: Any) -> bool:
    if pd.isna(form_type):
        return False
    form = str(form_type).strip().upper()
    return any(marker in form for marker in ANNUAL_FORM_MARKERS)


def infer_fiscal_period_meta(
    period_end: pd.Series | pd.Index | list[Any],
    form_type: pd.Series | list[Any] | None = None,
    period_start: pd.Series | pd.Index | list[Any] | None = None,
) -> pd.DataFrame:
    period_series = pd.Series(period_end, copy=False)
    form_series = pd.Series(form_type, copy=False) if form_type is not None else pd.Series([None] * len(period_series))
    start_series = pd.Series(period_start, copy=False) if period_start is not None else pd.Series([None] * len(period_series))
    if len(form_series) != len(period_series):
        form_series = form_series.reindex(range(len(period_series)))
    if len(start_series) != len(period_series):
        start_series = start_series.reindex(range(len(period_series)))

    temp = pd.DataFrame(
        {
            "period_end": pd.to_datetime(period_series, errors="coerce").dt.normalize(),
            "period_start": pd.to_datetime(start_series, errors="coerce").dt.normalize(),
            "form_type": form_series.astype("string"),
        }
    )
    temp = temp.loc[~temp["period_end"].isna()].copy()
    if temp.empty:
        return pd.DataFrame(columns=["period_end", "fiscal_year", "fiscal_quarter", "fiscal_label"])

    temp = temp.sort_values("period_end").drop_duplicates(subset=["period_end"], keep="last").reset_index(drop=True)
    duration_days = (temp["period_end"] - temp["period_start"]).dt.days.add(1)
    annual_form = temp["form_type"].map(is_annual_form)
    annual_anchor = duration_days.between(300, 380, inclusive="both") | (temp["period_start"].isna() & annual_form)

    temp["fiscal_year"] = pd.Series([pd.NA] * len(temp), dtype="Int64")
    temp["fiscal_quarter"] = pd.Series([pd.NA] * len(temp), dtype="Int64")
    anchor_dates = pd.DatetimeIndex(temp.loc[annual_anchor, "period_end"]).sort_values().unique()
    if len(anchor_dates) > 0:

        def _quarter_steps(days: int) -> int:
            return max(0, int(round(float(days) / 91.0)))

        for idx, period in enumerate(temp["period_end"]):
            ts = pd.Timestamp(period)
            if ts in anchor_dates:
                temp.at[idx, "fiscal_year"] = int(ts.year)
                temp.at[idx, "fiscal_quarter"] = 4
                continue

            prev_candidates = anchor_dates[anchor_dates < ts]
            next_candidates = anchor_dates[anchor_dates > ts]

            if len(next_candidates) > 0:
                next_anchor = pd.Timestamp(next_candidates[0])
                offset = _quarter_steps((next_anchor - ts).days)
                temp.at[idx, "fiscal_year"] = int(next_anchor.year) - (offset // 4)
                temp.at[idx, "fiscal_quarter"] = 4 - (offset % 4)
                continue

            if len(prev_candidates) > 0:
                prev_anchor = pd.Timestamp(prev_candidates[-1])
                offset = _quarter_steps((ts - prev_anchor).days)
                temp.at[idx, "fiscal_year"] = int(prev_anchor.year) + ((offset - 1) // 4) + 1
                temp.at[idx, "fiscal_quarter"] = ((offset - 1) % 4) + 1
                continue

    if temp["fiscal_year"].isna().all() or temp["fiscal_quarter"].isna().all():
        temp["fiscal_year"] = temp["period_end"].dt.year.astype("Int64")
        temp["fiscal_quarter"] = temp["period_end"].dt.quarter.astype("Int64")
    else:
        fallback_year = temp["period_end"].dt.year.astype("Int64")
        fallback_quarter = temp["period_end"].dt.quarter.astype("Int64")
        temp["fiscal_year"] = temp["fiscal_year"].where(temp["fiscal_year"].notna(), fallback_year)
        temp["fiscal_quarter"] = temp["fiscal_quarter"].where(temp["fiscal_quarter"].notna(), fallback_quarter)

    labels: list[str | None] = []
    for period, fiscal_year, fiscal_quarter in zip(
        temp["period_end"],
        temp["fiscal_year"],
        temp["fiscal_quarter"],
        strict=False,
    ):
        if pd.notna(fiscal_year) and pd.notna(fiscal_quarter):
            labels.append(f"{int(fiscal_year)}Q{int(fiscal_quarter)}")
        elif pd.notna(period):
            ts = pd.Timestamp(period)
            labels.append(f"{ts.year}Q{ts.quarter}")
        else:
            labels.append(None)
    temp["fiscal_label"] = labels
    return temp[["period_end", "fiscal_year", "fiscal_quarter", "fiscal_label"]]


def attach_fiscal_metadata(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    *,
    period_column: str = "PeriodEnd",
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    out = frame.copy()
    if source is None or source.empty:
        return out

    if {"fiscal_year", "fiscal_quarter", "fiscal_label"}.issubset(source.columns):
        meta = source.copy()
        meta["period_end"] = pd.to_datetime(meta.get(period_column), errors="coerce").dt.normalize()
        meta = meta.loc[~meta["period_end"].isna(), ["period_end", "fiscal_year", "fiscal_quarter", "fiscal_label"]]
    else:
        meta = infer_fiscal_period_meta(source.get(period_column), source.get("FormType"), source.get("PeriodStart"))
    if meta.empty:
        return out

    meta = meta.drop_duplicates(subset=["period_end"], keep="last").set_index("period_end")
    if period_column in out.columns:
        target_index = pd.to_datetime(out.get(period_column), errors="coerce").dt.normalize()
    else:
        target_index = pd.DatetimeIndex(out.index).normalize()
    for col in ("fiscal_year", "fiscal_quarter", "fiscal_label"):
        if col in meta.columns:
            out[col] = meta[col].reindex(target_index).to_numpy()
    return out


def period_year_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype="Int64")
    if "fiscal_year" in frame.columns:
        fiscal_year = pd.to_numeric(frame["fiscal_year"], errors="coerce")
        if isinstance(fiscal_year, pd.Series) and fiscal_year.notna().any():
            return fiscal_year.astype("Int64")
    return pd.Series(pd.DatetimeIndex(frame.index).year, index=frame.index, dtype="Int64")
