from __future__ import annotations

from datetime import UTC
from typing import Iterable

import numpy as np
import pandas as pd


DERIVED_COLUMNS: list[str] = [
    "period_end",
    "available_date",
    "basis",
    "roe",
    "roa",
    "roic",
    "gpa",
    "asset_turnover",
    "leverage",
    "debt_ratio",
    "current_ratio",
    "gross_margin",
    "op_margin",
    "net_margin",
    "cogs_ratio",
    "sga_ratio",
    "total_cost_ratio",
    "revenue_growth",
    "gross_profit_growth",
    "operating_income_growth",
    "net_income_growth",
    "cogs_growth",
    "sga_growth",
    "price_return",
    "ar_turnover",
    "inventory_turnover",
    "ap_turnover",
    "dso",
    "dio",
    "dpo",
    "operating_cycle",
    "cash_cycle",
    "ccr",
    "fcf",
    "eps",
    "bps",
    "sps",
    "ops",
    "oofps",
    "fcfps",
    "per",
    "pbr",
    "psr",
    "por",
    "pfcfr",
    "peg",
    "accruals_ratio",
    "cfo_to_ni",
    "ar_delta",
    "inv_delta",
    "ap_delta",
    "net_wc",
    "net_wc_delta",
    "filing_lag_days",
    "is_amendment",
    "is_nt",
    "punctuality_score",
    "source",
    "collected_at",
]


def _to_naive_ts_series(values: pd.Series | Iterable[object] | object) -> pd.Series:
    series = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(series, pd.Series):
        return series.dt.tz_convert(None)
    if isinstance(series, pd.DatetimeIndex):
        return pd.Series(series.tz_convert(None))
    if isinstance(series, pd.Timestamp):
        one = series.tz_convert(None) if series.tzinfo is not None else series
        return pd.Series([one])
    return pd.Series(pd.to_datetime(pd.Series([values]), errors="coerce", utc=True)).dt.tz_convert(None)


def _to_numeric_series(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    idx = frame.index
    for name in names:
        if name in frame.columns:
            col = pd.to_numeric(frame[name], errors="coerce")
            if col.notna().any():
                return col.reindex(idx)
    return pd.Series(np.nan, index=idx, dtype=float)


def _safe_div(num: pd.Series, den: pd.Series, *, positive_denominator: bool = False) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    if positive_denominator:
        d = d.where(d > 0, np.nan)
    else:
        d = d.replace(0, np.nan)
    return n / d


def _to_nullable_bool_series(values: pd.Series | Iterable[object] | object, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.reindex(index)
    else:
        series = pd.Series(values, index=index)
    return series.astype("boolean")


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").pct_change(periods=periods, fill_method=None) * 100.0
    return out.replace([np.inf, -np.inf], np.nan)


def _basis_eps_series(frame: pd.DataFrame, *, basis: str, net_income: pd.Series, shares: pd.Series) -> pd.Series:
    eps_calc = _safe_div(net_income, shares)
    if basis != "quarter":
        return eps_calc
    eps_raw = pd.to_numeric(frame.get("eps_raw"), errors="coerce")
    return eps_raw.where(eps_raw.notna(), eps_calc)


def _quarter_ttm_eps_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    q_net_income = pd.to_numeric(frame.get("net_income"), errors="coerce").rolling(window=4, min_periods=4).sum()
    q_shares = pd.to_numeric(frame.get("shares"), errors="coerce").replace(0, np.nan)
    return _safe_div(q_net_income, q_shares)


def _pick_price_series(prices: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
    if prices is None or prices.empty:
        return pd.Series(np.nan, index=target_index, dtype=float)
    close_col = None
    for cand in ("Adj Close", "adj_close", "Close", "close"):
        if cand in prices.columns:
            close_col = cand
            break
    if close_col is None:
        return pd.Series(np.nan, index=target_index, dtype=float)
    px = prices.copy()
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index, errors="coerce")
    px = px.loc[~px.index.isna()].sort_index()
    px.index = _to_naive_ts_series(px.index).dt.normalize()
    close = pd.to_numeric(px[close_col], errors="coerce")
    close.index = px.index
    return close.reindex(target_index, method="ffill")


def _prepare_filing_meta(filings: pd.DataFrame | None) -> pd.DataFrame:
    if filings is None or filings.empty:
        return pd.DataFrame()

    meta = filings.copy()
    period_end = _to_naive_ts_series(meta.get("period_end"))
    if period_end.isna().all():
        period_end = _to_naive_ts_series(meta.get("report_date"))
    meta["period_end"] = period_end.dt.normalize()
    meta["filing_date"] = _to_naive_ts_series(meta.get("filing_date")).dt.normalize()
    meta["accepted_at"] = _to_naive_ts_series(meta.get("accepted_at"))
    meta["available_date"] = _to_naive_ts_series(meta.get("available_date")).dt.normalize()
    meta["available_date"] = (
        meta["accepted_at"].dt.normalize().combine_first(meta["filing_date"]).combine_first(meta["available_date"])
    )
    meta["form_type"] = meta.get("form_type", pd.Series(dtype=object)).astype(str).str.upper().str.strip()
    meta = meta.loc[~meta["period_end"].isna()].copy()
    if meta.empty:
        return pd.DataFrame()

    meta["is_amendment"] = _to_nullable_bool_series(meta.get("is_amendment", False), meta.index).fillna(False).astype(bool) | meta["form_type"].str.endswith("/A")
    meta["is_nt"] = _to_nullable_bool_series(meta.get("is_nt", False), meta.index).fillna(False).astype(bool) | meta["form_type"].str.startswith("NT ")
    meta["expected_filing_lag_days"] = np.where(
        meta["form_type"].str.contains("10-K", na=False),
        90.0,
        np.where(meta["form_type"].str.contains("10-Q", na=False), 45.0, np.nan),
    )

    actual = meta.loc[~meta["is_nt"]].copy()
    if not actual.empty:
        actual = actual.sort_values(
            [c for c in ["period_end", "available_date", "filing_date", "accepted_at", "is_amendment"] if c in actual.columns]
        )
        actual = actual.groupby("period_end", as_index=False).tail(1)
        actual = actual.set_index("period_end")
    else:
        actual = pd.DataFrame(index=pd.DatetimeIndex([], name="period_end"))

    flags = meta.groupby("period_end", dropna=True).agg(
        is_amendment=("is_amendment", "max"),
        is_nt=("is_nt", "max"),
    )

    out = actual.join(flags, how="outer", rsuffix="_flag")
    if "is_amendment_flag" in out.columns:
        left = _to_nullable_bool_series(out.get("is_amendment", False), out.index).fillna(False).astype(bool)
        right = _to_nullable_bool_series(out["is_amendment_flag"], out.index).fillna(False).astype(bool)
        out["is_amendment"] = left | right
        out = out.drop(columns=["is_amendment_flag"])
    if "is_nt_flag" in out.columns:
        left = _to_nullable_bool_series(out.get("is_nt", False), out.index).fillna(False).astype(bool)
        right = _to_nullable_bool_series(out["is_nt_flag"], out.index).fillna(False).astype(bool)
        out["is_nt"] = left | right
        out = out.drop(columns=["is_nt_flag"])
    return out.sort_index()


def _prepare_quarter_frame(
    financials: pd.DataFrame,
    prices: pd.DataFrame | None,
    filings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if financials is None or financials.empty:
        return pd.DataFrame()

    out = financials.copy()
    if "PeriodEnd" not in out.columns and "StatementDate" in out.columns:
        out["PeriodEnd"] = out["StatementDate"]
    out["PeriodEnd"] = _to_naive_ts_series(out.get("PeriodEnd")).dt.normalize()
    out = out.loc[~out["PeriodEnd"].isna()].copy()
    if out.empty:
        return out

    out["FilingDate"] = _to_naive_ts_series(out.get("FilingDate")).dt.normalize()
    out["AcceptedAt"] = _to_naive_ts_series(out.get("AcceptedAt"))
    out["AvailableDate"] = _to_naive_ts_series(out.get("AvailableDate")).dt.normalize()
    out["AvailableDate"] = out["AvailableDate"].fillna(out["FilingDate"]).fillna(out["PeriodEnd"])

    sort_cols = [c for c in ["PeriodEnd", "AvailableDate", "FilingDate", "AcceptedAt", "collected_at", "CollectedAt"] if c in out.columns]
    out = out.sort_values(sort_cols)
    out = out.groupby("PeriodEnd", as_index=False).tail(1)
    out = out.sort_values("PeriodEnd").reset_index(drop=True)
    out = out.set_index("PeriodEnd", drop=False)
    out.index = pd.DatetimeIndex(out.index).normalize()

    revenue = _to_numeric_series(out, ["Revenue", "revenue", "Sales"])
    cogs = _to_numeric_series(out, ["COGS", "cogs", "Cost Of Revenue"])
    sga = _to_numeric_series(out, ["SG&A", "sga", "Selling General And Administrative"])
    gross_profit = _to_numeric_series(out, ["Gross Profit", "gross_profit", "GrossProfit"]).where(
        lambda s: s.notna(), revenue - cogs
    )
    operating_income = _to_numeric_series(out, ["Operating Income", "operating_income", "OperatingIncome"])
    net_income = _to_numeric_series(out, ["Net Income", "net_income", "NetIncome"])
    assets = _to_numeric_series(out, ["Total Assets", "assets", "Assets"])
    liabilities = _to_numeric_series(out, ["Total Liabilities", "liabilities", "Liabilities"])
    equity = _to_numeric_series(out, ["Shareholders Equity", "equity", "Stockholders Equity"])
    current_assets = _to_numeric_series(out, ["Current Assets", "current_assets", "CurrentAssets"])
    current_liabilities = _to_numeric_series(out, ["Current Liabilities", "current_liabilities", "CurrentLiabilities"])
    ar = _to_numeric_series(out, ["AR", "ar", "Accounts Receivable", "Receivables"])
    inventory = _to_numeric_series(out, ["Inventory", "inventory"])
    ap = _to_numeric_series(out, ["AP", "ap", "Accounts Payable"])
    cfo = _to_numeric_series(out, ["Operating Cash Flow", "cfo", "OperatingCashFlow", "CFO"])
    cfi = _to_numeric_series(out, ["Investing Cash Flow", "cfi", "InvestingCashFlow", "CFI"])
    cff = _to_numeric_series(out, ["Financing Cash Flow", "cff", "FinancingCashFlow", "CFF"])
    capex_raw = _to_numeric_series(
        out,
        ["Capital Expenditure", "capex", "capex_raw", "Capital Expenditures", "CapitalExpenditures", "CAPEX"],
    )
    pre_tax_income = _to_numeric_series(out, ["Pretax Income", "pre_tax_income", "Pre Tax Income"])
    tax_expense = _to_numeric_series(out, ["Tax", "tax_expense", "Income Tax Expense", "Tax Expense"])
    debt_short = _to_numeric_series(out, ["Debt Short", "debt_short", "Short Term Debt"])
    debt_long = _to_numeric_series(out, ["Debt Long", "debt_long", "Long Term Debt"])
    shares = _to_numeric_series(out, ["Shares", "shares", "Diluted Shares", "Basic Shares", "diluted_shares", "basic_shares"])
    eps_raw = _to_numeric_series(out, ["EPS", "eps", "Diluted EPS", "diluted_eps"])
    price = _pick_price_series(prices if prices is not None else pd.DataFrame(), pd.DatetimeIndex(out.index))

    base = pd.DataFrame(index=out.index)
    base["period_end"] = pd.DatetimeIndex(out.index).normalize()
    base["available_date"] = _to_naive_ts_series(out.get("AvailableDate")).dt.normalize().reindex(out.index)
    base["filing_date"] = _to_naive_ts_series(out.get("FilingDate")).dt.normalize().reindex(out.index)
    base["accepted_at"] = _to_naive_ts_series(out.get("AcceptedAt")).reindex(out.index)
    base["form_type"] = out.get("FormType", out.get("form_type", pd.Series(dtype=object))).reindex(out.index).astype(str)
    
    # Flags for amendment and NT if we have form_type or explicit columns
    is_amend = out.get("is_amendment", pd.Series(0, index=out.index)).astype(bool)
    is_amend = is_amend | base["form_type"].str.upper().str.endswith("/A")
    base["is_amendment"] = is_amend
    
    is_nt = out.get("is_nt", pd.Series(0, index=out.index)).astype(bool)
    is_nt = is_nt | base["form_type"].str.upper().str.startswith("NT")
    base["is_nt"] = is_nt
    base["filing_available_date"] = base["accepted_at"].dt.normalize().combine_first(base["filing_date"])
    base["expected_filing_lag_days"] = np.where(
        base["form_type"].str.upper().str.contains("10-K", na=False),
        90.0,
        np.where(base["form_type"].str.upper().str.contains("10-Q", na=False), 45.0, np.nan),
    )

    filing_meta = _prepare_filing_meta(filings)
    if not filing_meta.empty:
        aligned = filing_meta.reindex(base.index)
        for col in ("filing_date", "accepted_at", "available_date", "expected_filing_lag_days"):
            if col in aligned.columns:
                target = "filing_available_date" if col == "available_date" else col
                base[target] = base[target].combine_first(aligned[col])
        if "form_type" in aligned.columns:
            base["form_type"] = base["form_type"].where(base["form_type"].str.len() > 0, aligned["form_type"])
        if "is_amendment" in aligned.columns:
            base["is_amendment"] = base["is_amendment"] | _to_nullable_bool_series(aligned["is_amendment"], aligned.index).fillna(False).astype(bool)
        if "is_nt" in aligned.columns:
            base["is_nt"] = base["is_nt"] | _to_nullable_bool_series(aligned["is_nt"], aligned.index).fillna(False).astype(bool)

    base["revenue"] = revenue
    base["cogs"] = cogs
    base["sga"] = sga
    base["gross_profit"] = gross_profit
    base["operating_income"] = operating_income
    base["net_income"] = net_income
    base["assets"] = assets
    base["liabilities"] = liabilities
    base["equity"] = equity
    base["current_assets"] = current_assets
    base["current_liabilities"] = current_liabilities
    base["ar"] = ar
    base["inventory"] = inventory
    base["ap"] = ap
    base["cfo"] = cfo
    base["cfi"] = cfi
    base["cff"] = cff
    base["capex_raw"] = capex_raw
    base["capex_outflow"] = capex_raw.abs()
    base["fcf"] = base["cfo"] - base["capex_outflow"]
    base["pre_tax_income"] = pre_tax_income
    base["tax_expense"] = tax_expense
    base["debt_short"] = debt_short
    base["debt_long"] = debt_long
    base["debt_total"] = debt_short.fillna(0.0) + debt_long.fillna(0.0)
    base["shares"] = shares.replace(0, np.nan)
    base["eps_raw"] = eps_raw
    base["price"] = price
    base["market_cap"] = base["price"] * base["shares"]
    return base.sort_index()


def _annualize(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    flow_cols = [
        "revenue",
        "cogs",
        "sga",
        "gross_profit",
        "operating_income",
        "net_income",
        "cfo",
        "cfi",
        "cff",
        "capex_raw",
        "capex_outflow",
        "fcf",
        "pre_tax_income",
        "tax_expense",
    ]
    stock_cols = [c for c in base.columns if c not in set(flow_cols + ["period_end", "available_date"])]
    rows: list[dict[str, object]] = []
    years = sorted({ts.year for ts in pd.DatetimeIndex(base.index)})
    for year in years:
        chunk = base.loc[pd.DatetimeIndex(base.index).year == year]
        if chunk.empty:
            continue
        row: dict[str, object] = {
            "period_end": pd.Timestamp(year=year, month=12, day=31),
            "available_date": pd.to_datetime(chunk["available_date"], errors="coerce").max(),
        }
        for col in flow_cols:
            if col in chunk.columns:
                row[col] = pd.to_numeric(chunk[col], errors="coerce").sum(min_count=1)
        for col in stock_cols:
            vals = pd.to_numeric(chunk[col], errors="coerce")
            row[col] = vals.iloc[-1] if vals.notna().any() else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("period_end").sort_index()
    out["period_end"] = pd.DatetimeIndex(out.index).normalize()
    out["available_date"] = _to_naive_ts_series(out.get("available_date")).dt.normalize()
    return out


def _to_basis(base: pd.DataFrame, basis: str) -> tuple[pd.DataFrame, int]:
    if base.empty:
        return pd.DataFrame(), 1
    if basis == "annual":
        return _annualize(base), 1
    if basis == "ttm":
        out = base.copy()
        for col in [
            "revenue",
            "cogs",
            "sga",
            "gross_profit",
            "operating_income",
            "net_income",
            "cfo",
            "cfi",
            "cff",
            "capex_raw",
            "capex_outflow",
            "fcf",
            "pre_tax_income",
            "tax_expense",
        ]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce").rolling(window=4, min_periods=4).sum()
        out["capex_outflow"] = pd.to_numeric(out.get("capex_outflow"), errors="coerce").where(
            pd.to_numeric(out.get("capex_outflow"), errors="coerce").notna(),
            pd.to_numeric(out.get("capex_raw"), errors="coerce").abs(),
        )
        out["fcf"] = pd.to_numeric(out.get("cfo"), errors="coerce") - pd.to_numeric(out.get("capex_outflow"), errors="coerce")
        return out, 4
    return base.copy(), 4


def _compute_derived(frame: pd.DataFrame, *, basis: str, growth_lag: int, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=DERIVED_COLUMNS)
    out = pd.DataFrame(index=frame.index)
    out["period_end"] = pd.DatetimeIndex(frame.index).normalize()
    out["available_date"] = _to_naive_ts_series(frame.get("available_date")).dt.normalize().reindex(frame.index)
    out["basis"] = basis

    revenue = pd.to_numeric(frame.get("revenue"), errors="coerce")
    cogs = pd.to_numeric(frame.get("cogs"), errors="coerce")
    sga = pd.to_numeric(frame.get("sga"), errors="coerce")
    gross_profit = pd.to_numeric(frame.get("gross_profit"), errors="coerce")
    operating_income = pd.to_numeric(frame.get("operating_income"), errors="coerce")
    net_income = pd.to_numeric(frame.get("net_income"), errors="coerce")
    assets = pd.to_numeric(frame.get("assets"), errors="coerce")
    liabilities = pd.to_numeric(frame.get("liabilities"), errors="coerce")
    equity = pd.to_numeric(frame.get("equity"), errors="coerce")
    current_assets = pd.to_numeric(frame.get("current_assets"), errors="coerce")
    current_liabilities = pd.to_numeric(frame.get("current_liabilities"), errors="coerce")
    ar = pd.to_numeric(frame.get("ar"), errors="coerce")
    inventory = pd.to_numeric(frame.get("inventory"), errors="coerce")
    ap = pd.to_numeric(frame.get("ap"), errors="coerce")
    cfo = pd.to_numeric(frame.get("cfo"), errors="coerce")
    capex_outflow = pd.to_numeric(frame.get("capex_outflow"), errors="coerce")
    fcf = pd.to_numeric(frame.get("fcf"), errors="coerce")
    pre_tax_income = pd.to_numeric(frame.get("pre_tax_income"), errors="coerce")
    tax_expense = pd.to_numeric(frame.get("tax_expense"), errors="coerce")
    debt_total = pd.to_numeric(frame.get("debt_total"), errors="coerce")
    shares = pd.to_numeric(frame.get("shares"), errors="coerce").replace(0, np.nan)
    price = pd.to_numeric(frame.get("price"), errors="coerce")

    avg_assets = (assets + assets.shift(1)) / 2.0
    avg_equity = (equity + equity.shift(1)) / 2.0
    avg_debt = (debt_total + debt_total.shift(1)) / 2.0
    avg_ar = (ar + ar.shift(1)) / 2.0
    avg_inventory = (inventory + inventory.shift(1)) / 2.0
    avg_ap = (ap + ap.shift(1)) / 2.0

    if basis == "quarter":
        revenue_for_turnover = revenue.rolling(window=4, min_periods=4).sum()
        cogs_for_turnover = cogs.rolling(window=4, min_periods=4).sum()
    else:
        revenue_for_turnover = revenue
        cogs_for_turnover = cogs

    out["gross_margin"] = _safe_div(gross_profit, revenue) * 100.0
    out["op_margin"] = _safe_div(operating_income, revenue) * 100.0
    out["net_margin"] = _safe_div(net_income, revenue) * 100.0
    out["cogs_ratio"] = _safe_div(cogs, revenue) * 100.0
    out["sga_ratio"] = _safe_div(sga, revenue) * 100.0
    out["total_cost_ratio"] = _safe_div(cogs + sga, revenue) * 100.0

    out["revenue_growth"] = _pct_change(revenue, growth_lag)
    out["gross_profit_growth"] = _pct_change(gross_profit, growth_lag)
    out["operating_income_growth"] = _pct_change(operating_income, growth_lag)
    out["net_income_growth"] = _pct_change(net_income, growth_lag)
    out["cogs_growth"] = _pct_change(cogs, growth_lag)
    out["sga_growth"] = _pct_change(sga, growth_lag)
    out["price_return"] = _pct_change(price, growth_lag)

    out["roe"] = _safe_div(net_income, avg_equity) * 100.0
    out["roa"] = _safe_div(net_income, avg_assets) * 100.0
    out["gpa"] = _safe_div(gross_profit, avg_assets) * 100.0
    out["asset_turnover"] = _safe_div(revenue, avg_assets)
    out["leverage"] = _safe_div(avg_assets, avg_equity)
    out["debt_ratio"] = _safe_div(liabilities, equity) * 100.0
    out["current_ratio"] = _safe_div(current_assets, current_liabilities) * 100.0

    tax_rate = _safe_div(tax_expense, pre_tax_income.where(pre_tax_income > 0, np.nan))
    invested_capital = avg_equity + avg_debt
    out["roic"] = _safe_div(operating_income * (1.0 - tax_rate), invested_capital) * 100.0

    out["ar_turnover"] = _safe_div(revenue_for_turnover, avg_ar)
    out["inventory_turnover"] = _safe_div(cogs_for_turnover, avg_inventory)
    out["ap_turnover"] = _safe_div(cogs_for_turnover, avg_ap)
    out["dso"] = _safe_div(pd.Series(365.0, index=out.index), out["ar_turnover"], positive_denominator=True)
    out["dio"] = _safe_div(pd.Series(365.0, index=out.index), out["inventory_turnover"], positive_denominator=True)
    out["dpo"] = _safe_div(pd.Series(365.0, index=out.index), out["ap_turnover"], positive_denominator=True)
    out["operating_cycle"] = pd.to_numeric(out["dso"], errors="coerce") + pd.to_numeric(out["dio"], errors="coerce")
    out["cash_cycle"] = pd.to_numeric(out["operating_cycle"], errors="coerce") - pd.to_numeric(out["dpo"], errors="coerce")

    out["ccr"] = _safe_div(cfo, net_income.where(net_income > 0, np.nan))
    out["fcf"] = fcf

    eps = _basis_eps_series(frame, basis=basis, net_income=net_income, shares=shares)
    eps_for_per = eps
    if basis == "quarter":
        eps_for_per = _quarter_ttm_eps_series(frame).combine_first(pd.to_numeric(eps, errors="coerce"))
    out["eps"] = eps
    out["bps"] = _safe_div(equity, shares)
    out["sps"] = _safe_div(revenue, shares)
    out["ops"] = _safe_div(operating_income, shares)
    out["oofps"] = _safe_div(cfo, shares)
    out["fcfps"] = _safe_div(fcf, shares)
    out["per"] = _safe_div(price, eps_for_per, positive_denominator=True)
    out["pbr"] = _safe_div(price, out["bps"], positive_denominator=True)
    out["psr"] = _safe_div(price, out["sps"], positive_denominator=True)
    out["por"] = _safe_div(price, out["ops"], positive_denominator=True)
    out["pfcfr"] = _safe_div(price, out["fcfps"], positive_denominator=True)
    eps_growth = _pct_change(eps_for_per, growth_lag)
    out["peg"] = _safe_div(out["per"], eps_growth.where(eps_growth > 0, np.nan))

    out["accruals_ratio"] = _safe_div(net_income - cfo, assets)
    out["cfo_to_ni"] = _safe_div(cfo, net_income.where(net_income > 0, np.nan))
    out["ar_delta"] = ar - ar.shift(1)
    out["inv_delta"] = inventory - inventory.shift(1)
    out["ap_delta"] = ap - ap.shift(1)
    net_wc = ar.fillna(0) + inventory.fillna(0) - ap.fillna(0)
    out["net_wc"] = net_wc.where(ar.notna() | inventory.notna() | ap.notna(), np.nan)
    out["net_wc_delta"] = out["net_wc"] - out["net_wc"].shift(1)

    filing_date = _to_naive_ts_series(frame.get("filing_date")).dt.normalize()
    accepted_at = _to_naive_ts_series(frame.get("accepted_at"))

    # Priority for lag calculation: AcceptedAt, then FilingDate, fallback to AvailableDate
    filing_available = _to_naive_ts_series(frame.get("filing_available_date")).dt.normalize()
    calc_date = accepted_at.dt.normalize().combine_first(filing_date).combine_first(filing_available).combine_first(out["available_date"])
    out["filing_lag_days"] = (calc_date - out["period_end"]).dt.days
    out["is_amendment"] = frame.get("is_amendment", pd.Series(False, index=out.index)).astype(bool)
    out["is_nt"] = frame.get("is_nt", pd.Series(False, index=out.index)).astype(bool)
    expected_lag = pd.to_numeric(frame.get("expected_filing_lag_days"), errors="coerce")
    punctuality_raw = (100.0 - (out["filing_lag_days"] - expected_lag).abs() * 2.0).clip(lower=0.0, upper=100.0)
    out["punctuality_score"] = punctuality_raw.rolling(window=4, min_periods=1).mean()

    out["source"] = source
    out["collected_at"] = pd.Timestamp.now(tz="UTC")
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in DERIVED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[DERIVED_COLUMNS]


def build_derived_factors_quarterly(
    financials: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    filings: pd.DataFrame | None = None,
    *,
    source: str = "materialized",
) -> pd.DataFrame:
    base = _prepare_quarter_frame(financials=financials, prices=prices, filings=filings)
    if base.empty:
        return pd.DataFrame(columns=DERIVED_COLUMNS)

    results: list[pd.DataFrame] = []
    for basis in ("quarter", "ttm", "annual"):
        basis_frame, growth_lag = _to_basis(base, basis)
        if basis_frame.empty:
            continue
        derived = _compute_derived(basis_frame, basis=basis, growth_lag=growth_lag, source=source)
        if derived.empty:
            continue
        results.append(derived)

    if not results:
        return pd.DataFrame(columns=DERIVED_COLUMNS)
    out = pd.concat(results, ignore_index=True, sort=False)
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce").dt.date
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce").dt.date
    out["basis"] = out["basis"].astype(str).str.lower().str.strip()
    out = out.dropna(subset=["period_end", "available_date", "basis"]).copy()
    out = out.sort_values(["period_end", "available_date", "basis"]).drop_duplicates(
        subset=["period_end", "available_date", "basis"],
        keep="last",
    )
    return out.reset_index(drop=True)
