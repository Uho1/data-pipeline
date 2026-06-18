from __future__ import annotations

import importlib.util
import json
import re
import site
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from market_data.config import DATA_DIR
from market_data.utils import ensure_dir, now_utc_iso, retry_call

TERM_RE = re.compile(r"^\d{4}Q[1-4]$")
DEFAULT_RAW_TERMS_DIR = DATA_DIR / "finterstellar_term_cache" / "raw_terms"


@dataclass
class RawTermUpdateOptions:
    otp: str
    cache_dir: Path = DEFAULT_RAW_TERMS_DIR
    start_term: str | None = None
    end_term: str | None = None
    vol: int = 100000
    study: str = "N"
    retries: int = 3
    backoff_base: float = 1.0
    force: bool = False


def _normalize_term(term: str) -> str:
    text = str(term).strip().upper().replace("-", "").replace("_", "")
    if not TERM_RE.fullmatch(text):
        raise ValueError(f"Invalid term format: {term}. Use YYYYQn, e.g. 2024Q3")
    return text


def _term_to_period(term: str) -> pd.Period:
    return pd.Period(_normalize_term(term), freq="Q")


def _period_to_term(period: pd.Period) -> str:
    return f"{period.year}Q{period.quarter}"


def current_term() -> str:
    return _period_to_term(pd.Period(pd.Timestamp.today(), freq="Q"))


def next_term(term: str) -> str:
    return _period_to_term(_term_to_period(term) + 1)


def latest_cached_term(cache_dir: Path = DEFAULT_RAW_TERMS_DIR) -> str | None:
    if not cache_dir.exists():
        return None
    terms: list[pd.Period] = []
    for p in cache_dir.glob("*.parquet"):
        stem = p.stem
        if TERM_RE.fullmatch(stem):
            terms.append(_term_to_period(stem))
    if not terms:
        return None
    return _period_to_term(max(terms))


def _site_package_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        user = site.getusersitepackages()
        if user:
            roots.append(Path(user))
    except Exception:
        pass
    return roots


def _load_fn_consolidated():
    for root in _site_package_roots():
        mod_path = root / "finterstellar" / "financials.py"
        if not mod_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("_fs_financials_mod", mod_path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "fn_consolidated", None)
        if callable(fn):
            return fn
    raise ImportError("finterstellar financials module not found")


def _build_terms(start_term: str, end_term: str) -> list[str]:
    p_start = _term_to_period(start_term)
    p_end = _term_to_period(end_term)
    if p_start > p_end:
        return []
    return [_period_to_term(p) for p in pd.period_range(start=p_start, end=p_end, freq="Q")]


def _save_term_frame(out_dir: Path, term: str, df: pd.DataFrame) -> None:
    out = df.reset_index().copy()
    if "index" in out.columns and "symbol" not in out.columns:
        out = out.rename(columns={"index": "symbol"})
    pq = out_dir / f"{term}.parquet"
    csvp = out_dir / f"{term}.csv"
    manifest = out_dir / f"{term}.manifest.json"
    out.to_parquet(pq, index=False)
    out.to_csv(csvp, index=False)
    payload = {
        "saved_at": now_utc_iso(),
        "term": term,
        "rows": int(len(out)),
        "cols": int(out.shape[1]),
        "columns": out.columns.tolist(),
        "saved_parquet": str(pq.resolve()),
        "saved_csv": str(csvp.resolve()),
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rebuild_combined(out_dir: Path) -> tuple[int, str | None, str | None, Path]:
    frames: list[pd.DataFrame] = []
    for p in sorted(out_dir.glob("*.parquet")):
        if not TERM_RE.fullmatch(p.stem):
            continue
        df = pd.read_parquet(p)
        if "term" not in df.columns:
            df["term"] = p.stem
        frames.append(df)
    all_out = out_dir / "all_terms_2010Q1_to_latest_available.parquet"
    if not frames:
        pd.DataFrame().to_parquet(all_out, index=False)
        return 0, None, None, all_out
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df.to_parquet(all_out, index=False)
    t = all_df["term"].astype(str)
    return int(len(all_df)), str(t.min()), str(t.max()), all_out


def run_raw_term_update(
    opts: RawTermUpdateOptions,
    log_cb: Callable[[str], None] | None = None,
) -> int:
    def log(msg: str) -> None:
        if log_cb is None:
            print(msg)
        else:
            log_cb(msg)

    otp = (opts.otp or "").strip()
    if not otp:
        log("[ERROR] OTP is required.")
        return 2

    out_dir = Path(opts.cache_dir).expanduser()
    ensure_dir(out_dir)

    end_term = _normalize_term(opts.end_term) if opts.end_term else current_term()
    if opts.start_term:
        start_term = _normalize_term(opts.start_term)
    else:
        latest = latest_cached_term(out_dir)
        start_term = next_term(latest) if latest else end_term

    terms = _build_terms(start_term, end_term)
    if not terms:
        log(f"[DONE] no terms to update (start={start_term}, end={end_term})")
        return 0

    fn = _load_fn_consolidated()
    log(f"[RUN] finterstellar raw term update start={start_term} end={end_term} terms={len(terms)}")
    summary: list[dict[str, object]] = []

    for term in terms:
        term_file = out_dir / f"{term}.parquet"
        if term_file.exists() and not opts.force:
            log(f"{term}...SKIP")
            summary.append({"term": term, "status": "SKIP", "rows": 0, "error": ""})
            continue

        try:
            df = retry_call(
                lambda: fn(otp=otp, term=term, vol=opts.vol, study=opts.study),
                retries=opts.retries,
                backoff_base=opts.backoff_base,
                label=f"finterstellar:{term}",
            )
            if isinstance(df, str):
                msg = str(df).strip()
                if "Invalid OTP" in msg:
                    log(f"{term}...FAIL (Invalid OTP)")
                    return 2
                log(f"{term}...FAIL ({msg})")
                summary.append({"term": term, "status": "FAIL", "rows": 0, "error": msg})
                continue
            if df is None or df.empty:
                log(f"{term}...EMPTY")
                summary.append({"term": term, "status": "EMPTY", "rows": 0, "error": ""})
                continue
            _save_term_frame(out_dir, term, df)
            log(f"{term}...OK")
            summary.append({"term": term, "status": "OK", "rows": int(len(df)), "error": ""})
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Invalid OTP" in msg:
                log(f"{term}...FAIL (Invalid OTP)")
                return 2
            log(f"{term}...FAIL ({msg})")
            summary.append({"term": term, "status": "FAIL", "rows": 0, "error": msg})

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"run_summary_{ts}.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    all_rows, min_term, max_term, combined = _rebuild_combined(out_dir)
    ok = sum(1 for row in summary if row["status"] == "OK")
    skip = sum(1 for row in summary if row["status"] == "SKIP")
    empty = sum(1 for row in summary if row["status"] == "EMPTY")
    fail = sum(1 for row in summary if row["status"] == "FAIL")
    log(
        f"[DONE] ok={ok} skip={skip} empty={empty} fail={fail} "
        f"all_rows={all_rows} term_range={min_term}->{max_term}"
    )
    log(f"[SAVE] summary={summary_path}")
    log(f"[SAVE] combined={combined}")
    return 0 if fail == 0 else 2
