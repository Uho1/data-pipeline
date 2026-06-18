"""Inject yfinance prices into US ticker JSON files. Restarts every 80 tickers to avoid yfinance hang."""
import sys, os, json, time, subprocess
from pathlib import Path

us_dir = Path("data/tickers/us")

def count_need():
    need = []
    for f in sorted(us_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if "prices" in d and d["prices"].get("dates"):
            continue
        if "financials" not in d or not d["financials"].get("periods"):
            continue
        need.append(f.stem)
    return need

WORKER = r'''
import sys, os, json, time
sys.path.insert(0, r"C:\Users\yh\AppData\Roaming\Python\Python313\site-packages")
import yfinance as yf
import pandas as pd
from pathlib import Path

us_dir = Path("data/tickers/us")
tickers = sys.argv[1:]
ok = fail = 0

for t in tickers:
    try:
        p = us_dir / f"{t}.json"
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        periods = d["financials"]["periods"]
        dates_dt = pd.to_datetime(periods).drop_duplicates().sort_values()
        s = (dates_dt.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        e = (dates_dt.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        hist = yf.download(t, start=s, end=e, progress=False, auto_adjust=False, threads=False, timeout=8)
        if hist is None or hist.empty:
            fail += 1; continue
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        close = pd.to_numeric(hist["Close"], errors="coerce").sort_index()
        aligned = close.reindex(dates_dt, method="ffill").dropna()
        if aligned.empty:
            fail += 1; continue
        d["prices"] = {
            "dates": [d2.strftime("%Y-%m-%d") for d2 in aligned.index],
            "close": [round(float(v), 2) for v in aligned.values],
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
        ok += 1
    except Exception:
        fail += 1
    time.sleep(0.3)

print(f"ok={ok} fail={fail}")
'''

start = time.time()
total_ok = total_fail = 0
BATCH = 80

while True:
    need = count_need()
    if not need:
        break
    batch = need[:BATCH]
    print(f"[{len(need)} remaining] Processing {len(batch)} tickers...", flush=True)

    worktree_root = str(Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [r"C:\ProgramData\anaconda3\python.exe", "-u", "-c", WORKER] + batch,
        capture_output=True, text=True, timeout=180,
        cwd=worktree_root,
    )
    out = result.stdout.strip()
    if "ok=" in out:
        parts = out.split()[-1] if "\n" in out else out
        for line in out.splitlines():
            if "ok=" in line:
                ok = int(line.split("ok=")[1].split()[0])
                fail = int(line.split("fail=")[1].split()[0])
                total_ok += ok
                total_fail += fail
    elapsed = time.time() - start
    print(f"  -> {out} (total ok={total_ok} fail={total_fail}, {elapsed:.0f}s)", flush=True)
    time.sleep(2)  # pause between batches

elapsed = time.time() - start
print(f"DONE: total ok={total_ok} fail={total_fail} ({elapsed:.0f}s)", flush=True)
