#!/usr/bin/env python3
"""
Batch scrape butler.works financial summary for all Korean stocks.
Uses Playwright (headless Chromium) with butler.works cookies from Chrome.

Usage:
  python scripts/scrape_butler_all.py [--start 0] [--limit 100] [--concurrency 3]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

BUTLER_REF_DIR = Path(__file__).parent / "butler_reference"
BUTLER_REF_DIR.mkdir(exist_ok=True)


def get_chrome_cookies_for_butler() -> list[dict]:
    """Get butler.works cookies from Chrome via cookie export."""
    # Try to get cookies from the running Chrome MCP session
    # Fallback: use cookies from file if available
    cookie_file = Path("/tmp/butler_cookies.json")
    if cookie_file.exists():
        with open(cookie_file) as f:
            return json.load(f)
    return []


EXTRACT_JS = r"""
() => {
    function p(s){if(!s||s==='-'||s.trim()===''||s.trim()==='-')return null;s=s.trim();if(s.endsWith('%')){const n=parseFloat(s.replace(/%/g,'').replace(/,/g,''));return isNaN(n)?null:n}let neg=s.startsWith('-');if(neg)s=s.substring(1).trim();let t=0;const j=s.match(/([\d,.]+)\\s*조/),u=s.match(/([\d,.]+)\\s*억/),m=s.match(/([\d,.]+)\\s*만/);if(j)t+=parseFloat(j[1].replace(/,/g,''))*10000;if(u)t+=parseFloat(u[1].replace(/,/g,''));if(m&&!j&&!u)t+=parseFloat(m[1].replace(/,/g,''))*0.01;if(!j&&!u&&!m){const n=parseFloat(s.replace(/,/g,'').replace(/[원배]/g,''));if(!isNaN(n))return neg?-n:n;return null}return neg?-t:t}
    function pp(s){const m=s.match(/(\\d{2})\\.(\\d{2})/);return m?(parseInt(m[1])+2000)+'.'+m[2]:s}
    function et(table){if(!table)return null;const rows=table.querySelectorAll('tr');if(rows.length<2)return null;const hc=rows[0].querySelectorAll('th,td');const ps=[];for(let j=1;j<hc.length;j++)ps.push(pp(hc[j].textContent.trim()));const d={};const skip=new Set(['손익계산서','재무상태표','현금흐름표','사업부문별 매출액','사업지역별 매출액','합계','전체','소계','총계','가동률']);let sec=null;for(let r=1;r<rows.length;r++){const cells=rows[r].querySelectorAll('th,td');const l=cells[0].textContent.trim();if(skip.has(l)){if(l.includes('부문별'))sec='biz';else if(l.includes('지역별'))sec='geo';continue}const v=[];for(let j=1;j<cells.length&&j<=ps.length;j++)v.push(p(cells[j].textContent));d[sec?sec+':'+l:l]=v}return{p:ps,d:d}}
    // Click 분기 button
    const qb=Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='분기');
    if(qb)qb.click();
    // Extract tables
    const ts=document.querySelectorAll('table');
    const r={};
    if(ts[0])r.financials=et(ts[0]);if(ts[1])r.segments=et(ts[1]);if(ts[2])r.ratios=et(ts[2]);if(ts[3])r.valuation=et(ts[3]);
    const mc=ts[0]?.querySelector('tr th,tr td');r.period_mode=mc?mc.textContent.trim():'';
    const tm=document.title.match(/(.+?)\\((\\d{6})\\)/);r.ticker=tm?tm[2]:'';r.corp_name=tm?tm[1].trim():'';
    return r;
}
"""


def save_result(data: dict) -> str:
    """Save extraction result to butler_reference JSON."""
    ticker = data.get("ticker", "")
    if not ticker:
        return "NO_TICKER"

    ref_path = BUTLER_REF_DIR / f"{ticker}.json"
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {"ticker": ticker}

    existing["ticker"] = ticker
    existing["corp_name"] = data.get("corp_name", existing.get("corp_name", ""))
    existing["financials_scraped_at"] = str(date.today())

    for key in ("financials", "ratios", "valuation"):
        if data.get(key):
            existing[key] = data[key]

    seg_data = data.get("segments")
    if seg_data and seg_data.get("d"):
        periods = seg_data["p"]
        biz, geo = {}, {}
        for k, v in seg_data["d"].items():
            if k.startswith("biz:"):
                biz[k[4:]] = v
            elif k.startswith("geo:"):
                geo[k[4:]] = v
        if biz:
            existing["summary_biz_segments"] = {"p": periods, "d": biz}
        if geo:
            existing["summary_geo_segments"] = {"p": periods, "d": geo}

    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    fin = existing.get("financials", {})
    return f"{ticker} ({existing.get('corp_name', '?')}): {len(fin.get('d', {}))}m×{len(fin.get('p', []))}p"


def load_queue() -> list[tuple[str, str]]:
    """Load scraping queue: (ticker, corp_code) pairs."""
    queue = []
    for f in sorted(BUTLER_REF_DIR.glob("*.json")):
        if "_" in f.stem or f.stem.startswith("."):
            continue
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        tk = d.get("ticker", f.stem)
        cc = d.get("corp_code")
        if not cc:
            continue
        if d.get("financials") and d["financials"].get("d"):
            continue  # already scraped
        queue.append((tk, cc))
    return queue


async def scrape_batch(queue: list[tuple[str, str]], start: int, limit: int):
    """Scrape a batch of tickers using Playwright."""
    from playwright.async_api import async_playwright

    batch = queue[start : start + limit]
    if not batch:
        print("Nothing to scrape")
        return

    print(f"Scraping {len(batch)} tickers (from #{start})...")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            viewport={"width": 1400, "height": 900},
        )
        page = await context.new_page()

        ok = 0
        skip = 0
        fail = 0

        for i, (ticker, corp_code) in enumerate(batch):
            url = f"https://www.butler.works/ko/companies/{corp_code}?corpLeftPanel=summary&corpMiddlePanel=table"
            try:
                await page.goto(url, wait_until="load", timeout=20000)
                # Wait for React to render tables (key indicator)
                try:
                    await page.wait_for_selector("table tr td", timeout=10000)
                except Exception:
                    # Some pages have no tables — that's OK, will be caught below
                    pass

                # Dismiss any modal overlay
                try:
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(300)
                except Exception:
                    pass

                data = await page.evaluate(EXTRACT_JS)

                if not data or not data.get("ticker"):
                    skip += 1
                    if (i + 1) % 100 == 0:
                        print(f"  [{i+1}/{len(batch)}] {ticker}: no data (skip)", flush=True)
                    continue

                msg = save_result(data)
                ok += 1
                if (i + 1) % 100 == 0 or i < 3:
                    print(f"  [{i+1}/{len(batch)}] {msg}", flush=True)

            except Exception as e:
                fail += 1
                if fail <= 5 or (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{len(batch)}] {ticker}: ERROR {str(e)[:100]}", flush=True)

        await browser.close()
        print(f"\nDone: ok={ok} skip={skip} fail={fail}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=99999)
    args = parser.parse_args()

    queue = load_queue()
    print(f"Queue: {len(queue)} tickers to scrape")

    import asyncio
    asyncio.run(scrape_batch(queue, args.start, args.limit))


if __name__ == "__main__":
    main()
