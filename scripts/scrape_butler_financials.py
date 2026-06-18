#!/usr/bin/env python3
"""
Butler financial summary scraper — extracts quarterly financials, ratios,
and valuation from the butler.works summary table view.

This script provides:
  1. JS extraction code to run via Chrome MCP javascript_tool
  2. Python helper to save/merge results into butler_reference/{ticker}.json

Usage (from Claude Code conversation with Chrome MCP):
  - Navigate to butler summary table URL
  - Click '분기' button for quarterly standalone values
  - Run EXTRACT_FINANCIALS_JS via javascript_tool
  - Call save_financials_result(ticker, json_str) to save

The save function merges into existing JSON (preserving segment data).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

BUTLER_REF_DIR = Path(__file__).parent / "butler_reference"
BUTLER_REF_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# JavaScript extraction — run via Chrome MCP javascript_tool on the
# butler summary table page (after clicking '분기' for quarterly mode)
# ---------------------------------------------------------------------------
EXTRACT_FINANCIALS_JS = r"""
(function() {
  function parseKrNum(s) {
    if (!s || s === '-' || s.trim() === '' || s.trim() === '-') return null;
    s = s.trim();
    if (s.endsWith('%')) {
      const n = parseFloat(s.replace(/%/g, '').replace(/,/g, ''));
      return isNaN(n) ? null : n;
    }
    let neg = s.startsWith('-');
    if (neg) s = s.substring(1).trim();
    let total = 0;
    const joM = s.match(/([\d,.]+)\s*조/);
    const ukM = s.match(/([\d,.]+)\s*억/);
    const manM = s.match(/([\d,.]+)\s*만/);
    if (joM) total += parseFloat(joM[1].replace(/,/g, '')) * 10000;
    if (ukM) total += parseFloat(ukM[1].replace(/,/g, ''));
    if (manM && !joM && !ukM) total += parseFloat(manM[1].replace(/,/g, '')) * 0.01;
    if (!joM && !ukM && !manM) {
      const n = parseFloat(s.replace(/,/g, '').replace(/[원배]/g, ''));
      if (!isNaN(n)) return neg ? -n : n;
      return null;
    }
    return neg ? -total : total;
  }

  function parsePeriod(s) {
    const m = s.match(/(\d{2})\.(\d{2})/);
    if (!m) return s;
    return (parseInt(m[1]) + 2000) + '.' + m[2];
  }

  function extractTable(table) {
    if (!table) return null;
    const rows = table.querySelectorAll('tr');
    if (rows.length < 2) return null;
    const hCells = rows[0].querySelectorAll('th, td');
    const periods = [];
    for (let j = 1; j < hCells.length; j++) {
      periods.push(parsePeriod(hCells[j].textContent.trim()));
    }
    const data = {};
    const skipLabels = new Set([
      '손익계산서', '재무상태표', '현금흐름표',
      '사업부문별 매출액', '사업지역별 매출액',
      '합계', '전체', '소계', '총계', '가동률'
    ]);
    let currentSection = null;
    for (let r = 1; r < rows.length; r++) {
      const cells = rows[r].querySelectorAll('th, td');
      const label = cells[0].textContent.trim();
      if (skipLabels.has(label)) {
        if (label.includes('부문별')) currentSection = 'biz';
        else if (label.includes('지역별')) currentSection = 'geo';
        continue;
      }
      const vals = [];
      for (let j = 1; j < cells.length && j <= periods.length; j++) {
        vals.push(parseKrNum(cells[j].textContent));
      }
      const key = currentSection ? currentSection + ':' + label : label;
      data[key] = vals;
    }
    return { p: periods, d: data };
  }

  const tables = document.querySelectorAll('table');
  const result = {};

  // Table 0: Financial summary (억원)
  if (tables[0]) result.financials = extractTable(tables[0]);
  // Table 1: Segments (억원)
  if (tables[1]) result.segments = extractTable(tables[1]);
  // Table 2: Ratios (%)
  if (tables[2]) result.ratios = extractTable(tables[2]);
  // Table 3: Valuation (배/원)
  if (tables[3]) result.valuation = extractTable(tables[3]);

  // Period mode check
  const modeCell = tables[0]?.querySelector('tr th, tr td');
  result.period_mode = modeCell ? modeCell.textContent.trim() : '';

  // Title → ticker, corp_name
  const tm = document.title.match(/(.+?)\((\d{6})\)/);
  result.ticker = tm ? tm[2] : '';
  result.corp_name = tm ? tm[1].trim() : '';

  return JSON.stringify(result);
})()
"""

# ---------------------------------------------------------------------------
# JS to set up the page: click '분기' and switch to table view
# ---------------------------------------------------------------------------
SETUP_PAGE_JS = r"""
(function() {
  // Click '분기' button for quarterly standalone values
  const quarterBtn = Array.from(document.querySelectorAll('button'))
    .find(b => b.textContent.trim() === '분기');
  if (quarterBtn) quarterBtn.click();

  // Verify table view is active (check if tables exist)
  const tables = document.querySelectorAll('table');
  return JSON.stringify({
    clickedQuarter: !!quarterBtn,
    tableCount: tables.length,
    ready: tables.length >= 4
  });
})()
"""


def butler_summary_url(corp_code: str) -> str:
    """Build Butler summary table URL."""
    return (
        f"https://www.butler.works/ko/companies/{corp_code}"
        f"?corpLeftPanel=summary&corpMiddlePanel=table"
    )


def save_financials_result(json_str: str) -> str:
    """Parse extraction result and merge into existing butler_reference JSON.

    Returns ticker string on success, empty string on failure.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return ""

    ticker = data.get("ticker", "")
    if not ticker:
        return ""

    # Load existing reference file if present
    ref_path = BUTLER_REF_DIR / f"{ticker}.json"
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {"ticker": ticker}

    # Update metadata
    existing["ticker"] = ticker
    existing["corp_name"] = data.get("corp_name", existing.get("corp_name", ""))
    existing["financials_scraped_at"] = str(date.today())

    # Store extracted sections
    for key in ("financials", "ratios", "valuation"):
        if data.get(key):
            existing[key] = data[key]

    # Merge segment data from table 1 (biz: and geo: prefixed keys)
    seg_data = data.get("segments")
    if seg_data and seg_data.get("d"):
        periods = seg_data["p"]
        biz_segs = {}
        geo_segs = {}
        for k, v in seg_data["d"].items():
            if k.startswith("biz:"):
                biz_segs[k[4:]] = v
            elif k.startswith("geo:"):
                geo_segs[k[4:]] = v
        if biz_segs:
            existing["summary_biz_segments"] = {"p": periods, "d": biz_segs}
        if geo_segs:
            existing["summary_geo_segments"] = {"p": periods, "d": geo_segs}

    # Save
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    # Summary
    fin = existing.get("financials", {})
    n_periods = len(fin.get("p", []))
    n_metrics = len(fin.get("d", {}))
    return f"{ticker} ({existing.get('corp_name', '?')}): {n_metrics} metrics × {n_periods} periods"


def load_ticker_corp_map() -> dict[str, str]:
    """Load ticker → corp_code mapping from existing butler_reference JSONs."""
    mapping = {}
    for f in BUTLER_REF_DIR.glob("*.json"):
        if f.stem.startswith(".") or "_" in f.stem:
            continue
        try:
            with open(f, encoding="utf-8") as fp:
                d = json.load(fp)
            ticker = d.get("ticker", f.stem)
            corp_code = d.get("corp_code")
            if corp_code:
                mapping[ticker] = corp_code
        except (json.JSONDecodeError, KeyError):
            continue
    return mapping


# Priority tickers for testing/validation
PRIORITY_TICKERS = [
    # Known issues
    "035420",  # 네이버 - Revenue 매핑 오류
    "010130",  # 고려아연 - XBRL dimension 세그먼트 미지원
    # Top market cap
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "005380",  # 현대자동차
    "000270",  # 기아
    "005490",  # POSCO홀딩스
    "051910",  # LG화학
    "066570",  # LG전자
    "003550",  # LG
    "055550",  # 신한지주
    "105560",  # KB금융
    "034730",  # SK
    "096770",  # SK이노베이션
    "006400",  # 삼성SDI
    "207940",  # 삼성바이오로직스
    "068270",  # 셀트리온
    "035720",  # 카카오
    "028260",  # 삼성물산
    "012330",  # 현대모비스
    "009150",  # 삼성전기
    "018260",  # 삼성에스디에스
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        # Print URLs for priority tickers
        mapping = load_ticker_corp_map()
        for tk in PRIORITY_TICKERS:
            cc = mapping.get(tk)
            if cc:
                print(f"{tk}: {butler_summary_url(cc)}")
            else:
                print(f"{tk}: NO CORP_CODE")
    elif len(sys.argv) > 1 and sys.argv[1] == "save":
        # Read JSON from stdin and save
        json_str = sys.stdin.read()
        result = save_financials_result(json_str)
        print(result if result else "FAILED")
    else:
        print("Usage:")
        print("  python scrape_butler_financials.py list    # show priority URLs")
        print("  echo JSON | python scrape_butler_financials.py save  # save result")
