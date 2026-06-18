#!/usr/bin/env python3
"""
Butler segment data re-scrape server with auto-navigation driver page.

Usage:
  1. Start: .venv/bin/python scripts/butler_rescrape_server.py
  2. Open http://127.0.0.1:18923/driver in Chrome
  3. The driver page opens Butler in an iframe-like flow and auto-scrapes all tickers

Or manual mode:
  - Navigate Chrome to each Butler page
  - Run extraction JS (see EXTRACT_JS below)
  - JS posts results here
"""
import json
import urllib.parse
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

BUTLER_REF_DIR = Path(__file__).parent / "butler_reference"
BUTLER_REF_DIR.mkdir(exist_ok=True)

MAPPING_FILE = Path("/tmp/ticker_corpcode_map_full.json")
with open(MAPPING_FILE) as f:
    TICKER_CORP_MAP = json.load(f)

URLS_FILE = Path("/tmp/butler_scrape_urls.json")
with open(URLS_FILE) as f:
    ALL_URLS = json.load(f)

received_business: dict[str, bool] = {}
received_geography: dict[str, bool] = {}
ticker_data: dict[str, dict] = {}


# The JS that extracts data and posts to our server, then navigates to next URL
EXTRACT_AND_NEXT_JS = r"""
(function() {
  function parseKrNum(s) {
    if (!s || s === '-' || s.trim() === '') return null;
    s = s.trim(); var neg = s.startsWith('-');
    if (neg) s = s.substring(1).trim();
    var total = 0;
    var joM = s.match(/([\d,]+)\s*조/); var ukM = s.match(/([\d,.]+)\s*억/);
    if (joM) total += parseFloat(joM[1].replace(/,/g,''))*10000;
    if (ukM) total += parseFloat(ukM[1].replace(/,/g,''));
    if (!joM && !ukM) { var n = parseFloat(s.replace(/,/g,'')); if(!isNaN(n)) return neg?-n:n; return null; }
    return neg ? -total : total;
  }
  var titleMatch = document.title.match(/(.+?)\((\d{6})\)/);
  var ticker = titleMatch ? titleMatch[2] : '';
  var corp_name = titleMatch ? titleMatch[1].trim() : '';
  var url = window.location.href;
  var tab = 'business';
  if (url.includes('biz-geography')) tab = 'geography';
  else if (url.includes('biz-product')) tab = 'product';
  var h2s = Array.from(document.querySelectorAll('h2')).map(function(h){return h.textContent.trim()});
  var metricH2s = h2s.filter(function(t){return t.includes('매출') || t.includes('영업') || t.includes('이익')});
  var tables = Array.from(document.querySelectorAll('table')).filter(function(t) {
    var fc = t.querySelector('tr th, tr td');
    return fc && fc.textContent.includes('단위');
  });
  var data = {};
  tables.forEach(function(table, idx) {
    var rows = table.querySelectorAll('tr');
    if (rows.length < 2) return;
    var hCells = rows[0].querySelectorAll('th, td');
    var periods = [];
    for (var j = 1; j < hCells.length; j++) {
      var p = hCells[j].textContent.trim().split('(')[0].trim();
      if (p) periods.push(p);
    }
    if (periods.length === 0) return;
    var metricName = idx < metricH2s.length ? metricH2s[idx] : 'metric_' + idx;
    var segs = [];
    for (var r = 1; r < rows.length; r++) {
      var cells = rows[r].querySelectorAll('th, td');
      if (cells.length < 2) continue;
      var name = cells[0].textContent.trim();
      if (['합계','전체','소계','총계'].indexOf(name) >= 0) continue;
      var vals = [];
      for (var j2 = 1; j2 < cells.length && j2 <= periods.length; j2++) {
        vals.push(parseKrNum(cells[j2].textContent));
      }
      segs.push([name, vals]);
    }
    data[metricName] = { p: periods, s: segs };
  });
  var payload = { ticker: ticker, corp_name: corp_name, tab: tab, data: data };
  return fetch('http://127.0.0.1:18923/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(function(r){return r.json()}).then(function(r){
    return JSON.stringify({ok:true, ticker:ticker, tab:tab, next_url: r.next_url || null});
  }).catch(function(e){
    return JSON.stringify({ok:false, error:e.message});
  });
})()
"""


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        path = self.path

        if path == "/save":
            self._handle_save(body)
        else:
            self._handle_save(body)

    def _handle_save(self, body: bytes):
        try:
            data = json.loads(body)
            ticker = data.get("ticker", "unknown")
            tab = data.get("tab", "unknown")
            seg_data = data.get("data", {})
            corp_name = data.get("corp_name", "")

            if ticker not in ticker_data:
                ticker_data[ticker] = {
                    "ticker": ticker,
                    "corp_name": corp_name,
                    "corp_code": TICKER_CORP_MAP.get(ticker, ""),
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "business": {},
                    "geography": {},
                    "product": {},
                }

            if corp_name:
                ticker_data[ticker]["corp_name"] = corp_name
            ticker_data[ticker]["scraped_at"] = datetime.now(timezone.utc).isoformat()

            if tab == "business":
                ticker_data[ticker]["business"] = seg_data
                received_business[ticker] = True
            elif tab == "geography":
                ticker_data[ticker]["geography"] = seg_data
                received_geography[ticker] = True

            out_path = BUTLER_REF_DIR / f"{ticker}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ticker_data[ticker], f, ensure_ascii=False, indent=2)

            biz_done = len(received_business)
            geo_done = len(received_geography)
            total = len(TICKER_CORP_MAP)

            # Find next URL
            current_key = f"{ticker}_{tab}"
            next_url = None
            found_current = False
            for item in ALL_URLS:
                key = f"{item['ticker']}_{item['tab']}"
                if found_current:
                    next_url = item["url"]
                    break
                if key == current_key:
                    found_current = True

            print(
                f"  ✓ {ticker} {tab} ({corp_name}) "
                f"[biz:{biz_done}/{total} geo:{geo_done}/{total}]",
                flush=True,
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp = {"ok": True, "ticker": ticker, "tab": tab, "next_url": next_url}
            self.wfile.write(json.dumps(resp).encode())
        except Exception as e:
            print(f"  ✗ Error: {e}", flush=True)
            self.send_response(500)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/status":
            self._handle_status()
        elif path == "/next":
            self._handle_next()
        elif path == "/urls":
            self._handle_urls()
        else:
            self._handle_status()

    def _handle_status(self):
        total = len(TICKER_CORP_MAP)
        biz_done = len(received_business)
        geo_done = len(received_geography)
        both_done = len(set(received_business) & set(received_geography))
        status = {
            "total": total,
            "business_done": biz_done,
            "geography_done": geo_done,
            "both_done": both_done,
            "complete": biz_done == total and geo_done == total,
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(status, indent=2).encode())

    def _handle_next(self):
        """Return next unprocessed URL."""
        for item in ALL_URLS:
            tk = item["ticker"]
            tab = item["tab"]
            if tab == "business" and tk not in received_business:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(item).encode())
                return
            if tab == "geography" and tk not in received_geography:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(item).encode())
                return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"done": True}).encode())

    def _handle_urls(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(ALL_URLS).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    port = 18923
    server = HTTPServer(("127.0.0.1", port), Handler)
    total = len(TICKER_CORP_MAP)
    print(f"Butler re-scrape server on http://127.0.0.1:{port}", flush=True)
    print(f"  Output: {BUTLER_REF_DIR}", flush=True)
    print(f"  Target: {total} tickers × 2 tabs = {total*2} pages", flush=True)
    print(f"  Status: GET /status", flush=True)
    print(f"  Next:   GET /next", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    biz = len(received_business)
    geo = len(received_geography)
    print(f"\nFinal: business={biz}/{total}, geography={geo}/{total}")


if __name__ == "__main__":
    main()
