#!/usr/bin/env python3
"""
Butler capacity (가동률) data scraper.

Scrapes 생산능력/생산실적/가동률 tables from Butler's biz-capacity tab
and merges into existing butler_reference/*.json files.

Usage:
  1. Start server:  .venv/bin/python scripts/scrape_butler_capacity.py
  2. Ensure Chrome is logged into Butler
  3. Open http://127.0.0.1:18924/driver in Chrome
     → auto-navigates through all 134 tickers and extracts capacity data
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

BUTLER_REF_DIR = Path(__file__).parent / "butler_reference"
BUTLER_REF_DIR.mkdir(exist_ok=True)

MAPPING_FILE = Path("/tmp/ticker_corpcode_map_full.json")
with open(MAPPING_FILE) as f:
    TICKER_CORP_MAP: dict[str, str] = json.load(f)

# Build ordered URL list for all tickers
ALL_URLS: list[dict] = []
for ticker, corp_code in sorted(TICKER_CORP_MAP.items()):
    ALL_URLS.append({
        "ticker": ticker,
        "corp_code": corp_code,
        "url": f"https://www.butler.works/ko/companies/{corp_code}?corpLeftPanel=summary&corpMiddlePanel=biz-capacity",
    })

received_capacity: dict[str, bool] = {}

# JavaScript to extract capacity table data and post to server, then navigate next
EXTRACT_CAPACITY_JS = r"""
(function() {
  function parseKrNum(s) {
    if (!s || s === '-' || s.trim() === '') return null;
    s = s.trim();
    var neg = s.startsWith('-');
    if (neg) s = s.substring(1).trim();
    // percentage
    if (s.endsWith('%')) {
      var n = parseFloat(s.replace('%','').replace(/,/g,''));
      return isNaN(n) ? null : (neg ? -n : n);
    }
    var total = 0;
    var joM = s.match(/([\d,]+)\s*조/);
    var ukM = s.match(/([\d,]+)\s*억/);
    var manM = s.match(/([\d,]+)\s*만/);
    // remainder after removing 조/억/만 tokens
    var rest = s;
    if (joM) rest = rest.replace(joM[0], '');
    if (ukM) rest = rest.replace(ukM[0], '');
    if (manM) rest = rest.replace(manM[0], '');
    rest = rest.trim().replace(/,/g, '');
    var remNum = rest ? parseFloat(rest) : 0;
    if (isNaN(remNum)) remNum = 0;
    if (joM) total += parseFloat(joM[1].replace(/,/g, '')) * 1e12;
    if (ukM) total += parseFloat(ukM[1].replace(/,/g, '')) * 1e8;
    if (manM) total += parseFloat(manM[1].replace(/,/g, '')) * 1e4;
    total += remNum;
    if (!joM && !ukM && !manM) {
      var n = parseFloat(s.replace(/,/g, ''));
      return isNaN(n) ? null : (neg ? -n : n);
    }
    return neg ? -total : total;
  }

  var titleMatch = document.title.match(/(.+?)\((\d{6})\)/);
  var ticker = titleMatch ? titleMatch[2] : '';
  var corp_name = titleMatch ? titleMatch[1].trim() : '';

  // Find the capacity table — look for table with section headers 생산능력/생산실적/가동률
  var tables = Array.from(document.querySelectorAll('table'));
  var capTable = null;
  for (var ti = 0; ti < tables.length; ti++) {
    var txt = tables[ti].textContent;
    if (txt.includes('생산능력') && txt.includes('생산실적') && txt.includes('가동률')) {
      capTable = tables[ti];
      break;
    }
  }

  if (!capTable) {
    // No capacity data for this ticker
    var payload = { ticker: ticker, corp_name: corp_name, has_data: false, data: null };
    return fetch('http://127.0.0.1:18924/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function(r){return r.json()}).then(function(r){
      if (r.next_url) { window.location.href = r.next_url; }
      return JSON.stringify({ok: true, ticker: ticker, has_data: false, next_url: r.next_url});
    }).catch(function(e){ return JSON.stringify({ok: false, error: e.message}); });
  }

  var rows = capTable.querySelectorAll('tr');

  // First row: header with period columns
  var headerCells = rows[0].querySelectorAll('th, td');
  var periods = [];
  for (var j = 1; j < headerCells.length; j++) {
    var pText = headerCells[j].textContent.trim().split('(')[0].trim();
    if (pText) periods.push(pText);
  }

  // Parse remaining rows into sections
  var sections = {};
  var currentSection = null;
  var sectionNames = ['생산능력', '생산실적', '가동률'];

  for (var r = 1; r < rows.length; r++) {
    var cells = rows[r].querySelectorAll('th, td');
    if (cells.length < 2) continue;
    var firstCell = cells[0].textContent.trim();

    // Check if this is a section header
    var isSection = false;
    for (var si = 0; si < sectionNames.length; si++) {
      if (firstCell === sectionNames[si]) {
        currentSection = firstCell;
        sections[currentSection] = [];
        isSection = true;
        break;
      }
    }
    if (isSection) continue;
    if (!currentSection) continue;

    // This is a data row: segment name + values
    var segName = firstCell;
    if (!segName || segName === '합계' || segName === '전체' || segName === '소계') continue;
    var vals = [];
    for (var j2 = 1; j2 < cells.length && j2 <= periods.length; j2++) {
      vals.push(parseKrNum(cells[j2].textContent));
    }
    sections[currentSection].push([segName, vals]);
  }

  var result = { periods: periods };
  for (var key in sections) {
    result[key] = sections[key];
  }

  var payload = { ticker: ticker, corp_name: corp_name, has_data: true, data: result };
  return fetch('http://127.0.0.1:18924/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(function(r){return r.json()}).then(function(r){
    if (r.next_url) { window.location.href = r.next_url; }
    return JSON.stringify({ok: true, ticker: ticker, has_data: true, segments: Object.keys(sections).length, next_url: r.next_url});
  }).catch(function(e){ return JSON.stringify({ok: false, error: e.message}); });
})()
"""

# Driver page HTML — opens in Chrome, auto-extracts and navigates
DRIVER_HTML = """<!DOCTYPE html>
<html><head><title>Butler Capacity Scraper</title>
<style>
body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
#status { white-space: pre-wrap; max-height: 80vh; overflow-y: auto; }
.ok { color: #4caf50; } .skip { color: #ff9800; } .err { color: #f44336; }
h2 { color: #00bcd4; }
</style></head>
<body>
<h2>Butler Capacity (가동률) Scraper</h2>
<div>Total tickers: <span id="total">?</span> | Done: <span id="done">0</span> | With data: <span id="has_data">0</span></div>
<hr>
<div id="status"></div>
<script>
var EXTRACT_JS = """ + json.dumps(EXTRACT_CAPACITY_JS) + """;

var statusEl = document.getElementById('status');
var doneEl = document.getElementById('done');
var hasDataEl = document.getElementById('has_data');

// First get the URL list
fetch('http://127.0.0.1:18924/urls').then(r=>r.json()).then(function(urls){
  document.getElementById('total').textContent = urls.length;
  if (urls.length === 0) { statusEl.textContent = 'No URLs to process.'; return; }

  // Create hidden iframe for navigation
  var iframe = document.createElement('iframe');
  iframe.style.width = '100%';
  iframe.style.height = '600px';
  iframe.style.border = '1px solid #333';
  document.body.appendChild(iframe);

  var idx = 0;
  function processNext() {
    if (idx >= urls.length) {
      statusEl.innerHTML += '<div class="ok">\\n=== ALL DONE ===</div>';
      return;
    }
    var item = urls[idx];
    statusEl.innerHTML += '<div>Loading ' + item.ticker + ' (' + (idx+1) + '/' + urls.length + ')...</div>';
    statusEl.scrollTop = statusEl.scrollHeight;
    iframe.src = item.url;
    iframe.onload = function() {
      // Wait for table to render
      setTimeout(function() {
        try {
          var iframeDoc = iframe.contentWindow;
          // Execute extraction JS in iframe context
          iframeDoc.eval(EXTRACT_JS);
          // Wait for fetch to complete
          setTimeout(function() {
            idx++;
            fetch('http://127.0.0.1:18924/status').then(r=>r.json()).then(function(st){
              doneEl.textContent = st.done;
              hasDataEl.textContent = st.has_data;
            });
            processNext();
          }, 2000);
        } catch(e) {
          statusEl.innerHTML += '<div class="err">Error on ' + item.ticker + ': ' + e.message + '</div>';
          idx++;
          setTimeout(processNext, 1000);
        }
      }, 3000);
    };
  }
  processNext();
});
</script>
</body></html>
"""

# Alternative: simpler driver that just provides the JS for manual console use
MANUAL_DRIVER_HTML = """<!DOCTYPE html>
<html><head><title>Butler Capacity - Manual Mode</title>
<style>
body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
textarea { width: 100%; height: 200px; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; padding: 10px; font-family: monospace; font-size: 12px; }
h2 { color: #00bcd4; }
.info { color: #ff9800; margin: 10px 0; }
</style></head>
<body>
<h2>Butler Capacity Scraper - Manual Mode</h2>
<p class="info">1. Navigate to a Butler company's 가동률 tab in another tab</p>
<p class="info">2. Copy the JS below and paste into that tab's console</p>
<p class="info">3. The script extracts data, sends to server, and auto-navigates to next ticker</p>
<textarea id="js" readonly></textarea>
<script>
document.getElementById('js').value = """ + json.dumps(EXTRACT_CAPACITY_JS) + """;
</script>
<hr>
<h3>Server Status</h3>
<div id="status">Loading...</div>
<script>
setInterval(function(){
  fetch('http://127.0.0.1:18924/status').then(r=>r.json()).then(function(s){
    document.getElementById('status').innerHTML =
      'Total: ' + s.total + '<br>Done: ' + s.done + '<br>With data: ' + s.has_data +
      '<br>Complete: ' + (s.complete ? 'YES' : 'NO');
  }).catch(function(e){ document.getElementById('status').textContent = 'Server not reachable'; });
}, 3000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self._handle_save(body)

    def _handle_save(self, body: bytes):
        try:
            data = json.loads(body)
            ticker = data.get("ticker", "unknown")
            corp_name = data.get("corp_name", "")
            has_data = data.get("has_data", False)
            cap_data = data.get("data")

            # Load existing JSON
            out_path = BUTLER_REF_DIR / f"{ticker}.json"
            if out_path.exists():
                with open(out_path, encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = {
                    "ticker": ticker,
                    "corp_name": corp_name,
                    "corp_code": TICKER_CORP_MAP.get(ticker, ""),
                    "scraped_at": "",
                    "business": {},
                    "geography": {},
                    "product": {},
                }

            # Update capacity data
            if has_data and cap_data:
                existing["capacity"] = cap_data
                tag = "✓ data"
            else:
                existing["capacity"] = None
                tag = "○ no data"

            existing["capacity_scraped_at"] = datetime.now(timezone.utc).isoformat()
            if corp_name:
                existing["corp_name"] = corp_name

            # Save
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

            received_capacity[ticker] = has_data

            # Determine next URL
            current_idx = None
            for i, item in enumerate(ALL_URLS):
                if item["ticker"] == ticker:
                    current_idx = i
                    break
            next_url = None
            if current_idx is not None and current_idx + 1 < len(ALL_URLS):
                next_url = ALL_URLS[current_idx + 1]["url"]

            done = len(received_capacity)
            with_data = sum(1 for v in received_capacity.values() if v)
            total = len(ALL_URLS)
            print(
                f"  {tag} {ticker} ({corp_name}) [{done}/{total}, data:{with_data}]",
                flush=True,
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": True, "ticker": ticker,
                "next_url": next_url,
            }).encode())

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
        elif path == "/driver":
            self._handle_driver()
        elif path == "/manual":
            self._handle_manual()
        elif path == "/js":
            self._handle_js()
        else:
            self._handle_status()

    def _handle_status(self):
        done = len(received_capacity)
        with_data = sum(1 for v in received_capacity.values() if v)
        total = len(ALL_URLS)
        status = {
            "total": total,
            "done": done,
            "has_data": with_data,
            "complete": done == total,
        }
        self._json_response(status)

    def _handle_next(self):
        for item in ALL_URLS:
            if item["ticker"] not in received_capacity:
                self._json_response(item)
                return
        self._json_response({"done": True})

    def _handle_urls(self):
        # Return only unprocessed URLs
        remaining = [item for item in ALL_URLS if item["ticker"] not in received_capacity]
        self._json_response(remaining)

    def _handle_driver(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(DRIVER_HTML.encode("utf-8"))

    def _handle_manual(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(MANUAL_DRIVER_HTML.encode("utf-8"))

    def _handle_js(self):
        """Return just the extraction JS for copy-paste."""
        self.send_response(200)
        self.send_header("Content-Type", "text/javascript; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(EXTRACT_CAPACITY_JS.encode("utf-8"))

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    port = 18924
    server = HTTPServer(("127.0.0.1", port), Handler)
    total = len(ALL_URLS)
    print(f"Butler Capacity Scraper on http://127.0.0.1:{port}", flush=True)
    print(f"  Output:  {BUTLER_REF_DIR}", flush=True)
    print(f"  Target:  {total} tickers", flush=True)
    print(f"  Driver:  http://127.0.0.1:{port}/driver (auto mode - iframe)", flush=True)
    print(f"  Manual:  http://127.0.0.1:{port}/manual (copy JS to console)", flush=True)
    print(f"  JS only: http://127.0.0.1:{port}/js", flush=True)
    print(f"  Status:  http://127.0.0.1:{port}/status", flush=True)
    print(f"  Next:    http://127.0.0.1:{port}/next", flush=True)
    print(flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    done = len(received_capacity)
    with_data = sum(1 for v in received_capacity.values() if v)
    print(f"\nFinal: {done}/{total} processed, {with_data} with capacity data")


if __name__ == "__main__":
    main()
