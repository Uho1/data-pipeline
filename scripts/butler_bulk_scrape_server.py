#!/usr/bin/env python3
"""Local HTTP server to receive Butler segment data from Chrome MCP JS."""
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

BUTLER_REF_DIR = Path(__file__).parent / "butler_reference"
BUTLER_REF_DIR.mkdir(exist_ok=True)

received = {}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            ticker = data.get("ticker", "unknown")
            path = BUTLER_REF_DIR / f"{ticker}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            received[ticker] = True
            print(f"  ✓ Saved {ticker} ({len(body)} bytes) → {path.name}", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "ticker": ticker}).encode())
        except Exception as e:
            print(f"  ✗ Error: {e}", flush=True)
            self.send_response(500)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress default logging


def main():
    port = 18923
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Butler scrape server listening on http://127.0.0.1:{port}", flush=True)
    print(f"Output dir: {BUTLER_REF_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"\nReceived {len(received)} tickers: {sorted(received.keys())}")


if __name__ == "__main__":
    main()
