#!/usr/bin/env python3
"""
serve.py — CORS proxy for the Dynamo web UI.
Proxies requests from port 9999 to the Dynamo frontend on port 8000,
adding CORS headers so the browser-based chat UI can make requests.

Run on 192.168.1.26:
  python3 serve.py

Then open: http://192.168.1.26:9999/dynamo_chat.html
"""

import http.server
import urllib.request
import urllib.error
import os

UPSTREAM = "http://127.0.0.1:8000"
PORT = 9999
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress access logs

    def add_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self.add_cors_headers()
        self.end_headers()

    def do_GET(self):
        # Serve static files from the script directory
        if self.path == "/" or self.path == "/dynamo_chat.html":
            filepath = os.path.join(SCRIPT_DIR, "dynamo_chat.html")
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.add_cors_headers()
                self.end_headers()
                self.wfile.write(content)
                return
        self._proxy()

    def do_POST(self):
        self._proxy()

    def _proxy(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None

        url = f"{UPSTREAM}{self.path}"
        req = urllib.request.Request(url, data=body, method=self.command)
        req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))

        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                self.add_cors_headers()
                for key, val in resp.headers.items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.add_cors_headers()
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.add_cors_headers()
            self.end_headers()
            self.wfile.write(str(e).encode())


if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", PORT), ProxyHandler)
    print(f"CORS proxy running on http://0.0.0.0:{PORT}")
    print(f"Open: http://192.168.1.26:{PORT}/dynamo_chat.html")
    print(f"Proxying to: {UPSTREAM}")
    server.serve_forever()
