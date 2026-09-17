"""Minimal Telegram Bot API fixture for explicit deployment smoke tests."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a non-forwarding fake Telegram API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.delay <= 30:
        parser.error("--delay must be between 0 and 30 seconds")
    args.log.parent.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            size = int(self.headers.get("content-length", "0"))
            self.rfile.read(size)
            parts = self.path.strip("/").split("/", 1)
            bot = parts[0] if parts else ""
            method = parts[1] if len(parts) == 2 else ""
            token = bot.removeprefix("bot")
            record = {
                "bot_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "method": method,
            }
            with args.log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            time.sleep(args.delay)
            result = [] if method == "getUpdates" else {"message_id": 1}
            body = json.dumps({"ok": True, "result": result}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:
            pass

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
