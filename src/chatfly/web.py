"""Small local demo site for the Phase 5 chat layer (ADR 0009). Standard library only.

python -m chatfly.web configs/phase5_chat.yaml [--sim rate|spiking] [--seed N] [--port 8000]

Serves one page on 127.0.0.1 and POST /ask {"text": ...} -> chat.turn's result. Answers
come from a fly trained at startup; nothing is canned. Requests are handled one at a time
(the simulator is not thread-safe).
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files

from chatfly.chat import load, parse, turn


def make_handler(fly, ch, kind):
    page = files("chatfly").joinpath("web.html").read_bytes()
    lim = ch["web"]
    # state -> question keyword, for display only (e.g. s0 -> even)
    label = {s: k for k, acc in ch["questions"].items() for s in acc}
    info = {"inputs": ch["inputs"], "questions": list(ch["questions"]), "sim": kind,
            "labels": label, "threshold": ch["min_confidence"][kind]}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/info":
                self._send(200, json.dumps(info).encode())
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            if self.path != "/ask":
                return self._send(404, b'{"error": "not found"}')
            n = int(self.headers.get("Content-Length") or 0)
            if n > 4 * lim["max_chars"] + 100:
                return self._send(413, b'{"error": "too long"}')
            try:
                text = str(json.loads(self.rfile.read(n))["text"]).strip()
            except (ValueError, KeyError, TypeError):
                return self._send(400, b'{"error": "expected {\\"text\\": ...}"}')
            if not text or len(text) > lim["max_chars"]:
                return self._send(400, json.dumps(
                    {"error": f"ask in 1-{lim['max_chars']} characters"}).encode())
            toks = parse(text, ch["inputs"], list(ch["questions"]), ch["negations"]).get("tokens")
            if toks and len(toks) > lim["max_tokens"]:  # checked before the fly runs: long
                return self._send(400, json.dumps(  # loops are slow in the spiking model
                    {"error": f"at most {lim['max_tokens']} tokens per question"}).encode())
            self._send(200, json.dumps(turn(text, fly, ch, kind)).encode())

        def log_message(self, fmt, *args):
            pass

    return Handler


def main(argv=None):
    a, cfg, fly = load(argv, "Serve the chat demo page on localhost.")
    ch = cfg["chat"]
    srv = HTTPServer(("127.0.0.1", a.port), make_handler(fly, ch, a.sim))
    print(f"Demo at http://127.0.0.1:{a.port}  (Ctrl+C stops)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
