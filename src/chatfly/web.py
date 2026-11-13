"""Small local demo site for the chat layers. Standard library only.

python -m chatfly.web configs/phase5_chat.yaml [--sim rate|spiking] [--seed N] [--port 8000]
python -m chatfly.web configs/phase6_zumi.yaml  (Zumi, ADR 0010; picked when the config has `lang`)

Serves one page on 127.0.0.1 and POST /ask {"text": ...} -> the turn's result. Answers
come from a fly trained at startup; nothing is canned. Requests are handled one at a time
(the simulator is not thread-safe). Zumi keeps one conversation for every visitor.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files

from chatfly import zumi
from chatfly.chat import load, parse, turn


def chat_app(fly, ch, kind):
    """Phase 5 parity questions -> (info, limits, token count, ask)."""
    label = {s: k for k, acc in ch["questions"].items() for s in acc}  # s0 -> even (display)
    info = {"mode": "parity", "inputs": ch["inputs"], "questions": list(ch["questions"]),
            "sim": kind, "labels": label, "threshold": ch["min_confidence"][kind]}
    def n_tokens(t):
        return len(parse(t, ch["inputs"], list(ch["questions"]), ch["negations"]).get("tokens") or [])

    return info, ch["web"], n_tokens, lambda t: turn(t, fly, ch, kind)


def zumi_app(fly, lx, kind):
    """One Zumi conversation -> (info, limits, token count, ask). The reply carries the fly's
    place after the turn (None = lost) so the page can show its memory."""
    conv = zumi.Conversation(fly, lx, lx["min_confidence"][kind])
    info = {"mode": "zumi", "sim": kind, "threshold": conv.min_conf, "labels": lx["places"],
            "places": lx["places"], "moves": lx["moves"], "place": conv.place}

    def ask(text):
        return {**conv.turn(text), "place": conv.place}

    return info, lx["web"], lambda t: len(zumi.parse(t, lx).get("moves", [])), ask


def make_handler(info, lim, n_tokens, ask):
    page = files("chatfly").joinpath("web.html").read_bytes()

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
            if n_tokens(text) > lim["max_tokens"]:  # checked before the fly runs: long
                return self._send(400, json.dumps(  # loops are slow in the spiking model
                    {"error": f"at most {lim['max_tokens']} tokens per question"}).encode())
            out = ask(text)
            info["place"] = out.get("place", info.get("place"))
            self._send(200, json.dumps(out).encode())

        def log_message(self, fmt, *args):
            pass

    return Handler


def main(argv=None):
    a, cfg, fly = load(argv, "Serve the chat demo page on localhost.")
    app = zumi_app(fly, cfg["lang"], a.sim) if "lang" in cfg else chat_app(fly, cfg["chat"], a.sim)
    srv = HTTPServer(("127.0.0.1", a.port), make_handler(*app))
    print(f"Demo at http://127.0.0.1:{a.port}  (Ctrl+C stops)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
