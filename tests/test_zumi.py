from pathlib import Path

import yaml

from chatfly.zumi import Conversation, parse

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/phase6_zumi.yaml").read_text())
LX = CFG["lang"]


def test_parse():
    assert parse("zu ki zu nu?", LX) == {"restart": False, "suppose": False,
                                          "moves": ["zu", "ki", "zu"], "ask": "where"}
    assert parse("Zu, zu. Fi ba?", LX)["ask"] == "fi"
    assert parse("si ki nu", LX)["suppose"] is True
    assert parse("bzz", LX)["restart"] is True
    assert parse("zu.", LX)["ask"] is None
    assert parse("zu xo nu?", LX) == {"refuse": "unknown_word", "word": "xo"}
    assert parse("nu zu?", LX)["refuse"] == "word_order"
    assert parse("zu si nu?", LX)["refuse"] == "word_order"
    assert parse("si zu.", LX)["refuse"] == "suppose_needs_question"
    assert parse("?", LX)["refuse"] == "empty"
    assert parse("where is the fly?", LX)["refuse"] == "unknown_word"


def test_configs_agree_and_load_as_strings():
    fsm = yaml.safe_load((ROOT / "configs/phase6_zumi_fsm.yaml").read_text())
    assert fsm["tasks"][0]["delta"] == CFG["chat"]["task"]["delta"]
    assert set(CFG["chat"]["task"]["delta"]) == set(LX["places"])
    assert all(isinstance(v, str) for v in LX["replies"].values())


class FakeSim:
    """Decodes the true next state with confidence conf: tests the conversation logic only."""

    def __init__(self, task, conf):
        self.task, self.conf, self.n_comp = task, conf, len(task["states"])


def test_conversation(monkeypatch):
    import chatfly.zumi as z
    from chatfly.fsm import make_task

    task = make_task(CFG["chat"]["task"], None)
    calls = []

    def fake_ask(sim, pats, task, dur, moves, trace, start):
        calls.append(start)
        s = start
        for m in moves:
            s = task["delta"][s][m]
        return s, sim.conf

    monkeypatch.setattr(z, "ask_fly", fake_ask)
    conv = Conversation((FakeSim(task, 0.5), None, task, CFG), LX, min_conf=0.1)
    assert conv.turn("zu nu?")["reply"] == "fi"
    assert conv.turn("si zu nu?")["reply"] == "po"   # suppose: not committed
    assert conv.turn("nu?")["reply"] == "fi"
    assert conv.turn("ki ne ba?")["reply"] == "ya"
    assert conv.turn("po ba?")["reply"] == "na"
    assert conv.turn("ta.")["reply"] == "o"
    assert conv.turn("blah")["reply"] == "e"
    conv.sim.conf = 0.05                              # below threshold: the fly gets lost
    assert conv.turn("zu.")["reply"] == "hm"
    assert conv.turn("nu?")["reply"] == "hm" and conv.place is None
    conv.sim.conf = 0.5
    assert conv.turn("bzz nu?")["reply"] == "ne"


def test_web_demo_zumi():
    """Zumi demo server on a real rate fly: page, memory across turns, a refusal, the limit."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import HTTPServer
    from types import SimpleNamespace

    from chatfly.chat import fly_for
    from chatfly.web import make_handler, zumi_app

    a = SimpleNamespace(sim="rate", seed=0)
    srv = HTTPServer(("127.0.0.1", 0), make_handler(*zumi_app(fly_for("rate", CFG, 0), CFG, a)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"

    def ask(text):
        req = urllib.request.Request(url + "/ask", data=json.dumps({"text": text}).encode(),
                                     method="POST")
        try:
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    try:
        assert json.loads(urllib.request.urlopen(url + "/info").read())["mode"] == "zumi"
        out = ask("zu nu?")
        assert out["reply"] in ("fi", "hm") and out["start"] == "ne" and len(out["trace"]) == 1
        assert ask("nu?")["place"] == out["place"]          # remembered across requests
        assert ask("nu zu?")["reply"] == "e"
        assert ask("fire ka ho.")["reply"] == "o"
        out = ask("fire ke?")
        assert out["reply"] in ("ho", "hm") and len(out["memory"]) == len(LX["props"])
        assert "error" in ask("zu " * 41 + "nu?")           # max_tokens
    finally:
        srv.shutdown()


def test_parse_memory():
    assert parse("Fire ka mal ho.", LX) == {"noun": "fire", "kind": "teach", "props": ["mal", "ho"]}
    assert parse("fire ke?", LX) == {"noun": "fire", "kind": "what", "props": []}
    assert parse("fire ku ba?", LX) == {"noun": "fire", "kind": "ask", "props": ["ku"]}
    assert parse("fire ka", LX)["refuse"] == "ka_needs_property"
    assert parse("fire ka bon mal", LX)["refuse"] == "contradiction"
    assert parse("fire ka nice", LX) == {"refuse": "unknown_word", "word": "nice"}
    assert parse("zu ka bon", LX) == {"refuse": "not_a_noun", "word": "zu"}
    assert parse("r2d2 ke?", LX)["refuse"] == "not_a_noun"
    assert parse("fi ba?", LX)["ask"] == "fi"            # ring sentences are unchanged
    assert parse("fire bon", LX)["refuse"] == "unknown_word"


class FakeMemory:
    """Remembers what it was taught exactly; tests the conversation logic only."""

    def __init__(self):
        from chatfly.zumi import props

        self.prop, self.facts = props(LX), {}

    def teach(self, noun, prop):
        self.facts[(noun, self.prop[prop][0])] = self.prop[prop][1]

    def ask(self, noun, pair):
        side = self.facts.get((noun, pair))
        return (None, 0.0) if side is None else (side, 0.5)


def test_conversation_memory():
    from chatfly.fsm import make_task

    task = make_task(CFG["chat"]["task"], None)
    conv = Conversation((FakeSim(task, 0.5), None, task, CFG), LX, 0.1, FakeMemory(), 0.1)
    assert conv.turn("ice ke?")["reply"] == "hm"
    assert conv.turn("fire ka mal ho.")["reply"] == "o"
    assert conv.turn("fire ke?")["reply"] == "mal ho"
    assert conv.turn("fire ku ba?")["reply"] == "na"
    assert conv.turn("fire ho ba?")["reply"] == "ya"
    assert conv.turn("fire gu ba?")["reply"] == "hm"     # never told big/small
    assert conv.turn("fire ka bon.")["reply"] == "o"     # changed its mind
    assert conv.turn("fire ke?")["reply"] == "bon ho"
    assert conv.turn("fire ka bon mal.")["reply"] == "e"
    no_mem = Conversation((FakeSim(task, 0.5), None, task, CFG), LX, 0.1)
    assert no_mem.turn("fire ke?")["reply"] == "e"


def test_memory_learns():
    """A real rate memory fly: taught facts come back, a never-heard noun gets no answer."""
    from chatfly.zumi import Memory

    m = Memory("rate", CFG, 0)
    for noun, prop in (("fire", "ho"), ("ice", "ku"), ("sugar", "bon")):
        m.teach(noun, prop)
    assert m.ask("fire", 2)[0] == 0 and m.ask("ice", 2)[0] == 1 and m.ask("sugar", 0)[0] == 0
    assert m.ask("rock", 2)[1] < 0.05 < m.ask("fire", 2)[1]  # never heard: near-tie, below threshold
