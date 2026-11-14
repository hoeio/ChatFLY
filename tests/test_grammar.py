from pathlib import Path

import numpy as np
import yaml

from chatfly import grammar, speak
from chatfly.zumi import parse


def cfg():
    c = yaml.safe_load(Path("configs/phase7c_grammar.yaml").read_text())
    base = yaml.safe_load(Path(c["base"]).read_text())
    c = {**base, **c, "lang": {**base["lang"], **c["lang_overrides"]}}
    c["corpus"].update(n_train=20, n_dev=5, n_test=20)
    return c


def test_dfa_agrees_with_parser():
    c = cfg()
    lx, task = c["lang"], grammar.dfa(c["lang"])
    rng = np.random.default_rng(0)
    vocab = task["inputs"][:-1] + ["fire", "cat"]
    utts = [u for d in speak.corpus(c, 0, "test") for u, _, _ in d]
    utts += [[str(w) for w in rng.choice(vocab, rng.integers(1, 6))] for _ in range(5000)]
    for u in utts:
        ok = "refuse" not in parse(" ".join(u), lx)
        assert (grammar.true_state(u, task) in grammar.ACCEPT) == ok, u


def test_property_order_is_strict():
    lx = cfg()["lang"]
    assert "refuse" not in parse("sun ka bon gu ho.", lx)
    assert parse("sun ka gu bon.", lx)["refuse"] == "property_order"


def test_grammar_fly_tracks_utterances():
    c = cfg()
    g = grammar.GrammarFly("rate", c, 0, "routed_shared", "true", "state")
    g.thr = 0.05
    s = grammar.utterance_scores(g, speak.corpus(c, 0, "test"))
    assert s["utterance_acc"] > 0.95 and s["bad_rejected"] > 0.9
