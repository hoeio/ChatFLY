from pathlib import Path

import numpy as np
import yaml

from chatfly import speak, zumi3
from chatfly.zumi import parse


def cfg():
    c = yaml.safe_load(Path("configs/phase7e_zumi3.yaml").read_text())
    base = yaml.safe_load(Path(c["base"]).read_text())
    c = {**base, **c, "lang": {**base["lang"], **c["lang_overrides"]},
         "rng_streams": {**base["rng_streams"], **c["rng_streams"]}}
    c["corpus"].update(n_train=10)
    return c


def test_corpus3_replies_are_in_support_and_grammatical():
    c = cfg()
    lx = c["lang"]
    n = 0
    for d in zumi3.corpus3(c, 0, "train"):
        for u, reply, kind, support, _ in d:
            assert tuple(reply) in support
            assert all(speak.forms3(list(r), lx) in speak.FORMS3[kind] for r in support)
            n += lx["pronoun"] in u and "refuse" not in parse(" ".join(u), lx)
    assert n > 0  # the pronoun is used and parsed


def test_gru_gradient_matches_numeric():
    c = cfg()
    rc = {**c["baselines"]["rnn"], "hidden": 6, "noun_dim": 4}
    m = zumi3.GRU(speak.vocabulary(c["lang"]), rc, 0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 2, len(m.vocab) + 4))
    Y = rng.integers(len(m.vocab), size=(3, 2))
    _, g = m.loss_grad(X, Y)
    for k in ("Wz", "Un", "Wo"):
        i = (0, 1)
        m.p[k][i] += 1e-5
        lp, _ = m.loss_grad(X, Y)
        m.p[k][i] -= 2e-5
        lm, _ = m.loss_grad(X, Y)
        m.p[k][i] += 1e-5
        assert abs((lp - lm) / 2e-5 - g[k][i]) < 1e-6
