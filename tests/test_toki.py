from pathlib import Path

import numpy as np
import yaml

from chatfly.toki import Binder, Odors, parse, roles, slots


def test_grammar_roles():
    assert roles(["jan", "pona", "mi", "li", "moku", "e", "kili", "lon", "tomo"]) == [
        "subj", "subj", "subj", "li", "pred", "e", "obj", "prep", "pobj"]
    assert parse(["tenpo", "ni", "la", "mi", "wile", "pali", "e", "ijo", "pi", "pona", "mute"])
    assert parse(["li", "e", "jan"]) is None
    assert slots(["sina", "pona", "ala"]) == (("sina",), ("pona",), (), True)


def test_binder_is_deterministic_and_order_sensitive():
    cfg = yaml.safe_load(Path("configs/phase6_zumi.yaml").read_text())
    od = Odors(cfg, 1, 300, 0.1)
    b = Binder(cfg, 1, od, {"n_kc": 4000, "kc_theta": 650, "fan_out": 30, "signed": True, "weighted": False})
    k = b.phrase("subj", ("jan", "pona"))
    assert np.array_equal(k, b.phrase("subj", ("jan", "pona")))
    assert (k > 100).sum() == 30 and not np.array_equal(k, b.phrase("subj", ("jan", "ike")))
