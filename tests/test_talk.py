from pathlib import Path

import yaml

from chatfly import speak, talk


def cfg():
    c = yaml.safe_load(Path("configs/phase7d_talk.yaml").read_text())
    base = yaml.safe_load(Path(c["base"]).read_text())
    c = {**base, **c, "lang": {**base["lang"], **c["lang_overrides"]},
         "rng_streams": {**base["rng_streams"], **c["rng_streams"]}}
    c["corpus"].update(n_train=40, n_test=5)
    c["speak"]["n_train_dialogues"] = 40
    return c


def test_talker_answers_from_the_flies():
    c = cfg()
    tk = talk.Talker("rate", c, 0)
    assert tk.comp.accuracy() == 1.0
    r = speak.evaluate(tk, speak.corpus(c, 0, "test"), c["lang"])
    assert r["exact"] > 0.9
    sham = talk.Talker("rate", c, 0, "shuffled")
    assert speak.evaluate(sham, speak.corpus(c, 0, "test"), c["lang"])["exact"] < r["exact"] - 0.2
