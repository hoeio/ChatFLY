from pathlib import Path

import yaml

from chatfly import speak
from chatfly.zumi import parse


def cfg(**corpus):
    c = yaml.safe_load(Path("configs/phase7a_corpus.yaml").read_text())
    c = {**yaml.safe_load(Path(c["base"]).read_text()), **c}
    c["corpus"].update(n_train=30, n_dev=5, n_test=10, **corpus)
    return c


def test_corpus_is_consistent():
    c = cfg()
    lx = c["lang"]
    for d in speak.corpus(c, 0, "test"):
        for user, reply, kind in d:
            assert speak.kind_of(parse(" ".join(user), lx)) == kind
            assert speak.grammatical(reply, lx) and speak.appropriate(kind, reply, lx)


def test_oracle_scores_one_and_baselines_run():
    c = cfg()
    lx, test = c["lang"], speak.corpus(c, 0, "test")
    r = speak.evaluate(speak.Oracle(c), test, lx)
    assert r["exact"] == r["grammatical"] == r["appropriate"] == r["next_word_acc"] == 1.0
    vocab = speak.vocabulary(lx)
    big = speak.NGram(3, 0.01, vocab).fit(speak.corpus(c, 0, "train"))
    assert speak.evaluate(big, test, lx)["appropriate"] > 0.8
    uni = speak.evaluate(speak.NGram(1, 0.01, vocab), test, lx)
    assert uni["perplexity"] > 1.5


def test_reply_grammar():
    lx = cfg()["lang"]
    assert speak.grammatical(["mal", "ho"], lx) and not speak.grammatical(["ho", "mal"], lx)
    assert not speak.grammatical(["bon", "mal"], lx) and not speak.grammatical([], lx)
    assert speak.appropriate("where", ["fi"], lx) and not speak.appropriate("where", ["ya"], lx)
