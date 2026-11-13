from chatfly.chat import answer, parse, reply

IN, KW, NEG = ["A", "B"], ["even", "odd"], ["not", "no", "never", "nor", "n't"]
Q = {"even": ["s0"], "odd": ["s1"]}


def test_parse():
    assert parse("Is A B A even?", IN, KW, NEG) == {"tokens": ["A", "B", "A"], "ask": "even"}
    assert parse("Is the number of A's in ABBA odd?", IN, KW, NEG) == {"tokens": list("ABBA"), "ask": "odd"}
    assert parse("A, B: even or odd?", IN, KW, NEG) == {"tokens": ["A", "B"], "ask": None}
    assert parse("Is A B C even?", IN, KW, NEG)["refuse"] == "unknown_tokens"
    assert parse("Is A B even, and is B A odd?", IN, KW, NEG)["refuse"] == "ambiguous_sequence"
    assert parse("Is ABBA a good band?", IN, KW, NEG)["refuse"] == "no_question"
    assert parse("Is the empty sequence even?", IN, KW, NEG)["refuse"] == "no_sequence"
    for t in ("Is ABBA not odd?", "Isn't ABBA even?", "Is A B even or not?"):
        assert parse(t, IN, KW, NEG)["refuse"] == "negation"
    assert "tokens" in parse("Is ABBA even? Nothing else.", IN, KW, NEG)  # "no" only as a word


def test_answer_and_reply():
    assert answer("s0", 0.5, "even", Q, 0.1) == "yes"
    assert answer("s0", 0.5, "odd", Q, 0.1) == "no"
    assert answer("s1", 0.5, None, Q, 0.1) == "odd"
    assert answer("s1", 0.05, "odd", Q, 0.1) == "uncertain"  # below threshold
    assert answer(None, 0.0, "odd", Q, 0.0) == "uncertain"
    ch = {"inputs": IN, "questions": Q}
    p = {"tokens": ["A"], "ask": "even"}
    assert "uncertain" in reply(p, "uncertain", 0.05, ch)
    assert reply(p, "no", 0.5, ch).startswith("The fly says no")
    assert reply({"refuse": "unknown_tokens", "unknown": ["C"]}, None, 0, ch).startswith("I can't")


def test_config_translation():
    """The shipped config loads as strings, and dev / out-of-scope sets behave (no fly)."""
    from pathlib import Path

    import numpy as np
    import yaml

    from chatfly.chat import translation_eval

    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "phase5_chat.yaml").read_text())
    assert all(isinstance(n, str) for n in cfg["chat"]["negations"])
    tr = translation_eval(cfg, np.random.default_rng(0))
    assert tr["dev"]["acc"] == 1.0 and tr["out_of_scope"]["refused"] == 1.0
