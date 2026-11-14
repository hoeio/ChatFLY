from chatfly import tokipona as tp


def test_word_list_and_table_cover_every_sentence():
    assert len(tp.PU) == 120 and len(tp.CONTENT) == 112
    words = tp.words_for(25, 0)
    contexts = {(s, c): t for s, c, t in tp.table(words)}
    for msg in tp.messages(words, 200, 0):
        state, cues, k = "S0", [w for w in msg if w], 0
        for w in tp.sentence(msg) + [tp.EOS]:
            assert contexts[(state, cues[k] if k < len(cues) else tp.NONE)] == w
            if w == tp.EOS:
                break
            state = tp.after(state, w)
            assert state != "X"
            k += k < len(cues) and cues[k] == w


def test_repeats_balance_the_end_context():
    from chatfly.tokipona import NONE, repeats, table, words_for

    tab = table(words_for(112, 0))
    r = dict(zip([(s, c) for s, c, _ in tab], repeats(tab, "state")))
    assert r[("PRED", NONE)] == 112 and r[("PRED", "mi")] == 1 and r[("S0", "mi")] == 1
    assert repeats(tab, "none") == [1] * len(tab)


def test_grammar_task_accepts_sentences_and_counts_states():
    import numpy as np

    from chatfly.tokipona import ACCEPT, classes, final, grammar_task, utterance, words_for

    task, words = grammar_task(), words_for(112, 0)
    gc = {"p_bare": 0.3, "p_mods": [0.5, 0.3, 0.2], "p_object": 0.5}
    rng = np.random.default_rng(0)
    assert all(final(task, classes(utterance(rng, words, gc))) in ACCEPT for _ in range(500))
    assert final(task, classes(["mi", "li", "moku"])) == "X"
    assert final(task, classes(["jan", "pona", "li", "moku", "e", "kili"])) == "O1"
    assert len(task["states"]) <= 30
