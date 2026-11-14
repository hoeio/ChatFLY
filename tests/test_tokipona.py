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
