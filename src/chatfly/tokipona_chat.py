"""Phase 8d (ADR 0013): a Toki Pona question-answering exchange carried by flies (H4).

The user teaches facts ("jan pona li suli", "jan pona li suli ala") and asks
("jan pona li suli anu seme?"). The fly answers with a whole sentence: the statement, the
statement with `ala`, or "mi sona ala" (I don't know). Flies, all from earlier phases:
- class fly (8b-3): word -> class, here over the 8d classes;
- grammar fly (8b): the 8d grammar over classes, run on the user's words and on the reply;
- memory fly (8c): phrase -> yes/no per predicate word;
- speaker fly (8a-3): (grammar state, cue) -> next word.
Interface rules (glue, counted in ADR 0013): roles read off the grammar fly's states (subject
words, predicate word); the predicate word selects the memory output pair; the reply plan
(cue list) from the question's words and the memory's answer; the cue pointer.
"""

import copy
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chatfly.tokipona import (
    EOS,
    NONE,
    ClassFly,
    Speaker,
    phrase_memory,
    repeats,
    teaching,
    words_for,
)

log = logging.getLogger("chatfly")
CLASSES_D = ("B", "W", "li", "ala", "anu", "seme")
ACCEPT_D = {"P", "PN", "Q"}
UNKNOWN = ["mi", "sona", "ala"]


def classes_d(words):
    return [w if w in ("li", "ala", "anu", "seme") else "B" if w in ("mi", "sina") else "W" for w in words]


def grammar_d() -> dict:
    """Subject (mi/sina alone, or a word with up to two modifiers, then li), a one-word predicate,
    then `ala` (a negative statement, PN) or `anu seme` (a question, Q), or nothing (P)."""
    st = ["S0", "S1", "S2", "S3", "MI", "LI", "P", "PN", "Q1", "Q", "X"]
    d = {s: dict.fromkeys(CLASSES_D, "X") for s in st}
    d["S0"].update(B="MI", W="S1")
    d["S1"].update(B="S2", W="S2", li="LI")
    d["S2"].update(B="S3", W="S3", li="LI")
    d["S3"]["li"] = "LI"
    d["MI"].update(B="P", W="P")
    d["LI"].update(B="P", W="P")
    d["P"].update(ala="PN", anu="Q1")
    d["Q1"]["seme"] = "Q"
    return {"name": "tokipona_8d", "states": st, "inputs": list(CLASSES_D), "delta": d}


def speaker_table(content):
    """(state, cue) -> word for the replies: a subject of a head and one modifier, li, the
    predicate, optional ala; or mi sona ala."""
    t = [(s, w, w) for w in content for s in ("S0", "S1", "LI", "MI")]
    t += [("S2", w, "li") for w in content]
    return t + [("S0", "mi", "mi"), ("P", "ala", "ala"), ("P", NONE, EOS), ("PN", NONE, EOS)]


class Chat:
    def __init__(self, kind, cfg, seed, controls=()):
        dc, rs = cfg["chat8d"], np.random.default_rng([seed, 8, 12])
        self.vocab = words_for(112, seed) + ["li", "anu"]
        content = [w for w in self.vocab if w not in ("li", "anu", "ala", "seme")]
        c = copy.deepcopy(cfg)
        cf = dc["class_fly"]
        c["mb"]["n_kc"], c["rate"]["kc_theta"] = cf["n_kc"], cf["kc_theta"]
        self.cls = ClassFly(kind, c, seed, None, "no_kc" if "class_no_kc" in controls else None,
                            cf["balance"], vocab=self.vocab, cls=(CLASSES_D, classes_d))
        from chatfly.grammar import GrammarFly

        g, gf = grammar_d(), dc["grammar_fly"]
        c = copy.deepcopy(cfg)
        c["mb"]["n_kc"], c["rate"]["kc_theta"] = gf["n_kc"], gf["kc_theta"]
        seq, _ = teaching(g, "table", [], c["protocol"]["n_train"], np.random.default_rng([seed, 8, 7]))
        self.gram = GrammarFly(kind, c, seed, "single", "true", "pair", task=g, teach=seq)
        self.gram.thr = gf["threshold"]
        # subjects: heads x shared modifiers (8c level 1); predicates: the memory's output pairs
        pool = [str(w) for w in rs.permutation([w for w in content if w not in ("mi", "sina", "sona")])]
        nh, nm, npr = dc["n_heads"], dc["n_mods"], dc["n_preds"]
        self.preds, heads, mods = pool[:npr], pool[npr:npr + nh], pool[npr + nh:npr + nh + nm + 1]
        self.subjects = [(h, m) for h in heads for m in mods[:nm]]
        self.unknown = [(h, mods[nm]) for h in heads]
        mc = dc["memory_fly"]
        c = copy.deepcopy(cfg)
        c["mb"]["n_kc"], c["rate"]["kc_theta"] = mc["n_kc"], mc["kc_theta"]
        self.mem = phrase_memory(kind, c, seed, self.preds, "no_kc" if "memory_no_kc" in controls else None)
        self.mem_thr = mc["threshold"]
        sc = dc["speaker"]
        c = copy.deepcopy(cfg)
        c["capacity"] = dict(sc)
        self.sp = Speaker(kind, c, seed, content + ["ala"], sc["coding"], sc["n_kc"],
                          "no_kc" if "speaker_no_kc" in controls else None, states=g["states"])
        tab = speaker_table(content)
        idx = {w: i for i, w in enumerate(self.sp.outs)}
        order = np.repeat(np.arange(len(tab)), repeats(tab, sc["balance"]))
        for _ in range(c["protocol"]["n_train"]):
            for j in self.sp.rng.permutation(order):
                s, cue, t = tab[j]
                dan = np.ones(self.sp.n)
                dan[self.sp.code[idx[t]]] = 0.0
                self.sp.sim.present(self.sp.stim(s, cue), self.sp.dur, dan)

    def read(self, words):
        """Run the class and grammar flies; roles from the states. Returns (final state,
        subject words, predicate word)."""
        s, subj, pred = "S0", [], None
        for w in words:
            k = self.cls(w)
            nxt = "X" if k is None else self.gram.step(s, k)
            if nxt in ("S1", "S2", "S3", "MI"):
                subj.append(w)
            elif nxt == "P":
                pred = w
            s = nxt
        return s, subj, pred

    def hear(self, words):
        """A statement teaches the memory; a question gets a spoken reply (list of words)."""
        s, subj, pred = self.read(words)
        if s in ("P", "PN") and pred in self.preds:
            self.mem.teach(" ".join(subj), pred if s == "P" else f"{pred}_ala")
            return None
        if s != "Q" or pred not in self.preds:
            return ["?"]
        side, conf = self.mem.ask(" ".join(subj), self.preds.index(pred))
        plan = UNKNOWN if side is None or conf < self.mem_thr else subj + [pred] + (["ala"] if side else [])
        return self.speak(plan)

    def speak(self, cues, max_len=8):
        state, k, out = "S0", 0, []
        while len(out) < max_len and state != "X":
            w = self.sp.say(state, cues[k] if k < len(cues) else NONE)
            if w is None or w == EOS:
                break
            out.append(w)
            c = self.cls(w)
            state = "X" if c is None else self.gram.step(state, c)
            k += k < len(cues) and cues[k] == w
        return out


def _djob(args):
    kind, cfg, seed, controls = args
    t0, rng = time.time(), np.random.default_rng([seed, 8, 13])
    ch = Chat(kind, cfg, seed, controls)
    dc = cfg["chat8d"]
    facts = {}
    for subj in ch.subjects:
        for i in rng.choice(len(ch.preds), dc["preds_per_subject"], replace=False):
            facts[(subj, int(i))] = int(rng.integers(2))
    keys = list(facts)
    statements = [list(sj) + ["li", ch.preds[i]] + (["ala"] if facts[(sj, i)] else []) for sj, i in keys]
    taught_ok = [ch.hear(statements[j]) is None for j in rng.permutation(len(keys))]
    qs = [(sj, i, list(sj) + ["li", ch.preds[i]] + (["ala"] if facts[(sj, i)] else [])) for sj, i in keys]
    qs += [(sj, i, UNKNOWN) for sj in ch.unknown for i in range(len(ch.preds))]
    replies = []
    for j in rng.permutation(len(qs)):
        sj, i, want = qs[j]
        got = ch.hear(list(sj) + ["li", ch.preds[i], "anu", "seme"])
        replies.append((sj in ch.unknown, got == want, ch.read(got)[0] in ACCEPT_D if got else False))
    out = {"seed": seed, "controls": list(controls), "statements_read": float(np.mean(taught_ok)),
           "reply_exact": float(np.mean([e for _, e, _ in replies])),
           "taught_exact": float(np.mean([e for u, e, _ in replies if not u])),
           "unknown_exact": float(np.mean([e for u, e, _ in replies if u])),
           "grammatical": float(np.mean([g for _, _, g in replies]))}
    log.info("%s seed %d %s (%.0fs): %s", kind, seed, controls, time.time() - t0,
             {k: round(v, 3) for k, v in out.items() if isinstance(v, float)})
    return out


def exp_chat8d(cfg: dict, seed: int) -> dict:
    """8d: the whole exchange on the test seeds, with one control per fly role."""
    from chatfly.zumi3 import merged

    cfg, out = merged(cfg), []
    dc = cfg["chat8d"]
    runs = [()] + [(c,) for c in dc["controls"]]
    for kind in cfg["simulators"]:
        with ProcessPoolExecutor(cfg.get("workers", 1)) as ex:
            res = list(ex.map(_djob, [(kind, cfg, s, r) for r in runs for s in cfg["seeds"]]))
        rows = []
        for r in runs:
            rs = [x for x in res if tuple(x["controls"]) == r]
            rows.append({"controls": list(r), **{k: [float(np.mean([x[k] for x in rs])), float(np.min([x[k] for x in rs]))]
                         for k in ("statements_read", "reply_exact", "taught_exact", "unknown_exact", "grammatical")}})
            log.info("%s summary %s", kind, rows[-1])
        out.append({"simulator": kind, "rows": rows, "runs": res})
    return {"chat8d": out}
