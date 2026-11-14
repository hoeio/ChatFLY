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
from itertools import combinations

import numpy as np

from chatfly.tokipona import (
    EOS,
    NONE,
    ClassFly,
    Speaker,
    group_rates,
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
        self.objects_on = dc.get("objects", False)  # 8e
        self.vocab = words_for(112, seed) + ["li", "anu"] + (["e"] if self.objects_on else [])
        content = [w for w in self.vocab if w not in ("li", "anu", "ala", "seme", "e")]
        c = copy.deepcopy(cfg)
        cf = dc["class_fly"]
        c["mb"]["n_kc"], c["rate"]["kc_theta"] = cf["n_kc"], cf["kc_theta"]
        self.cls = ClassFly(kind, c, seed, None, "no_kc" if "class_no_kc" in controls else None,
                            cf["balance"], vocab=self.vocab,
                            cls=(CLASSES_D + ("e",), classes_e) if self.objects_on else (CLASSES_D, classes_d))
        from chatfly.grammar import GrammarFly

        g, gf = grammar_e() if self.objects_on else grammar_d(), dc["grammar_fly"]
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
        if self.objects_on:
            rest = pool[npr + nh + nm + 1:]
            self.opreds, self.objects = rest[:dc["n_object_preds"]], rest[dc["n_object_preds"]:][:dc["n_objects"]]
            om = dc["object_memory"]
            self.omem = ObjectMemory(kind, cfg, seed, self.vocab, self.opreds, self.objects, om,
                                     "no_kc" if "memory_no_kc" in controls else None)
        mc = dc["memory_fly"]
        c = copy.deepcopy(cfg)
        c["mb"]["n_kc"], c["rate"]["kc_theta"] = mc["n_kc"], mc["kc_theta"]
        self.routed = mc.get("routing", "pair") == "pair"  # "none" (8d-2): subject + predicate smelled together
        self.mem = phrase_memory(kind, c, seed, self.preds if self.routed else ["fact"],
                                 "no_kc" if "memory_no_kc" in controls else None)
        self.mem_thr = mc["threshold"]
        sc = dc["speaker"]
        c = copy.deepcopy(cfg)
        c["capacity"] = dict(sc)
        self.sp = Speaker(kind, c, seed, content + ["ala"], sc["coding"], sc["n_kc"],
                          "no_kc" if "speaker_no_kc" in controls else None, states=g["states"])
        tab = speaker_table(content)
        if self.objects_on:
            tab += [("P", o, "e") for o in self.objects] + [("E", o, o) for o in self.objects] + [("O", NONE, EOS)]
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
        subject words, predicate word, object word)."""
        s, subj, pred, obj = "S0", [], None, None
        for w in words:
            k = self.cls(w)
            nxt = "X" if k is None else self.gram.step(s, k)
            if nxt in ("S1", "S2", "S3", "MI"):
                subj.append(w)
            elif nxt == "P":
                pred = w
            elif nxt == "O":
                obj = w
            s = nxt
        return s, subj, pred, obj

    def hear(self, words):
        """A statement teaches the memory; a question gets a spoken reply (list of words)."""
        s, subj, pred, obj = self.read(words)
        if self.objects_on and s in ("O", "OQ") and pred in self.opreds:
            if s == "O":
                if obj in self.objects:
                    self.omem.teach(subj, pred, obj)
                    return None
                return ["?"]
            got = self.omem.ask(subj, pred)
            return self.speak(UNKNOWN if got is None else subj + [pred, got])
        if s in ("P", "PN") and pred in self.preds:
            key, prop = (" ".join(subj), pred) if self.routed else (" ".join(subj + [pred]), "fact")
            self.mem.teach(key, prop if s == "P" else f"{prop}_ala")
            return None
        if s != "Q" or pred not in self.preds:
            return ["?"]
        side, conf = (self.mem.ask(" ".join(subj), self.preds.index(pred)) if self.routed
                      else self.mem.ask(" ".join(subj + [pred]), 0))
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
        replies.append((sj in ch.unknown, got == want, got not in ([], ["?"]) and ch.read(got)[0] in ACCEPT_D))
    out = {"seed": seed, "controls": list(controls), "statements_read": float(np.mean(taught_ok)),
           "reply_exact": float(np.mean([e for _, e, _ in replies])),
           "taught_exact": float(np.mean([e for u, e, _ in replies if not u])),
           "unknown_exact": float(np.mean([e for u, e, _ in replies if u])),
           "grammatical": float(np.mean([g for _, _, g in replies]))}
    log.info("%s seed %d %s (%.0fs): %s", kind, seed, controls, time.time() - t0,
             {k: round(v, 3) for k, v in out.items() if isinstance(v, float)})
    return out


def _ejob(args):
    """8e: one conversation of n_turns: statements (yes/no and object facts, values may change),
    yes/no questions, "what" questions, and questions about never-mentioned subjects."""
    kind, cfg, seed, controls = args
    t0, rng, dc = time.time(), np.random.default_rng([seed, 8, 17]), cfg["chat8d"]
    ch = Chat(kind, cfg, seed, controls)
    yn, ob, changed, read_ok, qs = {}, {}, set(), [], []
    def sent(sj, p, tail):
        return list(sj) + ["li", p] + tail
    types = ("teach_yn", "teach_obj", "ask_yn", "ask_obj", "ask_unknown")
    for _ in range(dc["n_turns"]):
        t = types[rng.choice(len(types), p=dc["p_turn"])]
        if t == "teach_yn" or (t == "ask_yn" and not yn):
            sj, i, v = ch.subjects[rng.integers(len(ch.subjects))], int(rng.integers(len(ch.preds))), int(rng.integers(2))
            changed |= {("yn", sj, i)} if yn.get((sj, i), v) != v else set()
            yn[(sj, i)] = v
            read_ok.append(ch.hear(sent(sj, ch.preds[i], ["ala"] if v else [])) is None)
        elif t == "teach_obj" or (t == "ask_obj" and not ob):
            sj, j = ch.subjects[rng.integers(len(ch.subjects))], int(rng.integers(len(ch.opreds)))
            o = ch.objects[rng.integers(len(ch.objects))]
            changed |= {("ob", sj, j)} if ob.get((sj, j), o) != o else set()
            ob[(sj, j)] = o
            read_ok.append(ch.hear(sent(sj, ch.opreds[j], ["e", o])) is None)
        elif t == "ask_yn":
            (sj, i) = list(yn)[rng.integers(len(yn))]
            want = sent(sj, ch.preds[i], ["ala"] if yn[(sj, i)] else [])
            qs.append(("yn", ("yn", sj, i) in changed, ch.hear(sent(sj, ch.preds[i], ["anu", "seme"])) == want))
        elif t == "ask_obj":
            (sj, j) = list(ob)[rng.integers(len(ob))]
            want = sent(sj, ch.opreds[j], ["e", ob[(sj, j)]])
            qs.append(("obj", ("ob", sj, j) in changed, ch.hear(sent(sj, ch.opreds[j], ["e", "seme"])) == want))
        else:
            sj = ch.unknown[rng.integers(len(ch.unknown))]
            if rng.random() < 0.5:
                q = sent(sj, ch.preds[rng.integers(len(ch.preds))], ["anu", "seme"])
            else:
                q = sent(sj, ch.opreds[rng.integers(len(ch.opreds))], ["e", "seme"])
            qs.append(("unknown", False, ch.hear(q) == UNKNOWN))
    def frac(sel):
        xs = [ok for k, c, ok in qs if sel(k, c)]
        return float(np.mean(xs)) if xs else None
    out = {"seed": seed, "controls": list(controls), "statements_read": float(np.mean(read_ok)),
           "reply_exact": frac(lambda k, c: True), "yn_exact": frac(lambda k, c: k == "yn"),
           "obj_exact": frac(lambda k, c: k == "obj"), "unknown_exact": frac(lambda k, c: k == "unknown"),
           "changed_exact": frac(lambda k, c: c), "n_questions": len(qs)}
    log.info("%s seed %d %s (%.0fs): %s", kind, seed, controls, time.time() - t0,
             {k: round(v, 3) for k, v in out.items() if isinstance(v, float)})
    return out


def exp_chat8d(cfg: dict, seed: int) -> dict:
    """8d: the whole exchange on the test seeds, with one control per fly role."""
    from chatfly.zumi3 import merged

    cfg, out = merged(cfg), []
    dc = cfg["chat8d"]
    runs = [()] + [(c,) for c in dc["controls"]]
    chosen, job = None, _ejob if dc.get("objects") else _djob
    sk, conds = ("memory_fly", dc["memory_conds"]) if "memory_conds" in dc else (
        (dc["select"]["key"], dc["select"]["conds"]) if "select" in dc else (None, None))
    if conds:  # 8d-2 / 8e: one fly's setting chosen on select seeds (mean reply exact; ties: listed order)
        def with_set(c):
            return {**cfg, "chat8d": {**dc, sk: {**dc[sk], **c}}}
        with ProcessPoolExecutor(cfg.get("workers", 1)) as ex:
            sel = list(ex.map(job, [(cfg["simulators"][0], with_set(c), s, ()) for c in conds for s in cfg["select_seeds"]]))
        n = len(cfg["select_seeds"])
        score = [float(np.mean([r["reply_exact"] for r in sel[i * n:(i + 1) * n]])) for i in range(len(conds))]
        chosen = conds[max(range(len(score)), key=lambda i: (round(score[i], 3), -i))]
        log.info("select scores %s -> %s", score, chosen)
        cfg = with_set(chosen)
    for kind in cfg["simulators"]:
        with ProcessPoolExecutor(cfg.get("workers", 1)) as ex:
            res = list(ex.map(job, [(kind, cfg, s, r) for r in runs for s in cfg["seeds"]]))
        rows = []
        keys = [k for k, v in res[0].items() if isinstance(v, float)]
        for r in runs:
            rs = [x for x in res if tuple(x["controls"]) == r]
            rows.append({"controls": list(r), **{k: [float(np.mean([x[k] for x in rs if x[k] is not None])),
                                                     float(np.min([x[k] for x in rs if x[k] is not None]))]
                                                 for k in keys}})
            log.info("%s summary %s", kind, rows[-1])
        out.append({"simulator": kind, "chosen_memory": chosen, "rows": rows, "runs": res})
    return {"chat8d": out}


# ---- 8e: a whole conversation (objects, "what" questions, changes of mind) ----
ACCEPT_E = ACCEPT_D | {"O", "OQ"}


def grammar_e() -> dict:
    """8d's grammar plus an object: P + e -> E; E + word -> O (statement), E + seme -> OQ (question)."""
    g = grammar_d()
    st = g["states"][:-1] + ["E", "O", "OQ", "X"]
    d = {s: g["delta"].get(s, dict.fromkeys(CLASSES_D, "X")) for s in st}
    d["P"]["e"] = "E"
    d["E"].update(B="O", W="O", seme="OQ")
    return {"name": "tokipona_8e", "states": st, "inputs": list(CLASSES_D) + ["e"],
            "delta": {s: {**r, "e": r.get("e", "X")} for s, r in d.items()}}


def classes_e(words):
    return ["e" if w == "e" else c for w, c in zip(words, classes_d(words))]


class ObjectMemory:
    """(subject phrase, predicate) -> object word. The phrase smells like the union of its words;
    the predicate selects the KC->MBON weight set (routing glue, as the yes/no memory). The output
    is a k-of-n MBON code over the object words (8a's H1 code); teaching depresses every group
    outside the object's code, n_teach times. Asking reads the object with the highest mean rate
    over its groups; confidence = (top - second) / top."""

    def __init__(self, kind, cfg, seed, vocab, preds, objects, om, control=None):
        from chatfly.conditioning import token_patterns
        from chatfly.fsm import PNDirect, WeightSlot
        from chatfly.sim import SIMULATORS

        rs, c = cfg["rng_streams"], copy.deepcopy(cfg)
        k, n = (int(x) for x in om["coding"][1:].split("n"))
        sets = list(combinations(range(n), k))
        self.objects, self.preds = list(objects), list(preds)
        self.code = [list(sets[i]) for i in np.random.default_rng([seed, 8, 14]).permutation(len(sets))[:len(objects)]]
        c["mb"]["compartments"], c["tokens"]["names"] = [str(i) for i in range(n)], [f"w:{w}" for w in vocab]
        c["mb"]["n_kc"], c["rate"]["kc_theta"] = om["n_kc"], om["kc_theta"]
        self.pats = token_patterns(c, np.random.default_rng([seed, 8, 15]))
        sim = SIMULATORS[kind](c, seed + 5 * rs["fly"], noise_seed=seed * rs["noise"] + 11)
        self.slots = ([PNDirect(sim, c, n, np.random.default_rng([seed, 8, 16, i]), kind == "spiking") for i in preds]
                      if control == "no_kc" else [WeightSlot(sim) for _ in preds])
        self.n, self.dur, self.n_teach, self.thr = n, c["protocol"]["duration_ms"], om["n_teach"], om["threshold"]

    def smell(self, words):
        return np.max([self.pats["w:" + w] for w in words], axis=0)

    def teach(self, subj, pred, obj):
        dan = np.ones(self.n)
        dan[self.code[self.objects.index(obj)]] = 0.0
        s = self.slots[self.preds.index(pred)]
        for _ in range(self.n_teach):
            s.present(self.smell(subj), self.dur, dan)

    def ask(self, subj, pred):
        s = self.slots[self.preds.index(pred)]
        r = group_rates(s, s.present(self.smell(subj), self.dur)["mbon"])
        score = np.array([r[g].mean() for g in self.code])
        o = np.sort(score)[::-1]
        if o[0] <= 0 or (o[0] - o[1]) / o[0] < self.thr:
            return None
        return self.objects[int(np.argmax(score))]
