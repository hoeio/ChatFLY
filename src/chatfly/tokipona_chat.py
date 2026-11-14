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


def memory_odor(score, polar, split=False):
    """8g, the fixed interface from a memory fly to the speaker: four odor channels from the
    memory's output scores, top and second over top: yes, no (polar memories: the winning side's
    sureness), sure (object memory), doubt. No threshold; the speaker's KCs read the mixture.
    split (8g-2): the object memory's doubt has its own fifth channel."""
    o = np.sort(score)[::-1]
    if o[0] <= 0:
        return np.zeros(4)
    sure, doubt = (o[0] - o[1]) / o[0], o[1] / o[0]
    side = int(np.argmax(score))
    if polar:
        return np.array([sure * (side == 0), sure * (side == 1), 0.0, doubt])
    return np.array([0.0, 0.0, sure, 0.0, doubt]) if split else np.array([0.0, 0.0, sure, doubt])


def reply_odor_table(heads, mods, preds, opreds, objects, obj_glue=False):
    """8g: (state, cue, word, odor class) for every reply. The odors come from a practice session
    (Chat.practice). Cues: the subject, the predicate and the object word.
    obj_glue (8g-2): object replies keep the glue plan (threshold; "mi sona ala" as cues) and carry
    the object memory's odor (class "obj": known and unknown alike) only so that no reply is odor-free."""
    yes, no, sure, unk = "yes", "no", "obj" if obj_glue else "sure", "unk"
    t = []
    for r, ps in ((yes, preds), (no, preds), (sure, opreds)):
        t += [("S0", h, h, r) for h in heads] + [("S1", m, m, r) for m in mods]
        t += [("S2", p, "li", r) for p in ps] + [("LI", p, p, r) for p in ps]
    t += [("P", NONE, EOS, yes), ("P", NONE, "ala", no), ("PN", NONE, EOS, no)]
    t += [("P", o, "e", sure) for o in objects] + [("E", o, o, sure) for o in objects] + [("O", NONE, EOS, sure)]
    for h in heads:  # unknown: "mi sona ala"; the cue pointer stays on the head
        t += [("S0", h, "mi", unk), ("MI", h, "sona", unk), ("P", h, "ala", unk), ("PN", h, EOS, unk)]
    if obj_glue:
        t += [("S0", "mi", "mi", sure), ("MI", "sona", "sona", sure), ("P", "ala", "ala", sure), ("PN", NONE, EOS, sure)]
    return t


def grammar_fly(kind, cfg, seed, task, gf):
    """8b's grammar fly on a task; gf: n_kc, kc_theta, threshold, optional source (table | table_bal)."""
    from chatfly.fsm import transitions
    from chatfly.grammar import GrammarFly

    c = copy.deepcopy(cfg)
    c["mb"]["n_kc"], c["rate"]["kc_theta"] = gf["n_kc"], gf["kc_theta"]
    seq, _ = teaching(task, gf.get("source", "table"), [], c["protocol"]["n_train"], np.random.default_rng([seed, 8, 7]))
    g = GrammarFly(kind, c, seed, "single", "true", "pair", task=task, teach=seq)
    g.thr = gf["threshold"]
    valid = [(s, x, t) for s, x, t in transitions(task) if t != "X"]
    for _ in range(gf.get("rehearse", 0)):  # 8f: re-teach the valid transitions the fly gets wrong
        for s, x, t in [v for v in valid if g.step(*v[:2]) != v[2]]:
            g.flies.teach(g.stim(s, x), x, task["states"].index(t))
        if g.cache is not None:
            g.cache.clear()
    return g


def _gsel(args):
    kind, cfg, seed, gf = args
    return grammar_fly(kind, cfg, seed, grammar_e(), gf).step_acc()


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

        g = grammar_e() if self.objects_on else grammar_d()
        nr = {"rehearse": 0} if "no_rehearse" in controls else {}  # 8f ablation
        self.gram = grammar_fly(kind, cfg, seed, g, {**dc["grammar_fly"], **nr})
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
            def make_omem():
                return ObjectMemory(kind, cfg, seed, self.vocab, self.opreds, self.objects, om,
                                    "no_kc" if "memory_no_kc" in controls else None)
            self.omem = make_omem()
        mc = dc["memory_fly"]
        c = copy.deepcopy(cfg)
        c["mb"]["n_kc"], c["rate"]["kc_theta"] = mc["n_kc"], mc["kc_theta"]
        c["lang"] = {**c["lang"], "memory": {**c["lang"]["memory"], "n_teach": mc.get("n_teach", c["lang"]["memory"]["n_teach"])}}
        self.routed = mc.get("routing", "pair") == "pair"  # "none" (8d-2): subject + predicate smelled together
        cm = c
        def make_mem():
            return phrase_memory(kind, cm, seed, self.preds if self.routed else ["fact"],
                                 "no_kc" if "memory_no_kc" in controls else None)
        self.mem = make_mem()
        self.mem_thr = mc["threshold"]
        sc = {**dc["speaker"], **nr}
        c = copy.deepcopy(cfg)
        c["capacity"] = dict(sc)
        self.sp = Speaker(kind, c, seed, content + ["ala"], sc["coding"], sc["n_kc"],
                          "no_kc" if "speaker_no_kc" in controls else None, states=g["states"])
        tab = speaker_table(content)
        if self.objects_on:
            tab += [("P", o, "e") for o in self.objects] + [("E", o, o) for o in self.objects] + [("O", NONE, EOS)]
        reps = repeats(tab, sc["balance"])
        tab = [(s, cue, t, None) for s, cue, t in tab]
        self.odor = "mem_odor" in sc  # 8g: the memories' outputs reach the speaker as an odor
        self.odor_off = "odor_off" in controls
        if self.odor:  # every reply carries an odor, so the odor-free table is not taught
            self.obj_glue = sc["mem_odor"].get("objects") == "glue"
            tab = reply_odor_table(heads, mods[:nm], self.preds, self.opreds, self.objects, self.obj_glue)
            reps = [1] * len(tab)
            self.pool = self.practice(make_mem(), make_omem(), sc["mem_odor"]["n_practice"], np.random.default_rng([seed, 8, 18]))
        idx = {w: i for i, w in enumerate(self.sp.outs)}
        order = np.repeat(np.arange(len(tab)), reps)
        def odor(r, sample):  # 8g: a practice odor of class r (one drawn per presentation; the mean to check)
            if r is None or not isinstance(r, str):
                return r
            p = self.pool[r]
            return p[self.sp.rng.integers(len(p))] if sample else np.mean(p, axis=0)
        def teach(j):
            s, cue, t, r = tab[j]
            r = odor(r, True)
            dan = np.ones(self.sp.n)
            dan[self.sp.code[idx[t]]] = 0.0
            self.sp.sim.present(self.sp.stim(s, cue, r), self.sp.dur, dan)
        for _ in range(c["protocol"]["n_train"]):
            for j in self.sp.rng.permutation(order):
                teach(j)
        for _ in range(sc.get("rehearse", 0)):  # 8f: the teacher re-teaches the contexts the fly says wrong
            wrong = [j for j, (s, cue, t, r) in enumerate(tab) if self.sp.say(s, cue, odor(r, False)) != t]
            log.info("seed %d speaker rehearsal: %d of %d contexts wrong", seed, len(wrong), len(tab))
            for j in self.sp.rng.permutation(wrong):
                teach(j)
        self.role = self.pointer = None
        if "role_fly" in dc:  # 8h: the role of each word read off the reached state by a fly
            self.role = TokenFly(kind, cfg, seed, g["states"], ROLES, [((st,), role_of(st)) for st in g["states"]],
                                 dc["role_fly"], 1, "no_kc" if "role_no_kc" in controls else None)
        if "pointer_fly" in dc:  # 8h: whether the word just said uses up the cue (cue and word smelled together)
            pairs = sorted({(cue, t) for _, cue, t, _ in tab if cue != NONE and t != EOS})
            # cue and said word on separate PN patterns: a same-word union would be a PN subset of every
            # different pair holding that word, and the subset context loses (as in 8g)
            words = [w for w in self.sp.outs if w != EOS]  # every word the speaker can say
            toks = [f"c:{w}" for w in words] + [f"s:{w}" for w in words]
            self.pointer = TokenFly(kind, cfg, seed, toks, ("next", "stay"),
                                    [((f"c:{a}", f"s:{b}"), "next" if a == b else "stay") for a, b in pairs],
                                    dc["pointer_fly"], 2, "no_kc" if "pointer_no_kc" in controls else None)

    def read(self, words):
        """Run the class and grammar flies; roles from the states. Returns (final state,
        subject words, predicate word, object word)."""
        s, subj, pred, obj = "S0", [], None, None
        for w in words:
            k = self.cls(w)
            nxt = "X" if k is None else self.gram.step(s, k)
            role = role_of(nxt) if self.role is None else self.role.decide((nxt,))
            if role == "subj":
                subj.append(w)
            elif role == "pred":
                pred = w
            elif role == "obj":
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
            if self.odor and self.obj_glue:  # 8g-2: the glue plan, with the object memory's odor
                got, sc = self.omem.ask(subj, pred), self.omem.scores(subj, pred)
                od = np.zeros(5) if self.odor_off else memory_odor(sc, False, True)
                return self.speak(UNKNOWN if got is None else subj + [pred, got], od)
            if self.odor:  # 8g: the object word is the cue; knowing or not is left to the speaker
                sc = self.omem.scores(subj, pred)
                od = np.zeros(4) if self.odor_off else memory_odor(sc, False)
                return self.speak(subj + [pred, self.objects[int(np.argmax(sc))]], od)
            got = self.omem.ask(subj, pred)
            return self.speak(UNKNOWN if got is None else subj + [pred, got])
        if s in ("P", "PN") and pred in self.preds:
            key, prop = (" ".join(subj), pred) if self.routed else (" ".join(subj + [pred]), "fact")
            self.mem.teach(key, prop if s == "P" else f"{prop}_ala")
            return None
        if s != "Q" or pred not in self.preds:
            return ["?"]
        if self.odor:  # 8g: no confidence threshold, no reply branch; the speaker smells the memory
            return self.speak(subj + [pred], np.zeros(4) if self.odor_off else memory_odor(self.mem_rates(subj, pred), True))
        side, conf = (self.mem.ask(" ".join(subj), self.preds.index(pred)) if self.routed
                      else self.mem.ask(" ".join(subj + [pred]), 0))
        plan = UNKNOWN if side is None or conf < self.mem_thr else subj + [pred] + (["ala"] if side else [])
        return self.speak(plan)

    def practice(self, mem, omem, n, rng):
        """8g: before the conversation the teacher tells throwaway copies of the two memories n facts
        each (drawn as in a conversation) and keeps the odors they then give: per class yes, no,
        sure (object known), unknown. The speaker is taught with these."""
        own, self.mem, keep = self.mem, mem, self.omem
        self.omem, pool = omem, {"yes": [], "no": [], "sure": [], "unk": [], "obj": []}
        yn, ob = {}, {}  # facts drawn with replacement, so a later fact can change an earlier one
        for _ in range(n):
            sj = self.subjects[rng.integers(len(self.subjects))]
            p, v = self.preds[rng.integers(len(self.preds))], int(rng.integers(2))
            yn[(sj, p)] = v
            mem.teach(" ".join(sj), p if v == 0 else f"{p}_ala")
            p, o = self.opreds[rng.integers(len(self.opreds))], self.objects[rng.integers(len(self.objects))]
            ob[(sj, p)] = o
            omem.teach(list(sj), p, o)
        for (sj, p), v in yn.items():
            pool["no" if v else "yes"].append(memory_odor(self.mem_rates(sj, p), True))
        split = self.obj_glue
        for (sj, p), o in ob.items():
            pool["obj" if split else "sure"].append(memory_odor(omem.scores(list(sj), p), False, split))
        for s in self.unknown:
            pool["unk"].append(memory_odor(self.mem_rates(s, self.preds[rng.integers(len(self.preds))]), True))
            pool["obj" if split else "unk"].append(
                memory_odor(omem.scores(list(s), self.opreds[rng.integers(len(self.opreds))]), False, split))
        self.mem, self.omem = own, keep
        return {k: np.array(v) for k, v in pool.items() if v}

    def mem_rates(self, subj, pred):
        """The yes/no memory's two output rates (mean per compartment, n_ask sniffs)."""
        sl = self.mem.slots[self.preds.index(pred)]
        mb = sum(sl.present(self.mem.pattern(" ".join(subj)), self.mem.dur)["mbon"] for _ in range(self.mem.n_ask))
        return np.array([mb[sl.mbon_comp == c].mean() for c in range(sl.n_comp)])

    def speak(self, cues, odor=None, max_len=8):
        state, k, out = "S0", 0, []
        while len(out) < max_len and state != "X":
            w = self.sp.say(state, cues[k] if k < len(cues) else NONE, odor)
            if w is None or w == EOS:
                break
            out.append(w)
            c = self.cls(w)
            state = "X" if c is None else self.gram.step(state, c)
            if k < len(cues):
                k += (cues[k] == w) if self.pointer is None else (self.pointer.decide((f"c:{cues[k]}", f"s:{w}")) == "next")
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
    gsel = None
    if "grammar_select" in dc:  # 8e-3: the grammar fly's setting first: most select seeds at step 1.0, then mean
        gs = dc["grammar_select"]
        with ProcessPoolExecutor(cfg.get("workers", 1)) as ex:
            acc = list(ex.map(_gsel, [(cfg["simulators"][0], cfg, s, {**dc["grammar_fly"], **c})
                                      for c in gs["conds"] for s in gs["seeds"]]))
        n = len(gs["seeds"])
        per = [acc[i * n:(i + 1) * n] for i in range(len(gs["conds"]))]
        best = max(range(len(per)), key=lambda i: (sum(a == 1.0 for a in per[i]), round(float(np.mean(per[i])), 4), -i))
        gsel = {"conds": gs["conds"], "step_acc": per, "chosen": gs["conds"][best]}
        log.info("grammar select %s -> %s", [(sum(a == 1.0 for a in p), round(float(np.mean(p)), 4)) for p in per], gsel["chosen"])
        dc = {**dc, "grammar_fly": {**dc["grammar_fly"], **gsel["chosen"]}}
        cfg = {**cfg, "chat8d": dc}
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
        out.append({"simulator": kind, "chosen_memory": chosen, "grammar_select": gsel, "rows": rows, "runs": res})
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
        self.objects, self.routed = list(objects), om.get("routing", "pair") == "pair"
        self.preds = list(preds) if self.routed else ["any"]  # "none" (8e-2): subject + predicate smelled together
        self.code = [list(sets[i]) for i in np.random.default_rng([seed, 8, 14]).permutation(len(sets))[:len(objects)]]
        c["mb"]["compartments"], c["tokens"]["names"] = [str(i) for i in range(n)], [f"w:{w}" for w in vocab]
        c["mb"]["n_kc"], c["rate"]["kc_theta"] = om["n_kc"], om["kc_theta"]
        self.pats = token_patterns(c, np.random.default_rng([seed, 8, 15]))
        sim = SIMULATORS[kind](c, seed + 5 * rs["fly"], noise_seed=seed * rs["noise"] + 11)
        self.slots = ([PNDirect(sim, c, n, np.random.default_rng([seed, 8, 16, i]), kind == "spiking") for i in range(len(self.preds))]
                      if control == "no_kc" else [WeightSlot(sim) for _ in self.preds])
        self.n, self.dur, self.n_teach, self.thr = n, c["protocol"]["duration_ms"], om["n_teach"], om["threshold"]

    def smell(self, words):
        return np.max([self.pats["w:" + w] for w in words], axis=0)

    def teach(self, subj, pred, obj):
        dan = np.ones(self.n)
        dan[self.code[self.objects.index(obj)]] = 0.0
        s, key = self.slot(subj, pred)
        for _ in range(self.n_teach):
            s.present(self.smell(key), self.dur, dan)

    def slot(self, subj, pred):
        return (self.slots[self.preds.index(pred)], subj) if self.routed else (self.slots[0], list(subj) + [pred])

    def scores(self, subj, pred):
        s, key = self.slot(subj, pred)
        r = group_rates(s, s.present(self.smell(key), self.dur)["mbon"])
        return np.array([r[g].mean() for g in self.code])

    def ask(self, subj, pred):
        score = self.scores(subj, pred)
        o = np.sort(score)[::-1]
        if o[0] <= 0 or (o[0] - o[1]) / o[0] < self.thr:
            return None
        return self.objects[int(np.argmax(score))]


# ---- 8h: role assignment and the cue pointer as flies ----
ROLES = ("subj", "pred", "obj", "none")


def role_of(state):
    """8h teacher labels: the role of the word that led to this state (the old read() rule)."""
    return "subj" if state in ("S1", "S2", "S3", "MI") else {"P": "pred", "O": "obj"}.get(state, "none")


class TokenFly:
    """8h: smells the union (PN max) of one or more tokens and says a label (one MBON group per
    label), taught from a table of (tokens, label), every entry once per epoch, then rehearsal."""

    def __init__(self, kind, cfg, seed, tokens, labels, table, tf, salt, control=None):
        from chatfly.conditioning import token_patterns
        from chatfly.fsm import PNDirect
        from chatfly.parity import dan_for
        from chatfly.sim import SIMULATORS

        rs, c = cfg["rng_streams"], copy.deepcopy(cfg)
        self.labels = list(labels)
        c["mb"]["compartments"], c["tokens"]["names"] = self.labels, [f"t:{t}" for t in tokens]
        c["mb"]["n_kc"], c["rate"]["kc_theta"] = tf["n_kc"], tf["kc_theta"]
        self.pats = token_patterns(c, np.random.default_rng([seed, 8, 20, salt]))
        self.sim = SIMULATORS[kind](c, seed + (7 + salt) * rs["fly"], noise_seed=seed * rs["noise"] + 20 + salt)
        if control == "no_kc":
            self.sim = PNDirect(self.sim, c, len(self.labels), np.random.default_rng([seed, 8, 21, salt]), kind == "spiking")
        rng, self.dur = np.random.default_rng([seed, 8, 22, salt]), c["protocol"]["duration_ms"]
        self.cache = {} if kind == "rate" else None  # the rate model is deterministic
        def teach(i):
            toks, lab = table[i]
            self.sim.present(self.stim(toks), self.dur, dan_for(self.labels.index(lab), len(self.labels)))
        for _ in range(c["protocol"]["n_train"]):
            for i in rng.permutation(len(table)):
                teach(i)
        for _ in range(tf.get("rehearse", 0)):
            wrong = [i for i, (toks, lab) in enumerate(table) if self.decide(toks) != lab]
            self.cache = {} if self.cache is not None else None
            for i in rng.permutation(wrong):
                teach(i)

    def stim(self, toks):
        return np.max([self.pats[f"t:{t}"] for t in toks], axis=0)

    def decide(self, toks):
        from chatfly.fsm import argmax_or_none

        toks = tuple(toks)
        if self.cache is not None and toks in self.cache:
            return self.cache[toks]
        r = group_rates(self.sim, self.sim.present(self.stim(toks), self.dur)["mbon"])
        i = argmax_or_none(r)
        out = None if i is None else self.labels[i]
        if self.cache is not None:
            self.cache[toks] = out
        return out
