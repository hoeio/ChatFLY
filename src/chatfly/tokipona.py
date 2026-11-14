"""Phase 8a (ADR 0013): can the speaking fly hold a Toki Pona-sized output vocabulary?

The speaker smells (sentence state, next thing to say) and says one word, as the 7f speaker
does with (grammar state, cue). Sentences are "s li p [e o]", or "mi/sina p [e o]" without li.
Every content word must be said in three positions, so the table has about 6 contexts per word.

Output codings:
- "onehot": one MBON group per word (Phases 7a-7f), so the vocabulary needs as many groups.
- "k{k}n{n}": each word is a fixed random set of k of n groups (H1). Teaching depresses every
  group outside the target's set. Reading takes the word whose groups have the highest mean rate.

Interface rules: the true sentence state (8b replaces it with a grammar fly), the cue pointer
(moves on when the cue word is said), and stopping at EOS. Word list: the 120 nimi pu from
https://sona.pona.la/wiki/nimi_pu (CC BY-SA 4.0), recorded in docs/04-research-notes.md.
"""

import copy
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

from chatfly.conditioning import token_patterns
from chatfly.fsm import PNDirect, group_rates
from chatfly.sim import SIMULATORS

log = logging.getLogger("chatfly")
PU_TEXT = ("a akesi ala alasa ale anpa ante anu awen e en esun ijo ike ilo insa jaki jan jelo jo kala "
      "kalama kama kasi ken kepeken kili kiwen ko kon kule kulupu kute la lape laso lawa len lete li "
      "lili linja lipu loje lon luka lukin lupa ma mama mani meli mi mije moku moli monsi mu mun musi "
      "mute nanpa nasa nasin nena ni nimi noka o olin ona open pakala pali palisa pan pana pi pilin "
      "pimeja pini pipi poka poki pona pu sama seli selo seme sewi sijelo sike sin sina sinpin sitelen "
      "sona soweli suli suno supa suwi tan taso tawa telo tenpo toki tomo tu unpa uta utala walo wan "
      "waso wawa weka wile")  # as the source prints it
PU = PU_TEXT.split()
PARTICLES = {"a", "anu", "e", "en", "la", "li", "o", "pi"}
CONTENT = [w for w in PU if w not in PARTICLES]
BARE = ("mi", "sina")  # subjects that take no li
EOS, NONE = "#", "-"
STATES = ("S0", "SUBJ", "MI", "LI", "PRED", "E", "OBJ")


def words_for(v, seed):
    """mi, sina and v - 2 other content words, a fixed random subset per seed."""
    rest = [w for w in CONTENT if w not in BARE]
    pick = np.random.default_rng([seed, 8, 2]).permutation(len(rest))[:v - 2]
    return list(BARE) + [rest[i] for i in sorted(pick)]


def table(words):
    """Every (state, cue) -> word the speaker must know."""
    t = [(s, w, w) for w in words for s in ("S0", "MI", "LI", "E")]
    t += [(s, w, x) for w in words for s, x in (("SUBJ", "li"), ("PRED", "e"))]
    return t + [("PRED", NONE, EOS), ("OBJ", NONE, EOS)]


def after(state, w):
    """Sentence state after the speaker says w (the true grammar; X = not a sentence)."""
    nxt = {"S0": "MI" if w in BARE else "SUBJ", "SUBJ": "LI" if w == "li" else "X",
           "MI": "PRED", "LI": "PRED", "PRED": "E" if w == "e" else "X", "E": "OBJ"}
    return "X" if w in ("li", "e") and state not in ("SUBJ", "PRED") else nxt.get(state, "X")


def sentence(msg):
    s, p, o = msg
    return [s] + ([] if s in BARE else ["li"]) + [p] + (["e", o] if o else [])


def messages(words, n, seed):
    rng, others = np.random.default_rng([seed, 8, 3]), [w for w in words if w not in BARE]
    return [(str(rng.choice(words)), str(rng.choice(others)),
             str(rng.choice(others)) if rng.random() < 0.5 else None) for _ in range(n)]


def repeats(tab, by):
    """8a-3: times each context is taught per epoch. For each token of the context (the state, or
    with "token" also the cue), the most common target among contexts with that token, divided by
    the count of this context's target there; the context takes its largest ratio (1 if by is "none")."""
    if by == "none":
        return [1] * len(tab)
    keys = (lambda s, c: [("s", s)]) if by == "state" else (lambda s, c: [("s", s), ("c", c)])
    count = {}
    for s, c, t in tab:
        for k in keys(s, c):
            count.setdefault(k, {}).setdefault(t, 0)
            count[k][t] += 1
    return [max(round(max(count[k].values()) / count[k][t]) for k in keys(s, c)) for s, c, t in tab]


class Speaker:
    def __init__(self, kind, cfg, seed, words, coding, n_kc, control=None):
        rs, sp = cfg["rng_streams"], cfg["capacity"]
        self.outs = list(words) + ["li", "e", EOS]
        if coding == "onehot":
            n, self.code = len(self.outs), [[i] for i in range(len(self.outs))]
        else:
            k, n = (int(x) for x in coding[1:].split("n"))
            sets = list(combinations(range(n), k))
            if len(sets) < len(self.outs):
                raise ValueError(f"{coding} has {len(sets)} codes for {len(self.outs)} words")
            pick = np.random.default_rng([seed, 8, 1]).permutation(len(sets))[:len(self.outs)]
            self.code = [list(sets[i]) for i in pick]
        c = copy.deepcopy(cfg)
        c["mb"]["compartments"], c["mb"]["n_kc"] = [str(i) for i in range(n)], n_kc
        c["tokens"]["names"] = [f"s:{s}" for s in STATES] + [f"c:{w}" for w in list(words) + [NONE]]
        c["tokens"]["pn_active_frac"], c["rate"]["kc_theta"] = sp["pn_active_frac"], sp["kc_theta"]
        c["mb"]["n_pn"] = sp.get("n_pn", c["mb"]["n_pn"])
        self.pats = token_patterns(c, np.random.default_rng(seed + rs["tokens"]))
        self.sim = SIMULATORS[kind](c, seed, noise_seed=seed * rs["noise"] + 8)
        if control == "no_kc":
            self.sim = PNDirect(self.sim, c, n, np.random.default_rng([seed, 8, 5]), kind == "spiking")
        self.n, self.control, self.dur = n, control, c["protocol"]["duration_ms"]
        self.rng, self.balance = np.random.default_rng([seed, 8, 4]), sp.get("balance", "none")

    def stim(self, s, c):
        return np.maximum(self.pats["s:" + s], self.pats["c:" + c])

    def train(self, tab, epochs):
        idx = {w: i for i, w in enumerate(self.outs)}
        order = np.repeat(np.arange(len(tab)), repeats(tab, self.balance))
        for _ in range(epochs):
            for j in self.rng.permutation(order):
                s, c, t = tab[j]
                i = int(self.rng.integers(len(self.outs))) if self.control == "shuffled" else idx[t]
                dan = np.ones(self.n)
                dan[self.code[i]] = 0.0
                self.sim.present(self.stim(s, c), self.dur, dan)

    def say(self, s, c):
        r = group_rates(self.sim, self.sim.present(self.stim(s, c), self.dur)["mbon"])
        score = np.array([r[g].mean() for g in self.code])
        top = np.sort(score)[::-1]
        return None if r.sum() <= 0 or top[0] == top[1] else self.outs[int(np.argmax(score))]

    def speak(self, msg, max_len=6):
        state, cues, k, out = "S0", [w for w in msg if w], 0, []
        while len(out) < max_len and state != "X":
            w = self.say(state, cues[k] if k < len(cues) else NONE)
            if w is None or w == EOS:
                break
            out.append(w)
            state = after(state, w)
            k += k < len(cues) and cues[k] == w
        return out


def _job(args):
    kind, cfg, seed, v, coding, cond, control = args
    n_kc, theta = cond["n_kc"], cond["kc_theta"]
    cfg = {**cfg, "capacity": {**cfg["capacity"], **cond}}
    t0, cc = time.time(), cfg["capacity"]
    words = words_for(v, seed)
    sp = Speaker(kind, cfg, seed, words, coding, n_kc, control)
    tab = table(words)
    sp.train(tab, cfg["protocol"]["n_train"])
    step = float(np.mean([sp.say(s, c) == t for s, c, t in tab]))
    msgs = messages(words, cc["n_messages"], seed)
    exact = float(np.mean([sp.speak(m) == sentence(m) for m in msgs]))
    log.info("%s seed %d V %d %s %s %s: step %.3f sentence %.3f groups %d (%.0fs)", kind, seed,
             v, coding, cond, control, step, exact, sp.n, time.time() - t0)
    return {"seed": seed, "V": v, "coding": coding, "cond": cond, "n_kc": n_kc, "kc_theta": theta, "control": control,
            "groups": sp.n, "step_acc": step, "sentence_exact": exact}


def exp_capacity(cfg: dict, seed: int) -> dict:
    """8a: every (V, coding, (KC count, KC threshold)) on the config's seeds; controls at the largest V."""
    from chatfly.zumi3 import merged

    cfg, cc = merged(cfg), cfg["capacity"]
    conds = cc.get("conds") or [{"n_kc": n, "kc_theta": t} for n, t in cc["kc"]]
    out = []
    for kind in cfg["simulators"]:
        jobs = [(kind, cfg, s, v, cd, nk, None) for s in cfg["seeds"] for v in cc["vocab_sizes"]
                for cd in cc["codings"] for nk in conds]
        jobs += [(kind, cfg, s, max(cc["vocab_sizes"]), cd, nk, ctl) for s in cfg["seeds"]
                 for cd in cc["codings"] for nk in conds for ctl in cc["controls"]]
        with ProcessPoolExecutor(cfg.get("workers", 1)) as ex:
            res = list(ex.map(_job, jobs))
        rows = []
        def key(r):
            return r["V"], r["coding"], tuple(sorted(r["cond"].items())), str(r["control"])

        for k in sorted({key(r) for r in res}):
            rs = [r for r in res if key(r) == k]
            row = {"V": k[0], "coding": k[1], "cond": dict(k[2]), "control": k[3], "groups": rs[0]["groups"],
                   **{m: [float(np.mean([r[m] for r in rs])), float(np.min([r[m] for r in rs]))]
                      for m in ("step_acc", "sentence_exact")}}
            rows.append(row)
            log.info("%s summary %s", kind, row)
        out.append({"simulator": kind, "rows": rows, "runs": res})
    return {"capacity": out}


# ---- 8b: the grammar fly (ADR 0013) ----
CLASSES = ("B", "W", "li", "e")  # B = mi/sina, W = any other content word (tokenizer glue)
ACCEPT = {"P1", "P2", "P3", "O1", "O2", "O3"}


def grammar_task() -> dict:
    """The 8b subset as an FSM over word classes: subject (mi/sina alone, or a word with up to
    two modifiers, then li), predicate with up to two modifiers, optional `e` + object with up
    to two modifiers. Si = subject with i-1 modifiers; Pi, Oi likewise; X dead (absorbing).
    Content words after the head are modifiers, mi/sina included ("tomo mi")."""
    st = ["S0", "S1", "S2", "S3", "MI", "LI", "P1", "P2", "P3", "E", "O1", "O2", "O3", "X"]
    d = {s: dict.fromkeys(CLASSES, "X") for s in st}
    d["S0"].update(B="MI", W="S1")
    for a, b in (("S1", "S2"), ("S2", "S3"), ("P1", "P2"), ("P2", "P3"), ("O1", "O2"), ("O2", "O3")):
        d[a].update(B=b, W=b)
    for s in ("S1", "S2", "S3"):
        d[s]["li"] = "LI"
    for s, nxt in (("MI", "P1"), ("LI", "P1"), ("E", "O1")):
        d[s].update(B=nxt, W=nxt)
    for s in ("P1", "P2", "P3"):
        d[s]["e"] = "E"
    return {"name": "tokipona_8b", "states": st, "inputs": list(CLASSES), "delta": d}


def classes(words):
    return [w if w in ("li", "e") else "B" if w in BARE else "W" for w in words]


def utterance(rng, words, gc):
    """A valid sentence of the subset (words)."""
    others = [w for w in words if w not in BARE]
    def phrase(head):
        return [head] + [str(rng.choice(words)) for _ in range(rng.choice(3, p=gc["p_mods"]))]
    out = [str(rng.choice(BARE))] if rng.random() < gc["p_bare"] else phrase(str(rng.choice(others))) + ["li"]
    out += phrase(str(rng.choice(words)))
    return out + (["e"] + phrase(str(rng.choice(words))) if rng.random() < gc["p_object"] else [])


def salad(rng, words):
    pool = list(words) + ["li", "e"]
    return [str(rng.choice(pool)) for _ in range(rng.integers(2, 9))]


def final(task, xs):
    s = "S0"
    for x in xs:
        s = task["delta"][s][x]
    return s


def grammar_data(seed, words, gc, split):
    """train: valid sentences; dev/test: valid sentences and word salads, half each."""
    rng = np.random.default_rng([seed, 8, 6, ("train", "dev", "test").index(split)])
    if split == "train":
        return [utterance(rng, words, gc) for _ in range(gc["n_corpus"])]
    n = gc["n_eval"] // 2
    return [utterance(rng, words, gc) for _ in range(n)] + [salad(rng, words) for _ in range(n)]


def teaching(task, source, corpus, epochs, rng):
    """The presentation sequence. table: every valid transition once per epoch; table_bal: the
    same with `repeats` (8a-3, state and input tokens); dedup: every transition seen in the corpus
    once per epoch; raw: as many draws per epoch as dedup has, from the corpus occurrences
    (frequency-weighted, same budget)."""
    occ = []
    for u in corpus:
        s = "S0"
        for x in classes(u):
            occ.append((s, x, task["delta"][s][x]))
            s = occ[-1][2]
    valid = [(s, x, t) for s in task["states"] for x, t in task["delta"][s].items() if t != "X"]
    if source in ("table", "table_bal"):
        base = valid
    else:
        base = sorted(set(occ), key=valid.index)
    reps = repeats(base, "token") if source == "table_bal" else [1] * len(base)
    unit = [b for b, r in zip(base, reps) for _ in range(r)]
    if source == "raw":
        return [occ[i] for _ in range(epochs) for i in rng.integers(len(occ), size=len(unit))], len(base)
    return [unit[i] for _ in range(epochs) for i in rng.permutation(len(unit))], len(base)


def _gjob(args):
    from chatfly.grammar import GrammarFly

    kind, cfg, seed, run, split, thr = args
    arch, teacher, stim, source, n_kc, theta = run
    cfg = copy.deepcopy(cfg)
    cfg["mb"]["n_kc"], cfg["rate"]["kc_theta"] = n_kc, theta
    t0, gc, task = time.time(), cfg["grammar8"], grammar_task()
    words = words_for(gc["vocab"], seed)
    seq, n_taught = teaching(task, source, grammar_data(seed, words, gc, "train"), cfg["protocol"]["n_train"],
                             np.random.default_rng([seed, 8, 7]))
    g = GrammarFly(kind, cfg, seed, arch, teacher, stim, task=task, teach=seq)
    data = [classes(u) for u in grammar_data(seed, words, gc, split)]
    truth = [final(task, u) for u in data]
    out = {"seed": seed, "run": list(run), "split": split, "taught": n_taught, "by_threshold": {}}
    for t in [thr] if thr is not None else gc["conf_grid"]:
        g.thr = t
        got = [g.run(u) for u in data]
        out["by_threshold"][str(t)] = {
            "step_acc": g.step_acc(), "utterance_acc": float(np.mean([a == b for a, b in zip(got, truth)])),
            "good_accepted": float(np.mean([f in ACCEPT for f, t_ in zip(got, truth) if t_ in ACCEPT])),
            "bad_rejected": float(np.mean([f not in ACCEPT for f, t_ in zip(got, truth) if t_ not in ACCEPT]))}
    if thr is not None:
        out.update(out["by_threshold"][str(thr)], threshold=thr)
    log.info("%s seed %d %s %s taught %d (%.0fs) %s", kind, seed, run, split, n_taught, time.time() - t0,
             {k: round(v["utterance_acc"], 3) for k, v in out["by_threshold"].items()})
    return out


def exp_grammar8(cfg: dict, seed: int) -> dict:
    """8b: thresholds per run and the main run chosen on select seeds (dev split), then the
    main run, its controls and the references on test seeds (test split)."""
    from chatfly.zumi3 import merged

    cfg, out = merged(cfg), []
    gc = cfg["grammar8"]
    for kind in cfg["simulators"]:
        runs = [tuple(r) for r in gc["candidates"] + gc["references"]]
        with ProcessPoolExecutor(cfg.get("workers", 1)) as ex:
            sel = list(ex.map(_gjob, [(kind, cfg, s, r, "dev", None) for r in runs for s in cfg["select_seeds"]]))
        thr, score = {}, {}
        for r in runs:
            rs = [x for x in sel if tuple(x["run"]) == r]
            m = {t: np.mean([x["by_threshold"][str(t)]["utterance_acc"] for x in rs]) for t in gc["conf_grid"]}
            thr[r] = max(gc["conf_grid"], key=lambda t: (m[t], -t))
            score[r] = float(m[thr[r]])
            log.info("%s select %s: threshold %.2f, mean utterance acc %.3f", kind, r, thr[r], score[r])
        cands = [tuple(r) for r in gc["candidates"]]
        main = max(cands, key=lambda r: (round(score[r], 3), -cands.index(r)))  # ties (0.001): listed order
        ctl = [("single_pn", "true", *main[2:]), (main[0], "shuffled", *main[2:])]
        for r in ctl:  # controls get their own threshold on the select seeds too
            rs = [_gjob((kind, cfg, s, r, "dev", None)) for s in cfg["select_seeds"]]
            m = {t: np.mean([x["by_threshold"][str(t)]["utterance_acc"] for x in rs]) for t in gc["conf_grid"]}
            thr[r] = max(gc["conf_grid"], key=lambda t: (m[t], -t))
        test_runs = [main] + ctl + [r for r in runs if r != main]
        with ProcessPoolExecutor(cfg.get("workers", 1)) as ex:
            test = list(ex.map(_gjob, [(kind, cfg, s, r, "test", thr[r]) for r in test_runs for s in cfg["seeds"]]))
        rows = []
        for r in test_runs:
            rs = [x for x in test if tuple(x["run"]) == r]
            row = {"run": list(r), "role": "main" if r == main else "control" if r in ctl else "reference",
                   "threshold": thr[r], "taught": [x["taught"] for x in rs],
                   **{k: [float(np.mean([x[k] for x in rs])), float(np.min([x[k] for x in rs]))]
                      for k in ("step_acc", "utterance_acc", "good_accepted", "bad_rejected")}}
            rows.append(row)
            log.info("%s summary %s", kind, row)
        out.append({"simulator": kind, "main": list(main), "select_scores": {str(k): v for k, v in score.items()},
                    "thresholds": {str(k): v for k, v in thr.items()}, "rows": rows, "select": sel, "test": test})
    return {"grammar8": out}
