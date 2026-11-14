"""Phase 7c: the fly tracks the grammar state of the user's utterance (ADR 0012).

The Zumi utterance grammar (with strict property order, Zumi 3) is a finite-state machine
over words. `dfa` writes it down; it is used only for training targets and evaluation, and
`tests/test_grammar.py` checks it against `zumi.parse`. The fly learns every transition
(state, word) -> next state by the Phase 2 protocol and runs it in closed loop: its decoded
state is fed back with the next word. Architectures and controls are Phase 4's
(`fsm.Flies`): "routed_shared" = one fly whose KC->MBON weight set is selected by the current
word (a fixed routing rule: designer-set glue, as recorded in the revised Gate D verdict),
"single" = no routing, "routed_pn" = routing without a KC layer; plus a shuffled teacher.

The speaker (`GrammarSpeaker`) is 7b's split speaker with fly 0 smelling (grammar state,
last user word) instead of (second-last word, last word).
"""

import copy
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chatfly.conditioning import token_patterns
from chatfly.fsm import Flies, group_rates, present_pair, task_cfg, transitions
from chatfly.parity import dan_for
from chatfly.speak import EOS, NOUN, SEP, SpeakingFly, corpus, evaluate
from chatfly.zumi import props, reserved

log = logging.getLogger("chatfly")
ACCEPT = {"B", "M", "T0", "T1", "T2", "DONE"}


def dfa(lx) -> dict:
    """{"states", "inputs", "delta"} for one utterance, read word by word (nouns as N).
    S0 start · B after bzz · SI after si · M moves · SM moves after si · P place (needs ba) ·
    N noun · NP noun + property (needs ba) · KA noun ka · T0..T2 properties up to pair i ·
    DONE complete, nothing may follow · X dead (refused). Accepting: B M T0 T1 T2 DONE."""
    pr = props(lx)
    words = sorted(reserved(lx) - set(lx["replies"].values())) + [NOUN]
    states = ["S0", "B", "SI", "M", "SM", "P", "N", "NP", "KA", "T0", "T1", "T2", "DONE", "X"]
    d = {s: {w: "X" for w in words} for s in states}
    for s in ("S0", "B", "SI", "M", "SM"):
        sup = s in ("SI", "SM")
        for m in lx["moves"]:
            d[s][m] = "SM" if sup else "M"
        d[s][lx["where"]] = "DONE"
        for p in lx["places"]:
            d[s][p] = "P"
    d["S0"][lx["restart"]] = "B"
    d["S0"][lx["suppose"]] = d["B"][lx["suppose"]] = "SI"
    d["S0"][NOUN] = "N"
    if "pronoun" in lx:  # Zumi 3: `to` stands where a noun can
        d["S0"][lx["pronoun"]] = "N"
    d["P"][lx["yesno"]] = d["NP"][lx["yesno"]] = "DONE"
    d["N"][lx["is"]], d["N"][lx["what"]] = "KA", "DONE"
    last = len(lx["props"]) - 1
    for w, (i, _) in pr.items():
        d["N"][w] = "NP"
        nxt = "DONE" if i == last else f"T{i}"
        d["KA"][w] = nxt
        for k in range(last):
            if i > k:
                d[f"T{k}"][w] = nxt
    return {"name": "zumi_grammar", "states": states, "inputs": words, "delta": d}


def word_tok(w, task):
    return w if w in task["inputs"] else NOUN


def true_state(words, task) -> str:
    s = task["states"][0]
    for w in words:
        s = task["delta"][s][word_tok(w, task)]
    return s


class GrammarFly:
    """The grammar FSM learned by a fly (fsm.Flies) and run in closed loop.

    Selection-seed findings (ADR 0012):
    - Teaching every table entry, dead transitions included, fails: for most words most
      states lead to X, and KCs driven by the word alone settle on X.
    - Teaching only valid transitions and reading "no confident answer" as X (confidence <
      thr) works. The same idea as Zumi 2's memory: never taught -> no preference.
    stim "pair": present (state, word); "state": present the state token only. That makes
    sense only when the word already selects the readout (routed archs).
    The dead state X is absorbing (interface rule, like the ring's "lost")."""

    def __init__(self, kind, cfg, seed, arch, teacher="true", stim="state", task=None, teach=None):
        """task: the FSM (default: the Zumi grammar). teach: the whole presentation sequence of
        (state, input, next) (default: every valid transition per epoch, in random order)."""
        rs = cfg["rng_streams"]
        self.task = task or dfa(cfg["lang"])
        self.c = task_cfg(cfg, self.task)
        self.pats = token_patterns(self.c, np.random.default_rng(seed + rs["tokens"]))
        self.flies = Flies(kind, self.c, self.task, seed, arch, 1)
        self.stim_kind, self.thr, self.cache = stim, 0.0, {} if kind == "rate" else None
        rng, st = np.random.default_rng(seed + rs["task"]), self.task["states"]
        valid = [(s, x, t) for s, x, t in transitions(self.task) if t != "X"]
        if teach is None:
            teach = [valid[i] for _ in range(self.c["protocol"]["n_train"]) for i in rng.permutation(len(valid))]
        for s, x, t in teach:
            tgt = int(rng.integers(len(st))) if teacher == "shuffled" else st.index(t)
            self.flies.teach(self.stim(s, x), x, tgt)

    def stim(self, s, x):
        return self.pats[s] if self.stim_kind == "state" else present_pair(self.pats, s, x)

    def decide(self, s, x):
        """(state or None, confidence); cached for the deterministic rate model."""
        if self.cache is not None and (s, x) in self.cache:
            return self.cache[(s, x)]
        r = sum(group_rates(f, f.present(self.stim(s, x), self.flies.dur)["mbon"])
                for f in self.flies._which(x))
        o = np.sort(r)[::-1]
        out = (None, 0.0) if r.sum() <= 0 or o[0] == o[1] else (
            self.task["states"][int(np.argmax(r))], float((o[0] - o[1]) / r.sum()))
        if self.cache is not None:
            self.cache[(s, x)] = out
        return out

    def step(self, s, w):
        if s == "X":
            return "X"
        d, conf = self.decide(s, word_tok(w, self.task))
        return d if d is not None and conf >= self.thr else "X"

    def run(self, words):
        s = self.task["states"][0]
        for w in words:
            s = self.step(s, w)
        return s

    def step_acc(self):
        return float(np.mean([self.step(s, x) == t for s, x, t in transitions(self.task)]))


class GrammarSpeaker(SpeakingFly):
    """7b's split speaker; fly 0 smells (grammar state from the grammar fly, last user word).
    The grammar fly runs word by word as the user's words arrive (closed loop)."""

    def __init__(self, kind, cfg, seed, grammar, control=None):
        c = copy.deepcopy(cfg)
        c["speak"]["arch"] = "split"
        self.grammar = grammar
        super().__init__(kind, c, seed, control)
        extra = {f"g:{s}": None for s in grammar.task["states"]}
        cc = copy.deepcopy(self.flies[0].cfg)
        cc["tokens"]["names"] = list(extra)
        more = token_patterns(cc, np.random.default_rng(seed + cfg["rng_streams"]["tokens"] + 1))
        self.pats.update(more)

    def route(self, a, b, prev):
        if prev == SEP:  # here a is the grammar state, not a word
            return self.flies[0], np.maximum(self.pats["g:" + a], self.pats["b:" + b])
        return self.flies[1], np.maximum(self.pats["b:" + b], self.pats["p:" + prev])

    def train(self, dialogues, epochs):
        """Teacher forcing with the true grammar state."""
        n, task = len(self.vocab), self.grammar.task
        for _ in range(epochs):
            for d in dialogues:
                for user, reply, *_ in d:
                    g, b, prev = true_state(user, task), self.tok(user[-1]), SEP
                    for t in reply + [EOS]:
                        tgt = int(self.rng.integers(n)) if self.control == "shuffled_teacher" else self.index[t]
                        fly, stim = self.route(g, b, prev)
                        fly.present(stim, self.dur, dan_for(tgt, n))
                        prev = t

    def init(self):
        return (self.grammar.task["states"][0], "-", None)

    def step(self, st, t):
        g, _, prev = st
        if prev is None and t not in (SEP, EOS):
            return (self.grammar.step(g, t), self.tok(t), None), np.full(len(self.vocab), 1 / len(self.vocab))
        if t == EOS:
            return self.init(), np.full(len(self.vocab), 1 / len(self.vocab))
        return super().step(st, t)


def utterance_scores(g, dialogues) -> dict:
    rows = [(k, true_state(u, g.task), g.run(u)) for d in dialogues for u, _, k, *_ in d]
    return {"step_acc": g.step_acc(), "utterance_acc": float(np.mean([t == f for _, t, f in rows])),
            "bad_rejected": float(np.mean([f not in ACCEPT for k, _, f in rows if k == "bad"])),
            "good_accepted": float(np.mean([f in ACCEPT for k, _, f in rows if k != "bad"]))}


def _job(args):
    """One fly; scores at every threshold of the grid. With `thr` given (test), the speaker
    is trained and evaluated too when `speaker` is set."""
    kind, cfg, seed, run, split, thr, speaker = args
    t0 = time.time()
    arch, teacher, stim = run
    g = GrammarFly(kind, cfg, seed, arch, teacher, stim)
    data = corpus(cfg, seed, split)[:cfg["grammar"]["n_eval_dialogues"]]
    out = {"seed": seed, "run": list(run), "split": split, "by_threshold": {}}
    for t in ([thr] if thr is not None else cfg["grammar"]["conf_grid"]):
        g.thr = t
        out["by_threshold"][str(t)] = utterance_scores(g, data)
    if thr is not None:
        out.update(out["by_threshold"][str(thr)], threshold=thr)
        log.info("%s seed %d %s @%.2f: step %.3f utt %.3f bad-rej %.3f good-acc %.3f", kind, seed,
                 run, thr, out["step_acc"], out["utterance_acc"], out["bad_rejected"], out["good_accepted"])
    if speaker:
        train = corpus(cfg, seed, "train")[:cfg["speak"]["n_train_dialogues"]]
        sp = GrammarSpeaker(kind, cfg, seed, g)
        sp.train(train, cfg["speak"]["epochs"])
        out["speaker"] = r = evaluate(sp, data, cfg["lang"])
        log.info("%s seed %d speaker: gram %.3f approp %.3f exact %.3f ppl %.3f %s", kind, seed,
                 r["grammatical"], r["appropriate"], r["exact"], r["perplexity"],
                 {k: round(v, 3) for k, v in r["exact_by_kind"].items()})
    log.info("job %s seed %d %s done in %.0fs", run, seed, split, time.time() - t0)
    return out


def _pool(cfg, jobs):
    if cfg.get("workers", 1) > 1:
        with ProcessPoolExecutor(cfg["workers"]) as ex:
            return list(ex.map(_job, jobs))
    return [_job(j) for j in jobs]


def exp_grammar(cfg: dict, seed: int) -> dict:
    """7c: for every run (arch, teacher, stim), the confidence threshold is chosen on the
    select seeds (dev split: smallest grid value with the highest mean utterance accuracy),
    then the test seeds are run at that threshold. The main run also trains the speaker."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    base = yaml.safe_load((root / cfg["base"]).read_text())
    cfg = {**base, **cfg, "lang": {**base["lang"], **cfg.get("lang_overrides", {})}}
    gc, out = cfg["grammar"], []
    for kind in cfg["simulators"]:
        runs = [tuple(r) for r in gc["runs"]]
        sel = _pool(cfg, [(kind, cfg, s, r, "dev", None, False) for r in runs for s in cfg["select_seeds"]])
        thr = {}
        for r in runs:
            rs = [x for x in sel if tuple(x["run"]) == r]
            score = {t: np.mean([x["by_threshold"][str(t)]["utterance_acc"] for x in rs]) for t in gc["conf_grid"]}
            thr[r] = max(gc["conf_grid"], key=lambda t: (score[t], -t))
            log.info("%s %s: threshold %.2f chosen (select mean utterance acc %.3f)", kind, r, thr[r], score[thr[r]])
        test = _pool(cfg, [(kind, cfg, s, r, "test", thr[r], r == tuple(gc["main"]))
                           for r in runs for s in cfg["seeds"]])
        rows = []
        for r in runs:
            rs = [x for x in test if tuple(x["run"]) == r]
            row = {"run": list(r), "threshold": thr[r]}
            for k in ("step_acc", "utterance_acc", "bad_rejected", "good_accepted"):
                row[k] = [float(np.mean([x[k] for x in rs])), float(np.min([x[k] for x in rs]))]
            if "speaker" in rs[0]:
                for k in ("perplexity", "grammatical", "appropriate", "exact"):
                    row["speaker_" + k] = [float(np.mean([x["speaker"][k] for x in rs])),
                                           float(np.min([x["speaker"][k] for x in rs]))]
            rows.append(row)
            log.info("%s summary %s", kind, row)
        out.append({"simulator": kind, "thresholds": {str(k): v for k, v in thr.items()},
                    "rows": rows, "select": sel, "test": test})
    return {"grammar": out}
