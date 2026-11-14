"""Phase 7d: replies whose content comes from the flies, over whole dialogues (ADR 0012).

Parts (every answer comes from a fly; the interface routes and orders):
- grammar fly (7c, `grammar.GrammarFly`): its state, read word by word; X = refused.
- ring fly (Phase 6) and memory fly (Zumi 2), driven through `zumi.Conversation` exactly as
  in the Zumi 2 chat: the interface parses (fixed grammar) and feeds moves to the ring fly and
  nouns to the memory fly. Only their decoded outputs are used here, not Conversation's reply.
- comparator fly (`Comparator`): (asked word, the fly's answer word) -> ya / na / hm, so that
  "is it X?" is not answered by the interface comparing words. One fly, readout selected by the
  asked word (the 7c architecture).
- speaker (`ContentSpeaker`): 7b/7c's split speaker. Fly 0 smells (grammar state, first content
  word); fly 1 smells (previous word, next content word). Content words: the ring's place, the
  comparator's ya/na/hm, the memory's known properties in pair order ("what"), "hm" when a fly
  is unsure, "-" for no content. The interface's rules: which fly answers which question
  (by the parse), the order of the "what" list (pair order), and stop at EOS.
"""

import copy
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chatfly.chat import fly_for
from chatfly.conditioning import token_patterns
from chatfly.fsm import Flies
from chatfly.grammar import GrammarFly, true_state
from chatfly.parity import dan_for
from chatfly.speak import EOS, SEP, SpeakingFly, corpus, evaluate, kind_of
from chatfly.zumi import Conversation, Memory, parse, props

log = logging.getLogger("chatfly")
QUERIES = ("where", "yesno", "ask", "what")


class Comparator:
    """(asked word, answer word) -> ya / na / hm. Asked a place: answer a place or hm. Asked a
    property: answer that property, its opposite, or hm."""

    def __init__(self, kind, cfg, seed, shuffled=False):
        lx, rs = cfg["lang"], cfg["rng_streams"]
        r, pr = lx["replies"], props(lx)
        self.outs = [r["yes"], r["no"], r["unsure"]]
        self.table = {}
        for q in lx["places"]:
            for a in list(lx["places"]) + [r["unsure"]]:
                self.table[(q, a)] = r["unsure"] if a == r["unsure"] else (r["yes"] if a == q else r["no"])
        for w, (i, _) in pr.items():
            for a in list(lx["props"][i]) + [r["unsure"]]:
                self.table[(w, a)] = r["unsure"] if a == r["unsure"] else (r["yes"] if a == w else r["no"])
        asked = sorted({q for q, _ in self.table})
        answers = sorted({a for _, a in self.table})
        c = copy.deepcopy(cfg)
        c["mb"]["compartments"] = self.outs
        c["tokens"]["names"] = ["ans:" + a for a in answers]
        self.pats = token_patterns(c, np.random.default_rng(seed + rs["tokens"] + 2))
        task = {"states": self.outs, "inputs": asked}
        self.flies = Flies(kind, c, task, seed + 7 * rs["fly"], "routed_shared", 1)
        rng, entries = np.random.default_rng(seed + rs["task"] + 2), list(self.table.items())
        for _ in range(c["protocol"]["n_train"]):
            for i in rng.permutation(len(entries)):
                (q, a), t = entries[i]
                tgt = int(rng.integers(len(self.outs))) if shuffled else self.outs.index(t)
                self.flies.teach(self.pats["ans:" + a], q, tgt)

    def answer(self, asked, ans):
        d = self.flies.decide(self.pats["ans:" + ans], asked)
        return self.outs[2] if d is None else self.outs[d]

    def accuracy(self):
        return float(np.mean([self.answer(q, a) == t for (q, a), t in self.table.items()]))


def oracle_content(user, reply, kind):
    """Training targets for the speaker's content tokens (what a perfect set of flies gives)."""
    return list(reply) if kind in QUERIES else ["-"]


class ContentSpeaker(SpeakingFly):
    """Split speaker over (grammar state, content) and (previous word, content)."""

    def __init__(self, kind, cfg, seed, grammar_states, control=None):
        c = copy.deepcopy(cfg)
        c["speak"]["arch"] = "split"
        super().__init__(kind, c, seed, control)
        cc = copy.deepcopy(self.flies[0].cfg)
        cc["tokens"]["names"] = [f"g:{s}" for s in grammar_states] + [f"c:{w}" for w in self.vocab[:-1] + ["-"]]
        self.pats.update(token_patterns(cc, np.random.default_rng(seed + cfg["rng_streams"]["tokens"] + 3)))

    def route_c(self, g, content, prev, i):
        c = content[i] if i < len(content) else "-"
        if prev == SEP:
            return self.flies[0], np.maximum(self.pats["g:" + g], self.pats["c:" + c])
        return self.flies[1], np.maximum(self.pats["p:" + prev], self.pats["c:" + c])

    def train_content(self, dialogues, epochs, task):
        n = len(self.vocab)
        for _ in range(epochs):
            for d in dialogues:
                for user, reply, kind in d:
                    g, content, prev = true_state(user, task), oracle_content(user, reply, kind), SEP
                    for i, t in enumerate(reply + [EOS]):
                        tgt = int(self.rng.integers(n)) if self.control == "shuffled_teacher" else self.index[t]
                        fly, stim = self.route_c(g, content, prev, i)
                        fly.present(stim, self.dur, dan_for(tgt, n))
                        prev = t

    def dist(self, fly, stim):
        from chatfly.fsm import group_rates

        u = np.full(len(self.vocab), 1 / len(self.vocab))
        r = group_rates(fly, fly.present(stim, self.dur)["mbon"])
        p = r / r.sum() if r.sum() > 0 else u
        return (1 - self.floor) * p + self.floor * u


class Talker:
    """The whole system behind `speak.evaluate`'s model interface. init() starts a dialogue
    (fresh memory, ring back at the nest). The SEP step processes the utterance once: the
    parse routes it to the ring / memory flies (which learn or move), and the content words
    are fixed for the reply."""

    def __init__(self, kind, cfg, seed, control=None):
        tc = cfg["talk"]
        self.kind, self.cfg, self.seed, self.control, self.lx = kind, cfg, seed, control, cfg["lang"]
        shuffled = control == "shuffled"
        self.grammar = GrammarFly(kind, cfg, seed, "routed_shared", "true", "state")
        self.grammar.thr = tc["grammar_threshold"]
        self.ring = fly_for(kind, cfg, seed, random_teacher=shuffled)
        self.comp = Comparator(kind, cfg, seed, shuffled)
        self.speaker = ContentSpeaker(kind, cfg, seed, self.grammar.task["states"])
        train = corpus(cfg, seed, "train")[:cfg["speak"]["n_train_dialogues"]]
        self.speaker.train_content(train, cfg["speak"]["epochs"], self.grammar.task)
        self.vocab, self.index = self.speaker.vocab, self.speaker.index
        self.mem_control = {"shuffled": "shuffled_teacher", "no_kc_memory": "no_kc"}.get(control)
        self.conv = None

    def init(self):
        lx = self.lx
        mem = Memory(self.kind, self.cfg, self.seed, self.mem_control)
        self.conv = Conversation(self.ring, lx, lx["min_confidence"][self.kind], mem,
                                 lx["memory"]["min_confidence"][self.kind])
        return ((), self.grammar.task["states"][0], None, None, 0)

    def content(self, words):
        lx, r = self.lx, self.lx["replies"]
        text = " ".join(words)
        p = parse(text, lx)
        out = self.conv.turn(text)  # the ring and memory flies move / learn here
        if "refuse" in p:
            return ["-"]
        kind = kind_of(p)
        if kind == "teach":
            return ["-"]
        if kind == "statement":
            return ["-"] if out["final"] is not None else [r["unsure"]]
        if kind == "where":
            return [out["final"] or r["unsure"]]
        if kind == "yesno":
            return [self.comp.answer(p["ask"], out["final"] or r["unsure"])]
        if kind == "ask":
            return [self.comp.answer(p["props"][0], out["memory"][0][1] or r["unsure"])]
        return [w for _, w, _ in out["memory"] if w] or [r["unsure"]]  # what

    def step(self, st, t):
        words, g, content, prev, i = st
        u = np.full(len(self.vocab), 1 / len(self.vocab))
        if t == EOS:
            return ((), self.grammar.task["states"][0], None, None, 0), u
        if prev is None and t != SEP:
            return (words + (t,), self.grammar.step(g, t), None, None, 0), u
        if t == SEP:
            content, prev, i = tuple(self.content(list(words))), SEP, 0
        else:
            prev, i = t, i + 1
        st = (words, g, content, prev, i)
        return st, self.speaker.dist(*self.speaker.route_c(g, content, prev, i))


def _job(args):
    kind, cfg, seed, control = args
    t0 = time.time()
    tk = Talker(kind, cfg, seed, control)
    test = corpus(cfg, seed, "test")[:cfg["talk"]["n_eval_dialogues"]]
    r = evaluate(tk, test, cfg["lang"])
    q = [r["exact_by_kind"][k] for k in QUERIES]
    w = [sum(k2 == k for d in test for _, _, k2 in d) for k in QUERIES]
    r["content_exact"] = float(np.average(q, weights=w))
    r["comparator_acc"], r["grammar_step_acc"] = tk.comp.accuracy(), tk.grammar.step_acc()
    log.info("%s seed %d %s: exact %.3f content %.3f approp %.3f gram %.3f ppl %.3f comp %.3f %s (%.0fs)",
             kind, seed, control, r["exact"], r["content_exact"], r["appropriate"], r["grammatical"],
             r["perplexity"], r["comparator_acc"], {k: round(v, 3) for k, v in r["exact_by_kind"].items()},
             time.time() - t0)
    return {"seed": seed, "control": control, **r}


def exp_talk(cfg: dict, seed: int) -> dict:
    """7d: the full system and its controls on the test seeds. Nothing is tuned here: the
    grammar threshold is 7c's, the ring and memory thresholds Phase 6's."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    base = yaml.safe_load((root / cfg["base"]).read_text())
    cfg = {**base, **cfg, "lang": {**base["lang"], **cfg.get("lang_overrides", {})},
           "rng_streams": {**base["rng_streams"], **cfg.get("rng_streams", {})}}
    out = []
    for kind in cfg["simulators"]:
        jobs = [(kind, cfg, s, ctl) for s in cfg["seeds"] for ctl in cfg["talk"]["controls"]]
        if cfg.get("workers", 1) > 1:
            with ProcessPoolExecutor(cfg["workers"]) as ex:
                res = list(ex.map(_job, jobs))
        else:
            res = [_job(j) for j in jobs]
        rows = []
        for ctl in cfg["talk"]["controls"]:
            rs = [r for r in res if r["control"] == ctl]
            row = {"control": ctl, **{k: [float(np.mean([r[k] for r in rs])), float(np.min([r[k] for r in rs]))]
                                      for k in ("exact", "content_exact", "appropriate", "grammatical",
                                                "perplexity")},
                   "exact_by_kind": {k: [float(np.mean([r["exact_by_kind"][k] for r in rs])),
                                         float(np.min([r["exact_by_kind"][k] for r in rs]))]
                                     for k in rs[0]["exact_by_kind"]}}
            rows.append(row)
            log.info("%s summary %s", kind, row)
        out.append({"simulator": kind, "rows": rows, "runs": res})
    return {"talk": out}
