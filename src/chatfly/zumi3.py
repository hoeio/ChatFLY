"""Phase 7e: Zumi 3, "GPT-2 level in Zumi" (ADR 0012).

Zumi 3 adds, on top of Zumi 2 with strict property order:
- the pronoun `to` ("it": the last thing the user named; resolved by the interface, a
  fixed recency rule);
- fly replies of one or two sentences, chosen at random where several are right:
  teach -> "o." or "o. to <prop> ba?" (asks about the first pair it does not know),
  "no" about a thing -> "na." or "na. to ka <what it is>.",
  "no" about a place -> "na." or "na. <where it is>.", what -> "to ka <props>." or "hm.";
- free generation: every reply word is sampled from the speaker's normalized MBON rates.

The corpus oracle is stochastic: each turn carries the set of every correct reply
(`support`) and the cue list a perfect set of flies would give the speaker (`cue`).
The speaker (`CueSpeaker`) is the split speaker of 7b-7d. Fly 0 smells (grammar state, first
cue); fly 1 smells (previous word, current cue). The cue pointer moves on when the fly says
the cue's word (fixed rule). Cues: a:<answer word>, s:<property stated>, q:<property asked>,
c:ba, "-". They come from the ring, memory and comparator flies; the interface orders them
(answer, detail; the ask-back pair is the first unknown pair in pair order).
"""

import copy
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chatfly.chat import fly_for
from chatfly.conditioning import token_patterns
from chatfly.fsm import group_rates, make_task
from chatfly.grammar import GrammarFly, true_state
from chatfly.parity import dan_for
from chatfly.speak import (
    EOS,
    RNN,
    SEP,
    STOP,
    NGram,
    SpeakingFly,
    evaluate,
    kind_of,
    noun_word,
    utterance,
    vocabulary,
)
from chatfly.talk import Comparator
from chatfly.zumi import Conversation, Memory, parse, props, reserved

log = logging.getLogger("chatfly")
MEMORY_KINDS = ("ask", "what")


# --- corpus ---------------------------------------------------------------------------------

def oracle3(p, world, task, lx, rng, z) -> tuple:
    """-> (sampled reply, support set, cue list); updates world {"place", "facts", "topic"}."""
    r, pr, pron = lx["replies"], props(lx), lx["pronoun"]
    one = lambda w: ([w, STOP], {(w, STOP)})
    if "refuse" in p:
        return *one(r["huh"]), []
    if "noun" in p:
        n = world["topic"] if p["noun"] == pron else p["noun"]
        if n is None:
            return *one(r["huh"]), []
        world["topic"] = n
        f = world["facts"].setdefault(n, {})
        if p["kind"] == "teach":
            for w in p["props"]:
                f[pr[w][0]] = pr[w][1]
            unknown = [i for i in range(len(lx["props"])) if i not in f]
            base = [r["ok"], STOP]
            if not unknown:
                return base, {tuple(base)}, []
            q = next(iter(lx["props"][unknown[0]]))
            ask = base + [pron, q, lx["yesno"], STOP]
            return (ask if rng.random() < z["p_ask_back"] else base), {tuple(base), tuple(ask)}, \
                ["q:" + q, "c:" + lx["yesno"]]
        if p["kind"] == "what":
            words = [list(lx["props"][i])[f[i]] for i in sorted(f)]
            if not words:
                return *one(r["unsure"]), ["a:" + r["unsure"]]
            rep = [pron, lx["is"]] + words + [STOP]
            return rep, {tuple(rep)}, ["s:" + w for w in words]
        i, side = pr[p["props"][0]]
        if i not in f:
            return *one(r["unsure"]), ["a:" + r["unsure"]]
        if f[i] == side:
            return *one(r["yes"]), ["a:" + r["yes"]]
        true = list(lx["props"][i])[f[i]]
        base, more = [r["no"], STOP], [r["no"], STOP, pron, lx["is"], true, STOP]
        return (more if rng.random() < z["p_elaborate"] else base), {tuple(base), tuple(more)}, \
            ["a:" + r["no"], "s:" + true]
    s = task["states"][0] if p["restart"] else world["place"]
    for m in p["moves"]:
        s = task["delta"][s][m]
    if not p["suppose"]:
        world["place"] = s
    if p["ask"] is None:
        return *one(r["ok"]), []
    if p["ask"] == "where":
        return *one(s), ["a:" + s]
    if s == p["ask"]:
        return *one(r["yes"]), ["a:" + r["yes"]]
    base, more = [r["no"], STOP], [r["no"], STOP, s, STOP]
    return (more if rng.random() < z["p_elaborate"] else base), {tuple(base), tuple(more)}, \
        ["a:" + r["no"], "a:" + s]


def dialogue3(rng, lx, task, co, z) -> list:
    """[(user words, reply, kind, support, cue)]; `to` replaces a named thing with p_pronoun
    once there is a topic (it then means the topic)."""
    nouns = [noun_word(rng, co, reserved(lx)) for _ in range(co["n_nouns"])]
    world, out = {"place": task["states"][0], "facts": {}, "topic": None}, []
    for _ in range(co["turns"]):
        u = utterance(rng, lx, nouns, co)
        if u[0] in nouns and world["topic"] is not None and rng.random() < z["p_pronoun"]:
            u = [lx["pronoun"]] + u[1:]
        p = parse(" ".join(u), lx)
        reply, support, cue = oracle3(p, world, task, lx, rng, z)
        out.append((u, reply, kind_of(p), support, cue))
    return out


def corpus3(cfg, seed, split, which="corpus") -> list:
    co = cfg[which]
    rng = np.random.default_rng([seed, cfg["rng_streams"]["corpus"], 3, ("train", "dev", "test", "load").index(split)])
    task = make_task(cfg["chat"]["task"], None)
    return [dialogue3(rng, cfg["lang"], task, co, cfg["zumi3"]) for _ in range(co["n_" + split])]


# --- the speaker and the whole system --------------------------------------------------------

class CueSpeaker(SpeakingFly):
    """Split speaker over (grammar state, first cue) and (previous word, current cue)."""

    def __init__(self, kind, cfg, seed, grammar_states, control=None):
        c = copy.deepcopy(cfg)
        c["speak"]["arch"] = "split"
        super().__init__(kind, c, seed, control)
        lx = cfg["lang"]
        cues = (["a:" + w for w in list(lx["replies"].values()) + list(lx["places"])]
                + [f"{t}:{w}" for t in "sq" for w in props(lx)] + ["c:" + lx["yesno"], "-"])
        cc = copy.deepcopy(self.flies[0].cfg)
        cc["tokens"]["names"] = [f"g:{s}" for s in grammar_states] + ["cue:" + x for x in cues]
        self.pats.update(token_patterns(cc, np.random.default_rng(seed + cfg["rng_streams"]["tokens"] + 3)))

    def route_cue(self, g, cues, prev, k):
        c = cues[k] if k < len(cues) else "-"
        if prev == SEP:
            return self.flies[0], np.maximum(self.pats["g:" + g], self.pats["cue:" + c])
        return self.flies[1], np.maximum(self.pats["p:" + prev], self.pats["cue:" + c])

    @staticmethod
    def advance(cues, k, word):
        return k + 1 if k < len(cues) and cues[k].split(":", 1)[1] == word else k

    def train_cues(self, dialogues, epochs, task):
        n = len(self.vocab)
        for _ in range(epochs):
            for d in dialogues:
                for user, reply, _, _, cues in d:
                    g, prev, k = true_state(user, task), SEP, 0
                    for t in reply + [EOS]:
                        tgt = int(self.rng.integers(n)) if self.control == "shuffled_teacher" else self.index[t]
                        fly, stim = self.route_cue(g, cues, prev, k)
                        fly.present(stim, self.dur, dan_for(tgt, n))
                        prev, k = t, self.advance(cues, k, t)

    def dist(self, fly, stim):
        u = np.full(len(self.vocab), 1 / len(self.vocab))
        r = group_rates(fly, fly.present(stim, self.dur)["mbon"])
        p = r / r.sum() if r.sum() > 0 else u
        return (1 - self.floor) * p + self.floor * u


class Talker3:
    """Grammar fly + ring fly + memory fly + comparator fly + cue speaker, behind
    `speak.evaluate`'s model interface. control: None | "shuffled" (ring, memory and
    comparator teachers shuffled) | "no_kc_memory"."""

    def __init__(self, kind, cfg, seed, control=None):
        self.kind, self.cfg, self.seed, self.lx = kind, cfg, seed, cfg["lang"]
        shuffled = control == "shuffled"
        self.grammar = GrammarFly(kind, cfg, seed, "routed_shared", "true", "state")
        self.grammar.thr = cfg["talk"]["grammar_threshold"]
        self.ring = fly_for(kind, cfg, seed, random_teacher=shuffled)
        self.comp = Comparator(kind, cfg, seed, shuffled)
        self.speaker = CueSpeaker(kind, cfg, seed, self.grammar.task["states"])
        train = corpus3(cfg, seed, "train")[:cfg["speak"]["n_train_dialogues"]]
        self.speaker.train_cues(train, cfg["speak"]["epochs"], self.grammar.task)
        self.vocab, self.index = self.speaker.vocab, self.speaker.index
        self.mem_control = {"shuffled": "shuffled_teacher", "no_kc_memory": "no_kc"}.get(control)
        self.conv = self.topic = None

    def init(self):
        lx = self.lx
        mem = Memory(self.kind, self.cfg, self.seed, self.mem_control)
        self.conv = Conversation(self.ring, lx, lx["min_confidence"][self.kind], mem,
                                 lx["memory"]["min_confidence"][self.kind])
        self.topic = None
        return ((), self.grammar.task["states"][0], (), None, 0)

    def cues(self, words):
        """The utterance goes to the ring / memory flies (they move or learn); returns the cues."""
        lx, r = self.lx, self.lx["replies"]
        p = parse(" ".join(words), lx)
        if "refuse" in p:
            return []
        if "noun" in p:
            noun = self.topic if p["noun"] == lx["pronoun"] else p["noun"]
            if noun is None:
                return []
            self.topic = noun
            words = [noun] + list(words[1:])
        out = self.conv.turn(" ".join(words))
        kind, m = kind_of(p), self.conv.memory
        if kind == "teach":
            for i, pair in enumerate(lx["props"]):
                side, conf = m.ask(self.topic, i)
                if side is None or conf < self.conv.mem_conf:
                    return ["q:" + next(iter(pair)), "c:" + lx["yesno"]]
            return []
        if kind == "statement":
            return [] if out["final"] is not None else ["a:" + r["unsure"]]
        if kind == "where":
            return ["a:" + (out["final"] or r["unsure"])]
        if kind == "yesno":
            c = self.comp.answer(p["ask"], out["final"] or r["unsure"])
            return ["a:" + c] + (["a:" + out["final"]] if c == r["no"] and out["final"] else [])
        if kind == "ask":
            word = out["memory"][0][1]
            c = self.comp.answer(p["props"][0], word or r["unsure"])
            return ["a:" + c] + (["s:" + word] if c == r["no"] and word else [])
        known = [w for _, w, _ in out["memory"] if w]
        return ["s:" + w for w in known] or ["a:" + r["unsure"]]

    def step(self, st, t):
        words, g, cues, prev, k = st
        u = np.full(len(self.vocab), 1 / len(self.vocab))
        if t == EOS:
            return ((), self.grammar.task["states"][0], (), None, 0), u
        if prev is None and t != SEP:
            return (words + (t,), self.grammar.step(g, t), (), None, 0), u
        if t == SEP:
            cues, prev, k = tuple(self.cues(list(words))), SEP, 0
        else:
            prev, k = t, CueSpeaker.advance(cues, k, t)
        st = (words, g, cues, prev, k)
        return st, self.speaker.dist(*self.speaker.route_cue(g, cues, prev, k))


# --- a gated RNN baseline -------------------------------------------------------------------

class GRU(RNN):
    """GRU version of speak.RNN (same input coding, loss, training and selection)."""

    def __init__(self, vocab, rc, seed):
        super().__init__(vocab, rc, seed)
        rng = np.random.default_rng(seed + 11)
        v, d, h = len(vocab), rc["noun_dim"], rc["hidden"]
        s = lambda *shape: rng.normal(0, 1 / np.sqrt(shape[0]), shape)
        self.p = {"Wo": s(h, v), "bo": np.zeros(v)}
        for g in "zrn":
            self.p.update({f"W{g}": s(v + d, h), f"U{g}": s(h, h), f"b{g}": np.zeros(h)})

    def cell(self, x, h):
        p = self.p
        sig = lambda a: 1 / (1 + np.exp(-a))
        z = sig(x @ p["Wz"] + h @ p["Uz"] + p["bz"])
        r = sig(x @ p["Wr"] + h @ p["Ur"] + p["br"])
        n = np.tanh(x @ p["Wn"] + (r * h) @ p["Un"] + p["bn"])
        return (1 - z) * n + z * h, (z, r, n)

    def step(self, h, t):
        h, _ = self.cell(self.x(t), h)
        zz = h @ self.p["Wo"] + self.p["bo"]
        e = np.exp(zz - zz.max())
        return h, e / e.sum()

    def loss_grad(self, X, Y):
        p, (T, B, _) = self.p, X.shape
        H = np.zeros((T + 1, B, self.rc["hidden"]))
        P = np.zeros((T, B, len(self.vocab)))
        cache = []
        for t in range(T):
            H[t + 1], c = self.cell(X[t], H[t])
            cache.append(c)
            zz = H[t + 1] @ p["Wo"] + p["bo"]
            e = np.exp(zz - zz.max(1, keepdims=True))
            P[t] = e / e.sum(1, keepdims=True)
        m = Y >= 0
        n_ = max(m.sum(), 1)
        loss = -np.log(P[m, Y[m]] + 1e-12).sum() / n_
        dZ = P.copy()
        dZ[m, Y[m]] -= 1
        dZ *= m[..., None] / n_
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dh = np.zeros((B, self.rc["hidden"]))
        for t in range(T - 1, -1, -1):
            z, r, n = cache[t]
            h, x = H[t], X[t]
            g["Wo"] += H[t + 1].T @ dZ[t]
            g["bo"] += dZ[t].sum(0)
            dh = dh + dZ[t] @ p["Wo"].T
            dn, dz, dprev = dh * (1 - z), dh * (h - n), dh * z
            dan = dn * (1 - n ** 2)
            g["Wn"] += x.T @ dan
            g["Un"] += (r * h).T @ dan
            g["bn"] += dan.sum(0)
            drh = dan @ p["Un"].T
            dprev += drh * r
            dar = drh * h * r * (1 - r)
            daz = dz * z * (1 - z)
            for k, da in (("z", daz), ("r", dar)):
                g["W" + k] += x.T @ da
                g["U" + k] += h.T @ da
                g["b" + k] += da.sum(0)
                dprev += da @ p["U" + k].T
            dh = dprev
        return loss, g


# --- experiment -----------------------------------------------------------------------------

def memory_exact(r, test):
    """Exact on memory questions (ask, what), weighted by their counts."""
    w = [sum(k == x for d in test for _, _, k, *_ in d) for x in MEMORY_KINDS]
    return float(np.average([r["exact_by_kind"].get(x, 0.0) for x in MEMORY_KINDS], weights=w))


def _job(args):
    kind, cfg, seed, what = args
    t0, lx, zc = time.time(), cfg["lang"], cfg["zumi3"]
    rng = np.random.default_rng([seed, cfg["rng_streams"]["corpus"], 99])
    ev = lambda m, data: evaluate(m, data, lx, zc["max_len"], zc["temperature"], rng)
    if what == "baselines":
        train, dev, test = (corpus3(cfg, seed, s) for s in ("train", "dev", "test"))
        test = test[:cfg["talk"]["n_eval_dialogues"]]
        out = {}
        for n in cfg["baselines"]["ngram_orders"]:
            out[f"{n}-gram"] = ev(NGram(n, cfg["baselines"]["alpha"], vocabulary(lx)).fit(train), test)
        for name, cls in (("rnn", RNN), ("gru", GRU)):
            m = cls(vocabulary(lx), cfg["baselines"]["rnn"], seed)
            m.fit(train, dev[:cfg["baselines"]["dev_dialogues"]], lx, seed)
            out[name] = ev(m, test)
        res = {"seed": seed, "what": what, "models": out}
    else:
        which, control = {"fly": ("test", None), "shuffled": ("test", "shuffled"),
                          "load": ("load", None), "load_no_kc": ("load", "no_kc_memory")}[what]
        test = corpus3(cfg, seed, which, "corpus" if which == "test" else "load_corpus")
        test = test[:cfg["talk"]["n_eval_dialogues"]] if which == "test" else test
        tk = Talker3(kind, cfg, seed, control)
        r = ev(tk, test)
        r["memory_exact"] = memory_exact(r, test)
        res = {"seed": seed, "what": what, "models": {"system": r}}
    for name, r in res["models"].items():
        log.info("%s seed %d %s %s: ppl %.3f exact %.3f gram %.3f approp %.3f d2 %.3f maxlen %.3f%s %s",
                 kind, seed, what, name, r["perplexity"], r["exact"], r["grammatical"], r["appropriate"],
                 r["distinct_2"], r["max_len_hit"],
                 f" memory {r['memory_exact']:.3f}" if "memory_exact" in r else "",
                 {k: round(v, 3) for k, v in r["exact_by_kind"].items()})
    log.info("job %s seed %d done in %.0fs", what, seed, time.time() - t0)
    return res


def exp_zumi3(cfg: dict, seed: int) -> dict:
    """7e: baselines and the fly system on the Zumi 3 test corpus (sampled generation),
    the shuffled control, and the memory-load condition with the no-KC memory control."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    base = yaml.safe_load((root / cfg["base"]).read_text())
    cfg = {**base, **cfg, "lang": {**base["lang"], **cfg.get("lang_overrides", {})},
           "rng_streams": {**base["rng_streams"], **cfg.get("rng_streams", {})}}
    out = []
    for kind in cfg["simulators"]:
        jobs = [(kind, cfg, s, w) for s in cfg["seeds"] for w in cfg["zumi3"]["jobs"]]
        if cfg.get("workers", 1) > 1:
            with ProcessPoolExecutor(cfg["workers"]) as ex:
                res = list(ex.map(_job, jobs))
        else:
            res = [_job(j) for j in jobs]
        rows = {}
        for w in cfg["zumi3"]["jobs"]:
            rs = [r for r in res if r["what"] == w]
            for name in rs[0]["models"]:
                ms = [r["models"][name] for r in rs]
                keys = [k for k in ("perplexity", "exact", "grammatical", "appropriate", "distinct_2",
                                    "max_len_hit", "memory_exact") if k in ms[0]]
                rows[f"{w}/{name}"] = {k: [float(np.mean([m[k] for m in ms])), float(np.min([m[k] for m in ms])),
                                           float(np.max([m[k] for m in ms]))] for k in keys}
                log.info("%s summary %s/%s %s", kind, w, name, rows[f"{w}/{name}"])
        out.append({"simulator": kind, "rows": rows, "runs": res})
    return {"zumi3": out}
