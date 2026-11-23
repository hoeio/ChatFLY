"""Phase 9 (ADR 0014): a Toki Pona chatbot made of flies. Shared parts.

- `klass(word)`: the word class the grammar runs on (a lookup; 8b-3 showed a class fly learns it).
- `GRAMMAR`: the reference finite-state grammar over classes, with a role per state. It is the
  teacher's textbook: the grammar fly learns this table (H2 at full scale).
- `parse(words)`: run the reference table; returns the states (None if rejected).
"""

import copy
import re
import zlib
from itertools import combinations

import numpy as np

from chatfly.fsm import PNDirect, group_rates
from chatfly.sim import SIMULATORS
from chatfly.tokipona import PU

PREPS = ("lon", "tawa", "tan", "kepeken", "sama")
PARTS = ("li", "e", "la", "pi", "o", "en", "anu", "a")
CLASSES = ("B", "W", "P", *PARTS)
NAME = "NAME"  # capitalized unofficial words (jan Sonja): one token, class W
VOCAB = sorted(set(PU) | {"ali"})
CONTENT = [w for w in VOCAB if w not in PARTS]


def klass(w):
    if w in ("mi", "sina"):
        return "B"
    if w in PREPS:
        return "P"
    return w if w in PARTS else "W"


def tokens(text):
    """Clauses of lower-case words; names become NAME; None for a clause with a non-pu word."""
    out = []
    for part in re.split(r"[.!?:;]+", text):
        ws = [NAME if x[0].isupper() else x.lower() for x in re.findall(r"[A-Za-z]+", part)]
        if ws:
            out.append(ws if all(w == NAME or w in VOCAB for w in ws) else None)
    return out


# state -> role of the word that led into it (what memory and speaker need)
ROLE = {"MI": "subj", "SUBJ": "subj", "SEN": "subj", "SPI": "subj", "SPI1": "subj",
        "MIP": "pred", "PRED": "pred", "PPI": "pred", "PPI1": "pred", "PANU": "pred",
        "LI": "li", "OPRED": "o", "E": "e", "OBJ": "obj", "OPI": "obj", "OPI1": "obj", "OEN": "obj",
        "PREP": "prep", "PNP": "pobj", "NPI": "pobj", "NPI1": "pobj", "PEN": "pobj",
        "S0": "la", "A": "a"}
ACCEPT = {"MI", "SUBJ", "MIP", "PRED", "OBJ", "PNP", "PREP", "A"}


def _grammar():
    """(state, class) -> state; missing pairs are rejected (X)."""
    g = {s: {} for s in ROLE}
    c = ("W", "P", "B")  # content-like classes (mi/sina are content after the first word)
    def put(s, cl, t):
        for k in ([cl] if isinstance(cl, str) else cl):
            g[s][k] = t
    put("S0", ("W", "P"), "SUBJ"); put("S0", "B", "MI"); put("S0", "o", "OPRED"); put("S0", "a", "A")
    put("A", "a", "A"); put("A", ("W", "P"), "SUBJ"); put("A", "B", "MI"); put("A", "o", "OPRED")
    # bare mi / sina: the next content word is the predicate
    put("MI", c, "MIP"); put("MI", "en", "SEN"); put("MI", "la", "S0"); put("MI", "o", "OPRED"); put("MI", "a", "MI")
    put("MI", "li", "LI")  # "mi li" is non-standard but frequent in learner text (measured)
    # noun-phrase subject
    put("SUBJ", c, "SUBJ"); put("SUBJ", "pi", "SPI"); put("SUBJ", "li", "LI"); put("SUBJ", "o", "OPRED")
    put("SUBJ", ("en", "anu"), "SEN"); put("SUBJ", "la", "S0"); put("SUBJ", "a", "SUBJ")
    put("SPI", c, "SPI1"); put("SPI1", c, "SUBJ")
    put("SEN", c, "SUBJ")
    # predicate (after li / o / bare mi)
    for s in ("LI", "OPRED", "PANU"):
        put(s, c, "PRED")
    for s in ("MIP", "PRED"):
        put(s, ("W", "B"), s); put(s, "pi", "PPI"); put(s, "e", "E"); put(s, "P", "PREP"); put(s, "li", "LI")
        put(s, "la", "S0"); put(s, "anu", "PANU"); put(s, "a", s); put(s, "o", "OPRED")
    put("PPI", c, "PPI1"); put("PPI1", c, "PRED")
    # objects
    put("E", c, "OBJ")
    put("OBJ", ("W", "B"), "OBJ"); put("OBJ", "pi", "OPI"); put("OBJ", "e", "E"); put("OBJ", "P", "PREP")
    put("OBJ", "li", "LI"); put("OBJ", "la", "S0"); put("OBJ", ("en", "anu"), "OEN"); put("OBJ", "a", "OBJ")
    put("OPI", c, "OPI1"); put("OPI1", c, "OBJ"); put("OEN", c, "OBJ")
    # prepositional phrases
    put("PREP", c, "PNP"); put("PREP", "li", "LI"); put("PREP", "la", "S0"); put("PREP", "e", "E")
    put("PNP", ("W", "B"), "PNP"); put("PNP", "pi", "NPI"); put("PNP", "P", "PREP"); put("PNP", "la", "S0")
    put("PNP", "li", "LI"); put("PNP", "e", "E"); put("PNP", ("en", "anu"), "PEN"); put("PNP", "a", "PNP")
    put("NPI", c, "NPI1"); put("NPI1", c, "PNP"); put("PEN", c, "PNP")
    return g


GRAMMAR = _grammar()


def parse(words):
    s, out = "S0", []
    for w in words:
        s = GRAMMAR[s].get(klass(w))
        if s is None:
            return None
        out.append(s)
    return out if s in ACCEPT else None


def roles(words):
    st = parse(words)
    return None if st is None else [ROLE[s] for s in st]


# ---- flies (rate model; the spiking model shares the Simulator interface) ----


class Odors:
    """A fixed random PN pattern per key, e.g. ("subj", "jan"): the same word in another channel
    smells different (the 8h pointer-fly trick). Deterministic in (seed, key)."""

    def __init__(self, cfg, seed, n_pn, frac):
        t = cfg["tokens"]
        self.n_pn, self.k, self.on, self.off, self.seed, self.memo = n_pn, round(frac * n_pn), t["pn_on_hz"], t["pn_spont_hz"], seed, {}

    def __call__(self, *key):
        if key not in self.memo:
            r = np.full(self.n_pn, self.off)
            rng = np.random.default_rng([self.seed, zlib.crc32(repr(key).encode())])
            r[rng.choice(self.n_pn, self.k, replace=False)] = self.on
            self.memo[key] = r
        return self.memo[key]

    def top(self, drive, tie):
        """The k most driven PNs fire (a fixed per-PN jitter breaks ties)."""
        r = np.full(self.n_pn, self.off)
        r[np.argsort(-(drive + tie))[:self.k]] = self.on
        return r


def mb_cfg(cfg, n_out, n_kc, kc_theta, n_pn):
    c = copy.deepcopy(cfg)
    c["mb"].update(compartments=[str(i) for i in range(n_out)], n_kc=n_kc, n_pn=n_pn)
    c["rate"]["kc_theta"] = kc_theta
    return c


class Fly:
    """One mushroom body: PN pattern -> KCs -> n_out MBON groups, learning by the dopamine rule.
    Labels are k-of-n group codes (H1); teaching depresses every group outside the label's code."""

    def __init__(self, cfg, seed, salt, labels, f, control=None):
        k, n = f.get("code", (1, len(labels)))
        sets = list(combinations(range(n), k))
        if len(sets) < len(labels):
            raise ValueError(f"{k} of {n} has {len(sets)} codes for {len(labels)} labels")
        pick = np.random.default_rng([seed, 9, salt]).permutation(len(sets))[:len(labels)] if k > 1 else range(len(labels))
        self.labels, self.code = list(labels), [list(sets[i]) for i in pick]
        self.idx = {x: i for i, x in enumerate(self.labels)}
        c = mb_cfg(cfg, n, f["n_kc"], f["kc_theta"], f["n_pn"])
        self.sim = SIMULATORS["rate"](c, seed * 1000 + salt)
        if control == "no_kc":
            self.sim = PNDirect(self.sim, c, n, np.random.default_rng([seed, 9, salt, 1]), False)
        self.n, self.dur, self.control = n, c["protocol"]["duration_ms"], control
        self.rng = np.random.default_rng([seed, 9, salt, 2])

    def teach(self, x, label, times=1):
        i = int(self.rng.integers(len(self.labels))) if self.control == "shuffled" else self.idx[label]
        dan = np.ones(self.n)
        dan[self.code[i]] = 0.0
        for _ in range(times):
            self.sim.present(x, self.dur, dan)

    def scores(self, x):
        r = group_rates(self.sim, self.sim.present(x, self.dur)["mbon"])
        return np.array([r[g].mean() for g in self.code])

    def read(self, x):
        """(label or None, confidence). Confidence = (mean rate of the label's groups - mean rate
        of the other groups) / the label's mean: 0 for an untaught odor (all groups at w0), near 1
        for a taught one. (top - second) / top fails for k-of-n codes, whose runner-up shares a group
        with the winner (8g)."""
        r = group_rates(self.sim, self.sim.present(x, self.dur)["mbon"])
        s = np.array([r[g].mean() for g in self.code])
        o = np.sort(s)[::-1]
        if o[0] <= 0 or o[0] == o[1]:
            return None, 0.0
        i = int(np.argmax(s))
        rest = np.delete(r, self.code[i])
        return self.labels[i], float((s[i] - rest.mean()) / s[i])


class Binder:
    """A fixed (never taught) KC layer that binds two odors into a new one: the union drives KCs
    (a conjunction, the project's core assumption), and the active KCs drive a fixed random set of
    PNs; the most driven PNs fire (re-encoding). Chaining it gives any phrase a key odor.
    control "no_kc": the union itself is re-encoded (the same top-k rule on PN rates)."""

    def __init__(self, cfg, seed, odors, b, control=None):
        c = mb_cfg(cfg, 1, b["n_kc"], b["kc_theta"], odors.n_pn)
        self.sim, self.od, self.control = SIMULATORS["rate"](c, seed * 1000 + 77), odors, control
        rng = np.random.default_rng([seed, 9, 78])
        self.out = rng.integers(odors.n_pn, size=(b["n_kc"], b["fan_out"]))
        self.tie = rng.random(odors.n_pn) * 1e-3
        # signed: half of each KC's fan-out excites, half inhibits, so every PN's expected drive is
        # zero whatever KCs fire (no PN becomes popular along a chain)
        self.signed = b.get("signed", False)
        self.sign = np.where(np.arange(b["fan_out"]) % 2 == 0, 1.0, -1.0)
        self.dur, self.thr = c["protocol"]["duration_ms"], cfg["readout"]["kc_active_hz"]
        self.noise = np.random.default_rng([seed, 9, 79]) if b.get("poisson") else None
        self.window = b.get("window", 1.0)
        self.weighted = b.get("weighted", True)  # False: an active KC drives its targets by 1 (robust to small rate changes)
        # fixed per-PN gain: 1 / the PN's mean drive over random odor pairs, so no PN is in more
        # keys than others (without it, popular PNs feed forward along a chain: measured bias 2.4x)
        self.gain = np.ones(odors.n_pn)
        if b.get("calibrate", 0):
            cal = np.random.default_rng([seed, 9, 80])
            def rand():
                r = np.full(odors.n_pn, odors.off)
                r[cal.choice(odors.n_pn, odors.k, replace=False)] = odors.on
                return r
            mean = np.mean([self._drive(rand(), rand()) for _ in range(b["calibrate"])], axis=0)
            self.gain = 1.0 / np.maximum(mean, mean[mean > 0].min() if (mean > 0).any() else 1.0)

    def _drive(self, x, y):
        kc = self.sim.present(np.maximum(x, y), self.dur)["kc"]
        on = kc >= self.thr  # each active KC drives its fan-out PNs in proportion to its rate
        fo = self.out.shape[1]
        w = np.repeat(kc[on] if self.weighted else np.ones(on.sum()), fo) * (np.tile(self.sign, on.sum()) if self.signed else 1.0)
        return np.bincount(self.out[on].ravel(), w, minlength=self.od.n_pn)

    def __call__(self, x, y):
        u = np.maximum(x, y)
        if self.noise is not None:  # spike-count noise on the input (one presentation window)
            sec = self.dur / 1000.0 * self.window  # window: integration time in presentation lengths
            u = self.noise.poisson(u * sec) / sec
        if self.control == "no_kc":
            return self.od.top(u, self.tie)
        return self.od.top(self._drive(u, u) * self.gain, self.tie)

    def phrase(self, role, words):
        h = self.od(role, words[0])
        for w in words[1:]:
            h = self(h, self.od(role, w))
        return h


END = "#"


def slots(words):
    """Reference split of a parsed clause into slots (the teacher's view; the pipeline gets them
    from the grammar fly's states). None unless it is one plain `S li P [e O]` / `mi P [e O]`."""
    r = roles(words)
    if r is not None:  # drop the interjection a
        words, r = [w for w in words if w != "a"], [x for w, x in zip(words, r) if w != "a"]
    if r is None or not words or "la" in r or "o" in r or "prep" in r or r.count("li") > 1 or r.count("e") > 1:
        return None
    if words[0] in ("mi", "sina") and "li" not in r:
        r = ["subj"] + r[1:]
    s = [w for w, x in zip(words, r) if x == "subj"]
    p = [w for w, x in zip(words, r) if x == "pred"]
    o = [w for w, x in zip(words, r) if x == "obj"]
    if not s or not p or "ala" in s or "anu" in words or "seme" in words or "en" in words or "pi" in words:
        return None
    neg = "ala" in p
    p = [w for w in p if w != "ala"]
    return (tuple(s), tuple(p), tuple(o), neg) if p else None


class Recall:
    """Memory of phrases by chained keys: key -> first word, bind(key, word) -> next word, ... END.
    One Fly over the content words + END; one Binder for keys and chains."""

    def __init__(self, cfg, seed, binder, f, words, control=None):
        self.b, self.od = binder, binder.od
        self.fly = Fly(cfg, seed, 2, list(words) + [END], f, control)

    def tell(self, key, phrase, times):
        for w in list(phrase) + [END]:
            self.fly.teach(key, w, times)
            key = self.b(key, self.od("ans", w))

    def ask(self, key, thr, max_len=6):
        """The phrase, or None if any step is below the threshold (unknown)."""
        out = []
        while len(out) < max_len:
            w, c = self.fly.read(key)
            if w is None or c < thr:
                return None
            if w == END:
                return tuple(out)
            out.append(w)
            key = self.b(key, self.od("ans", w))
        return None
