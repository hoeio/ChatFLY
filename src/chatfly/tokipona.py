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
        self.pats = token_patterns(c, np.random.default_rng(seed + rs["tokens"]))
        self.sim = SIMULATORS[kind](c, seed, noise_seed=seed * rs["noise"] + 8)
        if control == "no_kc":
            self.sim = PNDirect(self.sim, c, n, np.random.default_rng([seed, 8, 5]), kind == "spiking")
        self.n, self.control, self.dur = n, control, c["protocol"]["duration_ms"]
        self.rng = np.random.default_rng([seed, 8, 4])

    def stim(self, s, c):
        return np.maximum(self.pats["s:" + s], self.pats["c:" + c])

    def train(self, tab, epochs):
        idx = {w: i for i, w in enumerate(self.outs)}
        for _ in range(epochs):
            for j in self.rng.permutation(len(tab)):
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
    kind, cfg, seed, v, coding, n_kc, control = args
    t0, cc = time.time(), cfg["capacity"]
    words = words_for(v, seed)
    sp = Speaker(kind, cfg, seed, words, coding, n_kc, control)
    tab = table(words)
    sp.train(tab, cfg["protocol"]["n_train"])
    step = float(np.mean([sp.say(s, c) == t for s, c, t in tab]))
    msgs = messages(words, cc["n_messages"], seed)
    exact = float(np.mean([sp.speak(m) == sentence(m) for m in msgs]))
    log.info("%s seed %d V %d %s KC %d %s: step %.3f sentence %.3f groups %d (%.0fs)", kind, seed, v,
             coding, n_kc, control, step, exact, sp.n, time.time() - t0)
    return {"seed": seed, "V": v, "coding": coding, "n_kc": n_kc, "control": control,
            "groups": sp.n, "step_acc": step, "sentence_exact": exact}


def exp_capacity(cfg: dict, seed: int) -> dict:
    """8a: every (V, coding, KC count) on the config's seeds; controls at the largest V."""
    from chatfly.zumi3 import merged

    cfg, cc = merged(cfg), cfg["capacity"]
    out = []
    for kind in cfg["simulators"]:
        jobs = [(kind, cfg, s, v, cd, nk, None) for s in cfg["seeds"] for v in cc["vocab_sizes"]
                for cd in cc["codings"] for nk in cc["n_kc"]]
        jobs += [(kind, cfg, s, max(cc["vocab_sizes"]), cd, nk, ctl) for s in cfg["seeds"]
                 for cd in cc["codings"] for nk in cc["n_kc"] for ctl in cc["controls"]]
        with ProcessPoolExecutor(cfg.get("workers", 1)) as ex:
            res = list(ex.map(_job, jobs))
        rows = []
        for key in sorted({(r["V"], r["coding"], r["n_kc"], str(r["control"])) for r in res}):
            rs = [r for r in res if (r["V"], r["coding"], r["n_kc"], str(r["control"])) == key]
            row = {"V": key[0], "coding": key[1], "n_kc": key[2], "control": key[3], "groups": rs[0]["groups"],
                   **{m: [float(np.mean([r[m] for r in rs])), float(np.min([r[m] for r in rs]))]
                      for m in ("step_acc", "sentence_exact")}}
            rows.append(row)
            log.info("%s summary %s", kind, row)
        out.append({"simulator": kind, "rows": rows, "runs": res})
    return {"capacity": out}
