"""Phase 6: Zumi, a tiny language the fly speaks (ADRs 0010, 0011; guide: docs/zumi-guide.md).

Two parts of the fly:
- Ring: the fly's place on a ring (its FSM states); each move word is an input token that
  permutes the places. The fly's last decoded place is kept between turns, so its memory of
  where it is lives in the feedback loop. Trained before the conversation.
- Memory (Zumi 2): any other word is a new odor-like PN pattern. "fire ka ho" pairs that
  pattern with dopamine in the hot/cold output pair, during the conversation; questions read
  the fly's MBONs. One KC layer, one KC->MBON weight set per property pair (fsm.WeightSlot).

This module only (1) splits an utterance into words by a fixed grammar, (2) feeds words to
the fly (closed loop for moves, plain presentation for nouns), (3) renders the fly's decoded
output as a word. It never reads the transition table or stores facts; ground truth exists
only in the evaluation.

Interactive: python -m chatfly.zumi configs/phase6_zumi.yaml [--sim rate|spiking] [--seed N]
"""

import copy
import hashlib
import logging
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chatfly.chat import ask_fly, fly_for, load, pick_threshold, score
from chatfly.fsm import PNDirect, WeightSlot, _warm
from chatfly.parity import dan_for, decode
from chatfly.sim import SIMULATORS

log = logging.getLogger("chatfly")


def reserved(lx) -> set:
    """Every Zumi word: none of them can be a noun."""
    one = {lx[k] for k in ("restart", "suppose", "where", "yesno", "is", "what")}
    one |= {lx["pronoun"]} if "pronoun" in lx else set()  # Zumi 3 (ADR 0012)
    return one | set(lx["moves"]) | set(lx["places"]) | set(props(lx)) | set(lx["replies"].values())


def props(lx) -> dict:
    """property word -> (pair index, side 0/1), e.g. bon -> (0, 0), mal -> (0, 1)."""
    return {w: (i, j) for i, pair in enumerate(lx["props"]) for j, w in enumerate(pair)}


def parse_memory(words, lx) -> dict | None:
    """noun ka prop+ (teach) | noun prop ba (ask) | noun ke (what). None if the sentence is
    not of these shapes (then it is a ring sentence)."""
    if len(words) < 2:
        return None
    noun, rest, pr = words[0], words[1:], props(lx)
    if rest[0] == lx["is"]:
        out = {"noun": noun, "kind": "teach", "props": rest[1:]}
    elif rest == [lx["what"]]:
        out = {"noun": noun, "kind": "what", "props": []}
    elif len(rest) == 2 and rest[0] in pr and rest[1] == lx["yesno"]:
        out = {"noun": noun, "kind": "ask", "props": rest[:1]}
    else:
        return None
    if (noun in reserved(lx) and noun != lx.get("pronoun")) or not noun.isalpha():
        return {"refuse": "not_a_noun", "word": noun}
    if out["kind"] == "teach":
        if not out["props"]:
            return {"refuse": "ka_needs_property"}
        bad = next((w for w in out["props"] if w not in pr), None)
        if bad:
            return {"refuse": "unknown_word", "word": bad}
        pairs = [pr[w][0] for w in out["props"]]
        if len(set(pairs)) < len(pairs):  # "bon mal", or "bon bon"
            return {"refuse": "contradiction"}
        if lx.get("prop_order") == "strict" and pairs != sorted(pairs):  # Zumi 3 (ADR 0012)
            return {"refuse": "property_order"}
    return out


def parse(text: str, lx: dict) -> dict:
    """Grammar: noun ka prop+ | noun prop ba | noun ke | [restart] [suppose] move* [where |
    place yesno]. Ring sentences return {"restart", "suppose", "moves", "ask"} with ask None
    (statement), "where" or a place; memory sentences {"noun", "kind", "props"};
    or {"refuse": reason, "word"?}. Never guesses."""
    words = re.findall(r"[^\s.,!?]+", text.lower())
    mem = parse_memory(words, lx)
    if mem is not None:
        return mem
    out = {"restart": False, "suppose": False, "moves": [], "ask": None}
    if words[:1] == [lx["restart"]]:
        out["restart"], words = True, words[1:]
    if words[:1] == [lx["suppose"]]:
        out["suppose"], words = True, words[1:]
    while words and words[0] in lx["moves"]:
        out["moves"].append(words.pop(0))
    if words == [lx["where"]]:
        out["ask"] = "where"
    elif len(words) == 2 and words[0] in lx["places"] and words[1] == lx["yesno"]:
        out["ask"] = words[0]
    elif words:
        known = {lx["restart"], lx["suppose"], lx["where"], lx["yesno"], *lx["moves"], *lx["places"]}
        bad = next((w for w in words if w not in known), None)
        return {"refuse": "unknown_word", "word": bad} if bad else {"refuse": "word_order"}
    if out["suppose"] and out["ask"] is None:
        return {"refuse": "suppose_needs_question"}
    if not (out["restart"] or out["moves"] or out["ask"]):
        return {"refuse": "empty"}
    return out


class Memory:
    """The fly's associative memory for nouns. One KC layer; one KC->MBON weight set with two
    output groups per property pair (fsm.WeightSlot): an "output pair" per property, like
    separate MB compartments. Teaching "X ka ho" presents X's pattern with the DAN on in the
    opposite group (cold), so X's KCs lose their drive to it. Asking presents X without dopamine
    (no weight change) and decodes the pair. control: None | "no_kc" (the same rule on PNs) |
    "shuffled_teacher" (each teaching uses a random side)."""

    def __init__(self, kind, cfg, seed, control=None):
        c = copy.deepcopy(cfg)
        c["mb"]["compartments"], c["tokens"]["names"] = ["first", "second"], []
        mem, rs = cfg["lang"]["memory"], cfg["rng_streams"]
        if "w_kc_mbon_mv" in mem:  # optional memory-only override of the spiking KC->MBON weight
            c["spiking"]["w_kc_mbon_mv"] = mem["w_kc_mbon_mv"]
        sim = SIMULATORS[kind](c, seed, noise_seed=seed * rs["noise"] + 2)
        n = len(cfg["lang"]["props"])
        if control == "no_kc":
            self.slots = [PNDirect(sim, c, 2, np.random.default_rng([seed, rs["words"], i]),
                                   kind == "spiking") for i in range(n)]
        else:
            self.slots = [WeightSlot(sim) for _ in range(n)]
        self.c, self.seed, self.stream, self.pats = c, seed, rs["words"], {}
        self.dur, self.n_teach, self.n_ask = c["protocol"]["duration_ms"], mem["n_teach"], mem["n_ask"]
        self.prop, self.shuffled = props(cfg["lang"]), control == "shuffled_teacher"
        self.rng = np.random.default_rng([seed, rs["words"]])

    def pattern(self, word):
        """A fixed random PN pattern per word and fly (like an odor): same rule as
        conditioning.token_patterns, seeded by the word's hash."""
        if word not in self.pats:
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
            rng, t, n = np.random.default_rng([self.seed, self.stream, h]), self.c["tokens"], self.slots[0].sim.n_pn
            r = np.full(n, t["pn_spont_hz"])
            r[rng.choice(n, round(t["pn_active_frac"] * n), replace=False)] = t["pn_on_hz"]
            self.pats[word] = r
        return self.pats[word]

    def teach(self, noun, prop):
        pair, side = self.prop[prop]
        side = int(self.rng.integers(2)) if self.shuffled else side
        for _ in range(self.n_teach):
            self.slots[pair].present(self.pattern(noun), self.dur, dan_for(side, 2))

    def ask(self, noun, pair) -> tuple[int | None, float]:
        """(side or None, confidence) from MBON rates summed over n_ask sniffs; no plasticity."""
        s = self.slots[pair]
        return decode(s, sum(s.present(self.pattern(noun), self.dur)["mbon"] for _ in range(self.n_ask)))


class Conversation:
    """One conversation with a trained fly. place = the fly's last decoded place (None = lost:
    a committed step was undecodable or below threshold, so later answers would be guesses).
    memory: a Memory (starts empty and learns during the conversation), or None."""

    def __init__(self, fly, lx, min_conf, memory=None, mem_conf=0.0):
        self.sim, self.pats, self.task, c = fly
        self.dur, self.lx, self.min_conf = c["protocol"]["duration_ms"], lx, min_conf
        self.place, self.low = self.task["states"][0], 1.0
        self.memory, self.mem_conf = memory, mem_conf

    def _memory_turn(self, p, out) -> dict:
        lx, r, m, noun = self.lx, self.lx["replies"], self.memory, p["noun"]
        gloss = {w: g for pair in lx["props"] for w, g in pair.items()}
        if p["kind"] == "teach":
            for w in p["props"]:
                m.teach(noun, w)
            return {**out, "reply": r["ok"],
                    "gloss": f"Okay: {noun} is {', '.join(gloss[w] for w in p['props'])}."}
        pairs = [m.prop[p["props"][0]][0]] if p["kind"] == "ask" else range(len(lx["props"]))
        known = []
        for i in pairs:
            side, conf = m.ask(noun, i)
            word = list(lx["props"][i])[side] if side is not None and conf >= self.mem_conf else None
            out["memory"].append([list(lx["props"][i]), word, conf])
            if word:
                known.append(word)
        if p["kind"] == "what":
            if not known:
                return {**out, "reply": r["unsure"], "gloss": f"I don't know anything about {noun} yet."}
            return {**out, "reply": " ".join(known), "gloss": f"{noun.capitalize()} is "
                    f"{', '.join(gloss[w] for w in known)}."}
        if not known:
            return {**out, "reply": r["unsure"],
                    "gloss": f"I don't know whether {noun} is {gloss[p['props'][0]]}."}
        yes = known[0] == p["props"][0]
        return {**out, "reply": r["yes"] if yes else r["no"],
                "gloss": f"{'Yes' if yes else 'No'}, {noun} is {gloss[known[0]]}."}

    def turn(self, text: str) -> dict:
        """-> {"parsed", "reply" (Zumi), "gloss" (English), "final", "conf", "trace"}."""
        lx, r = self.lx, self.lx["replies"]
        p = parse(text, lx)
        out = {"parsed": p, "final": None, "conf": 0.0, "trace": []}
        if "refuse" in p:
            why = {"unknown_word": f"I don't know the word '{p.get('word')}'",
                   "word_order": "the words are in the wrong order",
                   "suppose_needs_question": f"'{lx['suppose']}' needs a question at the end",
                   "empty": "say something",
                   "not_a_noun": f"'{p.get('word')}' is a Zumi word, not a thing",
                   "ka_needs_property": f"'{lx['is']}' needs a property after it",
                   "contradiction": "a thing can't be both sides of one pair",
                   "property_order": "properties go in pair order: bon/mal gu/pi ho/ku ra/lo"}[p["refuse"]]
            return {**out, "reply": r["huh"], "gloss": f"Huh? ({why}.)"}
        if "noun" in p:
            if self.memory is None:
                return {**out, "reply": r["huh"], "gloss": "Huh? (This fly has no memory part.)"}
            return self._memory_turn(p, {**out, "memory": []})
        if p["restart"]:
            self.place, self.low = self.task["states"][0], 1.0
        out["start"] = self.place
        if self.place is None:
            return {**out, "reply": r["unsure"],
                    "gloss": f"I lost track of where I am. Say '{lx['restart']}' to start over."}
        final, conf = ask_fly(self.sim, self.pats, self.task, self.dur, p["moves"], out["trace"],
                              start=self.place)
        conf = min(conf, self.low)  # an earlier shaky step makes every later answer shaky
        sure = final is not None and conf >= self.min_conf
        if not p["suppose"]:
            self.place, self.low = (final, conf) if sure else (None, 0.0)
        out.update(final=final if sure else None, conf=conf)
        if not sure:
            return {**out, "reply": r["unsure"], "gloss": "I'm not sure (I lost track)."}
        here = final  # the fly's states are named by the place words
        if p["ask"] is None:
            return {**out, "reply": r["ok"], "gloss": "Okay."}
        if p["ask"] == "where":
            return {**out, "reply": here, "gloss": f"{'I would be' if p['suppose'] else 'I am'} "
                                                   f"at the {lx['places'][here]}."}
        yes = here == p["ask"]
        return {**out, "reply": r["yes"] if yes else r["no"], "gloss": "Yes." if yes else "No."}


def random_utterance(lx, rng, ev) -> str:
    """A grammatical Zumi utterance; for the evaluation only."""
    moves = list(rng.choice(list(lx["moves"]), rng.integers(0, ev["max_moves"] + 1)))
    kind = rng.choice(["statement", "where", "yesno"], p=ev["p_question"])
    if kind == "statement":
        moves = moves or [next(iter(lx["moves"]))]
        return " ".join(moves) + "."
    ask = [lx["where"]] if kind == "where" else [rng.choice(list(lx["places"])), lx["yesno"]]
    sup = [lx["suppose"]] if rng.random() < ev["p_suppose"] else []
    return " ".join(sup + moves + ask) + "?"


def _conv_job(args):
    """Conversations with one fly: one row per question turn (final, conf, truth)."""
    kind, cfg, seed, random_teacher = args
    fly = fly_for(kind, cfg, seed, random_teacher)
    task, lx, ev = fly[2], cfg["lang"], cfg["lang"]["eval"]
    rng = np.random.default_rng([seed, cfg["rng_streams"]["chat"]])
    rows = []
    for c in range(ev["n_conversations"]):
        conv, truth = Conversation(fly, lx, min_conf=0.0), task["states"][0]
        for t in range(ev["turns"]):
            text = random_utterance(lx, rng, ev)
            p = parse(text, lx)
            end = truth
            for w in p["moves"]:
                end = task["delta"][end][w]
            o = conv.turn(text)
            if p["ask"] is not None:
                rows.append({"final": o["final"], "conf": o["conf"], "truth": end, "conv": c,
                             "turn": t, "text": text})
            truth = truth if p["suppose"] else end
    return {"seed": seed, "random_teacher": random_teacher, "rows": rows}


def exp_zumi(cfg: dict, seed: int) -> dict:
    """Conversations on select_seeds (threshold choice) and seeds (test), each with a
    teacher-shuffle control fly, for each simulator. Transition-level accuracy and the no-KC
    control come from the fsm_sweep config (phase6_zumi_fsm.yaml)."""
    out, grid, tgt = [], cfg["lang"]["conf_grid"], cfg["gate"]["answered_acc_target"]
    for kind in cfg["simulators"]:
        jobs = [(kind, cfg, s, rt) for s in cfg["select_seeds"] + cfg["seeds"] for rt in (False, True)]
        if kind == "spiking" and cfg.get("workers", 1) > 1:
            _warm(kind, cfg, cfg["chat"]["task"])
            with ProcessPoolExecutor(cfg["workers"]) as ex:
                res = list(ex.map(_conv_job, jobs))
        else:
            res = [_conv_job(j) for j in jobs]
        sel = [r for r in res if r["seed"] in cfg["select_seeds"]]
        test = [r for r in res if r["seed"] in cfg["seeds"]]
        thr = pick_threshold(sel, grid, tgt)
        for r in res:
            r["by_threshold"] = {str(t): score(r["rows"], t) for t in grid}
            r["at_threshold"] = score(r["rows"], thr)
            ok = {}
            for x in r["rows"]:  # a conversation is correct if every question was answered right
                ok[x["conv"]] = ok.get(x["conv"], True) and x["final"] == x["truth"] and x["conf"] >= thr
            r["conversations_all_correct"] = float(np.mean(list(ok.values())))
            if r["random_teacher"]:
                del r["rows"]  # keep the file small
        for r in test:
            log.info("%s seed %d%s @%.2f: %s conv-ok %.2f", kind, r["seed"],
                     " (teacher shuffle)" * r["random_teacher"], thr, r["at_threshold"],
                     r["conversations_all_correct"])
        out.append({"simulator": kind, "threshold": thr, "select": sel, "test": test})
    return {"fly": out}


def _memory_job(args):
    """Teach a fresh memory N nouns, then ask every (noun, pair) and some never-heard nouns.
    Returns per taught fact [correct, conf] and per untaught question [answered side?, conf]."""
    kind, cfg, seed, n, control = args
    ev, rs = cfg["memory_eval"], cfg["rng_streams"]
    m = Memory(kind, cfg, seed, control)
    rng = np.random.default_rng([seed, rs["chat"], n])
    letters = list("abcdefghijklmnopqrstuvwxyz")
    word = lambda: "".join(rng.choice(letters, ev["word_len"]))
    nouns, n_pairs = [word() for _ in range(n)], len(cfg["lang"]["props"])
    events = [(w, int(i), int(rng.integers(2))) for w in nouns
              for i in rng.choice(n_pairs, ev["props_per_noun"], replace=False)]
    rng.shuffle(events)
    later = [(w, i, 1 - v) for w, i, v in events[:round(ev["p_change"] * len(events))]]  # changed minds
    truth = {}
    for w, i, v in events + later:
        m.teach(w, list(cfg["lang"]["props"][i])[v])
        truth[(w, i)] = v
    taught, untaught = [], []
    for w in nouns + [word() for _ in range(ev["n_unknown"])]:
        for i in range(n_pairs):
            side, conf = m.ask(w, i)
            if (w, i) in truth:
                taught.append([side == truth[(w, i)], conf])
            else:
                untaught.append([side is not None, conf])
    return {"seed": seed, "n_nouns": n, "control": control, "taught": taught, "untaught": untaught}


def memory_score(r, thr) -> dict:
    ans = [ok for ok, c in r["taught"] if c >= thr]
    return {"coverage": len(ans) / len(r["taught"]),
            "answered_acc": float(np.mean(ans)) if ans else None,
            "untaught_answered": float(np.mean([a and c >= thr for a, c in r["untaught"]]))}


def exp_zumi_memory(cfg: dict, seed: int) -> dict:
    """Memory capacity (ADR 0011). cfg["base"] is the Zumi config; this config adds
    memory_eval. Threshold: smallest grid value at which every selection seed (n_select
    nouns, no control) has answered acc >= target and untaught answered rate <= limit."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    cfg = {**yaml.safe_load((root / cfg["base"]).read_text()), **cfg}
    ev, out = cfg["memory_eval"], []
    for kind in cfg["simulators"]:
        sizes = ev["n_nouns"][kind]
        jobs = [(kind, cfg, s, ev["n_select"], None) for s in cfg["select_seeds"]]
        jobs += [(kind, cfg, s, n, ctl) for s in cfg["seeds"] for n in sizes for ctl in ev["controls"]]
        if kind == "spiking" and cfg.get("workers", 1) > 1:
            Memory(kind, cfg, 0).ask("warmup", 0)  # compile once in the parent
            with ProcessPoolExecutor(cfg["workers"]) as ex:
                res = list(ex.map(_memory_job, jobs))
        else:
            res = [_memory_job(j) for j in jobs]
        sel, test = res[:len(cfg["select_seeds"])], res[len(cfg["select_seeds"]):]
        thr = max(ev["conf_grid"])
        for t in sorted(ev["conf_grid"]):
            sc = [memory_score(r, t) for r in sel]
            if all(x["answered_acc"] is not None and x["answered_acc"] >= ev["answered_acc_target"]
                   and x["untaught_answered"] <= ev["max_untaught_answered"] for x in sc):
                thr = t
                break
        rows = []
        for n in sizes:
            for ctl in ev["controls"]:
                sc = [memory_score(r, thr) for r in test if r["n_nouns"] == n and r["control"] == ctl]
                acc = [x["answered_acc"] if x["answered_acc"] is not None else 0.0 for x in sc]
                row = {"n_nouns": n, "control": ctl, "answered_acc_mean": float(np.mean(acc)),
                       "answered_acc_min": float(np.min(acc)),
                       "seeds_at_target": int(sum(a >= ev["answered_acc_target"] for a in acc)),
                       "coverage": float(np.mean([x["coverage"] for x in sc])),
                       "untaught_answered": float(np.mean([x["untaught_answered"] for x in sc]))}
                rows.append(row)
                log.info("%s @%.2f %s", kind, thr, row)
        for r in res:
            r["at_threshold"] = memory_score(r, thr)
            del r["taught"], r["untaught"]
        out.append({"simulator": kind, "threshold": thr, "rows": rows, "select": sel, "test": test})
    return {"memory": out}


def conversation(a, cfg, fly) -> Conversation:
    """A conversation with the trained ring fly and a fresh memory (CLI and web)."""
    lx = cfg["lang"]
    return Conversation(fly, lx, lx["min_confidence"][a.sim], Memory(a.sim, cfg, a.seed),
                        lx["memory"]["min_confidence"][a.sim])


def main(argv=None):
    a, cfg, fly = load(argv, "Talk to a trained fly in Zumi.")
    conv = conversation(a, cfg, fly)
    print("Speak Zumi (guide: docs/zumi-guide.md). Try 'zu nu?' or 'fire ka mal ho.' then "
          "'fire ke?'. Empty line quits.")
    try:
        while line := input("> ").strip():
            o = conv.turn(line)
            print(f"fly: {o['reply']}    ({o['gloss']})")
    except EOFError:  # end of piped input
        pass


if __name__ == "__main__":
    main()
