"""Phase 6: Zumi, a tiny language the fly speaks (ADR 0010, guide: docs/zumi-guide.md).

The fly's world is a ring of places (its FSM states); each move word is one of its input
tokens and permutes the places. A conversation keeps the fly's last decoded place between
turns, so the fly's memory of where it is lives in the feedback loop, not in this module.

This module only (1) splits an utterance into words by a fixed grammar, (2) feeds the move
words to the fly's closed loop (chat.ask_fly), (3) renders the fly's final place as a word.
It never reads the transition table; that exists only for training and in the evaluation.

Interactive: python -m chatfly.zumi configs/phase6_zumi.yaml [--sim rate|spiking] [--seed N]
"""

import logging
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chatfly.chat import ask_fly, fly_for, load, pick_threshold, score
from chatfly.fsm import _warm

log = logging.getLogger("chatfly")


def parse(text: str, lx: dict) -> dict:
    """Grammar: [restart] [suppose] move* [where | place yesno]. Returns
    {"restart", "suppose", "moves", "ask"} with ask None (statement), "where" or a place;
    or {"refuse": reason, "word"?}. Never guesses."""
    words = re.findall(r"[^\s.,!?]+", text.lower())
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


class Conversation:
    """One conversation with a trained fly. place = the fly's last decoded place (None = lost:
    a committed step was undecodable or below threshold, so later answers would be guesses)."""

    def __init__(self, fly, lx, min_conf):
        self.sim, self.pats, self.task, c = fly
        self.dur, self.lx, self.min_conf = c["protocol"]["duration_ms"], lx, min_conf
        self.place, self.low = self.task["states"][0], 1.0

    def turn(self, text: str) -> dict:
        """-> {"parsed", "reply" (Zumi), "gloss" (English), "final", "conf", "trace"}."""
        lx, r = self.lx, self.lx["replies"]
        p = parse(text, lx)
        out = {"parsed": p, "final": None, "conf": 0.0, "trace": []}
        if "refuse" in p:
            why = {"unknown_word": f"I don't know the word '{p.get('word')}'",
                   "word_order": "the words are in the wrong order",
                   "suppose_needs_question": f"'{lx['suppose']}' needs a question at the end",
                   "empty": "say something"}[p["refuse"]]
            return {**out, "reply": r["huh"], "gloss": f"Huh? ({why}.)"}
        if p["restart"]:
            self.place, self.low = self.task["states"][0], 1.0
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


def main(argv=None):
    a, cfg, fly = load(argv, "Talk to a trained fly in Zumi.")
    lx = cfg["lang"]
    conv = Conversation(fly, lx, lx["min_confidence"][a.sim])
    print("Speak Zumi (guide: docs/zumi-guide.md). Example: 'zu ki zu nu?'. Empty line quits.")
    while line := input("> ").strip():
        o = conv.turn(line)
        print(f"fly: {o['reply']}    ({o['gloss']})")


if __name__ == "__main__":
    main()
