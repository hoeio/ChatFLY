"""Phase 5 / Gate E: a fixed-schema chat layer over a trained fly (ADR 0009).

Natural language -> (token sequence, question) by a fixed rule-based parser; the fly runs the
sequence in its closed loop (fsm.run's feedback: decoded state fed back as the next state token);
the reply is rendered only from the fly's final state and its confidence. The chat layer never
evaluates the task's transition table: ground truth exists only in the evaluation code.

Interactive: python -m chatfly.chat configs/phase5_chat.yaml [--sim rate|spiking] [--seed N]
"""

import argparse
import logging
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chatfly.conditioning import token_patterns
from chatfly.fsm import _warm, make_task, present_pair, task_cfg, train
from chatfly.parity import decode
from chatfly.sim import SIMULATORS

log = logging.getLogger("chatfly")
# a run of all-caps words (A B A, ABBA, "A, B"), not glued to letters or an apostrophe (A's)
RUN = re.compile(r"(?<![\w'])[A-Z]+(?:[ ,]+[A-Z]+)*(?![\w'])")


def parse(text: str, inputs: list[str], keywords: list[str], negations: list[str]) -> dict:
    """{"tokens": [...], "ask": keyword | None (= "which?")} or {"refuse": reason}.
    Refuses rather than guesses: zero or several sequences, unknown tokens, negation (the parser
    has no logic to flip a question; ADR 0009), no keyword."""
    low = text.lower()
    # whole words ("not"), or anywhere for non-words ("n't" in "isn't")
    if any(re.search(rf"\b{n}\b" if n.isalpha() else re.escape(n), low) for n in negations):
        return {"refuse": "negation"}
    runs = [r for r in RUN.findall(text) if r != "I"]
    if len(runs) != 1:
        return {"refuse": "no_sequence" if not runs else "ambiguous_sequence"}
    toks = list(re.sub(r"[ ,]", "", runs[0]))
    unknown = sorted(set(toks) - set(inputs))
    if unknown:
        return {"refuse": "unknown_tokens", "unknown": unknown}
    words = set(re.findall(r"[a-z]+", low))
    asked = [k for k in keywords if k in words]
    if not asked:
        return {"refuse": "no_question"}
    return {"tokens": toks, "ask": asked[0] if len(asked) == 1 else None}


def ask_fly(sim, pats, task, dur, tokens, trace=None, start=None) -> tuple[str | None, float]:
    """Run the tokens through the closed loop from start (default: the task's start state). Returns (final state,
    lowest per-step confidence); (None, 0.0) if any step decodes as uncertain. trace (a list),
    if given, gets one (input, decoded state or None, confidence) per step."""
    s, low = start or task["states"][0], 1.0
    for x in tokens:
        d, c = decode(sim, sim.present(present_pair(pats, s, x), dur)["mbon"])
        if trace is not None:
            trace.append((x, None if d is None else task["states"][d], c))
        if d is None:
            return None, 0.0
        s, low = task["states"][d], min(low, c)
    return s, low


def answer(final, conf, ask, questions, min_conf) -> str:
    """"yes" | "no" | a keyword (for "which?") | "uncertain"."""
    if final is None or conf < min_conf:
        return "uncertain"
    if ask is None:
        return next(k for k, acc in questions.items() if final in acc)
    return "yes" if final in questions[ask] else "no"


def reply(parsed, ans, conf, ch) -> str:
    if "refuse" in parsed:
        why = {"no_sequence": "I found no sequence of the fly's tokens",
               "ambiguous_sequence": "I found more than one sequence and will not guess which",
               "negation": "I can't handle negated questions; please ask it without 'not'",
               "unknown_tokens": f"the fly only knows {', '.join(ch['inputs'])}, not "
                                 f"{', '.join(parsed.get('unknown', []))}",
               "no_question": f"I can only ask whether a sequence is {' or '.join(ch['questions'])}"}
        return f"I can't ask the fly that: {why[parsed['refuse']]}."
    seq = " ".join(parsed["tokens"])
    if ans == "uncertain":
        return f"The fly is uncertain about {seq} (lowest step confidence {conf:.2f})."
    if parsed["ask"] is None:
        return f"The fly says {seq} is {ans} (confidence {conf:.2f})."
    return f"The fly says {ans}: {seq} is{'' if ans == 'yes' else ' not'} {parsed['ask']} " \
           f"(confidence {conf:.2f})."


def truth(task, tokens) -> str:
    """Ground truth for evaluation only (the chat path never calls this)."""
    s = task["states"][0]
    for x in tokens:
        s = task["delta"][s][x]
    return s


def fly_for(kind, cfg, seed, random_teacher=False):
    rs = cfg["rng_streams"]
    task = make_task(cfg["chat"]["task"], np.random.default_rng(seed + rs["table"]))
    c = task_cfg(cfg, task)
    pats = token_patterns(c, np.random.default_rng(seed + rs["tokens"]))
    sim = SIMULATORS[kind](c, seed, noise_seed=seed * rs["noise"] + int(random_teacher))
    train(sim, pats, task, c, np.random.default_rng(seed + rs["task"]), random_teacher)
    return sim, pats, task, c


def translation_eval(cfg, rng) -> dict:
    """Parser only (no fly): templates x random sequences, plus out-of-scope utterances.
    A wrong parse that is not a refusal ("silent") is the failure that matters."""
    ch, ev = cfg["chat"], cfg["chat"]["eval"]
    out = {}
    for name, temps in ch["templates"].items():
        n = ok = silent = 0
        for t in temps:
            for _ in range(ev["n_parse_per_template"]):
                toks = list(rng.choice(ch["inputs"], rng.integers(1, ev["max_len"] + 1)))
                seq = " ".join(toks) if rng.random() < 0.5 else "".join(toks)
                p = parse(t["text"].format(seq=seq), ch["inputs"], list(ch["questions"]),
                          ch["negations"])
                good = p.get("tokens") == toks and p.get("ask") == t["ask"]
                n, ok, silent = n + 1, ok + good, silent + (not good and "refuse" not in p)
        out[name] = {"n": n, "acc": ok / n, "silent_wrong": silent / n}
    oos = [parse(t, ch["inputs"], list(ch["questions"]), ch["negations"])
           for t in ch["out_of_scope"]]
    out["out_of_scope"] = {"n": len(oos), "refused": float(np.mean(["refuse" in p for p in oos])),
                           "answered": [t for t, p in zip(ch["out_of_scope"], oos)
                                        if "refuse" not in p]}
    return out


def _fly_job(args):
    """Fly only, on gold token sequences: (final state, min confidence, truth, length)."""
    kind, cfg, seed, random_teacher = args
    sim, pats, task, c = fly_for(kind, cfg, seed, random_teacher)
    ev, rng = cfg["chat"]["eval"], np.random.default_rng([seed, cfg["rng_streams"]["chat"]])
    rows = []
    for _ in range(ev["n_questions"]):
        toks = list(rng.choice(task["inputs"], rng.integers(1, ev["max_len"] + 1)))
        f, conf = ask_fly(sim, pats, task, c["protocol"]["duration_ms"], toks)
        rows.append({"final": f, "conf": conf, "truth": truth(task, toks), "len": len(toks)})
    return {"seed": seed, "random_teacher": random_teacher, "rows": rows}


def score(rows, thr) -> dict:
    """Coverage (answered, not uncertain) and accuracy of the answered ones at threshold thr."""
    ans = [r for r in rows if r["final"] is not None and r["conf"] >= thr]
    return {"coverage": len(ans) / len(rows),
            "answered_acc": float(np.mean([r["final"] == r["truth"] for r in ans])) if ans else None,
            "raw_acc": float(np.mean([r["final"] == r["truth"] for r in rows]))}


def pick_threshold(res, grid, target) -> float:
    """Smallest grid threshold at which every (normally taught) selection seed's answered
    accuracy reaches target; the largest grid value if none does."""
    for t in sorted(grid):
        accs = [score(r["rows"], t)["answered_acc"] for r in res if not r["random_teacher"]]
        if all(a is not None and a >= target for a in accs):
            return t
    return max(grid)


def exp_chat(cfg: dict, seed: int) -> dict:
    """Translation eval, then the fly on chat_select_seeds (threshold choice) and seeds (test),
    each with a teacher-shuffle control fly, for each simulator."""
    ch = cfg["chat"]
    out = {"translation": translation_eval(cfg, np.random.default_rng(seed)), "fly": []}
    log.info("translation %s", out["translation"])
    for kind in cfg["simulators"]:
        seeds = cfg["select_seeds"] + cfg["seeds"]
        jobs = [(kind, cfg, s, rt) for s in seeds for rt in (False, True)]
        if kind == "spiking" and cfg.get("workers", 1) > 1:
            _warm(kind, cfg, ch["task"])
            with ProcessPoolExecutor(cfg["workers"]) as ex:
                res = list(ex.map(_fly_job, jobs))
        else:
            res = [_fly_job(j) for j in jobs]
        sel = [r for r in res if r["seed"] in cfg["select_seeds"]]
        test = [r for r in res if r["seed"] in cfg["seeds"]]
        thr = pick_threshold(sel, ch["conf_grid"], cfg["gate"]["answered_acc_target"])
        for r in res:
            r["by_threshold"] = {str(t): score(r["rows"], t) for t in ch["conf_grid"]}
            r["at_threshold"] = score(r["rows"], thr)
        for r in test:
            log.info("%s seed %d%s @%.2f: %s", kind, r["seed"],
                     " (teacher shuffle)" * r["random_teacher"], thr, r["at_threshold"])
        out["fly"].append({"simulator": kind, "threshold": thr, "select": sel, "test": test})
    return out


def turn(line, fly, ch, kind) -> dict:
    """One chat turn: parse, ask the fly if in scope, reply. fly = fly_for(...)."""
    sim, pats, task, c = fly
    pr = parse(line, ch["inputs"], list(ch["questions"]), ch["negations"])
    ans, conf, trace = None, 0.0, []
    if "refuse" not in pr:
        f, conf = ask_fly(sim, pats, task, c["protocol"]["duration_ms"], pr["tokens"], trace)
        ans = answer(f, conf, pr["ask"], ch["questions"], ch["min_confidence"][kind])
    return {"parsed": pr, "answer": ans, "confidence": conf, "trace": trace,
            "start": task["states"][0], "reply": reply(pr, ans, conf, ch)}


def load(argv, desc):
    """Shared CLI for chat and web: (args, cfg, trained fly)."""
    from pathlib import Path

    import yaml

    p = argparse.ArgumentParser(description=desc)
    p.add_argument("config", type=Path)
    p.add_argument("--sim", default="rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8000)  # web only
    a = p.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    print("Training the fly...")
    if "zumi3" in cfg or "chat8d" in cfg:  # Phase 7f / 8: the pipeline trains its own flies
        from chatfly.zumi3 import merged

        return a, merged(cfg), None
    return a, cfg, fly_for(a.sim, cfg, a.seed)


def main(argv=None):
    a, cfg, fly = load(argv, "Chat with a trained fly in the terminal.")
    ch = cfg["chat"]
    print(f"Ask whether a sequence of {', '.join(ch['inputs'])} is "
          f"{' or '.join(ch['questions'])}. Empty line quits.")
    while line := input("> ").strip():
        print(turn(line, fly, ch, a.sim)["reply"])


if __name__ == "__main__":
    main()
