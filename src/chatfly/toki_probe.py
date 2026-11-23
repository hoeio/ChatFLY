"""Phase 9 feasibility probes (ADR 0014), selection seeds only.

probe9_memory: can a memory hold facts about any (subject phrase, predicate phrase) without
predicate routing, so that known facts are answered and unknown pairs are declined (the 8d-2 /
8e-2 / 8i failure)? Compares 8i's union smell with chained binding (toki.Binder), plus controls.
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from chatfly.toki import CONTENT, Binder, Fly, Odors

log = logging.getLogger("chatfly")


def world(rng, m):
    """Subjects with shared heads and modifiers (a conjunction is needed), predicates of 1-2 words,
    facts (S, P, polarity) with changes of mind, and the questions with their answers."""
    words = list(rng.permutation([w for w in CONTENT if w not in ("mi", "sina")]))
    heads, mods, pw = words[:m["n_heads"]], words[m["n_heads"]:m["n_heads"] + m["n_mods"]], words[40:]
    subs = [(h,) for h in heads] + [(h, x) for h in heads for x in mods]
    subs += [(h, x, y) for h in heads for x in mods for y in mods if x != y]
    subs = [subs[i] for i in rng.permutation(len(subs))]
    known, unknown = subs[:m["n_subj"]], subs[m["n_subj"]:m["n_subj"] + m["n_unknown"]]
    preds = [(w,) for w in pw[:m["n_pred"] // 2]] + [(pw[i], pw[i + 1]) for i in range(m["n_pred"] // 2, m["n_pred"], 1)]
    pairs = [(s, p) for s in known for p in preds]
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[:m["n_facts"]]]
    told, truth = [], {}
    for s, p in pairs:
        v = bool(rng.integers(2))
        told.append((s, p, v))
        truth[(s, p)] = v
    for i in rng.permutation(len(pairs))[:m["n_change"]]:  # changes of mind, told later
        s, p = pairs[i]
        truth[(s, p)] = not truth[(s, p)]
        told.append((s, p, truth[(s, p)]))
    asked = set(pairs)
    q_unknown = [(s, preds[int(rng.integers(len(preds)))]) for s in unknown]
    q_untold = [(s, p) for s in known for p in preds if (s, p) not in asked]
    q_untold = [q_untold[i] for i in rng.permutation(len(q_untold))[:len(q_unknown)]]
    return told, truth, q_unknown + q_untold


def _job(args):
    method, cfg, seed = args
    m, t0 = cfg["probe"], time.time()
    rng = np.random.default_rng([seed, 9, 1])
    told, truth, q_none = world(rng, m)
    od = Odors(cfg, seed, cfg["mb"]["n_pn"], cfg["tokens"]["pn_active_frac"])
    if method.startswith("union"):
        f = {**m[method if method in m else "union"], "n_pn": od.n_pn}
        def key(s, p):
            return np.max([od("w", w) for w in s + p], axis=0)
    else:
        f = {**m["key_memory"], "n_pn": od.n_pn}
        b = Binder(cfg, seed, od, m["binder"], "no_kc" if method == "chain_binder_no_kc" else None)
        def key(s, p):
            return b(b.phrase("subj", s), b.phrase("pred", p))
    ctl = "no_kc" if method in ("union_no_kc", "chain_memory_no_kc") else None
    mem = Fly(cfg, seed, 1, [True, False], f, ctl)
    for s, p, v in told:
        mem.teach(key(s, p), v, m["n_teach"])
    known = [(mem.read(key(s, p)), v) for (s, p), v in truth.items()]
    none = [mem.read(key(s, p)) for s, p in q_none]
    out = {"method": method, "seed": seed, "s": round(time.time() - t0, 1)}
    for thr in m["thresholds"]:
        out[f"known@{thr}"] = float(np.mean([a == v and c >= thr for (a, c), v in known]))
        out[f"declined@{thr}"] = float(np.mean([a is None or c < thr for a, c in none]))
        out[f"both@{thr}"] = min(out[f"known@{thr}"], out[f"declined@{thr}"])
    log.info("%s", out)
    return out


def with_base(cfg):
    return {**yaml.safe_load((Path(__file__).resolve().parents[2] / cfg["base"]).read_text()), **cfg}


def exp_probe9_memory(cfg: dict, seed: int) -> dict:
    cfg = with_base(cfg)
    m = cfg["probe"]
    jobs = [(meth, cfg, s) for meth in m["methods"] for s in cfg["select_seeds"]]
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        rows = list(ex.map(_job, jobs))
    summary = {}
    for meth in m["methods"]:
        rs = [r for r in rows if r["method"] == meth]
        summary[meth] = {k: [round(float(np.mean([r[k] for r in rs])), 3), round(float(np.min([r[k] for r in rs])), 3)]
                         for k in rs[0] if "@" in k}
        best = max(m["thresholds"], key=lambda t: summary[meth][f"both@{t}"][1])
        log.info("%s best threshold %s: %s", meth, best, {k: v for k, v in summary[meth].items() if k.endswith(f"@{best}")})
    return {"rows": rows, "summary": summary}
