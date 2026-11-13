"""Phase 4: general finite-state transition tables on one fly (capacity sweep). ADR 0008.

A task is a DFA: states, inputs, delta[state][input] -> state. As in Phase 2, the state token
and the input token are presented together as the union of their PN patterns, the output is
one-hot over one MBON group per state, and the teacher depresses every group but the target.
Phase 2 code (parity.py) is left unchanged; its helpers are reused.
"""

import copy
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chatfly.conditioning import token_patterns
from chatfly.parity import dan_for, decode
from chatfly.sim import SIMULATORS, dopamine_rule

log = logging.getLogger("chatfly")


def make_task(spec: dict, rng) -> dict:
    """spec.kind: parity | mod (n) | no_aa | random (n_states, n_inputs) | table (delta given
    as {state: {input: next}}; the first state is the start state).
    Returns {"name", "states", "inputs", "delta": {s: {x: s'}}}."""
    k = spec["kind"]
    if k in ("parity", "mod"):
        n = 2 if k == "parity" else spec["n"]
        st = [f"s{i}" for i in range(n)]
        delta = {s: {"A": st[(i + 1) % n], "B": s} for i, s in enumerate(st)}
        return {"name": f"mod{n}", "states": st, "inputs": ["A", "B"], "delta": delta}
    if k == "no_aa":  # accept iff no two consecutive A; "dead" is absorbing
        delta = {"ok": {"A": "lastA", "B": "ok"}, "lastA": {"A": "dead", "B": "ok"},
                 "dead": {"A": "dead", "B": "dead"}}
        return {"name": "no_aa", "states": list(delta), "inputs": ["A", "B"], "delta": delta}
    if k == "random":  # every next-state drawn uniformly; table re-drawn per seed
        st = [f"s{i}" for i in range(spec["n_states"])]
        xs = [f"x{j}" for j in range(spec["n_inputs"])]
        delta = {s: {x: st[rng.integers(len(st))] for x in xs} for s in st}
        return {"name": f"rand{len(st)}x{len(xs)}", "states": st, "inputs": xs, "delta": delta}
    if k == "table":
        d = spec["delta"]
        return {"name": spec.get("name", "table"), "states": list(d), "inputs": list(d[next(iter(d))]),
                "delta": {s: dict(r) for s, r in d.items()}}
    raise ValueError(k)


def transitions(task):
    return [(s, x, task["delta"][s][x]) for s in task["states"] for x in task["inputs"]]


def task_cfg(cfg, task):
    """Config for this task: one output compartment per state, one token per state / input."""
    c = copy.deepcopy(cfg)
    c["mb"]["compartments"] = list(task["states"])
    c["tokens"]["names"] = list(task["states"]) + list(task["inputs"])
    return c


def present_pair(pats, s, x):
    return np.maximum(pats[s], pats[x])


def train(sim, pats, task, cfg, rng, random_teacher=False):
    p, st, tr = cfg["protocol"], task["states"], transitions(task)
    for _ in range(p["n_train"]):
        for i in rng.permutation(len(tr)):
            s, x, t = tr[i]
            tgt = int(rng.integers(len(st))) if random_teacher else st.index(t)
            sim.present(present_pair(pats, s, x), p["duration_ms"], dan_for(tgt, sim.n_comp))


def test(sim, pats, task, cfg):
    p, st = cfg["protocol"], task["states"]
    ok = [decode(sim, sim.present(present_pair(pats, s, x), p["duration_ms"])["mbon"])[0]
          == st.index(t) for s, x, t in transitions(task) for _ in range(p["n_test"])]
    per = np.array(ok).reshape(len(transitions(task)), -1).mean(1)
    return float(np.mean(ok)), per.tolist()


def pn_direct(sim, pats, task, cfg, rng, noisy):
    """Same rule and readout on PNs directly (no KC layer); argmax, ties scored wrong."""
    p, c, st = cfg["protocol"], cfg["controls"], task["states"]
    sec = p["duration_ms"] / 1000.0
    w = np.full((len(st), sim.n_pn), sim.w0)

    def rates(stim):
        return rng.poisson(stim * sec) / sec if noisy else stim

    tr = transitions(task)
    for _ in range(p["n_train"]):
        for i in rng.permutation(len(tr)):
            s, x, t = tr[i]
            dopamine_rule(w, rates(present_pair(pats, s, x)) >= c["pn_active_hz"],
                          dan_for(st.index(t), len(st)), sim.w0, sim.eta_dep, sim.eta_rec)
    ok = []
    for s, x, t in tr:
        for _ in range(p["n_test"]):
            o = w @ rates(present_pair(pats, s, x))
            ok.append(np.argmax(o) == st.index(t) and np.sort(o)[-1] > np.sort(o)[-2])
    return float(np.mean(ok))


def run_loop(sim, pats, task, cfg, rng, length):
    """Closed loop from the first state; decoded state fed back; each step from rest."""
    dur, st, xs = cfg["protocol"]["duration_ms"], task["states"], task["inputs"]
    true_s = fed = st[0]
    step_ok, track_ok = [], []
    for _ in range(length):
        x = xs[rng.integers(len(xs))]
        d, _ = decode(sim, sim.present(present_pair(pats, fed, x), dur)["mbon"])
        new = st[d] if d is not None else None
        step_ok.append(new == task["delta"][fed][x])
        true_s = task["delta"][true_s][x]
        track_ok.append(new == true_s)
        fed = new if new is not None else fed
    return float(np.mean(step_ok)), bool(all(track_ok))


def _job(args):
    kind, cfg, spec, s = args
    rs = cfg["rng_streams"]
    task = make_task(spec, np.random.default_rng(s + rs["table"]))
    c = task_cfg(cfg, task)
    rng = np.random.default_rng(s + rs["task"])
    pats = token_patterns(c, np.random.default_rng(s + rs["tokens"]))
    sim = SIMULATORS[kind](c, s, noise_seed=s * rs["noise"])
    train(sim, pats, task, c, rng)
    acc, per = test(sim, pats, task, c)
    out = {"seed": s, "task": task["name"], "step_acc": acc, "per_transition": per,
           "delta": task["delta"]}
    L = cfg["loop"]
    loops = [run_loop(sim, pats, task, c, rng, L["length"]) for _ in range(L["n_seq"])]
    out["loop_step_acc"] = float(np.mean([a for a, _ in loops]))
    out["loop_seq_ok"] = float(np.mean([b for _, b in loops]))
    out["pn_direct_rule"] = pn_direct(sim, pats, task, c, rng, noisy=kind == "spiking")
    s2 = SIMULATORS[kind](c, s, noise_seed=s * rs["noise"] + 1)
    train(s2, pats, task, c, rng, random_teacher=True)
    out["teacher_shuffle"] = test(s2, pats, task, c)[0]
    return out


def _warm(kind, cfg, spec, arch="single", k=1):
    """Compile once in the parent so worker processes only hit Brian2's code cache."""
    task = make_task(spec, np.random.default_rng(0))
    c = task_cfg(cfg, task)
    for sim in Flies(kind, c, task, cfg["seeds"][0], arch, k).sims:
        if hasattr(sim, "n_kc"):
            sim.present(np.zeros(sim.n_pn), cfg["warmup_ms"])


def exp_fsm_sweep(cfg: dict, seed: int) -> dict:
    """Every task in cfg["tasks"] x cfg["seeds"], for each simulator."""
    out = []
    tgt = cfg["gate"]["step_acc_target"]
    for kind in cfg["simulators"]:
        for spec in cfg["tasks"]:
            jobs = [(kind, cfg, spec, s) for s in cfg["seeds"]]
            if kind == "spiking" and cfg.get("workers", 1) > 1:
                _warm(kind, cfg, spec)
                with ProcessPoolExecutor(cfg["workers"]) as ex:
                    rows = list(ex.map(_job, jobs))
            else:
                rows = [_job(j) for j in jobs]
            acc = np.array([r["step_acc"] for r in rows])
            res = {"simulator": kind, "spec": spec, "task": rows[0]["task"],
                   "mean": float(acc.mean()), "min": float(acc.min()),
                   "frac_at_target": float(np.mean(acc >= tgt)),
                   "loop_seq_ok": float(np.mean([r["loop_seq_ok"] for r in rows])),
                   "pn_direct": float(np.mean([r["pn_direct_rule"] for r in rows])),
                   "teacher_shuffle": float(np.mean([r["teacher_shuffle"] for r in rows])),
                   "seeds": rows}
            out.append(res)
            log.info("%s %s: step %.3f (min %.3f, at-target %.1f) loop-seq %.2f pn-direct %.3f "
                     "teacher-shuffle %.3f", kind, res["task"], res["mean"], res["min"],
                     res["frac_at_target"], res["loop_seq_ok"], res["pn_direct"],
                     res["teacher_shuffle"])
    return {"runs": out}


# ---- Phase 4 step 2: several flies (ADR 0008) ----

def group_rates(sim, mbon):
    return np.array([mbon[sim.mbon_comp == c].mean() for c in range(sim.n_comp)])


def argmax_or_none(r):
    o = np.sort(r)[::-1]
    return None if r.sum() <= 0 or o[0] == o[1] else int(np.argmax(r))


class PNDirect:
    """Learning rule + one-hot readout applied to PN rates (no KC layer), Simulator-like API.
    noisy: Poisson PN counts, matching the spiking model's input."""

    def __init__(self, sim, cfg, n_comp, rng, noisy):
        self.w, self.n_comp = np.full((n_comp, sim.n_pn), sim.w0), n_comp
        self.mbon_comp = np.arange(n_comp)
        self.sim, self.rng, self.noisy = sim, rng, noisy
        self.sec, self.thr = cfg["protocol"]["duration_ms"] / 1000.0, cfg["controls"]["pn_active_hz"]

    def present(self, stim, dur, dan=None):
        r = self.rng.poisson(stim * self.sec) / self.sec if self.noisy else stim
        if dan is not None:
            s = self.sim
            dopamine_rule(self.w, r >= self.thr, dan, s.w0, s.eta_dep, s.eta_rec)
        return {"mbon": self.w @ r}


class WeightSlot:
    """One fly's KC layer with its own KC->MBON weight matrix: swapped in for each present."""

    def __init__(self, sim):
        self.sim, self.w = sim, np.full_like(sim.w, sim.w0)
        self.mbon_comp, self.n_comp, self.n_kc, self.n_pn = sim.mbon_comp, sim.n_comp, sim.n_kc, sim.n_pn

    def present(self, stim, dur, dan=None):
        self.sim.w = self.w  # learn() updates this array in place
        return self.sim.present(stim, dur, dan)


class Flies:
    """arch: "single" (one fly), "big" (one fly with k x n_kc KCs: size-matched control),
    "ensemble" (k flies, different wiring, same teaching; group rates summed: fixed rule),
    "routed" (one fly per input token; the input selects the fly: fixed routing rule),
    "routed_pn" (control: routed, each specialist is the rule on PNs without a KC layer),
    "routed_shared" (control: routed, one fly whose KC->MBON weights are a separate set per
    input: the same router glue with a single KC layer)."""

    def __init__(self, kind, cfg, task, seed, arch, k):
        rs = cfg["rng_streams"]
        if arch == "big":  # k x KCs; spiking APL and KC->MBON drives are per-spike sums, so
            cfg = copy.deepcopy(cfg)  # scale them by 1/k to keep the operating regime (the rate
            cfg["mb"]["n_kc"] *= k    # model already normalizes by n_kc)
            cfg["spiking"]["w_kc_mbon_mv"] /= k
            cfg["spiking"]["g_apl_mv"] /= k
        n = len(task["inputs"]) if arch == "routed" else (k if arch == "ensemble" else 1)
        n = 1 if arch in ("routed_pn", "routed_shared") else n
        # fly i gets its own wiring seed; fly 0 is the single fly of the same seed
        self.sims = [SIMULATORS[kind](cfg, seed + i * rs["fly"], noise_seed=seed * rs["noise"] + i)
                     for i in range(n)]
        self.arch, self.inputs, self.dur = arch, task["inputs"], cfg["protocol"]["duration_ms"]
        if arch == "routed_pn":  # control: routed, but each "fly" is the rule on PNs (no KCs)
            self.sims = [PNDirect(self.sims[0], cfg, len(task["states"]),
                                  np.random.default_rng([seed, rs["fly"], i]), kind == "spiking")
                         for i in range(len(task["inputs"]))]
        if arch == "routed_shared":  # control: routed, ONE fly (one KC layer) with a separate
            self.sims = [WeightSlot(self.sims[0]) for _ in task["inputs"]]  # KC->MBON w per input

    def _which(self, x):
        return [self.sims[self.inputs.index(x)]] if self.arch.startswith("routed") else self.sims

    def teach(self, stim, x, tgt):
        for sim in self._which(x):
            sim.present(stim, self.dur, dan_for(tgt, sim.n_comp))

    def decide(self, stim, x):
        return argmax_or_none(sum(group_rates(s, s.present(stim, self.dur)["mbon"])
                                  for s in self._which(x)))


def _multi_job(args):
    kind, cfg, spec, arch, k, s = args
    rs = cfg["rng_streams"]
    task = make_task(spec, np.random.default_rng(s + rs["table"]))
    c = task_cfg(cfg, task)
    rng = np.random.default_rng(s + rs["task"])
    pats = token_patterns(c, np.random.default_rng(s + rs["tokens"]))
    flies, st, tr, p = Flies(kind, c, task, s, arch, k), task["states"], transitions(task), c["protocol"]
    for _ in range(p["n_train"]):
        for i in rng.permutation(len(tr)):
            a, x, t = tr[i]
            flies.teach(present_pair(pats, a, x), x, st.index(t))
    ok = [flies.decide(present_pair(pats, a, x), x) == st.index(t)
          for a, x, t in tr for _ in range(p["n_test"])]
    seq_ok = []
    for _ in range(c["loop"]["n_seq"]):
        true_s = fed = st[0]
        good = True
        for _ in range(c["loop"]["length"]):
            x = task["inputs"][rng.integers(len(task["inputs"]))]
            d = flies.decide(present_pair(pats, fed, x), x)
            true_s = task["delta"][true_s][x]
            new = st[d] if d is not None else None
            good &= new == true_s
            fed = new if new is not None else fed
        seq_ok.append(good)
    return {"seed": s, "task": task["name"], "step_acc": float(np.mean(ok)),
            "loop_seq_ok": float(np.mean(seq_ok))}


def exp_multi_fly(cfg: dict, seed: int) -> dict:
    """cfg["tasks"] x cfg["archs"] ({arch, k}) x cfg["seeds"], for each simulator."""
    out, tgt = [], cfg["gate"]["step_acc_target"]
    for kind in cfg["simulators"]:
        for spec in cfg["tasks"]:
            for a in cfg["archs"]:
                jobs = [(kind, cfg, spec, a["arch"], a.get("k", 1), s) for s in cfg["seeds"]]
                if kind == "spiking" and cfg.get("workers", 1) > 1:
                    _warm(kind, cfg, spec, a["arch"], a.get("k", 1))
                    with ProcessPoolExecutor(cfg["workers"]) as ex:
                        rows = list(ex.map(_multi_job, jobs))
                else:
                    rows = [_multi_job(j) for j in jobs]
                acc = np.array([r["step_acc"] for r in rows])
                res = {"simulator": kind, "spec": spec, "arch": a, "task": rows[0]["task"],
                       "mean": float(acc.mean()), "min": float(acc.min()),
                       "frac_at_target": float(np.mean(acc >= tgt)),
                       "loop_seq_ok": float(np.mean([r["loop_seq_ok"] for r in rows])),
                       "seeds": rows}
                out.append(res)
                log.info("%s %s %s k=%d: step %.3f (min %.3f, at-target %.1f) loop-seq %.2f",
                         kind, res["task"], a["arch"], a.get("k", 1), res["mean"], res["min"],
                         res["frac_at_target"], res["loop_seq_ok"])
    return {"runs": out}


def margin_table(metrics: dict) -> dict:
    """Step-1 analysis: per-transition accuracy vs. margin = (row/column mates whose target is
    the strongest competing state) - (mates whose target is the correct state). Random tables
    only. Run: python -m chatfly.fsm results/<run_id>/metrics.json"""
    by = {}
    for run in metrics["runs"]:
        if run["spec"]["kind"] != "random":
            continue
        for sd in run["seeds"]:
            d = sd["delta"]
            st, xs = list(d), list(d[next(iter(d))])
            for i, (s, x) in enumerate((s, x) for s in st for x in xs):
                t = d[s][x]
                mates = [d[s][x2] for x2 in xs if x2 != x] + [d[s2][x] for s2 in st if s2 != s]
                m = max(mates.count(g) for g in st if g != t) - mates.count(t)
                by.setdefault(max(m, 0), []).append(sd["per_transition"][i] >= 0.5)
    return {k: {"n": len(v), "acc": float(np.mean(v))} for k, v in sorted(by.items())}


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    for k, v in margin_table(json.loads(Path(sys.argv[1]).read_text())).items():
        print(f"margin {'<=0' if k == 0 else k}: n {v['n']}, acc {v['acc']:.3f}")
