"""Phase 2 / Gate B: order discrimination (2a), parity transition table (2b), closed loop
(2c), and noise / step-spacing robustness (2d). ADR 0002.

The state token S and the input token x are presented together as the union (element-wise
max) of their PN patterns. Output compartments are one-hot states [even, odd]. Teaching
(ADR 0004 rule): DAN depresses every compartment except the target one.
"""

import copy
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chatfly.conditioning import token_patterns
from chatfly.sim import SIMULATORS, dopamine_rule

log = logging.getLogger("chatfly")
STATES = ["even", "odd"]


def delta(s: str, x: str) -> str:
    """Parity transition: A flips the state, B keeps it."""
    return s if x == "B" else STATES[1 - STATES.index(s)]


TRANS = [(s, x, delta(s, x)) for s in STATES for x in ["A", "B"]]


def pair(pats, s, x):
    return np.maximum(pats[s], pats[x])


def dan_for(target: int, n_comp: int) -> np.ndarray:
    """Teacher: DAN on (1) in every compartment except the target (0); levels definitional."""
    d = np.ones(n_comp)
    d[target] = 0.0
    return d


def decode(sim, mbon) -> tuple[int | None, float]:
    """argmax compartment-mean MBON rate; confidence = (top - second) / sum.
    None ("uncertain", scored as wrong) if all groups are silent or the top two tie."""
    comp = np.array([mbon[sim.mbon_comp == c].mean() for c in range(sim.n_comp)])
    o = np.sort(comp)[::-1]
    if comp.sum() <= 0 or o[0] == o[1]:
        return None, 0.0
    return int(np.argmax(comp)), float((o[0] - o[1]) / comp.sum())


def make_sim(kind, cfg, seed, k=0, wiring="random", g_apl_zero=False):
    """Simulator with the seed's wiring and its own noise stream k (independent noise)."""
    cfg = copy.deepcopy(cfg)
    if g_apl_zero:
        cfg["rate"]["g_apl"] = 0.0
        cfg["spiking"]["g_apl_mv"] = 0.0
    if wiring == "shuffled":  # degree-preserving PN->KC shuffle (sim.shuffle_edges)
        cfg["mb"]["wiring"] = "shuffle"
    return SIMULATORS[kind](cfg, seed, noise_seed=seed * cfg["rng_streams"]["noise"] + k)


def train_table(sim, pats, cfg, rng, random_teacher=False):
    """protocol.teach = "always" (default; ADR 0004 as used for Gate B) or "on_error":
    exploratory variant where the teacher fires only if the output is wrong or its
    confidence is below protocol.teach_margin (a stand-in for MBON->DAN error feedback)."""
    p = cfg["protocol"]
    on_error = p.get("teach", "always") == "on_error"
    for _ in range(p["n_train"]):
        for i in rng.permutation(len(TRANS)):
            s, x, t = TRANS[i]
            tgt = int(rng.integers(len(STATES))) if random_teacher else STATES.index(t)
            dan = dan_for(tgt, sim.n_comp)
            if not on_error:
                sim.present(pair(pats, s, x), p["duration_ms"], dan)
                continue
            out = sim.present(pair(pats, s, x), p["duration_ms"])
            d, c = decode(sim, out["mbon"])
            if d != tgt or c < p["teach_margin"]:
                sim.learn(out["kc"], dan)


def test_table(sim, pats, cfg):
    """Per-step accuracy on isolated (S, x) presentations; also returns PN/KC counts + labels."""
    p = cfg["protocol"]
    ok, conf, X_pn, X_kc, y = [], [], [], [], []
    for s, x, t in TRANS:
        for _ in range(p["n_test"]):
            out = sim.present(pair(pats, s, x), p["duration_ms"])
            d, c = decode(sim, out["mbon"])
            ok.append(d == STATES.index(t))
            conf.append(c)
            X_pn.append(out["pn"])
            X_kc.append(out["kc"])
            y.append(STATES.index(t))
    return {"acc": float(np.mean(ok)), "conf": conf, "ok": ok,
            "X_pn": np.array(X_pn), "X_kc": np.array(X_kc), "y": np.array(y)}


def linear_baseline(X, y, ridge, rng) -> float:
    """Ridge least-squares linear classifier, 2-fold split stratified per transition.
    The 'just a classifier' baseline of docs/03-validation.md."""
    idx = np.arange(len(y)).reshape(len(TRANS), -1)  # rows = transitions
    for row in idx:
        rng.shuffle(row)
    half = idx.shape[1] // 2
    tr, te = idx[:, :half].ravel(), idx[:, half:].ravel()
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    t = 2.0 * y - 1.0
    w = np.linalg.solve(Z[tr].T @ Z[tr] + ridge * np.eye(Z.shape[1]), Z[tr].T @ t[tr])
    return float(np.mean((Z[te] @ w > 0) == (t[te] > 0)))


def order_task(sim, pats, cfg, rng, seg_ms):
    """2a: AB -> class 0, BA -> class 1. The readout integrates the whole 2-segment window,
    so only order information carried over inside the network (membrane / APL state) can be
    used. Segment length is swept: carry-over can only matter when segments are short
    relative to the time constants. The rate model has no carry-over (exactly symmetric)."""
    p = cfg["order"]

    def show(a, b, dan=None):
        o1 = sim.present(pats[a], seg_ms, reset=True)
        o2 = sim.present(pats[b], seg_ms, reset=False)
        kc, mbon = (o1["kc"] + o2["kc"]) / 2, (o1["mbon"] + o2["mbon"]) / 2  # equal segments
        if dan is not None:
            sim.learn(kc, dan)
        return mbon

    seqs = [(("A", "B"), 0), (("B", "A"), 1)]
    for _ in range(p["n_train"]):
        for i in rng.permutation(2):
            (a, b), c = seqs[i]
            show(a, b, dan_for(c, sim.n_comp))
    dec = [(decode(sim, show(a, b))[0], c) for (a, b), c in seqs for _ in range(p["n_test"])]
    return {"acc": float(np.mean([d == c for d, c in dec])),
            "uncertain": float(np.mean([d is None for d, _ in dec]))}


def pn_direct(sim, pats, cfg, rng, noisy):
    """Structural control: the same rule and readout applied to PNs directly (no KC layer).
    noisy=True samples Poisson PN counts, matching the spiking model's PoissonGroup input."""
    p, c = cfg["protocol"], cfg["controls"]
    sec = p["duration_ms"] / 1000.0
    w = np.full((sim.n_comp, sim.n_pn), sim.w0)

    def rates(stim):
        return rng.poisson(stim * sec) / sec if noisy else stim

    for _ in range(p["n_train"]):
        for i in rng.permutation(len(TRANS)):
            s, x, t = TRANS[i]
            dopamine_rule(w, rates(pair(pats, s, x)) >= c["pn_active_hz"],
                          dan_for(STATES.index(t), sim.n_comp), sim.w0, sim.eta_dep, sim.eta_rec)
    ok = []
    for s, x, t in TRANS:
        for _ in range(p["n_test"]):
            out = w @ rates(pair(pats, s, x))
            ok.append(out[STATES.index(t)] > out[1 - STATES.index(t)])  # tie = wrong
    return float(np.mean(ok))


def run_loop(sim, pats, cfg, rng, length, gap_ms):
    """2c: closed loop. The decoded state is fed back (fixed rule, no learning). gap_ms=None:
    each step starts from rest; otherwise the network keeps its state and spontaneous PN input
    is played for gap_ms before the next step (residual activity can leak in)."""
    dur = cfg["protocol"]["duration_ms"]
    spont = np.full(sim.n_pn, cfg["tokens"]["pn_spont_hz"])
    true_s, fed = "even", "even"
    step_ok, track_ok = [], []
    for k in range(length):
        x = "A" if rng.random() < cfg["loop"]["p_input_a"] else "B"
        if gap_ms and k > 0:
            sim.present(spont, gap_ms, reset=False)
        # first step of every sequence starts from rest, so sequences are independent
        out = sim.present(pair(pats, fed, x), dur, reset=gap_ms is None or k == 0)
        d, _ = decode(sim, out["mbon"])
        new = STATES[d] if d is not None else None
        step_ok.append(new == delta(fed, x))
        true_s = delta(true_s, x)
        track_ok.append(new == true_s)
        fed = new if new is not None else fed  # silent readout: keep the last state
    return {"step_acc": float(np.mean(step_ok)), "seq_ok": bool(all(track_ok)),
            "final_ok": bool(track_ok[-1])}


def run_seed(args):
    kind, cfg, seed = args
    t0 = time.perf_counter()
    st = cfg["rng_streams"]
    rng = np.random.default_rng(seed + st["task"])
    pats = token_patterns(cfg, np.random.default_rng(seed + st["tokens"]))
    ridge = cfg["controls"]["ridge"]
    out = {"seed": seed}

    sim = make_sim(kind, cfg, seed)
    active = sim.present(pair(pats, "even", "A"), cfg["protocol"]["duration_ms"])["kc"]
    out["pair_sparseness"] = float((active >= sim.kc_active_hz).mean())
    out["order"] = {str(ms_): order_task(make_sim(kind, cfg, seed, k=i + 1), pats, cfg, rng, ms_)
                    for i, ms_ in enumerate(cfg["order"]["segment_ms"])}

    train_table(sim, pats, cfg, rng)
    tt = test_table(sim, pats, cfg)
    out["step_acc"] = tt["acc"]
    out["step_conf"] = float(np.mean(tt["conf"]))
    if kind == "spiking":  # rate model: repeats are identical, so a split is not held-out
        out["baseline_pn_linear"] = linear_baseline(tt["X_pn"], tt["y"], ridge, rng)
        out["baseline_kc_linear"] = linear_baseline(tt["X_kc"], tt["y"], ridge, rng)
    out["pn_direct_rule"] = pn_direct(sim, pats, cfg, rng, noisy=kind == "spiking")
    w_trained = sim.w.copy()

    loops = {}
    for L in cfg["loop"]["lengths"]:
        res = [run_loop(sim, pats, cfg, rng, L, None) for _ in range(cfg["loop"]["n_seq"])]
        loops[str(L)] = {k: float(np.mean([r[k] for r in res]))
                         for k in ["step_acc", "seq_ok", "final_ok"]}
    out["loop"] = loops

    if kind == "spiking":  # 2d: only the spiking model has noise and carry-over
        rob = {}
        L = cfg["robustness"]["length"]
        for gap in cfg["robustness"]["gaps_ms"]:
            sim.w = w_trained.copy()
            res = [run_loop(sim, pats, cfg, rng, L, gap) for _ in range(cfg["robustness"]["n_seq"])]
            rob[f"gap_{gap}"] = {k: float(np.mean([r[k] for r in res]))
                                 for k in ["step_acc", "seq_ok"]}
        for i, sigma in enumerate(cfg["robustness"]["noise_sigma_mv"]):
            c2 = copy.deepcopy(cfg)
            c2["spiking"]["noise_sigma_mv"] = sigma
            s2 = make_sim(kind, c2, seed, k=20 + i)
            train_table(s2, pats, c2, rng)
            rob[f"sigma_{sigma}"] = {"step_acc": test_table(s2, pats, c2)["acc"]}
        out["robustness"] = rob

    ctrl = {}
    for name, kw in [("teacher_shuffle", {"k": 30}),
                     ("wiring_shuffle", {"k": 31, "wiring": "shuffled"}),
                     ("apl_ablation", {"k": 32, "g_apl_zero": True})]:
        s2 = make_sim(kind, cfg, seed, **kw)
        train_table(s2, pats, cfg, rng, random_teacher=name == "teacher_shuffle")
        ctrl[name] = test_table(s2, pats, cfg)["acc"]
        if name == "apl_ablation":
            a2 = s2.present(pair(pats, "even", "A"), cfg["protocol"]["duration_ms"])["kc"]
            ctrl["apl_ablation_sparseness"] = float((a2 >= s2.kc_active_hz).mean())
    out["controls"] = ctrl
    out["wall_s"] = time.perf_counter() - t0
    return out


def exp_parity_table(cfg: dict, seed: int) -> dict:
    """Train + test the transition table only, for cfg["seeds"], in parallel. Optional
    cfg["grid"] = {"section.key": [values, ...]} runs every combination (a sweep); optional
    cfg["variants"] = [{"name": ..., "set": {"section.key": value}}] is crossed with it."""
    import itertools

    grid = cfg.get("grid", {})
    combos = [dict(zip(grid, v, strict=True)) for v in itertools.product(*grid.values())] or [{}]
    variants = cfg.get("variants", [{"name": "base", "set": {}}])
    combos = [{"variant": v["name"], **v["set"], **g} for v in variants for g in combos]
    out = []
    for kind in cfg["simulators"]:
        for combo in combos:
            c2 = copy.deepcopy(cfg)
            for path, v in combo.items():
                if path == "variant":
                    continue
                sec, key = path.split(".")
                c2[sec][key] = v
            with ProcessPoolExecutor(cfg["workers"]) as ex:
                rows = list(ex.map(_table_job, [(kind, c2, s) for s in cfg["seeds"]]))
            acc = np.array([r["step_acc"] for r in rows])
            out.append({"simulator": kind, "params": combo, "mean": float(acc.mean()),
                        "min": float(acc.min()), "frac_at_target":
                        float(np.mean(acc >= cfg["gate"]["step_acc_target"])), "seeds": rows})
            log.info("%s %s: mean %.3f min %.3f at-target %.1f", kind, combo, acc.mean(),
                     acc.min(), out[-1]["frac_at_target"])
    return {"runs": out}


def _table_job(args):
    kind, cfg, s = args
    st = cfg["rng_streams"]
    rng = np.random.default_rng(s + st["task"])
    pats = token_patterns(cfg, np.random.default_rng(s + st["tokens"]))
    sim = make_sim(kind, cfg, s)
    train_table(sim, pats, cfg, rng)
    tt = test_table(sim, pats, cfg)
    per = np.array(tt["ok"]).reshape(len(TRANS), -1).mean(1)
    return {"seed": s, "step_acc": tt["acc"], "per_transition": per.tolist()}


def summarize(seeds, cfg):
    def ms(xs):
        xs = np.asarray(xs, float)
        return {"mean": float(xs.mean()), "sd": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0,
                "min": float(xs.min())}

    out = {"step_acc": ms([s["step_acc"] for s in seeds]),
           "pair_sparseness": ms([s["pair_sparseness"] for s in seeds]),
           "order": {seg: {k: ms([s["order"][seg][k] for s in seeds]) for k in ("acc", "uncertain")}
                     for seg in seeds[0]["order"]},
           "pn_direct_rule": ms([s["pn_direct_rule"] for s in seeds]),
           "controls": {k: ms([s["controls"][k] for s in seeds]) for k in seeds[0]["controls"]},
           "loop": {L: {k: ms([s["loop"][L][k] for s in seeds]) for k in seeds[0]["loop"][L]}
                    for L in seeds[0]["loop"]}}
    if "baseline_pn_linear" in seeds[0]:
        for k in ("baseline_pn_linear", "baseline_kc_linear"):
            out[k] = ms([s[k] for s in seeds])
    p = np.array([s["step_acc"] for s in seeds])
    # p^n per seed, then averaged (averaging p first is biased low by Jensen's inequality)
    out["loop_pn_prediction"] = {L: float(np.mean(p ** int(L))) for L in seeds[0]["loop"]}
    if "robustness" in seeds[0]:
        out["robustness"] = {k: {m: ms([s["robustness"][k][m] for s in seeds])
                                 for m in seeds[0]["robustness"][k]}
                             for k in seeds[0]["robustness"]}
    tgt = cfg["gate"]["step_acc_target"]
    out["gate_b"] = {"step_acc_target": tgt, "frac_seeds_at_target": float(np.mean(p >= tgt)),
                     "step_acc_pass": bool(np.all(p >= tgt))}  # every seed, not just the mean
    return out


def exp_parity(cfg: dict, seed: int) -> dict:
    out = {}
    for kind in cfg["simulators"]:
        t0 = time.perf_counter()
        jobs = [(kind, cfg, seed + i) for i in range(cfg["n_seeds"])]
        if cfg.get("workers", 1) > 1 and kind == "spiking":
            # Compile every model variant once here so worker processes only hit the cache.
            for sigma in [cfg["spiking"]["noise_sigma_mv"], *cfg["robustness"]["noise_sigma_mv"]]:
                c2 = copy.deepcopy(cfg)
                c2["spiking"]["noise_sigma_mv"] = sigma
                for apl0 in (False, True):
                    warm = make_sim(kind, c2, seed, g_apl_zero=apl0)
                    warm.present(np.zeros(warm.n_pn), cfg["warmup_ms"])
            with ProcessPoolExecutor(cfg["workers"]) as ex:
                seeds = list(ex.map(run_seed, jobs))
        else:
            seeds = [run_seed(j) for j in jobs]
        out[kind] = {"summary": summarize(seeds, cfg), "seeds": seeds,
                     "wall_s": time.perf_counter() - t0}
        log.info("%s summary: %s (%.0fs)", kind, out[kind]["summary"], out[kind]["wall_s"])
    return out
