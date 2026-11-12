"""Phase 1 / Gate A: A+ / B- conditioning, extinction, reversal, and an ablation of the
recovery term (eta_rec = 0) to show which properties depend on it."""

import logging
import time

import numpy as np
from scipy.stats import wilcoxon

from chatfly.sim import SIMULATORS

log = logging.getLogger("chatfly")


def token_patterns(cfg: dict, rng) -> dict[str, np.ndarray]:
    """Each token drives a random subset of PNs at pn_on_hz; all other PNs fire at pn_spont_hz."""
    t = cfg["tokens"]
    if cfg["mb"].get("source", "synthetic") == "flywire":
        from chatfly.connectome import load_mb

        n_pn = load_mb(cfg["connectome"])["n_pn"]
    else:
        n_pn = cfg["mb"]["n_pn"]
    k = round(t["pn_active_frac"] * n_pn)
    pats = {}
    for name in t["names"]:
        r = np.full(n_pn, t["pn_spont_hz"])
        r[rng.choice(n_pn, k, replace=False)] = t["pn_on_hz"]
        pats[name] = r
    return pats


def valence(sim, mbon_rates) -> float:
    """(approach - avoid) / (approach + avoid) over compartment-mean MBON rates; 0 if silent."""
    comp = np.array([mbon_rates[sim.mbon_comp == c].mean() for c in range(sim.n_comp)])
    s = comp.sum()
    return float((comp[0] - comp[1]) / s) if s > 0 else 0.0


def decide(v: float, band: float) -> str:
    return "uncertain" if abs(v) < band else ("approach" if v > 0 else "avoid")


def run_seed(kind: str, cfg: dict, seed: int) -> dict:
    sim = SIMULATORS[kind](cfg, seed)
    pats = token_patterns(cfg, np.random.default_rng(seed + cfg["rng_streams"]["tokens"]))
    p = cfg["protocol"]
    dur = p["duration_ms"]
    # DAN per [approach, avoid]; 1 = teacher on, 0 = off (definitional, not a parameter)
    reward, punish, none = [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]

    def test(token):
        return float(np.mean([valence(sim, sim.present(pats[token], dur)["mbon"])
                              for _ in range(p["n_test"])]))

    def train(pairs):
        for _ in range(p["n_train"]):
            for token, dan in pairs:
                sim.present(pats[token], dur, dan)

    def extinction():
        for _ in range(p["n_ext"]):
            sim.present(pats["A"], dur, none)

    active = {k: sim.present(v, dur)["kc"] >= sim.kc_active_hz for k, v in pats.items()}
    names = list(active)
    overlap = [float((active[a] & active[b]).sum() / max(active[a].sum(), 1))
               for i, a in enumerate(names) for b in names[i + 1:]]
    out = {"sparseness": {k: float(v.mean()) for k, v in active.items()},
           "mean_overlap": float(np.mean(overlap))}
    out["pre"] = {"A": test("A"), "B": test("B")}
    train([("A", reward), ("B", punish)])
    out["post"] = {"A": test("A"), "B": test("B")}
    w_trained = sim.w.copy()
    eta_rec = sim.eta_rec
    for tag, rec in [("", eta_rec), ("_norec", 0.0)]:
        sim.eta_rec = rec
        sim.w = w_trained.copy()
        extinction()
        out["ext" + tag] = {"A": test("A")}
        sim.w = w_trained.copy()
        train([("A", punish), ("B", reward)])
        out["rev" + tag] = {"A": test("A"), "B": test("B")}
    sim.eta_rec = eta_rec
    band = cfg["readout"]["uncertain_band"]
    out["decisions"] = {ph: {t: decide(v, band) for t, v in out[ph].items()}
                        for ph in ["pre", "post", "ext", "rev"]}
    return out


def gate(seeds: list[dict], cfg: dict) -> dict:
    """Gate A checks across seeds (one-sided Wilcoxon signed-rank tests)."""
    g = cfg["gate"]
    lo, hi = g["sparseness_range"]

    def arr(ph, t):
        return np.array([s[ph][t] for s in seeds])

    def p_greater(x):  # H1: x > 0. Zeros are dropped, so min p = 2**-n_nonzero (reported).
        return float(wilcoxon(x, alternative="greater").pvalue) if np.any(x != 0) else 1.0

    sp = np.array([v for s in seeds for v in s["sparseness"].values()])
    diff = {ph: arr(ph, "A") - arr(ph, "B") for ph in ["pre", "post", "rev", "rev_norec"]}
    learn_p = p_greater(diff["post"] - diff["pre"])
    ext_drop = arr("post", "A") - arr("ext", "A")
    ext_p = p_greater(ext_drop)
    ext_p_norec = p_greater(arr("post", "A") - arr("ext_norec", "A"))
    rev_p = p_greater(-diff["rev"])
    rev_p_norec = p_greater(-diff["rev_norec"])
    a = g["alpha"]
    res = {
        "sparseness_mean": float(sp.mean()), "sparseness_min": float(sp.min()),
        "sparseness_max": float(sp.max()),
        "learning_delta_mean": float((diff["post"] - diff["pre"]).mean()), "learning_p": learn_p,
        "extinction_drop_mean": float(ext_drop.mean()), "extinction_p": ext_p,
        "extinction_p_norec": ext_p_norec,
        "reversal_diff_mean": float(diff["rev"].mean()), "reversal_p": rev_p,
        "reversal_p_norec": rev_p_norec,
        "n_nonzero": {"learning": int(np.count_nonzero(diff["post"] - diff["pre"])),
                      "extinction": int(np.count_nonzero(ext_drop)),
                      "reversal": int(np.count_nonzero(diff["rev"]))},
    }
    res["pass"] = {
        "sparseness": bool(lo <= sp.mean() <= hi),
        "learning": learn_p < a,
        "extinction": ext_p < a,
        "reversal": rev_p < a,
    }
    res["pass"]["all"] = all(res["pass"].values())
    return res


def exp_conditioning(cfg: dict, seed: int) -> dict:
    out = {}
    for kind in cfg["simulators"]:
        t0 = time.perf_counter()
        seeds = [run_seed(kind, cfg, seed + i) for i in range(cfg["n_seeds"])]
        out[kind] = {"gate": gate(seeds, cfg), "seeds": seeds,
                     "wall_s": time.perf_counter() - t0}
        log.info("%s gate: %s (%.0fs)", kind, out[kind]["gate"], out[kind]["wall_s"])
    # Gate A: rate and spiking agree qualitatively = same pass/fail on every property
    out["agreement"] = all(out[k]["gate"]["pass"] == out[cfg["simulators"][0]]["gate"]["pass"]
                           for k in cfg["simulators"])
    out["gate_a_pass"] = out["agreement"] and all(out[k]["gate"]["pass"]["all"]
                                                  for k in cfg["simulators"])
    log.info("agreement=%s gate_a_pass=%s", out["agreement"], out["gate_a_pass"])
    return out
