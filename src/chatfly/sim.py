"""Mushroom body simulators behind one interface (ADR 0001).

Populations: PN (input, rates given by the stimulus) -> KC (claws = PN->KC edges, APL
feedback inhibition) -> MBON (grouped into compartments). KC->MBON weights
`w[compartment, kc]` are plain numpy, shared by both models, and changed by the trial-level
dopamine-gated rule in `Simulator.learn` (ADR 0004/0005).

Wiring comes from `build_wiring`: synthetic (random claws, all-to-all KC->MBON, uniform APL)
or the FlyWire connectome (`chatfly.connectome`), optionally replaced by controls.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import brentq


def pn_kc_claws(n_pn: int, n_kc: int, claws: int, rng) -> np.ndarray:
    """(n_kc, claws) PN indices; each KC samples distinct PNs uniformly (synthetic wiring)."""
    return np.array([rng.choice(n_pn, claws, replace=False) for _ in range(n_kc)])


def dopamine_rule(w, elig, dan, w0, eta_dep, eta_rec):
    """ADR 0004. In place: depress on input x DAN coincidence, recover toward w0 when the input
    is active and the compartment's DAN is silent. elig: per input, bool or in [0, 1]."""
    e = np.asarray(elig, float)[None, :]
    d = dan[:, None]
    w += e * (-eta_dep * d * w + eta_rec * (1 - d) * (w0 - w))


def shuffle_edges(pn, kc, rng):
    """Degree-preserving bipartite shuffle: permute the PN end of every PN->KC edge (keeps
    each KC's in-degree and each PN's out-degree), then swap away duplicate (PN, KC) edges."""
    pn = rng.permutation(pn)
    while True:
        key = kc.astype(np.int64) * (pn.max() + 1) + pn
        _, first = np.unique(key, return_index=True)
        dup = np.setdiff1d(np.arange(len(pn)), first)
        if len(dup) == 0:
            return pn, kc
        for e in dup:
            o = rng.integers(len(pn))
            pairs = set(zip(kc.tolist(), pn.tolist()))
            if (kc[e], pn[o]) not in pairs and (kc[o], pn[e]) not in pairs:
                pn[e], pn[o] = pn[o], pn[e]


def random_edges(n_pn, n_kc, n_edges, rng):
    """Same-size random network: n_edges distinct PN->KC pairs drawn uniformly."""
    flat = rng.choice(n_pn * n_kc, n_edges, replace=False)
    return flat % n_pn, flat // n_pn


def build_wiring(cfg: dict, rng) -> dict:
    """{"n_pn", "n_kc", "pn_edge", "kc_edge", "mbon_comp" (compartment per MBON),
    "mask" (n_mbon, n_kc) KC->MBON weight factors or None = all-to-all with factor 1,
    "apl_in"/"apl_out" (n_kc,) mean 1}.

    Real KC->MBON factors are synapse counts (pairs >= min_synapses), scaled so every MBON's
    row sums to n_kc: the same total drive as an all-to-all MBON, so the configured gains keep
    their meaning, while relative strengths and which KCs reach which MBON stay real (ADR 0006).

    mb.source: "synthetic" | "flywire". mb.wiring: "real" (default) | "shuffle" | "random".
    mb.apl_uniform / mb.kc_mbon_all_to_all: ablations of the real APL / KC->MBON structure."""
    mb = cfg["mb"]
    n_comp = len(mb["compartments"])
    if mb.get("source", "synthetic") == "flywire":
        from chatfly.connectome import load_mb

        c = load_mb(cfg["connectome"])
        n_pn, n_kc = c["n_pn"], c["n_kc"]
        pn, kc = c["pn_edge"].copy(), c["kc_edge"].copy()
        counts = np.where(c["kc_mbon"] >= cfg["connectome"]["min_synapses"], c["kc_mbon"], 0)
        fed = counts.sum(1) > 0  # drop MBONs with no KC input: they cannot carry an output
        # Output groups: whole MBON types, shuffled, then dealt to compartments in turn
        # (arbitrary grouping, varies with the seed; ADR 0006), unless mb.mbon_groups maps each
        # compartment to a list of types (ADR 0007); unlisted types are then dropped.
        uniq = rng.permutation(np.unique(np.array(c["mbon_type"])[fed]))
        groups = mb.get("mbon_groups")
        if groups:
            comp_of_type = {t: mb["compartments"].index(name)
                            for name, ts in groups.items() for t in ts}
            fed &= np.isin(c["mbon_type"], list(comp_of_type))
        else:
            comp_of_type = {t: i % n_comp for i, t in enumerate(uniq)}
        # connectome.mbon_norm: "per_mbon" (ADR 0006: every MBON row sums to n_kc), "global"
        # (one scale, mean row sum n_kc, relative MBON strengths stay real) or "group" (one
        # scale per output group, mean row sum n_kc within each group; ADR 0007)
        if cfg["connectome"].get("mbon_norm", "per_mbon") == "global":
            mask = counts[fed] * (n_kc * fed.sum() / counts[fed].sum())
        else:
            mask = counts[fed] * (n_kc / counts[fed].sum(1, keepdims=True))
        mbon_comp = np.array([comp_of_type[t] for t in np.array(c["mbon_type"])[fed]])
        if cfg["connectome"].get("mbon_norm") == "group":  # one scale per output group
            for g in range(n_comp):
                r = mbon_comp == g
                mask[r] = counts[fed][r] * (n_kc * r.sum() / counts[fed][r].sum())
        apl_in, apl_out = c["kc_to_apl"], c["apl_to_kc"]
    else:
        n_pn, n_kc = mb["n_pn"], mb["n_kc"]
        claws = pn_kc_claws(n_pn, n_kc, mb["claws_per_kc"], rng)
        pn, kc = claws.ravel(), np.repeat(np.arange(n_kc), claws.shape[1])
        mbon_comp = np.repeat(np.arange(n_comp), mb["mbon_per_compartment"])
        mask, apl_in, apl_out = None, np.ones(n_kc), np.ones(n_kc)
    wiring = mb.get("wiring", "real")
    if wiring == "shuffle":
        pn, kc = shuffle_edges(pn, kc, rng)
    elif wiring == "random":
        pn, kc = random_edges(n_pn, n_kc, len(pn), rng)
    if mb.get("apl_uniform"):
        apl_in, apl_out = np.ones(n_kc), np.ones(n_kc)
    if mb.get("kc_mbon_all_to_all"):
        mask = None

    def norm(v):  # mean 1 so the gains in the configs keep their meaning
        v = np.asarray(v, float)
        return v / v.mean() if v.mean() > 0 else np.ones_like(v)

    return {"n_pn": n_pn, "n_kc": n_kc, "pn_edge": np.asarray(pn), "kc_edge": np.asarray(kc),
            "mbon_comp": mbon_comp, "mask": mask, "apl_in": norm(apl_in), "apl_out": norm(apl_out)}


class Simulator(ABC):
    """`present(pn_rates_hz, duration_ms, dan=None, reset=True)`
    -> {"pn": Hz (n_pn,), "kc": Hz (n_kc,), "mbon": Hz (n_mbon,)}.

    With reset=True a presentation starts from the resting network state; with reset=False
    the spiking model continues from the previous state (residual activity carries over;
    the rate model is a steady-state model and has no carry-over). If `dan`
    (one value in [0, 1] per compartment) is given, the KC->MBON weights are updated
    from this presentation's KC activity afterwards; otherwise the weights are untouched.
    """

    def __init__(self, cfg: dict, seed: int, noise_seed: int | None = None):
        """seed fixes wiring; noise_seed (default: seed) seeds the spiking model's noise and
        Poisson input, so controls can share wiring but draw independent noise."""
        self.cfg = cfg
        self.noise_seed = seed if noise_seed is None else noise_seed
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        wi = build_wiring(cfg, self.rng)
        self.n_pn, self.n_kc = wi["n_pn"], wi["n_kc"]
        self.pn_edge, self.kc_edge = wi["pn_edge"], wi["kc_edge"]
        self.mbon_comp, self.mask = wi["mbon_comp"], wi["mask"]  # compartment of each MBON
        self.apl_in, self.apl_out = wi["apl_in"], wi["apl_out"]
        self.n_comp = len(cfg["mb"]["compartments"])
        p = cfg["plasticity"]
        self.w0, self.eta_dep, self.eta_rec = p["w0"], p["eta_dep"], p["eta_rec"]
        self.kc_active_hz = cfg["readout"]["kc_active_hz"]
        self.elig_ref_hz = p.get("elig_ref_hz")  # None: binary eligibility (ADR 0004)
        self.w = np.full((self.n_comp, self.n_kc), self.w0)

    def present(self, pn_rates_hz, duration_ms, dan=None, reset=True):
        out = self._run(np.asarray(pn_rates_hz, float), duration_ms, reset)
        if dan is not None:
            self.learn(out["kc"], np.asarray(dan, float))
        return out

    def learn(self, kc_rates, dan):
        if self.elig_ref_hz is None:
            elig = kc_rates >= self.kc_active_hz
        else:  # graded: proportional to the KC rate, saturating at elig_ref_hz (ADR 0005)
            elig = np.clip(kc_rates / self.elig_ref_hz, 0.0, 1.0) * (kc_rates >= self.kc_active_hz)
        dopamine_rule(self.w, elig, dan, self.w0, self.eta_dep, self.eta_rec)

    @abstractmethod
    def _run(self, pn_rates_hz, duration_ms, reset) -> dict[str, np.ndarray]: ...


class RateSimulator(Simulator):
    """Threshold-linear KCs with APL feedback inhibition solved exactly at steady state.
    No noise."""

    def _run(self, pn_rates_hz, duration_ms, reset):
        r = self.cfg["rate"]
        summed = np.bincount(self.kc_edge, pn_rates_hz[self.pn_edge], minlength=self.n_kc)
        drive = r["w_pn_kc"] * summed - r["kc_theta"]

        def kc(apl):
            return r["kc_gain"] * np.maximum(drive - r["g_apl"] * self.apl_out * apl, 0.0)

        # APL activity a = weighted mean KC rate; mean(apl_in*kc(a)) - a is strictly
        # decreasing in a, so there is exactly one root.
        hi = (self.apl_in * kc(0.0)).mean()
        a = brentq(lambda a: (self.apl_in * kc(a)).mean() - a, 0.0, hi) if hi > 0 else 0.0
        kc_rates = kc(a)
        per_comp = self.w * kc_rates  # (n_comp, n_kc) synaptic drive per compartment
        if self.mask is None:
            mbon = (per_comp.sum(1) / self.n_kc)[self.mbon_comp]
        else:
            mbon = (self.mask * per_comp[self.mbon_comp]).sum(1) / self.n_kc
        return {"pn": pn_rates_hz, "kc": kc_rates, "mbon": r["mbon_gain"] * mbon}


class SpikingSimulator(Simulator):
    """Brian2 LIF network (ADR 0003): Poisson PNs, LIF KCs and MBONs, APL as a leaky KC-spike
    integrator that inhibits all KCs. The network has its own clock; state is restored
    before every presentation."""

    def __init__(self, cfg, seed, noise_seed=None):
        super().__init__(cfg, seed, noise_seed)
        import brian2 as b2

        self.b2 = b2
        s = cfg["spiking"]
        mV, ms = b2.mV, b2.ms
        b2.seed(self.noise_seed)  # global Brian2 RNG: give each instance its own noise_seed
        clock = b2.Clock(s["dt_ms"] * ms)
        ns = {"v_rest": s["v_rest_mv"] * mV, "sigma": s["noise_sigma_mv"] * mV,
              "tau_kc": s["tau_kc_ms"] * ms, "tau_mbon": s["tau_mbon_ms"] * ms,
              "tau_apl": s["tau_apl_ms"] * ms, "g_apl": s["g_apl_mv"] * mV}
        lif = {"threshold": f"v > {s['v_thresh_mv']}*mV", "reset": f"v = {s['v_reset_mv']}*mV",
               "refractory": s["refractory_ms"] * ms, "method": "euler", "clock": clock,
               "namespace": ns}
        self.pn = b2.PoissonGroup(self.n_pn, 0 * b2.Hz, clock=clock)
        self.kc = b2.NeuronGroup(self.n_kc, """
            dv/dt = (v_rest - v - I_apl) / tau_kc + sigma * sqrt(2 / tau_kc) * xi : volt (unless refractory)
            I_apl : volt
            apl_in : 1 (constant)
            apl_out : 1 (constant)""", **lif)
        self.kc.apl_in, self.kc.apl_out = self.apl_in, self.apl_out
        self.mbon = b2.NeuronGroup(len(self.mbon_comp), """
            dv/dt = (v_rest - v) / tau_mbon + sigma * sqrt(2 / tau_mbon) * xi : volt (unless refractory)
            """, **lif)
        self.apl = b2.NeuronGroup(1, "da/dt = -a / tau_apl : 1", clock=clock, namespace=ns,
                                  method="exact")
        pn_kc = b2.Synapses(self.pn, self.kc, on_pre=f"v_post += {s['w_pn_kc_mv']}*mV",
                            clock=clock)
        pn_kc.connect(i=self.pn_edge, j=self.kc_edge)
        kc_apl = b2.Synapses(self.kc, self.apl, on_pre="a_post += apl_in_pre", clock=clock)
        kc_apl.connect()
        apl_kc = b2.Synapses(self.apl, self.kc,
                             "I_apl_post = g_apl * apl_out_post * a_pre : volt (summed)",
                             clock=clock, namespace=ns)
        apl_kc.connect()
        self.kc_mbon = b2.Synapses(self.kc, self.mbon, "w : volt", on_pre="v_post += w",
                                   clock=clock)
        if self.mask is None:
            self.kc_mbon.connect()
        else:
            j, i = np.nonzero(self.mask)
            self.kc_mbon.connect(i=i, j=j)
        self.syn_factor = (np.ones(len(self.kc_mbon)) if self.mask is None
                           else self.mask[np.asarray(self.kc_mbon.j[:]), np.asarray(self.kc_mbon.i[:])])
        self.syn_comp = self.mbon_comp[np.asarray(self.kc_mbon.j[:])]
        self.syn_kc = np.asarray(self.kc_mbon.i[:])
        self.w_scale = s["w_kc_mbon_mv"] * mV
        self.kc.v = self.mbon.v = s["v_rest_mv"] * mV
        self.pn_spk = b2.SpikeMonitor(self.pn, record=False)
        self.kc_spk = b2.SpikeMonitor(self.kc, record=False)
        self.mbon_spk = b2.SpikeMonitor(self.mbon, record=False)
        self.net = b2.Network(self.pn, self.kc, self.mbon, self.apl, pn_kc, kc_apl, apl_kc,
                              self.kc_mbon, self.pn_spk, self.kc_spk, self.mbon_spk)
        self.target = s["codegen_target"]
        self.net.store("rest")

    def _run(self, pn_rates_hz, duration_ms, reset):
        b2 = self.b2
        b2.prefs.codegen.target = self.target  # global pref: set right before every run
        if reset:
            self.net.restore("rest")  # restores state, not the RNG: noise differs per call
        mons = (self.pn_spk, self.kc_spk, self.mbon_spk)
        before = [np.asarray(m.count[:]) for m in mons]
        self.pn.rates = pn_rates_hz * b2.Hz
        self.kc_mbon.w = self.w[self.syn_comp, self.syn_kc] * self.syn_factor * self.w_scale
        self.net.run(duration_ms * b2.ms)
        sec = duration_ms / 1000.0
        return {k: (np.asarray(m.count[:]) - b0) / sec
                for k, m, b0 in zip(("pn", "kc", "mbon"), mons, before, strict=True)}


SIMULATORS = {"rate": RateSimulator, "spiking": SpikingSimulator}
