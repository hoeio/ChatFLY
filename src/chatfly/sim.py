"""Mushroom body simulators behind one interface (ADR 0001).

Populations: PN (input, rates given by the stimulus) -> KC (sparse random claws,
global APL feedback inhibition) -> MBON (one group per compartment). KC->MBON
weights `w[compartment, kc]` are plain numpy, shared by both models, and changed
by the trial-level dopamine-gated rule in `Simulator.learn` (ADR 0004).
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import brentq


def pn_kc_claws(n_pn: int, n_kc: int, claws: int, rng) -> np.ndarray:
    """(n_kc, claws) PN indices; each KC samples distinct PNs uniformly (synthetic wiring)."""
    return np.array([rng.choice(n_pn, claws, replace=False) for _ in range(n_kc)])


class Simulator(ABC):
    """`present(pn_rates_hz, duration_ms, dan=None) -> {"kc": Hz (n_kc,), "mbon": Hz (n_mbon,)}`.

    Each presentation starts from the resting network state (no carry-over). If `dan`
    (one value in [0, 1] per compartment) is given, the KC->MBON weights are updated
    from this presentation's KC activity afterwards; otherwise the weights are untouched.
    """

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        mb = cfg["mb"]
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.n_pn, self.n_kc = mb["n_pn"], mb["n_kc"]
        self.n_comp, self.m = len(mb["compartments"]), mb["mbon_per_compartment"]
        self.claws = pn_kc_claws(self.n_pn, self.n_kc, mb["claws_per_kc"], self.rng)
        p = cfg["plasticity"]
        self.w0, self.eta_dep, self.eta_rec = p["w0"], p["eta_dep"], p["eta_rec"]
        self.kc_active_hz = cfg["readout"]["kc_active_hz"]
        self.w = np.full((self.n_comp, self.n_kc), self.w0)
        self.mbon_comp = np.repeat(np.arange(self.n_comp), self.m)  # compartment of each MBON

    def present(self, pn_rates_hz, duration_ms, dan=None):
        out = self._run(np.asarray(pn_rates_hz, float), duration_ms)
        if dan is not None:
            self.learn(out["kc"], np.asarray(dan, float))
        return out

    def learn(self, kc_rates, dan):
        """Depress on KC x DAN coincidence; recover toward w0 when KC is active and DAN silent."""
        act = kc_rates >= self.kc_active_hz
        d = dan[:, None]
        w = self.w[:, act]
        self.w[:, act] = w - self.eta_dep * d * w + self.eta_rec * (1 - d) * (self.w0 - w)

    @abstractmethod
    def _run(self, pn_rates_hz, duration_ms) -> dict[str, np.ndarray]: ...


class RateSimulator(Simulator):
    """Threshold-linear KCs with global APL inhibition solved exactly at steady state. No noise."""

    def _run(self, pn_rates_hz, duration_ms):
        r = self.cfg["rate"]
        drive = r["w_pn_kc"] * pn_rates_hz[self.claws].sum(1) - r["kc_theta"]

        def kc(apl):
            return r["kc_gain"] * np.maximum(drive - r["g_apl"] * apl, 0.0)

        # APL activity a = mean KC rate; mean(kc(a)) - a is strictly decreasing, so one root.
        hi = kc(0.0).mean()
        a = brentq(lambda a: kc(a).mean() - a, 0.0, hi) if hi > 0 else 0.0
        kc_rates = kc(a)
        mbon = r["mbon_gain"] * (self.w @ kc_rates) / self.n_kc
        return {"kc": kc_rates, "mbon": np.repeat(mbon, self.m)}


class SpikingSimulator(Simulator):
    """Brian2 LIF network (ADR 0003): Poisson PNs, LIF KCs and MBONs, APL as a leaky KC-spike
    integrator that inhibits all KCs. The network has its own clock; state is restored
    before every presentation."""

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        import brian2 as b2

        self.b2 = b2
        s = cfg["spiking"]
        mV, ms = b2.mV, b2.ms
        b2.seed(seed)
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
            I_apl : volt""", **lif)
        self.mbon = b2.NeuronGroup(self.n_comp * self.m, """
            dv/dt = (v_rest - v) / tau_mbon + sigma * sqrt(2 / tau_mbon) * xi : volt (unless refractory)
            """, **lif)
        self.apl = b2.NeuronGroup(1, "da/dt = -a / tau_apl : 1", clock=clock, namespace=ns,
                                  method="exact")
        pn_kc = b2.Synapses(self.pn, self.kc, on_pre=f"v_post += {s['w_pn_kc_mv']}*mV",
                            clock=clock)
        pn_kc.connect(i=self.claws.ravel(), j=np.repeat(np.arange(self.n_kc), self.claws.shape[1]))
        kc_apl = b2.Synapses(self.kc, self.apl, on_pre="a_post += 1", clock=clock)
        kc_apl.connect()
        apl_kc = b2.Synapses(self.apl, self.kc, "I_apl_post = g_apl * a_pre : volt (summed)",
                             clock=clock, namespace=ns)
        apl_kc.connect()
        self.kc_mbon = b2.Synapses(self.kc, self.mbon, "w : volt", on_pre="v_post += w",
                                   clock=clock)
        self.kc_mbon.connect()
        self.syn_comp = self.mbon_comp[np.asarray(self.kc_mbon.j[:])]
        self.syn_kc = np.asarray(self.kc_mbon.i[:])
        self.w_scale = s["w_kc_mbon_mv"] * mV
        self.kc.v = self.mbon.v = s["v_rest_mv"] * mV
        self.kc_spk = b2.SpikeMonitor(self.kc, record=False)
        self.mbon_spk = b2.SpikeMonitor(self.mbon, record=False)
        self.net = b2.Network(self.pn, self.kc, self.mbon, self.apl, pn_kc, kc_apl, apl_kc,
                              self.kc_mbon, self.kc_spk, self.mbon_spk)
        self.target = s["codegen_target"]
        self.net.store("rest")

    def _run(self, pn_rates_hz, duration_ms):
        b2 = self.b2
        b2.prefs.codegen.target = self.target  # global pref: set right before every run
        self.net.restore("rest")  # restores state, not the RNG: noise differs per presentation
        self.pn.rates = pn_rates_hz * b2.Hz
        self.kc_mbon.w = self.w[self.syn_comp, self.syn_kc] * self.w_scale
        self.net.run(duration_ms * b2.ms)
        sec = duration_ms / 1000.0
        return {"kc": np.asarray(self.kc_spk.count[:]) / sec,
                "mbon": np.asarray(self.mbon_spk.count[:]) / sec}


SIMULATORS = {"rate": RateSimulator, "spiking": SpikingSimulator}
