"""Simulator interface shared by the rate and spiking models (ADR 0001).

Phase 0 model is "empty": one population of unconnected LIF neurons driven
directly by the stimulus vector. Later phases add PN/KC/APL/MBON/DAN behind the
same `present()` signature.
"""

from abc import ABC, abstractmethod

import numpy as np


class Simulator(ABC):
    """Present a stimulus for a time window, return mean firing rates (Hz) per population."""

    def __init__(self, model: dict, seed: int):
        self.model = model
        self.n = model["n_neurons"]
        self.seed = seed

    @abstractmethod
    def present(self, stimulus: np.ndarray, duration_ms: float) -> dict[str, np.ndarray]:
        """stimulus: drive per neuron in mV, shape (n,). Returns {population: rates_hz (n,)}."""


class RateSimulator(Simulator):
    """Analytic LIF transfer function: the noise-free steady-state rate of the spiking model."""

    def present(self, stimulus, duration_ms):
        m = self.model
        gap = m["v_thresh_mv"] - m["v_reset_mv"]
        drive = np.asarray(stimulus, float) + m["v_rest_mv"] - m["v_reset_mv"]
        rates = np.zeros(self.n)
        on = drive > gap
        isi_ms = m["refractory_ms"] + m["tau_m_ms"] * np.log(drive[on] / (drive[on] - gap))
        rates[on] = 1000.0 / isi_ms
        return {"main": rates}


class SpikingSimulator(Simulator):
    """Brian2 LIF population (ADR 0003). The network is built once and reused across calls."""

    def __init__(self, model, seed):
        super().__init__(model, seed)
        import brian2 as b2

        self.b2 = b2
        b2.prefs.codegen.target = model["codegen_target"]
        b2.seed(seed)
        b2.defaultclock.dt = model["dt_ms"] * b2.ms
        eqs = """
        dv/dt = (v_rest - v + drive) / tau + sigma * sqrt(2 / tau) * xi : volt (unless refractory)
        drive : volt
        """
        ns = {
            "v_rest": model["v_rest_mv"] * b2.mV,
            "tau": model["tau_m_ms"] * b2.ms,
            "sigma": model["noise_sigma_mv"] * b2.mV,
        }
        self.group = b2.NeuronGroup(
            self.n,
            eqs,
            threshold=f"v > {model['v_thresh_mv']}*mV",
            reset=f"v = {model['v_reset_mv']}*mV",
            refractory=model["refractory_ms"] * b2.ms,
            method="euler",
            namespace=ns,
        )
        self.group.v = model["v_reset_mv"] * b2.mV
        self.spikes = b2.SpikeMonitor(self.group, record=False)
        self.net = b2.Network(self.group, self.spikes)

    def present(self, stimulus, duration_ms):
        b2 = self.b2
        self.group.drive = np.asarray(stimulus, float) * b2.mV
        before = np.array(self.spikes.count[:])
        self.net.run(duration_ms * b2.ms)
        return {"main": (np.array(self.spikes.count[:]) - before) / (duration_ms / 1000.0)}


SIMULATORS = {"rate": RateSimulator, "spiking": SpikingSimulator}
