# Phase 0 — run-time estimate by scale

Run: `results/20261112-155149_phase0_bench_s0` (config `configs/phase0_bench.yaml`, seed 0, commit 5083a26). Machine: Apple M4 Pro, 12 cores, macOS 15.3; Python 3.13.5, brian2 2.11.0 (Cython target, single thread).

Model: **unconnected** LIF population with noise, dt = 0.1 ms. One `present()` call simulates 500 ms. "Realtime factor" = wall time / simulated time.

| n neurons | rate: s per present | spiking: s per present | spiking realtime factor |
|---:|---:|---:|---:|
| 100 | 5e-6 | 0.042 | 0.08 |
| 1,000 | 1e-5 | 0.13 | 0.26 |
| 2,000 (≈ KCs per hemisphere) | 2e-5 | 0.18 | 0.35 |
| 10,000 | 7e-5 | 0.68 | 1.35 |
| 100,000 | 1e-3 | 6.1 | 12.2 |

Cython compilation of a new model (about 5 s) happened during the n = 1,000 build, and the compiled code is cached after that. The n = 100,000 "build" (6 s) is mostly its first `present()` call, not building the model.

## What this means

- Neuron updates scale linearly, at about 0.012–0.018 µs per neuron per time step (0.177 s / (2000 × 5000 steps); 6.1 s / (1e5 × 5000 steps)). This is a single machine with 3 repeats and no variance estimate.
- **There are no synapses yet.** PN→KC (about 5–6 claws per KC) is cheap. KC→MBON is all-to-few and plastic, so the cost per step grows with the number of active KCs and plastic synapses. Expect the Phase 1 MB to cost a small multiple of the 2000-neuron row. This will be measured again in Phase 1.
- Rough Phase 2 budget, assuming each step takes 1 s of simulated time (presentation plus spacing) at 3× the cost here: about 1 s of wall time per step. Gate B asks for 10 seeds × 100 sequences × length 100, which is 1e5 steps, about 28 h on a single core. That is too slow for sweeps. The rate model does the sweeps; seeds can run as parallel processes (12 cores, so about 2–3 h); Brian2 `cpp_standalone` is the next lever. These numbers exclude training presentations and plasticity updates, and the parallel-process figure is untested. I will re-estimate after Phase 1 before any large run.
- Rate/spiking check (`results/20261112-155148_phase0_empty_s0`, no noise, n = 2000): mean rate 37.19 Hz (rate) vs 37.16 Hz (spiking), with the same 49.65% of neurons active.

The Phase 0 empty-model code and configs were replaced in Phase 1; they are kept in git history at commit 5083a26. Phase 1 re-estimate: `docs/notes/phase1-gate-a.md`.
