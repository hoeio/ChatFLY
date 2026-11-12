# Phase 1 — Gate A result: PASS

Run: `results/20261112-160327_phase1_conditioning_s0` (config `configs/phase1_conditioning.yaml`, commit 6f02206, seeds 0–9, both simulators). Synthetic MB: 150 PNs → 2000 KCs (5 random claws each), global APL feedback inhibition, 2 MBON compartments (approach / avoid, 4 MBONs each), trial-level dopamine-gated rule (ADR 0004).

Protocol, per seed: pre-test → A+ / B− training (5 trials each) → post-test. From the trained weights, two branches: extinction (A presented 5 times with no US), and reversal (A− / B+, 5 trials each). Both branches are repeated with the recovery term switched off (η_rec = 0). Valence = (approach − avoid) / (approach + avoid) MBON rate. Tests are one-sided Wilcoxon signed-rank across 10 seeds, α = 0.01. p = 0.00098 is the minimum possible p with n = 10.

| Criterion | Rate | Spiking |
|---|---|---|
| KC sparseness, mean (min–max over tokens × seeds); target 0.05–0.15 | 0.106 (0.092–0.120) | 0.143 (0.121–0.165) |
| Learning: Δ(val A − val B), post − pre | +1.75, p = 0.001 | +1.99, p = 0.001 |
| Extinction: drop in val A | 0.77, p = 0.001 | 0.88, p = 0.001 |
| Reversal: val A − val B after reversal | −1.71, p = 0.001 | −1.99, p = 0.001 |
| Extinction with η_rec = 0 | p = 1.0 (no extinction) | p = 0.53 (no extinction) |
| Reversal with η_rec = 0 | p = 0.31 (fails) | p = 1.0 (fails; both MBON groups silent → "uncertain") |

Rate and spiking pass and fail the same criteria (qualitative agreement). **Gate A: pass.**

## Caveats (honest reading)

- Extinction and reversal come **entirely from the assumed recovery term** (ADR 0004): with η_rec = 0 both disappear. This is a property of the rule I chose, not evidence about the fly.
- Spiking sparseness passes on the mean, but single tokens reach 0.165, just above the 0.15 upper bound. The target range itself is an arbitrary default within the verified "< 0.2" bound (Honegger 2011).
- The spiking valence saturates (±1). After 5 pairings with η_dep = 0.5, the depressed MBON group is silent. This is fine for conditioning, but it means valence magnitude is not a graded confidence in this regime.
- To get KCs that detect claw coincidences (3 of 5 claws active → fire, ≤ 2 → silent), I needed PN on-rates of 200 Hz with 2 mV PN→KC jumps. PN rates are unverified. With larger jumps and lower rates, KCs fire from chance coincidences and sparseness is not controlled by claw count (scan in this session; not saved as a run).
- The MB here is 2 compartments, not 15, and has no MBON→DAN feedback or KC–KC connections. Those arrive with the connectome in Phase 3.

## Runtime (re-estimate)

Run `results/20261112-160624_bench_mb_s0`: one 500 ms presentation of the full MB with plasticity, single process on an M4 Pro:

| n_kc | rate | spiking | spiking realtime factor |
|---:|---:|---:|---:|
| 500 | 4e-5 s | 0.16 s | 0.32 |
| 2000 | 6e-5 s | 0.27 s | 0.53 |
| 8000 | 2e-4 s | 0.64 s | 1.29 |

The full Gate A spiking run (10 seeds × ~80 presentations) took 176 s.
