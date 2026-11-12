# Phase 2 — Gate B result: FAIL (spiking model)

Run: `results/20261112-161802_phase2_parity_s0` (config `configs/phase2_parity.yaml`, commit 812c1f7, seeds 0–9, both simulators, 1906 s wall time with 10 worker processes). Diagnostic run: `results/20261112-165149_phase2_diag_on_error_s0` (`configs/phase2_diag_on_error.yaml`).

Setup: same synthetic MB as Gate A. Tokens even/odd/A/B each drive 15% of PNs, and (S, x) is presented as the union of the two patterns. Two one-hot output compartments [even, odd]. The teacher depresses the non-target compartment (ADR 0004 rule). 10 training epochs over the 4 transitions; 20 test presentations per transition. A tie or silent output counts as wrong. The gate requires **every seed** ≥ 0.99 step accuracy.

## Results

| Metric (mean ± sd over 10 seeds; min) | Rate | Spiking |
|---|---|---|
| Step accuracy, isolated (S, x) | 1.000 ± 0 | **0.889 ± 0.164; min 0.575** |
| Seeds ≥ 0.99 | 10/10 | **6/10** (seeds 2, 4: 0.84; 5: 0.575; 9: 0.64) |
| KC sparseness for a pair | 0.126 | 0.159 |
| Loop step accuracy, L = 10 / 50 / 100 | 1 / 1 / 1 | 0.89 / 0.88 / 0.88 |
| Fully correct sequences, L = 10 / 50 / 100 | 1 / 1 / 1 | 0.65 / 0.48 / 0.52 |
| Final parity correct, L = 10 / 50 / 100 | 1 / 1 / 1 | 0.75 / 0.64 / 0.70 |
| p^L prediction (per-seed p, averaged) | 1 | 0.64 / 0.60 / 0.60 |

Sequence accuracy is **bimodal over seeds**: seeds with p = 1 are perfect at L = 100, and the others fail almost always. So the curve does not decay smoothly as the mean p^L would suggest. The p^L prediction matches the measured sequence accuracy closely (0.60 vs 0.52 at L = 100).

### Controls (spiking; rate in brackets)

| Control | Step accuracy | Reading |
|---|---|---|
| Teacher shuffle (random targets) | 0.48 [0.55] | chance, as expected |
| PN-direct: same rule and readout, no KC layer | 0.47 [0.58] | **the KC expansion is needed for this rule to learn XOR** |
| Degree-preserving PN→KC shuffle | 0.975 [1.0] | ≈ model. **Vacuous on synthetic iid wiring** (same distribution); meaningful only with the connectome (Phase 3) |
| Same-size random network | = the model itself | the synthetic MB *is* a random network; meaningful only in Phase 3 |
| APL ablation | 0.84 [1.0]; sparseness 0.21 [0.126] | APL is weak in this regime (in the rate model it does not change sparseness at all) |
| Ridge linear classifier on PN spike counts | 0.997 | **a plain linear classifier on raw PN input solves the 4 noisy transitions**: 4 points in 150 dimensions, plus the max-union nonlinearity |
| Ridge linear classifier on KC spike counts | 1.000 | |

### Order discrimination (2a)

The readout integrates both segments. Rate: 0.5 at every segment length (exactly symmetric by construction). Spiking: 0.88–1.0 of answers are "uncertain" at segment lengths of 10, 30 and 250 ms, because AB and BA activate the same KCs and training silences both outputs. **A single MB in this model carries no usable order information.**

### Robustness (2d, spiking)

- Step spacing: carry-over with a gap of 0 / 50 / 200 ms gives step accuracy 0.879 / 0.873 / 0.872, the same as reset (0.88). Residual activity is not the problem.
- Membrane noise σ = 1 / 2 (default) / 4 / 8 mV: step accuracy 0.98 / 0.89 / 0.59 / 0.50. **Margins are thin**, and the result is very noise-sensitive.

## Cause analysis

1. **The rule is not error-driven (main cause).** In failing seeds the two output groups fire almost equally on the failed transitions (seed 5, (even,B): 13.8 vs 14.6 Hz). This is a systematic tie, not readout noise. The rule teaches on every trial, so KCs shared between pairs that need opposite outputs (for example the "even"-only KCs in (even,A)→odd and (even,B)→even) are overwritten by whichever pair was trained last. Plenty of pure-conjunction KCs exist (116–174 per pair), so the KC code is not the limit.
2. **Error-gated teaching helps but is not enough.** The diagnostic teaches only when the output is wrong or its confidence is below 0.3 (a stand-in for MBON→DAN error feedback). It fixes seeds 2 and 4 (→ 1.0) but not 5 and 9 (→ 0.71, 0.68). In seed 5 the failure *moves* to the other two transitions, so the rule trades pairs against each other. Likely contributors, not yet tested: bounded weights (w ∈ [0, w0]) with a binary eligibility, large steps (η_dep = 0.5), and spike-count variability in which KCs count as "active".
3. **Thin margins under noise.** Even working seeds degrade fast: σ = 4 mV gives 0.59.
4. More MBONs per compartment or a longer step spacing will not fix (1). The group means are tied, and spacing had no effect.

## Honest caveats

- The rate model passes perfectly, but it is noise-free, so it is not evidence for the spiking result.
- The "connectome contributes" question cannot be asked yet: on synthetic wiring, the shuffle and random-network controls are the model itself.
- The linear-PN baseline shows that the 4-transition parity table is an easy problem for a generic classifier. The interesting claim can only be about whether *the MB's own rule and readout* can learn it. Without the KC layer they cannot (PN-direct = chance).
