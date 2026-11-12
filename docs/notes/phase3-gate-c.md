# Phase 3 / Gate C: parity on FlyWire v783 wiring

> **v1, superseded.** This version used the arbitrary MBON grouping and a buggy MBON selection (soma side instead of KC-input side; ADR 0007). It is kept for the record; see "Gate C v2" below once added.

**Result: FAIL as defined.** Phase 2 parity is not reproduced with the real connectivity as mapped in ADR 0006. The cause is localized to the readout mapping (KC→MBON mask combined with an arbitrary assignment of MBON types to the two outputs), not to the PN→KC connectome.

Gate C (docs/02-roadmap.md): "Phase 2 tasks are reproduced with real connectivity, and the contribution of each structural component is quantified." Phase 2 target: transition-table step accuracy ≥ 0.99 on every seed.

## Runs

| run | config | commit |
|---|---|---|
| `results/20261112-183344_phase3_controls_s0` | `configs/phase3_controls.yaml` (spiking) | 210c929 |
| `results/20261112-191516_phase3_controls_rate_s0` | `configs/phase3_controls_rate.yaml` (rate) | see meta.json |

The full-parity run on real wiring (`phase3_parity_flywire.yaml`: order, loop, robustness) crashed at warm-up from a config bug, which was then fixed in 60d0c88. It was not re-run: the transition table already misses the target, so the loop would only re-measure a failing configuration. Everything except the wiring is at the Gate B values (ADR 0005).

## Transition table, 10 seeds

Values are mean step accuracy over seeds, with the number of seeds at ≥ 0.99 in brackets. Chance is 0.5.

### Spiking, by membrane noise σ (mV; Phase 2 default is 2)

| variant | σ 2 | σ 4 | σ 6 | σ 8 |
|---|---|---|---|---|
| real | 0.889 (5) | 0.735 (2) | 0.530 (0) | 0.500 (0) |
| shuffle (degree-preserving PN→KC) | 0.851 (5) | 0.700 (0) | 0.512 (0) | 0.500 (0) |
| random PN→KC (same edge count) | 0.826 (3) | 0.654 (1) | 0.541 (0) | 0.516 (0) |
| apl_uniform | 0.896 (4) | 0.746 (1) | 0.515 (0) | 0.500 (0) |
| apl_off | 0.831 (3) | 0.642 (0) | 0.500 (0) | 0.500 (0) |
| **kc_mbon_all** (real PN→KC, all-to-all readout) | **0.994 (9)** | **0.980 (8)** | 0.826 (2) | 0.641 (1) |

For comparison, synthetic wiring in Phase 2 gave 10/10 at σ 2, was robust to σ 4, and scored 0.606 at σ 8 (loop, `docs/notes/phase2-gate-b.md`).

### Rate model (noise-free)

| variant | mean | min | seeds ≥ 0.99 |
|---|---|---|---|
| real | 0.900 | 0.750 | 6 |
| shuffle | 0.900 | 0.500 | 7 |
| random | 0.750 | 0.500 | 2 |
| apl_uniform | 0.900 | 0.750 | 6 |
| apl_off | 0.900 | 0.750 | 6 |
| kc_mbon_all | 1.000 | 1.000 | 10 |
| random + kc_mbon_all | 0.975 | 0.750 | 9 |

## Analysis

1. **The readout causes the failure.** Replacing the real KC→MBON mask with an all-to-all readout, keeping the real PN→KC wiring, restores Phase 2 performance: 10/10 seeds in the rate model, 9/10 in spiking at σ 2, and better noise robustness than any other variant. With the real mask, each output (a random half of the MBON types, ADR 0006 decision 4) sees only the KCs those types innervate. A transition whose KC code mostly hits MBONs on one side cannot be taught to the other side, so the result depends on the seed's grouping, which explains the bimodal seeds (min 0.5–0.75). The grouping is arbitrary because the model has no DANs to define compartments, so this is a failure of the **mapping**, not evidence about the fly.
2. **PN→KC wiring: weak, mixed evidence.** Under the real readout, real ≥ shuffle ≥ random at σ ≤ 4 in spiking. The real–shuffle differences (0.889 vs 0.851; 0.735 vs 0.700) are within seed-to-seed variance and were not tested. Random wiring is worse in the rate model (0.750 vs 0.900). With an all-to-all readout, random reaches 0.975 against 1.000 for real. **No clear evidence that the real PN→KC structure, beyond its degree distribution, contributes.**
3. **APL: no evidence of contribution from its real wiring.** Uniform APL weights score the same as the real ones at every σ. Removing APL entirely costs a little in spiking (0.831 vs 0.889 at σ 2) and nothing in the rate model, consistent with Phase 2, where APL was weak in this regime.
4. **Not quantified:** the KC–KC, MBON→DAN and MBON–MBON ablations and compartment removal listed in the roadmap. They need model components that do not exist yet (ADR 0006 decision 6). Order, loop and robustness on real wiring were not run (see above).

## Options (user decision, per CLAUDE.md)

- **A. Compartment-faithful grouping.** Assign MBON types to the two outputs by lobe/compartment from the annotations (for example by valence class, if annotated), instead of at random. This is the most biologically meaningful option, but the compartment-to-valence mapping must be verified from the literature first.
- **B. Coverage-balanced grouping.** Choose the split so that both outputs receive input from the same KCs. This is a fixed rule with no learning, but the choice is ours, not the fly's.
- **C. All-to-all readout.** Use the real PN→KC and APL wiring with a synthetic readout (`kc_mbon_all`), which already passes in the rate model and nearly passes in spiking. The honest framing is "real input side, idealized output side".
- **D. Per-compartment outputs.** Treat each MBON type as its own yes/no output, read out by a fixed rule.
- **E. Accept and report.** Record Gate C as failed and move on to Phase 4 on synthetic wiring.

---

# Gate C v2: DAN-cluster grouping, corrected MBON set (ADR 0007)

**Result: near-pass, FAIL on the strict criterion.** On real FlyWire wiring the spiking model reproduces parity almost to the Phase 2 level. However, 1 of 10 seeds falls just short of 0.99 in the full run, and 4 of 10 fall short in the controls run on the same seeds. The difference between the two runs is only the noise and task random draws, so this result sits on the gate boundary. The rate model passes everything.

Runs (all on commit 2394e2f or code-identical predecessors):
- `results/20261112-203652_phase3_parity_flywire_s0`: full Phase 2 protocol on real wiring.
- `results/20261112-193520_phase3_controls_s0`: spiking controls × noise.
- `results/20261112-193240_phase3_controls_rate_s0`: rate controls, seeds 0–9.
- `results/20261112-193424_phase3_heldout_rate_s0`: rate held-out check, seeds 20–29.

## Full protocol on real wiring (10 seeds; Phase 2 synthetic values from `phase2-gate-b.md` for comparison)

| measure | rate | spiking | Phase 2 spiking (synthetic) |
|---|---|---|---|
| step accuracy, mean / min | 1.000 / 1.000 | 0.997 / 0.975 | 1.0 / 1.0 |
| seeds ≥ 0.99 | 10/10 | **9/10** | 10/10 |
| loop, 100 steps: sequences fully correct | 1.000 | 0.920 (min 0.4) | 1.0 |
| loop step accuracy, lengths 10/50/100 | 1.0 / 1.0 / 1.0 | 0.998 / 0.998 / 0.999 | 1.0 |
| σ 1 / 4 / 8 mV step accuracy | n/a | 0.995 / 0.875 / 0.500 | robust to 4, 0.606 at 8 |
| step gaps 0 / 50 / 200 ms | n/a | 0.998 / 0.998 / 0.997 | ~1.0 |
| order (AB vs BA), 250 ms segments | 0.500 | 0.505 | not represented |
| PN-direct rule (no KC layer) | 0.525 | 0.576 | chance |
| teacher shuffle | 0.550 | 0.504 | chance |
| linear classifier on PN / KC counts | n/a | 0.988 / 1.000 | 0.993 / 1.0 |

## Spiking controls by membrane noise σ (mean, seeds ≥ 0.99 in brackets)

| variant | σ 2 | σ 4 | σ 6 | σ 8 |
|---|---|---|---|---|
| real (DAN groups, per-group scale) | 0.980 (6) | 0.879 (2) | 0.562 (0) | 0.501 (0) |
| swap (labels swapped) | 0.991 (9) | 0.935 (6) | 0.579 (0) | 0.503 (0) |
| arbitrary grouping (v1 rule) | 0.894 (4) | 0.734 (1) | 0.574 (0) | 0.546 (0) |
| per-MBON scale (v1 rule) | 0.642 (0) | 0.542 (0) | 0.500 (0) | 0.500 (0) |
| shuffle (degree-preserving PN→KC) | 0.979 (9) | 0.881 (4) | 0.539 (0) | 0.500 (0) |
| random PN→KC | 0.972 (8) | 0.941 (6) | 0.728 (1) | 0.595 (0) |
| apl_uniform | 0.978 (6) | 0.879 (1) | 0.569 (0) | 0.510 (0) |
| apl_off | 0.924 (6) | 0.626 (0) | 0.517 (0) | 0.500 (0) |
| kc_mbon_all (all-to-all readout) | 0.989 (9) | 0.985 (7) | 0.791 (2) | 0.635 (0) |

Rate model (seeds 0–9 / held-out 20–29, mean):
- real: 1.000 / 1.000
- arbitrary: 0.925 / 0.975
- per_mbon: 0.675 / 0.625
- shuffle: 0.775 / 1.000
- random: 0.900 / 0.975
- apl_uniform: 0.675 / 1.000
- apl_off: 0.650 / 1.000
- kc_mbon_all: 1.000 / 1.000

The seeds 0–9 rate column for the real variant comes from the diagnostic before `group` was set as the default (ADR 0007). The v2 controls_rate run used the per_mbon scale for every variant except where stated. **The held-out column is the clean one.**

## Contribution of each structural component

1. **Readout grouping and scale (KC→MBON):** large effect.
   - DAN-cluster groups with a per-group scale beat the v1 arbitrary grouping (0.980 vs 0.894 at σ 2; 0.879 vs 0.734 at σ 4).
   - The v1 per-MBON scale is near chance (0.642).
   - The all-to-all readout is still the most noise-robust: the real mask costs accuracy at σ ≥ 4.
2. **PN→KC wiring: no evidence of contribution.**
   - The degree-preserving shuffle equals real at every σ.
   - Uniform random wiring is *more* noise-robust than real (0.941 vs 0.879 at σ 4; 0.728 vs 0.562 at σ 6).
   - A plausible explanation (not tested) is the uneven real PN fan-out and claw counts (4.23 ± 1.61), which give some KCs fewer claws.
3. **APL:** the real per-KC APL weights show no effect (apl_uniform equals real at every σ). Having inhibition at all matters under noise (apl_off: 0.626 vs 0.879 at σ 4).
4. **Label assignment:** swap is slightly better than real at σ 2 and 4. With 10 seeds this is within variance (not tested), so the readout is roughly symmetric.
5. **Not quantified:** KC–KC, MBON→DAN and MBON–MBON connections, which are not modeled (ADR 0006/0007).

## Interpretation

The task is still solved by the KC expansion plus the dopamine rule. The PN-direct and teacher-shuffle controls are at chance. However, the PN linear classifier reaches 0.988, so as in Phase 2 the task is nearly linearly separable at the input. The real connectome's specific PN→KC and APL structure adds nothing measurable here. What makes the real MB work is the readout organization by DAN cluster, and even that is a little worse than an idealized all-to-all readout.
