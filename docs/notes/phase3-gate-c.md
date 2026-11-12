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
