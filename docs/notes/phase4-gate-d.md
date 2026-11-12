# Phase 4 / Gate D: what one fly cannot learn, and whether several flies can

Gate D (`docs/02-roadmap.md`): a clear yes/no on whether any task a single fly cannot do becomes possible with multiple flies. The definitions follow ADR 0008: one output group per state, a synthetic MB at the Gate B parameters, a per-seed criterion of step accuracy ≥ 0.99, and 10 seeds.

## Rate model

### Step 1: single-fly capacity (`results/20261112-220239_phase4_sweep_rate_s0`)

| task | step acc mean (min) | seeds ≥ 0.99 | loop 50 steps: sequences correct | PN-direct | teacher shuffle |
|---|---|---|---|---|---|
| parity (mod2) | 1.000 (1.000) | 10 | 1.00 | 0.525 | 0.400 |
| mod3 counter | 1.000 (1.000) | 10 | 1.00 | 0.400 | 0.417 |
| mod4 counter | 1.000 (1.000) | 10 | 1.00 | 0.400 | 0.338 |
| no consecutive A | 1.000 (1.000) | 10 | 1.00 | 0.750 | 0.333 |
| random 2×2 | 1.000 (1.000) | 10 | 1.00 | 0.825 | 0.575 |
| random 3×2 | 0.983 (0.833) | 9 | 0.90 | 0.800 | 0.317 |
| random 3×3 | 0.944 (0.778) | 6 | 0.76 | 0.633 | 0.311 |
| random 4×2 | 0.963 (0.750) | 8 | 0.80 | 0.688 | 0.250 |
| random 4×3 | 0.925 (0.833) | 3 | 0.31 | 0.600 | 0.333 |
| random 5×3 | 0.887 (0.733) | 3 | 0.31 | 0.413 | 0.233 |
| random 6×3 | 0.867 (0.722) | 0 | 0.09 | 0.389 | 0.128 |

Chance is 1/n_states. The structured tasks, including the 3- and 4-state counters and no-AA, are learned perfectly. Random tables degrade with size.

**Failure mechanism (from the per-transition results; 960 transitions over all random tables).** In the rate model, failures are all-or-nothing per transition. They are predicted by how the other entries in the same row and column vote. Let *margin* be the number of row/column mates whose target is the strongest competing state, minus the number whose target is the correct state.

| margin | ≤ 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| transitions | 271 | 218 | 362 | 91 | 17 | 1 |
| accuracy | 1.000 | 0.995 | 0.953 | 0.692 | 0.294 | 0.000 |

Interpretation:
- A KC driven by only one of the two tokens is active in every pair of that row or column. Its weights therefore settle toward the row/column majority target.
- Only the conjunctive KCs (driven by both tokens) carry the entry itself.
- When the majority disagrees with the entry by ≥ 3, the conjunctive KCs are outvoted.
- This depends on the *fraction* of conjunctive KCs, a statistic of the code. Neither more KCs nor different wiring changes it, as the controls below show.
- Counters and no-AA never create such majorities.

PN-pattern similarity does not explain the failures. The maximum Jaccard index to a conflicting pair is 0.481 for failed transitions and 0.456 for correct ones.

### Step 2: several flies (`results/20261112-220527_phase4_multi_rate_s0`)

| table | single | big (3× KCs) | ensemble (3 flies, summed) | **routed** (1 fly per input) | routed, no KC layer |
|---|---|---|---|---|---|
| 3×3 | 0.944 (6) | 0.956 (6) | 0.967 (7) | **1.000 (10)** | 0.989 (9) |
| 4×3 | 0.925 (3) | 0.933 (4) | 0.925 (3) | **1.000 (10)** | 0.950 (5) |
| 6×3 | 0.867 (0) | 0.867 (0) | 0.878 (1) | **1.000 (10)** | 0.911 (2) |

Values are mean step accuracy, with seeds ≥ 0.99 in brackets. The 50-step loop sequence accuracy for routed is 1.00 on all three tables, against 0.09–0.76 for single.

- More neurons (`big`) and more independent flies (`ensemble`) do not help. The limit is not capacity or noise, consistent with the voting mechanism above.
- **Routed specialists solve every table.** Each specialist sees only one input token, so within a fly no row has conflicting entries; only the state token varies.
- **The router alone is not enough.** The same routing with the rule applied to PNs (no KC layer) fails: 6×3 reaches 0.911 with 2/10 seeds. The specialists' KC expansion is still required.
- **Glue caveat:** the router uses the input token to pick the fly. This is a fixed, designer-set rule that does part of the conjunction. It is the "parallel + arbiter" structure, and its cost is stated in `01-architecture.md`.

## Spiking model

### Step 2: several flies (`results/20261112-220729_phase4_multi_spiking_s0`; loop with 5 sequences of 50 steps)

| table | single | big (3× KCs) | ensemble (3 flies) | **routed** | routed, no KC layer |
|---|---|---|---|---|---|
| 4×3 | 0.903 (1) | 0.960 (6) | 0.918 (3) | **1.000 (10)** | 0.945 (4) |
| 6×3 | 0.740 (0) | 0.960 (3) | 0.838 (0) | **0.996 (9)**, min 0.964 | 0.907 (1) |

Loop sequences fully correct (50 steps):
- 4×3: single 0.06, big 0.56, ensemble 0.22, routed 1.00, no-KC 0.56.
- 6×3: single 0.04, big 0.48, ensemble 0.04, routed 0.98, no-KC 0.12.

Differences from the rate model:
- **Size matters more in spiking.** A single fly is worse here than in the rate model (6×3: 0.740 vs 0.867), and the 3× bigger fly recovers much of the gap (0.960). Part of the spiking failure is therefore noise, which more KCs average out. Even so, the bigger fly reaches the criterion on only 3–6 of 10 seeds.
- **Routed specialists replicate:** 10/10 on 4×3, and 9/10 on 6×3, where one seed scores 0.964.
- **The no-KC routed control fails in spiking too:** 4/10 and 1/10 seeds.

### Step 1: single-fly capacity (spiking)

Running: `configs/phase4_sweep_spiking.yaml`. Results will be added.
