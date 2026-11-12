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

**Failure mechanism.** Recomputed from the per-transition results with `python -m chatfly.fsm <metrics.json>` (`fsm.margin_table`), random tables only: 720 transitions in rate and 390 in spiking. In the rate model, failures are all-or-nothing per transition. They are predicted by how the other entries in the same row and column vote. Let *margin* be the number of row/column mates whose target is the strongest competing state, minus the number whose target is the correct state.

| margin | ≤ 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| rate: transitions | 241 | 198 | 172 | 91 | 17 | 1 |
| rate: accuracy | 1.000 | 0.995 | 0.901 | 0.692 | 0.294 | 0.000 |
| spiking: transitions | 115 | 112 | 91 | 58 | 13 | 1 |
| spiking: accuracy | 1.000 | 0.955 | 0.769 | 0.638 | 0.231 | 0.000 |

Interpretation (a mechanism consistent with the data, not proven):
- A KC driven by only one of the two tokens is active in every pair of that row or column. Its weights therefore settle toward the row/column majority target.
- Only the conjunctive KCs (driven by both tokens) carry the entry itself, and they get outvoted as the margin grows.
- The counters have margin 2 on every A transition and still score 1.0 in rate. Margin alone is therefore not the whole story: in a counter the competing majority is split across transitions in a structured way. What the table shows is that random tables produce margins ≥ 3, which counters never do.
- An earlier, unscripted check found that PN-pattern similarity does not separate failed from correct transitions. It is not in the repository and is dropped here.

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
- **The `big` result in this table is confounded (independent review).** Spiking APL input and KC→MBON drive are per-spike sums, so 3× KCs also gave sparser KCs (0.127 vs 0.161 active) and about 2× the MBON rate. The control is re-run with both drives scaled by 1/k (see "Review fixes" below). The rate model normalizes by n_kc, so its `big` is unaffected.
- **Routed specialists replicate:** 10/10 on 4×3, and 9/10 on 6×3, where one seed scores 0.964.
- **The no-KC routed control fails in spiking too:** 4/10 and 1/10 seeds.

### Step 1: single-fly capacity (`results/20261113-000839_phase4_sweep_spiking_s0`; loop with 5 sequences of 50 steps)

| task | step acc mean (min) | seeds ≥ 0.99 | loop sequences correct | PN-direct | teacher shuffle |
|---|---|---|---|---|---|
| mod3 counter | 0.999 (0.992) | 10 | 1.00 | 0.417 | 0.321 |
| mod4 counter | 0.996 (0.981) | 8 | 0.82 | 0.351 | 0.224 |
| no consecutive A | 0.993 (0.933) | 9 | 0.92 | 0.826 | 0.290 |
| random 3×3 | 0.938 (0.767) | 4 | 0.56 | 0.622 | 0.271 |
| random 4×3 | 0.903 (0.808) | 1 | 0.06 | 0.558 | 0.214 |
| random 6×3 | 0.740 (0.586) | 0 | 0.04 | 0.344 | 0.066 |

- The structured tasks are learned at ≥ 0.99 mean in spiking too. The strict all-seeds criterion holds for mod3 only: mod4 reaches it on 8/10 seeds and no-AA on 9/10.
- No-AA is largely solvable without KCs (PN-direct 0.826). Its entries rarely need a conjunction.
- The killed first run reproduced exactly on mod3 and mod4 (same seeds, same numbers).

## Review fixes (independent agent review, 2026-11-13)

Rate: `results/20261113-010918_phase4_multi_rate_s0` (full re-run including the new control). Spiking: `results/20261113-011019_phase4_multi_spiking_fix_s0`.

| | big (3× KCs) | routed (separate flies) | **routed_shared** (one fly, one KC layer, KC→MBON weights per input) | routed, no KC |
|---|---|---|---|---|
| rate 3×3 | 0.956 (6) | 1.000 (10) | **1.000 (10)** | 0.989 (9) |
| rate 4×3 | 0.933 (4) | 1.000 (10) | **1.000 (10)** | 0.950 (5) |
| rate 6×3 | 0.867 (0) | 1.000 (10) | **0.994 (9)**, min 0.944 | 0.911 (2) |
| spiking 4×3 | 0.901 (0), drives scaled 1/k | 1.000 (10) | **0.996 (9)**, min 0.958 | 0.944 (4) |
| spiking 6×3 | 0.705 (0), drives scaled 1/k | 0.996 (9) | **0.994 (9)**, min 0.947 | 0.904 (1) |

Values are mean step accuracy, with seeds ≥ 0.99 in brackets. The spiking `routed` values are from the earlier run. The rate model's `routed_pn` now uses independent noise per specialist; rate is deterministic, so its numbers are unchanged.

- **A size-matched bigger fly does not help in spiking either**, once its APL and KC→MBON drives are scaled by 1/k. The earlier gain (0.960) came from the changed operating regime.
- **One fly with the same router does almost exactly as well as separate flies.** The router selects a separate KC→MBON weight set per input token. With a single shared KC layer that is enough: it reaches 9–10/10 seeds wherever separate flies do, and the remaining difference is at most one seed.
- **So the capability comes from input-indexed readout plasticity (selected by the fixed router) on top of a KC layer, not from having several mushroom bodies.** The no-KC control still fails, so a KC expansion is needed. One is enough.

## Gate D verdict (revised; pending user decision)

- **By ADR 0008 decision 5 as written: YES.** Routed flies meet the criterion where a single fly, the size-matched fly and the no-KC router do not.
- **With the review's control: the "multiple flies" part is not what does the work.** A single fly given the same router glue matches separate flies. The tasks a plain single fly cannot learn (random tables with conflicting row/column majorities) become learnable through an input-gated readout, whether it is implemented as several flies or as per-input weights in one fly.
- In this implementation the gating is designer-set glue. Whether a real MB could gate its plasticity by input (for example, via input-specific DAN activity) is untested and not claimed.

## Superseded: first Gate D verdict

Gate D verdict as first written: **YES**

The criterion is ADR 0008 decision 5: a single fly misses the per-seed 0.99 criterion, a multi-fly architecture meets it, and both the size-matched and the no-KC controls miss it.

| | single | big (3× KCs) | ensemble | routed | routed, no KC |
|---|---|---|---|---|---|
| rate 3×3 / 4×3 / 6×3 (seeds ≥ 0.99) | 6 / 3 / 0 | 6 / 4 / 0 | 7 / 3 / 1 | **10 / 10 / 10** | 9 / 5 / 2 |
| spiking 4×3 / 6×3 | 1 / 0 | 6 / 3 | 3 / 0 | **10 / 9** | 4 / 1 |

- **Yes, on random transition tables with conflicting row/column majorities.** Rate 3×3, 4×3, 6×3 and spiking 4×3 meet the full criterion. Spiking 6×3 is 9/10 for routed, with min 0.964, against ≤ 3/10 for every control.
- **What makes it work:** splitting the table by input token, so that no fly has to resolve a conjunction its single-token KCs outvote. Each specialist still needs its own KC expansion.
- **What does not:** more KCs in one fly, or more flies doing the same job with summed outputs.
- **Caveats:**
  1. The router (input → fly) is designer-set glue that does part of the conjunction (`01-architecture.md`, "parallel + arbiter").
  2. The number of flies grows with the number of input tokens.
  3. All on synthetic wiring. One output group per state is not biologically grounded beyond 2 groups (ADR 0007).
  4. Loops and series of flies, and cheaper groupings of inputs per fly, were not explored.
