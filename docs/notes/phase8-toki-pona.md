# Phase 8: Toki Pona hypotheses (ADR 0013)

## 8a: output capacity — gate NOT passed (1 of 5 test seeds), but the capacity question is answered

Test run `results/20261114-152955_phase8a_capacity_s0` (`configs/phase8a_capacity.yaml`, commit ebe4bfd,
rate model, seeds 60–64). The condition was chosen on selection seeds 10–14 (ADR 0013): `k2n20` (each
word = 2 of 20 MBON groups), 16000 KCs, KC threshold 850 (all 5 claws). One-hot uses the same settings
as reference. Vocabulary: 112 Toki Pona content words; 674 (state, cue) contexts.

| V = 112 | groups | step accuracy, mean (min) | sentence exact, mean (min) |
|---|---|---|---|
| **`k2n20`** | 20 | **0.998 (0.994)** | 0.895 (**0.500**) |
| one-hot | 115 | 1.000 (0.999) | 0.901 (0.506) |
| `k2n20`, no KC layer | 20 | 0.287 | 0.000 |
| `k2n20`, shuffled teacher | 20 | 0.005 | 0.000 |

Sentence exact by seed (`k2n20`): 1.000, 0.992, 0.982, 1.000, **0.500** (seed 64). At V = 25 and 60
both codings score 1.000.

**Verdict: not passed.**

| criterion | result |
|---|---|
| step accuracy ≥ 0.99 on every seed | passed |
| controls ≤ 0.70 | passed |
| sentence exact ≥ 0.95 on every seed | **failed** on seed 64 (0.500) |

What it shows:
- **H1 is supported at the step level.** A 2-of-20 code over 20 output groups says 112 words as
  well as 115 one-hot groups. The real fly has 35 MBON types per side, so this is the code that
  would fit it.
- **The KC layer does the work.** Without it, steps are 0.29 and sentences 0.
- **The first parameters failed for a clear reason** (selection run 1). With a KC threshold where 4
  of 5 claws suffice, about 2.8% of KCs fire per context, and each KC is shared by about 19 contexts
  with different targets. Adding KCs does not change that fraction. What fixed it was a **sparser
  code (all 5 claws, ~0.3%) together with more KCs**: 16000, which is 6 times the 2,597 KCs of one
  FlyWire hemisphere.
- **What still fails is one context**, ending the sentence after the predicate or the object. The
  empty cue `-` is one of 113 random 22-PN patterns among 150 PNs, so every PN is in about 17
  patterns and `-` cannot stay distinct. Its few KCs are also driven by the 112 (PRED, word) → `e`
  contexts, and `e` outvotes EOS. This is input crowding in PN space, plus the Phase 4
  majority conflict. On a seed where it happens, half the sentences (those without an object) fail.

Glue in 8a: the true sentence state (8b's job), the cue pointer, and stopping at EOS.

## 8a-2: FlyWire-sized PN layer (340 PNs)

Rejected at selection (seeds 10–19, `results/20261114-154242_phase8a2_capacity_select_s0`). The best mean sentence exact is still the 150-PN condition (0.751); the best 340-PN condition gives 0.742. Every condition is bimodal per seed (about 1.0 or about 0.5): one end-of-sentence context is lost by chance in about half the networks. More PNs do not fix it; the evidence points to a majority conflict on the state token, not input crowding. Details and the table: ADR 0013.

## 8a-3: frequency-balanced teaching — gate passed

Each rare context is taught as often as the majority sharing its state token ((PRED, `-`) → EOS 112 times per epoch). Selection seeds 10–19: sentence exact 0.990 (vs 0.751 without balance; balancing cue tokens too gave 0.940). Test seeds 80–84: steps ≥ 0.997 and sentence exact ≥ 0.980 on every seed; no-KC and shuffled controls ≤ 0.002. Needs 16000 KCs at an all-claws threshold; the schedule is teacher glue. Details: ADR 0013.
