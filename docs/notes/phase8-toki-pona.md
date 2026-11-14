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

## 8b: grammar fly for the subset — gate passed

The FSM has 14 states over 4 word classes (mi/sina, other content words, `li`, `e`), with up to two modifiers per phrase. One fly, with no routing, smells (state, class) together. Only valid transitions are taught; low confidence reads as dead.
- On test seeds 90–94 (2000 KCs, threshold 650, table teaching), step and utterance accuracy are 1.000 on every seed.
- Without the KC layer, 0.36; with a shuffled teacher, 0.30.
- Teaching each corpus transition once (`dedup`) beats frequency-weighted corpus teaching (`raw`) at the same budget: step 0.979 vs 0.807 (H5).
- The 7c routed design does worse here (0.952).
- Correction: the pre-run check on seed 10 used threshold 450, not 650 as first written.

Details: ADR 0013.

## 8b-2: the tokenizer as a fly — both ways fail

- **A, class fly in series:** class accuracy 0.953 and utterance accuracy 0.881 on test seeds 90–94 (gate: 1.0 and 0.99). The failing words are mostly silent: a single word's 22 PNs cannot drive KCs at the 4-of-5-claw threshold chosen for two-token stimuli.
- **B, grammar fly on words:** 1126 transitions; at best 0.62 utterance accuracy, even with the full table; most valid sentences are refused. Interference grows with the number of contexts, and nothing generalizes across words.

Details: ADR 0013.

## 8b-3: the class fly with single-word settings

- With threshold 450 and class-balanced teaching, the class fly classes all 114 words right on every test seed (95–99). The class fly → grammar fly pipeline gives 0.999, the same as with the tokenizer.
- But the no-KC control reaches 0.968 (perfect on 4/5 seeds). By the rule fixed in advance, the gate is not passed: the class step is a lookup that does not need the KC layer.

Details: ADR 0013.

## 8c: memory for noun phrases — gate passed (H3 supported)

The memory fly runs at 16000 KCs with threshold 850, chosen on selection seeds. Facts about 40 phrases go to a fresh fly per level, with shared modifiers so that a conjunction is needed. Test seeds 100–104:
- one modifier: 0.990 correct (min 0.975); no KC layer 0.500, shuffled 0.485;
- two modifiers: 0.823, as predicted (lost coverage; the answers given are 0.99 right);
- bare words at this setting: only 0.565 answered (all right). Single words are too sparse for the all-claws threshold, so no one setting serves words and phrases.

Details: ADR 0013.

## 8d: Toki Pona question answering by flies

The user teaches 80 facts ("jan pona li suli [ala]") and asks 120 questions ("... anu seme?"); the fly answers with a sentence. On test seeds 105–109, replies are exact 0.995 (min 0.983); unknown subjects get "mi sona ala" every time.
- Without the memory fly's KC layer: 0.455.
- Without the speaker's KC layer: 0.000.
- Without the class fly's KC layer: 0.937. This is above the 0.80 control limit, so the gate is formally not passed, as in 8b-3.

Details: ADR 0013.

## 8d-2: memory without routing — fails

The memory fly smells subject and predicate together, with one yes/no output, replacing the routing from predicate word to output pair. On test seeds 110–114, replies are 0.815 (taught questions 0.728). The fly declines about 31% of the facts (low confidence) but answers none wrong, and more KCs do not help. The routing stays as glue.

## 8e: a whole conversation — gate not passed

300 turns per seed: yes/no and object facts (with changes of mind), yes/no and "what" questions, unknown subjects. On test seeds 115–119:
- replies are exact 0.923 (min 0.730); "what" questions 0.974; after a change of mind 0.988;
- without the memory KC layers, 0.470; without the speaker's, 0.000.

The failures:
- Seed 117: the grammar fly, retrained on the larger 8e grammar with 8b's settings, misses S0 + mi → MI, so "mi sona ala" breaks (unknown subjects 0.0).
- Seed 119: yes/no memory under load, plus some speaker errors.

## 8e-2: object memory without routing — fails

"What" questions reach 0.926 (min 0.875) even at 64000 KCs, and unknown subjects 0.850. The object routing stays as glue.
