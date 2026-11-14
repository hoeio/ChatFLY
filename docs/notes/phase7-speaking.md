# Phase 7: a fly that speaks Zumi (ADR 0012)

## 7a: corpus, metrics, baselines

Run `results/20261114-081637_phase7a_corpus_s0` (`configs/phase7a_corpus.yaml`, seeds 60–64).
Each seed has 2,000 train, 200 dev and 500 test dialogues of 30 turns each (366k train tokens).
Test turn mix: teach 25%, ask 20%, what 10%, statement 10%, where 12%, yes/no 12%, bad 10%.

Sample (seed 60, test dialogue 0):

```
jdp gu ba        -> hm        ohd ka bon gu ra -> o
si ki zu po ba   -> ya        ohd ku ba        -> hm
si ki ki ta nu   -> fi        zu ki zu         -> o
```

Test results, mean over 5 seeds (the minimum is within 0.005 of the mean everywhere). Replies are generated greedily from the gold history:

| model | reply perplexity | next word | grammatical | appropriate | exact |
|---|---|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1-gram | 15.1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2-gram | 3.07 | 0.599 | 1.000 | 0.353 | 0.353 |
| 3-gram | 1.60 | 0.794 | 1.000 | 0.947 | 0.652 |
| 5-gram | 1.37 | 0.873 | 1.000 | 0.994 | 0.789 |
| 8-gram | 1.44 | 0.875 | 1.000 | 0.994 | 0.794 |
| RNN (128) | 1.34 | 0.874 | 1.000 | 0.999 | 0.790 |

Exact by kind:

| model | teach | statement | bad | ask | what | where | yes/no |
|---|---|---|---|---|---|---|---|
| 5-gram | 1.000 | 1.000 | 0.944 | 0.745 | 0.569 | 0.430 | 0.669 |
| 8-gram | 1.000 | 1.000 | 0.944 | 0.742 | 0.563 | 0.467 | 0.680 |
| RNN | 1.000 | 1.000 | 0.988 | 0.745 | 0.560 | 0.402 | 0.676 |

Readings:

- **The form is easy.** A 5-gram already replies with the right kind of answer 99.4% of the
  time, and the RNN 99.9%. Grammaticality and appropriateness alone can't show that the fly
  does anything: they must be compared with these numbers.
- **The content needs memory, and no baseline has it.**
  - *Ask:* 0.745 for every model from the 3-gram up. A 3-gram sees only `N prop ba`, so this is the rate of `hm` (never taught).
  - *What:* about 0.56, again the `hm` rate.
  - *Yes/no:* about 0.67, about the "no" rate (2 of 3 places are wrong).
  - *Where:* 0.40–0.47, a little above chance (1/3), from moves inside the utterance.
  - The RNN learned neither the facts taught earlier in the dialogue nor the ring place carried across turns.
- **The bar for the fly.** Exact replies on ask / what / where / yes-no above these levels,
  using the memory fly and the ring loop (7d). That is where the fly has something the
  baselines lack: plasticity during the conversation, and a state fed back through its loop.

Caveats: the RNN is a plain numpy Elman net with no tuning beyond the dev-chosen epoch. Its
dev perplexity was still creeping down (1.33 at epoch 30). A gated RNN might learn the ring
or the facts in context; it would be added if a fly result depends on beating it.

**7a acceptance: passed.** The corpus agrees with the parser (`tests/test_speak.py`), the
oracle scores 1.0 on every metric, and the baselines are reported on the test seeds.

## Proposed 7b criteria (to confirm with the user)

A speaking fly v0 sees the user's words and its own previous reply word. Its output groups
are the 18 reply words plus end-of-reply. On the test seeds it should reach:

- grammatical ≥ 0.95 and appropriate ≥ 0.95 (the 5-gram gets 0.994);
- controls (shuffled teacher, no KC layer, degree-preserving shuffle of PN→KC) clearly below.

Content (exact on ask / what / where / yes-no) is not expected before 7d.

## 7b: speaking fly v0 — gate NOT passed

Run `results/20261114-090830_phase7b_speak_s0` (`configs/phase7b_speak.yaml`, rate model).
Architecture "split" (ADR 0012): fly 0 says the first word from the user's last two words,
and fly 1 says every later word from (last user word, previous fly word). Training used 300
dialogues × 3 epochs. Evaluation used 200 dialogues per seed.

Selection (seeds 10–14, dev; mean appropriate / grammatical): (0.1, 450) 0.875 / 1.0 ·
(0.15, 450) 0.760 / 1.0 · **(0.15, 650) 0.940 / 1.0** · (0.2, 650) 0.855 / 1.0.

Test (seeds 60–64, test split), mean (min):

| | perplexity | grammatical | appropriate | exact |
|---|---|---|---|---|
| **fly (split)** | 1.79 (1.74) | 1.000 (1.000) | **0.935 (0.905)** | 0.680 (0.619) |
| shuffled teacher | 31.2 | 0.039 | 0.003 | 0.003 |
| no KC (same rule on PNs) | 3.01 | 1.000 | 0.611 (0.381) | 0.437 |
| shuffled wiring | 1.80 | 1.000 | 0.931 (0.908) | 0.665 |
| context table (ceiling for these inputs) | 1.42 | 0.987 | 0.976 | 0.765 |
| 5-gram (7a) | 1.37 | 1.000 | 0.994 | 0.789 |

Exact by kind:

| | teach | statement | bad | ask | what | where | yes/no |
|---|---|---|---|---|---|---|---|
| fly | 1.000 | 1.000 | 0.421 | 0.616 | 0.565 | 0.322 | 0.536 |
| context table | 1.000 | 1.000 | 0.767 | 0.746 | 0.565 | 0.379 | 0.673 |

**Verdict: not passed.**
- The gate needed appropriate ≥ 0.95 on every test seed; the fly's mean was 0.935 and its minimum 0.905.
- Grammatical passed (1.000).
- Shuffled teacher and no-KC are well below.
- The shuffled-wiring control (0.931) is not below 0.90.

Analysis:
- **The miss is word salad.** "Bad" turns (10%) must get `e`; the fly says it 42% of the
  time (ceiling 77%). On the other 90% of turns its appropriateness is about
  (0.935 − 0.099 × 0.421) / 0.901 ≈ 0.99, and the table's about 0.999.
  - With only the last two words, salad that ends like a real sentence can't be detected.
  - The fly also does worse than the table because rare salad contexts share KCs with frequent valid ones.
- **The shuffled-wiring control is uninformative on synthetic wiring.** A degree-preserving
  shuffle of random PN→KC wiring is another random network, so it is expected to match. It
  becomes a real test only on FlyWire wiring (Phase 3 data).
- **The KC layer matters.** Without it, the same rule and readout reaches appropriate 0.61.
- The fly is below the table on ask and yes/no as well: it sometimes says `ya`/`na` where the
  most frequent answer is `hm`/`na`. This costs exactness, not form.

Options (for the user):
1. Move on to 7c: the fly tracks the grammar state of the whole utterance (several flies),
   which is what salad detection needs. The 7b result stays as "not passed".
2. Keep working on 7b: more training, or a finer routing of fly 0.
3. Re-scope 7b's gate to non-salad turns. This changes a criterion after seeing results, so it
   is listed only for completeness and is not recommended.

## 7c: grammar state tracked by the fly — gate NOT passed (control)

Run `results/20261114-092733_phase7c_grammar_s0` (`configs/phase7c_grammar.yaml`, rate model).
The utterance grammar is a 14-state FSM over 21 words, with strict property order (Zumi 3).
Only valid transitions are taught, and a decoded confidence below the threshold reads as "dead" (ADR 0012).
Thresholds were chosen per run on seeds 10–14 (dev).

Test seeds 60–64, mean (min):

| run | threshold | step acc | utterance acc | salad rejected | valid accepted |
|---|---|---|---|---|---|
| **routed readout, state token** | 0.05 | 1.000 (1.000) | 1.000 (1.000) | 1.000 | 1.000 |
| teacher shuffle | 0.01 | 0.776 | 0.101 | 0.947 | 0.092 |
| **routed, no KC layer** | 0.15 | 0.997 (0.986) | **1.000 (0.999)** | 0.999 | 1.000 |
| single fly (state + word) | 0.00 | 0.254 | 0.733 | 0.385 | 0.903 |
| routed readout, state + word | 0.10 | 0.858 | 0.956 | 0.966 | 0.960 |

Speaker (7b split speaker; fly 0 smells the fly's grammar state + last user word):

| | perplexity | grammatical | appropriate | exact |
|---|---|---|---|---|
| 7c speaker | 1.77 (1.70) | 1.000 (1.000) | **0.999 (0.997)** | 0.673 (0.620) |
| 7b speaker | 1.79 | 1.000 | 0.935 (0.905) | 0.680 |
| 5-gram (7a) | 1.37 | 1.000 | 0.994 | 0.789 |

Exact by kind (7c speaker): teach 1.000 · statement 1.000 · bad **0.992** (7b: 0.421) · ask 0.502 ·
what 0.565 · where 0.319 · yes/no 0.198 (7b: 0.536).

**Verdict (ADR 0012 gate): not passed.**
- Every fly criterion is met on every test seed: step and utterance accuracy 1.000, speaker grammatical 1.000, speaker appropriate ≥ 0.9965.
- The control criterion fails. The same routing **without a KC layer** reaches utterance accuracy 0.9997, above the 0.90 limit.

Reading:
- **The 7b failure is fixed.** With the fly's own grammar state, word salad gets `e` 99.2% of
  the time (7b 42%). The speaker's form is right 99.9% of the time, above the 5-gram's 0.994.
- **But the KC layer is not what tracks the grammar.** Routing by word selects a readout, and
  within it the 14 state patterns are a linear lookup. This is the revised Gate D finding again:
  the input-gated readout (designer-set glue) is the mechanism. Without routing, the one-fly
  version (where KC conjunctions would have to do it) reaches only 0.73.
- **Content dropped on yes/no** (0.536 → 0.198). Fly 0 now sees (DONE, `ba`) for both
  `<place> ba?` and `<noun> <prop> ba?`, so their reply distributions are merged. Content is 7d's job
  (memory fly and ring), not part of this gate.

Options (for the user):
1. **Keep this architecture and go on to 7d**, recording that grammar tracking rests on the
   word-routed readout, not on the KC layer. The chatbot works; the claim about the fly is narrower.
2. **Look for a design in which the KC layer is needed**: no word routing, with the grammar split into sub-machines small
   enough for one fly (≤ 3–4 states, Phase 4). This is research with an uncertain outcome.
3. Re-run with a different control limit. This changes a criterion after seeing results; not recommended.
