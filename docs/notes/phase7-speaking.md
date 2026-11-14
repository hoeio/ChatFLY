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

## 7d: content from the flies — gate NOT passed (control), fly criteria met

Run `results/20261114-102055_phase7d_talk_s0` (`configs/phase7d_talk.yaml`, rate model, test
seeds 60–64, 200 dialogues × 30 turns per seed). Nothing was tuned in 7d.

| | exact | content exact | appropriate | grammatical | perplexity |
|---|---|---|---|---|---|
| **flies** | **0.990 (0.980)** | **0.986 (0.964)** | 0.996 (0.991) | 0.998 (0.991) | 1.20 |
| all teachers shuffled | 0.683 | 0.423 | 0.996 | 0.998 | 1.95 |
| memory without KC layer | 0.978 (0.968) | **0.963 (0.943)** | 0.996 | 0.998 | 1.23 |
| 5-gram (7a, Zumi 2 corpus) | 0.789 | ~0.62 | 0.994 | 1.000 | 1.37 |
| RNN (7a) | 0.790 | ~0.62 | 0.999 | 1.000 | 1.34 |

Exact by kind, mean (min):

| | teach | statement | bad | ask | what | where | yes/no |
|---|---|---|---|---|---|---|---|
| flies | 1.000 | 1.000 | 0.981 (0.932) | **1.000** | 0.961 (0.828) | 0.981 | 0.988 |
| shuffled | 1.000 | 1.000 | 0.981 | 0.368 | 0.667 | 0.385 | 0.357 |
| memory no KC | 1.000 | 1.000 | 0.981 | 0.975 | 0.888 (0.770) | 0.981 | 0.988 |

**Verdict (ADR 0012 gate): not passed.**
- Every fly criterion is met on every test seed: exact ≥ 0.980, content ≥ 0.964, appropriate and grammatical ≥ 0.991.
- The shuffled control is at 0.42.
- The memory-without-KC control has content exact 0.963, above the 0.70 limit.

Reading:
- **The dialogue content now comes from the flies.** Content exact 0.986, against about 0.6 for every
  7a baseline, which have no memory across turns. Facts taught earlier in the dialogue are
  answered right (ask 1.000), and so is the ring place carried across turns (where 0.98,
  yes/no 0.99).
- **The control criterion was too blunt, but it stands as written.** The no-KC control only
  replaces the memory fly. Where and yes/no come from the ring fly in both runs, which is about
  45% of content questions. On memory questions alone the KC layer helps (ask 1.000 vs 0.975,
  what 0.961 vs 0.888), but at this load (6 things per dialogue) the PN-level rule is not far
  behind. Zumi 2 showed the gap opens with more things (no KC: 0.90 at 40, 0.75 at 80).
- `what` has one weak seed (0.828): every pair must be right, including leaving untaught pairs out.
- Remaining glue, as recorded in ADR 0012: the parse routes questions to the flies, the "what" list
  is in pair order, the grammar and comparator readouts are selected by the word, and X and
  the ring's "lost" state are absorbing.

## 7e: Zumi 3, sampled replies — gate NOT passed (perplexity, memory under load)

Run `results/20261114-111206_phase7e_zumi3_s0` (`configs/phase7e_zumi3.yaml`, commit ab15a4d, rate
model, test seeds 60–64). Corpus: 200 dialogues × 30 turns per seed. Load condition: 20 dialogues × 200
turns, 40 things each. Every reply word is sampled at T = 0.1 (chosen on selection seeds 10–14,
ADR 0012). Values are means, with the minimum over seeds in brackets.

| | perplexity | exact | grammatical | appropriate | distinct-2 | max-len | memory exact |
|---|---|---|---|---|---|---|---|
| **flies** | **1.469** | **0.978 (0.967)** | 0.995 (0.985) | 0.989 (0.980) | 0.0034 | 0.0007 | 0.953 (0.906) |
| flies, load (40 things) | 1.543 | 0.953 (0.928) | 0.987 | 0.978 | 0.0045 | 0.0017 | **0.917 (0.875)** |
| control: all teachers shuffled | 2.787 | **0.670** | 0.995 | 0.988 | 0.0031 | 0.0005 | 0.427 |
| control: load, memory without KC | 2.356 | 0.742 | 0.981 | 0.969 | 0.0045 | 0.0023 | **0.534** |
| 3-gram | 1.578 | 0.615 | 0.959 | 0.910 | 0.0021 | 0 | |
| 5-gram | 1.384 | 0.758 | 0.962 | 0.958 | 0.0021 | 0.0005 | |
| 8-gram | 1.319 | 0.767 | 1.000 | 0.996 | 0.0028 | 0 | |
| RNN | 1.280 | 0.760 | 1.000 | 0.999 | 0.0032 | 0 | |
| **GRU** | **1.268** | 0.764 | 1.000 | 0.998 | 0.0032 | 0 | |

Oracle (correct) replies on the same test dialogues have distinct-2 0.0032 and 98–100 distinct
replies per seed.

**Verdict (ADR 0012 gate): not passed.**

| criterion | result |
|---|---|
| 1. Perplexity below every baseline | **failed**: 1.469 against the GRU's 1.268, as expected from selection seed 10 |
| 2. Grammatical, appropriate and exact ≥ 0.95 on every seed | passed (minimum 0.967) |
| 3. Max-length hits ≤ 0.01 | passed (≤ 0.0027) |
| 4. Diversity | the fly's distinct-2 (0.0034) matches the oracle's (0.0032), with no runaway replies |
| 5a. Memory exact under load ≥ 0.95 on every seed | **failed**: mean 0.917, minimum 0.875 |
| 5b. Shuffled control exact ≤ 0.70 | passed (0.670), by a small margin |
| 5c. No-KC memory control under load ≤ 0.90 | passed, clearly (0.534) |

Reading:
- **The sampled replies are right, and the baselines' are not.** The flies give one of the
  correct replies 97.8% of the time, against about 76% for the baselines, which have no memory across turns.
  They use the pronoun and ask back, and their diversity matches the oracle's.
- **Perplexity is the fly's weak point.** Normalized MBON rates are flat (median p(correct word)
  ≈ 0.74). Good generation needs a sharp decoder (T = 0.1), and that decoder is an interface choice.
  The gated RNN models the next-word distribution better. The fly wins on content, the GRU on
  calibrated probabilities.
- **The KC layer is shown to matter for the first time in Phase 7.** Under memory load, the no-KC
  memory rule falls to 0.53 while the KC memory keeps 0.92. This is the control 7c and 7d lacked.
  It still falls short of the 0.95 bar at 40 things, so the memory fly's capacity is the limit.
- Glue, unchanged from 7d plus: pronoun resolution (topic = last named thing), the cue pointer, and the temperature.

## 7f: larger speaker KC layer — gate NOT passed (perplexity; load memory on 2 of 5 seeds)

Run `results/20261114-121352_phase7f_zumi3_s0` (`configs/phase7f_zumi3.yaml`, commit 2ed4308).
Fresh test seeds 70–74; otherwise as 7e, except that the speaker has 8000 KCs (ADR 0012, 7f).
Values are means, with the minimum in brackets.

| | perplexity | exact | grammatical | appropriate | memory exact |
|---|---|---|---|---|---|
| **flies** | 1.459 | **0.988 (0.979)** | 0.999 | 0.999 | 0.973 (0.947) |
| flies, load (40 things) | 1.536 | 0.974 (0.961) | 0.998 | 0.998 | **0.952 (0.924)** |
| control: shuffled | 2.768 | 0.667 | 0.999 | 0.998 | 0.442 |
| control: load, memory without KC | 2.306 | 0.769 | 0.998 | 0.997 | **0.549** |
| 8-gram | 1.322 | 0.765 | 1.000 | 0.996 | |
| RNN | 1.283 | 0.750 | 1.000 | 0.998 | |
| GRU | **1.267** | 0.761 | 1.000 | 0.998 | |

Memory exact under load, by seed: 70 0.924, 71 0.956, 72 0.953, 73 0.948, 74 0.977.

**Verdict: not passed.**
- Perplexity is above the GRU's (1.459 vs 1.267).
- Memory exact under load is below 0.95 on seeds 70 and 73.
- Every other criterion passes: sampled grammatical, appropriate and exact ≥ 0.97 on every seed, no runaway replies, and both controls fail.
- Against 7e (other test seeds), the larger speaker raises load memory from 0.917 to 0.952 and exact from 0.978 to 0.988.

## Phase 7 closing summary (2026-11-14, closed at the user's request)

What the system can do, on held-out test seeds:
- **It speaks Zumi word by word.** Every reply word is sampled from MBON group rates. Replies are grammatical and appropriate 99.9% of the time. They use the pronoun `to`, ask back, and have the oracle's diversity.
- **What it says is right**, because the answer comes from flies:
  - the ring fly's place;
  - the memory fly's facts, taught earlier in the same dialogue;
  - the comparator fly's yes / no.

  Exact replies: 0.988. The n-gram, RNN and GRU baselines get about 0.75–0.77, since they cannot remember across turns.
- **The KC layer is needed for memory under load**: without it, memory exact is 0.55; with it, 0.95.

What it cannot do, and where the credit goes:
- **Perplexity is not GPT-2 like.** Normalized MBON rates are a poorly calibrated next-word distribution; a gated RNN is better (1.27 vs 1.46). Good samples need a sharp decoder (T = 0.1), which the interface sets.
- **Long "what" lists are the weak point.** The memory fly holds 40 things almost perfectly. The speaker loses rare (previous word, cue) contexts to frequent end words, so long lists sometimes break off.
- **Glue (designer rules):**
  - the parse routes questions to the flies, and the readouts are selected by the word;
  - the cue lists and their pointer;
  - pronoun resolution;
  - absorbing dead and lost states;
  - the split between the first word and later words;
  - the temperature.
- **The routed control was not ruled out for grammar or the speaker.** In 7c, grammar tracking also works without the KC layer under the same routing. The KC layer's own contribution is shown only for memory.
- The 7e/7f gate, the user's "GPT-2 level in Zumi", is **not passed**.
