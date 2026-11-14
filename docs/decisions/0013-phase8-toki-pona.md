# 0013. Phase 8: can the current fly learn a subset of Toki Pona? (hypotheses)

- Status: Accepted (plan and 8a design); 8a gate not passed (1 of 5 test seeds); 8a-2 (340 PNs) rejected at selection; 8a-3 (balanced teaching) **passed the 8a gate** on test seeds 80–84; 8b in progress, see `docs/notes/phase8-toki-pona.md`
- Date: 2026-11-14

## Context

Phase 7 closed without reaching "GPT-2 level in Zumi". The user and I agreed that general language ability is out of
reach for this model: one-layer associative readout, a decoded-state loop, a depression-and-recovery
dopamine rule. The user then asked to try the opposite way: assume it is impossible, and form
hypotheses for making the current fly learn **Toki Pona**, a small human-made language.

What Toki Pona is (verified, `docs/04-research-notes.md`):
- 120 core words (nimi pu; https://sona.pona.la/wiki/nimi_pu);
- particles carry the grammar: `li` marks the predicate (not after a bare `mi` or `sina`), `e` the object, `la` the context, `pi` groups modifiers, `o` commands.

Versus Zumi 3:
- about 5 times the output vocabulary;
- content words that serve as noun, verb or modifier;
- noun phrases of several words;
- every word is in a closed vocabulary, so there is no noun copying.

## Hypotheses (each with what would reject it)

| | hypothesis | rejected if |
|---|---|---|
| H1 | **Compact output codes beat the one-group-per-word limit.** The real fly has 48 MBONs of 35 types per side (FlyWire v783, `docs/04-research-notes.md`), fewer than the 115 output groups one word per group would need. A word as k of n MBON groups keeps accuracy at the Toki Pona vocabulary size with ≤ 24 groups. | the codes fail where one group per word works, or both fail together |
| H2 | **Particles carry the grammar**, so a 7c-style grammar fly with content words as one class tracks it (expected 15–30 states). | more than 30 states are needed, or `pi` nesting cannot be bounded |
| H3 | **KC conjunctions give noun phrases their meaning.** `jan pona` and `jan ike` are different smells to the memory fly. One modifier works; two or more degrade (7b: 3-token unions failed). | one modifier already confuses phrases, or the no-KC control does as well |
| H4 | **A closed vocabulary removes the copying problem** and turns generation into the 7f speaker problem at a larger scale. It needs H1 and/or a larger KC layer. | accuracy falls with vocabulary size whatever the KC count (a capacity limit of the rule) |
| H5 | **Learning from example sentences works** if each observed transition is taught once per epoch (7c's corpus failure was frequency weighting). | the deduplicated corpus does no better than raw corpus training |

Glue is listed and counted in every phase. Known so far:
- tokenizing;
- word-selected readouts (H2, as 7c);
- the `mi`/`sina` perspective swap;
- routing questions to the memory fly;
- the cue pointer.

## Phases

| | content | gate |
|---|---|---|
| **8a** | output capacity: the speaker says Toki Pona sentences from (state, cue); vocabulary 25 / 60 / 112; one-hot vs k-of-n codes (H1, H4) | below; if no coding passes, Toki Pona stops here |
| 8b | grammar fly for a Toki Pona subset, with table teaching and deduplicated corpus teaching (H2, H5) | set before 8b |
| 8c | memory for noun phrases with modifiers (H3) | set before 8c |
| 8d | speaker over a closed vocabulary with content from flies (H4) | set before 8d |
| 8e | a whole subset-Toki Pona conversation | set before 8e |

The starting subset is `s li p`, `s li p e o`, `mi/sina p (e o)`. `pi`, `la`, questions and `ona`
come later.

## Decisions (8a)

- **Task** (`src/chatfly/tokipona.py`).
  - The speaker smells (sentence state, cue) and says one word.
  - States: S0, SUBJ, MI, LI, PRED, E, OBJ. They are the true sentence states; 8b replaces them with a grammar fly.
  - Cues: the content words of the message in order, then `-`.
  - The table has 6 contexts per word plus 2 end contexts, 674 at V = 112. It is taught for `protocol.n_train` epochs in random order.
  - Content words: the 120 nimi pu minus the 8 particles a anu e en la li o pi, which gives 112. `mi` and `sina` are always included.
- **Measures:**
  - step accuracy over the whole table;
  - sentence exact over 500 random messages (`s li p` or `s li p e o`, object with p 0.5), said in closed loop.
- **Codings:**
  - one group per word (V + 3 groups);
  - `k2n20` (2 of 20 groups);
  - `k3n12` (3 of 12 groups).

  A word's code is a fixed random subset per seed. Teaching sets DAN on in every group outside the target's code. Reading picks the word with the highest mean rate over its groups.
- **Fixed in advance:** token density 0.15 and KC threshold 650 (the 7b–7f speaker); KC count 2000 or 8000.
- **Controls** at V = 112, for every coding and KC count: no KC layer (the rule on PNs), and a shuffled teacher.
- **Selection** (`configs/phase8a_capacity_select.yaml`, seeds 10–14).
  - The **chosen condition** is the (code, KC count) with ≤ 24 groups that has the highest mean sentence exact at V = 112. Ties go to fewer KCs.
  - The best one-hot condition is picked the same way, for reference.
- **Gate** (`configs/phase8a_capacity.yaml`, test seeds 60–64). The chosen condition must reach, at V = 112:
  - step accuracy ≥ 0.99 and sentence exact ≥ 0.95 on every test seed;
  - mean sentence exact ≤ 0.70 for both of its controls.
- **Reading the gate:**
  - If it passes, H1 is supported and 8b uses the code.
  - If it fails but one-hot passes, H1 is not supported. The vocabulary is still learnable with one group per word, and the user decides whether to continue with it.
  - If both fail, the output vocabulary is the barrier, and Toki Pona stops at 8a.

### First selection run, and the grid extension (before any test run)

- **First run** (`results/20261114-144525_phase8a_capacity_select_s0`, KC threshold 650): everything collapsed as V grew. Sentence exact, mean:

  | coding | V = 25 | V = 60 | V = 112 |
  |---|---|---|---|
  | one-hot | 0.72 | 0.38 | 0.05 |
  | `k2n20` | 0.69 | 0.24 | 0.01 |
  | `k3n12` | 0.55 | 0.10 | 0.02 |

  2000 vs 8000 KCs made almost no difference.
- **Diagnosis (selection seeds 10–11):** at threshold 650 a KC fires when 4 of its 5 claws are on, and about 2.8% of KCs fire per context, whatever the KC count. At V = 112 each KC is shared by about 19 of the 674 contexts, so more KCs do not reduce interference. At threshold 850 (all 5 claws, about 0.3% of KCs):

  | KCs | seed 10, one-hot | seed 10, `k2n20` |
  |---|---|---|
  | 8000 | 0.98 | 0.93 |
  | 16000 | 1.00 | 1.00 |
  | 32000 | 1.00 | 1.00 |

  Seed 11 stays at 0.52. Its one failing context is (PRED, `-`) → EOS, which is outvoted by the 112 (PRED, word) → `e` contexts: their PN patterns share 6 PNs (about 3 expected) and only 11 KCs fire.
- **Decision:** the selection grid gets the KC threshold as part of the KC axis: (2000, 650), (8000, 650), (8000, 850), (16000, 850), (32000, 850). The selection rule and the gate stay as written.
- **Biology:** 16000–32000 KCs is 6–12 times the 2,597 KCs of one FlyWire hemisphere (`docs/04-research-notes.md`), and an all-claws threshold is sparser than the 5–15% target of Phase 1. If the gate passes only there, it says what the circuit would need, not that the fly has it.

### Selection result (`results/20261114-145745_phase8a_capacity_select_s0`)

Means over seeds 10–14 at V = 112. The minimum is in brackets.

| coding | KCs, threshold | step | sentence |
|---|---|---|---|
| one-hot | 2000, 650 | 0.63 | 0.03 |
| one-hot | 16000, 850 | 0.999 | 0.719 (0.518) |
| **`k2n20`** | **16000, 850** | **0.998** | **0.712 (0.512)** |
| `k3n12` | 16000, 850 | 0.993 | 0.709 (0.502) |
| any coding, no KC layer | | 0.25–0.31 | 0.00 |
| any coding, shuffled | | ≤ 0.01 | 0.00 |

- **Chosen** by the rule (≤ 24 groups, best mean sentence exact): `k2n20`, 16000 KCs, threshold 850. The one-hot reference is at the same settings.
- **Finding before the test:**
  - Sentence exact is 1.00 on seeds 10 and 14, and about 0.5 on 11–13.
  - Steps are ≥ 0.996 on every seed, so the remaining error is one end-of-sentence context ((PRED, `-`) or (OBJ, `-`)). Its few KCs are shared with the many (state, word) contexts.
  - With 113 cue tokens of 22 PNs among 150 PNs, every PN is in about 17 patterns, so the empty cue `-` cannot stay distinct. This is input crowding in PN space, and it is expected to fail the "every seed ≥ 0.95" gate. The gate is not changed.

### 8a-2: a FlyWire-sized PN layer (decided by the user after the 8a gate failure; written before any run)

- **Hypothesis.** The end-of-sentence failure is input crowding: 113 cue tokens of 22 PNs among 150 PNs. FlyWire v783 has 139 uniglomerular + 201 multiglomerular ALPNs per side (`docs/04-research-notes.md`), so 340 PNs is the measured size. With the same density, the expected overlap fraction between tokens is unchanged, but rare large chance overlaps (seed 64, seed 11) become rarer. With a lower density, tokens can stay as sparse as in 8a (22 PNs) and overlap less.
- **Rejected if** no 340-PN condition beats the 150-PN condition on mean sentence exact over the selection seeds, or the chosen condition fails the gate.
- **Conditions** (`configs/phase8a2_capacity_select.yaml`; V = 112, `k2n20`):

  | PNs | token density | KCs | KC threshold |
  |---|---|---|---|
  | 150 | 0.15 | 16000 | 850 (8a's chosen condition) |
  | 340 | 0.15 | 16000 | 850 |
  | 340 | 0.10 | 16000 | 650 |
  | 340 | 0.065 | 16000 | 650 |
  | 340 | 0.065 | 32000 | 650 |

- **Selection** on seeds 10–19 (ten seeds, because the failure is per seed). The chosen condition has the highest mean sentence exact. Ties (within 0.005) go to fewer PNs, then fewer KCs, then the lower threshold.
- **Gate:** the same as 8a, on new test seeds 80–84 (`configs/phase8a2_capacity.yaml`).
  - step accuracy ≥ 0.99 and sentence exact ≥ 0.95 on every test seed;
  - mean sentence exact ≤ 0.70 for both controls (no KC layer, shuffled teacher).
- **If it fails,** the result is reported and the user decides. Criteria are not changed.

#### 8a-2 selection result (`results/20261114-154242_phase8a2_capacity_select_s0`)

Sentence exact per seed (10–19) at V = 112, `k2n20`:

| PNs, density, KCs, threshold | mean | seeds ≥ 0.95 | per seed |
|---|---|---|---|
| **150, 0.15, 16000, 850** | **0.751** | 5/10 | 1.00 0.51 0.99 0.53 0.53 0.98 1.00 1.00 0.49 0.48 |
| 340, 0.15, 16000, 850 | 0.660 | 3/10 | 0.52 0.52 0.49 1.00 0.54 1.00 0.52 0.53 1.00 0.49 |
| 340, 0.10, 16000, 650 | 0.460 | 0/10 | all 0.39–0.51 (steps 0.975) |
| 340, 0.065, 16000, 650 | 0.742 | 5/10 | 0.50 0.50 0.49 0.98 0.98 0.48 0.51 0.99 1.00 0.99 |
| 340, 0.065, 32000, 650 | 0.704 | 4/10 | 0.52 1.00 0.49 0.54 1.00 0.48 0.52 1.00 1.00 0.49 |

Controls, every condition: no KC layer steps 0.26–0.35, sentences 0.00; shuffled steps ≤ 0.01, sentences 0.00.

- **The hypothesis is rejected** by its own criterion: no 340-PN condition beats 150 PNs. The rule picks 150 PNs, which is 8a's tested condition, so no new test run is made.
- **What the data say instead.** Every condition is bimodal: a seed gives about 1.0 or about 0.5, and which seeds fail changes with the condition. About 0.5 means one of the two end contexts ((PRED, `-`) or (OBJ, `-`)) is lost, and with it every sentence of that shape. This looks like the Phase 4 majority conflict (one EOS context against 112 contexts with the same state token), decided by chance in each network, not like crowding that more PNs would remove.
- 8a-2 does not pass. The next step is the user's decision.

### 8a-3: frequency-balanced teaching (decided by the user after 8a-2; written before any run)

- **Hypothesis.** The lost end context is a majority conflict: (PRED, `-`) → EOS shares the PRED token with 112 contexts → `e`, and each is taught once per epoch, so the majority's depression wins in about half the networks. Teaching the rare context as often as the majority removes the conflict without new inputs or readouts.
- **Rule** (`tokipona.repeats`, fixed before any run). Per epoch a context is taught r times. For each of its tokens, take the most common target among table contexts with that token, divided by the count of this context's target there; r is the largest ratio, rounded.
  - `none`: r = 1 (8a).
  - `state`: state tokens only. At V = 112, (PRED, `-`) gets 112 and everything else 1; 785 presentations per epoch instead of 674.
  - `token`: state and cue tokens. Also (SUBJ, w) → `li` and (PRED, w) → `e` get 4 against the four (·, w) → w contexts; 1457 per epoch.
  - This is a teaching schedule computed from the table, not an inference rule, so it adds no glue at run time. It does add presentations, so a gain could partly be more training; `none` with the same epochs is the reference.
- **Conditions** (`configs/phase8a3_capacity_select.yaml`): 8a's chosen condition (150 PNs, density 0.15, 16000 KCs, threshold 850, `k2n20`, V = 112) with balance `none`, `state`, `token`. Controls (no KC layer, shuffled teacher) with the same schedule.
- **Rejected if** neither `state` nor `token` beats `none` on mean sentence exact over seeds 10–19, or the chosen condition fails the gate.
- **Selection:** the highest mean sentence exact. Ties (within 0.005) go to the simpler schedule (`none`, then `state`, then `token`).
- **Gate:** the same as 8a, on test seeds 80–84 (`configs/phase8a3_capacity.yaml`; the 340-PN conditions of 8a-2 never ran on them).

#### 8a-3 selection result (`results/20261114-160427_phase8a3_capacity_select_s0`)

Seeds 10–19, V = 112, `k2n20`, 150 PNs, 16000 KCs, threshold 850:

| balance | step, mean (min) | sentence, mean | per seed |
|---|---|---|---|
| `none` | 0.998 (0.994) | 0.751 | 1.00 0.51 0.99 0.53 0.53 0.98 1.00 1.00 0.49 0.48 |
| **`state`** | **0.998 (0.996)** | **0.990** | 1.00 0.98 1.00 0.98 1.00 0.97 1.00 1.00 0.98 0.99 |
| `token` | 0.984 (0.972) | 0.940 | 0.94 0.91 0.95 0.96 0.92 0.89 1.00 0.95 0.94 0.94 |

Controls with every schedule: no KC layer steps 0.21–0.28, sentences 0.00; shuffled steps ≤ 0.01, sentences 0.00.

- **Chosen:** `state`. The test config keeps only it.

#### 8a-3 test result (`results/20261114-161156_phase8a3_capacity_s0`): gate passed

| test seed | 80 | 81 | 82 | 83 | 84 |
|---|---|---|---|---|---|
| step | 1.000 | 0.997 | 0.997 | 1.000 | 0.999 |
| sentence | 1.000 | 0.994 | 0.982 | 1.000 | 0.980 |

- No KC layer: steps 0.08–0.20, sentence exact ≤ 0.002. Shuffled teacher: steps ≤ 0.015, sentence exact 0.000.
- **8a passes as 8a-3.** H1 is supported at this setting: a 2-of-20 MBON code (20 groups, fewer than the 48 MBONs per side) says all 112 content words plus `li`, `e` and EOS in sentences.
- **Limits:**
  - It needs 16000 KCs with an all-claws threshold, 6 times one hemisphere's 2,597 KCs and sparser than the Phase 1 target. It says what the circuit would need, not that the fly has it.
  - The fix is a teaching schedule: rare contexts are taught as often as the majority that shares their state token. Counting the table is done by the designer, so this is teacher glue (not run-time glue).
  - States are still the true sentence states and the cue list comes from the interface; 8b replaces the states with a grammar fly.

## Decisions (8b; written before the run)

- **Grammar** (`tokipona.grammar_task`): an FSM over word classes.
  - Classes: `B` = mi/sina, `W` = every other content word, `li`, `e`. Classing is done by the tokenizer (glue, as nouns were `N` in 7c).
  - Sentences: a subject (mi/sina alone, or a word with up to two modifiers, then `li`); a predicate with up to two modifiers; an optional `e` and an object with up to two modifiers. Content words after a head are modifiers, mi/sina included ("tomo mi").
  - 14 states (S0, S1–S3, MI, LI, P1–P3, E, O1–O3, X) over 4 inputs; 26 valid transitions. Accepting: P1–P3, O1–O3. **H2's state count holds by construction (14 ≤ 30)**; `pi` is not in this subset.
- **Fly:** 7c's method. Only valid transitions are taught; an answer below the confidence threshold reads as X (dead, absorbing).
  - **Main runs use one fly without routing:** state and word class are smelled together, so the KC conjunction has to do the work. This removes 7c's router glue.
  - 7c's routed design (`routed_shared`, state only) is a reference.
- **Teaching sources (H5):**
  - `table`: every valid transition once per epoch.
  - `table_bal`: the same with 8a-3's `repeats` over state and input tokens.
  - `dedup`: every transition seen in 500 training sentences, once per epoch.
  - `raw`: the same number of draws per epoch from the corpus occurrences (frequency-weighted, same budget).
  - Training sentences: subject mi/sina with p 0.3; 0/1/2 modifiers per phrase with p 0.5/0.3/0.2; object with p 0.5; V = 112.
- **Evaluation:** 1000 utterances per split, half valid sentences, half word salads (2–8 random words, `li` and `e` included).
  - step accuracy over all 56 (state, class) transitions, X included;
  - utterance accuracy (final state, closed loop);
  - good accepted and bad rejected.
- **Check on selection seed 10 before fixing the grid:**
  - Default KC layer (2000 KCs, threshold 650): utterance accuracy 0.51–0.74 for every source.
  - 16000 KCs, threshold 650 or 850, `table`: 1.000.
  - So the KC count and threshold join the candidates: (2000, 650), (16000, 650), (16000, 850), each with all four sources.
- **Selection** on seeds 10–14 (dev split).
  - The threshold is chosen per run from `conf_grid`: the highest mean utterance accuracy, smallest threshold on ties.
  - The **main run** is the candidate with the highest mean utterance accuracy. Ties (to 0.001) go to the order in `configs/phase8b_grammar.yaml`: fewer KCs, then table, dedup, table_bal, raw.
  - The controls get their own thresholds on the same seeds:
    - no KC layer (`single_pn`, the rule on PNs);
    - shuffled teacher.
- **Gate** (test seeds 90–94, test split). The main run must reach:
  - step accuracy ≥ 0.99 and utterance accuracy ≥ 0.99 on every test seed;
  - mean utterance accuracy ≤ 0.90 for each control.
- **H5 verdict:** `dedup` vs `raw` at the main run's KC setting, on the test seeds. It is rejected if `dedup` does not beat `raw` on step accuracy.
- **Biology:** as in 8a, 16000 KCs is 6 times a hemisphere. 2000 KCs is about the real count.

## Consequences

- 8a is cheap (about 1 s per job in the rate model) and decides whether the rest is worth doing.
- H3 is the project's core assumption (KC conjunctions) tested on a language task. 7b's three-token failure predicts a limit at two modifiers.
