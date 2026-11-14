# 0013. Phase 8: can the current fly learn a subset of Toki Pona? (hypotheses)

- Status: Accepted (plan and 8a design); 8a gate not passed (1 of 5 test seeds); 8a-2 (340 PNs) rejected at selection; 8a-3 (balanced teaching) **passed the 8a gate** on test seeds 80–84; **8b passed** on test seeds 90–94; 8b-2 (tokenizer as a fly): A and B fail; **8c passed** on test seeds 100–104; 8d: replies 0.995, gate not passed only by the class-fly control (as 8b-3); 8d-2 (unrouted memory) fails (0.815); 8e: replies 0.923, gate not passed (seed 117's grammar fly misses one transition); 8e-2 (unrouted object memory) fails (0.926); 8e-3 (re-selection) 0.956, one seed's speaker context fails; **8f (rehearsal) passed the gate** (0.997), but its test seeds had no failing context (ablation identical); on the old failing seeds rehearsal repairs all three; **8f-2 passed** on 20 fresh seeds (min 0.986; the one seed failing without rehearsal, 0.852, reaches 1.000); 8b-3: the class fly works (1.000) but the no-KC control does too (0.968), so its gate is not passed. **The user accepted it** (option 1): the class fly stays in the pipeline with the note "KC layer not needed", and from now on new glue is first tried as a fly, see `docs/notes/phase8-toki-pona.md`
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
  - Default KC layer: utterance accuracy 0.51–0.74 for every source. **Correction after the run:** this was the base config's threshold 450 (3 of 5 claws), not 650 as first written here. (2000, 650) was not checked before the run.
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

### 8b result (`results/20261114-170227_phase8b_grammar_s0`): gate passed

- **Selection** (seeds 10–14, mean utterance accuracy):
  - `table` and `table_bal` reach 1.000 at every KC setting, and `dedup` 0.998.
  - `raw` reaches 0.942–0.982.
  - The routed reference reaches 0.950.
  - The rule picks the first in order: **one fly, `table`, 2000 KCs, threshold 650**.
- **Test** (seeds 90–94; mean, minimum in brackets):

  | run | step | utterance | good accepted | bad rejected |
  |---|---|---|---|---|
  | **main: one fly, table, 2000 KCs, 650** | **1.000 (1.000)** | **1.000 (1.000)** | 1.000 | 1.000 |
  | control: no KC layer | 0.607 | 0.358 | 0.000 | 1.000 |
  | control: shuffled teacher | 0.550 | 0.299 | 0.000 | 0.83 |
  | `dedup`, 2000, 650 | 0.979 (0.964) | 0.997 (0.992) | 0.995 | 1.000 |
  | `table_bal`, 2000, 650 | 1.000 | 1.000 | 1.000 | 1.000 |
  | `raw`, 2000, 650 | 0.807 (0.786) | 0.928 (0.768) | 0.950 | 0.915 |
  | `raw`, 16000, 850 | 0.750 | 0.933 (0.763) | 0.966 | 0.907 |
  | routed (7c design), table, 2000, 650 | 0.975 | 0.952 (0.762) | 0.935 | 1.000 |

- **Gate passed:** main step and utterance accuracy are 1.000 on every test seed, and both controls are ≤ 0.36.
- **H2 supported** for this subset: 14 states, and the KC conjunction of (state, class) learns them **without routing**, at the real KC count (2000) and 4-of-5-claw sparsity. In 7c the routed readout did the work; here the no-KC control fails (0.36). The difference from 7c is the input side: 4 word classes instead of 21 words, and a sparser KC threshold (650, not 450).
- **H5 supported:** `dedup` beats `raw` at the same budget (step 0.979 vs 0.807; utterance 0.997 vs 0.928). Raw frequencies under-teach rare transitions, as in 7c and 8a. `dedup` misses 0–2 of the 26 transitions that the 500 training sentences never show (mi/sina as a modifier is rare). Those transitions read as X, so step accuracy stays below 0.99.
- **Limits:**
  - The tokenizer gives word classes, so the fly never sees which content word it is (glue: classing).
  - The subset has no `pi`, `la`, `o`, questions or several predicates.
  - The rate model only.

## Decisions (8b-2: the tokenizer's classes; asked by the user; written before the runs)

The user asked whether the tokenizer's word → class step can be a fly. Two ways, both run.

### A: a class fly in series (`configs/phase8b2_classfly.yaml`)

- **Class fly** (`tokipona.ClassFly`): it smells one word (114: content words, `li`, `e`) and says its class (one MBON group per class).
  - It is taught from the word list's labels (the dictionary), every word per epoch.
  - The decoded class is the grammar fly's input (a fixed rule, like the state loop). A silent or tied class fly gives X.
  - The class knowledge moves from the run-time tokenizer into the teacher's labels; it is still set by the designer.
- **Grammar fly:** 8b's chosen run (one fly, table, 2000 KCs, threshold 650; confidence threshold 0.1), unchanged.
- **Check on selection seed 10:** with 2000 KCs, threshold 650 and no balance, class accuracy is 0.965 and utterance accuracy 0.904. 110 `W` words outvote the 4 others (the 8a-3 majority conflict). So the class fly's condition is selected.
- **Conditions,** in tie order: (2000, 650) and (16000, 850), each with balance `none` or `class`. `class` teaches each word (largest class size / its class size) times per epoch, which is 8a-3's rule with the class as the only key.
- **Selection** on seeds 10–14 (dev split): the highest mean utterance accuracy of the class fly → grammar fly pipeline. Ties (to 0.001) go to the listed order.
- **Gate** on 8b's test seeds 90–94 (the grammar fly there is the tested one):
  - class accuracy 1.0 (all 114 words) and utterance accuracy ≥ 0.99 on every seed;
  - mean utterance accuracy ≤ 0.90 for both class-fly controls (no KC layer, shuffled teacher).
  - If the no-KC control also passes, the class step is a lookup that does not need the KC layer. That is recorded, as in 7c.

### B: no classes (`configs/phase8b2_words.yaml`)

- The grammar fly reads the words themselves: 14 states × 114 inputs, 1126 valid transitions (`tokipona.word_task`).
- Random word patterns share nothing, so the fly cannot carry what it learns about one word to another. Each (state, word) transition must be taught.
- **Runs:**
  - `table` (every valid transition; built from the classes, so it is an upper bound) at (2000, 650), (16000, 650), (16000, 850);
  - `dedup` from 500, 2000 and 8000 training sentences at the same three settings.
- **Procedure:** as 8b. Thresholds are chosen per run on seeds 10–14. The "main" run is the best `table` run; its controls are no KC layer and a shuffled teacher. Everything runs on test seeds 90–94.
- **B is a measurement, with no gate.** It reports:
  - whether word-level table teaching reaches 8b's level (step and utterance ≥ 0.99 on every test seed);
  - the smallest corpus where `dedup` does, if any.

  Expected: the table works with enough KCs. `dedup` is limited by coverage, because unseen (state, word) pairs read as dead.

### 8b-2 results: both fail

**A** (`results/20261114-171133_phase8b2_classfly_s0`):
- **Selection** (mean utterance accuracy): (2000, 650) 0.900 with no balance and 0.900 with `class` balance; (16000, 850) 0.455 either way. The rule picks (2000, 650, none).
- **Test** (seeds 90–94):

  | | class accuracy, mean (min) | utterance (class fly → grammar fly), mean (min) |
  |---|---|---|
  | class fly | 0.953 (0.930) | 0.881 (0.799) |
  | no KC layer | 0.965 | 0.486 |
  | shuffled teacher | 0.218 | 0.360 |
  | (8b tokenizer, same grammar fly) | | 1.000 |

- **Gate not passed.**
- **Diagnosis** (after the gate, nothing changed): the wrong words are mostly silent, not misclassified. On seeds 90 and 91, 3 words each give no class at all (for example `pan`, `pimeja`, `sin`). One word is a single token of 22 PNs. At threshold 650 a KC needs 4 of its 5 claws inside that set, so some words drive no KCs. The thresholds were set for two-token stimuli. The same explains (16000, 850) doing worse. Class balance adds a few misclassifications (for example `walo` → `B`).

**B** (`results/20261114-171206_phase8b2_words_s0`), test seeds 90–94:

| run | step, mean (min) | utterance, mean (min) | good accepted | bad rejected |
|---|---|---|---|---|
| table, 16000, 850 (chosen) | 0.949 (0.928) | 0.617 (0.442) | 0.419 | 0.908 |
| table, 2000, 650 | 0.927 | 0.453 | 0.069 | 0.965 |
| table, 16000, 650 | 0.966 | 0.466 | 0.046 | 0.956 |
| dedup 500 / 2000 / 8000 sentences, 16000, 850 | 0.943 / 0.982 / 0.984 | 0.690 / 0.606 / 0.629 | 0.27–0.54 | 0.93–0.99 |
| controls: no KC layer / shuffled | 0.43 / 0.31 | 0.36 / 0.36 | 0.00 | 1.00 |

- 500 sentences show 843 of the 1126 valid transitions; 2000 show 1102; 8000 show all.
- **No run reaches 0.99.** Even the full table gives 0.62. Steps are 0.93–0.98, and a sentence of 5–10 words compounds the error. Most valid sentences are refused.
- With 1126 contexts this is 8a's interference at a larger size: 8a needed 16000 KCs at threshold 850 and balanced teaching for 674 contexts. And the fly cannot share what it learns across words.
- So without the tokenizer's classes, the current fly does not track the grammar.

### 8b-3: the class fly with single-word settings (the user chose option 1 after 8b-2; written before the run)

- **Hypothesis.** 8b-2 A failed because single words were silent at threshold 650. A threshold or token density made for single-word stimuli gives every word a class, and the pipeline then reaches 8b's level.
- **Conditions** (`configs/phase8b3_classfly.yaml`), in tie order. The grammar fly is unchanged (8b's run, threshold 0.1).
  - 2000 KCs, threshold 450 (3 of 5 claws, the base default), no balance;
  - 2000 KCs, threshold 550, no balance;
  - 2000 KCs, threshold 650, word tokens at density 0.3 (44 PNs);
  - 2000 KCs, threshold 450, `class` balance.
- **Selection:** as 8b-2 A, on seeds 10–14 (dev split).
- **Gate:** as 8b-2 A, on new test seeds 95–99:
  - class accuracy 1.0 and pipeline utterance accuracy ≥ 0.99 on every seed;
  - mean utterance accuracy ≤ 0.90 for each control.
  - A passing no-KC control is recorded as "the class step does not need the KC layer".
- The classes are still the dictionary's labels (designer knowledge in the teacher).

#### 8b-3 result (`results/20261114-172601_phase8b3_classfly_s0`)

- **Selection** (mean pipeline utterance accuracy):
  - 0.484 (450, none), 0.485 (550, none), 0.484 (650, density 0.3).
  - **1.000 (450, `class` balance)**, which is chosen.
  - Without balance at these settings, the 110 `W` words take over the other classes.
- **Test** (seeds 95–99):

  | | class accuracy, mean (min) | pipeline utterance, mean (min) |
  |---|---|---|
  | **class fly → grammar fly** | **1.000 (1.000)** | **0.999 (0.996)** |
  | (tokenizer → same grammar fly) | | 0.999 (0.996) |
  | no KC layer | 0.996 (0.982) | 0.968 (0.838) |
  | shuffled teacher | 0.256 | 0.367 |

- The class fly itself passes: every word is classed right on every seed, and the pipeline matches the tokenizer. The one error on seed 99 is the grammar fly's.
- **Gate not passed, by the control rule.** The no-KC control's mean (0.968) is above 0.90; it is perfect on 4 of 5 seeds. As written in advance, the class step is a lookup that does not need the KC layer: 114 random word patterns into 4 classes are almost linearly separable on PNs (as 7c's routed readout).
- **What it shows:**
  - The tokenizer's word → class step can be done by a fly, with balanced teaching and a single-word threshold.
  - The KC layer is not what does it. The class labels still come from the dictionary.

## Decisions (8c: memory for noun phrases, H3; written before the run)

- **Memory fly:** Zumi 2's (`zumi.Memory`: 5 pairings per fact, one sniff per question, confidence threshold 0.2).
  - Its properties are 4 Toki Pona predicate words, each a yes/no output pair.
  - A phrase smells like the union (PN max) of its words' patterns, like a mixture of odors. This is the interface rule. Word order is not in the smell, so "jan pona" and "pona jan" are the same smell (a known limit, not tested).
- **Facts** (`tokipona.phrase_sets`): a fresh fly per level, 40 phrases per level, 2 of the 4 predicates each with a random yes/no (80 facts).
  - Level 0: 40 bare words.
  - Level 1: 10 heads × 4 shared modifiers. Every modifier is used with every head, so a phrase's value is not the sum of its words' values; a conjunction is needed.
  - Level 2: 10 heads × 2 × 2 shared modifiers.
  - Never-taught probes: unused words (level 0), or the taught heads with a never-used modifier.
- **Measure:** correct = answered with confidence ≥ 0.2 and right, over all 80 facts. Coverage and the probes' answered rate are reported.
- **Check on selection seed 10:** Zumi 2's KC setting (2000 KCs, threshold 450) gives level 0 1.000, but level 1 0.487 and level 2 0.400 (no KC layer: 0.575 / 0.463 / 0.525). Unions of 2–3 words drive more KCs (the 8a interference). So the KC setting is chosen.
- **Conditions,** in tie order: (2000, 450), (2000, 650), (16000, 650), (16000, 850).
- **Selection** on seeds 10–14: the highest mean level-1 correct.
- **Gate** (test seeds 100–104):
  - level-1 correct ≥ 0.95 on every seed;
  - mean level-1 correct ≤ 0.80 for each control (no KC layer, shuffled teacher).
- **H3 reading:**
  - Supported if the gate passes and level 2 is below level 1, as 7b predicts.
  - Rejected if level 1 fails, or the no-KC control does as well.
  - If level 2 also reaches 0.95, the predicted two-modifier limit is not seen.

### 8c result (`results/20261114-173036_phase8c_phrase_memory_s0`): gate passed

- **Selection** (mean level-1 correct, seeds 10–14): 0.555 (2000, 450), 0.855 (2000, 650), 0.860 (16000, 650), **0.980 (16000, 850)**, which is chosen.
- **Test** (seeds 100–104; correct = answered and right, over 80 facts; mean, minimum in brackets):

  | level | memory fly | no KC layer | shuffled | coverage | answered accuracy | never-taught probes answered |
  |---|---|---|---|---|---|---|
  | 0 (bare word) | 0.565 (0.475) | 0.613 | 0.275 | 0.565 | 1.000 | 0.000 |
  | **1 (one modifier)** | **0.990 (0.975)** | **0.500** | **0.485** | 0.990 | 1.000 | 0.000 |
  | 2 (two modifiers) | 0.823 (0.775) | 0.483 | 0.413 | 0.828 | 0.994 | 0.015 |

- **Gate passed:** level 1 is ≥ 0.975 on every seed, and both controls are ≤ 0.50.
- **H3 supported:**
  - "jan pona" and "jan ike" are kept apart where the same modifiers go with every head. The no-KC rule cannot do it (0.50; it also answers 69% of never-taught phrases).
  - Two modifiers degrade (0.82) as 7b predicted. The loss is coverage (the fly declines to answer); what it does answer is still 0.99 right.
- **New limit:** at this setting a bare word is answered only 57% of the time (all of those right). A single word's PNs drive almost no KCs at the all-claws threshold (as in 8b-2). No single KC setting serves both words and phrases: the 2000/450 setting gives bare words 1.000 (seed 10 check) but phrases 0.55. 8d has to deal with this, for example with a memory fly per phrase length.
- **Glue:** the phrase union is an interface rule (a mixture of the words' odors). The phrase boundaries come from the grammar fly's states.

## Decisions (8d: a question-answering exchange carried by flies, H4; written before the run)

- **Exchange** (`src/chatfly/tokipona_chat.py`):
  - The user teaches 80 facts by statements: "jan pona li suli", or "jan pona li suli ala" for no.
  - Then the user asks 120 questions: "jan pona li suli anu seme?", the `anu seme` question form verified in `docs/04-research-notes.md`. 80 are about taught facts; 40 use a subject never mentioned (a head with an unused modifier).
  - The fly replies with a whole sentence: the statement, the statement with `ala`, or "mi sona ala" (I don't know).
  - Subjects are a head plus one modifier (8c level 1); there are 4 predicate words.
- **8d grammar** (`grammar_d`): 11 states over 6 classes (mi/sina, other content words, `li`, `ala`, `anu`, `seme`). A subject, `li`, a one-word predicate, then nothing (P), `ala` (PN) or `anu seme` (Q).
- **Flies**, each with the setting chosen in its own phase (nothing is selected in 8d):
  - class fly (8b-3: 2000 KCs, threshold 450, class balance), over the 8d classes;
  - grammar fly (8b: one fly, table, 2000 KCs, threshold 650, confidence 0.1), on the 8d grammar. It reads the user's words, and it gives the speaker its state while the speaker talks (closed loop);
  - memory fly (8c: 16000 KCs, threshold 850, confidence 0.2);
  - speaker (8a-3: `k2n20`, 16000 KCs, threshold 850, state balance). Its states are the grammar fly's, and its table is the reply sentences.
- **Glue** (interface rules):
  - subject and predicate roles are read off the grammar fly's states;
  - the predicate word selects the memory's output pair (as Zumi 2's property pairs);
  - the reply plan (cue list) is built from the question's words and the memory's answer;
  - the cue pointer.

  Following the user's standing request, a fly replacement for a glue item is tried after 8d's own test.
- **Check on selection seed 10:** reply exact 0.992.
- **Controls,** one per fly role: the memory fly, the speaker and the class fly each without a KC layer.
- **Gate** (test seeds 105–109):
  - reply exact ≥ 0.95 on every seed, over all 120 questions;
  - mean reply exact ≤ 0.80 for each control.

  A control that passes is recorded as "that fly's KC layer is not needed here".

### 8d result (`results/20261114-184117_phase8d_chat_s0`)

| test seed | 105 | 106 | 107 | 108 | 109 | mean |
|---|---|---|---|---|---|---|
| **flies** | 1.000 | 0.992 | 0.983 | 1.000 | 1.000 | **0.995** |
| memory fly without KC layer | 0.450 | 0.492 | 0.383 | 0.450 | 0.500 | 0.455 |
| speaker without KC layer | 0 | 0 | 0 | 0 | 0 | 0.000 |
| class fly without KC layer | 1.000 | 0.992 | 0.692 | 1.000 | 1.000 | 0.937 |

- With the flies:
  - all 80 statements are read and taught on every seed;
  - taught questions are answered exactly 0.993 of the time, unknown subjects 1.000 ("mi sona ala");
  - every reply is grammatical by the grammar fly.
- **The flies pass the reply criterion** (≥ 0.983 on every seed). The memory and speaker controls fail, so those KC layers are needed.
- **Gate not passed, by the class-fly control** (0.937 > 0.80). This is the 8b-3 finding again, which the user accepted: the class step does not need its KC layer. On seed 107, 30% of words are misread without it.
- **H4 supported** for this exchange: with a closed vocabulary, every reply word comes from the speaker fly, with content from the memory fly and no copying.

### 8d-2: the memory fly without routing (the user's standing request; written before the run)

- **Glue replaced:** in 8d the predicate word selects the memory fly's output pair (routing). In 8d-2 the memory fly smells the subject words and the predicate together (a three-word union) and has one yes/no pair, so the KC conjunction must keep the facts apart.
- **Expectation:** three-word unions behaved like 8c's level 2 (0.82 at 16000 KCs), so more KCs may be needed.
- **Conditions** (memory KCs, threshold 850), in tie order: 16000, 32000, 64000. The other flies are as in 8d.
- **Selection:** the highest mean reply exact on seeds 10–14.
- **Gate** (new test seeds 110–114, `configs/phase8d2_chat.yaml`):
  - reply exact ≥ 0.95 on every seed;
  - mean ≤ 0.80 for the memory fly without its KC layer.
- **Biology:** 64000 KCs is 25 times a hemisphere.

#### 8d-2 result (`results/20261114-184338_phase8d2_chat_s0`): gate not passed

- **Selection** (mean reply exact): 0.782 (16000), 0.793 (32000), 0.793 (64000). The rule picks 32000.
- **Test** (seeds 110–114):
  - reply exact 0.815 (min 0.808), taught questions 0.728, unknown subjects 0.990;
  - without the memory fly's KC layer, 0.373.
- **Diagnosis** (seed 110, after the gate, nothing changed): the unrouted memory declines 31% of taught facts (confidence below 0.2) and answers none wrong. Three-word unions spread each fact over more shared KCs, and more KCs do not help (16000 to 64000), like 8c's level 2.
- **Verdict:** replacing the predicate routing with a KC conjunction fails at this rule. The routing stays as recorded glue, and the 8d pipeline (routed memory) is kept.

## Decisions (8e: a whole conversation; written before the run)

- **Conversation** (`tokipona_chat._ejob`): 300 turns per test seed, in random order with these probabilities:
  - teach a yes/no fact (0.3): "S li P" / "S li P ala";
  - teach an object fact (0.2): "S li P e O";
  - ask yes/no (0.2): "S li P anu seme?";
  - ask what (0.2): "S li P e seme?", reply "S li P e O";
  - ask about a never-mentioned subject (0.1): reply "mi sona ala".

  Facts may be re-taught with a new value (a change of mind). Questions about changed facts are scored separately. Subjects are 40 (head + modifier); there are 4 yes/no predicates, 2 object predicates and 8 objects.
- **8e grammar** (`grammar_e`): 8d's plus `e`: P + `e` → E; E + word → O; E + `seme` → OQ. 14 states over 7 classes.
- **New fly: object memory** (`ObjectMemory`).
  - (subject phrase, predicate) → object word, as a `k2n20` MBON code over the 8 objects (8a's H1 code).
  - The predicate selects the weight set: the same routing glue as the yes/no memory.
  - Teaching depresses every group outside the object's code, 5 times. Asking reads the best object; it declines when (top − second) / top is below the threshold.
- **The other flies keep their 8d settings.**
  - The speaker's table gains (P, object) → `e`, (E, object) → object, and (O, `-`) → EOS.
  - The class and grammar flies are retrained on the 8e classes and grammar with their 8d settings.
- **Check on selection seed 10:** reply exact 1.000 over 147 questions. 8d is unchanged by the code (seed 10: 0.992 as before).
- **Selection** (seeds 10–14): the object memory's KC count (16000, 32000) × threshold (0.05, 0.1, 0.2), in that tie order. The highest mean reply exact wins.
- **Controls:**
  - both memory flies without their KC layers;
  - the speaker without its KC layer.
- **Gate** (test seeds 115–119):
  - reply exact ≥ 0.95 on every seed, over all questions;
  - mean ≤ 0.80 for each control.
- **Glue, new:** routing the object memory by predicate. A fly replacement (unrouted, subject + predicate smelled together) follows the test, as in 8d-2.

### 8e result (`results/20261114-192004_phase8e_chat_s0`): gate not passed

- **Selection** (mean reply exact): 0.988 / **1.000** / 0.995 (16000 KCs at thresholds 0.05 / 0.1 / 0.2) and 0.992 / 0.999 / 0.999 (32000). Chosen: 16000 KCs, threshold 0.1.
- **Test** (seeds 115–119, 123–152 questions per seed):

  | seed | 115 | 116 | 117 | 118 | 119 | mean |
  |---|---|---|---|---|---|---|
  | reply exact | 0.963 | 0.993 | **0.730** | 0.993 | **0.935** | 0.923 |
  | yes/no | 1.000 | 1.000 | 0.981 | 0.979 | 0.863 | 0.965 |
  | what | 0.909 | 0.983 | 1.000 | 1.000 | 0.980 | 0.974 |
  | unknown subject | 1.000 | 1.000 | **0.000** | 1.000 | 1.000 | 0.800 |
  | after a change of mind | 0.938 | 1.000 | 1.000 | 1.000 | 1.000 | 0.988 |

  - Both memory flies without KC layers: 0.470. The speaker without its KC layer: 0.000.
  - Statements were read on every seed (1.000).
- **Diagnosis** (after the gate, nothing changed):
  - **Seed 117:** the grammar fly, trained on the 8e grammar with 8b's settings, misses one transition, S0 + `mi`/`sina` → MI (low confidence, read as dead). So "mi sona ala" breaks after "mi". Its step accuracy is 0.990; the other test seeds are 1.000. The 8e grammar fly was not validated on its own before the test.
  - **Seed 119:** of the 63 yes/no-form questions, 12 got "mi sona ala" (some of them correctly, for unknown subjects) and 7 were malformed replies. This is memory under conversational load plus some speaker errors.
- **What works:** objects, "what" questions (0.974), changes of mind (0.988), and the KC layers of both memories and the speaker are needed.

### 8e-2: the object memory without routing (standing request; written before the run)

- The object memory smells the subject words and the predicate together, with one weight set (`routing: none`).
- **Selection** on seeds 10–14 (mean reply exact): KC count 16000, 32000, 64000 × threshold 0.05, 0.1, 0.2, in that tie order.
- **Gate** (new test seeds 120–124, `configs/phase8e2_chat.yaml`). It judges the replaced glue:
  - "what" questions exact ≥ 0.95 on every seed;
  - mean ≤ 0.80 without the memory KC layers.
- Expected to fail like 8d-2 (three-word unions). The object code has 8 choices, not a yes/no.

#### 8e-2 result (`results/20261114-192532_phase8e2_chat_s0`): gate not passed

- **Selection:** 64000 KCs, threshold 0.1 (mean reply 0.950). The other conditions gave 0.913–0.946.
- **Test** (seeds 120–124):
  - "what" questions 0.926 (min 0.875), against the 0.95 gate;
  - unknown subjects 0.850 (min 0.688): the unrouted memory answers about subjects it never heard, because they share the predicate's smell;
  - all replies 0.936.
  - Without the memory KC layers, "what" questions fall to 0.265.
- **Verdict:** closer than 8d-2, but replacing the object routing with a three-word conjunction still fails, even at 64000 KCs (25 times a hemisphere). Both predicate routings stay as glue.

### 8e-3: the grammar fly and the yes/no memory chosen for the 8e conversation (the user chose option 3; written before the run)

- **Stage 1, grammar fly** on the 8e grammar.
  - The conditions (`configs/phase8e3_chat.yaml`, in tie order) vary the KC count, threshold, source (`table` or 8a-3-style `table_bal`) and confidence threshold. The first is 8b's setting.
  - Chosen: the most selection seeds (10–19) with step accuracy 1.0, then the highest mean.
- **Stage 2, yes/no memory,** with the chosen grammar fly.
  - Confidence threshold 0.1 / 0.2 / 0.3 × 5 or 10 pairings per fact.
  - Chosen: the highest mean reply exact on seeds 10–14; ties in the listed order.
  - The object memory and speaker stay as in 8e.
- **Gate:** 8e's, on new test seeds 125–129:
  - reply exact ≥ 0.95 on every seed;
  - mean ≤ 0.80 for each control.

#### 8e-3 result (`results/20261114-194344_phase8e3_chat_s0`): gate not passed

- **Stage 1** (grammar fly, seeds 10–19): 8b's setting (2000 KCs, threshold 650, `table`, confidence 0.1) has step accuracy 1.0 on 10/10 seeds. 16000 KCs ties it, and the lower confidence thresholds do worse (1/10). The tie order keeps 8b's.
- **Stage 2** (yes/no memory, seeds 10–14): 1.000 at threshold 0.2 with 5 pairings, tied with others. The order keeps 8c's. So 8e-3 runs 8e's flies unchanged, on new seeds.
- **Test** (seeds 125–129): reply exact 1.000 / 0.993 / **0.802** / 0.993 / 0.993 (mean 0.956).
  - Without the memory KC layers, 0.465; without the speaker's, 0.000.
- **Diagnosis, seed 127** (after the gate):
  - 32 of 162 replies have a broken frame, for example "kili li sike li tawa e lili" for "kili sike li tawa e seme?". The speaker says `li` where it should say the modifier `sike` (state S1). That context is outvoted by the 112 (S2, word) → `li` contexts, the 8a failure mode in the reply table.
  - The grammar fly then accepts the broken reply, because on this seed it answers the untaught (P, `li`) instead of reading it as dead. Its step accuracy is 0.990; the other seeds are 1.000.
- **Reading:**
  - Across 8e and 8e-3, 8 of 10 test seeds are ≥ 0.96.
  - The two failures are single contexts in a fly (grammar on 117; speaker plus grammar on 127), each going wrong on one seed. Per-seed reliability of single contexts is the limit, as in 8a.

### 8f: rehearsal for single contexts (the user chose option 1 after 8e-3; written before the run)

- **Problem:** the 8e and 8e-3 failures are single contexts outvoted on one seed: (S0, `mi`) → MI in the grammar fly (seed 117), and (S1, modifier) → modifier in the speaker (seed 127, outvoted by the 112 (S2, word) → `li` contexts). 8a-3's state balance does not touch the speaker case: every S1 context has its own target, so all its repeats are 1. Cue balance would teach (S2, `sike`) → `li` more, the wrong way (it also did worse in 8a-3).
- **Change (H6):** after the usual teaching, the teacher runs `rehearse` rounds. Each round asks the fly every taught context and teaches again, once, the ones it gets wrong. For the grammar fly these are the valid transitions only (dead ones stay untaught, as in 8b). This uses the same teaching signal as before. The only glue is the teacher's schedule (which contexts, how often), like 8a-3's balance.
- **Selection:** seeds 10–19 (`configs/phase8f_chat.yaml`). Stage 1: grammar fly with `rehearse` 1 or 3 (8b's setting otherwise), most seeds at step accuracy 1.0, then mean, ties to the listed order. Stage 2: speaker `rehearse` 1 or 3, mean reply exact, ties to the listed order. Everything else is 8e-3.
- **Test:** new seeds 130–134.
- **Gate (same as 8e):** reply exact ≥ 0.95 on every test seed; mean ≤ 0.80 for `memory_no_kc` and `speaker_no_kc` (both with rehearsal).
- **Ablation, not gated:** `no_rehearse` (both flies without rehearsal) on the same test seeds, to show what rehearsal adds there.
- **What would count against H6:** a seed still fails with one context wrong after rehearsal, or a no-KC control with rehearsal clears 0.80 (then the schedule, not the KC code, does the work).

#### 8f result (`results/20261114-202506_phase8f_chat_s0`): gate passed, but the test seeds do not test H6

- **Selection** (seeds 10–19): `rehearse` 1 and 3 tie in both stages (grammar fly: step accuracy 1.0 on 10/10; speaker: reply exact 0.997), so the listed order picks 1 for both.
- **Test** (seeds 130–134): reply exact mean 0.997, min 0.993; "what" questions, unknown subjects and changes of mind 1.000 on every seed.
  - `memory_no_kc` 0.476, `speaker_no_kc` 0.000, both with rehearsal. **Gate passed.**
  - **Ablation `no_rehearse`: identical (0.997, min 0.993).** None of these five seeds had an outvoted context, so the pass is the 8 of 10 good-seed rate, not evidence for rehearsal.
- **Diagnosis after the gate**, on the old failing test seeds (already used, so not a test; `diag_old_seeds.txt` in the run folder). Without rehearsal the old values come back exactly:

  | seed | no rehearsal | rehearse 1 |
  |---|---|---|
  | 117 (grammar: S0 + `mi`) | 0.730 | 0.967 |
  | 119 (memory under load, speaker) | 0.935 | 0.992 |
  | 127 (speaker: S1 modifier) | 0.802 | 1.000 |

- **Reading:**
  - One round of re-teaching the wrong contexts repairs every single-context failure seen in 8e and 8e-3.
  - The no-KC speaker stays at 0.000 with rehearsal, so the schedule does not do the work by itself.
  - Over the 15 test seeds of 8e, 8e-3 and 8f, the chosen setting is ≥ 0.96 on all 15, but 3 of them are seeds it was diagnosed on. A clean test of H6 needs seeds where the plain fly fails. The next step would be a larger fresh test set, about 20 seeds, so that about 4 failing seeds are expected.
  - Rehearsal is teacher glue: the teacher checks the fly's answers. It is the same kind as 8a-3's balance.

### 8f-2: rehearsal on 20 fresh seeds (the user agreed after 8f; written before the run)

- **Why:** 8f's test seeds had no failing context, so they could not test H6. The plain-fly failure rate so far is about 2 in 10 seeds, so 20 seeds should hold about 4 failing ones.
- **Setting:** 8f's choice fixed, no selection: `rehearse` 1 for the speaker and the grammar fly (`configs/phase8f2_chat.yaml`). Test seeds 135–154.
- **Gate:**
  - reply exact ≥ 0.95 on every seed;
  - mean ≤ 0.80 for `memory_no_kc` and `speaker_no_kc`;
  - **H6:** on every seed where `no_rehearse` is below 0.95, rehearsal reaches ≥ 0.95.
- If no seed fails without rehearsal, H6 is still untested, and that will be reported as such.

#### 8f-2 result (`results/20261114-203740_phase8f2_chat_s0`): gate passed, H6 met on the one failing seed

- **Test** (seeds 135–154): reply exact mean 0.998, min 0.986; every seed ≥ 0.95.
  - `memory_no_kc` 0.495, `speaker_no_kc` 0.000.
- **`no_rehearse`:** mean 0.990. One seed is below 0.95: **seed 137, 0.852 → 1.000 with rehearsal** ("what" questions are its weak part without rehearsal, 0.755 at the minimum). The other 19 seeds are the same with and without rehearsal.
- **Gate:** all three criteria are met.
- **Reading:**
  - With rehearsal, all 25 fresh test seeds of 8f and 8f-2 reach ≥ 0.986. Without it, 1 of those 25 fails.
  - H6 now has one fresh failing seed rescued, plus the three old seeds from the diagnosis. That is consistent, but n = 1 on fresh seeds.
  - The plain-fly failure rate here (1 in 20) is lower than in 8e / 8e-3 (3 in 10). With small counts this is within chance.
  - Rehearsal never made a seed worse.

## 8g: the reply plan and "mi sona ala" as a fly (the user chose this glue item after 8f-2; written before the run)

- **Glue removed:**
  - both memories' confidence thresholds (yes/no 0.2, object 0.1);
  - the reply branch that builds the cue list (answer, `ala` or not, or "mi sona ala") from the memory's answer.
  - The cue list is now always the question's own words, plus the object memory's top word for "what" questions. The speaker decides whether to answer, deny or say "mi sona ala".
- **Interface (fixed, unlearned):** a memory's output scores reach the speaker as a mixture of four odors (PN patterns on the speaker's PNs), each scaled 0..1. With top and second the two highest scores:
  - `yes` / `no` = (top − second) / top on the winning side (yes/no memory);
  - `sure` = the same for the object memory;
  - `doubt` = second / top.
  - **Why channels and not the raw rates:** with raw rates over max, "yes" (1, 0.3) is a PN subset of "unknown" (1, 1). Under depression learning the subset context loses: on selection seed 10 this gave yes/no 0.0. The same holds for odor-free contexts, so every reply carries an odor and the odor-free reply table is not taught.
- **Teaching:**
  - The speaker's table is the reply sentences over the conversation's lexicon (heads, modifiers, predicates, objects), each context with an odor class (yes, no, sure, unknown).
  - The odors come from a **practice session.** Before the conversation, throwaway copies of the two memories are told `n_practice` facts, drawn as in a conversation, so some are changes of mind. The speaker is taught with the odors those copies then give, one drawn per presentation.
  - On seed 10, hand prototypes (1 on one channel) failed for objects. Practice at 20 facts was too light a load: 0.925. At 60 facts it gave 0.973.
  - The practice is teacher glue, like the balance and the rehearsal.
  - Teaching once per context (no state balance: with the new contexts `mi` became the majority in S0 and the balance taught 102 contexts 12 times). Then rehearsal.
- **Selection** (seeds 10–19, `configs/phase8g_chat.yaml`): `n_practice` 60 or 120 × `rehearse` 1 or 3, mean reply exact, ties to the listed order. Everything else is 8f-2.
- **Test:** new seeds 155–164.
- **Gate:**
  - reply exact ≥ 0.95 on every test seed;
  - mean ≤ 0.80 for each control:
    - `memory_no_kc` (both memories without KC layer, so the odor carries no fact);
    - `speaker_no_kc`;
    - `odor_off` (the speaker gets no odor at test). On seed 10, the odor-free speaker still says object replies right (1.000) but fails on yes/no (0.321) and unknown subjects (0.0).
  - Reference: 8f-2 with the glue, 0.998.

### 8g result (`results/20261114-210100_phase8g_chat_s0`): gate not passed

- **Selection** (seeds 10–19): mean reply exact 0.898 / 0.910 / 0.899 / 0.909. `n_practice` 120 with `rehearse` 1 was chosen. Already below 0.95 at selection.
- **Test** (seeds 155–164): reply exact mean 0.877, min 0.725 (seed 160).
  - yes/no 0.972 (min 0.912);
  - unknown subjects 0.985 (min 0.889);
  - **"what" questions 0.717 (min 0.269);**
  - changes of mind 0.894.
- **Controls:** `memory_no_kc` 0.302, `speaker_no_kc` 0.000, `odor_off` 0.591. All three are ≤ 0.80; `odor_off` still says object replies right, 1.000, as on seed 10.
- **Verdict:** the first criterion fails on 8 of 10 seeds.
- **Diagnosis** (after the gate, on selection seeds 11 and 12, 54 object questions each):
  - The main failure is "mi sona ala" for a fact the object memory holds, with its top word right: 17 and 22 cases.
  - Broken frames: 1 and 6.
  - The object memory's `sure` channel is structurally weak, about 0.3–0.45. In its 2-of-20 code the runner-up object shares a group with the winner. An odor scaled to 0.3 barely drives the speaker's KCs at the all-claws threshold, so known facts and unknown subjects smell alike: both are mostly `doubt`.
  - The old glue threshold was 0.1 on the same quantity.
  - The yes/no memory gives `yes`/`no` at about 0.7, which the speaker reads, though below the glue pipeline (0.972 against 0.993 in 8f-2).
- **Reading:** taking the reply decision away from the code works for the yes/no memory, but not yet at the glue's accuracy. It fails for the object memory, because a graded odor near 0.3 is below what the speaker's sparse KC code can see. The glue (confidence thresholds and reply branch) stays in the pipeline until the user decides.

### 8g-2: the reply decision as a fly for the yes/no memory only (the user chose option 1 after 8g; written before the run)

- **Change from 8g:** object questions go back to the glue: the object memory's threshold, and the plan (the object reply, or "mi sona ala" as cues). Yes/no questions keep 8g's design: no threshold, no branch; the speaker decides from the odor.
- **Interface:**
  - The object memory's odor still reaches the speaker (sure, plus its own fifth doubt channel). Otherwise object replies would be odor-free, and an odor-free context is a PN subset of the yes/no ones (8g).
  - Its doubt is kept apart from the yes/no memory's doubt, so a known object (weak `sure`) does not smell like an unknown yes/no subject.
  - In the table, object replies carry one class ("obj", known and unknown practice odors alike); "mi sona ala" for objects is taught on its own cues (`mi`, `sona`, `ala`).
- **Selection** as in 8g (`configs/phase8g2_chat.yaml`). **Test:** new seeds 165–174.
- **Check on selection seed 10:** reply 0.959, yes/no 0.893, objects 1.000. Without odor, yes/no 0.679; no-KC memory 0.643.
- **Gate:**
  - reply exact ≥ 0.95 on every test seed;
  - for each control (`memory_no_kc`, `speaker_no_kc`, `odor_off`), mean **yes/no** exact ≤ 0.80. Overall reply exact no longer measures the fly's decision, since the object path is glue; on seed 10, `odor_off` is 0.803 overall.

#### 8g-2 result (`results/20261114-222259_phase8g2_chat_s0`): gate passed

- **Selection** (seeds 10–19): 0.977 / 0.983 / 0.977 / 0.983. `n_practice` 120 with `rehearse` 1 was chosen; rehearse 3 changed nothing.
- **Test** (seeds 165–174): reply exact mean 0.992, min 0.980.
  - yes/no 0.989 (min 0.970);
  - objects 1.000;
  - unknown subjects 0.981 (min 0.885, seed 174);
  - changes of mind 0.992.
- **Controls, yes/no exact:**
  - `memory_no_kc` 0.235;
  - `speaker_no_kc` 0.000;
  - `odor_off` 0.368. Without the odor, objects still reach 0.940, since that path is glue.
- **Gate:** both criteria are met.
- **Reading:**
  - For yes/no questions the speaker now decides between answering, `ala` and "mi sona ala" from the memory's odor. That removes a confidence threshold and the reply branch, at a small cost against the glue pipeline (8f-2: 0.998 overall, 0.998 yes/no).
  - The memory's KC layer and the odor both carry the decision (controls 0.235, 0.368).
  - **What still is glue:**
    - the odor interface itself, which computes sureness and doubt from the top two scores: fixed, no threshold;
    - the practice session (teacher);
    - the object path's threshold and plan.

## 8h: role assignment and the cue pointer as flies (the user agreed after 8g-2; written before the run)

- **Glue removed:**
  - **Roles:** `read()` no longer maps states to roles in code. A role fly smells the state the grammar fly reached and says the role of the word: subject, predicate, object or none. It is taught from the old rule's table (14 states), then one rehearsal round, at the single-token setting (2000 KCs, threshold 450, as the class fly). At 650, 13 of 14 states were right; at 16000 / 850, 4 of 14, with single tokens too sparse.
  - **Cue pointer:** `speak()` no longer compares the said word with the cue. A pointer fly smells the cue and the said word together and says `next` or `stay`. It is taught on every (cue, word) pair of the speaker's table (75 on seed 10), then rehearsal.
- **Interface for the pointer:** the cue (the plan) and the word just said come in on **separate PN patterns**, as two senses. With one pattern per word, the "same" pair is a single odor, a PN subset of every "different" pair holding the word, and it loses (65 of 75 at best). With separate patterns, 75 of 75 at 2000 KCs / 650 and at 16000 / 850; without KCs, 44 of 75. Whether to advance after `mi`, `sona` or `ala` depends on the cue (the object path's "mi sona ala" uses them as cues), so a conjunction is needed.
- **Selection** (seeds 10–19, `configs/phase8h_chat.yaml`): the pointer fly at 2000 / 650 or 16000 / 850, mean reply exact, ties to the listed order. The role fly is fixed. Everything else is 8g-2's chosen setting.
- **Test:** new seeds 175–184.
- **Check on selection seed 10:** reply 0.980. `role_no_kc` 0.980, `pointer_no_kc` 0.075, `memory_no_kc` 0.333.
- **Gate:**
  - reply exact ≥ 0.95 on every test seed;
  - yes/no exact ≤ 0.80 for `memory_no_kc`, `speaker_no_kc` and `odor_off`;
  - reply exact ≤ 0.80 for `role_no_kc` and `pointer_no_kc` (8b-3's rule for a new fly).
- **Prediction:** the role fly is a lookup on the state, and seed 10 shows no KC need. So `role_no_kc` will very likely be above 0.80, and the gate then fails formally on it, as 8b-3 did. That would read: the role glue can be a fly, but the KC layer adds nothing there.

### 8h result (`results/20261114-232704_phase8h_chat_s0`): gate not passed

- **Selection** (seeds 10–19): both pointer settings 0.983; 2000 KCs / 650 was chosen.
- **Test** (seeds 175–184): reply exact mean 0.978, min **0.837 (seed 178**: unknown subjects 0.276). The other nine seeds are ≥ 0.98.
- **Controls:**

  | control | measure | value |
  |---|---|---|
  | `memory_no_kc` | yes/no exact | 0.251 |
  | `speaker_no_kc` | yes/no exact | 0.000 |
  | `odor_off` | yes/no exact | 0.430 |
  | `pointer_no_kc` | reply exact | 0.157 |
  | `role_no_kc` | reply exact | **0.978**, identical to the full run on every seed, as predicted |

- **Verdict:** two criteria fail:
  - seed 178 is below 0.95;
  - `role_no_kc` is above 0.80.
- **Diagnosis** (after the gate, on test seed 178):
  - Replacing the role and pointer flies by the old rules gives the same 0.837, so the new flies are not the cause.
  - The failure is in 8g-2's object path: for an unknown subject's "what" question the glue plan is "mi sona ala", but the speaker says "mi sona open". The one context (P, cue `ala`, object odor) is wrong.
  - Rehearsal should have caught it, but it checks each context with the **mean** practice odor of its class. 8g-2's single "obj" class mixes known objects (sure about 0.5) with unknown ones (doubt 1.0), so the mean (0.48 / 0.52) is neither.
  - This is a flaw in 8g-2's design that showed up on this seed only. It is not a property of 8h.
- **Reading:**
  - **The pointer fly works and needs its KC layer** (0.157 without it). The cue pointer can be a fly.
  - **The role fly works, but its KC layer adds nothing** (8b-3's outcome): the role is a lookup on the state.

### 8h-2: unknown-object replies taught on their own odors (the user chose option 1 after 8h; written before the run)

- **Change:** in 8g-2's object path, the unknown-object reply ("mi sona ala" on the cues `mi`, `sona`, `ala`) is taught and rehearsed with unknown-object practice odors (class `ounk`). Known-object replies keep the known-object odors (`obj`).
  - Before, both used one mixed class, whose mean odor rehearsal checked against and which resembled neither: seed 178 in 8h.
  - No new glue: the glue plan already tells the two cases apart.
- **Role fly:** kept, with the note "KC layer not needed" (the user's decision, as the class fly after 8b-3). `role_no_kc` is reported but not gated.
- **Settings:** 8h's chosen settings, fixed, no selection (`configs/phase8h2_chat.yaml`). **Test:** new seeds 185–194.
- **Check on selection seed 10:** reply 0.980 (as 8h), `pointer_no_kc` 0.075.
- **Gate:**
  - reply exact ≥ 0.95 on every test seed;
  - yes/no exact ≤ 0.80 for `memory_no_kc`, `speaker_no_kc` and `odor_off`;
  - reply exact ≤ 0.80 for `pointer_no_kc`.

#### 8h-2 result (`results/20261114-233619_phase8h2_chat_s0`): gate not passed

- **Test** (seeds 185–194): reply exact mean 0.987, min **0.909 (seed 187**, yes/no 0.812). The other nine seeds are ≥ 0.98.
  - objects 0.998;
  - unknown subjects 0.997 (the 8h failure mode is gone).
- **Controls:**
  - yes/no exact: `memory_no_kc` 0.328, `speaker_no_kc` 0.000, `odor_off` 0.367;
  - reply exact: `pointer_no_kc` 0.177;
  - `role_no_kc` (reported only): 0.987, identical to the full run.
- **Verdict:** the first criterion fails on seed 187.
- **Diagnosis** (after the gate, seed 187, 76 yes/no facts after 100 statements):
  - 65 right;
  - 8 wrong polarity (`ala` said or dropped), 2 "mi sona ala", 1 broken frame. In all 11 the old glue decision (threshold 0.2) would have been right.
  - The yes/no memory's odor under load is weak in these cases (e.g. yes 0.39, doubt 0.61), and the speaker misreads it. This is 8g-2's graded-odor decision, not the role or pointer fly.
  - Across the 30 fresh test seeds of 8g-2, 8h and 8h-2, yes/no exact is below 0.95 on 2 (178: 0.934, 187: 0.812).
- **Reading:**
  - The pointer fly needs its KC layer and works on every seed.
  - The odor-based yes/no decision is the weak part. It is right on most seeds, but under load it is less reliable than the threshold it replaced.

### 8h-3: a speaker robust to weak yes/no odors (the user chose option 2, then option 1 if it fails; written before the run)

- **Cause addressed (8h-2, seed 187):** under load the yes/no memory's odor gets weak (yes about 0.4, doubt about 0.6), and the speaker misreads polarity.
- **Two changes tried:**
  - a heavier practice load: 240 facts instead of 120, about 2.5 times a conversation's yes/no load, so weak odors are among those taught;
  - a larger speaker KC layer: 32000 instead of 16000.
- **Selection** on 20 select seeds (10–29), since the failure is on about 1 seed in 15 (`configs/phase8h3_chat.yaml`). The four combinations are ranked by the **worst seed's** reply exact, then the mean, then the listed order (8h-2's setting first).
- **Decision rule, fixed now:** if the chosen setting's worst select seed is below 0.95, option 2 has failed, and option 1 follows on new seeds 205–214. Option 1 is the yes/no decision back to the threshold glue, keeping the role and pointer flies. Test seeds 195–204 are run in the same job and count as used either way.
- **Test** (if the rule passes): seeds 195–204, with 8h-2's gate.

#### 8h-3 result (`results/20261114-234051_phase8h3_chat_s0`): option 2 failed by the rule fixed before the run

- **Selection** (seeds 10–29): worst seed 0.907 / 0.914 / 0.901 / 0.901, mean 0.981 / 0.980 / 0.980 / 0.983, in the listed order. 240 practice facts at 16000 KCs was chosen.
  - Its worst select seed, 0.914, is below 0.95, so **option 2 has failed** and option 1 follows (8h-4).
  - More practice load and twice the speaker KCs do not lift the worst seed.
- **Test** (seeds 195–204, run in the same job, so they count as used): reply exact mean 0.982, min 0.945 (seed 198); yes/no 0.968 (min 0.880, seed 204). The 8h-2 gate would not have been passed either.
- **Controls:**
  - yes/no exact: `memory_no_kc` 0.118, `speaker_no_kc` 0.000, `odor_off` 0.244;
  - reply exact: `pointer_no_kc` 0.186.
- **Reading:** the odor-based yes/no decision is right on most seeds but has a tail of weak seeds (about 1 in 7 below 0.95 here) that these changes do not remove.

### 8h-4: option 1, the yes/no decision back to the threshold (written before the run)

- **Pipeline:** 8f-2's pipeline (thresholds and reply plan for both memories; the 8f odor-free speaker table, with state balance and rehearsal) plus 8h's role fly and pointer fly. The pointer fly is taught on every (cue, word) pair of that speaker table (`configs/phase8h4_chat.yaml`).
- **Settings:** fixed, no selection. Role fly 2000 KCs / 450; pointer fly 2000 / 650; both with rehearse 1.
- **Test:** new seeds 205–214.
- **Gate:**
  - reply exact ≥ 0.95 on every test seed;
  - reply exact ≤ 0.80 for `memory_no_kc`, `speaker_no_kc` and `pointer_no_kc`;
  - `role_no_kc` reported only (KC layer not needed, the user's decision).
- **Check on selection seed 10:** reply 1.000; `pointer_no_kc` 0.177.

#### 8h-4 result (`results/20261114-234836_phase8h4_chat_s0`): gate passed

- **Test** (seeds 205–214): reply exact mean 0.997, min 0.986.
  - yes/no 0.998;
  - objects 0.996;
  - unknown subjects 0.994;
  - changes of mind 1.000.
- **Controls, reply exact:**
  - `memory_no_kc` 0.480;
  - `speaker_no_kc` 0.000;
  - `pointer_no_kc` 0.167;
  - `role_no_kc` 0.997, identical to the full run (reported only).
- **Gate:** both criteria are met.
- **Reading:**
  - The cue pointer is now a fly whose KC layer is needed, at no loss against 8f-2 (0.998).
  - The role assignment is a fly too, but its KC layer adds nothing.
  - The yes/no reply decision stays glue (threshold and plan). 8g-2 showed it can be a fly on most seeds, but 8h-2 and 8h-3 showed a tail of weak seeds.
- **Glue left in the Toki Pona pipeline:**
  - tokenizing;
  - both predicate routings;
  - both memories' confidence thresholds and the reply plan;
  - the `mi`/`sina` swap (none in this task);
  - teacher schedules (balance, rehearsal).

### 8i: predicate routing, third try (the user chose this glue item after 8h-4): stopped at the probe stage

- **Idea:** the unrouted yes/no memory (8d-2: subject words and predicate smelled together, one yes/no pair) with levers the earlier tries lacked:
  - more teaching per fact;
  - a lower confidence threshold;
  - sparser word odors (fewer active PNs per word, so a three-word union is less dense);
  - the FlyWire-sized PN layer (340 PNs, 8a-2).
  - A separate PN channel for the predicate (the pointer-fly trick) does nothing here, since predicate words are never subject words.
- **Probe on selection seeds only** (, no test seeds used). The memory alone was told 90 facts, as in a conversation, about 40 subjects and 4 predicates; unknown subjects were asked about every predicate.

  | PNs, active fraction | KCs, threshold | seed 10: right and confident at 0.1 / 0.2; unknown declined at 0.1 / 0.2 | seed 11, same |
  |---|---|---|---|
  | 150, 0.15 (8d-2) | 32000, 850 | 0.822 / 0.74*; – | – |
  | 150, 0.07 | 32000, 850 | 0.896 / 0.806; 0.725 / 0.900 | 0.887 / 0.873; 0.800 / 0.975 |
  | 150, 0.07 | 32000, 650 | 0.746 / 0.657; 0.450 / 0.850 | 0.761 / 0.620; 0.625 / 0.875 |
  | 340, 0.05 | 32000, 850 | 0.776 / 0.687; 0.825 / 0.950 | 0.648 / 0.620; 0.925 / 0.950 |
  | 340, 0.05 | 32000, 650 | 0.866 / 0.716; 0.550 / 0.875 | 0.831 / 0.775; 0.700 / 1.000 |
  | 340, 0.03 | 32000, 850 | 0.045 (silent) | 0.099 (silent) |
  | 340, 0.03 | 32000, 650 | 0.866 / 0.776; 0.650 / 0.900 | 0.873 / 0.831; 0.725 / 0.950 |

  \* 8d-2's setting, from a first probe: 0.74 of facts confident at 0.2, 0.92 right.
- **Whole conversation** (seed 10, 32000 KCs): yes/no 0.839 (8d-2's setting), 0.911 at threshold 0.1, 0.875 with 10 teachings.
- **Reading:**
  - Sparser odors raise the facts answered (0.74 to about 0.88 at threshold 0.2), but no setting gets both known facts and unknown subjects to about 0.95.
  - A setting that knows more also answers more about unknown subjects. Without routing, an unknown subject shares its head word and the predicate's smell with known facts, the 8e-2 failure.
  - This is the third failure of the same kind (8d-2, 8e-2, 8i). The test was not run, since the selection data already rule out the gate.
  - **Predicate routing stays as glue.**

## Consequences

- 8a is cheap (about 1 s per job in the rate model) and decides whether the rest is worth doing.
- H3 is the project's core assumption (KC conjunctions) tested on a language task. 7b's three-token failure predicts a limit at two modifiers.
