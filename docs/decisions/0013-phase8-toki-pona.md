# 0013. Phase 8: can the current fly learn a subset of Toki Pona? (hypotheses)

- Status: Accepted (plan and 8a design); 8a gate not passed (1 of 5 test seeds); 8a-2 (340 PNs) rejected at selection; 8a-3 (balanced teaching) **passed the 8a gate** on test seeds 80–84; **8b passed** on test seeds 90–94; 8b-2 (tokenizer as a fly): A and B fail; **8c passed** on test seeds 100–104; 8d: replies 0.995, gate not passed only by the class-fly control (as 8b-3); 8b-3: the class fly works (1.000) but the no-KC control does too (0.968), so its gate is not passed. **The user accepted it** (option 1): the class fly stays in the pipeline with the note "KC layer not needed", and from now on new glue is first tried as a fly, see `docs/notes/phase8-toki-pona.md`
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

## Consequences

- 8a is cheap (about 1 s per job in the rate model) and decides whether the rest is worth doing.
- H3 is the project's core assumption (KC conjunctions) tested on a language task. 7b's three-token failure predicts a limit at two modifiers.
