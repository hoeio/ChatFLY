# 0013. Phase 8: can the current fly learn a subset of Toki Pona? (hypotheses)

- Status: Accepted (plan and 8a design); started at the user's request on 2026-11-14
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

## Consequences

- 8a is cheap (about 1 s per job in the rate model) and decides whether the rest is worth doing.
- H3 is the project's core assumption (KC conjunctions) tested on a language task. 7b's three-token failure predicts a limit at two modifiers.
