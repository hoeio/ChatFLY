# 0014. Phase 9: a Toki Pona chatbot made of flies (feasibility research and plan)

- Status: **Proposed** (feasibility probes done on selection seeds; the plan and its gates wait for the user's approval)
- Date: 2026-11-23

## Context

The user (2026-11-23): build an actual Toki Pona chatbot on this project; destructive restructuring is allowed; keep researching until it is clear how it can be built.

Phase 8 ended with a 300-turn conversation (0.997) about a toy world: 40 invented subjects, 4 yes/no and 2 object predicates, fixed sentence forms. Three things blocked a real chatbot:
1. **Predicate routing** was glue. Three tries to remove it failed (8d-2, 8e-2, 8i): with subject and predicate smelled together, known facts and unknown subjects could not both be right.
2. The memories held one predicate word per output pair, so a fact could not be about any predicate, and "what" questions worked only for objects.
3. The grammar covered `S li P [e O]` only. Real Toki Pona uses `la`, `pi` and prepositions in about one clause in five (below).

"Real" is measured here on real text: the Tatoeba Toki Pona corpus (80,461 sentences, CC BY 2.0 FR, `data/README.md`).

## What real Toki Pona looks like (Tatoeba, measured 2026-11-23)

- 101,755 clauses (split at `. ! ? : ;`); **95.8%** use only the 120 nimi pu (names excepted). The commonest other words are `kin`, `monsuta`, `kipisi`, `oko`.
- Clause length: median 6 to 7 words; 90% have 3 to 11 words.
- Share of clauses with: `li` 0.61, `e` 0.48, `tawa` 0.22, `ala` 0.21, `lon` 0.20, `la` 0.20, `pi` 0.19, `kama` 0.13, `seme` 0.12, `wile` 0.10, `o` 0.09.
- The 100 commonest words cover 98% of tokens, the 120 nimi pu 99.7%.

## Feasibility probes (selection seeds only; rate model, synthetic wiring)

New code: `src/chatfly/toki.py` (grammar, `Odors`, `Fly`, `Binder`, `Recall`), `src/chatfly/toki_probe.py` (probes).

### P0: reference grammar (`toki.GRAMMAR`)

A finite-state grammar over 11 word classes (`mi`/`sina`, content, the 5 prepositions, `li e la pi o en anu a`), 24 states, 119 valid transitions, one role per state (subject, predicate, object, preposition, prepositional object, context). It accepts **98.1%** of the 97,453 nimi-pu clauses. The rest are mostly non-standard (`e` after a subject, `pi` after a preposition, a bare `o` at the end). H2 of ADR 0013 (particles carry the grammar; ≤ 30 states) holds at full scale.

### P1: memory without routing (`configs/phase9_probe_memory.yaml`)

The idea: never smell three or more words at once. A **binder**, a fixed KC layer that is never taught, smells two odors together. Its active KCs drive a fixed random set of PNs, and the most driven PNs form a new odor (re-encoding). Chaining it gives any phrase, and any (subject, predicate) pair, one key odor that is a conjunction of two things at every step. The memory fly then smells one key.

100 subjects (bare heads, one or two modifiers, shared words), 10 predicates (one or two words), 400 yes/no facts, 40 changes of mind, unknown subjects and untold pairs. `results/20261123-191621_phase9_probe_memory_s0`, seeds 10–19:

| | known right | unknown declined |
|---|---|---|
| **chained keys** (memory 64000 KCs, 4 of 5 claws) | **0.994 (min 0.988)** | **1.000** |
| 8i union smell, 64000 KCs, threshold 850 | 0.35 | 0.52 |
| 8i union smell, threshold 650 | 0.22 | 0.60 |
| binder without KC layer | 0.38 | 0.49 |
| memory without KC layer | 0.29 | 0.45 |

(Best threshold per row, chosen for the worst seed.) The memory's KC threshold was picked on seed 10 (sweep, 400 facts): 16000 KCs at 3 of 5 claws (367 KCs per key) gave 0.91 at 150 facts; 64000 KCs at 3 of 5 (1536 per key) 0.70 at 400; at 4 of 5 claws, 16000 / 32000 / 64000 KCs (33 / 60 / 117 per key) all gave 1.00 at 400 facts, with known confidence p10 ≈ 0.5 vs unknown p90 0.11–0.25. 64000 was kept for the widest margin.

### P2: phrases recalled word by word, facts from real sentences (`configs/phase9_probe_recall.yaml`)

The fact is a real Tatoeba clause `S li P [e O]` (or `mi/sina P [e O]`). One memory fly over the 120 words, a name token and an end mark (2-of-20 code, H1) stores three questions per fact, each under its own key:
- `S li P e seme?` → object;
- `S li seme?` → predicate;
- `seme li P [e O]?` → subject.

The answer phrase is recalled one word at a time: key → first word; bind(key, word) → next word; … → end. Unknown clauses must be declined. Seeds 10–14, 100 unknown clauses.

Three runs. The failures matter, because they fixed the design:
- **Run 1** (`results/20261123-181317_...`): about 0.4–0.5. Diagnosed on seed 10:
  - (a) **Confidence.** (top − second)/top is low for a k-of-n code, since the runner-up shares a group with the winner (the 8g finding). With an ideal hash in place of the binder, recall was 0.99/0.97/0.88 at threshold 0 but 0.55 at 0.2. **New confidence:** (mean rate of the answer's groups − mean of all other groups) / the answer's mean. It is 0 for an untaught odor and near 1 for a taught one.
  - (b) **Popular PNs.** Some PNs were in 35% of keys (expected 15%). A PN that is often in keys drives more KCs at the next binding, so the bias grows along a chain. A fixed per-PN gain did not remove it. A **signed** fan-out (half of each KC's targets excited, half inhibited, so every PN's expected drive is zero) did.
  - (c) **Key space.** 150 PNs at 15% leaves keys that share words too close (overlap above chance). **300 PNs at 10%** (FlyWire has 340 ALPNs per side, ADR 0013 8a-2) brought seed 10 to 1.00/0.98/1.00.
- **Run 3** (`results/20261123-185405_...`, binary signed binder; run 2 was the rate-weighted version):

  | facts | object | predicate | subject | unknown declined |
  |---|---|---|---|---|
  | **100** | **1.000** | **0.992 (0.98)** | **1.000** | **0.999 (0.996)** |
  | **300** | 0.977 (0.967) | 0.963 (0.950) | 0.979 (0.970) | 0.939 (0.925) |
  | binder without KCs, 100 | 0.68 | 0.22 | 0.04 | 0.08 |
  | memory without KCs, 100 (run 2) | 0.67 | 0.01 | 0.02 | 0.00 |
  | **spike-count noise**, 2 s window, 100 | 0.866 | 0.868 | 0.830 | 0.991 |

  Object questions without the KC layer still score 0.67–0.72 because many real clauses have no object: "end" at once is easy.

### P3: grammar fly for the full grammar (`configs/phase9_probe_grammar.yaml`)

- **Setup:** one fly smells (state, class) and says the next state or X (reject). It is taught the 288-entry table (20 epochs, 2 rehearsal rounds) and run on 3000 random real clauses (seeds 10–14).
- **Pass mark:** its states must equal the reference parse, and its rejections the reference's rejections.

| | table | Tatoeba clauses as the reference | word salad rejected |
|---|---|---|---|
| **16000 KCs, 5 of 5 claws** | **1.000** | **1.000 (min 1.000)** | **1.000** |
| 2000 KCs (8b's setting) | 0.999 | 0.9996 | 1.000 |
| without KC layer | 0.50 | 0.017 | 0.74 |

### P4: speaker over the whole vocabulary (`configs/phase9_probe_speaker.yaml`)

The speaker smells (grammar state, next cue tagged with its slot) and says a word, `li`, `e` or the end (1255 contexts; 8a-3's balance, 8f's rehearsal). It speaks the slots of 500 real clauses. Seeds 10–14:

| | table | real sentences exact |
|---|---|---|
| 16000 KCs | 0.9998 | 0.996 (0.992) |
| **32000 KCs** | **1.000** | **0.996 (0.992)** |
| without KC layer | 0.08 | 0.000 |

The failures are clauses like `mi mute li …`, which the table does not teach.

### P5: end to end (`configs/phase9_probe_integrate.yaml`, `results/20261123-191432_phase9_probe_integrate_s0`)

- **Telling:** the user tells 100 real clauses in Toki Pona. The grammar fly reads them, and the slots come from its states.
- **Asking:** the user asks the three questions about each, and the two first about 50 unknown clauses.
- **Replying:** recall gives the answer phrase. The speaker says the reply sentence, with `mi`/`sina` swapped, or "mi sona ala".
- **Results** (seeds 10–14):
  - told clauses understood: **1.000**;
  - **reply exact 0.992 (min 0.987)**: object 0.985, predicate 0.990, subject 0.996;
  - unknown declined: 0.997.
- Two scoring bugs were found in this probe and fixed before the reported run: the reference applied the swap only to the subject, and the several-right-subjects check was wrong.

## What the probes show

1. **The routing glue is gone.** Chained two-at-a-time binding keeps every KC conjunction at two items, the size Phase 2 showed the KC layer separates. One memory fly then answers yes/no, object, predicate and subject questions about any real sentence, and declines unknown ones. Both KC layers are needed (binder and memory controls fail).
2. **The grammar of real Toki Pona fits one fly** (24 states, 98.1% of Tatoeba).
3. **Speaking the whole vocabulary works** with the Phase 8 recipe.
4. **The parts compose:** 0.992 end to end on real sentences without new glue beyond the interface rules below.

## Risks (measured, not solved)

1. **Spiking.** A hash has no margins, so spike-count noise breaks it.
   - With a 500 ms window, recall fails (~0).
   - A binary KC drive and a 2 s window give 0.83–0.87. At an 8 s window, chain keys match 100% (key-overlap test).
   - The real Brian2 binder is untested. The window is a free parameter (arbitrary, `phase6_zumi.yaml`), but an 8 s step is slow.
2. **Capacity.** 300 facts: 0.94 (min 0.925). 100 facts: 0.99. A chat session of a few hundred facts needs more memory KCs or sparser keys; untested.
3. **The re-encoding is not mushroom-body anatomy.** KCs driving a new PN pattern through fixed signed weights is an interface assumption. The closest real circuits (KC→APL/DPM, MBON→DAN loops) do not do this. It is fixed and unlearned (CLAUDE.md principle 2), and the controls show it does nothing without the KC layers. It must be named as glue.
4. **Dialogue beyond facts** (greetings, thanks, "sina seme?", commands, ask-back) is not probed. It is a small table, the kind the role and class flies learned (8b-3, 8h), so low risk.
5. **Forms not yet in the memory:** `la` contexts, prepositional phrases (`lon seme?`), `X ala X`, `pi`, `en`. The grammar parses them; the memory keys for them are the same mechanism, untested.

## Proposed plan (gates to be confirmed by the user)

**Destructive restructuring:**
- `toki.py` becomes the core: grammar, flies, binder, recall.
- A new `toki_bot.py` holds the conversation.
- `tokipona.py`, `tokipona_chat.py`, `talk.py`, `speak.py`, `zumi*.py` and `grammar.py` leave the pipeline. Their results stay reproducible from the recorded commits.
- Delete them once 9a passes, keeping `sim.py`, `fsm.py`, `conditioning.py`, `parity.py` (shared), `run.py`, `web.py`.

| | content | gate (test seeds, set before the run) |
|---|---|---|
| 9a | core refactor; the conversation corpus: real Tatoeba clauses as statements, generated questions of every form, social turns; the oracle | the oracle scores 1.0; every probe above reproduces on the new core |
| 9b | memory for all question forms: `anu seme` and `X ala X` (yes/no), `seme` in subject/predicate/object/preposition slots, `la` contexts, changes of mind; capacity sweep to 300 facts | each form ≥ 0.97 on every seed; unknown declined ≥ 0.97; binder and memory no-KC controls ≤ 0.5 |
| 9c | dialogue acts as flies: an act fly over the grammar states (statement, question form, greeting, thanks, farewell, command); a reply-plan fly; persona facts told by the teacher ("mi pipi", "nimi mi li …") | act and plan tables 1.0; no-KC controls reported (expected lookups, as 8b-3) |
| 9d | **whole conversations**: 300 turns of real sentences, questions, social turns, unknown things | reply exact ≥ 0.95 on every test seed; controls ≤ 0.5; ideal-hash memory reported as the ceiling |
| 9e | the chatbot: terminal and web demo (rate model), the user types any Toki Pona | runs; replies are grammatical by the reference grammar |
| 9f | spiking: binder and memory in Brian2 at reduced scale; window sweep | report; gate set after 9d |

**Interface rules (glue) in this plan:**
- word classes: a lookup, or the class fly (8b-3);
- roles read from the grammar fly's states, plus the `mi X li` fix;
- the `mi`/`sina` swap;
- which keys a statement is stored under, and which key a question asks: a fixed rule per question form;
- the binder's re-encoding;
- the confidence threshold;
- the cue pointer (a fly in 8h);
- the teaching schedules.

Each new rule is tried as a fly first, as the user asked in Phase 8.

## Consequences

- The probes take minutes: the rate model with faster learning (`sim.Simulator.learn` updates only active KCs; same numbers, 2.7× faster).
- Out of scope: words outside nimi pu (4% of real clauses), and inference ("soweli li moku" does not imply "soweli li lon"). The bot answers from what it was told, and says "mi sona ala" otherwise.
