# 0010. Phase 6: Zumi, a small conversational language for one fly

- Status: Accepted (goal set by the user, 2026-11-13: "make a really simple language that can talk well, with a guidebook, and train chatfly on it")
- Date: 2026-11-13

## Context

Phase 5 only answers "is this A/B sequence even or odd?". The user asked for a language that
is simple, learnable by a person, and lets them hold a conversation with the fly. The
constraints come from earlier phases:

- One fly is a learned transition table with one output group per state (ADR 0008).
- Phase 4: one fly fails tables whose entries are outvoted by their row/column mates
  (margin ≥ 3); structured tables with margin ≤ 2 are learned.
- Real wiring grounds only 2 output groups (ADR 0007). A 3-state language therefore uses the
  synthetic MB at the Gate B / Phase 4 parameters.
- The interface must not do the fly's work (CLAUDE.md principle 2).

## Options

1. More question types over parity: still 1 bit; nothing to talk about.
2. A world the fly tracks: states = places, input tokens = move words, kept across turns in
   the feedback loop. Conversation = telling the fly to move and asking where it is.
3. Several flies, one per topic, routed by word: more to talk about, but the router is glue
   and Gate D found no gain from several flies.

## Decision

Option 2, as the language **Zumi** (`docs/zumi-guide.md`):

1. **World:** a ring of three places `ne → fi → po → ne`. Moves `zu` (+1), `ki` (−1), `ta` (stay).
   Each move permutes the places, so the table is a Latin square and every margin is ≤ 2.
   The vocabulary was chosen with `configs/phase6_vocab_sweep.yaml` on seeds 10–19 (rate,
   `results/20261113-195555_phase6_vocab_sweep_s0`; seeds ≥ 0.99 of 10):

   | vocabulary | seeds | | vocabulary | seeds |
   |---|---|---|---|---|
   | ring3 zu ki | 10 | | ring4 zu ki ta | 8 |
   | **ring3 zu ki ta** | **10** | | ring4 zu ki ho | 1 |
   | ring3 zu ki ho ("go home") | 4 | | ring4 zu ki ta pa (+2) | 1 |
   | ring3 zu ki ta ho | 1 | | ring5 zu ki ta | 5 |

   A state-independent word (`ho`) breaks learning, as the margin analysis predicts. Margin
   ≤ 2 is not sufficient, though: the full 4-place Latin square (with `pa` = +2) fails too.
   The largest vocabulary learned on every seed is the 3-place ring with three moves.
2. **Grammar** (regular, fixed parser `zumi.parse`): `[bzz] [si] move* [nu | <place> ba]`.
   Unknown words and wrong order are refused (`e`), never guessed.
3. **Who does what:**
   - The fly computes every step (place, move) → place in its closed loop. Its last decoded place is
     kept between turns, so its memory is the same loop as in Phase 2.
   - The interface does fixed rules only: word → token, final state → word, equality for
     `ba`, discarding the result for `si`, reset to the start for `bzz`. It never reads the table.
4. **Honesty:** a turn's confidence is the lowest step confidence since the last reset. Below
   the threshold the fly says `hm`. It stays `hm` until `bzz`, since a later answer would be a guess.
5. **Evaluation** (`configs/phase6_zumi.yaml`, `configs/phase6_zumi_fsm.yaml`):
   - transition accuracy, 50-step loops, PN-direct (no KC) and teacher-shuffle controls;
   - random grammatical conversations (8 turns), scored per question turn;
   - threshold chosen on seeds 10–19, reported on seeds 0–9;
   - target: answered accuracy ≥ 0.99 on every test seed, as Gate E (c);
   - teacher shuffle behind the same interface should answer at chance (1/3).
6. **Plasticity step:** `eta_dep` 0.3 instead of 0.5, for Zumi only. v1 (0.5) failed on test
   seeds 0–9 (rate 9/10, spiking 7/10, with silent wrong answers). 0.3 was chosen on dev
   seeds, then passed on fresh test seeds 60–69: every answer correct in both models
   (`docs/notes/phase6-zumi.md`).

## Consequences

- The language is small: 3 places, 3 moves, 2 question forms, "what if", restart. Growing it
  needs either a new structure the fly can learn (margin ≤ 2) or routed glue (ADR 0008).
- `bzz` is an interface reset, not learned; a learned "go home" fails (sweep).
- Synthetic wiring only. The no-KC control is what shows the KC layer is needed; there is no
  connectome-structure claim.
