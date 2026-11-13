# 0011. Zumi 2: learning about any word during the conversation

- Status: Accepted
- Date: 2026-11-14

## Context

Zumi v1 (ADR 0010) only talks about a three-place ring. The user asked to talk with the fly
about "more universal things". The fly can't produce words, so it can't talk about the world
in general. It can do what the mushroom body does: associate a new smell with a valence and
recall it later. So Zumi 2 lets the user tell the fly what things are like during the chat
and ask it back later. The fly learns those facts **online** by plasticity, and the interface
must store no facts (CLAUDE.md principle 2).

## Options

1. A lookup table in the interface. Rejected: the fly would do nothing.
2. Train a fixed vocabulary in advance. Rejected: it isn't "any word", and it is no
   different in kind from v1.
3. **Online associative memory.** Each unknown word gets a fixed random PN pattern (a
   "smell"), and properties are pairs of opposites, each with its own two-output KC→MBON
   weight set. Teaching is the same dopamine-gated rule used everywhere else; asking reads
   the outputs without dopamine.

## Decision

Option 3 (`src/chatfly/zumi.py`, `Memory`).

- **Words → smells.** The PN pattern is drawn from an RNG seeded by sha256(word) and the
  `words` rng stream (`pn_active_frac` 0.15, the same as the task tokens). The interface
  uses no word meaning.
- **Four property pairs:** bon/mal (good/bad), gu/pi (big/small), ho/ku (hot/cold) and
  ra/lo (fast/slow). Each pair is an `fsm.WeightSlot`, a separate 2-group KC→MBON weight
  set, all on one KC layer. This is like separate MB compartments that read the same KCs.
- **Teach** (`thing ka prop…`): `n_teach` presentations with `dan_for(side, 2)`, so
  dopamine depresses the opposite output. **Ask** (`thing prop ba?`, `thing ke?`): `n_ask`
  presentations with no dopamine, then decode the summed MBON rates (argmax, confidence =
  (top − second) / sum). Below the threshold the fly says `hm`, which is how it can say
  "never told".
- **Changing a fact** is just teaching the other side, and the same rule relearns it.
- **Criteria** (`configs/phase6_zumi2_memory.yaml`): on every test seed (60–69), answered
  accuracy ≥ 0.99 **and** at most 5% of questions about never-taught things answered with
  ya/na. Capacity is the largest number of taught things that passes. Thresholds are chosen
  on seeds 10–19. 20% of facts are changed later in each run.
- **Controls.** `no_kc`: the same rule on PN→MBON directly (`fsm.PNDirect`). `shuffled_teacher`:
  the dopamine goes to a random side.

### Tuning (selection seeds only)

- The rate model worked with `n_teach` 5 and `n_ask` 1, with threshold 0.2.
- With 1 pairing, spiking answered accuracy was 0.63–0.75 at 40 things.
  - Averaging 4 sniffs (`n_ask` 4) or doubling the KC→MBON drive alone left most taught
    facts below threshold.
  - Doubling the drive (`w_kc_mbon_mv` 2.0, memory slots only) plus `n_teach` 5 reached
    1.000 answered accuracy at 40 things with coverage 0.77 at threshold 0.3 (0.95 at 0.2).
  - `n_teach` 3 had lower coverage.
- Denser noun smells (`pn_active_frac` 0.25) raised KC overlap and made rate worse.
  Rejected.

## Consequences

- The fly can learn about any word, and the guide teaches four property pairs and three
  sentence forms. The ring from v1 is unchanged.
- Capacity is limited by KC overlap between smells: past it, the fly starts answering about
  things it was never told. The results are in `docs/notes/phase6-zumi.md`.
- Facts are not compositional (no "fire is hotter than ice"), and properties are only the
  four fixed pairs. Adding a pair adds one weight slot.
- The browser demo runs a JavaScript port of the rate model and was checked against the
  same metrics in node.
