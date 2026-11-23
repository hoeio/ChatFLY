# ChatFLY: progress summary (2026-11-15)

"The fly" is a digital fly: a computational model of the fly mushroom body (MB), not a living
fly. Every number below is from held-out test seeds unless noted. The details are in
`docs/notes/` and `docs/decisions/`.

## What was built

| | what | verdict |
|---|---|---|
| Phase 0 | foundations: synthetic MB (150 PNs → 2000 KCs, APL, MBONs), rate and spiking (Brian2) models with one interface | done |
| Phase 1 | associative conditioning: learning, extinction, reversal | **Gate A pass** (both models) |
| Phase 2 | feedback loop = state-transition table; parity of a sequence | **Gate B pass** after a user-approved rule fix (graded eligibility). Spiking: 0 errors in ~16,000 loop steps |
| Phase 3 | real FlyWire wiring | **near-pass**, accepted by the user. Rate 10/10 seeds; spiking 9/10 ≥ 0.99. No evidence that the real PN→KC or APL structure helps |
| Phase 4 | capacity; several flies | **Gate D: NO.** One fly fails tables with conflicting majorities. Routing solves them, but routing is designer glue, not multiple mushroom bodies |
| Phase 5 | chat layer over the parity fly | **near-pass**, accepted by the user. No silent parser errors; spiking ≥ 0.99 on 8/10 seeds |
| Phase 6 | Zumi, a 10-word language; the fly tracks its place on a ring of three places | **pass** on new test seeds. 1,115/1,115 questions right per model; shuffled teacher at chance |
| Zumi 2 | talk about anything: the fly learns facts during the conversation | **pass** up to 80 things (rate) and 40 (spiking). Without the KC layer the same rule fails at 40 |
| Phase 7 | the fly says its own Zumi replies, word by word ("GPT-2 level in Zumi") | 7a pass; **7b–7f gates not passed**. Closed by the user. Results below |
| Phase 8 | a subset of Toki Pona (112 content words): eight flies hold a whole conversation | **pass** (8h-4), closed by the user. Hypotheses H1–H6 supported. Results below |
| Phase 9 | a Toki Pona chatbot over real sentences (Tatoeba) | feasibility probes done (ADR 0014): routing glue removed by chained binding, end to end 0.992 on selection seeds; plan waits for approval |

## Phase 7 in one table (7f, test seeds 70–74)

| | perplexity | exact reply | grammatical | memory under load (40 things) |
|---|---|---|---|---|
| **flies** | 1.46 | **0.988** | 0.999 | 0.952 (min 0.924) |
| without KC layer (memory) | | | | **0.549** |
| shuffled teachers | 2.77 | 0.667 | 0.999 | |
| GRU (best baseline) | **1.27** | 0.761 | 1.000 | |
| 8-gram | 1.32 | 0.765 | 1.000 | |

The fly's replies are sampled from its MBON rates. They are right far more often than the
baselines', which cannot remember across turns. As a probability model it is worse than a
GRU, and two seeds fall below 0.95 on memory under load.

## Phase 8 in one table (8h-4, test seeds 205–214, rate model, synthetic wiring)

The conversation is 300 turns: yes/no and object facts (with changes of mind), yes/no and "what"
questions, and subjects never mentioned. Eight flies: class, grammar, role, two routed memories
(yes/no, object), speaker, cue pointer.

| | exact reply, mean (min) |
|---|---|
| **flies** | **0.997 (0.986)** |
| memories without KC layer | 0.480 |
| speaker without KC layer | 0.000 |
| cue pointer without KC layer | 0.167 |
| role fly without KC layer | 0.997 (a lookup: KC layer not needed) |

What made it work:
- **A compact output code:** each word is 2 of 20 MBON groups, which says 112 words as well as one group per word.
- **Sparse KC codes:** 16000 KCs, all five claws needed.
- **Teaching schedules:** rare contexts taught as often as the majority, and **rehearsal**, i.e. re-teaching what the fly gets wrong, which repaired every single-context failure seen.

## What the evidence supports

1. **A mushroom-body loop learns small finite-state machines.** Parity and the three-place
   ring are learned, in rate and spiking models, with controls at chance (Phases 2, 6).
2. **One-shot memory in the KC→MBON synapses, used in conversation.** The KC layer is needed,
   and controls show it: without it, memory fails at 40 things (Zumi 2) and under load in
   Phase 7 (0.55 vs 0.95).
3. **A system of flies can hold a Zumi conversation in its own words.** Grammar, content and
   every reply word come from flies (Phase 7).
4. **The same approach scales to a Toki Pona subset.** About five times Zumi's vocabulary,
   noun phrases with a modifier, and objects. KC conjunctions keep phrases apart (8c: 0.990, no KC
   layer 0.500). A 2-of-20 MBON code fits the real fly's 35 MBON types (Phase 8).

## What it does not support

- **The real connectome is not shown to help.** On FlyWire wiring, parity works about as well
  as on random wiring (Phase 3).
- **Capacity comes from routing.** Bigger tasks needed input-selected readouts, and the same
  router without a KC layer also works for grammar (Phases 4, 7c). That is designer glue.
- **No general language ability.** Zumi has 20+ words and a regular grammar; replies need cue
  lists, pronoun resolution and a sharp decoder (temperature 0.1) from the interface.
- **Not GPT-2 level** by the agreed criteria: perplexity is above a small GRU's.
- **Routing is still needed in Phase 8.** Three tries to replace the memories' predicate routing
  with a KC conjunction failed (8d-2, 8e-2, 8i). Without routing, known facts and unknown subjects
  cannot both be right.
- **Phase 8 is rate model on synthetic wiring only.**

## Glue that remains (designer rules, not learned)

- the parser and the routing of each question to a fly;
- input-selected readouts (grammar, comparator);
- the speaker's split between the first word and later words;
- the cue lists and their pointer;
- pronoun resolution;
- absorbing "lost" and "dead" states;
- the sampling temperature.

In Phase 8:
- **Replaced by flies:**
  - the cue pointer (its KC layer is needed);
  - word classes and roles (lookups: KC layer not needed).
- **Still glue:**
  - tokenizing;
  - predicate routing;
  - the memories' confidence thresholds and reply plan. A speaker deciding from the memory's odor works on most seeds (8g-2) but not reliably on all (8h-2, 8h-3);
  - teaching schedules.

## Try it

- Published demo (rate model in the browser; the fly speaks Zumi 3): https://claude.ai/artifact/5aDVbagQmhxcNkvmW1fdKt
- Guidebook: https://claude.ai/artifact/LxR9CQSG3n6uf4cUb8EduG (also `docs/zumi-guide.md`)
- Local demos:
  - `python -m chatfly.web configs/phase7f_zumi3.yaml` (Zumi 3, rate);
  - `python -m chatfly.web configs/phase6_zumi.yaml --sim spiking` (Zumi 2, spiking);
  - `python -m chatfly.web configs/phase8h4_chat.yaml` (Toki Pona, 8h-4, rate). In the terminal: `python -m chatfly.tokipona_chat configs/phase8h4_chat.yaml`. The page lists the words this seed's flies know (`--seed N` picks another lexicon).

## Open directions (not started)

- Calibrated speaker outputs, so that perplexity can compete without a sharp decoder.
- Removing glue: learned routing (three failed tries in Phase 8), a reliable reply decision without thresholds.
- A spiking version of Phase 7 or Phase 8, and Phase 8 on FlyWire wiring.
- A task where the real connectome could matter (Phase 3 found none so far).
