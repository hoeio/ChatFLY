# Phase 5 / Gate E: chat layer over the parity fly

## Update (2026-11-13): negation refusal, held-out set v2

The user confirmed the ADR 0009 criteria and the parser fix. Negated questions are now refused (`chat.negations`). Re-run: `results/20261113-180326_phase5_chat_s0` (commit 607f7f3). An earlier attempt, `results/20261113-180242_phase5_chat_s0`, failed because YAML read the bare `no` as false; the words are now quoted, and a test covers it.

| set | n | correct | silent wrong |
|---|---|---|---|
| dev | 100 | 1.000 | 0.000 |
| held-out v1 (seen; kept for the record) | 160 | 0.500 | 0.000 |
| **held-out v2** (written after the fix, before the parser was run on it) | 200 | **0.800** | **0.000** |
| out-of-scope | 8 | refused 8/8 | – |

- v1 drops from 0.625 to 0.500 because its two negated templates are now refused instead of inverted.
- The fly results are identical to the first run (checked: equal `fly` metrics).

| criterion | result |
|---|---|
| (a) dev parse 100%, held-out silent-wrong 0 | **pass** (v2) |
| (b) out-of-scope refused 100% | pass |
| (c) answered acc ≥ 0.99 on every test seed | rate pass; **spiking 8/10: open** |
| (d) teacher-shuffle ≤ 0.6 | pass |

**Gate E verdict: near-pass, accepted by the user (2026-11-13).** (a), (b) and (d) pass. (c) passes for rate and holds on 8/10 spiking test seeds. On the two failing seeds the fly mostly answers "uncertain" instead of answering wrongly (seed 8: 8% coverage, answered accuracy 0.875; seed 5: 52%, 0.942). The weak (s0, A) transition under some training orders is the same fragility as Gate C and is left as is. Interactive mode uses the chosen thresholds (`chat.min_confidence`).

## First run

Run: `results/20261113-170848_phase5_chat_s0` (config `configs/phase5_chat.yaml`, commit 034f870, 25 min with 10 workers). Design and the proposed criteria are in ADR 0009. The fly is the Gate C parity fly (FlyWire v783 right MB, DAN-cluster readout). Each seed answers 100 questions on random A/B sequences of length 1–10.

## Translation (parser only, no fly)

| set | n | correct | silent wrong (not refused) |
|---|---|---|---|
| dev templates (5) | 100 | 1.000 | 0.000 |
| held-out templates (8) | 160 | 0.625 | **0.125** |
| out-of-scope utterances (8) | 8 | refused 8/8 | – |

- All silent errors come from one held-out template, "Is {seq} not odd?". The parser has no negation, so it reads the question as "odd?" and the answer comes out inverted.
- Two held-out templates are refused, not misparsed: "A question: is {seq} even?" and "Is the count of A in {seq} even?". A capital "A" used as an ordinary word is a second all-caps run, so the parser refuses it as ambiguous.
- The parser was not changed after seeing these results, so the held-out set is still clean.

## Fly behind the chat layer

Answered accuracy means the accuracy of the questions answered rather than marked "uncertain". Coverage is the fraction answered. The threshold applies to the lowest per-step confidence and was chosen on seeds 10–19 (rate: 0.0, spiking: 0.05).

| | answered acc mean (min) | seeds ≥ 0.99 | coverage mean (min) | teacher-shuffle answered acc |
|---|---|---|---|---|
| rate, test seeds 0–9 | 1.000 (1.000) | 10 | 1.000 (1.000) | 0.490 |
| spiking, selection seeds 10–19 | 1.000 (1.000) | 10 | 0.881 (0.370) | 0.524 |
| spiking, test seeds 0–9 | 0.982 (0.875) | **8** | 0.799 (0.080) | 0.528 |

Spiking test seeds, threshold vs. outcome:

| threshold | 0.0 | 0.02 | 0.05 (chosen) | 0.1 | 0.15 | 0.2 |
|---|---|---|---|---|---|---|
| seeds ≥ 0.99 | 8 | 8 | 8 | 10 | 10 | 10 |
| min answered acc | 0.535 | 0.562 | 0.875 | 1.000 | 1.000 | 1.000 |
| mean coverage | 0.998 | 0.905 | 0.799 | 0.643 | 0.413 | 0.274 |

- **The failing seeds are 8 and 5.** Without a threshold, seed 8 answers only 53% of questions correctly and seed 5 answers 73%. On isolated steps, (s0, A) → s1 is correct only 60% of the time for seed 8 and 70% for seed 5; the other transitions are at 1.0 (step accuracy 0.90 and 0.925). In Phase 3 the same seeds reached 1.0. Phase 3 ran the order task first on the same task RNG, so it trained these seeds in a different order. This is a realization of the Gate C fragility (9/10 seeds there), not a chat-layer bug.
- **Confidence is informative.** The errors sit in the low-confidence answers. At a threshold of 0.1 every test seed is at 1.0, at a mean coverage of 0.64. Seed 8 marks 92% of its questions "uncertain" at the chosen threshold, so the weak fly mostly declines rather than answering wrongly.
- **Glue control passes.** A fly trained with a shuffled teacher, behind the same parser and reply code, answers at about chance (0.49 rate, 0.53 spiking). The chat layer does not compute parity.

## Gate E against the proposed criteria (ADR 0009)

| criterion | result |
|---|---|
| (a) dev parse 100%, held-out silent-wrong 0 | **fail**: held-out silent-wrong 0.125 (negation) |
| (b) out-of-scope refused 100% | pass |
| (c) answered acc ≥ 0.99 on every test seed | rate pass; **spiking 8/10** at the held-out-chosen threshold |
| (d) teacher-shuffle ≤ 0.6 | pass |

**Gate E is not passed.** Options for the user:

1. Parser: refuse any question with negation ("not", "n't", "no"), then write a new held-out template set. The current set has been seen, so it cannot be reused to test the fix.
2. Fly: (i) accept 8/10 as a near-pass, like Gate C; (ii) choose the threshold by a coverage-aware rule (0.1 would pass every test seed here, but it was not the held-out choice); or (iii) address the weak (s0, A) transition at the training level (more epochs or on-error teaching), which reopens Phase 2/3 parameters.
3. LLM translator: needs an API key and network access. It must be held to (a) and (b) too.
