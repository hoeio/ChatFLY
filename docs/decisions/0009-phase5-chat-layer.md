# 0009. Phase 5 chat layer: fixed-schema parser, closed-loop query, Gate E criteria

- Status: Accepted (Gate E criteria confirmed by the user, 2026-11-13)
- Date: 2026-11-13

## Context

Phase 5 puts a natural-language interface on a trained fly. Gate D answered NO, so the
interface talks to one single-fly module. The Gate C parity fly (FlyWire v783 right MB, DAN-cluster
readout, ADR 0007) is the most validated module, so Phase 5 starts with it. The roadmap's Gate E
("question answering within the learned language works, uncertain / out-of-scope input is
handled honestly") needs measurable criteria.

The translator must not answer in place of the fly (CLAUDE.md principle 2). This environment
has no `anthropic` package and no API key.

## Options

1. LLM translator (Claude API): handles paraphrases well, but needs a key and network, is not
   deterministic, and is harder to audit for "does the translator solve the task".
2. Fixed rule-based parser over a small schema: deterministic, testable, cannot compute parity
   by construction, but brittle on paraphrases.

## Decision

1. **Parser (option 2, for now).** `chat.parse`: exactly one run of all-caps tokens (`A B A`,
   `ABBA`) plus exactly one question keyword (`even` / `odd`; both = "which?"). Anything
   else is refused with a reason (no sequence, several sequences, unknown tokens, negation, no
   question). It never guesses. Negation (`chat.negations`: not, no, never, nor, n't) is refused
   because the parser has no logic to flip a question. This was added after held-out set v1 showed
   silent inversions ("Is ABBA not odd?"). v1 is therefore seen, and the fix is tested on a new
   held-out set v2, written before the parser was run on it. An LLM can replace `parse` later behind the same return format; that would need
   the same held-out evaluation.
2. **Fly query.** The parsed tokens run through the closed loop from the start state, as in
   Phase 2/3 (decoded state fed back as the next state token, each step from rest). The answer
   is built only from the final decoded state. The confidence is the lowest per-step decode
   confidence `(top - second) / sum`. If any step is undecodable or the confidence is below a
   threshold, the reply is "uncertain".
3. **No ground truth in the chat path.** The transition table is used only by the evaluation
   (`chat.truth`).
4. **Threshold choice.** Held-out seeds 10–19 are used to pick the smallest threshold in
   `chat.conf_grid` at which every seed's answered accuracy reaches 0.99. Test seeds 0–9 are
   reported at that threshold.
5. **Gate E criteria (confirmed by the user):**
   - (a) Translation, measured without the fly. Dev templates are parsed 100% correctly. On
     held-out paraphrase templates (written before the first run, not used to tune the parser),
     accuracy is reported, and the **silent-wrong rate** (wrong parse, not refused) must be 0.
   - (b) Out-of-scope utterances are refused 100% of the time.
   - (c) The fly, measured on gold token sequences of length 1–10: answered accuracy ≥ 0.99 on
     every test seed at the chosen threshold, with coverage (fraction not "uncertain") reported.
   - (d) Glue control: a teacher-shuffled fly behind the same chat layer answers at about
     chance (mean answered accuracy ≤ 0.6). This shows the chat layer does not answer.

## Consequences

- The parser is brittle on paraphrases. That shows up as refusals, or as silent errors that (a)
  catches.
- Swapping in an LLM needs a key, network access and a user decision, and must be re-run
  against (a) and (b).
- Only parity is wired up. Other single-fly tasks need only a `chat.task` and `chat.questions`
  change in the config.
