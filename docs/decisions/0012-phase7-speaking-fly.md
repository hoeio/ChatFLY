# 0012. Phase 7: a fly that says its Zumi replies (7a: corpus, metrics, baselines)

- Status: Accepted (7a)
- Date: 2026-11-14

## Context

The user asked for a "GPT-2 level" chatbot, with Zumi as the language. GPT-2 picks the next
word from a long context. ChatFLY's fly can't produce words: the interface assembles every
reply, and one fly reliably holds 3 loop states. General GPT-2 ability is out of reach.
The user agreed to this definition instead: **a fly that generates its Zumi replies word by
word**. The replies must be grammatical, fit the question, and agree with what it was told,
and the interface must not build template replies.

Zumi's grammar is regular, so a finite-state machine can generate it. FSMs are what the
fly's loop learns (ADR 0002). The open question is the number of states (Phase 4 capacity).

Phases:

| | content | gate |
|---|---|---|
| 7a | corpus, metrics, non-fly baselines | the corpus agrees with the parser and the oracle scores 1.0 |
| 7b | speaking fly v0: MBON groups = reply words, previous word fed back | to be set from 7a |
| 7c | grammar state tracked by the fly (several flies) | branch point |
| 7d | replies use the memory fly and the ring, over many turns | |

## Decisions (7a)

- **Corpus** (`src/chatfly/speak.py`).
  - A dialogue is 30 turns of a user utterance followed by the correct fly reply.
  - Each dialogue has a fresh world: the fly starts at the nest, knows no facts, and 6 new random nouns are used.
  - Turn kinds: teach / ask / what / ring / bad, at 25 / 20 / 10 / 35 / 10%. "Bad" turns are word salad that the parser refuses.
  - The correct reply comes from `oracle`, which lives only in the evaluation.
  - Replies use only the fly's words: o ya na hm e, places, and properties (one per pair, in pair order). Replies never contain nouns, so the output vocabulary is fixed.
- **Stream:** user words, `|`, reply words, `#`.
- **Metrics.**
  - Teacher-forced reply perplexity and next-word accuracy.
  - Greedy free generation of each reply from the gold history, scored as:
    - *grammatical*: fits the reply grammar;
    - *appropriate*: the reply's form fits the question kind, e.g. a place or `hm` for `nu`;
    - *exact*: equals the oracle reply, reported overall and by kind.
- **Baselines.**
  - n-grams (orders 1, 2, 3, 5, 8), with nouns mapped to one class and backoff to the longest seen context.
  - An Elman RNN (128 units) trained on reply words only. It sees each noun as a fixed random ±1 vector from sha256(word), the same information the fly gets. The epoch is chosen on a dev split.
  - Because nouns are new in every dialogue, exact replies about facts need **in-context memory**: the RNN must learn it, and the n-grams can't have it.
- **No external LLM.** The corpus is generated from the grammar.

## Consequences

- 7b is compared with these numbers (`docs/notes/phase7-speaking.md`). The fly earns credit
  only for what it does beyond the n-grams (which get the form right) and what it does with
  memory.
- A ponytail note: the RNN is plain numpy (no torch in the environment). It is a baseline,
  not a tuned model, and a stronger one (GRU or LSTM) can be added if the fly gets close.
