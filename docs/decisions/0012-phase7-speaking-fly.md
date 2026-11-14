# 0012. Phase 7: a fly that says its Zumi replies (7a: corpus, metrics, baselines)

- Status: Accepted (7a, 7b design); final goal agreed with the user on 2026-11-14
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
| 7e | Zumi 3: "GPT-2 level in Zumi" (below) | final |

### Final goal (7e), agreed with the user

The user asked to raise the bar to "GPT-2 level language use". In Zumi this means:

- **Free generation.** The fly samples each word from its MBON output, and one turn can hold several sentences. The interface only stops at the end word.
- **Zumi 3.** Zumi gains the pronoun `to` ("it" = the current topic): `to ke?`, and the fly says `to ka mal ho.` The fly can ask back (`sun ka gu.` → `o. sun ho ba?`). A pronoun is used instead of copying nouns, which output groups can't do.
- **Criteria**, on test seeds:
  1. reply perplexity ≤ the strongest baseline (5-gram, RNN, and a gated RNN to be added);
  2. sampled sentences grammatical ≥ 0.95 and fitting the question ≥ 0.95;
  3. statements about taught facts and the ring place agree with the truth ≥ 0.95;
  4. diversity (distinct-n) reported, with no degenerate repetition;
  5. the controls (shuffled teacher, no KC, shuffled wiring) fail.

  Risk: Zumi 3 has more grammar states (7c).

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

## Decisions (7b)

- **Output groups:** 16 reply words (o ya na hm e, 3 places, 8 properties) plus EOS = 17.
- **Context:** the last two user words (`a`, `b`; nouns as one class N) and the fly's previous word `p` (SEP at the start of a reply). Each token is a PN pattern. The next word is the argmax group, fed back as `p`.
- **One fly smelling (a, b, p) failed** on selection seed 10.
  - Grammatical was 0.06–0.55 at token density 0.05–0.15. With a higher KC threshold (850) it reached 0.93.
  - Contexts that differ only in `p` share 28–48% of their active KCs. EOS, the target of every reply's last step, takes the shared KCs over, so the fly ends replies at once.
- **Two flies with a fixed routing rule** (like ADR 0008).
  - Fly 0 says the first word from (a, b); fly 1 says every later word from (b, p). These are two-token conjunctions, as in Phase 2.
  - Prototype on selection seeds 10–11: grammatical 1.000, appropriate 0.950–0.952 at density 0.15 and KC threshold 650.
  - Routing on "first word or not" is an interface rule; the fly still chooses every word.
- **Ceiling:** a lookup table with the most frequent next word for each context the flies see. On seeds 10–11 it gets appropriate 0.977–0.978, and "bad" (word salad → `e`) only 0.77. Telling salad from a real sentence needs the whole utterance, which is the grammar state of 7c.
- **Selection:** (token density, KC threshold) from a 4-point grid on seeds 10–14 (dev split), by mean appropriate.
- **Gate (set before the test run):** on every test seed grammatical ≥ 0.95 and appropriate ≥ 0.95, and each fly control's mean appropriate ≤ 0.90. Content (exact on ask / what / where / yes-no) is not part of the 7b gate.

## Decisions (7c)

- **Grammar as an FSM** (`grammar.dfa`). It has 14 states over 21 inputs (Zumi words, nouns as N). The state machine is used only for training targets and evaluation; `tests/test_grammar.py` checks it against `zumi.parse` on the corpus and 5,000 random word salads.
- **Zumi 3 rule:** properties must come in pair order (good/bad, size, heat, speed), like adjective order. Without it, contradiction checks need the set of pairs used (more states). `lang.prop_order: strict`; Zumi 2 is unchanged.
- **Teaching every table entry failed** on selection seeds.
  - With word-routed readout, transitions were 0.87 correct and almost every sentence was rejected. For most words most states lead to the dead state X, and word-only KCs settle on X (the Phase 4 margin mechanism).
  - Learning only from corpus experience gave the opposite: valid sentences accepted, but only 70–74% of word salads rejected.
  - Mixing corpus and table passes did not help.
- **Teaching only valid transitions** and reading low confidence as "dead". It is the Zumi 2 memory idea: never taught → no preference.
  - Presenting the state token alone in the word-selected readout gave 1.000 on selection seeds 10–11 at threshold 0.05. Taught transitions had confidence ≥ 0.82, untaught ones ≤ 0.015 (95th percentile).
  - X is absorbing (interface rule, like the ring's "lost").
- **Finding before the test run:** with the same routing and state-only input, the rule **without a KC layer** also reaches 1.000 on the selection seeds, at threshold 0.15–0.2. Each word's readout is a lookup over 14 random state patterns, which are linearly separable.
  - So in this design the router and the rule do the work, not the KC layer. This repeats the revised Gate D verdict.
  - Without routing (one fly, state + word together), only 0.75–0.80 of valid transitions are argmax-correct.
- **Gate (set before the test run):** on every test seed:
  - grammar step accuracy ≥ 0.99 and utterance accuracy ≥ 0.99;
  - speaker grammatical ≥ 0.95 and appropriate ≥ 0.95;
  - each control's mean utterance accuracy ≤ 0.90.
- Thresholds are chosen per run on select seeds, so the controls get the same tuning.

## Consequences

- 7b is compared with these numbers (`docs/notes/phase7-speaking.md`). The fly earns credit
  only for what it does beyond the n-grams (which get the form right) and what it does with
  memory.
- A ponytail note: the RNN is plain numpy (no torch in the environment). It is a baseline,
  not a tuned model, and a stronger one (GRU or LSTM) can be added if the fly gets close.
