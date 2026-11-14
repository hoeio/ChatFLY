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

## Decisions (7d)

- **The user chose option 1 after 7c** (2026-11-14): keep the word-routed grammar fly and record that grammar tracking rests on the routed readout, not on the KC layer.
- **Content comes from flies** (`src/chatfly/talk.py`):
  - the ring fly (Phase 6) and the memory fly (Zumi 2), driven through `zumi.Conversation` as in the Zumi 2 chat;
  - a new **comparator fly**: (asked word, the fly's answer) → ya / na / hm, so the interface does not compare words. It has a routed readout, as in 7c.
- **Speaker:** the split speaker smells (grammar state, first content word), then (previous word, next content word). Content words are the ring's place, the comparator's answer, the memory's known properties in pair order, `hm` when a fly is unsure, and `-` for none.
- **Interface rules:** which fly answers which question (by the parse), the pair order of the "what" list, and stopping at EOS.
- **Nothing is tuned in 7d:** the grammar threshold is 7c's; the ring and memory thresholds are Phase 6's. A check on selection seed 10 (30 dialogues) gave exact 0.998.
- **Controls:**
  - `shuffled`: the ring, memory and comparator teachers are shuffled;
  - `no_kc_memory`: the memory rule on PNs.
- **Gate (set before the test run):** on every test seed:
  - exact ≥ 0.95;
  - content exact (ask / what / where / yes-no pooled) ≥ 0.95;
  - appropriate ≥ 0.95 and grammatical ≥ 0.95;
  - each control's mean content exact ≤ 0.70.

## Decisions (7e)

- **The user chose to go to 7e after 7d** (2026-11-14), with the control criteria set before the run.
- **Zumi 3** (`src/chatfly/zumi3.py`).
  - Pronoun `to` means the last named thing. The interface resolves it (a fixed rule, like the parse).
  - Properties come in strict pair order (as in 7c).
  - The oracle is stochastic. A teach turn may be followed by an ask-back (`o . to q ba .`), and a "na" answer by what is true. Exact means "one of the correct replies" (the support set).
- **Speaker** (`CueSpeaker`).
  - It is the split speaker smelling (grammar state, cue) and (previous word, cue).
  - Cues come from the flies: ring place, comparator answer, memory properties, and the first unknown pair for an ask-back.
  - A pointer over the cue list moves on when the fly says the cue's word. This is an interface rule.
- **Sampling.** Each word is drawn from the normalized MBON group rates raised to 1/T.
  - The config first said T = 1.0, "fixed in advance", but that was before any run. On selection seed 10, T = 1 gave exact 0.37: rate-proportional distributions are flat (median p(correct word) 0.74). Greedy gave 0.99.
  - **T is chosen on selection seeds 10–14** (dev split, 30 dialogues each) from {0.1, 0.2, 0.3}, as the largest T with exact ≥ 0.95 on every seed.
    - T = 0.1 has exact 0.958–0.999.
    - T = 0.2 falls to 0.926 on seed 14.
    - **T = 0.1**, and the baselines are sampled at the same T.
  - Distinct-2 at T = 0.1 is 0.017–0.021. The oracle's own replies on seed 10 give 0.019: Zumi replies are short and repetitive by design, so the 4th criterion is read against the oracle's value.
- **Perplexity** is measured on the normalized rates (T = 1), as defined in 7a, for every model.
- **Finding before the test run** (selection seed 10). Fly perplexity is 1.559. The baselines give: 3-gram 1.581, 5-gram 1.384, 8-gram 1.325, RNN 1.278, GRU 1.275.
  - Raising the rates to 1/T does not close the gap. The best value is 1.281 at T = 0.5, and exact is 0.76 there.
  - Criterion 1 is therefore expected to fail. The criterion is not changed.
- **Baselines:** n-grams (3, 5, 8), the Elman RNN, and a numpy GRU (gradient checked in `tests/test_zumi3.py`).
- **Memory load:** 20 dialogues of 200 turns with 40 things each (Zumi 2's no-KC memory failed at 40).
- **Gate (set before the test run)** (`configs/phase7e_zumi3.yaml`). On every test seed:
  - perplexity below every baseline's mean;
  - sampled grammatical ≥ 0.95, appropriate ≥ 0.95 and exact ≥ 0.95;
  - max-length hits ≤ 0.01;
  - memory exact in the load condition ≥ 0.95.

  Control means must stay at or below these values:

  | control | measure | max |
  |---|---|---|
  | shuffled | exact | 0.70 |
  | no-KC memory under load | memory exact | 0.90 |

## Consequences

- 7b is compared with these numbers (`docs/notes/phase7-speaking.md`). The fly earns credit
  only for what it does beyond the n-grams (which get the form right) and what it does with
  memory.
- A ponytail note: the RNN is plain numpy (no torch in the environment). It is a baseline,
  not a tuned model, and a stronger one (GRU or LSTM) can be added if the fly gets close.
