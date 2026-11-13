# Phase 7: a fly that speaks Zumi (ADR 0012)

## 7a: corpus, metrics, baselines

Run `results/20261114-081637_phase7a_corpus_s0` (`configs/phase7a_corpus.yaml`, seeds 60–64).
Each seed has 2,000 train, 200 dev and 500 test dialogues of 30 turns each (366k train tokens).
Test turn mix: teach 25%, ask 20%, what 10%, statement 10%, where 12%, yes/no 12%, bad 10%.

Sample (seed 60, test dialogue 0):

```
jdp gu ba        -> hm        ohd ka bon gu ra -> o
si ki zu po ba   -> ya        ohd ku ba        -> hm
si ki ki ta nu   -> fi        zu ki zu         -> o
```

Test results, mean over 5 seeds (the minimum is within 0.005 of the mean everywhere). Replies are generated greedily from the gold history:

| model | reply perplexity | next word | grammatical | appropriate | exact |
|---|---|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1-gram | 15.1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2-gram | 3.07 | 0.599 | 1.000 | 0.353 | 0.353 |
| 3-gram | 1.60 | 0.794 | 1.000 | 0.947 | 0.652 |
| 5-gram | 1.37 | 0.873 | 1.000 | 0.994 | 0.789 |
| 8-gram | 1.44 | 0.875 | 1.000 | 0.994 | 0.794 |
| RNN (128) | 1.34 | 0.874 | 1.000 | 0.999 | 0.790 |

Exact by kind:

| model | teach | statement | bad | ask | what | where | yes/no |
|---|---|---|---|---|---|---|---|
| 5-gram | 1.000 | 1.000 | 0.944 | 0.745 | 0.569 | 0.430 | 0.669 |
| 8-gram | 1.000 | 1.000 | 0.944 | 0.742 | 0.563 | 0.467 | 0.680 |
| RNN | 1.000 | 1.000 | 0.988 | 0.745 | 0.560 | 0.402 | 0.676 |

Readings:

- **The form is easy.** A 5-gram already replies with the right kind of answer 99.4% of the
  time, and the RNN 99.9%. Grammaticality and appropriateness alone can't show that the fly
  does anything: they must be compared with these numbers.
- **The content needs memory, and no baseline has it.**
  - *Ask:* 0.745 for every model from the 3-gram up. A 3-gram sees only `N prop ba`, so this is the rate of `hm` (never taught).
  - *What:* about 0.56, again the `hm` rate.
  - *Yes/no:* about 0.67, about the "no" rate (2 of 3 places are wrong).
  - *Where:* 0.40–0.47, a little above chance (1/3), from moves inside the utterance.
  - The RNN learned neither the facts taught earlier in the dialogue nor the ring place carried across turns.
- **The bar for the fly.** Exact replies on ask / what / where / yes-no above these levels,
  using the memory fly and the ring loop (7d). That is where the fly has something the
  baselines lack: plasticity during the conversation, and a state fed back through its loop.

Caveats: the RNN is a plain numpy Elman net with no tuning beyond the dev-chosen epoch. Its
dev perplexity was still creeping down (1.33 at epoch 30). A gated RNN might learn the ring
or the facts in context; it would be added if a fly result depends on beating it.

**7a acceptance: passed.** The corpus agrees with the parser (`tests/test_speak.py`), the
oracle scores 1.0 on every metric, and the baselines are reported on the test seeds.

## Proposed 7b criteria (to confirm with the user)

A speaking fly v0 sees the user's words and its own previous reply word. Its output groups
are the 18 reply words plus end-of-reply. On the test seeds it should reach:

- grammatical ≥ 0.95 and appropriate ≥ 0.95 (the 5-gram gets 0.994);
- controls (shuffled teacher, no KC layer, degree-preserving shuffle of PN→KC) clearly below.

Content (exact on ask / what / where / yes-no) is not expected before 7d.
