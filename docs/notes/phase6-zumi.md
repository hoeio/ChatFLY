# Phase 6: Zumi, a conversational language for one fly

Goal (user, 2026-11-13): a really simple language that can hold a conversation, with a
guidebook for people (`docs/zumi-guide.md`), and a fly trained to use it. Design and criteria:
ADR 0010. Synthetic MB at the Phase 4 parameters, except `eta_dep` (v2, below).

## 1. Choosing the vocabulary (`results/20261113-195555_phase6_vocab_sweep_s0`)

Rate model, selection seeds 10–19, seeds with step accuracy ≥ 0.99:

| vocabulary | seeds ≥ 0.99 | PN-direct |
|---|---|---|
| ring3 zu ki | 10 | 0.433 |
| **ring3 zu ki ta (Zumi)** | **10** | 0.344 |
| ring3 zu ki ho ("go home") | 4 | 0.611 |
| ring3 zu ki ta ho | 1 | 0.525 |
| ring4 zu ki ta | 8 | 0.275 |
| ring4 zu ki ho | 1 | 0.575 |
| ring4 zu ki ta pa | 1 | 0.256 |
| ring5 zu ki ta | 5 | 0.253 |

- A state-independent "go home" word breaks learning, as the Phase 4 margin analysis predicts.
- Margin ≤ 2 is not sufficient: the 4-place Latin square with `pa` (+2) fails.
- The 3-place ring with three moves is the largest vocabulary learned on every seed. Its
  PN-direct (no KC) control is at chance (1/3), so the KC layer is needed.

## 2. v1: baseline parameters on test seeds 0–9 — not passed

Transition level (`results/20261113-195613_phase6_zumi_fsm_s0`):

| | step acc mean (min) | seeds ≥ 0.99 | 50-step loops | PN-direct | teacher shuffle |
|---|---|---|---|---|---|
| rate | 0.989 (0.889) | 9 | 0.90 | 0.333 | 0.400 |
| spiking | 0.989 (0.939) | 7 | 0.72 | 0.329 | 0.334 |

Conversations (`results/20261113-200458_phase6_zumi_s0`, threshold 0.0 chosen on seeds 10–19):

- **rate:** 9/10 seeds answer every question correctly. Seed 5 has one transition, `fi zu → po`,
  that it always gets wrong, with high confidence. That gives **silent wrong answers**:
  answered accuracy 0.624, and 30% of conversations fully correct.
- **spiking:** answered accuracy ≥ 0.99 on 7/10 seeds. Seeds 0, 1 and 5 score 0.901, 0.934
  and 0.844. Their failures cluster on the same `fi zu → po` entry.

## 3. Fix: slower depression (`eta_dep` 0.5 → 0.3), chosen on dev seeds

The test seeds were not used for tuning. Candidates were run on dev seeds 20–59 (rate) and
20–29 (spiking), with the script below:

| change (rate, 40 dev seeds) | seeds ≥ 0.99 |
|---|---|
| baseline | 36 |
| n_train 40 | 39 |
| pn_active_frac 0.10 | 40 |
| pn_active_frac 0.20 | 14 |
| n_kc 4000 | 40 |
| **eta_dep 0.3** | **40** |
| eta_rec 0.1 | 40 |
| kc_theta 650 (rate only) | 40 |

On spiking (10 dev seeds), only parameters shared by both models were tried:

| change | seeds ≥ 0.99 |
|---|---|
| pn_active_frac 0.10 | 0, mean 0.566 (KCs too silent) |
| **eta_dep 0.3** | **10**, min 0.994, 50-step loops 1.00 |

So `eta_dep` = 0.3 was chosen, for the Zumi configs only. Other phases keep 0.5.
A plausible reading, not tested: smaller steps let the conjunctive KCs' weights settle before
the row/column-majority KCs push a transition the wrong way.

Tuning script (run from the repo root; it printed the two spiking rows above):

```python
import copy, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from chatfly.fsm import _job, _warm
base = yaml.safe_load(open("configs/phase6_zumi_fsm.yaml"))  # at commit 8543892 (eta_dep 0.5)
spec = base["tasks"][0]
CANDS = {"pn_active_frac 0.10": ("tokens", "pn_active_frac", 0.10),
         "eta_dep 0.3": ("plasticity", "eta_dep", 0.3)}
if __name__ == "__main__":
    jobs, names = [], []
    for name, (sec, key, v) in CANDS.items():
        c = copy.deepcopy(base); c[sec][key] = v
        _warm("spiking", c, spec)
        jobs += [("spiking", c, spec, s) for s in range(20, 30)]; names += [name] * 10
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(_job, jobs))
    # rate rows: the same _job with "rate", seeds 20-59, no pool
```

## 4. v2: `eta_dep` 0.3 on fresh test seeds 60–69 — passed

Transition level (`results/20261113-203848_phase6_zumi_fsm_s0`, commit 343105e):

| | step acc mean (min) | seeds ≥ 0.99 | 50-step loops | PN-direct | teacher shuffle |
|---|---|---|---|---|---|
| rate | 1.000 (1.000) | 10 | 1.00 | 0.411 | 0.289 |
| spiking | 1.000 (1.000) | 10 | 1.00 | 0.408 | 0.339 |

Conversations (`results/20261113-204727_phase6_zumi_s0`): 20 conversations × 8 turns per
seed, of which 1,115 are question turns per simulator. The threshold chosen on seeds 10–19 is
0.0 for both models.

| | answered acc, every test seed | coverage | conversations fully correct | teacher shuffle (mean answered acc) |
|---|---|---|---|---|
| rate | 1.000 | 1.00 | 1.00 | 0.335 |
| spiking | 1.000 | 1.00 | 1.00 | 0.339 |

**Verdict (ADR 0010 criteria): pass.** Answered accuracy ≥ 0.99 on every test seed for both
models, and the teacher-shuffle fly behind the same interface answers at chance (1/3).

The guidebook's example conversations and exercises (16 utterances) were run through
`python -m chatfly.zumi` on seed 0 with rate and with spiking. Every reply matched the guide.

## Caveats

- **Seen once, then fixed.** v1 failed on test seeds 0–9; the fix was chosen on other seeds
  and passed on new ones. The v1 result stays above.
- **The world is small**: 3 places and 3 moves. Larger rings fail (section 1).
- **Synthetic wiring only.** Three output groups are not grounded on real wiring (ADR 0007).
  There is no claim about connectome structure. The no-KC control (0.41, against chance
  0.33) shows that the KC layer does the work.
- The PN-direct control is a little above chance in v2 (0.41). It is still far below the fly.
- `bzz` (start over) is an interface rule, not learned.

# Zumi 2: talking about anything (ADR 0011)

Any word that is not a Zumi word is a thing. `fire ka mal ho.` teaches the fly (dopamine in
the matching property pair's outputs), and `fire ho ba?` / `fire ke?` read its outputs with
no learning. The interface stores no facts.

## Tuning (selection seeds 10–19 only)

Spiking, 40 things, `n_teach` 3 vs 5, with drive ×2 (`w_kc_mbon_mv` 2.0):

| n_teach | threshold | answered acc min | coverage | untaught answered |
|---|---|---|---|---|
| 3 | 0.2 | 1.000 | 0.676 | 0.008 |
| 5 | 0.2 | 1.000 | 0.946 | 0.056 |
| 5 | 0.3 | 1.000 | 0.766 | 0.009 |

- With 1 pairing, answered accuracy was 0.63–0.75.
- Averaging 4 sniffs (`n_ask` 4) alone reached a minimum answered accuracy of 0.958 at threshold 0.2, but with coverage 0.35.
- Drive ×2 alone gave coverage 0.36 at threshold 0.1.
- `pn_active_frac` 0.25 for things was worse for rate. Rejected.
- The chosen setting is `n_teach` 5, `n_ask` 1, drive ×2.
- Thresholds chosen on seeds 10–19 (40 things): rate 0.2, spiking 0.3.

## Test (seeds 60–69)

Run `results/20261114-002847_phase6_zumi2_memory_s0`. Each fly is taught 2 properties per thing,
20% of facts are changed later, and it is asked about every taught fact plus 10 things it never heard.
"Untaught answered" is the fraction of questions about never-taught things answered ya/na
(should be `hm`). Pass = answered acc ≥ 0.99 on all 10 seeds **and** untaught answered ≤ 0.05.

| model | things | answered acc mean (min) | seeds ≥ 0.99 | coverage | untaught answered | pass |
|---|---|---|---|---|---|---|
| rate | 10 | 1.000 (1.000) | 10 | 1.00 | 0.000 | yes |
| rate | 20 | 1.000 (1.000) | 10 | 1.00 | 0.000 | yes |
| rate | 40 | 1.000 (1.000) | 10 | 1.00 | 0.002 | yes |
| rate | 80 | 1.000 (1.000) | 10 | 0.96 | 0.021 | yes |
| rate | 160 | 0.997 (0.988) | 9 | 0.82 | 0.110 | no |
| rate | 320 | 0.977 (0.962) | 1 | 0.57 | 0.161 | no |
| spiking | 20 | 1.000 (1.000) | 10 | 0.77 | 0.001 | yes |
| spiking | 40 | 1.000 (1.000) | 10 | 0.80 | 0.004 | yes |
| spiking | 80 | 1.000 (1.000) | 10 | 0.76 | 0.058 | **no** (untaught) |

Controls (same interface and threshold):

| model | things | no KC: acc mean (min), untaught answered | teacher shuffle: acc mean |
|---|---|---|---|
| rate | 10 | 0.994 (0.944), 0.158 | 0.485 |
| rate | 40 | 0.899 (0.806), 0.442 | 0.490 |
| rate | 80 | 0.746 (0.626), 0.538 | 0.511 |
| rate | 320 | 0.579 (0.522), 0.489 | 0.500 |
| spiking | 20 | 0.990 (0.938), 0.108 | 0.490 |
| spiking | 40 | 0.938 (0.860), 0.257 | 0.501 |
| spiking | 80 | 0.796 (0.657), 0.361 | 0.523 |

**Verdict.** Capacity is 80 things for rate and 40 for spiking.
- Spiking at 80 things answered every question it was sure about correctly. It fails only because it answers too many questions about things it was never taught.
- The teacher-shuffle fly is at chance.
- Without the KC layer, the same rule fails the criteria at every size. Even at 10 things it answers 16% of never-taught questions, so the sparse KC code is what keeps facts apart.

## Caveats

- The spiking fly says `hm` to about 20–25% of taught facts at its threshold.
- The tuning (n_teach, drive ×2) was needed for spiking and was chosen on selection seeds only.
- Properties are four fixed pairs. Facts are independent: there are no relations or comparisons.
- Synthetic wiring. Word smells are random PN patterns, not real odors.
- The browser artifact runs a JavaScript port of the rate model (its own RNG). On 20 flies in node, at 40 things, it scored 1.000 answered accuracy, 0.993 coverage and 0.006 untaught answered.
