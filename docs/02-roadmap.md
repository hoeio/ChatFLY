# 02. Roadmap and Gates

Each phase must pass its acceptance criteria before the next one begins. If it does not pass, summarize the results and report to the user.

## Phase 0 — Foundations

- Project skeleton, config system, seeding / logging, abstract `Simulator` interface
- Compare and choose a simulator candidate (Brian2, NEST) → ADR
- Start surveying the needed literature and parameters (`04-research-notes.md`)
- **Acceptance criteria:** An empty model runs a simulation, and results are saved to `results/<run_id>/` together with config, seed, and commit. A run-time estimate by scale is documented.

## Phase 1 — Associative learning in a single mushroom body (synthetic connectivity)

- Implement synthetic PN→KC connectivity, APL, KC→MBON, DAN, and the plasticity rule
- Classical conditioning: after learning A+ / B−, read approach / avoid from the MBONs
- Explore the plasticity rule and parameter ranges with the rate model, then port to the spiking model
- **Acceptance criteria (Gate A):**
  - KC sparseness is within the target range.
  - MBON readout changes statistically significantly between before and after learning (multiple seeds).
  - Basic properties such as extinction and reversal are reproduced.
  - Rate and spiking results agree qualitatively.

## Phase 2 — Order discrimination and feedback-loop parity ★ branch point

- 2a. AB vs BA order discrimination (feedforward). Check whether a single fly represents order information
- 2b. Transition-table learning: present (state, input) conjunctions simultaneously and learn the four XOR transitions
- 2c. Close the loop: extend parity to length 10 → 50 → 100 and measure the error-accumulation curve
- 2d. Improve reliability through denoising (re-encoding), population readout, step spacing, etc.
- **Acceptance criteria (Gate B):**
  - After learning the four transitions, per-step accuracy ≥ 99% (a target value; if unreachable, report the achievable level and the cause)
  - The sequence-accuracy-vs-length curve is reported and robustness to noise levels is measured.
  - Compared against at least two controls (connectivity shuffle, same-size random network).
- **On failure:** Do not proceed to Phase 3 or later. Analyze the cause (KC sparseness, insufficient conjunction separation, noise accumulation, plasticity rule, etc.) and present redefinition options (change the rule, change the readout, simplify the problem) to the user.

## Phase 3 — Swap in the FlyWire connectome

- Extract KC→MBON, MBON→DAN, MBON–MBON, and APL connectivity, neurotransmitter predictions, etc. from FlyWire (the data may need to be provided by the user)
- Compare performance with the synthetic model on the same tasks
- Ablations: KC–KC connections, MBON→DAN feedback, removal of specific compartments
- **Acceptance criteria (Gate C):** Phase 2 tasks are reproduced with real connectivity, and the contribution of each structural component is quantified. If there is no contribution, report that fact.

## Phase 4 — Vocabulary and grammar expansion, multiple flies

- Tokens 4 → 6, additional rules (mod-3 counter, "no consecutive A", etc.)
- Two-fly loops, series, and parallel + arbiter comparisons
- **Acceptance criteria (Gate D):** A clear answer (yes or no) to whether any task a single fly cannot do becomes possible with multiple flies

## Phase 5 — Chatbot

- Natural-language ↔ token translation layer (LLM) and confidence-based responses
- Evaluate conversational scenarios: measure translation accuracy and fly response accuracy separately
- **Acceptance criteria (Gate E):** Question answering within the learned language works, and uncertain / out-of-scope input is handled honestly.

## Phase 6 — Zumi, a conversational language (added at the user's request, 2026-11-13)

- A tiny regular language (`docs/zumi-guide.md`) whose meaning is a world the fly tracks across conversation turns in its loop
- **Acceptance criteria (ADR 0010):** answered accuracy ≥ 0.99 on every test seed (rate and spiking) in random grammatical conversations, threshold chosen on held-out seeds; the teacher-shuffle control behind the same interface at chance; the no-KC control reported
- **Zumi 2 (ADR 0011):** the fly learns properties of any word during the conversation. Criteria: on every test seed, answered accuracy ≥ 0.99 and ≤ 5% of never-taught questions answered; capacity = the most things taught that pass; thresholds chosen on held-out seeds; no-KC and teacher-shuffle controls

## Phase 7 — a fly that speaks Zumi (added at the user's request, 2026-11-14)

- Goal (ADR 0012): the fly generates its Zumi replies word by word: grammatical, fitting the question, consistent with what it was told; no template replies from the interface
- **7a:** dialogue corpus from the grammar, generation metrics, n-gram and RNN baselines. Acceptance: the corpus agrees with the parser, the oracle scores 1.0 on every metric, baselines reported on test seeds
- **7b:** speaking fly v0 (MBON groups = reply words, previous word fed back); criteria set from 7a
- **7c (branch point):** the fly tracks the grammar state itself (several flies; Phase 4 capacity)
- **7d:** replies use the memory fly and the ring across turns
- **7e/7f:** Zumi 3 (pronoun, ask-back, sampled multi-sentence replies), GPT-2-level criteria
- **Status (2026-11-14): closed.** 7a passed. 7b–7f gates not passed; the results and what they show are in `docs/notes/phase7-speaking.md`



The mushroom body is a circuit designed to learn through a small number of trials, so transition-table learning itself (four transitions) is expected to be cheap. The time-consuming parts will be **hyperparameter tuning, achieving reliability, and long-sequence evaluation**. Always estimate simulation time before running.
