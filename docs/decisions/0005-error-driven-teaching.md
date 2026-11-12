# 0005. Error-driven teaching for transition-table learning

- Status: Accepted (user decision after the Gate B failure, 2026-11-12)
- Date: 2026-11-12
- Amends: ADR 0004 (the plasticity rule itself is unchanged; this changes *when* the teacher DAN fires, and adds optional graded eligibility)

## Context

Gate B failed for the spiking model: 6/10 seeds reached 99% step accuracy and the mean was 0.889 (`docs/notes/phase2-gate-b.md`). In the failing seeds, both outputs fire almost equally on some transitions. The teacher fires on every trial, so KCs shared between transitions with opposite targets are overwritten by whichever transition was trained last. Shared KCs fire faster than pure-conjunction KCs (9.5–12 Hz vs ~6.4 Hz) and carry about half of the MBON drive. The user chose to change the rule (option "error-driven rule").

Literature support for error-driven DAN activity (see `04-research-notes.md`):
- In a spiking larval MB model, DANs encode prediction error through a feedback motif within the MB (Jürgensen et al.).
- MBON→DAN feedback exists and lets memory units interact (Huang et al., Nature 2024).
- Extinction-related MBONs drive DANs (Felsenberg 2017).
These support the *existence* of MBON→DAN feedback that could gate teaching. They do not show that the fly's DANs implement exactly this error gate.

## Options

1. Always teach (ADR 0004 as used for Gate B): fails Gate B.
2. Error-gated: the teacher DAN fires only if the decoded output is wrong or its confidence is below a margin. This is a stand-in for MBON→DAN error feedback; the gate is a fixed rule computed from the MBON readout, not learned.
3. A graded prediction-error DAN signal (DAN = target − MBON prediction). Closer to Jürgensen et al., but it adds a modeled feedback circuit and more parameters.

## Decision

Option 2, with step sizes and eligibility chosen by a sweep on **held-out seeds 10–19** (`configs/phase2_sweep_on_error.yaml`). Gate B is then re-run on seeds 0–9 with the chosen values, and Gate A is re-run with the same plasticity parameters so there is one model.

The chosen values and the sweep result are recorded in `docs/notes/phase2-gate-b.md` (rerun section).

## Consequences

- The error gate lives in the task's teacher (`parity.train_table`), outside the MB simulator. It is glue between MBON and DAN, so CLAUDE.md principle 2 applies: it is a fixed threshold rule on the output and is reported as such.
- The conditioning protocol (Gate A) keeps an always-on US: the US is the reinforcement, not a teacher. Only the plasticity parameters are shared.
- In Phase 3 the gate can be replaced by connectome MBON→DAN connections, which turns this assumption into something testable by ablation.
