# 0005. Graded (rate-proportional) eligibility; error-gated teaching tested but not adopted

- Status: Accepted
- Date: 2026-11-12
- Amends: ADR 0004 (the form of the rule is unchanged; the eligibility becomes graded)

## Context

Gate B failed for the spiking model: 6/10 seeds reached 99% step accuracy and the mean was 0.889 (`docs/notes/phase2-gate-b.md`). In the failing seeds, both outputs fire almost equally on some transitions. KCs shared between transitions with opposite targets are overwritten by whichever transition was trained last. They also fire faster than pure-conjunction KCs (9.5–12 Hz vs ~6.4 Hz) and carry about half of the MBON drive. The readout weights KCs by rate, but the binary eligibility of ADR 0004 does not.

After the failure, the user chose to fix the learning rule; my recommended option was an error-driven rule. I therefore tested two changes on the transition table, on **held-out seeds 10–19**, not the Gate B seeds 0–9.

## Options tested

Spiking step accuracy over 10 held-out seeds (mean, seeds ≥ 0.99). Runs: `results/20261112-172735_phase2_sweep_on_error_s0`, `results/20261112-173849_phase2_sweep_always_s0`.

| Eligibility | Always-on teaching | Error-gated teaching (margin 0.3) |
|---|---|---|
| Binary (ADR 0004), η_dep 0.5 / η_rec 0.3 | 0.956, 6/10 | 0.882, 2/10 |
| Graded, elig_ref 20 Hz, η_dep 0.5 / η_rec 0.3 | **1.000, 10/10** | 1.000, 10/10 |

With graded eligibility, all 6 tested (η_dep, η_rec) combinations under error-gating reached 1.000 and 10/10. With binary eligibility, results ranged from mean 0.875 (min 0.025) to 1.000 depending on the step sizes.

1. Error-gated teaching: the teacher fires only if the output is wrong or uncertain, a stand-in for MBON→DAN error feedback (Jürgensen et al.; Huang et al. 2024). It is not sufficient with binary eligibility, and not needed with graded eligibility. It is also glue outside the MB model (CLAUDE.md principle 2).
2. Graded eligibility: `elig_k = clip(rate_k / elig_ref_hz, 0, 1)` for active KCs, so a synapse changes in proportion to its KC's firing, the same way it contributes to the MBON. This is enough on its own.

## Decision

Option 2 with always-on teaching, keeping η_dep = 0.5 and η_rec = 0.3 (the Gate A values), elig_ref_hz = 20 (arbitrary; chosen on held-out seeds). Error-gated teaching stays available as `protocol.teach: on_error`, but it is not used for the gates. Gate A and Gate B are both re-run with this rule, so there is one model.

## Consequences

- This departs from the literal "error-driven" choice. The evidence showed that the decisive fix is in the rule's eligibility, and the error gate would only add glue. If the user still wants error-driven teaching, it is one config flag and has been validated on held-out seeds.
- Graded eligibility is biologically plausible (more presynaptic activity, more cAMP/plasticity), but the exact dependence is unverified (see `04-research-notes.md`, plasticity row).
- The recovery-term caveat of ADR 0004 still applies.
