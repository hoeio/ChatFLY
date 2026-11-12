# 0004. Trial-level dopamine-gated KC→MBON rule with a recovery term

- Status: Accepted (the recovery term is an assumption; see Consequences)
- Date: 2026-11-12

## Context

Phase 1 needs a KC→MBON plasticity rule that is the same in the rate and spiking models (ADR 0001), and that produces conditioning, extinction and reversal (Gate A).

What the literature supports (see `04-research-notes.md`):
- Coincident KC and DAN activity causes depression of the KC→MBON synapse. This is well supported (Hige 2015, and Cohn 2015 / Berry 2018 / Handler 2019 as cited in Yamada et al. 2024).
- Potentiation is reported, but only in some conditions: it depends on the training regime, the initial response size (eNeuro 2023), the DA receptor type and timing (Handler 2019, as cited), and on NO/cGMP: cGMP-pathway activation *paired with KC activity* induces presynaptic LTP, slower than cAMP-triggered LTD, and NO is co-released by a subset of DANs (Yamada, Davidson & Hige 2024, J Physiol 602:2019–2045). KC activity alone has **not** been shown to reliably potentiate.
- Extinction is explained by a parallel opposing memory (Felsenberg 2018), not by erasure.

## Options

1. Depression only: best supported. But a pure depression rule cannot do reversal, because once both compartments are depressed for an odor they stay depressed. Extinction would need a modeled MBON→DAN feedback loop.
2. Depression plus recovery toward baseline when the KC is active and the compartment's DAN is silent. Simple, and gives extinction and reversal. The recovery term is an assumption. The closest biological mechanism is cGMP/NO-driven LTP paired with KC activity, but that is driven by a DAN co-transmitter, not by DAN silence.
3. A time-resolved rule with eligibility traces inside Brian2. Higher fidelity, but the timing data is unverified and this would make the rate model and the spiking model diverge.

## Decision

Option 2, applied **once per trial** from that trial's KC activity, in shared numpy code (`Simulator.learn`), for both models:

`Δw[c,k] = active_k · ( −η_dep · dan_c · w[c,k] + η_rec · (1 − dan_c) · (w0 − w[c,k]) )`

Here `active_k` means KC k fired at least `kc_active_hz` in the trial, and `dan_c ∈ [0, 1]` is the teacher signal for compartment c. Test presentations (`dan=None`) never change weights.

## Consequences

- The recovery term is an assumption. Every Gate A result is also reported with `η_rec = 0` (the `*_norec` metrics), so it is visible which properties depend on it.
- Extinction here is *recovery of the depressed synapses*, not the opposing-memory mechanism reported by Felsenberg 2018. This is a known mismatch with biology. Modeling MBON→DAN feedback is deferred to Phase 3, where the connectome provides that connectivity.
- There is no within-trial timing dependence. If Phase 2 needs it (for example, a teacher signal delivered at a delay), move the rule into Brian2 as an eligibility trace, and keep the rate model equivalent by integrating over the trial.
- Sources (accessed 2026-11-12): https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068 , https://www.eneuro.org/content/10/10/ENEURO.0275-23.2023 , https://www.elifesciences.org/articles/76712 , https://pesquisa.bvsalud.org/portal/resource/pt/mdl-38488688 (Yamada 2024; PMID 38488688 as listed there) , https://pubmed.ncbi.nlm.nih.gov/30245010/
