# ChatFLY

A project to teach a very simple structured "language" to a **connectome-based spiking simulation** of the fly mushroom body built from FlyWire, and to build a chatbot that can converse with it in natural language.

> "The fly" means a **digital fly (a computational model)** built from the connectome. This project does not claim communication with a living fly.

## Idea

1. The mushroom body's output is the approach/avoid valence signal of the MBONs. A conversation with the fly therefore takes the form of yes / no / uncertain answers to "does this sequence follow the learned rule?".
2. Closing a **feedback loop** that returns the output as input turns the mushroom body into a finite-state machine that learns the transition (state, input) → next state by associative learning. The first task is parity.
3. If needed, several flies are connected in series or loops to divide up more complex rules.
4. An LLM translation layer turns natural language into token sequences and turns the fly's answers back into natural language.

## Documents

| Document | Contents |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Standing instructions for Claude Code sessions |
| [docs/00-vision.md](docs/00-vision.md) | Goals, scope, honest framing |
| [docs/01-architecture.md](docs/01-architecture.md) | Pipeline, token/loop specification, model composition |
| [docs/02-roadmap.md](docs/02-roadmap.md) | Phased plan and acceptance criteria (gates) |
| [docs/03-validation.md](docs/03-validation.md) | Controls, ablations, evaluation metrics |
| [docs/04-research-notes.md](docs/04-research-notes.md) | Literature and parameters to verify, open questions |
| [docs/decisions/](docs/decisions/) | Architecture decision records (ADRs) |

## Status

- Phase 0 (foundations): done. `docs/notes/phase0-runtime.md`
- Phase 1 (conditioning, synthetic MB): Gate A pass. `docs/notes/phase1-gate-a.md`
- Phase 2 (parity loop): Gate B failed with the first rule, then passed after a user-approved rule change (graded eligibility, ADR 0005). `docs/notes/phase2-gate-b.md`
- Phase 3 (FlyWire connectome): Gate C v2 near-pass (spiking 9/10 seeds ≥ 0.99, rate 10/10) after the MBON selection fix and the DAN-cluster readout (ADR 0007); strict criterion not met. No evidence that the real PN→KC or APL structure contributes. Accepted by the user as a near-pass. `docs/notes/phase3-gate-c.md`
- Phase 4 (capacity, multiple flies): one fly fails random transition tables with conflicting row/column majorities. Input-routed specialist flies solve them, but so does a single fly given the same router (a KC->MBON weight set per input). The gain is input-gated readout, not multiple mushroom bodies. Gate D answered NO (user decision). `docs/notes/phase4-gate-d.md`
- Phase 5 (chat layer, ADR 0009): a fixed-schema parser over the parity fly. Out-of-scope input is refused, and the teacher-shuffle control answers at chance. Gate E not passed: held-out paraphrases with negation are misparsed silently (12.5%), and spiking reaches answered accuracy ≥ 0.99 on 8/10 test seeds. Pending a user decision. `docs/notes/phase5-gate-e.md`

Run an experiment: `python -m chatfly.run configs/<name>.yaml` (results in `results/<run_id>/`).
