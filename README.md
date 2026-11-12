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

Phase 0 (foundations) done: shared `Simulator` interface (rate + Brian2 spiking), run pipeline (`python -m chatfly.run configs/<name>.yaml`), run-time estimate in `docs/notes/phase0-runtime.md`. Next: Phase 1.
