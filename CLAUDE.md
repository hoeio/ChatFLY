# CLAUDE.md — ChatFLY

Standing instructions for Claude Code sessions working in this repository. Read this file and `docs/` before starting any work.

## One-line summary

A research and development project: teach a very simple structured "language" to a **connectome-based spiking simulation of the fly mushroom body (MB)** built from FlyWire, and build a **chatbot that can converse with that simulation in natural language** (including a natural-language ↔ fly-language translator).

"The fly" here is a **digital fly (a connectome-based computational model)**. This project does not claim communication with a living fly. Do not phrase it that way in docs, code, commits, or the README.

## Core design ideas (details in docs/)

- The MB's output is only the approach/avoid **valence signal** of the MBONs. The fly cannot "speak"; conversation takes the form of a question-answering acceptor (yes / no / uncertain).
- The key idea is **feedback loop = state transition table**. Feed the MBON output back as a state token, presented together with the next input; the fly then learns the transition (state, input) → next state by associative learning, acting as a finite-state machine. The first target task is **parity** (is the number of A's even?).
- The KC sparse expansion code is assumed to represent the **conjunction** of the state token and the input token, which is the assumed reason an XOR-structured transition table can be solved. **This is an assumption and is subject to verification.**
- Grammars a single fly cannot handle are to be explored later by connecting several flies in series or loops.

## Principles

1. **Prove that the structure does the work.** Every major result must come with controls: degree-preserving connectivity shuffle, same-size random network, lookup-table / small-RNN baselines, and ablations of key connections (KC–KC, MBON→DAN feedback, etc.). Never write "it works thanks to the connectome" without controls. (`docs/03-validation.md`)
2. **Minimize glue code.** Keep the natural-language ↔ token conversion (LLM layer) outside the fly model, and keep fly-to-fly and fly-to-decoder interfaces as fixed, unlearned, simple rules. Continually check that the translator is not solving the problem in place of the fly.
3. **Never cite literature or numbers from memory.** Verify papers and parameters (KC count, claw count, time constants, plasticity rule, etc.) via web search and record them in `docs/04-research-notes.md` with source URLs. Leave anything unverified marked "unverified".
4. **Negative results are results.** If a gate is not passed, do not hide it; report it to the user. Do not quietly change the goal.
5. **Reproducibility.** Every experiment saves its seed, config file (`configs/*.yaml`), and code commit hash under `results/<run_id>/`. No hard-coded magic numbers outside configs.
6. **Rate and spiking models share one interface.** The user prefers spiking, and the main experiments use spiking. The rate model is for fast validation and sweeps of the plasticity rule and loop logic. (`docs/decisions/0001`)

## Phases and gates

Follow the phase order in `docs/02-roadmap.md`. Each phase has **acceptance criteria**, and you do not move on until they pass. **Phase 2 (loop parity) is the branch point.** If it fails, do not proceed to Phase 3 or later; analyze the cause, lay out redefinition options, and report to the user.

## Data policy

- External connectome data (FlyWire/Codex, etc.) is **never committed.** `data/` is gitignored; record how to download it in `data/README.md`.
- Before using any data, check its license and terms of use and record them in `data/README.md`.
- The work environment's network is allowlist-based, so FlyWire access may be blocked. If blocked, do not work around it (retrying with other commands, mirrors, etc.); ask the user to provide the data files.
- Phases 0–2 are designed to run on **synthetic connectivity** (e.g., statistically matched random PN→KC wiring). The real connectome is needed from Phase 3.

## Development rules

- Python 3.13+ (required by Brian2, ADR 0003), package in `src/chatfly/`. Tests with `pytest`, linting with `ruff`.
- Everything in this repository is written in English: code, identifiers, comments, documentation, and commit messages.
- Simulator candidates: Brian2 (including GPU backends) or NEST. Record the rationale for the choice as an ADR in `docs/decisions/`. Verify versions and support status yourself.
- Write an ADR as `docs/decisions/NNNN-title.md` for each meaningful design decision (template: `docs/decisions/0000-template.md`).
- Commit in small units. **Push only when the user asks.**
- For long simulations, keep checkpoints and logs, estimate run time first, and if it is too large, verify at reduced scale before scaling up.

## Ask the user first about

- Changing the goal or evaluation criteria
- Redirecting after a gate failure
- Runs that need large compute resources
- Needing external data to be provided

Make other detailed implementation choices yourself and record the reasons in an ADR.

## Commands

```bash
pip install -e ".[dev]"        # install (simulator via extras: ".[sim]")
pytest                          # tests
ruff check src tests            # lint
```
