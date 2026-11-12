# 00. Vision and Scope

## Goal

Build a system in which a mushroom-body simulation constructed from the connectome **learns a very simple structured language**, and a person can interact with it **as if conversing** in natural language.

## Honest framing

- This system is not a conversation with a real fly. It is a **digital fly simulation for exploring the computational capabilities of connectome circuits**.
- The only "utterances" the mushroom body can produce are valence signals (good / bad / novel) and combinations of them. Conversation is closer to a question-answering acceptor (yes / no / uncertain).
- The more the "translator" does, the more the contribution of the fly circuit is obscured. The scientific value of this project lies in **evidence that the fly circuit actually contributes** (controls, ablations).

## In scope

- Spiking simulation of a FlyWire-based (or synthetic) mushroom body
- Associative learning through dopamine-gated plasticity
- Languages defined as token sequences (rules expressible as finite-state machines)
- Feedback loops and, if needed, multi-fly series / loop / parallel structures
- A natural-language ↔ token translation layer (LLM) and a conversational interface

## Out of scope (for now)

- Experiments on living flies
- Nested / recursive grammars (AⁿBⁿ, i.e., context-free and above). Without external memory (a stack) these are not a goal.
- Whole-brain simulation (extended only partially, and only when the needed subcircuits are identified)
- Free-form open-ended conversation. Conversation stays within the small learned language.

## Success criteria (summary)

1. The connectome-based mushroom body reproduces classical olfactory conditioning.
2. Through the feedback loop it performs state-based rules such as parity with high per-step accuracy, and error accumulation is controlled.
3. There is evidence that the connectome structure contributes meaningfully compared with controls (shuffled / random / baselines).
4. A natural-language chatbot interface works on top of the above results.

Detailed acceptance criteria are in `02-roadmap.md`.
