# 0002. Define the feedback loop as state-transition-table learning

- Status: Accepted (the hypothesis is verified at Gate B)
- Date: 2026-11-12

## Context

A feedforward mushroom body makes a valence judgment on a single input, so it has no state. Grammatical processing requires memory of earlier inputs (state). It is hard to hold long-sequence memory through plasticity alone.

## Options

1. Feedforward + fixed-length window: simple, but arbitrary-length rules are impossible.
2. Add an external stack / memory: increases expressive power, but the external structure ends up doing the computation.
3. A loop that returns the output as input: the state is carried in a discrete token, and the fly learns only the transition (state, input) → next state.

## Decision

Option 3. The first task is parity (four transitions, XOR structure). It rests on the hypothesis that the KC sparse expansion code separates (S, x) conjunctions so that XOR transitions can be solved by a linear readout.

## Consequences

- This is a **hypothesis**. Gate B verifies whether a single fly plus the loop works reliably.
- Error accumulation is the key risk. Countermeasures are discretization / re-encoding at every step, population readout, and step spacing.
- On failure, do not proceed to Phase 3 or later; discuss redefinition options (readout method, rule, simplifying the problem) with the user.
