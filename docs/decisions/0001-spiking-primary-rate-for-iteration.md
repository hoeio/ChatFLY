# 0001. Spiking model for the main experiments, rate model for iteration

- Status: Accepted
- Date: 2026-11-12

## Context

A choice is needed between fidelity of the mushroom body simulation and development speed. The user prefers a spiking model. A single mushroom body is small (a few thousand KCs or fewer per hemisphere plus a small number of MBONs / DANs / APL), so spiking is realistic. In addition, in the loop configuration, how spiking noise contributes to error accumulation is a key object of observation.

## Options

1. Spiking only: high fidelity and direct observation of noise effects. Initial parameter exploration is slow.
2. Rate only: fast, but misses noise-driven error accumulation.
3. Both, behind a common interface so they are swappable: secures development efficiency and fidelity together. Requires slightly more implementation.

## Decision

Option 3. Provide an abstract `Simulator` interface; use the rate model for fast validation of the plasticity rule and loop logic and for parameter sweeps, and the spiking model for the main experiments and reported results. Gate A checks whether the two implementations agree qualitatively.

## Consequences

- The choice of simulator library (Brian2 / NEST, etc.) is decided in a separate ADR in Phase 0.
- If the two models disagree, analyze the cause and give priority to the spiking results.
