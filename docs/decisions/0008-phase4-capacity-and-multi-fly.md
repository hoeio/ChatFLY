# 0008. Phase 4: transition-table capacity of one fly, and multi-fly architectures

- Status: Accepted (the user chose the "capacity sweep" definition of a single fly's limit)
- Date: 2026-11-12

## Context

Gate D asks whether some task a single fly cannot do becomes possible with several flies. With the 2-group biological readout, one fly holds 1 bit of state, which would settle the question by construction. The user chose to define the limit empirically instead: a single fly gets one output group per state, and the task size is swept.

## Decisions

1. **Tasks** (`fsm.make_task`):
   - parity; mod-3 and mod-4 counters (A increments, B keeps);
   - "no consecutive A" (states ok / lastA / dead);
   - random tables of n_states × n_inputs, re-drawn per seed with stream `table`.
   - The protocol is Phase 2's, unchanged: union of the state and input tokens, one output group per state, the teacher depresses all non-target groups, graded eligibility, 20 training epochs, 20 tests per transition.
2. **Model:** synthetic MB at the Gate B parameters, so the sweep is fast. Real wiring cannot give more than 2 biologically grounded output groups (ADR 0007), so it is not used for the sweep.
3. **Multi-fly architectures** (`fsm.Flies`). Every interface is a fixed rule with no learning.
   - `ensemble`: k flies with different wiring, the same teaching, and group rates summed.
   - `routed`: one fly per input token. The current input token selects which fly is taught and asked. Each fly still receives the (state, input) union. Routing is a fixed rule set by the designer: the "parallel + arbiter" structure of `01-architecture.md`, with its stated caution about glue.
4. **Controls:**
   - `big`: one fly with k × n_kc KCs, size-matched to a k-fly ensemble;
   - `routed_pn`: the same routing, but each specialist is the rule applied to PNs, with no KC layer;
   - PN-direct and teacher shuffle for single flies.
5. **Gate D answer format:** "yes" requires a task family on which a single fly misses the per-seed 0.99 criterion while a multi-fly architecture meets it, *and* the size-matched and no-KC controls also miss it. Without those controls, the success could be extra neurons or the router alone.

## Consequences

- The routed architecture moves part of the conjunction (which input is present) into a fixed router. This is reported as glue, together with the `routed_pn` result that shows the flies' KC layers are still required.
- Loops and series of flies with learned interfaces are not explored.
