# 04. Research Notes

> Only record as fact what has been **verified by web search**. Leave items that could not be verified as "unverified". Write sources with their URL and access date.

## Parameters to verify (unverified)

| Item | Use | Status | Source |
|---|---|---|---|
| KC count per hemisphere, subtypes | Model scale | Unverified | |
| Number of PN claws per KC, randomness of PN→KC connectivity | Synthetic connectivity | Unverified | |
| KC activation sparseness (fraction per stimulus) | Target sparseness | Unverified | |
| APL connectivity and inhibition mechanism | Maintaining sparseness | Unverified | |
| Number of MBONs / DANs, compartment structure | Model composition | Unverified | |
| Dopamine-gated plasticity rule (time dependence, sign, eligibility trace time constant) | Learning rule | Unverified | |
| Membrane and synaptic time constants | LIF parameters | Unverified | |
| Glomerulus response profiles per odor | Token encoder | Unverified | |

## Prior work to verify (unverified)

- The FlyWire connectome paper, how to access the data, license / terms of use (including Codex)
- Whole-brain LIF simulation studies based on FlyWire, and their public code
- Computational models of the mushroom body (sparse coding, heterogeneous dopamine signals, learning rules)
- Configural / non-elemental learning (e.g., negative patterning) in behavioral experiments on flies
- Evidence for sequence (order) learning in flies / insects
- Current support status of neural simulators (Brian2, NEST, GPU backends)
- References on regular languages and the serial decomposition of finite-state machines (Krohn–Rhodes)

## Open questions

1. Can a single MB reliably learn XOR transitions of (state, input) conjunctions? At what KC sparseness / APL strength is it possible?
2. How much does residual KC activity between consecutive steps cause errors, and how much step spacing is needed?
3. How much does the choice of plasticity rule drive the results? A rule-sensitivity analysis is needed.
4. Can multi-valued states (three or more) be represented stably by the MBON readout?
5. Which connectome components (KC–KC connections, MBON→DAN feedback, etc.) contribute to sequence processing?
6. How can individual variation be modeled when only the wiring of a single connectome is available?

## Research log

(Add what was investigated here, together with dates and source URLs.)
