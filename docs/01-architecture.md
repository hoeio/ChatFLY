# 01. Architecture

> The numbers and biological assumptions below are **starting points**. Parameters are moved into `configs/` only after their sources are verified in `04-research-notes.md`. Values not yet verified are arbitrary defaults.

## Pipeline

```
natural language
  → [LLM: intent / slot extraction]            ← glue layer (outside the model)
  → token sequence (e.g., A B A)
  → [odor encoder: token → PN activation pattern]
  → [mushroom body simulator: PN→KC→MBON, APL, DAN]   ← the core of the research
  → [decoder: MBON activity → approach / avoid / uncertain]
  → [LLM: natural-language response generation]
```

In the loop configuration, the decoder output returns as the input token (state token) of the next step.

## Modules and interfaces

| Module | Responsibility | Interface principle |
|---|---|---|
| `tokens` | Vocabulary definition, token ↔ PN activation vector | Vocabulary lives in config files. Record the overlap between token PN patterns as a metric |
| `mb` | Mushroom body model (PN, KC, APL, MBON, DAN) | Abstract `Simulator` interface. Rate and spiking implementations share one signature |
| `plasticity` | Dopamine-gated plasticity rule | Rules are swappable. The rule is a major degree of freedom of the model, so record it in an ADR |
| `readout` | MBON population → discrete output | Time-window averaging, thresholds, confidence. Output is always a discrete token plus confidence |
| `loop` | Feed the output token back as the next input | Fixed rule (no learning). This is where denoising happens |
| `tasks` | Task definitions (conditioning, order, parity, ...) and teacher signal | The DAN teacher signal is given only on correct transitions |
| `eval` | Metrics, controls, ablations | `03-validation.md` |
| `chat` (later) | LLM translation layer | Separate from the model core. The translator does not replace the fly |

## Mushroom body model (draft)

- **PN → KC:** Mostly random connectivity (a small number of PN claws per KC). Phases 0–2 use synthetic random connectivity; Phase 3 swaps in FlyWire connectivity.
- **KC sparseness:** Global APL inhibition keeps roughly 5–10% of KCs active per stimulus (target value to be fixed after checking the literature).
- **KC → MBON:** The synapses that are learned. MBONs and DANs are distinguished per compartment.
- **DAN:** Per-compartment activation as the teacher signal (when correct) or as a reward / punishment signal.
- **Plasticity:** Depression of KC→MBON synapses when DAN activity and KC activity overlap in time, including an eligibility trace. The exact form is decided after checking the literature.
- **Neurons / synapses:** LIF (conductance-based as a candidate). Spiking noise is deliberately kept. How noise accumulates in the loop is itself an object of observation.

## Viewing the loop as a state transition table

The heart of the feedback loop is that the fly learns a **transition function δ(S, x) → S'** by associative learning.

- When the state token S and the input token x are presented simultaneously (or close in time), the KC sparse code represents the **conjunction** of (S, x).
- The four parity transitions: (even, A)→odd, (odd, A)→even, (even, B)→even, (odd, B)→odd. This is an XOR structure.
- Learning uses a teacher signal that activates the corresponding compartment's DAN on correct transitions.
- The output S' is read from the MBON population, **discretized**, and re-encoded as the state token for the next step.
- With three or more states (e.g., a mod-3 counter) the output must be multi-valued. In that case, consider splitting the MBONs into groups and reading them as one-hot.

## Error accumulation

If per-step accuracy is p, the accuracy over a sequence of length n is about pⁿ (p=0.99, n=100 gives about 37%).

- Use population readout with time-window averaging.
- Re-encode as a discrete token at every step so that noise is cleaned up.
- Leave enough spacing between steps so that residual KC activity from the previous step does not mix in. The spacing is a parameter to be reported together with the error rate.

## Multiple flies (later exploration)

| Structure | Description | Caution |
|---|---|---|
| Series | Fly 1 → Fly 2 → … | Without state it is limited to a fixed-length window. Information bottleneck at each stage |
| Parallel + arbiter | Per-rule specialists + a combiner | The arbitration rule is set by the designer (more glue) |
| Loop | Output returned as input (1–2 flies) | Provides state. Current top priority |

The interface between flies must be uniform: MBON output (a discrete token) → fixed rule → the next fly's input token. Individual variation is modeled by noise in the initial learning state or, where available, by mixing in the MB of other connectomes.

## Chatbot layer (later)

- Slot extraction that converts natural language into token sequences of the fly language is restricted to a **small fixed schema**.
- When turning the fly's response (yes / no / uncertain + confidence) back into natural language, do not invent information the fly cannot produce.
- If confidence is low, respond "uncertain". The translator must not hide this.
