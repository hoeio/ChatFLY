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
| [docs/progress-summary.md](docs/progress-summary.md) | What has been achieved so far, and what it does and does not show |
| [docs/decisions/](docs/decisions/) | Architecture decision records (ADRs) |

## Status

- Phase 0 (foundations): done. `docs/notes/phase0-runtime.md`
- Phase 1 (conditioning, synthetic MB): Gate A pass. `docs/notes/phase1-gate-a.md`
- Phase 2 (parity loop): Gate B failed with the first rule, then passed after a user-approved rule change (graded eligibility, ADR 0005). `docs/notes/phase2-gate-b.md`
- Phase 3 (FlyWire connectome): Gate C v2 near-pass (spiking 9/10 seeds ≥ 0.99, rate 10/10) after the MBON selection fix and the DAN-cluster readout (ADR 0007); strict criterion not met. No evidence that the real PN→KC or APL structure contributes. Accepted by the user as a near-pass. `docs/notes/phase3-gate-c.md`
- Phase 4 (capacity, multiple flies): one fly fails random transition tables with conflicting row/column majorities. Input-routed specialist flies solve them, but so does a single fly given the same router (a KC->MBON weight set per input). The gain is input-gated readout, not multiple mushroom bodies. Gate D answered NO (user decision). `docs/notes/phase4-gate-d.md`
- Phase 5 (chat layer, ADR 0009): a fixed-schema parser over the parity fly. Out-of-scope input is refused, and the teacher-shuffle control answers at chance. After the negation refusal, the parser makes no silent errors (held-out v2: 0.80 correct, the rest refused). Spiking reaches answered accuracy ≥ 0.99 on 8/10 test seeds, and the weak seeds mostly answer "uncertain". Gate E near-pass, accepted by the user. Chat: `python -m chatfly.chat configs/phase5_chat.yaml [--sim spiking]`. Demo site: `python -m chatfly.web configs/phase5_chat.yaml [--sim spiking] [--port 8000]`, then open http://127.0.0.1:8000. `docs/notes/phase5-gate-e.md`

- Phase 6 (Zumi, ADR 0010): a 10-word language for talking with the fly. The fly keeps track of where it is on a ring of three places, across turns, through its own feedback loop. A guidebook for people: [docs/zumi-guide.md](docs/zumi-guide.md). The first parameters failed on the test seeds. After a fix chosen on other seeds (`eta_dep` 0.3), it passed on new test seeds: every question correct in 200 conversations (1,115 questions) per model, rate and spiking, with the teacher-shuffle control at chance. Synthetic wiring. Talk to it: `python -m chatfly.zumi configs/phase6_zumi.yaml [--sim spiking]`, or in the browser: `python -m chatfly.web configs/phase6_zumi.yaml [--sim spiking] [--port 8000]`. `docs/notes/phase6-zumi.md`
- Zumi 2 (ADR 0011): talk about anything. Any other word is a thing; tell the fly what it is like (four pairs: good/bad, big/small, hot/cold, fast/slow) and it learns during the conversation by dopamine-gated plasticity, then answers from its synapses. Passes on test seeds up to 80 things (rate) and 40 things (spiking; 80 fails narrowly, 5.8% of never-taught questions answered against a 5% limit). Without the KC layer the same rule fails at 40 things. `docs/notes/phase6-zumi.md`
- Phase 7 (ADR 0012, closed): a fly that says its Zumi replies word by word, sampling each word from its MBON rates. The content comes from the ring, memory and comparator flies. On test seeds, replies are grammatical and appropriate 99.9% of the time and exactly right 98.8% of the time (n-gram, RNN and GRU baselines: about 76%, since they forget across turns). Without the KC layer, memory under load falls from 0.95 to 0.55. The final "GPT-2 level in Zumi" gate was not passed: perplexity is worse than a GRU's (1.46 vs 1.27), and memory under load is below 0.95 on 2 of 5 seeds. Much of the structure is designer routing. Talk to it: `python -m chatfly.web configs/phase7f_zumi3.yaml` (rate model). `docs/notes/phase7-speaking.md`
- Phase 8 (ADR 0013, in progress): hypotheses for teaching the fly a subset of Toki Pona (112 content words). 8a, output capacity: with a sparse KC code (16,000 KCs, all 5 claws) a 2-of-20 MBON code says the words as well as 115 one-word groups do, with step accuracy ≥ 0.994 on every test seed; without the KC layer, 0.29. On 1 of 5 seeds the end-of-sentence decision is outvoted. 8a-2 (340 PNs, the FlyWire count) does not help: on seeds 10–19 about half the networks lose one end-of-sentence context whatever the PN count, which points to a majority conflict, not input crowding. **8a-3 passes the gate:** teaching each rare context as often as the majority that shares its state token gives sentence exact ≥ 0.98 and steps ≥ 0.997 on every test seed (controls ≤ 0.002), at 16,000 KCs. **8b passes:** one fly without routing tracks the subset's grammar (14 states over word classes, up to two modifiers per phrase) with step and utterance accuracy 1.000 on every test seed at 2,000 KCs; without the KC layer, 0.36. Deduplicated corpus teaching beats frequency-weighted teaching (H5). 8b-2: replacing the tokenizer's word classes by a fly fails so far (class fly in series: utterance 0.88; grammar fly on the words themselves: 0.62). 8b-3: with a single-word threshold and balanced teaching the class fly classes every word (pipeline 0.999), but without its KC layer it does nearly as well (0.968), so the gate is not passed. The user kept the class fly (KC layer not needed). **8c passes:** the memory fly keeps facts about phrases with one modifier apart (0.990; no KC layer 0.500); two modifiers degrade to 0.823, as predicted (H3). **8d:** a Toki Pona question-answering exchange carried by four flies replies exactly 0.995 (memory without KC layer 0.455, speaker without KC layer 0.000); the class fly's KC layer is again not needed. Replacing the memory's predicate routing with a three-word KC conjunction fails (0.815), so that routing stays as glue. **8e** (a whole conversation: objects, "what" questions, changes of mind) replies 0.923 on test seeds; gate not passed (one seed's grammar fly misses one transition, one seed's memory under load). Unrouted object memory: 0.926. `docs/notes/phase8-toki-pona.md`

Run an experiment: `python -m chatfly.run configs/<name>.yaml` (results in `results/<run_id>/`).
