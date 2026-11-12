# 04. Research Notes

> Only record as fact what has been **verified by web search**. Leave items that could not be verified as "unverified". Write sources with their URL and access date.

## Parameters to verify (unverified)

| Item | Use | Status | Source |
|---|---|---|---|
| KC count per hemisphere, subtypes | Model scale | Verified (search-result excerpts of the article): ~2000 KCs, 7 types; MB ≈ 2200 neurons (Aso 2014) | https://elifesciences.org/articles/04577 |
| Number of PN claws per KC, randomness of PN→KC connectivity | Synthetic connectivity | Partly verified: EM 5.2 ± 1.6 claws/KC (Zheng 2022); 6.4 (Butcher 2012, as cited in a secondary source); one PN bouton per claw, apparently random glomerular combinations (Caron 2013); non-random biases exist (Zheng 2022). Exact numbers from Caron full text unverified | https://pmc.ncbi.nlm.nih.gov/articles/PMC9413950 , https://typeset.io/pdf/random-convergence-of-olfactory-inputs-in-the-drosophila-3hez23v7jq.pdf |
| KC activation sparseness (fraction per stimulus) | Target sparseness | Partly verified: population sparseness < 0.2 even at high concentration (Honegger 2011, Fig. 6). The per-odor values (e.g., 5–10%) are unverified | https://pmc.ncbi.nlm.nih.gov/articles/PMC3180869 |
| APL connectivity and inhibition mechanism | Maintaining sparseness | Verified (search-result excerpts of the article): KC→APL→KC negative feedback; disrupting it lowers sparseness, raises inter-odor correlation, impairs fine discrimination (Lin 2014) | https://pubmed.ncbi.nlm.nih.gov/24561998/ |
| Number of MBONs / DANs, compartment structure | Model composition | Verified (search-result excerpts of the article): 21 MBON types (34 cells/hemisphere), 15 compartments, 20 DAN types; each DAN type → 1–2 compartments (Aso 2014) | https://elifesciences.org/articles/04577 , https://elifesciences.org/articles/04580 |
| Dopamine-gated plasticity rule (time dependence, sign, eligibility trace time constant) | Learning rule | Partly verified: odor + DAN pairing → odor-specific KC→MBON depression (Hige 2015). Timing dependence / bidirectionality (Handler 2019) and the time constants are unverified. Plasticity mode may depend on the initial response size (eNeuro 2023) | https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068 , https://www.eneuro.org/content/10/10/ENEURO.0275-23.2023 |
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

### 2026-11-12 (Phase 0)

- **FlyWire:** Dorkenwald et al. 2024, Nature 634:124–138. 139,255 neurons, ~5×10⁷ chemical synapses, adult female. Companion paper: Schlegel et al. 2024 (8,453 cell types). https://pmc.ncbi.nlm.nih.gov/articles/PMC11228034 . KC count in FlyWire: unverified.
- **FlyWire license:** published data under CC BY-NC 4.0 (public release v630); the underlying EM data (Zheng, Lauritzen et al. 2018) is also CC BY-NC 4.0. Codex needs a Google login and offers downloads and static exports. https://flywire.ai/guidelines , https://edit.flywire.ai/tos.html , https://codex.flywire.ai
- **Visual KCs:** >90% of KCs get mainly olfactory input; ~8% get mainly visual input (Nat Commun 2024). https://doaj.org/article/6a55e744b753405e948e343f64597d75
- **Whole-brain LIF:** Shiu et al. 2024, Nature 634:210–219, doi 10.1038/s41586-024-07763-9. Official code link not verified. A community reimplementation exists: https://github.com/eonsystemspbc/fly-brain (Brian2 / NEST GPU / Brian2GeNN backends).
- **Simulators:** Brian2 2.11.0 (requires Python ≥ 3.13); NEST 3.10 (2026, pip-installable). Brian2CUDA / Brian2GeNN current status unverified. See ADR 0003.
- Still unverified: membrane/synaptic time constants, eligibility-trace time constant, glomerulus odor response profiles, configural learning, sequence learning, Krohn–Rhodes references.

### 2026-11-12 (Phase 3): measured directly from FlyWire v783 data

Source: Zenodo 10676866 (`proofread_connections_783.feather`) plus flywire_annotations `Supplemental_file1_neuron_annotations.tsv`; see `data/README.md`. These are counts from the data, not literature values.

- Annotated neurons: 139,248. Kenyon cells: 2,597 right / 2,580 left (5,177 in total; KCg 1,271 R / 1,219 L, KCab 799 R / 844 L, 527 R / 517 L with no subclass label). This is more than the "~2000 per hemisphere" of Aso 2014.
- Uniglomerular ALPNs 139 R / 138 L; multiglomerular 201 R / 199 L. This is consistent with "~150 uPNs" (Choi 2022) and with the FAFB count of more multiglomerular PNs.
- MBONs 48 per side, 35 types; DANs 165–166 per side; 1 APL per side (GABA).
- Right MB, claw = uPN→KC pair with ≥ 5 synapses: 2,329 KCs have ≥ 1 claw; 4.23 ± 1.61 claws per KC (range 1–9). PN fan-out is very uneven, from 0 to 480 KCs per PN (mean 70.8), which visibly departs from uniform random wiring (cf. Zheng 2022).
- KC→MBON (≥ 5 synapses): 146 KCs per MBON on average (0–957); 93% of KCs reach at least one MBON; 13 of 48 MBONs receive no KC input above threshold.
- KC↔APL: every olfactory KC connects to APL in both directions (21.7 and 17.6 synapses per KC on average).
- KC→KC: 153,139 synapses, but only 648 pairs have ≥ 5 synapses (weak and diffuse).
- License note: the Zenodo record is CC BY 4.0; FlyWire states CC BY-NC 4.0 for its data in general. The annotations repo has no license file but asks for citations.
