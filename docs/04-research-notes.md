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

### 2026-11-12 (Phase 3): MBON compartments and DAN clusters (ADR 0007)

The eLife pages were bot-blocked, so these were read on PMC: Li et al. 2020 (https://pmc.ncbi.nlm.nih.gov/articles/PMC7909955/), Aso et al. 2014a (https://pmc.ncbi.nlm.nih.gov/articles/PMC4273437/), Aso et al. 2014b (https://pmc.ncbi.nlm.nih.gov/articles/PMC4273436/).

- **PPL1 vs PAM (verified, Li 2020, Fig 6 legend):** "the PPL1 neurons innervate the α' lobe, α lobe compartments 2 and 3, and γ lobe compartments 1 and 2"; "All other compartments are innervated by PAM DANs".
- **MBON01–22 names (verified, Aso 2014b Table 1):** 01 γ5β'2a, 02 β2β'2a, 03 β'2mp, 04 β'2mp_bilateral, 05 γ4>γ1γ2, 06 β1>α, 07 α1, 08 γ3, 09 γ3β'1, 10 β'1, 11 γ1pedc>α/β, 12 γ2α'1, 13 α'2, 14 α3, 15 α'1, 16 α'3ap, 17 α'3m, 18 α2sc, 19 α2p3p, 20 γ1γ2, 21 γ4γ5, 22 calyx.
  - Li 2020 says hemibrain numbers are synonyms ("MBON14 for MBON-α3").
  - **Unverified:** whether hemibrain MBON10 (GABAergic, 3 cells, atypical) is Aso's MBON-10 (β'1, 8 cells).
  - MBON08 was "not found in the hemibrain volume".
- **MBON24–32 (verified, Li 2020, Fig 8 supplements):** 24 β2γ5, 25 γ1γ2, 26 β'2d, 27 γ5d, 28 α'3, 29 γ4γ5, 30 γ1γ2γ3, 31 α'1, 32 γ2.
  - **Unverified** (page summary only): 33 γ2γ3, 34 γ2, 35 γ2.
  - MBON23: compartment not found.
- **Atypical MBONs (verified, Li 2020):** "14 cell types, that have part of their dendritic arbors outside the MB lobes".
- **Valence logic (verified, Aso 2014b):**
  - "PPL1- γ1pedc DANs play a major role to mediate punishment signals"; "reward signals are mediated by a distributed set of PAM cluster DANs".
  - "all MBONs eliciting aversion were glutamatergic and all the MBONs eliciting attraction were either GABAergic or cholinergic".
  - Dopamine pairing gives "depression of the synapses from the active KCs onto MBONs innervating that compartment" (Li 2020 intro).
- **Measured (FlyWire v783):** MBON `side` is the soma side. For MBON01, 04, 05 and 06 the right-soma cell receives its KC input from left KCs, and vice versa. Example: right-soma MBON05 gets 9,926 synapses from left KCs and 0 from right KCs.

### 2026-11-14 (Phase 8): Toki Pona (ADR 0013)

- **Word list (verified):** https://sona.pona.la/wiki/nimi_pu (revision 37151, edited 9 July 2026). It has "120 main nimi pu, if ali is counted as a pronunciation variant of ale".
  - The page also lists 3 "synonyms" (kin, namako, oko), not counted.
  - The list is in `src/chatfly/tokipona.py` (`PU`).
  - Page license: CC BY-SA 4.0. The words are used as a list with attribution; no glosses are copied.
- **Word counts (verified, search summary of sona.pona.la "How many words"):**
  - pu (2014 book): 120 advertised, 123 with the synonyms;
  - ku (Toki Pona Dictionary, 2021): "137 essential words", up to 187 in total.
- **Grammar (verified, sona.pona.la Grammar and Particles pages and a flashcard summary; the community sources differ in detail, and the book is the main reference):**
  - `li` introduces the predicate, but not after a subject that is only `mi` or only `sina`;
  - `e` marks the direct object;
  - `la` ends a context phrase;
  - `pi` groups modifiers (`jan pi pona mute`);
  - `o` marks commands and vocatives.

  Sources: https://sona.pona.la/wiki/Grammar, https://sona.pona.la/wiki/Particles, https://sona.pona.la/wiki/mi_li_and_sina_li
- **Yes/no questions (verified, search summary of sona.pona.la Questions and V ala V pages and lipu sona lesson 7, 2026-11-14):**
  - a statement plus `anu seme` ("or what?"), or the verb repeated around `ala` (`sina kama ala kama?`);
  - yes is answered by repeating the verb, no by the verb plus `ala` (or `ala` alone).
  - 8d uses `anu seme`, so the grammar fly marks the question by its state. The reply is a whole sentence: the statement, or the statement with `ala`.

  Sources: https://sona.pona.la/wiki/Questions?oldid=669, https://sona.pona.la/wiki/V_ala_V, https://lipu-sona.pona.la/en/7
- **License of Lang's texts (partly verified, sources conflict):**
  - Wikisource lists the 2021 dictionary as CC0 (https://en.wikisource.org/wiki/Toki_Pona_Dictionary). Older sona pona revisions say CC BY 4.0.
  - Wikipedia says the 2014 English book was dedicated to the public domain in April 2026 (https://en.wikipedia.org/wiki/Toki_Pona).
  - **Unverified** on Lang's own site. We use only the word list and our own grammar generator, no text from the books.

### 2026-11-23 (Phase 9): Toki Pona corpus and grammar (ADR 0014)

- **Tatoeba (verified):** 80,461 Toki Pona sentences in the weekly export of 2026-11-23 (`data/README.md`); text license CC BY 2.0 FR (some CC0), per https://sona.pona.la/wiki/Tatoeba and https://en.wikipedia.org/wiki/Tatoeba . The wiki's "over 56,000 sentences" (March 2024) is older than our count.
- **Measured on that file:** 95.8% of clauses use only nimi pu; median length 6–7 words; `la` 20%, `pi` 19%, `lon` 20%, `tawa` 22% of clauses (ADR 0014).
- **Prepositions (verified, search summary of https://sona.pona.la/wiki/Prepositions):** kepeken, lon, sama, tan, tawa; all are also content words, so some sentences are ambiguous.
- **Preverbs (verified, same source family, https://sona.pona.la/wiki/verb):** the core preverbs in *Toki Pona: The Language of Good* are awen, kama, ken, lukin, sona, wile; no particle marks them. Our grammar treats them as predicate words (no class of their own).
- **`la` (verified, https://sona.pona.la/wiki/la):** what comes before `la` is context for the rest; a fronted `lon` phrase may drop `lon`; it can mark a condition.
- **FlyWire ALPNs per side:** 139 uniglomerular + 201 multiglomerular = 340 (already recorded for ADR 0013 8a-2); Phase 9 uses a 300-PN key space at 10% (arbitrary, chosen on seed 10).
