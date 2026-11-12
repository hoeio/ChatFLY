# 0006. How the FlyWire connectome enters the MB model

- Status: Accepted; decisions 3 and 4 and the MBON selection superseded by ADR 0007
- Date: 2026-11-12

## Context

Phase 3 swaps the synthetic wiring for FlyWire v783 (`data/README.md`). The model has only point neurons, one APL, two output compartments and no DANs, so several modeling choices are needed to map the connectome onto it. Everything outside the wiring stays at the Gate B values, so any difference comes from wiring.

## Decisions

1. **Hemisphere and cells.** Right hemisphere. PNs are uniglomerular ALPNs (139). Multiglomerular PNs are excluded, because they are mostly not cholinergic MB inputs (unverified) and the token code is defined over uPNs. KCs are all `Kenyon_Cell`s with at least one uPN claw (2329 kept, 268 dropped; mostly non-olfactory KCs).
2. **Claws.** A uPN→KC pair with ≥ 5 synapses (summed over neuropils) is one claw with the same weight as a synthetic claw. This gives 4.23 ± 1.61 claws per KC. The threshold is arbitrary; EM papers report 5.2 ± 1.6 claws (Zheng 2022), so ≥ 5 synapses is slightly strict.
3. **KC→MBON.** Pairs with ≥ 5 synapses. Their synapse counts are the weight factors, scaled so every MBON's factors sum to n_kc: an all-to-all synthetic MBON has the same total drive. Without this scaling MBONs fire at 0.4–2.3 Hz and the readout is starved. MBONs with no KC input are dropped (35 of 48 kept). *Correction (ADR 0007): MBONs were selected by soma side, which is wrong for types with contralateral somata (MBON01, 04–06); now selected by KC-input side.*
4. **Output compartments.** Whole MBON *types* are shuffled with the seed and dealt alternately to the two outputs (even / odd). The fly's compartments are defined by DAN innervation, which this model does not have, so this grouping is arbitrary. It is a source of seed-to-seed variance, which is intended.
5. **APL.** Per-KC KC→APL and APL→KC synapse counts, normalized to mean 1, scale the APL input and inhibition of each KC. With uniform vectors this is exactly the synthetic model.
6. **Not modeled yet:** KC–KC synapses (153k, but only 648 pairs have ≥ 5 synapses), MBON→DAN feedback (there are no DAN neurons in the model), MBON–MBON connections. Their ablations listed in `03-validation.md` need modeling work first; this is reported, not faked.

## Controls and ablations (Gate C)

- `shuffle`: degree-preserving PN→KC shuffle (each KC keeps its claw count, each PN its fan-out; no duplicate pairs).
- `random`: same number of PN→KC edges, uniform random.
- `apl_uniform`: real APL weights replaced by uniform ones. `apl_off`: no APL.
- `kc_mbon_all`: real KC→MBON mask and weights replaced by all-to-all with factor 1.

## Consequences

Effects at the default noise will probably be at ceiling, as in Phase 2, so the variants are compared under increasing membrane noise. If nothing differs, the report says "no evidence of contribution" (03-validation interpretation rule).
