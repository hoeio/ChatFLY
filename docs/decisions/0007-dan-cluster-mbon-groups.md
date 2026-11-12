# 0007. MBON output groups by DAN cluster, per-group readout scale

- Status: Accepted (user chose "compartment grouping" after Gate C v1 failed)
- Date: 2026-11-12

## Context

In Gate C v1 (`docs/notes/phase3-gate-c.md`), each seed dealt MBON types at random to the two outputs. The fly's own grouping is set by its dopamine neurons (DANs): each compartment is innervated by either the PPL1 or the PAM cluster. While implementing that grouping, two more problems surfaced:

1. **MBON hemisphere bug (ADR 0006).** MBONs were selected by the annotation `side`, which is the soma side. Several types (e.g. MBON01, 04, 05, 06) have their soma contralateral to their MB dendrites. The v1 "right" MBON set therefore missed these MBONs of the right MB and included their left-MB twins, which get almost no right-KC input.
2. **Per-MBON scale.** ADR 0006 scaled each MBON's KC input to sum to n_kc. A 2-KC atypical MBON then weighs as much in its group mean as a 957-KC MBON. With DAN groups, the PPL1 group (many small atypical MBONs) lost both transitions that target it, under either label assignment (rate, seeds 0–9).

## Decisions

1. **MBON selection:** all MBONs whose KC input comes mostly from this hemisphere's KCs (`connectome.py`, cache v2). This gives 48 MBONs, 39 of them with ≥ 1 KC edge of ≥ 5 synapses.
2. **Output groups (`mb.mbon_groups`, lists in the configs under `mbon_clusters`).** The rule, from Li et al. 2020 (verbatim): "the PPL1 neurons innervate the α' lobe, α lobe compartments 2 and 3, and γ lobe compartments 1 and 2"; "All other compartments are innervated by PAM DANs". MBON compartments come from Aso et al. 2014b Table 1 (MBON01–22) and Li et al. 2020 figure legends (MBON24–32). MBON33–35 come from page summaries only and are unverified. Sources and details are in `docs/04-research-notes.md`.
   - Output 0 (even) = PPL1 group (17 types). Output 1 (odd) = PAM group (13 types).
   - Excluded: MBON10 (hemibrain and Aso 2014 numbering may differ), 22 (calyx), 23 (compartment not found), and 30 / 33 (γ1γ2γ3 / γ2γ3, which span both clusters).
   - For "β1>α" and "γ4>γ1γ2", the compartment before ">" is taken as the dendritic one. This reading of the naming convention is inferred, not quoted.
   - KC coverage: 1,717 of 2,329 KCs reach both groups, 106 only PPL1, 468 only PAM, 38 neither.
   - The labels are arbitrary (parity has no valence). The `swap` variant tests the label assignment.
3. **Readout scale (`connectome.mbon_norm: group`).** Real synapse counts with one scale factor per output group, so the mean MBON input within each group sums to n_kc. Relative strengths within a group stay real. The decoder then compares two groups with equal mean drive, as the synthetic model does by symmetry. This is a fixed, unlearned rule.
4. **Selection bias, disclosed.** The `group` scale was chosen after seeing rate results on seeds 0–9 (per_mbon 0.675, global 0.875, group 1.000). It was confirmed on held-out seeds 20–29 (rate: 10/10 at 1.000; per_mbon on the same seeds: 0/10, 0.625; `results/20261112-193424_phase3_heldout_rate_s0`). The spiking model was not used for the choice.

## Consequences

- Gate C v1 results used the buggy MBON set and are superseded, but kept for the record.
- The PPL1/PAM split gives a biological meaning to the two outputs. It does not make the outputs valence: this task has no valence.
- MBON→DAN feedback and MBON–MBON connections are still not modeled.
