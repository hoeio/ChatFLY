# data/

This is where external connectome data (FlyWire / Codex, etc.) is placed. The contents of this directory are not committed (`.gitignore`).

## Datasets in use (downloaded 2026-11-12)

### FlyWire connectivity, release v783 (snapshot October 2023)

- Source: Zenodo record 10676866, "FlyWire Whole-brain Connectome Connectivity Data", version 783.0, published 2024-06-02, DOI 10.5281/zenodo.10676866. https://zenodo.org/records/10676866
- License of this record: **CC BY 4.0** (Zenodo metadata). The FlyWire project states CC BY-NC 4.0 for its published data in general (below). The user confirmed non-commercial use (2026-11-12), which satisfies both.
- Files, in `data/flywire_783/`; the md5 values are from Zenodo:
  - `proofread_connections_783.feather` (852 MB, md5 f48f972d262323a102aed49af1396b8a): neuron-to-neuron connections with synapse counts, per neuropil, with neurotransmitter probabilities.
  - `proofread_root_ids_783.npy` (1.1 MB, md5 e0e6c19732fd8c7a4e39a2d170105421): proofread neuron IDs.
  - Not downloaded: `flywire_synapses_783.feather` (9.5 GB, per-synapse points; not needed for connection counts) and the per-neuropil count files.
- Download:
  ```bash
  mkdir -p data/flywire_783 && cd data/flywire_783
  for f in proofread_root_ids_783.npy proofread_connections_783.feather; do
    curl -L -o $f "https://zenodo.org/records/10676866/files/$f?download=1"; done
  ```
- Cite: Dorkenwald et al. 2024; Buhmann et al. 2021; Heinrich et al. 2018; Eckstein, Bates et al. 2024 (as listed in the Zenodo description).

### FlyWire neuron annotations (cell types)

- Source: https://github.com/flyconnectome/flywire_annotations , `supplemental_files/Supplemental_file1_neuron_annotations.tsv` (31.7 MB, `main` branch, downloaded 2026-11-12) → `data/flywire_783/`.
- License: the repository has **no license file**. Its README asks to cite Berg et al. (2026), Schlegel et al. (2024), Tastekin et al. (2026), Matsliah et al. (2024) and Dorkenwald et al. (2024). Used here for non-commercial research with citation; not redistributed.
- Download: `curl -L -o data/flywire_783/Supplemental_file1_neuron_annotations.tsv https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv`

Preprocessing is in `src/chatfly/connectome.py`.

## License (checked 2026-11-12)

FlyWire published data: CC BY-NC 4.0 (non-commercial, attribution). Follow the citing guidelines at https://flywire.ai/guidelines . Codex (https://codex.flywire.ai) requires a Google login and agreement to the FlyWire ToS. Not downloaded yet.

## Notes

The work environment's network is allowlist-based, so FlyWire access may be blocked. If blocked, ask the user to provide the files.
Phases 0–2 use synthetic connectivity, so they can start without this data.
