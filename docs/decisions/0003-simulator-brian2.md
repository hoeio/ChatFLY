# 0003. Spiking simulator: Brian2

- Status: Accepted
- Date: 2026-11-12

## Context

ADR 0001 needs a spiking backend behind the shared `Simulator` interface. The model is small (about 2000 KCs per hemisphere plus tens of MBONs/DANs; see `04-research-notes.md`). The main unknown is the plasticity rule (dopamine-gated, eligibility trace), which will be changed often.

## Options

1. **Brian2**: the latest release is 2.11.0 on PyPI (requires Python ≥ 3.13, numpy ≥ 2.3). Neuron and synapse models, including plasticity, are written as equations in Python, so a new DAN-gated rule is a string change. The GPU backends Brian2CUDA and Brian2GeNN exist; their current release status is unverified.
2. **NEST**: the latest release is 3.10 (2026), which is now pip-installable and drops SLI. It is fast at large scale with MPI. Custom neuron and plasticity models need NESTML or C++, which adds friction for a rule that will change often.

## Decision

Brian2. The network is small, so NEST's scaling advantage does not matter, and changing the plasticity rule must be cheap. Verified locally: brian2 2.11.0 installs and runs on Python 3.13.5 / numpy 2.5.4 (macOS arm64), with the Cython runtime target.

## Consequences

- The project now needs Python ≥ 3.13 (`pyproject.toml`). With Python 3.11, pip resolves an old Brian2 that fails to import with numpy 2 (`np.ndarray.ptp` was removed).
- Default target is `cython`. The first build of a new model compiles (seconds) and is cached afterwards. `numpy` is the fallback target (used in tests).
- Revisit if large sweeps need multi-core or GPU: consider Brian2 standalone mode (`cpp_standalone`), or Brian2CUDA after checking its status.

Sources (accessed 2026-11-12): https://pypi.org/pypi/Brian2 , https://brian2.readthedocs.io/en/stable/introduction/release_notes.html , https://nest-simulator.readthedocs.io/en/stable/whats_new/v3.8/index.html , https://nest-simulator.org/mailinglist/hyperkitty/list/users@nest-simulator.org/2026/5/ , https://www.frontiersin.org/articles/10.3389/fninf.2022.883700/text
