# 03. Validation Plan

Core question: **"Does the connectome structure actually contribute to the computation, or is it doing something any small classifier could do?"**

## Metrics

| Metric | Definition |
|---|---|
| KC sparseness | Fraction of KCs active per stimulus (distribution) |
| Conjunction separability | Distance / overlap between KC activity patterns of (S, x) pairs |
| Step accuracy | Fraction of transitions output correctly |
| Sequence accuracy vs length | Curve of the fraction of sequences fully correct at length n |
| Confidence calibration | Agreement between output confidence and actual accuracy |
| Noise robustness | Performance under input noise, synaptic noise, and background firing rate |
| Learning efficiency | Number of trials needed to reach the target accuracy |

Report every metric as a mean and variance (or confidence interval) over **multiple seeds** (at least 10 recommended).

## Controls

1. **Connectivity shuffle:** Randomize connectivity while preserving synapse counts / degrees (e.g., separately for KC→MBON and MBON→DAN)
2. **Same-size random network:** Same neuron count, synapse count, and firing characteristics
3. **Baseline computers:** A lookup table and a small RNN / logistic regression on conjunction features. These are the upper bound and the "just a classifier" baseline
4. **Teacher-signal control:** Check whether performance drops when the correct signal is shuffled or delayed

## Ablations (Phase 3)

- Remove KC–KC connections
- Remove MBON→DAN feedback
- Remove APL inhibition (effect when sparseness collapses)
- Remove MBONs / DANs per compartment
- Interpretation rule: if an ablation breaks performance, record "that component contributes to the computation"; if nothing changes, record "no evidence of contribution".

## Generalization tests

- Sequences of lengths not used in training
- Order patterns not used in training (same transitions but new paths)
- New token combinations (when the vocabulary is extended). Rule transfer is likely to be hard, so record failure as a normal result

## Reporting format

Each experiment leaves the following in `results/<run_id>/`: a copy of the config file, the seed, the commit hash, metrics (JSON/CSV), and figures. Short per-experiment summaries go into notes under `docs/`, including negative results.
