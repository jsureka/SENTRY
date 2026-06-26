# SENTRY — Results Summary

A training-free reliability framework for code classifiers: temperature calibration +
reliability-gated *k*NN retrieval + selective abstention, over a frozen model. Every number
reproduces from the `analysis/` scripts; no value is hand-edited.

## Reliability wins

**0. Baseline (measured, not novel): code classifiers are over-confident.** Mean confidence ≫
accuracy (vuln·CodeBERT 0.81 vs 0.61). Temperature scaling removes it — mean ECE 0.13 → 0.03 on
multiclass, 0.20 → 0.06 on vulnerability. Calibration of code models is known (Guo 2017; Zhou et al.,
ICSE 2024); SENTRY applies it. This is table stakes, reported because the SE detector papers omit it,
not claimed as the contribution.

**1. Accuracy and calibration together (separable tasks) — the part calibration cannot do.** Pure
calibration is accuracy-neutral; reliability-gated retrieval also raises accuracy.

| | accuracy | ECE | significance |
|---|---|---|---|
| Defect, CodeBERT (fine-tuned) | 0.818 → **0.831** | 0.082 → **0.017** | McNemar p = 4e-6 |
| Defect, GraphCodeBERT | 0.806 → **0.835** | 0.069 → **0.007** | p = 2e-19 |
| Multiclass (frozen, mean) | **+3.1pp** (7/10) | ↑ | risk–coverage (AURC) +0.021 |
| Clone, BigCloneBench | **+2.5pp** (4/4) | ↑ | — |

**2. Selective abstention.** The gate's reliability signal (neighbour distance + vote agreement)
improves risk–coverage on separable tasks (gate AURC Δ +0.021 multiclass).

**3. Never-harm fallback.** Where retrieval is unreliable the gate reverts to the calibrated model:
accuracy preserved on 17/18 vulnerability points; calibration still fixed.

## When it works — the separability scope

Retrieval engages iff the task is **separable** in the frozen embedding, independent of arity:

| Task family | separable | base acc | retrieval effect |
|---|---|---|---|
| Multiclass (CodeChef defect, POJ-104) | yes | 0.61–0.92 | accuracy +3.1pp |
| Binary clone (BigCloneBench) | yes | 0.84 | accuracy +2.5pp |
| Binary vuln (Devign, ReVeal, PrimeVul, DiverseVul) | no | 0.94–0.97\* | retrieval-neutral |

\*majority collapse (base F1-macro ≈ 0.49). Clone detection is the control: **binary yet separable**,
helped like multiclass — so the factor is separability, not binary-ness. Confirmed across **7 datasets
× 4 encoders = 32 points**.

## Limitations & future work

- On non-separable tasks (binary vulnerability — a representability ceiling, cf. Ding et al.,
  ICSE 2024) retrieval cannot help; SENTRY contributes calibration and abstention only.
- Separability is characterised post-hoc, not predicted before deployment.
- Future: auto-gate from the separability signal; compose retrieval with uncertainty estimators
  (kNN-UE, Hashimoto et al., NAACL 2025); broaden tasks and languages.

## Reproduce

```
analysis/methods_table.py     # calibration + accuracy + risk-coverage across the grid
analysis/clone_point.py       # BigCloneBench separability control
kNN-Prediction/reproduce_results.py --task defect_codebert   # fine-tuned anchor
kNN-Prediction/embed_clone.py # builds the clone caches
```
Outputs: `results/{methods_grid.csv, methods_summary.json}`.
