# Corrected & Verified Results (2026-06-22)

Clean strict-load re-verification of all four (task × model) combos using
`kNN-Prediction/repro_patch.py` + `mcnemar_test.py` (py3.11 CPU). Every saved
ECE/Brier number is wrong; accuracy reproduces. Numbers below are the correct ones.

## 1. The saved ECE/Brier column is corrupt (all 4 combos)

Accuracy reproduces on load (codebert exact, graphcodebert within config noise), so
predictions are correct. The saved ECE is impossible given the model's own confidence —
confirmed by the binning-independent identity `ECE ≈ mean_conf − acc`:

| Combo | saved B1 ECE | **real B1 ECE** | mean_conf − acc | temp T\* |
|---|---|---|---|---|
| defect · codebert | 0.376 | **0.082** | 0.900 − 0.818 = 0.082 | 1.42 |
| defect · graphcodebert | 0.391 | **0.069** | 0.874 − 0.806 = 0.069 | 1.33 |
| vuln · codebert | 0.051 | **0.197** | 0.809 − 0.612 = 0.197 | 3.19 |
| vuln · graphcodebert | 0.052 | **0.228** | 0.836 − 0.609 = 0.228 | 3.75 |

Defect ECE was **inflated ~5×**, vuln ECE **deflated ~4×**. All models are over-confident;
temperature scaling fixes calibration in every case (e.g. vuln-gcb 0.228 → 0.061 at T=3.75).
**Action: regenerate every ECE/Brier in slides/tables. Acc/F1/MCC are safe.**

## 2. Full method tables (correct numbers)

### Defect — kNN + patches WIN (both model families)

| | Acc | F1-M | MCC | ECE | Brier |
|---|---|---|---|---|---|
| **codebert** B1 base | 0.818 | 0.779 | 0.734 | 0.082 | 0.288 |
| B3 +temp | 0.818 | 0.779 | 0.734 | 0.023 | 0.276 |
| M1 unpatched | 0.822 | 0.787 | 0.741 | 0.035 | 0.267 |
| **M1+ patched (P1+P3)** | **0.831** | **0.803** | **0.755** | **0.016** | 0.261 |
| **graphcodebert** B1 base | 0.806 | 0.761 | 0.717 | 0.069 | 0.300 |
| B3 +temp | 0.806 | 0.761 | 0.717 | 0.016 | 0.292 |
| M1 unpatched | 0.819 | 0.781 | 0.737 | 0.026 | 0.273 |
| **M1+ patched (P1+P3)** | **0.835** | **0.806** | **0.762** | **0.007** | 0.255 |

### Vuln — kNN HURTS; use temperature scaling alone (both families)

| | Acc | F1-M | MCC | ECE | Brier |
|---|---|---|---|---|---|
| **codebert** B1 base | 0.612 | 0.612 | 0.243 | 0.197 | 0.527 |
| **B3 +temp (best)** | 0.612 | 0.612 | 0.243 | **0.055** | 0.442 |
| M1+ patched | 0.597 | 0.591 | 0.237 | 0.046† | 0.464 |
| **graphcodebert** B1 base | 0.609 | 0.603 | 0.259 | 0.228 | 0.560 |
| **B3 +temp (best)** | 0.609 | 0.603 | 0.259 | **0.061** | 0.446 |
| M1+ patched | 0.596 | 0.596 | 0.198 | 0.085† | 0.457 |

† after P4 mixture recalibration. B3 (temp-only, no kNN) dominates every kNN variant on vuln.

## 3. Significance (McNemar, continuity-corrected χ² + exact binomial)

| Comparison | defect·cb | defect·gcb | vuln·cb | vuln·gcb |
|---|---|---|---|---|
| base → patched method | +84, **p=4e-6** | +194, **p=2e-19** | −41, p=0.05 (harm) | −36, p=0.09 ns |
| patches vs unpatched | +58, **p=5e-5** | +105, **p=3e-10** | +7, p=0.49 ns | −42, **p=0.008 (harm)** |
| original method vs base | +26, p=0.044 | +89, p=3e-8 | −48, p=0.015 (harm) | +6, p=0.77 ns |

Note: the saved "p<1e-12" for the original codebert method is also overstated — its real
gain over base is marginal (p=0.044). The **patches** convert it into a strong result.

## 4. The dichotomy (thesis spine)

Retrieval-augmentation + patches **help iff the representation separates classes**:

- **Defect** (MCC ≈ 0.72–0.76, separating): significant accuracy + calibration gains, both
  families (p ≤ 4e-6). P2 selective: retrieval-reliability **beats** model confidence
  (acc@50% cov: cb 0.937 vs 0.932; gcb 0.938 vs 0.918).
- **Vuln** (MCC ≈ 0.26, overlapping per ReVeal): kNN flat-to-harmful; patches neutral
  (cb) to **significantly harmful** (gcb, p=0.008). P2: model confidence ≥ retrieval.

One mechanism, opposite outcomes, significant on both sides across two model families.

## 5. What was actually wrong / fixed
- **knn_temperature=10** on L2-normalized embeddings made distance-weighting ~uniform
  (dead). Fixed: `'auto'`/0.1 (P1).
- kNN mixture never re-calibrated → inflated M1 ECE. Fixed: P4 temp on the blend.
- No class-prior correction → minority classes lost. Fixed: P3 (helps defect, hurts vuln).
- Saved graphcodebert datastores are a different embedding space (block_size/config) than a
  plain re-extraction → spurious kNN crash. Fixed by `--rebuild_datastore`.

Harness: `repro_patch.py --task {defect,vuln}_{codebert,graphcodebert} [--rebuild_datastore]`,
`mcnemar_test.py --task ...`, per-combo `patch_results_*.json`, cached `out_*_*.npz`.
