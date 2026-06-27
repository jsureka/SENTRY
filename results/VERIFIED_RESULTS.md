# Verified Results — fine-tuned anchors

Rigorous, strict-load verification of the reliability framework's flagship results on the four
fine-tuned anchors (CodeBERT / GraphCodeBERT × defect / vulnerability), via
`kNN-Prediction/reproduce_results.py` + `significance_test.py` (py3.11, CPU). Accuracy reproduces on
load; significance is paired McNemar. The full 7-dataset × 4-encoder grid is summarised in
[`FINAL_VERDICT.md`](FINAL_VERDICT.md).

## 1. Reliability win — defect (separable): accuracy *and* calibration

Gated-*k*NN engages and improves both accuracy and calibration, on both model families:

| | Acc | F1-M | MCC | ECE | Brier |
|---|---|---|---|---|---|
| **CodeBERT** base | 0.818 | 0.779 | 0.734 | 0.082 | 0.288 |
| +temperature | 0.818 | 0.779 | 0.734 | 0.023 | 0.276 |
| **SENTRY (gated-*k*NN)** | **0.831** | **0.803** | **0.755** | **0.017** | 0.261 |
| **GraphCodeBERT** base | 0.806 | 0.761 | 0.717 | 0.069 | 0.300 |
| +temperature | 0.806 | 0.761 | 0.717 | 0.016 | 0.292 |
| **SENTRY (gated-*k*NN)** | **0.835** | **0.806** | **0.762** | **0.007** | 0.255 |

Pure calibration (temperature) is accuracy-neutral; SENTRY additionally raises accuracy (+1.3–2.9pp)
and F1, while driving ECE below the calibration-only value.

## 2. Vulnerability (non-separable): calibration only, never-harm

The representation does not separate the classes, so retrieval is unreliable and the gate falls back
to the calibrated model — accuracy preserved, calibration still fixed:

| | Acc | F1-M | MCC | ECE | Brier |
|---|---|---|---|---|---|
| **CodeBERT** base | 0.612 | 0.612 | 0.243 | 0.197 | 0.527 |
| **+temperature (SENTRY)** | 0.612 | 0.612 | 0.243 | **0.055** | 0.442 |
| **GraphCodeBERT** base | 0.609 | 0.603 | 0.259 | 0.228 | 0.560 |
| **+temperature (SENTRY)** | 0.609 | 0.603 | 0.259 | **0.061** | 0.446 |

Temperature scaling fixes calibration (ECE 0.20–0.23 → 0.05–0.06) on a comparable base; retrieval is
correctly skipped (forcing kNN here is flat-to-harmful — see §3).

## 3. Significance (McNemar, continuity-corrected χ² + exact binomial)

| Comparison | defect·cb | defect·gcb | vuln·cb | vuln·gcb |
|---|---|---|---|---|
| base → SENTRY | +84, **p=4e-6** | +194, **p=2e-19** | −41, p=0.05 | −36, p=0.09 ns |

Defect: large, highly significant gains. Vulnerability: forcing retrieval is neutral-to-harmful — the
gate skips it, which is the design.

## 4. When retrieval engages — the separability scope

One mechanism, opposite outcomes by representation:

- **Defect** (MCC ≈ 0.72–0.76, separable): retrieval engages → significant accuracy + calibration
  gains; the gate's reliability signal beats model confidence for selective abstention
  (acc@50% coverage: cb 0.937 vs 0.932; gcb 0.938 vs 0.918).
- **Vulnerability** (MCC ≈ 0.26, overlapping): retrieval skipped → calibration-only; model confidence
  ≥ retrieval for abstention.

This anchor result is the binary-vs-multiclass slice of the broader separability scope (extended to
clone detection and the full grid in [`FINAL_VERDICT.md`](FINAL_VERDICT.md)).

## 5. Verification & data-integrity notes

- **Accuracy reproduces on strict load** (CodeBERT exact; GraphCodeBERT within config noise), so the
  predictions are correct.
- **The originally-saved ECE/Brier column was wrong** and has been regenerated. The binning-independent
  identity `ECE ≈ mean_conf − acc` confirms the corrected values (defect was inflated ~5×, vuln
  deflated ~4×): defect·cb 0.900 − 0.818 = 0.082; vuln·cb 0.809 − 0.612 = 0.197. Accuracy/F1/MCC were
  unaffected. Fitted temperatures corroborate (defect T≈1.4 mild; vuln T≈3.2–3.8 strong).
- **What was fixed in the predictor:** `knn_temperature=10` on L2-normalised embeddings made
  distance-weighting ~uniform (now `'auto'`); kNN mixture re-calibrated; class-prior correction added
  (helps defect, skipped on vuln). GraphCodeBERT datastores need `--rebuild_datastore` (shipped store
  is a different embedding space).

Harness: `reproduce_results.py --task {defect,vuln}_{codebert,graphcodebert} [--rebuild_datastore]`,
`significance_test.py --task ...`.
