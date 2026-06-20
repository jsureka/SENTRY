# 4. Experimental Evaluation

We evaluate the **Confidence-Informed Gated $k$NN (CI-Gated $k$NN)** framework on the two
benchmarks used by CodeImprove (Rathnasuriya et al., ICSE 2025): the **CodeChef** dataset for
multi-class defect prediction (4 classes; 21,647 / 5,411 / 6,764 train/val/test) and the
**Devign** dataset for binary vulnerability detection (21,854 / 2,732 / 2,732). Using the same
datasets and splits lets us position our results directly against CodeImprove's reported baselines.
We study two encoders, **CodeBERT** and **GraphCodeBERT**, and ask three questions:

- **RQ1 (Accuracy & Calibration):** Does CI-Gated $k$NN reduce confidence miscalibration and/or
  improve discriminative performance, post-hoc and without retraining?
- **RQ2 (Safety under shift):** Does Split Conformal Prediction maintain its coverage guarantee
  when the test distribution shifts via semantic-preserving transformations (SPTs)?
- **RQ3 (Efficiency):** Does the entropy-adaptive gate let us recover always-on-$k$NN quality while
  skipping retrieval on confident inputs? *(Not yet measured — see §4.5, Limitations.)*

## 4.1 Configurations

We report four configurations under one metric harness (Accuracy, F1-Macro, ECE [15-bin], Brier;
McNemar paired test for significance):

| Config | Description |
|---|---|
| **B1** | Base fine-tuned model, raw softmax. |
| **B3** | + Temperature scaling (calibration only; argmax unchanged → same Acc/F1 as B1). |
| **B4** | + Always-on $k$NN interpolation (every sample). |
| **M1 [ours]** | + Confidence-gated, entropy-adaptive $k$NN (full CI-Gated $k$NN). |

**Reference points (CodeImprove, ICSE 2025, Table III; same datasets):** base-model accuracy
CodeBERT/GraphCodeBERT = **81.98% / 81.91%** on CodeChef defect and **62.74% / 62.40%** on Devign;
their *adapted* method adds up to **+8.78%** accuracy via input transformation, and their
out-of-scope **detector reaches AUC 0.924**. The retrieval-only kNN-for-Vuln method (Du et al.,
EMNLP 2022) reports **F1 ≈ 0.66** on Devign.

> Note: an earlier draft cited "0.613 / 0.715" literature baselines; these do not correspond to
> CodeChef and have been removed. The correct same-dataset reference is CodeImprove Table III above.

## 4.2 Accuracy & Calibration (RQ1)

### Table 1 — CodeChef Defect Prediction (4-class)

| Arch | Method | Acc | F1-M | ECE↓ | Brier↓ | McNemar vs B1 |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| CodeBERT | B1 base | 0.788 | 0.736 | 0.359 | 0.513 | — |
| CodeBERT | B3 +temp | 0.788 | 0.736 | 0.053 | 0.324 | n/a (argmax = B1) |
| CodeBERT | B4 always-on $k$NN | **0.811** | **0.773** | 0.032 | 0.283 | p = 6e-15 ✓ |
| CodeBERT | **M1 CI-gated $k$NN** | 0.808 | 0.767 | **0.028** | 0.283 | **p = 6e-13 ✓** |
| GraphCodeBERT | B1 base | 0.806 | 0.762 | 0.372 | 0.502 | — |
| GraphCodeBERT | B3 +temp | 0.806 | 0.762 | 0.044 | 0.303 | n/a |
| GraphCodeBERT | B4 always-on $k$NN | **0.821** | **0.784** | 0.041 | 0.269 | p = 2e-7 ✓ |
| GraphCodeBERT | **M1 CI-gated $k$NN** | 0.821 | 0.783 | **0.034** | 0.269 | **p = 3e-8 ✓** |

**Discussion (defect).** Three honest observations:
1. **Calibration is dominated by temperature scaling, not $k$NN.** B1 is severely overconfident
   (ECE 0.36–0.37); a single scalar temperature (B3) collapses this to 0.04–0.05. $k$NN adds only a
   marginal further reduction (to ~0.03). We state this plainly rather than attribute the
   calibration win to retrieval.
2. **$k$NN delivers a small but statistically significant accuracy gain.** B1→M1 improves accuracy
   by +2.0 pp (CodeBERT) and +1.5 pp (GraphCodeBERT), significant under McNemar (p < 1e-12 and
   p < 1e-7). This is the substantive retrieval contribution.
3. **Our base model is comparable to — not better than — CodeImprove's.** Our B1 (78.8 / 80.6) sits
   slightly below CodeImprove's reported base (81.98 / 81.91); with $k$NN, GraphCodeBERT/M1 (82.1)
   reaches their base level. We therefore make **no claim of accuracy superiority over CodeImprove**;
   our contribution is a post-hoc reliability layer, evaluated on a comparable base.

### Table 2 — Devign Vulnerability Detection (binary) — *negative result*

| Arch | Method | Acc | F1-M | ECE↓ | McNemar vs B1 |
|:--|:--|:--:|:--:|:--:|:--:|
| CodeBERT | B1 base | 0.622 | 0.621 | 0.057 | — |
| CodeBERT | M1 CI-gated $k$NN | 0.627 | 0.626 | 0.092 (worse) | p = 0.36 ✗ |
| GraphCodeBERT | B1 base | 0.619 | 0.614 | 0.036 | — |
| GraphCodeBERT | M1 CI-gated $k$NN | 0.611 | 0.604 | 0.162 (much worse) | p = 0.06 ✗ |

**Discussion (vulnerability) — reported as a negative result.** On binary detection the framework
does **not** help: accuracy changes are within noise and **not significant** (p = 0.36 / 0.06), and
ECE *degrades*. Mechanistic explanation: with two classes the predictive entropy is near its
maximum for almost every sample, so the entropy gate routes nearly all inputs to $k$NN; in the
768-dim binary embedding space the neighbourhood signal is noisy, so interpolation adds variance
rather than correcting errors. Both our base (62.2 / 61.9) and CodeImprove's base (62.74 / 62.40)
sit well below the Devign accuracy SOTA (~66–67%: CodeT5, CoTexT, CodeT5+) and below the EMNLP'22
retrieval F1 (0.66). **This bounds the method's applicability: multi-class tasks with separable
embeddings; binary sigmoid tasks need a different gating signal (margin or neighbour agreement).**

## 4.3 Distribution-Free Safety under SPT (RQ2)

We perturb the CodeChef test set with semantic-preserving transformations (SPT-0 clean … SPT-3
severe), calibrate Split Conformal Prediction (RAPS, $\alpha=0.05$, $\hat q = 0.962$) **once** on
the clean validation set, and evaluate coverage across shift levels without re-calibration
(CodeBERT/defect).

### Table 3 — Conformal coverage under SPT (CodeBERT/defect)

| SPT | Target | Marginal coverage | Avg set size | Point accuracy |
|:--|:--:|:--:|:--:|:--:|
| SPT-0 (clean) | ≥95% | **98.70%** | 2.79 | 82.5% |
| SPT-1 | ≥95% | **98.45%** | 2.79 | 80.9% |
| SPT-2 | ≥95% | **98.15%** | 2.79 | 79.1% |
| SPT-3 (severe) | ≥95% | **97.81%** | 2.79 | 78.0% |

**Discussion (safety).** Coverage stays above the 95% target across all shift levels even as
deterministic point accuracy falls 82.5%→78.0% — the set-valued guarantee degrades gracefully where
the argmax does not. **Honest caveat:** average set size is 2.79 of 4 classes (~70% of the label
space), so high coverage is relatively easy to achieve; the decisive efficiency metric — set size
versus the base model at equal coverage — is not yet computed and is listed as future work.

## 4.4 Positioning

| Work | Mechanism | Acts on | Detect OOS | Calibration | Formal guarantee | Retrain |
|---|---|---|:--:|:--:|:--:|:--:|
| CodeBERT / GraphCodeBERT (base) | fine-tune | — | ✗ | ✗ (ECE ~0.37) | ✗ | ✓ |
| kNN-for-Vuln (Du et al., EMNLP'22) | contrastive + $k$NN | output | ✗ | ✗ | ✗ | ✓ |
| **CodeImprove (ICSE'25)** | OOS detect + input transform | input | ✓ (AUC 0.924) | ✗ | ✗ | ✗ |
| **CI-Gated $k$NN [ours]** | temp-scale + gated $k$NN + conformal | output | partial (AUC ~0.77) | ✓ (ECE→0.03) | ✓ (cov ≥95%) | ✗ |

Our distinguishing contribution is the pairing of **calibration** with a **distribution-free
coverage guarantee**, applied **post-hoc on the output side without retraining**. This is
**complementary** to CodeImprove, which adapts the *input*; the two could in principle compose. We
do **not** beat CodeImprove on detection (AUC 0.77 vs 0.924) or on accuracy.

## 4.5 Limitations & Future Work

- **RQ3 efficiency unmeasured.** The gate's value is skipping retrieval on confident inputs; guard
  ratio and latency for B1/B4/M1 are not yet logged (`KNNPredictor.predict()` already returns
  `guard_ratio`). This is the highest-value next experiment and needs no retraining.
- **Conformal set tightness vs. baseline** not yet computed (§4.3 caveat).
- **Binary-task gate.** Entropy gating is pathological for sigmoid binary output; a margin-based or
  neighbour-agreement gate is the natural fix.
- **Detection AUC below CodeImprove** (0.77 vs 0.924); our OOS detector is not the contribution.

## References
[1] Rathnasuriya, Zhao, Yang. "CodeImprove: Program Adaptation for Deep Code Models." ICSE 2025. arXiv:2501.15804.
[2] Du, Kuang, Zhao. "Code Vulnerability Detection via Nearest Neighbor Mechanism." Findings of EMNLP 2022.
[3] Zhou et al. "Devign." NeurIPS 2019. (via CodeXGLUE Defect-Detection)
[4] Guo et al. "On Calibration of Modern Neural Networks." ICML 2017. [Temperature scaling]
[5] Khandelwal et al. "Generalization through Memorization: Nearest Neighbor LMs." ICLR 2020.
[6] Angelopoulos & Bates. "A Gentle Introduction to Conformal Prediction." FnT ML 2023.
[7] Angelopoulos et al. "Uncertainty Sets for Image Classifiers using Conformal Prediction." ICLR 2021. [RAPS]
[8] Vovk et al. "Algorithmic Learning in a Random World." Springer 2005.
[9] Lu et al. "CodeXGLUE." NeurIPS Datasets & Benchmarks 2021. [CodeBERT/GraphCodeBERT Devign baselines: 62.1 / 63.2]
