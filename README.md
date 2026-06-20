# SENTRY

**S**elective **EN**tropy-gated re**TR**ieval — a training-free reliability layer for code classifiers.

![status](https://img.shields.io/badge/status-research-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

SENTRY wraps a fine-tuned code classifier (CodeBERT / GraphCodeBERT) with a post-hoc,
**no-retraining** reliability layer. It is the framework; the underlying method is **CI-Gated kNN**
(Confidence-Informed Gated *k*NN). Masters-thesis work building on and **complementary to**
**CodeImprove** (Rathnasuriya et al., ICSE 2025, [arXiv:2501.15804](https://arxiv.org/abs/2501.15804)):
CodeImprove adapts the *input* (detect out-of-scope code, transform it back in-scope); SENTRY wraps
the *output*.

## What it does

At inference time only, for any fine-tuned classifier:

1. **Temperature scaling** — fixes severe overconfidence (ECE ≈ 0.37 → ≈ 0.04).
2. **Confidence-gated kNN retrieval** — a FAISS datastore over training embeddings; an
   entropy-adaptive sigmoid gate routes only *uncertain* queries to kNN and interpolates.
3. **Split conformal prediction (RAPS)** — distribution-free prediction sets whose coverage
   guarantee holds under semantic-preserving transformations (SPTs).

## Results

- **Defect prediction (CodeChef, 4-class) — works.** kNN gives a small but **statistically
  significant** accuracy gain over the base model (McNemar p < 1e-12 CodeBERT, p < 1e-7
  GraphCodeBERT); calibration handled by temperature scaling; conformal coverage stays ≥ 97.8%
  under perturbation. Accuracy is **on par with CodeImprove's base model** (82.1% vs 81.9%,
  GraphCodeBERT).
- **Vulnerability detection (Devign, binary) — negative result.** Flat-to-harmful, not significant
  (p = 0.36 / 0.06), ECE degrades. Cause: binary softmax → near-maximal entropy → the gate
  over-triggers. Bounds the method to multi-class tasks with separable embeddings.
- **Positioning.** Output-side and complementary to CodeImprove; contributes calibration + a formal
  guarantee. **Not** an accuracy-SOTA claim (Devign SOTA ≈ 66–67%; base ≈ 62%) and does **not** beat
  CodeImprove's detector (AUC ≈ 0.77 vs 0.924).

Full numbers & discussion: [`writeup/evaluation_draft.md`](writeup/evaluation_draft.md) ·
raw metrics: [`writeup/Aggregated_Raw_Results.md`](writeup/Aggregated_Raw_Results.md) ·
talk cheat-sheet: [`PRESENTATION_PLAN.md`](PRESENTATION_PLAN.md).

## Repository layout

| Path | Contents |
|---|---|
| `kNN-Prediction/` | **Core pipeline**: datastore, predictor, calibration, conformal, SPT simulator, OOS metrics, `run_knn.py`, master notebook. See `writeup/CI_Gated_kNN_Files_Summary.md`. |
| `Defect-Prediction/`, `Vulnerability-Detection/` | Per-task `dataset/` + fine-tuning `code/` (`run.py`, `model.py`, train/test scripts) + CodeXGLUE `evaluator/`. |
| `results/` | Per-(task × model) outputs: final tables, ablations, OOD, conformal, significance, plots. (FAISS datastores are git-ignored.) |
| `models/` | Fine-tuned weights `*.bin` (git-ignored; ~965M, kept locally). |
| `writeup/` | Thesis drafts: methodology, related work, evaluation, raw results, analysis. |
| `aggregate.py` | Collates `results/*/` JSON → `writeup/Aggregated_Raw_Results.md`. |
| `Images/` | Figures. |

## Setup & reproduce

```bash
pip install -r kNN-Prediction/requirements.txt
python aggregate.py        # rebuild writeup/Aggregated_Raw_Results.md from results/
```

The evaluation (calibration / kNN interpolation / conformal) re-runs from the per-experiment JSON in
`results/` without a GPU. Rebuilding the **FAISS datastores** or the **model weights** (both
git-ignored to keep the repo lean) requires the training data and a GPU — see the master notebook
`kNN-Prediction/CI_Gated_kNN_Master.ipynb` (Colab/Drive oriented).

## Attribution

SENTRY's own code (`kNN-Prediction/`, `aggregate.py`, `writeup/`) is MIT-licensed. Datasets
(**Devign**, **CodeChef**) and base task code under `Defect-Prediction/` and
`Vulnerability-Detection/` derive from **CodeXGLUE** and **CodeImprove** and remain under their
respective upstream licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite CodeImprove (arXiv:2501.15804).
