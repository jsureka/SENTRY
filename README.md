# SENTRY

**A training-free reliability framework for code classifiers.**

![status](https://img.shields.io/badge/status-research-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

SENTRY wraps a **frozen** code classifier (CodeBERT, GraphCodeBERT, UniXcoder, CodeT5+) with three
post-hoc, **no-retraining** components — temperature calibration, reliability-gated *k*NN retrieval,
and selective abstention — so deployed defect / vulnerability detectors are well-calibrated and, where
retrieval is reliable, more accurate. Masters-thesis work, **complementary to** CodeImprove
(Rathnasuriya et al., ICSE 2025, [arXiv:2501.15804](https://arxiv.org/abs/2501.15804)): CodeImprove
adapts the *input*; SENTRY wraps the *output*.

## What SENTRY does

We first measure that deployed code classifiers are badly over-confident (e.g. CodeBERT on
vulnerability: 0.81 mean confidence at 0.61 accuracy). Calibration itself is a *known* fix
(temperature scaling, Guo 2017; for code models, Zhou et al. ICSE 2024) and SENTRY applies it —
that part is table stakes, not our contribution. What SENTRY adds on top:

- **Accuracy *and* calibration together, where the representation separates classes — which
  calibration alone cannot do.** Calibration is accuracy-neutral by construction; SENTRY's
  reliability-gated retrieval also raises accuracy. **Defect prediction:** 0.818 → **0.831**
  (CodeBERT, McNemar p = 4e-6) / 0.806 → **0.835** (GraphCodeBERT, p = 2e-19), ECE 0.082 → **0.017**.
  Across separable tasks: **+3.1pp** accuracy (multiclass), **+2.5pp** (clone), with improved
  risk–coverage (AURC).
- **A characterisation of *when* it works — separability** (below): the contribution that ties the
  results together, with binary clone detection as the control.
- **Selective abstention.** The gate's reliability signal (neighbour distance + vote agreement) drives
  abstention; risk–coverage improves on separable tasks.
- **Never-harm by design.** Where retrieval is unreliable the gate falls back to the calibrated model,
  so accuracy is preserved (vulnerability: retrieval-neutral, calibration still fixed).

For reference, calibration coverage: ECE drops 3–5× on every task (multiclass 0.13 → 0.03;
vulnerability 0.20 → 0.06) — expected from temperature scaling, reported here because the SE task
papers that build these detectors do not.

## When it works — the separability scope

Retrieval engages exactly when the task is **separable** in the frozen embedding — independent of
binary vs. multiclass:

| Task family | separable | base acc | retrieval effect |
|---|---|---|---|
| Multiclass (CodeChef defect, POJ-104) | yes | 0.61–0.92 | accuracy **+3.1pp**, calibration ↑ |
| Binary clone (BigCloneBench) | yes | 0.84 | accuracy **+2.5pp** |
| Binary vuln (Devign, ReVeal, PrimeVul, DiverseVul) | no | 0.94–0.97\* | retrieval-neutral → calibration only |

\*vulnerability accuracy is majority collapse (base F1-macro ≈ 0.49). **Clone detection is the
control:** it is *binary* yet *separable*, and retrieval helps it like multiclass — so the
determining factor is **separability, not binary-ness**. The gate's reliability signal selects which
regime a task is in.

## Limitations & future work

- Retrieval cannot help non-separable tasks (binary vulnerability = a representability ceiling, cf.
  Ding et al., ICSE 2024); there SENTRY contributes calibration and abstention only.
- Separability is characterised post-hoc, not predicted before deployment.
- **Future:** auto-gate from the separability signal; compose retrieval with uncertainty estimators
  (e.g. kNN-UE, Hashimoto et al., NAACL 2025); broaden tasks and languages.

## Datasets & models

- **7 datasets** — CodeChef (defect, 4-class), POJ-104 (104-class), BigCloneBench (clone), Devign,
  ReVeal, PrimeVul (ICSE'24), DiverseVul (vulnerability).
- **4 encoders** — CodeBERT, GraphCodeBERT, UniXcoder, CodeT5+ (frozen probes), plus **4 fine-tuned
  anchors** (CodeBERT / GraphCodeBERT × defect / vulnerability). 7 × 4 = 32 evaluation points.

## Repository layout

| Path | Contents |
|---|---|
| `kNN-Prediction/` | **Core framework**: `knn_datastore.py`, `knn_predictor.py`, `calibration.py`, `oos_metrics.py`, `separability.py`; grid drivers `run_grid.py`, `methods_grid.py`, `embed_clone.py`. **Verification harness**: `reproduce_results.py`, `significance_test.py`. Frozen-embedding caches in `grid_emb/` (git-ignored, regenerable). |
| `analysis/` | Evaluation scripts (read cached embeddings, 0-credit): `methods_table.py` (reliability metrics across the grid), `clone_point.py` (BigCloneBench separability control); plus probes `reliability_headtohead.py`, `knn_ue_headtohead.py`, `cross_encoder.py`, `transform_probe.py`, `ood_calib.py`. |
| `Defect-Prediction/`, `Vulnerability-Detection/` | Per-task `dataset/` + fine-tuning `code/` + CodeXGLUE `evaluator/`. |
| `results/` | `methods_grid.csv`, `methods_summary.json`, per-(task×model) fine-tuned outputs, significance, plots; **[`FINAL_VERDICT.md`](results/FINAL_VERDICT.md)** + **[`VERIFIED_RESULTS.md`](results/VERIFIED_RESULTS.md)**. FAISS datastores git-ignored. |
| `paper/` | Paper sections (`00`–`06`, `references.md`) + `figures/`. **In progress** — do not cite as final. |
| `docs/` | **[`REPRODUCE.md`](docs/REPRODUCE.md)** (run instructions), **[`RELATED_WORK.md`](docs/RELATED_WORK.md)** (positioning). |
| `models/` | Fine-tuned weights `*.bin` (git-ignored; 4×476 MB). See **Models & data** below. |

## Models & data

Fine-tuned checkpoints (4×476 MB) and FAISS datastores are git-ignored (regenerable). For exact
reproduction without retraining, download the bundle:

<!-- DRIVE_LINK: paste the shared Google Drive folder URL here -->
**Models + datastores (Google Drive):** _link pending upload — see `drive_bundle/MANIFEST.md`._

Or regenerate from the per-task `code/` training scripts; frozen-embedding grid caches rebuild via
`python kNN-Prediction/run_grid.py` + `python kNN-Prediction/embed_clone.py`.

## Setup & reproduce

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r kNN-Prediction/requirements.txt
export KMP_DUPLICATE_LIB_OK=TRUE          # macOS libomp

# reliability metrics across the grid (0-credit, cached embeddings):
python analysis/methods_table.py          # calibration + accuracy + risk-coverage per method
python analysis/clone_point.py            # BigCloneBench separability control

# fine-tuned defect/vuln anchors (CPU, no retraining):
cd kNN-Prediction
python reproduce_results.py --task defect_codebert
python significance_test.py --task defect_codebert
```

Full instructions: [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Attribution

SENTRY's own code (`kNN-Prediction/`, `analysis/`) is MIT-licensed. Datasets (Devign, CodeChef,
POJ-104, BigCloneBench, ReVeal, PrimeVul, DiverseVul) and base task code derive from CodeXGLUE,
CodeImprove, and their respective sources under upstream licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite CodeImprove (arXiv:2501.15804).
