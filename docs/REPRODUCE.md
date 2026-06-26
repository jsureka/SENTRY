# Reproducing SENTRY's Results

Two reproduction paths, both CPU/MPS, no retraining:

- **Fine-tuned anchors** ([`results/VERIFIED_RESULTS.md`](../results/VERIFIED_RESULTS.md)) — the
  rigorous defect/vuln verification, via `kNN-Prediction/{reproduce_results,significance_test}.py`
  (§1–5 below).
- **The full grid** (7 datasets × 4 encoders, [`results/FINAL_VERDICT.md`](../results/FINAL_VERDICT.md))
  — calibration + accuracy + risk–coverage over frozen embeddings, via `analysis/` (§6 below).

The anchor scripts:

- `reproduce_results.py` — baselines + the framework (gated-*k*NN), per combo
- `significance_test.py` — paired McNemar significance (continuity-corrected χ² + exact binomial)

## 1. Environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "torch" "transformers>=4.20,<5" "faiss-cpu" "scikit-learn" "scipy" "numpy<2" "tqdm" "matplotlib"
```

macOS only: torch and faiss both link libomp. Export these (the scripts also set them):

```bash
export KMP_DUPLICATE_LIB_OK=TRUE      # allow the duplicate libomp
export OMP_NUM_THREADS=4
```

## 2. Inputs (gitignored — supply locally)

| What | Path | Notes |
|---|---|---|
| 4 model checkpoints | `models/{codebert,graphcodebert}_{defect,vuln}_model.bin` | ~500M each; fine-tuned CodeBERT/GraphCodeBERT (CodeImprove splits). Gitignored. |
| Datasets | `Defect-Prediction/dataset/`, `Vulnerability-Detection/dataset/` | `train/dev(valid)/test.jsonl`. Tracked. |
| Datastores | `results/CI_kNN_<task>_results/datastore/full/` | FAISS index over train embeddings. Gitignored; rebuildable (see `--rebuild_datastore`). |

Checkpoint state dicts use the CodeImprove `encoder.` prefix; the loader strips it and loads
strict (`missing=0/unexpected=0`).

## 3. Run

```bash
cd kNN-Prediction
# tasks: vuln_codebert | defect_codebert | defect_graphcodebert | vuln_graphcodebert
python reproduce_results.py --task defect_codebert
python significance_test.py --task defect_codebert
```

First run extracts dev+test embeddings (CPU, ~3–7 min/split) and caches them to
`out_<task>_<split>.npz`; reruns are instant. Per-combo metrics are written to
`results/CI_kNN_<task>_results/verified_metrics.json`.

### GraphCodeBERT note — rebuild the datastore
The shipped GraphCodeBERT datastores were built in a slightly different embedding space
(block-size/config), which makes a plain re-extraction's queries mismatch the store. For a
faithful kNN test, rebuild it from this run's own train extraction (adds one ~40-min train pass):

```bash
python reproduce_results.py --task defect_graphcodebert --rebuild_datastore
```

## 4. Validation gate (proves the checkpoint loaded correctly)

`reproduce_results.py` prints a gate comparing **B1 accuracy** (config-independent) to the saved
table. Accuracy must match; ECE is *expected to differ* — that is the point (the originally
recorded ECE column was wrong; see [`results/VERIFIED_RESULTS.md`](../results/VERIFIED_RESULTS.md) §1).

| Combo | B1 acc (expected) | B1 ECE (real) |
|---|---|---|
| vuln_codebert | 0.612 | 0.197 |
| defect_codebert | 0.818 | 0.082 |
| defect_graphcodebert | 0.806 | 0.069 |
| vuln_graphcodebert | 0.609 | 0.228 |

## 5. Expected headline numbers

| Combo | base acc → SENTRY acc | base ECE → SENTRY ECE | McNemar (base→SENTRY) |
|---|---|---|---|
| defect_codebert | 0.818 → 0.831 | 0.082 → 0.017 | p = 4e-6 |
| defect_graphcodebert | 0.806 → 0.835 | 0.069 → 0.008 | p = 2e-19 |
| vuln_codebert | 0.612 → 0.612 (temp-only) | 0.197 → 0.055 | kNN harms (p≈0.05), use temp-only |
| vuln_graphcodebert | 0.609 → 0.609 (temp-only) | 0.228 → 0.061 | kNN harms, use temp-only |

Defect: the gated-kNN engages and improves accuracy + calibration (significant). Vuln: the
representation does not separate, so kNN is skipped and temperature scaling alone fixes
calibration while preserving accuracy.

## 6. The full grid (7 datasets × 4 encoders)

The reliability metrics across the grid run over **frozen** embeddings — no fine-tuned checkpoints
needed. First build the embedding caches (one GPU/MPS pass, then 0-credit reruns), then evaluate:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
cd kNN-Prediction
python run_grid.py        # frozen-embedding caches: 6 datasets × 4 encoders -> grid_emb/*.npz
python embed_clone.py     # BigCloneBench pair embeddings -> grid_emb/*__bigclone.npz

cd ..
python analysis/methods_table.py   # calibration + accuracy + risk-coverage per method -> results/methods_grid.csv
python analysis/clone_point.py     # the binary-but-separable control
```

Datasets load from Hugging Face (Devign/CodeChef local; POJ-104, ReVeal, PrimeVul, DiverseVul,
BigCloneBench via `datasets`). `grid_emb/` caches are git-ignored and rebuilt by the two scripts
above. Expected headline: retrieval/gating improve accuracy on separable tasks (multiclass +3.1pp,
clone +2.5pp) and are retrieval-neutral on vulnerability; temperature scaling reduces ECE on every
task. See [`results/FINAL_VERDICT.md`](../results/FINAL_VERDICT.md).
