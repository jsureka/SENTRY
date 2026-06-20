# CI-Gated kNN Pipeline Files Overview

This document summarizes the purpose and usage of all the peripheral Python scripts that the `CI_Gated_kNN_Master.ipynb` notebook leverages to construct the full pipeline for Confidence-Informed (CI) Gated kNN. It serves as a centralized knowledge base for AI tooling and project architecture.

## 1. Core kNN Architecture (`kNN-Prediction/`)

### `knn_datastore.py`
* **Purpose**: Constructs and manages a high-performance FAISS-based embedding datastore strictly from the training corpus.
* **Usage in Pipeline**: Operates throughout **Phase 3**. It converts training instances into high-dimensional hidden-state vectors using the fine-tuned base models (CodeBERT/GraphCodeBERT) and stores them in FAISS indexes for instantaneous similarity searches at inference time. The Master Notebook dynamically patches parsing logic inside this file to correctly interpret Devign vs. Defect dataset keys (e.g., bridging `func` vs `input` JSON field disparities) before execution.

### `knn_predictor.py`
* **Purpose**: Performs the inference phase by routing, evaluating, and mixing underlying model predictions with retrieval-augmented neighbor labels. 
* **Usage in Pipeline**: Called in **Phase 4** to execute the CI-Gated kNN and Selective prediction mechanism. It interpolates softmax outputs from the base neural network with a distance-weighted neighborhood vote. Handles uncertainty gating parameters (λ) identifying when to fallback on raw model features vs when to trust neighbor proximity.

### `conformal_prediction.py`
* **Purpose**: Wraps the classifier outputs using the Regularized Adaptive Prediction Sets (RAPS) conformal prediction framework.
* **Usage in Pipeline**: Used in **Phase 5** to map point-estimate probabilities to statistically guaranteed sets. It guarantees a bounded, distribution-free coverage of the true labels (i.e. $P(y \in C(X)) \ge 1-\alpha$). Used to prove the reliability of the kNN predictions under intense distributional shifts.

### `calibration.py`
* **Purpose**: Implements post-hoc probabilities calibration on multiclass classifiers.
* **Usage in Pipeline**: Acts as an intermediary smoothing utility. Specifically, it applies localized Temperature Scaling optimizations on raw logits against checking parameters like Brier Scores and Expected Calibration Errors (ECE) preventing underconfidence/overconfidence prior to kNN integration.

### `oos_metrics.py`
* **Purpose**: Formulates the primary Out-of-Scope (OOS) testing metrics and novel probabilistic score extensions.
* **Usage in Pipeline**: Evaluated heavily in **Phase 4.5**. Computes traditional CodeImprove thresholds including Coverage Ratio (CVR), AUC, and Misclassification Verification Rate (MVR). Extends natively into cutting edge OOD heuristic calculations such as Relative Mahalanobis Distance and Energy Scores.

### `spt_simulator.py`
* **Purpose**: A pure-Python implementation acting as a Semantic-Preserving Transformation (SPT) engine.
* **Usage in Pipeline**: Executed in **Phase 1.5**. Replicates CodeImprove's older robust structural heuristics (like TXL-based insertions of dead code or renaming abstract variables) without needing pre-compiled TXL binary dependencies safely porting OOD perturbation dataset generation to Colab environments locally.

---

## 2. Base CodeImprove Integrations (`code/` directories)

### `run.py`
* **Purpose**: The primary fine-tuning, training, and base-model evaluation script heavily adopted from HuggingFace/CodeXGLUE specifications.
* **Usage in Pipeline**: Invoked via command line subprocesses during **Phase 1**. It loads up underlying causal/masked configurations, trains them across epochs, and yields the initial base `model.bin`. Like `knn_datastore`, the Notebook patches standard mappings directly within this file locally.

### `rundissector.py` & `emsemble.py`
* **Purpose**: Legacy internal tracking scripts provided by the baseline CodeImprove ecosystem covering internal structural representation evaluations (PVScore analysis) and manual aggregation metrics, respectively.
* **Usage in Pipeline**: Discovered and processed during **Phase 0**. Because the overarching environment leverages recent `transformers` releases (≥4.32), the notebook targets these files with automated Regex patches migrating out-dated `AdamW` optimizer imports back to PyTorch's native `torch.optim` to ensure execution safety for subsequent fallback operations.
