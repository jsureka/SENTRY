# 3. Methodology

We present **Confidence-Informed Gated kNN (CI-Gated kNN)**, a post-hoc reliability layer that wraps any fine-tuned transformer classifier for code intelligence tasks without modifying its parameters. The framework executes in five sequential phases: (1) base model fine-tuning and semantic extraction, (2) post-hoc temperature calibration, (3) FAISS datastore construction, (4) confidence-guarded, entropy-adaptive kNN interpolation, and (5) conformal prediction for distribution-free set-valued guarantees. We also define a synthetic out-of-distribution (OOD) evaluation protocol via Semantic-Preserving Transformations (SPTs). Figure 1 illustrates the complete inference-time flow.

---

## 3.1 Base Model Fine-Tuning and Semantic Extraction

### 3.1.1 Task Setup

Let $\mathcal{D}_{train} = \{(x_i, y_i)\}_{i=1}^{N}$ be the training corpus where $x_i$ is a code snippet and $y_i \in \mathcal{Y}$ is its label. For defect prediction (CodeChef), $\mathcal{Y} = \{0,1,2,3\}$ corresponding to four execution outcomes (Runtime Error, Wrong Answer, Accepted, Time Limit Exceeded). For vulnerability detection (Devign), $\mathcal{Y} = \{0,1\}$ (not vulnerable, vulnerable).

We fine-tune one of two transformer encoders: **CodeBERT** (125M parameters, RoBERTa backbone pre-trained on code–documentation pairs) or **GraphCodeBERT** (125M parameters, additionally pre-trained with data-flow graphs). Both architectures produce a sequence of hidden states; we extract the final-layer representation of the `[CLS]` token, $h(x) \in \mathbb{R}^{768}$, as the semantic embedding.

**Implementation** (`Defect-Prediction/code/run.py`, `Vulnerability-Detection/code/run.py`):
- Fine-tuning uses `AdamW` with learning rate $2 \times 10^{-5}$, max sequence length 400 tokens, and `CrossEntropyLoss` for defect (4-class) or binary cross-entropy with `sigmoid` for vulnerability.
- The fine-tuned weights are persisted as `saved_models/checkpoint-best-acc/model.bin`, then loaded in inference via `Model.load_state_dict()` (`Defect-Prediction/code/model.py`, `Vulnerability-Detection/code/model.py`).

### 3.1.2 Embedding Extraction Strategy

**Implementation** (`kNN-Prediction/knn_datastore.py` → `class EmbeddingExtractor`):

The `EmbeddingExtractor` class calls `model.encoder(input_ids, output_hidden_states=True)` to obtain the full hidden-state tuple and supports four extraction strategies:
- `last_cls` — final layer, `[CLS]` token: $h = \text{hidden\_states}[-1][:,0,:]$
- `second_last_cls` — penultimate layer, `[CLS]` token
- `avg_last4_cls` — mean of the last four layers, `[CLS]` token
- `mean_pool_last` — attention-masked mean-pool over all tokens in the last layer

The default strategy used throughout all experiments is `last_cls`. All embeddings are L2-normalized before FAISS indexing (`faiss.normalize_L2()`), converting L2 search to cosine-equivalent retrieval.

---

## 3.2 Post-Hoc Temperature Calibration

### 3.2.1 Motivation

Raw softmax probabilities from fine-tuned code transformers are systematically overconfident, exhibiting Expected Calibration Error (ECE) values of up to 0.37 (observed on CodeBERT/defect baseline B1). To use confidence as a reliable gate trigger, probabilities must first be calibrated.

### 3.2.2 Temperature Scaling

**Implementation** (`kNN-Prediction/calibration.py` → `class TemperatureScaler`):

Using the held-out validation set $\mathcal{D}_{val}$, we optimize a single scalar $T > 0$ by minimizing the Negative Log-Likelihood (NLL) via `scipy.optimize.minimize` (Nelder-Mead):

$$\mathcal{L}_{NLL}(T) = -\frac{1}{|\mathcal{D}_{val}|} \sum_{i} \log \hat{p}_{T}(y_i \mid x_i), \quad \hat{p}_{T}(y=c \mid x) = \frac{\exp(z_c / T)}{\sum_{j} \exp(z_j / T)}$$

The fitted $T$ is saved to `temperature.npy` via `TemperatureScaler.save()` and loaded on subsequent runs via `TemperatureScaler.load()`. Because $T$ is applied uniformly to all logits, the argmax prediction (and therefore accuracy) is unchanged — only the probability mass distribution is adjusted.

**Calibrated probabilities** (`TemperatureScaler.calibrate(logits)`) are the direct input to all downstream components. The raw logits are preserved separately for FAISS embedding extraction (which bypasses the calibration head).

---

## 3.3 FAISS Datastore Construction

**Implementation** (`kNN-Prediction/knn_datastore.py` → `class KNNDatastore`):

Prior to test-time inference, we perform a single forward pass over $\mathcal{D}_{train}$ to construct a key–value datastore:

$$(\mathcal{K}, \mathcal{V}) = \{(h(x_i),\, y_i) \mid x_i \in \mathcal{D}_{train}\}$$

The `KNNDatastore.build()` method:
1. Iterates over training batches via `DataLoader` + `SequentialSampler`
2. Calls `EmbeddingExtractor.extract(input_ids)` per batch
3. Concatenates all embeddings into a flat `(N, 768)` float32 matrix
4. L2-normalizes the matrix: `faiss.normalize_L2(all_embeddings)`
5. Adds to a FAISS exact-search index: `faiss.IndexFlatL2(768)`

The index, label array (`labels.npy`), and ID array (`ids.npy`) are persisted via `KNNDatastore.save(output_dir)` and restored via `KNNDatastore.load(output_dir)`, avoiding full re-embedding on repeat runs.

**Variant: Prototype Datastore** (`kNN-Prediction/knn_datastore.py` → `class ProtokNNDatastore`):

As a novel efficiency variant, we introduce a centroid-based datastore that replaces all $N$ training embeddings with $K \cdot C$ prototypes (default $K=50$ centroids per class), computed via `MiniBatchKMeans`. This reduces datastore size by ~200× while improving calibration smoothness by averaging out label noise. The prototype store uses the same `search()` interface as the full datastore.

---

## 3.4 Confidence-Guarded, Entropy-Adaptive kNN Prediction

**Implementation** (`kNN-Prediction/knn_predictor.py` → `class KNNPredictor`):

This is the core contribution. The `KNNPredictor` class implements a two-level decision mechanism: a **confidence guard** that prevents kNN from destabilizing already-reliable predictions, and an **entropy gate** that dynamically controls how much weight is given to kNN retrieval.

### 3.4.1 Confidence Guard (Level 1 — Hard Bypass)

For each test sample $x$, we first compute the maximum calibrated confidence:

$$\kappa(x) = \max_{c \in \mathcal{Y}} \hat{p}_T(y=c \mid x)$$

If $\kappa(x) > \tau$ (where $\tau$ is the confidence guard threshold selected via validation-set grid search over $\{0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90\}$), the sample is **guarded**: kNN retrieval is bypassed entirely and $\lambda$ is hard-set to $1.0$.

```python
# knn_predictor.py, KNNPredictor.predict(), lines 139–142
if self.confidence_guard_threshold is not None:
    guard_mask = model_probs.max(axis=1) > self.confidence_guard_threshold
else:
    guard_mask = np.zeros(len(model_probs), dtype=bool)
```

This guard is critical for the binary vulnerability task (Devign): without it, binary sigmoid logits near 0.5 produce near-maximal entropy for all samples, causing the entropy gate to always route to kNN, which introduces noise and degrades ECE from 0.036 to 0.162.

### 3.4.2 kNN Retrieval

For non-guarded samples, we retrieve $k$ nearest neighbors from the FAISS datastore:

$$(\mathbf{d}_i, \mathbf{y}_i^{kNN}) = \text{FAISS.search}(h(x), k), \quad \mathbf{d}_i \in \mathbb{R}^k,\; \mathbf{y}_i^{kNN} \in \mathcal{Y}^k$$

**Implementation** (`KNNDatastore.search(query_embeddings, k)`): queries are L2-normalized before search, then `faiss.IndexFlatL2.search()` returns L2 distances and neighbor indices. Labels are looked up by index from `self.labels`.

### 3.4.3 Distance-Weighted kNN Probability (`compute_knn_probs`)

Three voting strategies are available; the default is `distance_weighted`:

$$p_{kNN}(y=c \mid x) = \frac{\sum_{j=1}^{k} w_j \cdot \mathbb{I}(y_j^{kNN}=c)}{\sum_{j=1}^{k} w_j}, \quad w_j = \frac{\exp(-d_j / t)}{\sum_{l=1}^{k} \exp(-d_l / t)}$$

where $t$ (`knn_temperature`, default 10.0) controls the sharpness of the distance weighting. Implemented via `scipy.special.softmax(-distances / self.knn_temperature, axis=1)`.

The alternative `threshold_filtered` strategy restricts voting to neighbors within the median distance, further suppressing outlier neighbors.

### 3.4.4 Entropy-Adaptive Sigmoid Gate (Level 2 — Soft Interpolation)

For samples not bypassed by the confidence guard, we compute predictive entropy of the calibrated probabilities:

$$U(x) = -\sum_{c=1}^{C} \hat{p}_T(y=c \mid x) \log \hat{p}_T(y=c \mid x)$$

**Implementation** (`kNN-Prediction/calibration.py` → `compute_entropy(probs)`): computes $-\sum p \log p$ with clipping to $[10^{-10}, 1.0]$.

The entropy is mapped through an inverted sigmoid to produce the per-sample model weight $\lambda(x) \in (0, 1)$:

$$\lambda(x) = 1 - \sigma\bigl(a \cdot U(x) + b\bigr) = 1 - \frac{1}{1 + \exp\bigl(-(a \cdot U(x) + b)\bigr)}$$

where $a=1.0$ and $b=-0.5$ (fixed, not tuned). High entropy → large $\sigma$ output → low $\lambda$ → more kNN weight. Low entropy → high $\lambda$ → model dominates.

```python
# knn_predictor.py, KNNPredictor.predict(), lines 154–158
lambda_per_sample = 1.0 / (1.0 + np.exp(-(gate_a * calibrated_entropy + gate_b)))
lambda_per_sample = 1.0 - lambda_per_sample   # invert: high entropy → lower λ
lambda_per_sample = lambda_per_sample.reshape(-1, 1)
```

The confidence guard then overrides $\lambda$ to 1.0 for guarded samples:

```python
# lines 163–165
if self.confidence_guard_threshold is not None:
    lambda_per_sample = lambda_per_sample.copy()
    lambda_per_sample[guard_mask] = 1.0
```

### 3.4.5 Final Interpolation

The final probability distribution is the convex combination:

$$p_{final}(y \mid x) = \lambda(x) \cdot \hat{p}_T(y \mid x) + (1 - \lambda(x)) \cdot p_{kNN}(y \mid x)$$

followed by row-normalization. The argmax of $p_{final}$ yields the point prediction. The `predict()` method returns `(final_probs, predictions, knn_probs, details)`, where `details` includes `lambda_per_sample`, `n_guarded`, and `guard_ratio`.

---

## 3.5 Evaluation Protocol: Methods and Baselines

**Implementation** (`kNN-Prediction/knn_evaluate.py` → `class KNNEvaluator`, `KNNEvaluator.run_evaluation()`):

We evaluate four configurations under a unified `compute_metrics()` function that computes Accuracy, F1-Macro, F1-Weighted, MCC, ECE (15 bins), Brier Score, mean entropy, mean confidence, and the selective risk curve (`compute_selective_risk` from `calibration.py`):

| Config | Label | Description |
|--------|-------|-------------|
| `KNNPredictor.predict_model_only()` | **B1** | Base model only, no adaptation |
| `TemperatureScaler.calibrate()` → argmax | **B3** | Temperature scaling only |
| `KNNPredictor.predict(uncertainty_gated=False)` | **B4** | kNN interpolation, fixed $\lambda$, no gating |
| `KNNPredictor.predict(uncertainty_gated=True)` | **M1** | Full CI-Gated kNN (ours) |

Results are saved to `evaluation_results.json`, and calibration reliability diagrams and selective risk curves are generated automatically via `_plot_calibration()` and `_plot_selective_risk()`.

---

## 3.6 OOD Robustness via Semantic-Preserving Transformations

**Implementation** (`kNN-Prediction/spt_simulator.py`):

To evaluate robustness under covariate shift, we synthetically perturb the test set using **Semantic-Preserving Transformations (SPTs)**, a pure-Python simulation of the TXL-based transformations used in CodeImprove. Seven transformation types are implemented:

| SPT | Function | Effect |
|-----|----------|--------|
| SPT-1 | `spt_rename_vars()` | Renames local variables to `var_0, var_1, ...` |
| SPT-2 | `spt_insert_dead_code()` | Inserts unreachable `if(0){...}` block |
| SPT-3 | `spt_add_noop()` | Inserts `(void)0;` no-ops every 3rd semicolon |
| SPT-4 | `spt_switch_if_else()` | Swaps if/else with negated condition |
| SPT-5 | `spt_add_redundant_cast()` | Adds `(int)(...)` casts around literals |
| SPT-6 | `spt_wrap_compound()` | Wraps bare `if`-body in braces |
| SPT-7 | `spt_remove_comments()` | Strips `//` and `/* */` comments |

The `generate_spt_shift_levels(input_path, output_dir, max_n=3)` function generates four test variants: `test_0_spt.jsonl` (original) through `test_3_spt.jsonl` (3 random SPTs applied per sample). SPT selection per sample uses a per-sample seed (`seed + i`) for reproducibility. All SPTs preserve the true label, constituting genuine covariate shift rather than label shift.

**Extended OOD Detection** (`kNN-Prediction/oos_metrics.py`): We additionally compute class-conditional Mahalanobis distance (`compute_mahalanobis_ood_scores`), energy score (`compute_energy_ood_scores`), and Relative Mahalanobis Distance (`compute_relative_mahalanobis_ood_scores`) as post-hoc OOD detectors, benchmarked against the entropy baseline.

---

## 3.7 Distribution-Free Safety via Conformal Prediction

**Implementation** (`kNN-Prediction/conformal_prediction.py` → `class KNNConformalPredictor`):

To provide formal safety guarantees beyond point predictions, we apply **Split Conformal Prediction** to the output of M1. The calibration set $\mathcal{D}_{cal}$ (the validation split) is used to compute a nonconformity threshold $\hat{q}$, after which the test set is never revisited during calibration.

### 3.7.1 Nonconformity Score Functions

We implement two scoring methods:

**LAC** (`_lac_scores`): $s(x, y) = 1 - p_{final}(y \mid x)$. Higher score = more nonconforming.

**RAPS** (`_raps_scores`, default): regularized cumulative score that sums softmax probabilities in descending order until the true class is included, plus a regularization penalty $\lambda_{reg} \cdot \max(0, \text{rank} - k_{reg})$ for each class beyond rank $k_{reg}$ (defaults: $\lambda_{reg}=0.1$, $k_{reg}=2$).

### 3.7.2 Calibration and Threshold Computation

**Implementation** (`KNNConformalPredictor.calibrate(calib_probs, calib_labels)`):

$$\hat{q} = \text{Quantile}\!\left(\{s(x_i, y_i)\}_{i=1}^{n},\; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

The finite-sample corrected quantile level ensures the coverage guarantee holds exactly for any $n$ (Vovk et al., 2005). The calibration state (including $\hat{q}$, $\alpha$, method, $\lambda_{reg}$, $k_{reg}$) is persisted via `KNNConformalPredictor.save()` to `conformal_calibration.json`.

### 3.7.3 Prediction Sets and Coverage Guarantee

**Implementation** (`KNNConformalPredictor.predict_set(test_probs)`):

For RAPS, the prediction set $\mathcal{C}(x)$ is constructed by including classes in descending probability order until the cumulative regularized score exceeds $\hat{q}$. This provides the marginal coverage guarantee:

$$\mathbb{P}\!\left(y^* \in \mathcal{C}(x)\right) \geq 1 - \alpha$$

with no distributional assumptions beyond exchangeability (satisfied by i.i.d. sampling).

### 3.7.4 SPT Evaluation

**Implementation** (`run_conformal_spt_evaluation(calib_probs, calib_labels, spt_probs_dict, spt_labels_dict, alpha, method)`):

The CP threshold $\hat{q}$ is calibrated once on the clean validation set, then evaluated across all four SPT shift levels without re-calibration. This tests whether the formal coverage guarantee degrades under distribution shift. Metrics reported per SPT level: `marginal_coverage`, `avg_set_size`, `singleton_rate`, `empty_rate`, `point_accuracy`.

---

## 3.8 Hyperparameter Selection Protocol

All hyperparameters are selected on the validation set $\mathcal{D}_{val}$ before any test-set evaluation. The protocol mirrors the CI-Gated kNN notebook (Phase 3 ablation grid):

| Hyperparameter | Search Range | Selection Criterion |
|----------------|-------------|---------------------|
| $k$ (neighbors) | $\{2, 4, 8, 16, 32\}$ | Val F1-Macro |
| $\lambda$ (base weight) | $\{0.1, 0.3, 0.5, 0.7, 0.9\}$ | Val F1-Macro |
| $\tau$ (confidence guard) | $\{$None$, 0.55, \ldots, 0.90\}$ | Val F1-Macro |
| $\alpha$ (CP coverage) | $0.05, 0.10$ | Fixed by user requirement |
| Embedding strategy | `last_cls` (fixed) | — |
| Gate parameters $a, b$ | $1.0, -0.5$ (fixed) | — |

The confidence guard threshold $\tau$ is treated identically to $k$ and $\lambda$: selected once on val, frozen before any test-set inference. No test labels are observed during hyperparameter search.

---

## References

[1] Feng, Z. et al. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages." EMNLP (2020).  
[2] Guo, D. et al. "GraphCodeBERT: Pre-training Code Representations with Data Flow." ICLR (2021).  
[3] Guo, C. et al. "On Calibration of Modern Neural Networks." ICML (2017).  
[4] Khandelwal, U. et al. "Generalization through Memorization: Nearest Neighbor Language Models." ICLR (2020).  
[5] Johnson, J., Douze, M., and Jégou, H. "Billion-scale similarity search with GPUs." IEEE Trans. Big Data (2019).  
[6] Angelopoulos, A. N., and Bates, S. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." Foundations and Trends in ML (2023).  
[7] Angelopoulos, A. N. et al. "Uncertainty Sets for Image Classifiers using Conformal Prediction." ICLR (2021). [RAPS]  
[8] Vovk, V. et al. "Algorithmic Learning in a Random World." Springer (2005).  
[9] Zhao, Y. et al. "Code Vulnerability Detection via Nearest Neighbor Mechanism." Findings of EMNLP (2022).  
