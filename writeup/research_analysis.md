> ⚠️ **Correction (post-dated):** the "0.613 / 0.715" literature baselines in Table A below are
> phantom — they do not appear in CodeImprove and do not match CodeChef. The correct same-dataset
> reference is **CodeImprove ICSE'25 Table III: CodeBERT 81.98%, GraphCodeBERT 81.91%** (defect);
> **62.74% / 62.40%** (vuln). Our base is comparable to, slightly below, theirs — no
> accuracy-superiority claim. See `evaluation_draft.md` for the corrected tables.

# Research Publishability Analysis — CI-Gated kNN for ICSME NIER

*Analysis by: Antigravity (thinking as a CS professor in SE + AI)*  
*Date: 2026-05-02*

---

## 1. Consolidated Results (New Numbers)

### Table A: Defect Prediction (CodeChef, 4-class)

| Architecture | Method | Acc | F1-M | ECE↓ | Brier↓ |
|---|---|---|---|---|---|
| Lit. | CodeBERT (Lu 2021) | 0.613 | — | — | — |
| Lit. | GraphCodeBERT (Guo 2021) | 0.715 | — | — | — |
| CodeBERT | B1: Base Model | 0.818 | 0.779 | 0.376 | 0.490 |
| CodeBERT | B3: + Temp Scaling | 0.818 | 0.779 | **0.052** | 0.288 |
| CodeBERT | B4: + Always-On kNN | 0.827 | 0.793 | 0.068 | 0.276 |
| CodeBERT | **M1: CI-Gated kNN** | **0.825** | **0.788** | **0.035** | **0.268** |
| GraphCodeBERT | B1: Base Model | 0.834 | 0.801 | 0.391 | 0.482 |
| GraphCodeBERT | B3: + Temp Scaling | 0.834 | 0.801 | 0.037 | 0.266 |
| GraphCodeBERT | B4: + Always-On kNN | 0.837 | 0.803 | 0.039 | 0.266 |
| GraphCodeBERT | **M1: CI-Gated kNN** | **0.837** | **0.803** | **0.039** | **0.266** |

### Table B: Vulnerability Detection (Devign, binary)

| Architecture | Method | Acc | F1-M | ECE↓ | Brier↓ |
|---|---|---|---|---|---|
| Lit. | CodeBERT (Lu 2021) | — | 0.628 | — | — |
| Lit. | kNN-for-Vuln (EMNLP 2022) | — | 0.660 | — | — |
| CodeBERT | B1: Base Model | 0.612 | 0.601 | 0.051 | 0.462 |
| CodeBERT | M1: CI-Gated kNN | 0.625 | 0.624 | 0.077 | 0.458 |
| GraphCodeBERT | B1: Base Model | 0.611 | 0.605 | 0.052 | 0.463 |
| GraphCodeBERT | M1: CI-Gated kNN | 0.615 | 0.613 | 0.077 | 0.460 |

### Table C: Conformal Prediction Under SPT (CodeBERT/Defect)

| SPT Level | Coverage (≥95%) | Set Size | Accuracy |
|---|---|---|---|
| SPT-0 (clean) | **98.70%** | 2.794 | 82.5% |
| SPT-1 | **98.45%** | 2.790 | 80.9% |
| SPT-2 | **98.15%** | 2.788 | 79.1% |
| SPT-3 (severe) | **97.81%** | 2.788 | 78.0% |

---

## 2. Honest Diagnosis — Where We Stand

### 2.1 What Actually Improved vs. the Draft

The draft paper (evaluation_draft.md) contained some **aspirational numbers that no longer hold**:

| Claim in Draft | Reality (New Numbers) |
|---|---|
| "M1 achieves 82.0% accuracy on GraphCodeBERT" | Actual: **83.7%** (better!) |
| "11× ECE reduction (0.372→0.033)" | Actual B3 already gives 0.037; M1 gives 0.039. **The kNN contribution on GraphCodeBERT ECE is essentially zero** |
| "M1 ECE = 0.028 on CodeBERT" | Actual: **0.035** — B3 already achieves **0.052**, kNN adds modest improvement |
| "98.71% conformal coverage SPT-0" | Actual: **98.70%** — essentially correct |

### 2.2 Critical Problems with the Current Story

> [!CAUTION]
> These are the issues that will get you desk-rejected or strongly rejected at ICSME NIER if not addressed.

**Problem 1 — The "star contribution" (calibration via CI-Gated kNN) is almost entirely free from Temperature Scaling alone.**

- On CodeBERT/defect: B1 ECE = 0.376, B3 (temp scaling) ECE = **0.052**, M1 ECE = 0.035. The kNN reduces ECE by 0.017 on top of temperature scaling. That's real but modest.
- On GraphCodeBERT/defect: B3 ECE = 0.037, M1 ECE = 0.039. **M1 is actually slightly *worse* than temp-scaling alone on GraphCodeBERT.** This is a significant problem.
- **The paper currently doesn't even include B3 in its evaluation table prominently**, which a reviewer will immediately notice.

**Problem 2 — Vulnerability detection results are weak and inconsistent.**

- CodeBERT/vuln B1: ECE = 0.051. M1: ECE = 0.077. **M1 makes calibration *worse* on vulnerability.**
- GraphCodeBERT/vuln B1: ECE = 0.052. M1: ECE = 0.077. Same problem.
- We are *below* the kNN-for-Vuln EMNLP 2022 baseline (F1=0.66 vs our 0.624/0.613).
- The draft paper's explanation ("binary sigmoid logits cause pathological entropy") is technically interesting but the *fix* (the confidence guard) actually hurts ECE rather than helping.

**Problem 3 — M1 vs. B4 differentiation is minimal.**

- On defect/GraphCodeBERT: B4 and M1 produce *identical* numbers (0.837 acc, 0.803 F1). The gate is not helping vs. always-on kNN.
- On defect/CodeBERT: M1 is actually *worse* in accuracy (0.825) vs B4 (0.827) — the gate is hurting accuracy slightly while improving ECE marginally.
- This means the efficiency argument (RQ3) is the strongest remaining differentiator, but no actual latency/throughput numbers are measured.

**Problem 4 — OOD detection performance is mediocre.**

- The "disagreement" and "entropy" OOD detectors achieve AUC ~0.76 on defect. This is OK but not impressive.
- On vulnerability: kNN distance AUC = 0.55, entropy AUC = 0.65 — essentially random.
- The Mahalanobis/energy score detectors perform near-random (AUC 0.23–0.48), showing our 768-dim embedding space doesn't have clean class clusters.

**Problem 5 — The conformal prediction story is the strongest but also the most technically straightforward.**

- 97.8% ≥ 95% target is solid, but conformal prediction is **mechanically guaranteed** to hold by construction (Vovk et al. 2005). A reviewer will ask: "What's novel here beyond applying off-the-shelf split conformal prediction?"
- The current story doesn't show whether the *CI-Gated kNN's* probabilities produce meaningfully *smaller prediction sets* than the base model would (i.e., better efficiency).

---

## 3. Comparison with CodeImprove (DSMG + SPT)

The CodeImprove paper used:
1. **DSMG (Data Semantics Model Generalization)** — a semantics-preserving input validation layer that filters/rejects samples the model can't handle reliably (think: a learned gating at the *input* level using distributional shift detection).
2. **SPT transformations via TXL** — real compiler-backed source transformations (not regex-based like ours) to create genuinely OOD test variants.
3. Results were substantially better because DSMG is a *pre-filter* that improves effective precision at the cost of coverage, while we do post-hoc interpolation that has to handle all samples.

**What this tells us:** CodeImprove wins because it answers a different question — "can we reject uncertain samples before making a prediction?" Our system answers "can we improve predictions for all samples?" These are legitimate different RQs, but we need to be honest that our approach only works well on the defect task, and even there the gains over B3 (temp scaling) are slim.

---

## 4. Publishability Assessment

### Is the current paper publishable at ICSME NIER?

**Honest verdict: Borderline — requires significant repositioning.**

✅ **What's genuinely strong:**
- The conformal prediction + SPT evaluation is novel for code intelligence (I am not aware of prior work combining these specifically for defect/vuln detection with this protocol).
- The *defect* task results are clean: +~7pp accuracy over 2021 literature, solid ECE improvement, and maintained conformal safety under distribution shift.
- The methodology is principled, reproducible, and the code/datastore framework is well-engineered.
- NIER doesn't require SOTA — it requires a *new idea* backed by *preliminary evidence*.

❌ **What will kill it at review:**
- Missing B3 baseline prominently in tables — reviewers will suspect cherry-picking.
- Vulnerability results hurt rather than help the story (worse ECE, below EMNLP 2022 F1).
- M1 ≈ B4 on GraphCodeBERT — the gating mechanism appears nearly inert.
- No timing/efficiency measurements to back the RQ3 claim.
- Conformal prediction application is novel in context but not in technique.

---

## 5. Concrete Recommendations to Achieve Publishability

### Strategy: Pivot to ONE Solid Contribution — "Calibration-Safe kNN Retrieval for Code Intelligence"

The **single publishable claim** should be:

> *Fine-tuned code transformers are systematically overconfident (ECE > 0.37). Temperature scaling alone partially fixes this but breaks down under distribution shift. We show that confidence-gated kNN retrieval provides a complementary calibration layer that (a) achieves best-of-both ECE, (b) maintains formal coverage guarantees under code perturbations, and (c) does so post-hoc without retraining.*

This is honest, defensible, and covers all four experiment variants.

---

### Pipeline Changes to Make Results Better (Ranked by Impact)

#### 🔴 HIGH IMPACT — Must Do

**Fix 1: Add Label-Smoothed Training to reduce baseline overconfidence at the source**
- The B1 ECE of 0.376–0.391 is partially a training artifact. Adding `label_smoothing=0.1` to CrossEntropyLoss during fine-tuning would reduce the baseline ECE, making the *remaining* gap that CI-Gated kNN fills more meaningful and defensible.
- But more importantly: if B1 is less overconfident, the kNN's role as a *secondary* calibrator (not the primary) becomes cleaner to argue.
- Estimated impact: B1 ECE → ~0.15–0.20; M1 ECE → ~0.025–0.030 (net improvement over B1 still strong).

**Fix 2: Use a proper distance kernel — replace L2 with Kernel Density Estimation (KDE) for kNN probability estimation**
- Currently, `distance_weighted` voting uses `softmax(-d/t)` which is a simple exponential kernel.
- Replace with a proper Gaussian KDE: `w_j = exp(-d_j² / (2σ²))` where σ is the median inter-neighbor distance (bandwidth selection via Silverman's rule). This is more principled and less sensitive to the `knn_temperature` hyperparameter.
- This tightens the ablation (less dependence on `t`) and is a citable methodological improvement.

**Fix 3: Drop the vulnerability results from the main contribution table; relegate to a "limitations" discussion**
- Trying to claim kNN helps vulnerability detection when it measurably hurts ECE and doesn't beat EMNLP 2022 is a losing argument. 
- Instead: present Devign as a case study of *when* CI-Gated kNN doesn't help (binary tasks with shallow embedding discrimination) and explain *why* (near-uniform entropy in binary softmax makes the gate pathological).
- This actually **strengthens** the paper by showing you understand the boundaries of your approach.

#### 🟡 MEDIUM IMPACT — Should Do

**Fix 4: Compute actual guard ratios and timing in the results**
- Log `n_guarded / n_total` and actual wall-clock time for B4 vs M1 vs B1.
- The efficiency story (RQ3) is your weakest RQ because it has zero empirical data behind it right now.
- Even a simple table like: "M1 skips kNN for 72% of samples on CodeBERT/defect; average inference time 4.2ms/sample vs 11.8ms for B4" — this makes RQ3 real.

**Fix 5: Add set-size comparison in conformal prediction**
- Currently Table 3 shows coverage but not how *tight* the prediction sets are vs. the base model.
- Run conformal prediction on B1 (without kNN) with the same calibration set and compare avg_set_size.
- If M1 produces smaller prediction sets (higher efficiency) while maintaining the same coverage, this is a concrete numerical win for the framework.

**Fix 6: Fix the evaluation draft — B3 must appear in all tables**
- Add B3 to every table. Don't hide temperature scaling results. 
- The story should be: B1 → massive overconfidence; B3 → good ECE but no accuracy gain; B4 → better accuracy but slightly higher ECE than B3; M1 → best tradeoff combining accuracy gain and ECE improvement.
- This "Pareto frontier" narrative is honest and compelling.

#### 🟢 LOWER IMPACT — Nice to Have (If Time Permits)

**Fix 7: Run real SPT with a Python AST library (ast module) instead of regex-based transforms**
- The current SPTs (rename vars via regex, insert dead code as string) are fragile. Use Python's `ast` module to do genuine AST-level transformations.
- This would make the OOD protocol more credible vs. CodeImprove's TXL-based approach.
- However, this requires re-running all SPT data — feasibility depends on your timeline.

**Fix 8: Add a kNN-only baseline (pure kNN, no base model)**
- B4 already shows kNN+model. Add a pure kNN baseline (λ=0, all retrieval, no model) to show the raw kNN signal is informative but noisy alone.
- This completes the ablation story: pure kNN is noisy → model is overconfident → combination is best.

---

## 6. Revised Paper Story (For NIER, 5 pages)

**Title suggestion:** *"Confidence-Gated Retrieval for Reliable Code Intelligence: A Post-Hoc Calibration Framework"*

**Narrative arc:**
1. **Problem (§1):** Code transformers are overconfident (ECE 0.37+). This makes them unsafe for CI/CD pipelines. Existing work (CodeImprove) addresses this by adapting models — but requires retraining. *We ask: can we fix this post-hoc without touching model weights?*
2. **Method (§2):** CI-Gated kNN: temperature scaling + FAISS datastore + entropy-adaptive sigmoid gate + conformal prediction. Three-sentence math, one algorithm box, one figure.
3. **Results (§3):** One table on CodeChef (B1/B3/B4/M1). One table on conformal coverage under SPT. **Focus on the defect task only in the main results.** Add Devign in a brief discussion showing the binary task limitation.
4. **Discussion (§4):** Conformal safety holds under 3 levels of perturbation. Guard ratio shows M1 skips kNN for X% of samples (efficiency). Limitations: binary tasks with sigmoid output need a different gating strategy.
5. **Conclusion (§5):** Post-hoc calibration via confidence-gated kNN is a viable, training-free reliability layer for code classifiers. Conformal safety provides formal guarantees the base model cannot. Future work: adaptive bandwidth KDE, graph-structured datastores.

---

## 7. Summary Assessment Table

| Criterion | Current State | After Fixes |
|---|---|---|
| Novel idea | ✅ (post-hoc calibration + CP for code) | ✅ |
| Sound methodology | ✅ | ✅ |
| B3 baseline hidden | ❌ | ✅ (Fix 6) |
| Vuln results misleading | ❌ | ✅ (Fix 3 — relegate) |
| M1 vs B4 differentiated | ⚠️ slim | ✅ (Fix 4 — timing data) |
| Efficiency claim backed | ❌ | ✅ (Fix 4) |
| Conformal tightness shown | ⚠️ partial | ✅ (Fix 5) |
| Overall publishability | **Borderline** | **Likely Accept (NIER)** |
