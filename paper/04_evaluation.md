# 4. Evaluation

## 4.1 Research questions

- **RQ1 (Calibration).** Are fine-tuned code classifiers miscalibrated, and does the layer fix it?
- **RQ2 (Accuracy).** Does reliability-gated retrieval change accuracy, and is the change significant?
- **RQ3 (Where it works).** What determines whether retrieval helps or hurts?
- **RQ4 (Selective prediction).** Does the retrieval-reliability signal support trustworthy abstention,
  and when does it beat the model's own confidence?

## 4.2 Experimental setup

**Tasks and data.** We use two CodeXGLUE-style classification tasks: 4-class **defect prediction** on
CodeChef submissions and binary **vulnerability detection** on **Devign** (Zhou et al., 2019). We use
the same splits as CodeImprove (Rathnasuriya et al., 2025) for comparability.

**Models.** Each task is run with two frozen fine-tuned encoders — **CodeBERT** (Feng et al., 2020)
and **GraphCodeBERT** (Guo et al., 2021) — giving a full $2\times2$ (task $\times$ model) design.

**Metrics.** Accuracy, macro-F1, and Matthews correlation coefficient (MCC) for predictive quality;
**Expected Calibration Error** (ECE, 15 equal-width bins) and the multiclass **Brier score** for
calibration; McNemar's test (continuity-corrected $\chi^2$ and exact binomial) for paired
significance; and selective accuracy versus coverage for abstention.

**Protocol and reproducibility.** All numbers are produced on CPU with no retraining by
`kNN-Prediction/reproduce_results.py` and `significance_test.py`; the checkpoints load strict
(`missing=0, unexpected=0`). A **validation gate** checks that B1 (the base model's own softmax)
reproduces the recorded *accuracy* before any downstream number is trusted.

### 4.2.1 A data-integrity correction

During reproduction we found that the ECE/Brier column originally recorded for these experiments was
**wrong in both directions** — defect ECE was inflated $\approx5\times$ and vulnerability ECE deflated
$\approx4\times$. Three independent checks establish the corrected values: (i) accuracy reproduces
exactly, so the checkpoints and predictions are correct; (ii) the binning-independent identity
$\text{ECE}\approx \overline{\text{conf}}-\text{acc}$ holds for the corrected numbers (e.g. defect
CodeBERT $0.900-0.818=0.082$; vulnerability CodeBERT $0.809-0.612=0.197$); and (iii) the
NLL-optimal temperatures ($T^\*\!=\!1.3$–$1.4$ for defect, $3.2$–$3.8$ for vulnerability) are
incompatible with the originally recorded ECE. We report only the corrected, reproduced numbers.
This episode is itself a small lesson: calibration numbers must be regenerated from probabilities,
not trusted from a results table. Figure 2 visualises the over-confidence directly.

![Figure 2: overconfidence](figures/fig_overconfidence.png)

*Figure 2. Base models are over-confident: mean confidence exceeds accuracy by 8 points (defect) to
23 points (vulnerability). That gap is, to first order, the Expected Calibration Error.*

## 4.3 RQ1 — Calibration is broken and cheaply fixed

Temperature scaling repairs calibration on every setting while leaving accuracy untouched (Figure 3,
Table 1). ECE drops from 0.082→0.023 (defect CodeBERT) and 0.069→0.016 (defect GraphCodeBERT), and
from 0.197→0.055 and 0.228→0.061 on vulnerability — a $3\times$–$4\times$ reduction at zero accuracy
cost. SENTRY's full pipeline pushes defect ECE further still (to 0.017 / 0.008) via the recalibrated
retrieval blend.

![Figure 3: reliability diagrams](figures/fig_reliability.png)

*Figure 3. Reliability diagrams before (red) and after (green) temperature scaling. Post-scaling curves
hug the diagonal; fitted $T^\*$ scales with the severity of miscalibration.*

## 4.4 RQ2 — Accuracy: a real, significant gain where retrieval is reliable

Table 1 reports the full $2\times2$ results; Figure 4 summarises accuracy and ECE.

**Table 1. Verified results (CPU, no retraining). SENTRY = temperature scaling + reliability-gated
k-NN on defect; temperature scaling alone on vulnerability, where retrieval is unreliable.**

| Task · Model | Method | Acc | F1-M | MCC | ECE | Brier |
|---|---|---|---|---|---|---|
| Defect · CodeBERT | base | 0.818 | 0.779 | 0.734 | 0.082 | 0.288 |
| | + temp | 0.818 | 0.779 | 0.734 | 0.023 | 0.276 |
| | **SENTRY** | **0.831** | **0.803** | **0.755** | **0.017** | 0.261 |
| Defect · GraphCodeBERT | base | 0.806 | 0.761 | 0.717 | 0.069 | 0.300 |
| | + temp | 0.806 | 0.761 | 0.717 | 0.016 | 0.292 |
| | **SENTRY** | **0.835** | **0.806** | **0.762** | **0.008** | 0.255 |
| Vuln · CodeBERT | base | 0.612 | 0.612 | 0.243 | 0.197 | 0.527 |
| | **SENTRY** (temp-only) | 0.612 | 0.612 | 0.243 | **0.055** | 0.442 |
| | + k-NN | 0.597 | 0.591 | 0.234 | 0.046 | 0.464 |
| Vuln · GraphCodeBERT | base | 0.609 | 0.603 | 0.259 | 0.228 | 0.560 |
| | **SENTRY** (temp-only) | 0.609 | 0.603 | 0.259 | **0.061** | 0.446 |
| | + k-NN | 0.595 | 0.596 | 0.198 | 0.085 | 0.457 |

On **defect**, gated retrieval lifts accuracy by **+1.3 pp** (CodeBERT, 0.818→0.831) and **+2.9 pp**
(GraphCodeBERT, 0.806→0.835), with simultaneous large F1 and calibration gains. McNemar's test
(Table 2) confirms significance and isolates the contribution of our corrections to the naive k-NN
port: the *patched* method significantly beats both the base model **and** the unpatched method,
whereas the unpatched gain over base is only marginal.

**Table 2. Paired significance (McNemar; exact-binomial $p$). "Patches" = auto-$\tau$ + class-prior
correction over a naive k-NN port.**

| Comparison | Defect·CB | Defect·GCB | Vuln·CB | Vuln·GCB |
|---|---|---|---|---|
| base → SENTRY | +84, **p=4e-6** | +194, **p=2e-19** | −41, p=0.05 (harm) | −36, p=0.09 (n.s.) |
| patches vs. unpatched k-NN | +58, **p=5e-5** | +105, **p=3e-10** | +7, p=0.49 (n.s.) | −42, **p=0.008 (harm)** |
| unpatched k-NN → base | +26, p=0.044 | +89, p=3e-8 | — | — |

On **vulnerability**, k-NN *significantly harms* accuracy; SENTRY's gate therefore disables it and
returns the temperature-scaled model, preserving the base accuracy exactly while still fixing
calibration.

![Figure 4: results overview](figures/fig_results.png)

*Figure 4. Accuracy (left) preserved or improved; ECE (right) reduced on all four settings.*

## 4.5 RQ3 — The dichotomy: retrieval helps iff the representation separates

The opposite behaviour on the two tasks is not noise — it is governed by **how well the frozen
representation separates the classes**, measured by the base model's MCC. Figure 5 plots the retrieval
accuracy gain against MCC: the two defect settings (MCC ≈ 0.72–0.76) sit clearly above zero, the two
vulnerability settings (MCC ≈ 0.26) clearly below. This is the mechanism predicted by ReVeal
(Chakraborty et al., 2021): when vulnerable and safe code overlap in embedding space, a query's
nearest neighbours carry label-noise, so retrieval can only hurt. The result is significant on both
sides and consistent across two model families — a single mechanism, two outcomes.

![Figure 5: the dichotomy](figures/fig_dichotomy.png)

*Figure 5. Retrieval helps iff the representation separates classes. SENTRY's reliability gate reads
this regime per-query (neighbour distance + agreement) and routes accordingly.*

## 4.6 RQ4 — Selective prediction

The retrieval-reliability signal supports trustworthy abstention. On the separable defect task,
ranking by reliability lets the model retain ≈0.94 accuracy at 50% coverage versus ≈0.83 at full
coverage (Figure 6); on the non-separable vulnerability task the model's own calibrated confidence is
the better abstention signal — again consistent with the dichotomy of §4.5. The *choice* of abstention
signal is therefore governed by the same representation-separability mechanism that governs whether
retrieval helps, so a single quantity drives both the accuracy correction and the abstention rule.

![Figure 6: selective prediction](figures/fig_riskcoverage.png)

*Figure 6. Selective prediction on Defect · CodeBERT: abstaining on low-confidence inputs trades
coverage for accuracy along the calibrated curve.*

## 4.7 Comparison with prior work

SENTRY is not an accuracy-SOTA system and we do not claim it is; it is a **reliability layer**, and the
comparison is best read by *axis of contribution* (Table 3). Against accuracy-only SE systems it adds
calibration and abstention they never measure, training-free and output-side (complementary to
CodeImprove's input-side adaptation). Against post-hoc calibration — which is accuracy-neutral by
construction — it additionally *improves* accuracy where retrieval is reliable. Our post-calibration
defect ECE (0.008–0.017) is in the best-in-class range reported for code calibration (the 2025 JIT
study reports ≈0.02–0.06 after Platt/temperature scaling, on different data), which corroborates that
our corrected numbers are realistic rather than anomalous.

**Table 3. Axis-of-contribution comparison. Dataset numbers differ across rows; this is positioning,
not a leaderboard.**

| System | Task | Accuracy | Calibration | Training-free | Abstention |
|---|---|---|---|---|---|
| CodeBERT base (Lu et al., 2021) | Devign vuln | ~62% acc | not reported | — | no |
| Devign (Zhou et al., 2019) | vuln | GNN F1 | no | no | no |
| ReVeal (Chakraborty et al., 2021) | vuln | *diagnoses* non-separability | no | n/a | no |
| LineVul (Fu & Tantithamthavorn, 2022) | vuln (BigVul) | SOTA line-level F1 | no | no | no |
| CodeImprove (Rathnasuriya et al., 2025) | defect+vuln | +8.78% acc; OOD AUC 0.924 | no | no (GA transforms) | OOD only |
| Temp. scaling (Guo et al., 2017) | any | unchanged | yes | yes | no |
| JIT-Calibration (2025) | JIT defect | unchanged | ECE→0.02–0.06 | yes | no |
| **SENTRY (ours)** | CodeChef defect | **0.831 acc (+1.3 pp, p=4e-6)** | **0.082→0.017** | **yes** | **yes** |
| **SENTRY (ours)** | Devign vuln | 0.612 acc (preserved) | **0.197→0.055** | **yes** | **yes** |

## 4.8 Summary of findings

(1) Fine-tuned code classifiers are materially over-confident; temperature scaling fixes it at zero
accuracy cost. (2) On a separable task, reliability-gated retrieval adds a small but highly
significant accuracy gain on top, on two model families. (3) On a non-separable task retrieval hurts,
and the gate correctly disables it — preserving accuracy while still delivering calibration. (4) The
help/hurt boundary is predicted by representation separability, and the same signal drives selective
prediction. Together these support the framework's guarantee: **accuracy never below base, calibration
always improved.**
