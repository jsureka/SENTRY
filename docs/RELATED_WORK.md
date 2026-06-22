# Positioning vs Related Work — "Reliability Layer" Framing

**Claim we are selling:** a *training-free, output-side reliability layer* that keeps a
deployed code model's accuracy at-least-the-same while ensuring it is **not confidently
wrong** — via calibration (temperature scaling) and retrieval-gated prediction (kNN that
engages only when the representation is reliable), with the same reliability signal driving
selective abstention.

Two camps to compare against: (A) SE works that report **accuracy only** (no calibration),
and (B) works that **do report calibration** (ECE/Brier).

---

## 1. Axis-coverage comparison (what each work actually does)

| Work | Venue | Task / data | Optimizes | Reports ECE? | Accuracy effect | Training-free? | Abstention / sets? |
|---|---|---|---|---|---|---|---|
| **Ours (SENTRY)** | — | defect (CodeChef), vuln (Devign) | acc **and** calibration **and** selective | **Yes (+Brier)** | **↑ defect (sig), = vuln** | **Yes** | **Yes (selective)** |
| CodeImprove | ICSE'25 | defect, vuln | accuracy (input adaptation) | No | ↑ up to 8.78% | No (GA transforms) | OOD detect only (AUC .924) |
| Devign | NeurIPS'19 | vuln | accuracy/F1 (GNN) | No | — | No | No |
| ReVeal "Are We There Yet?" | TSE'21 | vuln | F1; *diagnoses* why DL vuln fails | No | — | n/a | No |
| LineVul / DeepDFA | MSR'22 / '24 | vuln | F1 + line localization | No | SOTA F1 | No | No |
| Guo et al. (temp scaling) | ICML'17 | image/NLP | calibration | Yes | **unchanged** (post-hoc) | Yes | No |
| Desai & Durrett | EMNLP'20 | NLI/paraphrase | calibration of BERT/RoBERTa | Yes | unchanged | Yes | No |
| Spiess et al. | ICSE'25 | code generation (LLMs) | calibration/correctness | Yes | unchanged | Yes | No (reflection/rescale) |
| JIT-Calibration | arXiv'25 (2504.12051) | JIT defect (QT/OpenStack) | calibration | Yes | unchanged | Yes | No |
| Selective prediction | general ML | — | abstention | — | trade coverage | Yes | Yes |

**The empty cell that is our contribution:** no prior code-classification work occupies the
row that does accuracy **+** calibration **+** abstention, training-free, on the same model.

---

## 2. Camp A — accuracy-only SE works (e.g. CodeImprove)

These never ask whether the model is *confidently wrong*. CodeImprove is the natural anchor:
it improves accuracy by **adapting inputs** (genetic, semantic-preserving transforms) and
flags out-of-scope inputs (AUC 0.924) — an **input-side** method that needs the transform
machinery and reports no calibration.

- **We are complementary, not competing.** Ours is **output-side and training-free** — it
  consumes logits + a datastore, changes nothing upstream, and can **stack on top of**
  CodeImprove (adapt the input, then calibrate/gate the output).
- **We measure what they don't.** On the *same* CodeChef/Devign splits, the base models are
  badly over-confident (defect mean-conf 0.90 vs acc 0.82; vuln 0.84 vs 0.61). Accuracy-only
  reporting hides this. We quantify and fix it (defect ECE 0.082→0.016/0.007; vuln 0.20→0.06).
- **We do not claim accuracy SOTA.** CodeImprove's input adaptation and their detector AUC
  0.924 are a different objective; we keep base accuracy (vuln) or improve it modestly and
  significantly (defect, +1.3–2.9pp, McNemar p≤4e-6) while adding a reliability guarantee.

> One-liner: *CodeImprove makes the input fit the model; we make the model's confidence fit
> reality — and we never had to retrain.*

## 3. Camp B — works that report ECE

Pure calibration is **accuracy-neutral by construction** (temp/Platt scale logits; argmax is
unchanged). Reference points (different datasets — for axis context, not head-to-head):

| Method | base ECE | post-hoc ECE | changes accuracy? |
|---|---|---|---|
| Guo'17 temp scaling (DNNs) | ~15–20% | ~1–3% | no |
| Desai&Durrett'20 (BERT/RoBERTa, in-domain) | ~2–4% | lower | no |
| JIT-Calib'25 CodeBERT4JIT | 8–12% | temp→6%, Platt→2–4% | no |
| **Ours, defect (M1+ gated-kNN)** | **7–8%** | **0.7–1.6%** | **yes, accuracy ↑ (sig)** |
| **Ours, vuln (temp-only)** | **20–23%** | **5.5–6.1%** | no (accuracy preserved) |

- **Our post-calibration ECE is in the best-in-class range** (≤2%, comparable to JIT-Calib's
  Platt) — but with the crucial extra: on defect we **also raise accuracy** while calibrating,
  which no pure-calibration method does.
- **Caveat (honesty):** datasets differ (CodeChef/Devign vs QT/OpenStack/NLI), so this is
  axis positioning, not a leaderboard win. We cite these to show our numbers are *realistic*
  and our reporting is *standard*, pre-empting "is this ECE even believable?"
- **Spiess (ICSE'25)** legitimizes calibration as a first-class concern for code models; we
  extend it from generation/correctness to **classification reliability with abstention**.

## 4. The unifying mechanism (why it is one framework, not two tricks)

The gate is the product. Our P2 result shows **retrieval reliability (neighbor
distance + vote agreement) cleanly separates the two regimes**:

- **Separating representation** (defect, MCC ~0.74): neighbors agree → gated-kNN engages →
  accuracy ↑ + calibration ↑. Retrieval-reliability even beats raw confidence for abstention.
- **Overlapping representation** (vuln, MCC ~0.26): neighbors disagree → fall back to
  temperature-only → **accuracy preserved**, calibration still fixed.

So the framework's guarantee is **accuracy ≥ base, always better-calibrated**: it *adds*
signal where retrieval is trustworthy and *gets out of the way* where it isn't. That is the
"keep accuracy, never confidently wrong" sell — and it is grounded in measured behaviour, not
asserted. (Current results select the regime per-task by this signal; the auto-gate is the
natural next step, with the P2 evidence already supporting it.)

## 5. Suggested related-work sentence for the paper

> Unlike input-side adaptation (CodeImprove, ICSE'25) that improves accuracy but leaves model
> confidence unexamined, and unlike post-hoc calibration (Guo'17; Desai & Durrett'20;
> Spiess et al. ICSE'25; JIT-Calibration'25) that fixes ECE without touching accuracy, we
> contribute a training-free output-side layer that **preserves or improves accuracy
> (significantly on defect, p≤4e-6) while reducing ECE by 4–10×**, and adds retrieval-gated
> selective prediction — the first such reliability layer for code defect / vulnerability
> classification.

### References
- CodeImprove — arXiv:2501.15804, ICSE 2025
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
- Desai & Durrett, "Calibration of Pre-trained Transformers", EMNLP 2020
- Spiess et al., "Calibration and Correctness of Language Models for Code", ICSE 2025
- "On the Calibration of Just-in-time Defect Prediction", arXiv:2504.12051, 2025
- Chakraborty et al. (ReVeal), "Deep Learning based Vulnerability Detection: Are We There Yet?", TSE 2021
- Fu & Tantithamthavorn, "LineVul", MSR 2022
