# CI-Gated kNN — Results & Discussion Plan (1-week sprint)

> ⚠️ **SUPERSEDED (2026-06-22).** The numbers below predate the data-integrity correction:
> the recorded ECE column was wrong (defect inflated ~5×, vuln deflated ~4×) and the McNemar
> p-values were overstated. **Use [`results/VERIFIED_RESULTS.md`](results/VERIFIED_RESULTS.md)
> and the rebuilt slide.** Corrected story: defect improves accuracy *and* calibration
> (McNemar p=4e-6 codebert, p=2e-19 graphcodebert); vuln = retrieval can't help (representation
> doesn't separate, MCC≈0.26), so temperature scaling alone fixes calibration with accuracy
> preserved. Kept for historical context only.

*Prepared as research-guide triage. Read top to bottom. The numbers below are pulled
directly from `Aggregated_Raw_Results.md` and the `*_results/` JSON — nothing invented.*

---

## TL;DR (the honest verdict)

You have **one working result and one negative result**, plus a **safety wrapper that
holds**. That is a perfectly good "Results & Discussion" — *if you stop trying to sell it
as a win on both tasks*.

- **Defect prediction (CodeChef, 4-class): WORKS.** kNN gives a small but
  **statistically significant** accuracy gain over the base model (McNemar p < 1e-12 on
  CodeBERT, p < 1e-7 on GraphCodeBERT). Temperature scaling kills the calibration error;
  conformal prediction holds coverage under perturbation. This is your headline.
- **Vulnerability detection (Devign, binary): DOES NOT WORK.** kNN is flat-to-harmful,
  not significant (p = 0.36 / 0.06), and ECE gets *worse*. This is a **negative result** —
  present it as one. It is scientifically honest and it explains *why* (binary sigmoid +
  entropy gate is pathological). Negative results are presentable; spin is not.
- **The "barely moving" feeling is correct for two reasons** you should name out loud:
  1. On vuln, it genuinely doesn't move.
  2. On defect, **M1 (gated) ≈ B4 (always-on kNN)** — the *gate* adds almost nothing to
     accuracy. The gate's only justification is **efficiency** (skip retrieval on confident
     samples) and **you never measured it**. That empty cell is the single highest-value
     thing to fill this week.

**Do NOT pivot to a new method.** With one week and a results deadline, switching
approaches is how you end up with nothing. Harvest what exists, fill two gaps, reframe.

---

## Consolidated honest results (ready to paste into slides)

### Table 1 — Defect Prediction (CodeChef, 4-class). **This is the headline.**

> **Baseline note (corrected):** the old "0.613 / 0.715 (Lu/Guo 2021)" rows were phantom —
> those numbers appear nowhere in CodeImprove and don't match CodeChef. The correct same-dataset
> reference is **CodeImprove ICSE'25 Table III base accuracy: CodeBERT 81.98%, GraphCodeBERT
> 81.91%** (CodeChef defect). Our base sits *slightly below* theirs — no accuracy-superiority claim.

| Arch | Method | Acc | F1-M | ECE↓ | Brier↓ | vs B1 (McNemar) |
|---|---|---|---|---|---|---|
| Ref | CodeImprove base CodeBERT (ICSE'25) | .820 | — | — | — | — |
| Ref | CodeImprove base GraphCodeBERT (ICSE'25) | .819 | — | — | — | — |
| CodeBERT | B1 base | .788 | .736 | .359 | .513 | — |
| CodeBERT | B3 +temp scaling | .788 | .736 | **.053** | .324 | n.s. (acc identical) |
| CodeBERT | B4 always-on kNN | **.811** | **.773** | .032 | .283 | p=6e-15 ✓ |
| CodeBERT | **M1 CI-gated kNN** | .808 | .767 | **.028** | .283 | **p=6e-13 ✓** |
| GraphCodeBERT | B1 base | .806 | .762 | .372 | .502 | — |
| GraphCodeBERT | B3 +temp scaling | .806 | .762 | .044 | .303 | n.s. |
| GraphCodeBERT | B4 always-on kNN | **.821** | **.784** | .041 | .269 | p=2e-7 ✓ |
| GraphCodeBERT | **M1 CI-gated kNN** | **.821** | .783 | **.034** | .269 | **p=3e-8 ✓** |

**Read this honestly in the talk:** "Calibration is fixed by temperature scaling (B3).
kNN buys a real, significant accuracy gain on top (B4/M1). The gate (M1) matches always-on
kNN (B4) — its value is efficiency, shown in Table 3."

### Table 2 — Vulnerability Detection (Devign, binary). **Frame as negative result / limitation.**

| Arch | Method | Acc | F1-M | ECE↓ | vs B1 |
|---|---|---|---|---|---|
| Lit | CodeBERT (Lu 2021) | — | .628 | — | — |
| Lit | kNN-for-Vuln (EMNLP 2022) | — | .660 | — | — |
| CodeBERT | B1 base | .622 | .621 | .057 | — |
| CodeBERT | M1 CI-gated kNN | .627 | .626 | .092 (worse) | p=0.36 ✗ n.s. |
| GraphCodeBERT | B1 base | .619 | .614 | .036 | — |
| GraphCodeBERT | M1 CI-gated kNN | .611 | .604 | .162 (much worse) | p=0.06 ✗ n.s. |

**Read honestly:** "On binary detection the method fails. We below the 2022 kNN baseline,
the gain is not significant, and ECE degrades. Diagnosis: with two classes, the softmax
entropy is near-maximal for almost every sample, so the entropy gate routes everything to
kNN, and a noisy 768-dim binary embedding space adds noise rather than signal. This bounds
where the method applies: **multi-class tasks with separable embeddings.**"

### Table 3 — Conformal coverage under SPT perturbation (CodeBERT/defect). **Safety story.**

| SPT level | Target | Coverage | Set size | Point acc |
|---|---|---|---|---|
| SPT-0 clean | 95% | 98.7% | 2.79 | 82.5% |
| SPT-1 | 95% | 98.4% | 2.79 | 80.9% |
| SPT-2 | 95% | 98.2% | 2.79 | 79.1% |
| SPT-3 severe | 95% | 97.8% | 2.79 | 78.0% |

**Caveat you must pre-empt (a committee will ask):** set size 2.79 of 4 classes ≈ 70% of
the label space — coverage that high is *easy* with big sets. The number that gives this
teeth is **set size vs. the base model at equal coverage** (does M1 produce *tighter* sets
than B1?). That comparison is missing — see Gap 2.

---

## The two gaps worth filling this week (need GPU inference, NO retraining)

### Gap 1 — RQ3 efficiency (HIGHEST value, lowest risk)
The gate's whole reason to exist. Currently zero data.
- Log `guard_ratio` (fraction of samples that bypass kNN) — `KNNPredictor.predict()`
  already returns `n_guarded` / `guard_ratio` in `details`. You may already have it in the
  saved JSON; if not, one re-run.
- Time three configs end-to-end on the test set: B1 (no kNN), B4 (always-on), M1 (gated).
- Payoff sentence: *"M1 matches B4 accuracy/ECE while skipping FAISS retrieval on X% of
  samples, cutting mean inference latency from A ms to B ms."* Converts "M1≈B4, why bother"
  into the contribution.

### Gap 2 — Conformal set-size vs. baseline (gives Table 3 teeth)
- Run split conformal on **B1** probabilities with the same calibration set and α.
- Compare `avg_set_size`: if M1 sets are smaller at equal coverage, that's a concrete win.
  If they're the same, say so — still honest.

Both reuse the existing pipeline. No fine-tuning. If you have a Colab GPU session, this is
an afternoon, not a week.

### Optional stretch (only if Gaps 1–2 done and time left, still no retraining)
- **Rescue vuln (risky):** swap the entropy gate for a **margin gate** `|p−0.5|` on the
  binary task, or gate on **kNN neighbor agreement**. Might lift Devign; might not. Time-box
  to one day, and if it doesn't move, keep vuln as the negative result.
- **Embedding ablation for vuln:** re-extract the datastore with `mean_pool_last` or
  `avg_last4_cls` (already supported in `EmbeddingExtractor`) instead of `last_cls`. Binary
  kNN may be failing because `[CLS]` alone is a weak separator. No retraining — just
  re-embed + re-run. One re-run.

---

## What is NOT presentable / cut it

- **LLM comparison (`*_vs_llm_results/`): n = 5 samples.** Statistically meaningless. Either
  re-run on ≥300 samples (needs API budget) or **delete it from the deck entirely.** Do not
  show 5-sample accuracy on a slide.
- **B5 ProtokNN:** not significant anywhere (p ≈ 0.25–0.91). Drop to an appendix line or cut.
- **Mahalanobis / energy / relative-Mahalanobis OOD detectors:** AUC 0.23–0.49 ≈ random or
  worse. Cut, or show only as "these failed, embedding space lacks clean class clusters."
  Keep `disagreement`/`entropy`/`composite` (AUC ~0.77 on defect) if you want an OOD slide.

---

## Suggested 1-week schedule

| Day | Task | Needs |
|---|---|---|
| 1 | Lock narrative (headline = defect; vuln = limitation; safety = conformal). Rebuild Tables 1–3 in slides from numbers above. | none |
| 2 | Gap 1: guard ratio + latency for B1/B4/M1 on both defect models. | GPU |
| 3 | Gap 2: conformal set-size B1 vs M1. Update Table 3. | GPU |
| 4 | (Stretch) margin-gate or embedding ablation for vuln. Time-boxed. | GPU |
| 5 | Write Results & Discussion prose. Cut LLM/Proto/Maha slides. | none |
| 6 | Dry-run the talk. Pre-write answers to the 3 questions below. | none |
| 7 | Buffer. | — |

If **no GPU access this week**: do Days 1, 5, 6 only. You still have a complete, honest
Results & Discussion from existing numbers — just without the efficiency/set-size cells.

---

## Three questions a committee WILL ask — have answers ready

1. *"Isn't your calibration win just temperature scaling?"* → **Yes, and we say so (B3).**
   kNN's contribution is the significant accuracy gain (B4/M1 vs B1), not the calibration.
2. *"M1 ≈ B4 — why the gate?"* → Efficiency (Gap 1 numbers). If Gap 1 undone, this is your
   weakest point; do Gap 1.
3. *"Conformal coverage is trivial with big sets."* → Set-size-vs-baseline (Gap 2).

---

## Bottom line

This is a **defensible Results & Discussion today**, even with zero new compute:
a significant accuracy improvement on multi-class defect prediction via retrieval, full
calibration via temperature scaling, a distribution-free safety guarantee that survives
perturbation, and an honest negative result on binary vulnerability detection that maps the
method's boundary. Fill Gaps 1–2 if you can run inference, and you upgrade it from
"defensible" to "solid." Don't chase a new method.
