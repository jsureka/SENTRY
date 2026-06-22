# 1. Introduction

Pre-trained code models are increasingly deployed as automated gatekeepers: defect predictors that
flag risky commits, vulnerability detectors that triage code for review. In these settings a model's
**confidence** is as consequential as its prediction — a detector that is *confidently wrong* will
auto-approve a vulnerable change, while one that signals its uncertainty can defer to a human. Yet the
software-engineering literature evaluates these models almost exclusively on accuracy and F1, and
takes their softmax probabilities at face value. We show those probabilities are not trustworthy: a
fine-tuned CodeBERT on binary vulnerability detection reports a mean confidence of 0.81 while being
correct only 61% of the time.

The machine-learning community has tools for exactly this problem — temperature scaling for
calibration (Guo et al., 2017), retrieval augmentation for cheap accuracy gains (Khandelwal et al.,
2020), and selective prediction for principled abstention (Geifman & El-Yaniv, 2019) — but they are
seldom brought to bear on code classification, and never together on a frozen model. The closest software-engineering system, CodeImprove (Rathnasuriya et
al., 2025), improves accuracy by adapting the model's *input*; it does not touch calibration.

We ask a complementary, output-side question: **given a frozen, deployed code classifier, can a cheap
post-hoc layer keep its accuracy while making it stop being confidently wrong?** Our answer is
SENTRY, a training-free reliability layer with a single guarantee — *accuracy never below the base
model, calibration always improved* — achieved by gating a retrieval correction on whether the
representation can be trusted, and falling back to a calibrated model otherwise.

**Contributions.**

1. **SENTRY**, a training-free, inference-only reliability layer composing temperature scaling with
   reliability-gated, class-prior-corrected k-NN retrieval over a frozen CodeBERT/GraphCodeBERT
   classifier, exposing a retrieval-reliability signal for selective abstention (§3).
2. A full $2\times2$ (task $\times$ model) evaluation showing the layer improves accuracy *and*
   calibration significantly on defect prediction (+1.3–2.9 pp, $p\le4\times10^{-6}$; ECE 0.08→0.01)
   and preserves accuracy while fixing calibration on vulnerability detection (ECE 0.20→0.06) (§4).
3. **A mechanistic account of when retrieval helps**: the help/hurt boundary is governed by
   representation separability (base MCC), significant on both sides and across two model families —
   one mechanism, two outcomes, rather than a win claimed everywhere (§4.5).
4. A documented **data-integrity correction** of previously recorded calibration numbers, and a
   CPU-only reproduction harness that regenerates every figure and table from cached model outputs
   (§4.2.1, released).

We deliberately do not claim accuracy state-of-the-art; on vulnerability detection our base detector
matches the CodeXGLUE CodeBERT baseline, and line-level / data-flow methods remain stronger. SENTRY's
value is trustworthiness added on top of a comparable base, at no training cost.
