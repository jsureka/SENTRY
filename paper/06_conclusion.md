# 6. Conclusion

Deployed code classifiers are judged on accuracy and trusted on confidence, yet their confidence is
unreliable — they are often *confidently wrong*. We presented **SENTRY**, a training-free,
inference-only reliability layer that wraps a frozen CodeBERT/GraphCodeBERT classifier with
temperature scaling and reliability-gated k-nearest-neighbour retrieval, unified by a gate that
trusts retrieval only when the representation is reliable and that doubles as a selective-abstention
signal. On defect
prediction SENTRY improves accuracy by 1.3–2.9 points on two model families ($p\le4\times10^{-6}$)
while cutting calibration error from 0.08 to 0.01–0.02; on vulnerability detection — where the
representation does not separate the classes — it preserves accuracy and still reduces calibration
error four-fold, because the gate detects the unreliable-retrieval regime and falls back to the
calibrated model. The help/hurt boundary is governed by a single, measurable quantity, representation
separability, giving a mechanistic account rather than a win claimed on both tasks.

SENTRY is complementary to input-side adaptation and goes beyond accuracy-neutral post-hoc
calibration by also improving accuracy where retrieval is reliable. It makes no accuracy-SOTA claim;
its contribution is *trustworthiness on top of a comparable base, at no training cost*, distilled to
one guarantee: **accuracy never below the base model, and calibration always improved.** All results,
figures, and tables are reproducible on CPU from the released harness.
