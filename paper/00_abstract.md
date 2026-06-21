# SENTRY: A Training-Free Reliability Layer for Code Classifiers

## Abstract

Fine-tuned code classifiers such as CodeBERT and GraphCodeBERT are deployed as defect and
vulnerability detectors, yet their predicted confidences are trusted blindly and are, in fact,
badly miscalibrated: across four task–model settings we measure mean confidence exceeding accuracy
by 8 to 23 points. We present **SENTRY**, a training-free, inference-only reliability layer that
wraps a *frozen* classifier and is designed to keep its accuracy while ensuring it is not
*confidently wrong*. SENTRY composes three post-hoc components — temperature scaling for
calibration, reliability-gated k-nearest-neighbour retrieval over a datastore of training
embeddings, and split-conformal prediction sets — unified by a gate that trusts retrieval only when
the underlying representation is reliable. On 4-class defect prediction the gated retrieval improves
accuracy by 1.3–2.9 points on two model families (McNemar $p=4\times10^{-6}$ and $2\times10^{-19}$)
while cutting Expected Calibration Error from 0.08 to 0.01–0.02. On binary vulnerability detection,
where vulnerable and safe code overlap in representation space, k-NN significantly *harms* accuracy;
the gate detects this regime, disables retrieval, and returns the calibrated model — preserving
accuracy while still reducing ECE four-fold (0.20→0.06). The help/hurt boundary is predicted by a
single quantity, representation separability, giving an honest and mechanistic account rather than a
win claimed on both tasks. SENTRY is complementary to input-side adaptation (CodeImprove) and goes
beyond accuracy-neutral post-hoc calibration by also improving accuracy where retrieval is reliable.
We additionally document and correct a data-integrity error in previously recorded calibration
numbers, and release a CPU-only reproduction harness for every result.
