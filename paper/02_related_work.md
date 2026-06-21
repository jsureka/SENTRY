# 2. Related Work

SENTRY sits at the intersection of four lines of research: pre-trained models for code
classification, post-hoc calibration of neural classifiers, retrieval-augmented prediction, and
selective / conformal prediction. We review each, then situate our work against the most closely
related software-engineering systems — deep vulnerability detectors and input-side program
adaptation.

## 2.1 Pre-trained models for code classification

Transformer encoders pre-trained on source code are the de-facto backbone for code-understanding
tasks. CodeBERT (Feng et al., 2020) adapts the BERT masked-language-modelling objective to bimodal
(natural-language / programming-language) data; GraphCodeBERT (Guo et al., 2021) augments the input
with data-flow edges to inject structural information. Both are evaluated through CodeXGLUE (Lu et
al., 2021), a benchmark suite of ten code-intelligence tasks, of which *defect detection* (the
Devign dataset) and *clone detection* are the classification tasks most relevant here. On CodeXGLUE
defect detection, fine-tuned CodeBERT reaches roughly 62% accuracy — a number we reproduce exactly
for our binary vulnerability setting (§4). These models are optimised and reported almost
exclusively on **accuracy and F1**; their predicted probabilities are taken at face value and never
audited. SENTRY treats such a fine-tuned encoder as a frozen black box and asks an orthogonal
question: *can its outputs be trusted, and can a cheap post-hoc layer make them more trustworthy
without retraining?*

## 2.2 Calibration of neural classifiers

A classifier is *calibrated* if its confidence matches its empirical accuracy. Modern deep networks
are systematically **over-confident** (Guo et al., 2017): their softmax probabilities are far higher
than the frequency with which they are correct. Guo et al. introduced **temperature scaling** — a
single scalar $T$ that rescales the logits, $p = \mathrm{softmax}(z/T)$, fit by minimising negative
log-likelihood on held-out data — and showed it is a remarkably strong, accuracy-preserving baseline
(it does not change the arg-max, only the confidence). Calibration quality is most often summarised
by the **Expected Calibration Error** (ECE) of Naeini et al. (2015), the bin-weighted gap between
confidence and accuracy, and visualised with reliability diagrams. ECE has known weaknesses — it is
sensitive to binning and is not a proper scoring rule — and a body of work revisits it (Nixon et
al., 2019; Minderer et al., 2021) and complements it with the Brier score. In NLP, Desai & Durrett
(2020) showed pre-trained transformers are reasonably calibrated in-domain but degrade out-of-domain,
with temperature scaling recovering much of the gap.

Calibration has only very recently reached code models. Spiess et al. (2025) study the calibration
and correctness of language models for code generation, arguing that trustworthy confidence is a
prerequisite for developer adoption. Closest to our setting, a 2025 study of just-in-time defect
prediction reports ECE for CodeBERT-based detectors (≈8–12%) and shows temperature and Platt scaling
reduce it to ≈2–6%. SENTRY differs in two ways. First, it makes calibration the *primary* lens on a
defect/vulnerability classification pipeline rather than an afterthought. Second — and unlike pure
post-hoc calibration, which is accuracy-neutral by construction — SENTRY's retrieval component can
*improve accuracy* where it is reliable, so the framework delivers calibration **and** an accuracy
gain on the tasks where retrieval helps.

## 2.3 Retrieval-augmented prediction

Rather than encoding all knowledge in parameters, retrieval-augmented models consult an external
datastore at inference. The k-nearest-neighbour language model (kNN-LM; Khandelwal et al., 2020)
interpolates a parametric LM with a distribution computed from the nearest training contexts in
representation space, improving perplexity *with no additional training*; kNN-MT (Khandelwal et al.,
2021) extends the idea to machine translation. In classification, Deep k-Nearest Neighbours (Papernot
& McDaniel, 2018) use the conformity of a test point to its neighbours across layers as a credibility
and robustness signal. SENTRY adopts the kNN-LM interpolation recipe for a classifier —
$p = \lambda\, p_{\text{model}} + (1-\lambda)\, p_{k\text{NN}}$ over a FAISS datastore of training
embeddings — but adds two things the original recipe lacks for our setting: a **reliability gate**
that decides *when* retrieval should be trusted (rather than always interpolating), and an awareness
that retrieval quality is bounded by how well the host representation separates the classes (§2.5).

## 2.4 Selective prediction and conformal prediction

A reliable deployed model should be able to *abstain*. Selective prediction, or classification with a
reject option (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017), trades coverage for accuracy by
declining to predict on low-confidence inputs; SelectiveNet (Geifman & El-Yaniv, 2019) learns the
reject head end-to-end. Conformal prediction (Vovk et al., 2005) instead returns *prediction sets*
with a finite-sample, distribution-free coverage guarantee; Regularized Adaptive Prediction Sets
(RAPS; Angelopoulos et al., 2021) produce small, adaptive sets and are especially effective with many
classes. These frameworks are well developed in vision and NLP but are rarely applied to code
defect / vulnerability classification. SENTRY brings both to bear: it exposes a **retrieval-reliability
score** (neighbour distance and vote agreement) as a selective-prediction signal, and wraps the
calibrated output in split-conformal RAPS sets whose coverage we verify under
semantic-preserving program transformations.

## 2.5 Deep vulnerability detection and its limits

Function-level vulnerability detection has been a flagship application of deep code models. Devign
(Zhou et al., 2019) learns over a joint graph of control- and data-dependencies; subsequent work
builds richer graph and sequence models (LineVul, Fu & Tantithamthavorn 2022; DeepDFA, Steenhoek et
al. 2024; statement-level LineVD). Crucially, the reality-check study **ReVeal** (Chakraborty et al.,
2021, *Are We There Yet?*) showed that these models generalise poorly and that, in representation
space, *vulnerable and non-vulnerable code overlap heavily* — the embeddings simply do not separate
the classes, and class imbalance compounds the problem. This finding is central to our results: it
**predicts** that retrieval over such a representation cannot help, because the nearest neighbours of
a query are label-noise. Our vulnerability experiments confirm this quantitatively (low MCC ≈ 0.26;
k-NN significantly *harms* accuracy), and the contribution is mechanistic — SENTRY's gate *detects*
the unreliable-retrieval regime and falls back to calibration, preserving accuracy. We make no
accuracy-SOTA claim on vulnerability detection; line-level and data-flow methods remain stronger on
that axis, and our base detector matches the CodeXGLUE CodeBERT baseline rather than the SOTA.

## 2.6 Input-side adaptation: CodeImprove

The most directly comparable software-engineering system is **CodeImprove** (Rathnasuriya et al.,
ICSE 2025), which improves a deployed code model by adapting its *input*: it scores whether a program
is out-of-scope (reported AUC ≈ 0.924) and uses genetic, semantic-preserving transformations to map
out-of-scope inputs back in-scope, raising accuracy by up to 8.78%. CodeImprove is **input-side** and
requires a transformation/search apparatus; it reports accuracy and out-of-scope detection but **no
calibration**. SENTRY is deliberately complementary: it is **output-side and training-free**, leaves
the input untouched, and contributes exactly what CodeImprove does not measure — calibrated
confidence and principled abstention. The two could be stacked (adapt the input, then calibrate and
gate the output).

## 2.7 Out-of-distribution detection

For completeness, we note post-hoc OOD detectors — maximum softmax probability (Hendrycks & Gimpel,
2017), Mahalanobis distance (Lee et al., 2018), and energy scores (Liu et al., 2020) — which we
evaluated as alternative gate signals. In our setting these were near-random and are not part of the
final framework; the model's own confidence and the retrieval-reliability signal proved stronger.

## 2.8 Positioning summary

To our knowledge, no prior work on code defect / vulnerability classification occupies the cell that
simultaneously (i) preserves or improves accuracy, (ii) corrects calibration, and (iii) offers
selective prediction with a coverage guarantee, all **training-free** on a frozen model. Accuracy-only
SE systems (CodeImprove, Devign, LineVul) ignore (ii)–(iii); post-hoc calibration (Guo et al., 2017;
Desai & Durrett, 2020; Spiess et al., 2025) is accuracy-neutral and ignores (iii); selective /
conformal methods supply (iii) in the abstract but are not instantiated for this domain. SENTRY fills
that cell, and its gate makes the accuracy/abstention trade-off *adaptive* to whether the underlying
representation is trustworthy.
