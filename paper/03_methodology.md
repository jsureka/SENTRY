# 3. Methodology

## 3.1 Problem formulation

Let $f$ be a fine-tuned code classifier (CodeBERT or GraphCodeBERT with a linear head) that maps a
code snippet $x$ to logits $z(x)\in\mathbb{R}^{C}$ over $C$ classes, with softmax probabilities
$p_{\text{model}}(x)=\mathrm{softmax}(z(x))$ and prediction $\hat{y}=\arg\max_c p_{\text{model}}(x)_c$.
We treat $f$ as **frozen**: SENTRY never updates its weights. Given only (i) the logits $z(x)$, (ii)
the penultimate-layer `[CLS]` embedding $h(x)\in\mathbb{R}^{d}$, and (iii) read access to the training
set, SENTRY produces a *reliability-enhanced* output — recalibrated probabilities, an optional
retrieval correction, and a conformal prediction set — at **inference time only, with no retraining**.

The design goal is a guarantee that is honest on every task:

> **accuracy $\ge$ the base model, and calibration always improved.**

SENTRY achieves this by *adding* signal where retrieval is trustworthy and *getting out of the way*
where it is not. The architecture is shown in Figure 1.

![Figure 1: SENTRY architecture](figures/fig_architecture.png)

*Figure 1. SENTRY is a training-free, inference-only layer over a frozen encoder. Logits flow through
temperature scaling; the `[CLS]` embedding queries a FAISS datastore of training embeddings. A
reliability gate decides how much to trust retrieval and interpolates; the calibrated output is
wrapped in a conformal prediction set with an abstention option.*

## 3.2 Component 1 — Temperature scaling

Fine-tuned code classifiers are badly over-confident: across our four settings the mean top-class
confidence exceeds accuracy by 8–23 points (§4). We correct this with temperature scaling (Guo et
al., 2017). A single scalar $T>0$ is fit on the held-out validation split by minimising
negative log-likelihood,

$$T^\* = \arg\min_{T}\; -\frac{1}{N}\sum_{i} \log \mathrm{softmax}\!\left(z(x_i)/T\right)_{y_i},$$

and applied at test time as $p_{\text{cal}}(x)=\mathrm{softmax}(z(x)/T^\*)$. Temperature scaling is
**accuracy-preserving** — it does not change the arg-max — so it can never hurt the prediction while
it repairs confidence. Empirically $T^\*$ is mild on the well-separated defect task ($\approx1.3$–
$1.4$) and large on the over-confident binary vulnerability task ($\approx3.2$–$3.8$), exactly
tracking the severity of miscalibration.

## 3.3 Component 2 — Embedding datastore

We build a datastore $\mathcal{D}=\{(h(x_i), y_i)\}$ from every training example, storing the frozen
encoder's `[CLS]` embedding and its label. Embeddings are L2-normalised and indexed with FAISS
(exact inner-product / L2 search; the training sets here are ~22k items, so exact search is cheap).
At test time a query embedding $h(x)$ retrieves its $k$ nearest neighbours
$\mathcal{N}_k(x)=\{(h_j, y_j)\}$ with distances $\{\delta_j\}$.

## 3.4 Component 3 — Reliability-gated k-NN

Following kNN-LM (Khandelwal et al., 2020), we form a retrieval distribution by distance-weighted
voting and interpolate it with the model:

$$p_{k\text{NN}}(x)_c \;\propto\; \sum_{j:\,y_j=c} w_j \big/ \pi_c, \qquad
  w_j=\mathrm{softmax}\!\left(-\delta_j / \tau\right)_j,$$
$$p_{\text{final}}(x) \;=\; \lambda(x)\, p_{\text{cal}}(x) \;+\; \big(1-\lambda(x)\big)\, p_{k\text{NN}}(x).$$

Three details matter for correctness and are part of our contribution over a naive port of kNN-LM:

- **Calibrated softmax temperature $\tau$ (auto).** Because embeddings are L2-normalised, FAISS
  distances lie in a narrow range; a fixed large $\tau$ collapses the soft-max weights to near-uniform
  and silently disables distance weighting. We set $\tau$ from the per-query distance spread so that
  near neighbours actually dominate.
- **Class-prior correction $\pi_c$.** Dividing the vote mass by the class frequency counteracts the
  majority-class bias of an imbalanced datastore (e.g. the 34/40/15/11% class split of the 4-class
  defect task), recovering minority-class recall and macro-F1.
- **Reliability gate $\lambda(x)$.** The gate decides *how much* to trust retrieval from a
  **retrieval-reliability** signal $r(x)$ computed from the mean neighbour distance and the neighbour
  vote agreement (the top class's share among the $k$ neighbours). When neighbours are close and agree,
  $\lambda(x)$ lowers, admitting the retrieval correction; when they are distant or split — the regime
  ReVeal (Chakraborty et al., 2021) identifies for non-separable representations — $\lambda(x)\to 1$
  and SENTRY falls back to the calibrated model, preserving accuracy. The same $r(x)$ doubles as a
  selective-prediction score for abstention.

Because the gate can always fall back to $p_{\text{cal}}$ (which is itself accuracy-preserving), the
retrieval stage can only *add* accuracy where it is reliable — it cannot drag the framework below the
base model when the gate is respected. The blended distribution is re-calibrated with a second
temperature so the interpolation does not re-introduce miscalibration.

## 3.5 Component 4 — Conformal prediction

Finally we wrap the calibrated output in **split-conformal Regularized Adaptive Prediction Sets**
(RAPS; Angelopoulos et al., 2021). Using the validation split as the calibration set, we compute the
RAPS non-conformity scores and the threshold $\hat{q}$ for a target miscoverage $\alpha$; at test time
the prediction set $\mathcal{C}(x)=\{c : \text{cumulative regularised score} \le \hat{q}\}$ satisfies
the marginal coverage guarantee $\Pr(y\in\mathcal{C}(x))\ge 1-\alpha$. We additionally evaluate
coverage under **semantic-preserving transformations** (SPTs) of the test inputs to probe robustness
of the guarantee to mild distribution shift. (On binary vulnerability detection, sets are trivially
$\{0\},\{1\}$ or $\{0,1\}$; the conformal component is most informative on the multi-class task.)

## 3.6 Summary

SENTRY is the composition: temperature scaling (always on, accuracy-preserving) → reliability-gated,
class-prior-corrected k-NN (engages only when retrieval is reliable) → conformal set construction
with abstention. Every component is post-hoc and training-free, and the gate is what turns three
independent tricks into a single framework with a defensible guarantee.
