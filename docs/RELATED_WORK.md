# Positioning vs Related Work — a training-free reliability framework

**What we contribute:** a training-free, output-side **reliability framework** for code classifiers
— temperature calibration + reliability-gated *k*NN retrieval + selective abstention over a frozen
model — that improves calibration on every task and, where the representation is separable, accuracy
and risk–coverage as well. We position it against **recent** (2023–2025) calibration / uncertainty
work and input-side adaptation; older methods appear as the lineage of the techniques we apply.

---

## 1. The closest recent work

| Work | Venue | Relation to us |
|---|---|---|
| **kNN-UE** — Hashimoto, Kamigaito, Watanabe | **Findings of NAACL 2025** | The direct recent competitor: *k*NN-based uncertainty via neighbour distance + label ratio. We implement it faithfully (Eq. 4–5) and compare on selective AURC + ECE; it is the uncertainty-scoring reference, while our complementary axis is prediction correction (accuracy), which calibration does not change. Never previously applied to *code*. |
| **Zhou et al.** — On Calibration of Pre-trained Code Models | **ICSE 2024** | Shows code models are miscalibrated in- and out-of-distribution; recommends temperature scaling. We apply calibration as one of the intervention families in the dichotomy study. |
| **Ding et al.** — Vulnerability Detection: How Far Are We? (PrimeVul) | **ICSE 2024** | Independent evidence that vulnerability detection is near-random *even fine-tuned* — i.e. a **representability ceiling**, which our binary-vulnerability result confirms from the frozen-embedding side. |
| **DiverseVul** — Chen et al. | **2023** | Recent large vuln benchmark; another point on our binary-vuln (non-separable) side. |
| **Spiess et al.** — Calibration and Correctness of LMs for Code | **ICSE 2025** | Calibration as a first-class concern for *generative* code LLMs; we work on encoder *classifiers* and add the separability characterisation. |
| **CodeImprove** — Rathnasuriya et al. | **ICSE 2025** | Input-side adaptation (semantic transforms, OOD detection) that raises accuracy. We are **output-side and complementary**: we characterise when output-side training-free help is even possible, and can stack on top. |

**The gap we fill:** no prior code-classification work asks *for which tasks* training-free
retrieval/gating/ensembling help, nor isolates the answer to **embedding separability** via a
binary-but-separable control (clone detection).

## 2. Technique lineage (applied, not contributed — cited for provenance only)

These are the building blocks we *use*; we do not claim them, and they are not our comparison
targets beyond the head-to-heads above.

- **Retrieval / *k*NN inference** — *k*NN-LM (Khandelwal et al., 2020); deep *k*NN (Papernot &
  McDaniel, 2018). We apply retrieval to frozen code-encoder classification.
- **Calibration** — temperature scaling (Guo et al., 2017). Used as the calibration baseline; it is
  rank-preserving (cannot change accuracy or selective ranking), which is exactly why prediction
  correction is a distinct axis.
- **Post-hoc OOD / uncertainty scores** — Mahalanobis (Lee et al., 2018), Energy (Liu et al., 2020),
  Relative Mahalanobis (Ren et al., 2021). Run as named baselines in `reliability_headtohead.py`.

## 3. Positioning

Post-hoc calibration and uncertainty methods — temperature scaling (Guo'17), Mahalanobis, energy,
and kNN-UE (NAACL 2025) — are **rank-preserving**: they improve calibration / uncertainty but leave
accuracy unchanged. SENTRY's reliability-gated retrieval additionally improves **accuracy** on
separable tasks (defect, McNemar p = 4e-6; +2.5–3.1pp across multiclass and clone), an axis those
methods do not touch, and falls back to calibration where retrieval is unreliable. Input-side
adaptation (CodeImprove, ICSE 2025) improves accuracy but reports no calibration; SENTRY is
output-side, reports both, and can stack on top. The separability characterisation (across 7 datasets
× 4 encoders) tells a deployer **when** the retrieval component will contribute.

## 4. Suggested related-work sentence for the paper

> Recent work calibrates code models post-hoc (Zhou et al., ICSE 2024; Spiess et al., ICSE 2025) and
> estimates uncertainty by retrieval (kNN-UE, Hashimoto et al., NAACL 2025) — all accuracy-neutral —
> while input-side adaptation raises accuracy but not calibration (CodeImprove, ICSE 2025). We present
> a training-free, output-side reliability framework that improves calibration on every task and
> additionally **raises accuracy** where the representation is separable (defect, p = 4e-6), with the
> gate falling back to calibration otherwise; the separability characterisation tells a deployer when
> the retrieval component contributes (corroborating the vulnerability ceiling of Ding et al., ICSE
> 2024).

### References (recent-first)
- Hashimoto, Kamigaito, Watanabe, "Efficient Nearest Neighbor based Uncertainty Estimation for NLP Tasks", Findings of NAACL 2025 ([arXiv:2407.02138](https://arxiv.org/abs/2407.02138)).
- Zhou et al., "On the Calibration of Pre-trained Code Models", ICSE 2024 (DOI 10.1145/3597503.3639126).
- Ding et al., "Vulnerability Detection with Code Language Models: How Far Are We?" (PrimeVul), ICSE 2024 ([arXiv:2403.18624](https://arxiv.org/abs/2403.18624)).
- Spiess et al., "Calibration and Correctness of Language Models for Code", ICSE 2025.
- Chen et al., "DiverseVul", 2023 ([arXiv:2304.00409](https://arxiv.org/abs/2304.00409)).
- Rathnasuriya et al., "CodeImprove", ICSE 2025 ([arXiv:2501.15804](https://arxiv.org/abs/2501.15804)).
- Khandelwal et al., "Generalization through Memorization: Nearest Neighbor Language Models", ICLR 2020.
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
- Lee et al., "A Simple Unified Framework for Detecting OOD Samples", NeurIPS 2018; Liu et al., "Energy-based OOD Detection", NeurIPS 2020.
