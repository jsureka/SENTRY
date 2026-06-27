# SENTRY — Paper

> **BEING REVISED to the representability-dichotomy framing (target: APSEC 2026).** The sections
> below predate the clone-detection control and the kNN-UE (NAACL'25) comparison and are superseded.
> Authoritative current results: [`../results/FINAL_VERDICT.md`](../results/FINAL_VERDICT.md);
> positioning: [`../docs/RELATED_WORK.md`](../docs/RELATED_WORK.md).

Academic write-up of SENTRY. Read **[`PAPER.md`](PAPER.md)** for the assembled document, or the
individual sections:

- [Abstract](00_abstract.md)
- [1. Introduction](01_introduction.md)
- [2. Related Work](02_related_work.md) — thorough literature review
- [3. Methodology](03_methodology.md)
- [4. Evaluation](04_evaluation.md) — results, significance, comparison
- [5. Discussion](05_discussion.md) — threats to validity, limitations
- [6. Conclusion](06_conclusion.md)
- [References](references.md)

Figures live in [`figures/`](figures/) and regenerate from the cached model outputs with
`python figures/make_figures.py` (no GPU; see [../docs/REPRODUCE.md](../docs/REPRODUCE.md)). Every
number traces to [`../results/VERIFIED_RESULTS.md`](../results/VERIFIED_RESULTS.md).

> Format note: written in Markdown for review and GitHub rendering; straightforward to port to a
> LaTeX template (ACM/IEEE) for submission.
