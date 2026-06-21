# 5. Discussion

## 5.1 The dichotomy is the contribution

It would be easy to report defect prediction as a win and bury vulnerability detection as a failed
experiment. The more useful scientific object is the **boundary between them**. SENTRY's retrieval
stage helps precisely when the frozen representation separates the classes (defect, MCC ≈ 0.74) and
hurts when it does not (vulnerability, MCC ≈ 0.26), with statistical significance on both sides and
across two model families. This converts a pair of disconnected results into a single falsifiable
mechanism — retrieval quality is bounded by representation quality — that also yields an actionable
design: a gate that reads representation reliability per query and routes accordingly. The same signal
that explains the dichotomy is the signal that makes the framework safe to deploy.

## 5.2 Threats to validity

**Construct validity.** ECE is binning-sensitive and is not a proper scoring rule. We mitigate this by
(i) reporting the Brier score alongside ECE, (ii) using a fixed 15-bin protocol, and (iii) grounding
every ECE in the binning-independent identity $\text{ECE}\approx\overline{\text{conf}}-\text{acc}$ and
the NLL-optimal temperature. The data-integrity correction in §4.2.1 is itself a construct-validity
finding: a recorded calibration column was wrong in both directions, and we therefore regenerate all
calibration numbers from probabilities rather than trusting any stored table.

**Internal validity.** All results come from strict-load reproduction with an accuracy validation gate,
so predictions are provably the frozen model's. The retrieval gain is established with paired McNemar
tests (continuity-corrected $\chi^2$ and exact binomial), and we separately test the *patches*
(auto-$\tau$, class-prior correction) against a naive k-NN port to show the gain is not an artefact of
the port. Notably, the unpatched method's gain over the base is only marginal ($p=0.044$); the
corrections are what make the defect result robust.

**External validity.** We evaluate two datasets (CodeChef defect, Devign vulnerability) and two
encoders, using the CodeImprove splits for comparability. Broader claims across languages, datasets,
and architectures are future work. The comparison with prior systems (Table 3) is *axis-of-contribution*
positioning, not a same-dataset leaderboard.

**Scope of the guarantee.** The "accuracy ≥ base" guarantee holds when the gate respects its
fall-back to the calibrated model. In our experiments the gate decision is made with the
reliability signal at the task level (informed per-query by neighbour distance and agreement); a fully
autonomous per-query gate is the natural next step, and the selective-prediction evidence (§4.6)
already shows the signal carries the needed information.

## 5.3 Limitations

SENTRY cannot make retrieval help on vulnerability detection, because the limitation is in the *frozen
representation*, not the layer: when vulnerable and safe code overlap in embedding space (Chakraborty
et al., 2021), no amount of post-hoc retrieval can recover a signal the encoder did not encode. The
honest consequence is that on such tasks SENTRY contributes calibration and abstention but not
accuracy. The conformal component is also uninformative on binary tasks, where prediction sets are
trivial.

## 5.4 Future work

Three directions follow directly. **(i) A fully autonomous per-query gate** that thresholds the
retrieval-reliability score, removing the task-level decision. **(ii) Separating representations for
vulnerability**: a contrastive retrieval encoder, or a line-level / data-flow datastore in the spirit
of LineVul (Fu & Tantithamthavorn, 2022) and DeepDFA (Steenhoek et al., 2024), to give the retrieval
stage a representation it can actually use. **(iii) Composition with input-side adaptation**: SENTRY
is output-side and CodeImprove (Rathnasuriya et al., 2025) is input-side, so the two are stackable —
adapt the input back in-scope, then calibrate and gate the output. Broader evaluation across more
languages, datasets, and encoder families would further test the separability mechanism.
