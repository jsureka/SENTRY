# SENTRY deck — animation guide

## Already baked in (automatic, no action needed)

- **Slide transitions on every slide.** Open in PowerPoint → Slide Show. Slides cross-**Fade**.
- **Morph on the walkthrough (slides 10 → 11 → 12).** The pipeline stations and the moving "code
  card" glide/grow between these slides, so the data looks like it flows. (Falls back to Fade on
  PowerPoint older than 2019 / older LibreOffice.)
- The walkthrough (slides 9–12) is also built as a **build sequence**: even with no animation, each
  click advances the scene one logical step.

> Why not full per-object animations baked in? PowerPoint's file format does not let tools reliably
> write shape entrance/exit animations (only transitions). The 5-minute manual pass below adds them.

## Optional 5-minute manual pass (for click-by-click reveals)

In PowerPoint: select an object → **Animations** tab → pick effect → set **Start: On Click** →
reorder in the **Animation Pane**.

| Slide | Objects → effect (in click order) |
|---|---|
| 3 Problem | red meter bar → **Wipe (L→R)**; right analogy card → **Fade** |
| 4 Why it matters | the 3 cards → **Fade**, one per click |
| 5 Existing work | the 3 cards → **Fade**, one per click |
| 7 Our idea | the 3 green tags → **Fade/Float In**, one per click |
| 8 Analogy | the 2 rows → **Fade**, one per click |
| 9–12 Walkthrough | leave to Morph; optionally **Fade** the caption line last |
| 13 RQs | the 3 RQ cards → **Fade**, one per click |
| 14 Calibration | "Before 0.37" → **Fade**; arrow → **Fade**; "After 0.03" → **Grow/Fade** |
| 15 RQ2 | bar chart → **Wipe (Up)**; the 2 side cards → **Fade** |
| 16 RQ3 | green "works" card → **Fade**; red "fails" card → **Fade** |
| 17 Compare | table rows top→bottom → **Fade**; highlight the SENTRY row last |
| 18 Discussion | the 3 columns → **Fade**, one per click |

Keep effects to **Fade / Wipe / Grow** — subtle and professional. Avoid spins/bounces.

## Presenter tip
Set transition **Duration ≈ 0.8–1.0s** (Transitions tab) for a smooth Morph. Use **arrow keys**
to step the walkthrough so the Morph plays one segment at a time.
