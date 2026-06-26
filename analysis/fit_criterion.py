"""
Stage-3: does separability S predict retrieval gain delta?
Consumes results/grid_results.csv (+ optional results/stage0_pilot.json) and runs:

  1. Spearman rho(S, delta)                         -- raw association
  2. Partial correlation rho(S, delta | base_acc)   -- ANTI "it's just accuracy"
  3. OLS delta ~ base_acc + S, S's t-stat/p         -- does S add signal over acc?
  4. Leave-one-DATASET-out CV: predict held-out points' delta from S vs from
     base_acc-only baseline (MAE + help/hurt sign accuracy) -- ANTI "correlation
     not prediction"; this is the actual deployable claim.

Picks the best S by LODO sign-accuracy and writes the scatter figure.
Pure numpy/scipy/sklearn/matplotlib.
"""
import os, json, csv
import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "grid_results.csv")
PILOT = os.path.join(REPO, "results", "stage0_pilot.json")
SCORES = ["S1_local_label_consistency", "S2_fisher_ratio", "S3_silhouette", "S4_mahalanobis_margin"]


def load_rows(include_pilot=True):
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append({**r, "dataset": r["dataset"], "kind": r.get("kind", "frozen_probe")})
    if include_pilot and os.path.exists(PILOT):
        for r in json.load(open(PILOT)):
            ds = r["combo"].split("_", 1)[1] if "_" in r["combo"] else r["combo"]
            ds = "codechef" if "defect" in r["combo"] else ("devign" if "vuln" in r["combo"] else ds)
            rows.append({**r, "encoder": r["combo"].split("_")[-1], "dataset": ds,
                         "kind": "fine_tuned", "point": r["combo"]})
    # coerce numerics
    for r in rows:
        for k in ["base_acc", "delta_retrieval_gain", "base_ece", "base_aurc"] + SCORES:
            r[k] = float(r[k])
    return rows


def partial_corr(x, y, z):
    """Pearson partial correlation of x,y controlling for z (residual method)."""
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    rx, ry = resid(x, z), resid(y, z)
    r, p = stats.pearsonr(rx, ry)
    return r, p


def ols_t(X, y):
    """OLS with t-stats. X already includes intercept col. Returns (beta, t, p)."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, t, p


def lodo_cv(rows, score, predictor_cols):
    """Leave-one-dataset-out: fit delta ~ predictor_cols on other datasets,
    predict held-out. Returns MAE and help/hurt sign accuracy."""
    datasets = sorted({r["dataset"] for r in rows})
    errs, sign_hits, n = [], 0, 0
    for held in datasets:
        tr = [r for r in rows if r["dataset"] != held]
        te = [r for r in rows if r["dataset"] == held]
        if len(tr) < 3 or not te:
            continue
        Xtr = np.array([[1.0] + [r[c] for c in predictor_cols] for r in tr])
        ytr = np.array([r["delta_retrieval_gain"] for r in tr])
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        for r in te:
            x = np.array([1.0] + [r[c] for c in predictor_cols])
            pred = x @ beta
            errs.append(abs(pred - r["delta_retrieval_gain"]))
            # sign agreement: predicted-helps vs actually-helps (threshold 0)
            sign_hits += int((pred > 0) == (r["delta_retrieval_gain"] > 0))
            n += 1
    return float(np.mean(errs)) if errs else np.nan, (sign_hits / n if n else np.nan), n


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} points "
          f"({sum(r['kind']=='fine_tuned' for r in rows)} fine-tuned, "
          f"{sum(r['kind']=='frozen_probe' for r in rows)} frozen)")
    acc = np.array([r["base_acc"] for r in rows])
    delta = np.array([r["delta_retrieval_gain"] for r in rows])
    print(f"delta range [{delta.min():.3f}, {delta.max():.3f}]  acc range [{acc.min():.3f}, {acc.max():.3f}]")

    print("\n=== baseline: can base_acc alone predict delta? ===")
    rho_a, p_a = stats.spearmanr(acc, delta)
    mae_a, sign_a, n = lodo_cv(rows, None, ["base_acc"])
    print(f"  spearman(acc,delta)={rho_a:+.3f} p={p_a:.3g} | LODO acc-only: MAE={mae_a:.4f} sign-acc={sign_a:.2f}")

    print("\n=== per-score: raw rho, partial rho|acc, OLS S-coef, LODO ===")
    results = {}
    for s in SCORES:
        sv = np.array([r[s] for r in rows])
        rho, p = stats.spearmanr(sv, delta)
        pr, pp = partial_corr(delta, sv, acc)
        X = np.c_[np.ones(len(rows)), acc, sv]
        beta, t, pvals = ols_t(X, delta)
        mae_s, sign_s, _ = lodo_cv(rows, s, [s])
        mae_sa, sign_sa, _ = lodo_cv(rows, s, ["base_acc", s])
        results[s] = dict(rho=rho, p=p, partial_r=pr, partial_p=pp,
                          ols_S_t=float(t[2]), ols_S_p=float(pvals[2]),
                          lodo_S_mae=mae_s, lodo_S_sign=sign_s,
                          lodo_accS_mae=mae_sa, lodo_accS_sign=sign_sa)
        print(f"  {s:30s} rho={rho:+.3f}(p={p:.2g}) partial|acc r={pr:+.3f}(p={pp:.2g}) "
              f"OLS t={t[2]:+.2f}(p={pvals[2]:.2g}) LODO[S]sign={sign_s:.2f} mae={mae_s:.4f}")

    best = max(SCORES, key=lambda s: (results[s]["lodo_S_sign"], -results[s]["lodo_S_mae"]))
    print(f"\nBEST score by LODO sign-accuracy: {best}  "
          f"(sign={results[best]['lodo_S_sign']:.2f} vs acc-only {sign_a:.2f})")

    # verdict
    r = results[best]
    verdict = ("STRONG" if (r["partial_p"] < 0.05 and r["lodo_S_sign"] > max(sign_a, 0.7))
               else "MARGINAL" if r["partial_p"] < 0.1 else "WEAK")
    print(f"VERDICT: {verdict}  "
          f"(partial p={r['partial_p']:.2g}, S adds over acc: "
          f"{'YES' if r['lodo_accS_sign']>=sign_a else 'NO'})")

    json.dump({"verdict": verdict, "best": best, "acc_baseline": {"spearman": rho_a, "lodo_sign": sign_a},
               "scores": results}, open(os.path.join(REPO, "results", "criterion_fit.json"), "w"), indent=2)
    _figure(rows, best)
    print(f"saved -> results/criterion_fit.json , results/fig_criterion.png")


def _figure(rows, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = sorted({r["dataset"] for r in rows})
    cmap = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds, c in zip(datasets, cmap):
        pts = [r for r in rows if r["dataset"] == ds]
        ax.scatter([r[best] for r in pts], [r["delta_retrieval_gain"] for r in pts],
                   color=c, label=ds, s=60,
                   marker="*" if pts[0]["kind"] == "fine_tuned" else "o",
                   edgecolor="k", linewidth=0.4)
    x = np.array([r[best] for r in rows]); y = np.array([r["delta_retrieval_gain"] for r in rows])
    b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(b, xs), "k--", lw=1, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel(f"separability  ({best})"); ax.set_ylabel("retrieval gain  Δacc (kNN − base)")
    ax.set_title("Does representation separability predict retrieval utility?")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(REPO, "results", "fig_criterion.png"), dpi=140)


if __name__ == "__main__":
    main()
