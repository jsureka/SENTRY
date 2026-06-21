"""
make_figures.py — regenerate every paper figure from the verified caches.

Reads kNN-Prediction/out_<combo>_{test,dev}.npz (model logits/labels, cached by
reproduce_results.py) and results/CI_kNN_<combo>_results/verified_metrics.json.
No model/GPU needed. Outputs PNGs into this directory.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax

HERE = os.path.dirname(os.path.abspath(__file__))            # paper/figures
REPO = os.path.dirname(os.path.dirname(HERE))                # SENTRY
KP = os.path.join(REPO, "kNN-Prediction")
sys.path.insert(0, KP)
from calibration import TemperatureScaler, compute_ece      # noqa: E402

COMBOS = ["defect_codebert", "defect_graphcodebert", "vuln_codebert", "vuln_graphcodebert"]
PRETTY = {"defect_codebert": "Defect · CodeBERT", "defect_graphcodebert": "Defect · GraphCodeBERT",
          "vuln_codebert": "Vuln · CodeBERT", "vuln_graphcodebert": "Vuln · GraphCodeBERT"}
DEV = lambda t: "dev" if t.startswith("defect") else "valid"
RED, GREEN, TEAL, GREY, INK = "#c0392b", "#27ae60", "#0e7c86", "#9DB2BD", "#1b2a32"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def cache(combo, split):
    return np.load(os.path.join(KP, f"out_{combo}_{split}.npz"))


def base_cal(combo):
    """Return base probs, temp-scaled probs, labels, temperature."""
    te, dv = cache(combo, "test"), cache(combo, DEV(combo))
    ts = TemperatureScaler(); ts.fit(dv["logits"], dv["labels"].astype(int))
    return (softmax(te["logits"], axis=1), ts.calibrate(te["logits"]),
            te["labels"].astype(int), float(ts.temperature))


def vm(combo):
    pr = json.load(open(os.path.join(REPO, "results", f"CI_kNN_{combo}_results", "verified_metrics.json")))
    g = lambda pred: next(v for k, v in pr.items() if pred(k))
    return dict(b1=g(lambda k: k.startswith("B1")), b3=g(lambda k: k.startswith("B3")),
                b4=g(lambda k: k.startswith("B4") and "P1" in k),
                m1=g(lambda k: k.startswith("M1+") and "++" not in k),
                m1pp=g(lambda k: "++" in k))


def sentry_point(combo):
    """Framework's task-appropriate result: defect=gated-kNN, vuln=temp-only."""
    v = vm(combo)
    if combo.startswith("defect"):
        return v["m1"]["acc"], v["m1pp"]["ece"]      # accuracy from M1+, ECE recalibrated
    return v["b1"]["acc"], v["b3"]["ece"]            # vuln: accuracy preserved, ECE via temp


# ----------------------------------------------------------------------------- #
def fig_reliability():
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, c in zip(axes.ravel(), COMBOS):
        base, cal, y, T = base_cal(c)
        for probs, col, lab in [(base, RED, "base"), (cal, GREEN, f"+temp (T={T:.2f})")]:
            ece, info = compute_ece(probs, y, n_bins=10)
            ax.plot(info["confidence"], info["accuracy"], marker="o", ms=4, color=col,
                    label=f"{lab}: ECE={ece:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_title(PRETTY[c]); ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Reliability diagrams — temperature scaling restores calibration", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig_reliability.png", dpi=160); plt.close(fig)


def fig_overconfidence():
    labels, conf, acc = [], [], []
    for c in COMBOS:
        base, _, y, _ = base_cal(c)
        labels.append(PRETTY[c]); conf.append(base.max(1).mean()); acc.append((base.argmax(1) == y).mean())
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, conf, w, label="mean confidence", color=RED)
    ax.bar(x + w/2, acc, w, label="accuracy", color=TEAL)
    for i, (cf, ac) in enumerate(zip(conf, acc)):
        ax.text(i, max(cf, ac) + 0.02, f"gap {cf-ac:+.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability"); ax.legend()
    ax.set_title("Base models are over-confident (confidence ≫ accuracy)")
    fig.tight_layout(); fig.savefig(f"{HERE}/fig_overconfidence.png", dpi=160); plt.close(fig)


def fig_results():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(COMBOS)); w = 0.38
    bacc = [vm(c)["b1"]["acc"] for c in COMBOS]; sacc = [sentry_point(c)[0] for c in COMBOS]
    bece = [vm(c)["b1"]["ece"] for c in COMBOS]; sece = [sentry_point(c)[1] for c in COMBOS]
    a1.bar(x - w/2, bacc, w, label="base", color=GREY); a1.bar(x + w/2, sacc, w, label="SENTRY", color=TEAL)
    a1.set_title("Accuracy"); a1.set_ylim(0.5, 0.9); a1.set_ylabel("accuracy")
    a2.bar(x - w/2, bece, w, label="base", color=GREY); a2.bar(x + w/2, sece, w, label="SENTRY", color=GREEN)
    a2.set_title("Expected Calibration Error (lower = better)"); a2.set_ylabel("ECE")
    for ax in (a1, a2):
        ax.set_xticks(x); ax.set_xticklabels([PRETTY[c].replace(" · ", "\n") for c in COMBOS], fontsize=8); ax.legend()
    fig.suptitle("SENTRY: accuracy preserved or improved, calibration always improved", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig_results.png", dpi=160); plt.close(fig)


def fig_dichotomy():
    mcc = [vm(c)["b1"]["mcc"] for c in COMBOS]
    dacc = [(vm(c)["b4"]["acc"] - vm(c)["b1"]["acc"]) * 100 for c in COMBOS]   # retrieval Δacc (pp)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for c, mx, dy in zip(COMBOS, mcc, dacc):
        col = GREEN if dy > 0 else RED
        ax.scatter(mx, dy, s=130, color=col, zorder=3, edgecolor=INK, lw=0.6)
        ax.annotate(PRETTY[c], (mx, dy), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.set_xlabel("representation separability  (base MCC)")
    ax.set_ylabel("retrieval accuracy gain  (kNN − base, pp)")
    ax.set_title("Retrieval helps iff the representation separates classes")
    ax.text(0.30, 2.3, "kNN helps", color=GREEN, fontsize=11, weight="bold")
    ax.text(0.30, -1.6, "kNN hurts", color=RED, fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(f"{HERE}/fig_dichotomy.png", dpi=160); plt.close(fig)


def fig_riskcoverage():
    """Selective prediction on defect·CodeBERT: accuracy retained vs coverage, by confidence."""
    base, cal, y, _ = base_cal("defect_codebert")
    fig, ax = plt.subplots(figsize=(7, 5))
    for probs, col, lab in [(base, GREY, "base confidence"), (cal, TEAL, "SENTRY (calibrated)")]:
        s = probs.max(1); order = np.argsort(-s)
        correct = (probs.argmax(1) == y)[order].astype(float)
        cov = np.arange(1, len(correct) + 1) / len(correct)
        acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        ax.plot(cov, acc, color=col, lw=2, label=lab)
    ax.set_xlabel("coverage"); ax.set_ylabel("selective accuracy")
    ax.set_title("Selective prediction (Defect · CodeBERT): abstain on low-confidence inputs")
    ax.legend(); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig_riskcoverage.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    figs = [fig_reliability, fig_overconfidence, fig_results, fig_dichotomy, fig_riskcoverage]
    for f in figs:
        try:
            f(); print(f"OK  {f.__name__}")
        except Exception as e:
            print(f"FAIL {f.__name__}: {e}")
    print("done")
