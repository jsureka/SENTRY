"""
HEAD-TO-HEAD reliability comparison: does SENTRY's retrieval-based confidence
signal rank errors better (lower selective-risk / AURC) than established
uncertainty methods from the literature?

Selective prediction with base-model predictions fixed; only the abstention score
varies. Lower AURC (area under risk-coverage) is better, isolating the reliability
signal.

Competitors:
  msp        Maximum Softmax Probability   (Hendrycks & Gimpel, ICLR 2017)
  energy     Energy score                  (Liu et al., NeurIPS 2020)
  maha       Class-cond. Mahalanobis       (Lee et al., NeurIPS 2018)
  knn_dist   Deep-kNN nearest distance     (Papernot & McDaniel 2018, kNN-LM lineage)
  sentry     gated-kNN composite (OURS)    distance+disagreement+entropy+conf-delta

A win is SENTRY AURC below the best baseline AURC with a bootstrap 95% CI on the
delta excluding 0.

Run: python analysis/reliability_headtohead.py
"""
import os, glob, csv
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import faiss; faiss.omp_set_num_threads(1)
from sklearn.linear_model import LogisticRegression

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "kNN-Prediction"))
from oos_metrics import (validity_score_knn_distance, validity_score_composite)
from knn_datastore import (compute_energy_ood_scores, compute_mahalanobis_ood_scores)

EMB = os.path.join(REPO, "kNN-Prediction", "grid_emb")
BINARY_VULN = {"devign", "primevul", "reveal", "diversevul"}
HIGH_SEP = {"clone", "bigclone"}            # binary but separable (filled when clone cache lands)
K = 8
RNG = np.random.default_rng(0)


def relabel(*labs):
    u = sorted(set(np.concatenate(labs).tolist()))
    m = {v: i for i, v in enumerate(u)}
    return [np.array([m[v] for v in l.tolist()]) for l in labs], len(u)


def l2(x):
    x = x.astype(np.float32).copy(); faiss.normalize_L2(x); return x


def knn_vote(tr, trl, te, C):
    """distance-weighted kNN -> (knn_probs, knn_pred, dist_matrix)."""
    trn, ten = l2(tr), l2(te)
    idx = faiss.IndexFlatL2(trn.shape[1]); idx.add(trn)
    D, I = idx.search(ten, K)
    nl = trl[I]; w = 1.0 / (1.0 + D)
    P = np.zeros((len(te), C))
    for c in range(C):
        P[:, c] = (w * (nl == c)).sum(1)
    P /= P.sum(1, keepdims=True) + 1e-12
    return P, P.argmax(1), D


def aurc_from_conf(err, conf):
    """Area under risk-coverage. err: 0/1 wrong. conf: higher=more confident.
    Matches separability.aurc trapz convention. Lower = better."""
    order = np.argsort(-conf)
    e = err[order].astype(float)
    cum_risk = np.cumsum(e) / (np.arange(len(e)) + 1)
    coverage = (np.arange(len(e)) + 1) / len(e)
    return float(np.trapz(cum_risk, coverage))


def family(ds):
    if ds in BINARY_VULN: return "binary_vuln"
    if ds in HIGH_SEP:    return "binary_clone"
    return "multiclass"


def run_point(npz):
    name = os.path.basename(npz)[:-4]
    enc, ds = name.split("__")
    d = np.load(npz)
    if "train_lab" not in d.files:
        return None
    (trl, tel), C = relabel(d["train_lab"], d["test_lab"])
    tr, te = d["train_emb"], d["test_emb"]

    base = LogisticRegression(max_iter=2000).fit(tr, trl)
    probs = base.predict_proba(te)
    pred = probs.argmax(1)
    err = (pred != tel).astype(float)
    df = base.decision_function(te)
    logits = np.column_stack([-df, df]) if df.ndim == 1 else df

    knn_probs, knn_pred, D = knn_vote(tr, trl, te, C)

    # confidence scores (higher = more confident); validity_* return high=unreliable -> negate
    conf = {
        "msp":      probs.max(1),
        "energy":   compute_energy_ood_scores(logits),                       # high=in-dist
        "maha":     compute_mahalanobis_ood_scores(tr, trl, te),             # high=in-dist
        "knn_dist": -validity_score_knn_distance(D),
        "sentry":   -validity_score_composite(D, pred, knn_pred, probs, knn_probs),
    }
    aurc = {k: aurc_from_conf(err, c) for k, c in conf.items()}

    # bootstrap CI on delta(baseline - sentry); positive = sentry better
    n = len(err)
    out = {"enc": enc, "ds": ds, "family": family(ds), "C": C, "n": n, "base_acc": round(1 - err.mean(), 4)}
    for k in aurc:
        out[f"aurc_{k}"] = round(aurc[k], 4)
    for base_k in ["msp", "energy", "maha"]:
        deltas = []
        for _ in range(1000):
            idx = RNG.integers(0, n, n)
            e_b = err[idx]
            d_b = aurc_from_conf(e_b, conf[base_k][idx]) - aurc_from_conf(e_b, conf["sentry"][idx])
            deltas.append(d_b)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out[f"win_vs_{base_k}"] = round(aurc[base_k] - aurc["sentry"], 4)
        out[f"ci_{base_k}"] = f"[{lo:+.4f},{hi:+.4f}]"
        out[f"sig_{base_k}"] = bool(lo > 0)        # sentry significantly better
    return out


def main():
    files = sorted(glob.glob(os.path.join(EMB, "*.npz")))
    rows = [r for f in files if (r := run_point(f)) is not None]
    cols = (["enc", "ds", "family", "C", "n", "base_acc",
             "aurc_msp", "aurc_energy", "aurc_maha", "aurc_knn_dist", "aurc_sentry",
             "win_vs_msp", "sig_msp", "ci_msp", "win_vs_energy", "sig_energy",
             "win_vs_maha", "sig_maha"])
    out_csv = os.path.join(REPO, "results", "reliability_headtohead.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in cols})

    # summary per family
    print(f"{'point':>24} {'fam':>13} {'base':>6} | {'msp':>7}{'enrg':>7}{'maha':>7}{'knn':>7}{'SENTRY':>8} | sig(msp/enrg/maha)")
    fam_win = {}
    for r in rows:
        sigs = "".join("Y" if r[f"sig_{b}"] else "." for b in ["msp", "energy", "maha"])
        best_base = min(r["aurc_msp"], r["aurc_energy"], r["aurc_maha"], r["aurc_knn_dist"])
        star = " *WIN" if r["aurc_sentry"] < best_base else ""
        print(f"{r['enc']+'/'+r['ds']:>24} {r['family']:>13} {r['base_acc']:>6} | "
              f"{r['aurc_msp']:>7}{r['aurc_energy']:>7}{r['aurc_maha']:>7}{r['aurc_knn_dist']:>7}{r['aurc_sentry']:>8} | {sigs}{star}")
        fam_win.setdefault(r["family"], []).append(r)

    print(f"\n{'='*70}\nHEAD-TO-HEAD SUMMARY (SENTRY vs each, mean AURC delta; +=SENTRY better)\n{'='*70}")
    for fam, rs in sorted(fam_win.items()):
        print(f"\n{fam} (n={len(rs)}):")
        for b in ["msp", "energy", "maha"]:
            d = np.mean([r[f"win_vs_{b}"] for r in rs])
            nsig = sum(r[f"sig_{b}"] for r in rs)
            print(f"  vs {b:7s}: mean AURC delta {d:+.4f}   significant wins {nsig}/{len(rs)}")
    print(f"\nsaved -> results/reliability_headtohead.csv")


if __name__ == "__main__":
    main()
