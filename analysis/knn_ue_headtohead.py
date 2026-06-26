"""
SENTRY gated-kNN composite vs kNN-UE (Hashimoto, Kamigaito, Watanabe,
Findings of NAACL 2025).

kNN-UE (faithful, Eq. 4-5 of the paper):
  p(y|x) = softmax( W_kNN * z )                                            (Eq.4)
  W_kNN  = (alpha/K) * sum_k exp(-d_k / tau)  +  lambda*( S(yhat)/K + b )  (Eq.5)
    d_k    : L2 distance to k-th train neighbour (datastore = train reps)
    S(yhat): # of the K neighbours whose label == model's predicted label
    alpha,tau,lambda >=0, b in R  — fit on validation NLL via L-BFGS-B; K=32.
W>0 scalar per sample => predictions unchanged (argmax z preserved); only the
confidence is re-scaled. So it competes as a SELECTIVE / CALIBRATION signal.

Compared on AURC (selective prediction, base predictions fixed) and ECE (calibration).
Run: python analysis/knn_ue_headtohead.py
"""
import os, glob, csv
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import faiss; faiss.omp_set_num_threads(1)
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
import sys; sys.path.insert(0, os.path.join(REPO, "kNN-Prediction"))
from oos_metrics import validity_score_composite

EMB = os.path.join(REPO, "kNN-Prediction", "grid_emb")
BINARY_VULN = {"devign", "primevul", "reveal", "diversevul"}
HIGH_SEP = {"bigclone"}
K = 32
RNG = np.random.default_rng(0)


def relabel(*labs):
    u = sorted(set(np.concatenate(labs).tolist())); m = {v: i for i, v in enumerate(u)}
    return [np.array([m[v] for v in l.tolist()]) for l in labs], len(u)


def l2(x):
    x = x.astype(np.float32).copy(); faiss.normalize_L2(x); return x


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(probs, labels, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def aurc(err, conf):
    order = np.argsort(-conf); e = err[order].astype(float)
    cr = np.cumsum(e) / (np.arange(len(e)) + 1); cov = (np.arange(len(e)) + 1) / len(e)
    return float(np.trapz(cr, cov))


def family(ds):
    if ds in BINARY_VULN: return "binary_vuln"
    if ds in HIGH_SEP:    return "binary_clone"
    return "multiclass"


def knn_stats(tr, trl, q, pred_q, C):
    """returns (D KxN distances, S(yhat) per query, knn_dist_matrix, knn_probs) for composite."""
    trn, qn = l2(tr), l2(q)
    idx = faiss.IndexFlatL2(trn.shape[1]); idx.add(trn)
    D, I = idx.search(qn, K)
    nl = trl[I]
    S = (nl == pred_q[:, None]).sum(1)                 # neighbours matching predicted label
    # distance-weighted knn vote (for composite score, k=8 slice to match other harness)
    w = 1.0 / (1.0 + D[:, :8]); nl8 = nl[:, :8]
    P = np.zeros((len(q), C))
    for c in range(C):
        P[:, c] = (w * (nl8 == c)).sum(1)
    P /= P.sum(1, keepdims=True) + 1e-12
    return D, S, P


def W_knn(params, D, S):
    a, tau, lam, b = params
    dist = (a / K) * np.exp(-D / max(tau, 1e-3)).sum(1)
    return np.clip(dist + lam * (S / K + b), 1e-3, 50.0)


def fit_knnue(z_va, va_l, D_va, S_va):
    def nll(p):
        W = W_knn(p, D_va, S_va)
        P = softmax(W[:, None] * z_va)
        return -np.log(P[np.arange(len(va_l)), va_l] + 1e-12).mean()
    res = minimize(nll, x0=[1.0, 1.0, 1.0, 0.0], method="L-BFGS-B",
                   bounds=[(0, 50), (1e-2, 50), (0, 50), (-5, 5)])
    return res.x


def run(npz):
    name = os.path.basename(npz)[:-4]; enc, ds = name.split("__")
    d = np.load(npz)
    if "train_lab" not in d.files: return None
    (trl, val, tel), C = relabel(d["train_lab"], d["val_lab"], d["test_lab"])
    tr, va, te = d["train_emb"], d["val_emb"], d["test_emb"]

    base = LogisticRegression(max_iter=2000).fit(tr, trl)
    def logits(X):
        df = base.decision_function(X)
        return np.column_stack([-df, df]) if df.ndim == 1 else df
    z_va, z_te = logits(va), logits(te)
    pred_va, pred_te = z_va.argmax(1), z_te.argmax(1)
    p_te = softmax(z_te)
    err = (pred_te != tel).astype(float)

    D_va, S_va, _ = knn_stats(tr, trl, va, pred_va, C)
    D_te, S_te, knnP_te = knn_stats(tr, trl, te, pred_te, C)

    # --- kNN-UE (NAACL'25) ---
    params = fit_knnue(z_va, val, D_va, S_va)
    W_te = W_knn(params, D_te, S_te)
    ue_probs = softmax(W_te[:, None] * z_te)
    conf_ue = ue_probs.max(1)

    # --- SENTRY composite (ours) --- (high=unreliable -> negate for confidence)
    conf_sentry = -validity_score_composite(D_te[:, :8], pred_te, knnP_te.argmax(1), p_te, knnP_te)
    conf_msp = p_te.max(1)

    out = {"enc": enc, "ds": ds, "family": family(ds), "n": len(tel),
           "base_acc": round(1 - err.mean(), 4),
           "aurc_msp": round(aurc(err, conf_msp), 4),
           "aurc_knnue": round(aurc(err, conf_ue), 4),
           "aurc_sentry": round(aurc(err, conf_sentry), 4),
           "ece_base": round(ece(p_te, tel), 4),
           "ece_knnue": round(ece(ue_probs, tel), 4)}
    # bootstrap CI on (aurc_knnue - aurc_sentry); positive = sentry better
    n = len(err); deltas = []
    for _ in range(1000):
        ix = RNG.integers(0, n, n); eb = err[ix]
        deltas.append(aurc(eb, conf_ue[ix]) - aurc(eb, conf_sentry[ix]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    out["aurc_win_vs_knnue"] = round(out["aurc_knnue"] - out["aurc_sentry"], 4)
    out["ci"] = f"[{lo:+.4f},{hi:+.4f}]"; out["sentry_sig_better"] = bool(lo > 0)
    out["knnue_sig_better"] = bool(hi < 0)
    return out


def main():
    rows = [r for f in sorted(glob.glob(os.path.join(EMB, "*.npz"))) if (r := run(f)) is not None]
    cols = ["enc", "ds", "family", "n", "base_acc", "aurc_msp", "aurc_knnue", "aurc_sentry",
            "aurc_win_vs_knnue", "ci", "sentry_sig_better", "knnue_sig_better", "ece_base", "ece_knnue"]
    with open(os.path.join(REPO, "results", "knn_ue_headtohead.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print(f"{'point':>24} {'fam':>13} | {'msp':>7}{'kNN-UE':>8}{'SENTRY':>8} aurc | {'win':>8} sig | ECE base->UE")
    for r in rows:
        s = "S" if r["sentry_sig_better"] else ("U" if r["knnue_sig_better"] else ".")
        print(f"{r['enc']+'/'+r['ds']:>24} {r['family']:>13} | "
              f"{r['aurc_msp']:>7}{r['aurc_knnue']:>8}{r['aurc_sentry']:>8} | {r['aurc_win_vs_knnue']:>+8} {s}  | "
              f"{r['ece_base']:.3f}->{r['ece_knnue']:.3f}")

    print(f"\n{'='*66}\nSENTRY composite vs kNN-UE (NAACL'25) — selective AURC\n{'='*66}")
    byf = {}
    for r in rows: byf.setdefault(r["family"], []).append(r)
    for fam, rs in sorted(byf.items()):
        d = np.mean([r["aurc_win_vs_knnue"] for r in rs])
        ns = sum(r["sentry_sig_better"] for r in rs); nu = sum(r["knnue_sig_better"] for r in rs)
        print(f"  {fam:>13} (n={len(rs)}): mean AURC delta {d:+.4f} | SENTRY sig-wins {ns}, kNN-UE sig-wins {nu}")
    allr = rows
    print(f"\n  OVERALL: SENTRY sig-better {sum(r['sentry_sig_better'] for r in allr)}/{len(allr)}, "
          f"kNN-UE sig-better {sum(r['knnue_sig_better'] for r in allr)}/{len(allr)}, "
          f"mean delta {np.mean([r['aurc_win_vs_knnue'] for r in allr]):+.4f}")
    print(f"  ECE (calibration, kNN-UE's home turf): base {np.mean([r['ece_base'] for r in allr]):.4f} "
          f"-> kNN-UE {np.mean([r['ece_knnue'] for r in allr]):.4f}")
    print("saved -> results/knn_ue_headtohead.csv")


if __name__ == "__main__":
    main()
