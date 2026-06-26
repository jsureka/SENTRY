"""
Whether prediction correction lowers risk-coverage (AURC) relative to kNN-UE, which
re-ranks confidence without changing predictions.

AURC has two levers: the error set (predictions) and the confidence ranking. kNN-UE
optimises ranking only (W>0 keeps argmax). SENTRY+calib applies both:
  1. confidence-gated kNN blend -> corrected predictions
  2. kNN-UE-style adaptive scaling on the corrected logits -> calibrated confidence
Expectation: SENTRY+calib below kNN-UE on AURC for separable tasks (multiclass, clone),
similar on vulnerability. Compares SENTRY+calib against kNN-UE-on-base, each with its
own adaptive calibration.

Run: python analysis/sentry_calib_vs_knnue.py
"""
import os, glob, csv
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import faiss; faiss.omp_set_num_threads(1)
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
EMB = os.path.join(REPO, "kNN-Prediction", "grid_emb")
BINARY_VULN = {"devign", "primevul", "reveal", "diversevul"}; HIGH_SEP = {"bigclone"}
K = 32; RNG = np.random.default_rng(0)


def relabel(*labs):
    u = sorted(set(np.concatenate(labs).tolist())); m = {v: i for i, v in enumerate(u)}
    return [np.array([m[v] for v in l.tolist()]) for l in labs], len(u)
def l2(x): x = x.astype(np.float32).copy(); faiss.normalize_L2(x); return x
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def aurc(err, conf):
    o = np.argsort(-conf); e = err[o].astype(float)
    cr = np.cumsum(e) / (np.arange(len(e)) + 1); cov = (np.arange(len(e)) + 1) / len(e)
    return float(np.trapz(cr, cov))
def family(ds):
    return "binary_vuln" if ds in BINARY_VULN else ("binary_clone" if ds in HIGH_SEP else "multiclass")


def knn_all(tr, trl, q, C):
    """faiss search: returns D (NxK), neighbour labels NxK, distance-weighted vote probs (k=8)."""
    trn, qn = l2(tr), l2(q)
    idx = faiss.IndexFlatL2(trn.shape[1]); idx.add(trn)
    D, I = idx.search(qn, K); nl = trl[I]
    w = 1.0 / (1.0 + D[:, :8]); P = np.zeros((len(q), C))
    for c in range(C): P[:, c] = (w * (nl[:, :8] == c)).sum(1)
    P /= P.sum(1, keepdims=True) + 1e-12
    return D, nl, P


def W_knn(p, D, S):
    a, tau, lam, b = p
    return np.clip((a / K) * np.exp(-D / max(tau, 1e-3)).sum(1) + lam * (S / K + b), 1e-3, 50.0)


def fit_W(z_va, va_l, D_va, S_va):
    def nll(p):
        P = softmax(W_knn(p, D_va, S_va)[:, None] * z_va)
        return -np.log(P[np.arange(len(va_l)), va_l] + 1e-12).mean()
    return minimize(nll, [1., 1., 1., 0.], method="L-BFGS-B",
                    bounds=[(0, 50), (1e-2, 50), (0, 50), (-5, 5)]).x


def gate_blend(model_p, knn_p, c, lam):
    """route to kNN blend only when model is unsure (maxprob < c). prob space (for val acc)."""
    unsure = model_p.max(1) < c
    out = model_p.copy()
    out[unsure] = (1 - lam) * model_p[unsure] + lam * knn_p[unsure]
    return out


def gate_blend_logit(zm, zk, model_p, c, lam):
    """gate blend in LOGIT space — keeps geometry kNN-UE's temperature expects."""
    unsure = model_p.max(1) < c
    out = zm.copy()
    out[unsure] = (1 - lam) * zm[unsure] + lam * zk[unsure]
    return out


def run(npz):
    name = os.path.basename(npz)[:-4]; enc, ds = name.split("__")
    d = np.load(npz)
    if "train_lab" not in d.files: return None
    (trl, val, tel), C = relabel(d["train_lab"], d["val_lab"], d["test_lab"])
    tr, va, te = d["train_emb"], d["val_emb"], d["test_emb"]

    base = LogisticRegression(max_iter=2000).fit(tr, trl)
    def lg(X):
        df = base.decision_function(X); return np.column_stack([-df, df]) if df.ndim == 1 else df
    z_va, z_te = lg(va), lg(te); p_va, p_te = softmax(z_va), softmax(z_te)

    D_va, nl_va, knn_va = knn_all(tr, trl, va, C)
    D_te, nl_te, knn_te = knn_all(tr, trl, te, C)

    # ---- baseline system: kNN-UE on BASE predictions ----
    pred_base_va, pred_base_te = z_va.argmax(1), z_te.argmax(1)
    S_va_b = (nl_va == pred_base_va[:, None]).sum(1); S_te_b = (nl_te == pred_base_te[:, None]).sum(1)
    Wb = fit_W(z_va, val, D_va, S_va_b)
    ue_conf = softmax(W_knn(Wb, D_te, S_te_b)[:, None] * z_te).max(1)
    err_base = (pred_base_te != tel).astype(float)
    aurc_ue = aurc(err_base, ue_conf)

    # ---- SENTRY+calib: gate-corrected predictions + kNN-UE-style calibration ----
    # fit gate (c,lam) on val accuracy
    best, ba = (0.9, 0.4), -1
    for c in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        for lam in [0.2, 0.4, 0.6, 0.8, 1.0]:
            a = (gate_blend(p_va, knn_va, c, lam).argmax(1) == val).mean()
            if a > ba: ba, best = a, (c, lam)
    c, lam = best
    zk_va, zk_te = np.log(knn_va + 1e-12), np.log(knn_te + 1e-12)      # kNN vote as logits
    zg_va = gate_blend_logit(z_va, zk_va, p_va, c, lam)               # blend in LOGIT space
    zg_te = gate_blend_logit(z_te, zk_te, p_te, c, lam)
    pred_g_va, pred_g_te = zg_va.argmax(1), zg_te.argmax(1)
    S_gva = (nl_va == pred_g_va[:, None]).sum(1); S_gte = (nl_te == pred_g_te[:, None]).sum(1)
    Wg = fit_W(zg_va, val, D_va, S_gva)
    sc_conf = softmax(W_knn(Wg, D_te, S_gte)[:, None] * zg_te).max(1)
    err_g = (pred_g_te != tel).astype(float)
    aurc_sc = aurc(err_g, sc_conf)

    n = len(tel); deltas = []
    for _ in range(1000):
        ix = RNG.integers(0, n, n)
        deltas.append(aurc(err_base[ix], ue_conf[ix]) - aurc(err_g[ix], sc_conf[ix]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"enc": enc, "ds": ds, "family": family(ds), "n": n,
            "acc_base": round(1 - err_base.mean(), 4), "acc_sc": round(1 - err_g.mean(), 4),
            "aurc_knnue": round(aurc_ue, 4), "aurc_sentrysc": round(aurc_sc, 4),
            "win": round(aurc_ue - aurc_sc, 4), "ci": f"[{lo:+.4f},{hi:+.4f}]",
            "sc_sig_better": bool(lo > 0), "knnue_sig_better": bool(hi < 0)}


def main():
    rows = [r for f in sorted(glob.glob(os.path.join(EMB, "*.npz"))) if (r := run(f)) is not None]
    cols = ["enc", "ds", "family", "n", "acc_base", "acc_sc", "aurc_knnue", "aurc_sentrysc",
            "win", "ci", "sc_sig_better", "knnue_sig_better"]
    with open(os.path.join(REPO, "results", "sentry_calib_vs_knnue.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print(f"{'point':>24} {'fam':>13} | acc {'base':>6}->{'sc':>6} | aurc {'kNN-UE':>7}{'sc':>8} | {'win':>8} sig")
    for r in rows:
        s = "V2" if r["sc_sig_better"] else ("UE" if r["knnue_sig_better"] else "..")
        print(f"{r['enc']+'/'+r['ds']:>24} {r['family']:>13} | {r['acc_base']:>6}->{r['acc_sc']:>6} | "
              f"{r['aurc_knnue']:>7}{r['aurc_sentrysc']:>8} | {r['win']:>+8} {s}")
    print(f"\n{'='*64}\nSENTRY+calib (gate-correct + kNN-UE calib) vs kNN-UE-on-base — AURC\n{'='*64}")
    byf = {}
    for r in rows: byf.setdefault(r["family"], []).append(r)
    for fam, rs in sorted(byf.items()):
        d = np.mean([r["win"] for r in rs])
        nv = sum(r["sc_sig_better"] for r in rs); nu = sum(r["knnue_sig_better"] for r in rs)
        da = np.mean([r["acc_sc"] - r["acc_base"] for r in rs])
        print(f"  {fam:>13} (n={len(rs)}): acc Δ {da:+.4f} | AURC win {d:+.4f} | sc sig-wins {nv}, kNN-UE sig-wins {nu}")
    print(f"\n  OVERALL: sc sig-better {sum(r['sc_sig_better'] for r in rows)}/{len(rows)}, "
          f"kNN-UE sig-better {sum(r['knnue_sig_better'] for r in rows)}/{len(rows)}")
    print("saved -> results/sentry_calib_vs_knnue.csv")


if __name__ == "__main__":
    main()
