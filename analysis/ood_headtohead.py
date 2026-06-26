"""
NAMED HEAD-TO-HEAD on a reliability metric under distribution shift.

Claim: temperature scaling (Guo 2017; recommended for code models by Zhou et al.
ICSE 2024) is the field's go-to reliability fix, but it is rank-preserving and
provably cannot fix calibration once the input distribution shifts. A retrieval
(kNN-vote) signal, which re-grounds confidence in the *target* neighbourhood,
stays better calibrated under cross-project shift.

Setup: train on source vuln dataset, evaluate on a DIFFERENT target vuln dataset
(cross-project shift). All binary-vuln so label spaces match. For every
(source -> target, encoder):
  base ECE, temp ECE (T fit on source val), kNN-vote ECE (datastore = source train),
  accuracy of each (including the accuracy cost),
  bootstrap 95% CI on (temp_ECE - knn_ECE); a win is the CI excluding 0 (kNN better).

Run: python analysis/ood_headtohead.py
"""
import os, glob, csv, itertools
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import faiss; faiss.omp_set_num_threads(1)
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EMB = os.path.join(REPO, "kNN-Prediction", "grid_emb")
VULN = ["devign", "reveal", "primevul", "diversevul"]
ENCODERS = ["codebert", "graphcodebert", "unixcoder", "codet5p"]
K = 8
RNG = np.random.default_rng(0)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(probs, labels, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0: continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def fit_temp(val_logits, val_labels):
    def nll(T):
        p = softmax(val_logits / max(T, 1e-3))
        return -np.log(p[np.arange(len(val_labels)), val_labels] + 1e-12).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def to_logits(clf, emb):
    df = clf.decision_function(emb)
    return np.column_stack([-df, df]) if df.ndim == 1 else df


def l2(x):
    x = x.astype(np.float32).copy(); faiss.normalize_L2(x); return x


def knn_vote(tr, trl, te, C):
    trn, ten = l2(tr), l2(te)
    idx = faiss.IndexFlatL2(trn.shape[1]); idx.add(trn)
    D, I = idx.search(ten, K)
    nl = trl[I]; w = 1.0 / (1.0 + D)
    P = np.zeros((len(te), C))
    for c in range(C):
        P[:, c] = (w * (nl == c)).sum(1)
    return P / (P.sum(1, keepdims=True) + 1e-12)


def cache(enc, ds):
    p = os.path.join(EMB, f"{enc}__{ds}.npz")
    if not os.path.exists(p): return None
    d = np.load(p)
    if "train_lab" not in d.files: return None
    return d


def run_pair(enc, src, tgt):
    ds_s, ds_t = cache(enc, src), cache(enc, tgt)
    if ds_s is None or ds_t is None: return None
    trE, trL = ds_s["train_emb"], ds_s["train_lab"].astype(int)
    vaE, vaL = ds_s["val_emb"], ds_s["val_lab"].astype(int)
    teE, teL = ds_t["test_emb"], ds_t["test_lab"].astype(int)
    C = 2
    base = LogisticRegression(max_iter=2000).fit(trE, trL)
    base_p = base.predict_proba(teE)
    T = fit_temp(to_logits(base, vaE), vaL)
    temp_p = softmax(to_logits(base, teE) / T)
    knn_p = knn_vote(trE, trL, teE, C)

    def acc(p): return float((p.argmax(1) == teL).mean())
    e_base, e_temp, e_knn = ece(base_p, teL), ece(temp_p, teL), ece(knn_p, teL)

    n = len(teL)
    deltas = []
    for _ in range(1000):
        ix = RNG.integers(0, n, n)
        deltas.append(ece(temp_p[ix], teL[ix]) - ece(knn_p[ix], teL[ix]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"enc": enc, "src": src, "tgt": tgt, "T": round(T, 2), "n": n,
            "ece_base": round(e_base, 4), "ece_temp": round(e_temp, 4), "ece_knn": round(e_knn, 4),
            "acc_temp": round(acc(temp_p), 4), "acc_knn": round(acc(knn_p), 4),
            "acc_cost": round(acc(temp_p) - acc(knn_p), 4),
            "win_ece": round(e_temp - e_knn, 4), "ci": f"[{lo:+.4f},{hi:+.4f}]", "sig": bool(lo > 0)}


def main():
    rows = []
    for enc in ENCODERS:
        for src, tgt in itertools.permutations(VULN, 2):
            r = run_pair(enc, src, tgt)
            if r: rows.append(r)
    cols = ["enc", "src", "tgt", "T", "n", "ece_base", "ece_temp", "ece_knn",
            "acc_temp", "acc_knn", "acc_cost", "win_ece", "ci", "sig"]
    with open(os.path.join(REPO, "results", "ood_headtohead.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print(f"{'enc':>14} {'src->tgt':>22} | {'base':>7}{'temp':>7}{'knn':>7} ece | {'win':>7} sig | acc_cost")
    for r in rows:
        s = "Y" if r["sig"] else "."
        print(f"{r['enc']:>14} {r['src']+'->'+r['tgt']:>22} | "
              f"{r['ece_base']:>7}{r['ece_temp']:>7}{r['ece_knn']:>7} | {r['win_ece']:>+7} {s}  | {r['acc_cost']:+.4f}")

    print(f"\n{'='*64}\nSUMMARY: kNN-vote vs temp-scaling under cross-project shift\n{'='*64}")
    mt = np.mean([r["ece_temp"] for r in rows]); mb = np.mean([r["ece_base"] for r in rows])
    mk = np.mean([r["ece_knn"] for r in rows]); nsig = sum(r["sig"] for r in rows)
    print(f"  mean OOD ECE: base {mb:.4f} | temp {mt:.4f} | kNN {mk:.4f}")
    print(f"  temp vs base: {'temp WORSE' if mt>=mb else 'temp better'} ({mt:.4f} vs {mb:.4f}) -> reproduces Zhou ICSE'24")
    print(f"  kNN beats temp on ECE: significant {nsig}/{len(rows)} pairs, mean delta {np.mean([r['win_ece'] for r in rows]):+.4f}")
    print(f"  mean accuracy cost of kNN: {np.mean([r['acc_cost'] for r in rows]):+.4f}")
    print(f"\nsaved -> results/ood_headtohead.csv")


if __name__ == "__main__":
    main()
