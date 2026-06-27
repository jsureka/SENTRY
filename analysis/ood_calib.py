"""
Whether a kNN-aware calibrator improves on temperature scaling under distribution
shift for code classifiers. Zhou et al. (ICSE 2024) report that temperature scaling's
effectiveness diminishes out-of-distribution.

Setup: train a linear probe + kNN datastore on dataset A (frozen encoder embeddings),
evaluate on A-test (in-distribution) AND B-test (cross-project shift, same encoder,
same binary vuln label space). Devign <-> ReVeal are both binary C/C++ vulnerability,
different projects = genuine covariate shift.

Methods (all training-free, post-hoc):
  base      — softmax(probe logits)
  temp      — temperature scaling, T fit on A-val NLL            (the Zhou baseline)
  knn       — distance-weighted kNN vote distribution from A datastore
  blend     — lambda*base + (1-lambda)*knn, lambda fit on A-val acc
  distemp   — kNN-distance-adaptive temperature: T_i = T*(1+alpha*z_i), z_i = standardized
              mean-neighbor-distance (far from training data -> hotter -> less confident).
              alpha fit on A-val ECE. The "OOD-aware calibrator".

Report ECE(15-bin), AURC, accuracy on ID and OOD. The claim target: under SHIFT,
temp ECE/AURC degrade while a kNN-aware method stays lower. Reads cached grid_emb.
"""
import os, glob, itertools
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import faiss; faiss.omp_set_num_threads(1)
from sklearn.linear_model import LogisticRegression
import sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "kNN-Prediction"))
from calibration import compute_ece
from separability import aurc
from methods_grid import softmax, fit_temperature

EMB = os.path.join(HERE, "kNN-Prediction", "grid_emb")
ENCODERS = ["codebert", "graphcodebert", "unixcoder", "codet5p"]
PAIRS = [("devign", "reveal"), ("reveal", "devign")]   # (train A, shift-test B)
K = 8


def load(enc, ds):
    d = np.load(os.path.join(EMB, f"{enc}__{ds}.npz"))
    return {s: (d[f"{s}_emb"].astype(np.float32), d[f"{s}_lab"].astype(int)) for s in ["train", "val", "test"]}


class Store:
    def __init__(self, emb, lab):
        e = emb.copy(); faiss.normalize_L2(e)
        self.idx = faiss.IndexFlatL2(e.shape[1]); self.idx.add(e)
        self.lab = lab
    def query(self, emb, k=K, C=2):
        q = emb.copy(); faiss.normalize_L2(q)
        D, I = self.idx.search(q, k)
        nl = self.lab[I]
        w = 1.0 / (1.0 + D)
        P = np.zeros((len(emb), C))
        for c in range(C):
            P[:, c] = (w * (nl == c)).sum(1)
        P /= P.sum(1, keepdims=True)
        return P, D.mean(1)        # knn probs, mean neighbor distance


def metrics(probs, labels):
    pred = probs.argmax(1)
    ece, _ = compute_ece(probs, labels, n_bins=15)
    return dict(acc=round(float((pred == labels).mean()), 4),
                ece=round(float(ece), 4), aurc=round(float(aurc(probs, labels)), 4))


def aurc_score(correct, score):
    """selective-prediction AURC: fix predictions, rank ABSTENTION by `score` (high=keep).
    Lower = better at sending wrong predictions to abstain. correct: bool array."""
    order = np.argsort(-score)
    c = correct[order].astype(float)
    n = np.arange(1, len(c) + 1)
    risk = np.cumsum(1 - c) / n          # cumulative error over kept set
    cov = n / len(c)
    return float(np.trapz(risk, cov))


def run_pair(enc, A, B):
    da, db = load(enc, A), load(enc, B)
    tr_e, tr_l = da["train"]; va_e, va_l = da["val"]
    C = 2
    probe = LogisticRegression(max_iter=2000).fit(tr_e, tr_l)
    def logits(e):
        d = probe.decision_function(e)
        return np.c_[np.zeros_like(d), d] if d.ndim == 1 else d
    store = Store(tr_e, tr_l)

    # calibration params fit on A-val
    T = fit_temperature(logits(va_e), va_l)
    _, va_dist = store.query(va_e, C=C)
    dmu, dsd = va_dist.mean(), va_dist.std() + 1e-9
    # blend lambda on A-val acc
    va_lg = logits(va_e); va_base = softmax(va_lg); va_knn, _ = store.query(va_e, C=C)
    lam = max([i/10 for i in range(11)],
              key=lambda l: ((l*va_base+(1-l)*va_knn).argmax(1) == va_l).mean())
    # distemp alpha on A-val ECE
    def distemp_probs(lg, dist):
        z = (dist - dmu) / dsd
        return softmax(lg / (T * (1 + np.clip(alpha, 0, None) * np.clip(z, -0.9, None))[:, None]))
    best_a, best_ece = 0.0, 1e9
    for alpha in [0.0, 0.25, 0.5, 1.0, 2.0]:
        z = (va_dist - dmu) / dsd
        p = softmax(va_lg / (T * (1 + alpha * np.clip(z, -0.9, None))[:, None]))
        e, _ = compute_ece(p, va_l, n_bins=15)
        if e < best_ece:
            best_ece, best_a = e, alpha
    alpha = best_a

    out = {}
    for tag, (e, l) in [("ID", db_split(da)), ("OOD", db_split(db))]:
        lg = logits(e); base = softmax(lg); knn, dist = store.query(e, C=C)
        z = (dist - dmu) / dsd
        distemp = softmax(lg / (T * (1 + alpha * np.clip(z, -0.9, None))[:, None]))
        out[tag] = {
            "base": metrics(base, l), "temp": metrics(softmax(lg / T), l),
            "knn": metrics(knn, l), "blend": metrics(lam * base + (1 - lam) * knn, l),
            "distemp": metrics(distemp, l),
        }
        # SELECTIVE PREDICTION: fix base predictions, compare abstention SCORES.
        correct = (base.argmax(1) == l)
        out[tag]["sel"] = {
            "msp":        aurc_score(correct, base.max(1)),               # softmax confidence
            "neg_dist":   aurc_score(correct, -dist),                     # close to data = keep
            "knn_agree":  aurc_score(correct, knn.max(1)),                # neighbor agreement
            "reliability": aurc_score(correct, base.max(1) * knn.max(1)),  # msp x agreement
            "msp_div_dist": aurc_score(correct, base.max(1) / (dist + 1e-6)),  # combo
        }
    out["params"] = dict(T=round(T, 3), lam=lam, alpha=alpha)
    return out


def db_split(d):
    return d["test"]


def main():
    METHODS = ["base", "temp", "knn", "blend", "distemp"]
    agg = {m: {"ID": {k: [] for k in ["acc", "ece", "aurc"]},
               "OOD": {k: [] for k in ["acc", "ece", "aurc"]}} for m in METHODS}
    sel_agg = []
    for enc, (A, B) in itertools.product(ENCODERS, PAIRS):
        r = run_pair(enc, A, B)
        sel_agg.append({"ID": r["ID"]["sel"], "OOD": r["OOD"]["sel"]})
        print(f"\n{enc}  {A}->{B}  (T={r['params']['T']} lam={r['params']['lam']} alpha={r['params']['alpha']})")
        print(f"  {'method':<9}" + "".join(f"{s+'_'+m:>12}" for s in ["ID", "OOD"] for m in ["ece", "aurc"]))
        for meth in METHODS:
            line = f"  {meth:<9}"
            for s in ["ID", "OOD"]:
                line += f"{r[s][meth]['ece']:>12}{r[s][meth]['aurc']:>12}"
            print(line)
            for s in ["ID", "OOD"]:
                for k in ["acc", "ece", "aurc"]:
                    agg[meth][s][k].append(r[s][meth][k])

    print("\n" + "=" * 72 + "\nMEAN over 8 (encoder x direction) runs\n" + "=" * 72)
    print(f"{'method':<9}{'OOD_ece':>10}{'OOD_aurc':>10}{'OOD_acc':>10}{'ID_ece':>10}{'ID_aurc':>10}")
    for meth in METHODS:
        a = agg[meth]
        print(f"{meth:<9}{np.mean(a['OOD']['ece']):>10.4f}{np.mean(a['OOD']['aurc']):>10.4f}"
              f"{np.mean(a['OOD']['acc']):>10.4f}{np.mean(a['ID']['ece']):>10.4f}{np.mean(a['ID']['aurc']):>10.4f}")

    print("\n" + "=" * 72 + "\nSELECTIVE PREDICTION: abstention-score AURC (lower=better), base preds fixed\n" + "=" * 72)
    print(f"{'score':<14}{'ID_AURC':>10}{'OOD_AURC':>10}")
    for sc in ["msp", "neg_dist", "knn_agree", "reliability", "msp_div_dist"]:
        idv = np.mean([sel_agg[s]["ID"][sc] for s in range(len(sel_agg))])
        odv = np.mean([sel_agg[s]["OOD"][sc] for s in range(len(sel_agg))])
        print(f"{sc:<14}{idv:>10.4f}{odv:>10.4f}")
    print("\nwin = a kNN-distance score (neg_dist / reliability / msp_div_dist) beats msp on OOD_AURC.")


if __name__ == "__main__":
    main()
