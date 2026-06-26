"""
Whether a cheap supervised transform of frozen embeddings makes training-free kNN
retrieval work on tasks where it otherwise does not (binary vulnerability).

Raw kNN helps only where classes already separate. We test whether
transforming the space first changes that. Transforms (all closed-form / no encoder
retraining, fit on train only):
  raw      — frozen embeddings as-is (baseline)
  white    — within-class whitening: W = Sw^{-1/2}; Euclidean in W-space = within-class
             Mahalanobis. Shrinks within-class spread, relatively expands between-class.
             Keeps full dim (unlike LDA) so it doesn't collapse binary tasks.
  lda      — Fisher LDA projection (multiclass only; binary -> 1D, reported but expected weak)
  nca      — Neighbourhood Components Analysis: linear transform that directly maximizes
             kNN leave-one-out accuracy (subsampled train for speed). The principled
             "make kNN work" transform.

Question per dataset: does kNN-in-transformed-space beat BOTH the base linear
probe AND a linear classifier in that same space?  If kNN only matches the linear model,
retrieval adds nothing and the transform just reproduces a linear boundary -> not a win.
If kNN > max(base, linear), retrieval has genuine local/non-linear value the transform
unlocked -> a genuine accuracy gain.

Reads cached kNN-Prediction/grid_emb/*.npz. Prints a table; focus = vuln rows.
"""
import os, glob
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import faiss; faiss.omp_set_num_threads(1)
from numpy.linalg import eigh
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

EMB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "kNN-Prediction", "grid_emb")
BINARY = {"devign", "primevul", "reveal", "diversevul"}
K = 8


def relabel(*labs):
    u = sorted(set(np.concatenate(labs).tolist()))
    m = {v: i for i, v in enumerate(u)}
    return [np.array([m[v] for v in l.tolist()]) for l in labs], len(u)


def l2(x):
    x = x.astype(np.float32).copy(); faiss.normalize_L2(x); return x


def knn_acc(tr, trl, te, tel, C):
    """distance-weighted kNN in given space; returns acc, f1, neighbor purity."""
    trn, ten = l2(tr), l2(te)
    idx = faiss.IndexFlatL2(trn.shape[1]); idx.add(trn)
    D, I = idx.search(ten, K)
    nl = trl[I]
    w = 1.0 / (1.0 + D)
    P = np.zeros((len(te), C))
    for c in range(C):
        P[:, c] = (w * (nl == c)).sum(1)
    pred = P.argmax(1)
    purity = float((nl == tel[:, None]).mean())   # fraction of neighbors sharing true label
    return float((pred == tel).mean()), float(f1_score(tel, pred, average="macro")), purity


def within_class_whiten(tr, trl, shrink=0.1):
    d = tr.shape[1]
    Sw = np.zeros((d, d))
    for c in np.unique(trl):
        X = tr[trl == c]
        Xc = X - X.mean(0)
        Sw += Xc.T @ Xc
    Sw /= len(tr)
    Sw += shrink * (np.trace(Sw) / d) * np.eye(d)
    vals, vecs = eigh(Sw)
    W = vecs @ np.diag(np.maximum(vals, 1e-8) ** -0.5) @ vecs.T
    return W


def run(npz):
    name = os.path.basename(npz)[:-4]
    enc, ds = name.split("__")
    d = np.load(npz)
    (trl, v4, tel), C = relabel(d["train_lab"], d["val_lab"], d["test_lab"])
    tr, te = d["train_emb"], d["test_emb"]

    base = LogisticRegression(max_iter=2000).fit(tr, trl)
    base_acc = float((base.predict(te) == tel).mean())

    out = {"enc": enc, "ds": ds, "C": C, "base": round(base_acc, 4)}

    # raw kNN
    a, f, pur = knn_acc(tr, trl, te, tel, C)
    out["knn_raw"], out["pur_raw"] = round(a, 4), round(pur, 3)

    # within-class whitening
    mu = tr.mean(0)
    W = within_class_whiten(tr, trl)
    trw, tew = (tr - mu) @ W, (te - mu) @ W
    a, f, pur = knn_acc(trw, trl, tew, tel, C)
    lin_w = float((LogisticRegression(max_iter=2000).fit(trw, trl).predict(tew) == tel).mean())
    out["knn_white"], out["pur_white"], out["lin_white"] = round(a, 4), round(pur, 3), round(lin_w, 4)

    # LDA (full Fisher) — for multiclass mainly
    try:
        lda = LDA().fit(tr, trl)
        trl_, tel_ = lda.transform(tr), lda.transform(te)
        a, f, pur = knn_acc(trl_, trl, tel_, tel, C)
        out["knn_lda"] = round(a, 4)
    except Exception:
        out["knn_lda"] = None

    # NCA (principled kNN-optimal linear map), subsample train for speed
    try:
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        n = min(3000, len(tr))
        ridx = np.random.default_rng(0).choice(len(tr), n, replace=False)
        ncomp = min(128, tr.shape[1])
        nca = NeighborhoodComponentsAnalysis(n_components=ncomp, max_iter=30, random_state=0)
        nca.fit(tr[ridx], trl[ridx])
        a, f, pur = knn_acc(nca.transform(tr), trl, nca.transform(te), tel, C)
        out["knn_nca"], out["pur_nca"] = round(a, 4), round(pur, 3)
    except Exception as e:
        out["knn_nca"], out["pur_nca"] = None, None
    return out


def main():
    files = [f for f in sorted(glob.glob(os.path.join(EMB, "*.npz")))
             if "train_lab" in np.load(f).files]   # skip partial/broken caches
    rows = [run(f) for f in files]
    cols = ["enc", "ds", "C", "base", "knn_raw", "knn_white", "lin_white", "knn_lda", "knn_nca",
            "pur_raw", "pur_white", "pur_nca"]
    print(" | ".join(f"{c:>9}" for c in cols))
    for r in rows:
        print(" | ".join(f"{str(r.get(c)):>9}" for c in cols))

    # read: on each space, does kNN beat base? and on vuln specifically?
    def gain(r, key): return None if r.get(key) is None else round(r[key] - r["base"], 4)
    print("\n=== retrieval gain vs base linear probe (kNN_space - base) ===")
    print(f"{'point':>22} | {'raw':>7} {'white':>7} {'lda':>7} {'nca':>7}  (vuln=binary)")
    for r in rows:
        tag = " *VULN" if r["ds"] in BINARY else ""
        print(f"{r['enc']+'/'+r['ds']:>22} | {str(gain(r,'knn_raw')):>7} {str(gain(r,'knn_white')):>7} "
              f"{str(gain(r,'knn_lda')):>7} {str(gain(r,'knn_nca')):>7}{tag}")

    # mean kNN gain on VULN per space
    print("\n=== mean kNN gain vs base, VULN only ===")
    vuln = [r for r in rows if r["ds"] in BINARY]
    for key in ["knn_raw", "knn_white", "knn_lda", "knn_nca"]:
        gs = [gain(r, key) for r in vuln if gain(r, key) is not None]
        # also vs linear-in-white for the white case (does kNN add over linear there?)
        print(f"  {key:10s} mean gain {np.mean(gs):+.4f}  (wins {sum(g>0 for g in gs)}/{len(gs)})")
    addl = [r["knn_white"] - r["lin_white"] for r in vuln if r.get("lin_white") is not None]
    print(f"  --> kNN_white - linear_white (does retrieval add OVER linear in that space?): "
          f"mean {np.mean(addl):+.4f}  (wins {sum(a>0 for a in addl)}/{len(addl)})")


def _selfcheck():
    # within-class whitening must raise neighbor purity on a 2-class set with strong
    # within-class anisotropy (classes are parallel elongated blobs offset along a thin axis).
    rng = np.random.default_rng(0)
    n, d = 400, 20
    a = rng.normal(0, 1, (n, d)); a[:, 1:] *= 6.0       # huge variance on noise dims
    b = a.copy();
    X = np.vstack([a, b]); y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    X[n:, 0] += 2.0                                      # classes separated only on dim 0
    _, _, pr_raw = knn_acc(X[:n//1], y[:n], X, y, 2) if False else (0,0,0)
    # simple purity check raw vs whitened
    tr, trl = X, y
    def purity(sp):
        return knn_acc(sp, trl, sp, trl, 2)[2]
    mu = tr.mean(0); W = within_class_whiten(tr, trl)
    assert purity((tr - mu) @ W) >= purity(tr) - 1e-9
    print("selfcheck OK: whitening purity %.3f >= raw %.3f" % (purity((tr-mu)@W), purity(tr)))


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
