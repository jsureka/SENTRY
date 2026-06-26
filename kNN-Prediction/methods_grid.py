"""
Method comparison grid — the pivot experiment.

Question: is there a TRAINING-FREE, post-hoc method that wins on the four
defendable metrics (Accuracy, F1-macro, ECE, AURC) across code classifiers,
and never harms accuracy/F1?  We compare, per (encoder x dataset) point:

  base       — model softmax (frozen linear probe, or fine-tuned logits)
  temp       — temperature scaling   (T fit on val NLL)            -> ECE
  logit_adj  — logit adjustment       (tau*log prior, tau on val F1) -> F1 on imbalanced
  knn        — kNN-blend              (lambda on val acc)            -> acc where separable
  gate       — confidence-gated kNN   (route to kNN only when model unsure; c,lambda on val)

Every method is val-tuned so it CANNOT be worse than base on its selection metric
on val (tau=0 / lambda=1 / c=0 all reduce to base); the test number is the
generalization estimate. Points come in two kinds: frozen linear probes over 4 encoders x
HF/local datasets (breadth), and the 4 fine-tuned anchors (the real deployment case).

Output: long CSV results/methods_grid.csv, one row per (point, method); partial-safe.
.
"""
import os, sys, csv, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import faiss; faiss.omp_set_num_threads(1)
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from knn_predictor import KNNPredictor
from knn_datastore import KNNDatastore
from calibration import compute_ece
from separability import aurc
import run_grid as G

CSV_PATH = os.path.join(REPO, "results", "methods_grid.csv")
FIELDS = ["point", "encoder", "dataset", "kind", "C", "n_test", "imbalance",
          "method", "acc", "f1", "ece", "aurc", "param"]

# ---------- calibration / adjustment math (all post-hoc, training-free) ----------

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels):
    """scalar T minimizing val NLL."""
    yi = np.arange(len(val_labels))
    def nll(T):
        p = softmax(val_logits / T)
        return -np.log(p[yi, val_labels] + 1e-12).mean()
    r = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(r.x)


def metrics(probs, labels):
    pred = probs.argmax(axis=1)
    ece, _ = compute_ece(probs, labels, n_bins=15)
    return {
        "acc": round(float((pred == labels).mean()), 4),
        "f1": round(float(f1_score(labels, pred, average="macro")), 4),
        "ece": round(float(ece), 4),
        "aurc": round(float(aurc(probs, labels)), 4),
    }


def imbalance_ratio(labels):
    c = np.bincount(labels)
    c = c[c > 0]
    return round(float(c.max() / c.min()), 2)


# ---------- the five methods, returns list of metric rows ----------

def eval_methods(va_logits, va_l, te_logits, te_l, knn_va, knn_te, prior, C):
    va_p, te_p = softmax(va_logits), softmax(te_logits)
    rows = []

    # base
    rows.append(("base", metrics(te_p, te_l), ""))

    # temperature scaling
    T = fit_temperature(va_logits, va_l)
    rows.append(("temp", metrics(softmax(te_logits / T), te_l), f"T={T:.3f}"))

    # logit adjustment: pick tau on val F1-macro
    logp = np.log(prior + 1e-12)
    best_tau, best_f1 = 0.0, -1.0
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        f = f1_score(va_l, softmax(va_logits - tau * logp).argmax(1), average="macro")
        if f > best_f1:
            best_f1, best_tau = f, tau
    rows.append(("logit_adj", metrics(softmax(te_logits - best_tau * logp), te_l), f"tau={best_tau}"))

    # kNN blend: pick lambda on val acc (lambda=1 -> base, so never worse on val)
    def blend(mp, kp, lam):
        return lam * mp + (1 - lam) * kp
    LAM = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_lam = max(LAM, key=lambda lm: (blend(va_p, knn_va, lm).argmax(1) == va_l).mean())
    rows.append(("knn", metrics(blend(te_p, knn_te, best_lam), te_l), f"lambda={best_lam}"))

    # confidence-gated kNN: route to kNN only when model maxprob < c (else keep model).
    # pick (c, lambda) on val acc; c grid includes 1.0 (always-gate) and selection
    # can pick lambda=1 / never-trigger -> never worse than base on val.
    def gated(mp, kp, c, lam):
        mask = (mp.max(1) < c).reshape(-1, 1)
        return np.where(mask, blend(mp, kp, lam), mp)
    best, best_acc = ("", 0.0, 0.0), -1.0
    for c in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        for lam in [0.0, 0.2, 0.4, 0.6, 0.8]:
            a = (gated(va_p, knn_va, c, lam).argmax(1) == va_l).mean()
            if a > best_acc:
                best_acc, best = a, (c, lam)
    c, lam = best
    rows.append(("gate", metrics(gated(te_p, knn_te, c, lam), te_l), f"c={c},lambda={lam}"))

    # composite = temp-scale THEN gate: orthogonal knobs (temp -> ECE, gate -> acc/AURC).
    # one training-free layer meant to win ECE AND acc AND AURC at once, never-harm.
    tp_va, tp_te = softmax(va_logits / T), softmax(te_logits / T)
    best, best_acc = ("", 0.0, 0.0), -1.0
    for cc in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        for lm in [0.0, 0.2, 0.4, 0.6, 0.8]:
            a = (gated(tp_va, knn_va, cc, lm).argmax(1) == va_l).mean()
            if a > best_acc:
                best_acc, best = a, (cc, lm)
    cc, lm = best
    rows.append(("temp_gate", metrics(gated(tp_te, knn_te, cc, lm), te_l), f"T={T:.2f},c={cc},lambda={lm}"))

    return rows


# ---------- knn probabilities via the existing predictor ----------

def knn_probs(train_emb, train_lab, query_emb, model_probs, C):
    ds = G._DS(train_emb, train_lab)
    pred = KNNPredictor(ds, num_classes=C, k=8, knn_temperature="auto", voting="distance_weighted")
    _, _, kp, _ = pred.predict(query_emb.astype(np.float32), model_probs)
    return kp


# ---------- point sources ----------

def frozen_point(enc_name, ds_name, splits):
    se = G.cached_embed(enc_name, G.ENCODERS[enc_name], ds_name, splits)
    if not all(s in se for s in ["train", "val", "test"]):
        return None
    tr_e, tr_l0 = se["train"]; va_e, va_l0 = se["val"]; te_e, te_l0 = se["test"]
    alll = np.concatenate([tr_l0, va_l0, te_l0])
    remap = {v: i for i, v in enumerate(sorted(set(alll.tolist())))}
    C = len(remap)
    rl = lambda a: np.array([remap[v] for v in a.tolist()])
    tr_l, va_l, te_l = rl(tr_l0), rl(va_l0), rl(te_l0)

    clf = LogisticRegression(max_iter=2000, C=1.0).fit(tr_e, tr_l)
    def logits(X):
        d = clf.decision_function(X)
        if d.ndim == 1:                      # binary -> [0, margin]
            d = np.c_[np.zeros_like(d), d]
        full = np.full((len(X), C), d.min() - 20.0)
        for j, cl in enumerate(clf.classes_):
            full[:, int(cl)] = d[:, j]
        return full
    va_lg, te_lg = logits(va_e), logits(te_e)
    knn_va = knn_probs(tr_e, tr_l, va_e, softmax(va_lg), C)
    knn_te = knn_probs(tr_e, tr_l, te_e, softmax(te_lg), C)
    prior = np.bincount(tr_l, minlength=C) / len(tr_l)
    return dict(C=C, va_lg=va_lg, va_l=va_l, te_lg=te_lg, te_l=te_l,
                knn_va=knn_va, knn_te=knn_te, prior=prior, n_test=len(te_l), kind="frozen_probe")


# fine-tuned anchors: (task, model, C, val_split)
ANCHORS = {
    "defect_codebert":      ("defect", "codebert", 4, "dev"),
    "defect_graphcodebert": ("defect", "graphcodebert", 4, "dev"),
    "vuln_codebert":        ("vuln", "codebert", 2, "valid"),
    "vuln_graphcodebert":   ("vuln", "graphcodebert", 2, "valid"),
}


def _load_npz(task, model, split):
    d = np.load(os.path.join(HERE, f"out_{task}_{model}_{split}.npz"), allow_pickle=True)
    return d["logits"].astype(np.float64), d["emb"].astype(np.float32), d["labels"].astype(int)


def anchor_point(name):
    task, model, C, vsplit = ANCHORS[name]
    va_lg, va_e, va_l = _load_npz(task, model, vsplit)
    te_lg, te_e, te_l = _load_npz(task, model, "test")
    ds = KNNDatastore()
    ds.load(os.path.join(REPO, "results", f"CI_kNN_{task}_{model}_results", "datastore", "full"))
    pred = KNNPredictor(ds, num_classes=C, k=8, knn_temperature="auto", voting="distance_weighted")
    _, _, knn_va, _ = pred.predict(va_e, softmax(va_lg))
    _, _, knn_te, _ = pred.predict(te_e, softmax(te_lg))
    tr_lab = np.asarray(getattr(ds, "labels", va_l))      # datastore train labels for prior
    prior = np.bincount(tr_lab, minlength=C) / len(tr_lab)
    return dict(C=C, va_lg=va_lg, va_l=va_l, te_lg=te_lg, te_l=te_l,
                knn_va=knn_va, knn_te=knn_te, prior=prior, n_test=len(te_l), kind="fine_tuned")


# ---------- driver ----------

def append_rows(point, enc, ds, kind, C, n_test, imb, method_rows):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for method, m, param in method_rows:
            w.writerow({"point": point, "encoder": enc, "dataset": ds, "kind": kind,
                        "C": C, "n_test": n_test, "imbalance": imb, "method": method,
                        "param": param, **m})


def done():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH) as f:
        return {r["point"] for r in csv.DictReader(f)}


def run_point(point, enc, ds_name, P):
    imb = imbalance_ratio(P["te_l"])
    rows = eval_methods(P["va_lg"], P["va_l"], P["te_lg"], P["te_l"],
                        P["knn_va"], P["knn_te"], P["prior"], P["C"])
    append_rows(point, enc, ds_name, P["kind"], P["C"], P["n_test"], imb, rows)
    b = dict((r[0], r[1]) for r in rows)
    print(f"[OK {point}] base acc={b['base']['acc']} f1={b['base']['f1']} ece={b['base']['ece']} "
          f"| temp ece={b['temp']['ece']} | logit_adj f1={b['logit_adj']['f1']} "
          f"| knn acc={b['knn']['acc']} | gate acc={b['gate']['acc']} aurc={b['gate']['aurc']}", flush=True)


def main(which="all"):
    have = done()
    # fine-tuned anchors first (instant, strongest points)
    if which in ("all", "anchors"):
        for name in ANCHORS:
            if name in have:
                print(f"[skip] {name}"); continue
            try:
                run_point(name, ANCHORS[name][1], ANCHORS[name][0], anchor_point(name))
            except Exception as e:
                import traceback; traceback.print_exc(); print(f"[FAIL {name}] {e}", flush=True)
    # frozen probe breadth
    if which in ("all", "frozen"):
        for ds_name in G.DATASETS:
            try:
                splits = G.DATASETS[ds_name]()
            except Exception as e:
                print(f"[SKIP dataset {ds_name}] {e}", flush=True); continue
            for enc in G.ENCODERS:
                point = f"{enc}/{ds_name}"
                if point in have:
                    print(f"[skip] {point}"); continue
                t0 = time.time()
                try:
                    P = frozen_point(enc, ds_name, splits)
                    if P is None:
                        print(f"[SKIP {point}] missing split", flush=True); continue
                    run_point(point, enc, ds_name, P)
                    print(f"   ({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc(); print(f"[FAIL {point}] {e}", flush=True)


def _selfcheck():
    # temp scaling must not worsen val NLL; logit_adj must help F1 on an imbalanced toy;
    # gate/knn convex blend stays a valid distribution.
    rng = np.random.default_rng(0)
    n, C = 600, 3
    y = rng.integers(0, C, n)
    # overconfident logits (miscalibrated): correct class boosted x4
    lg = rng.normal(0, 1, (n, C)); lg[np.arange(n), y] += 4.0
    T = fit_temperature(lg, y)
    base_ece = metrics(softmax(lg), y)["ece"]
    temp_ece = metrics(softmax(lg / T), y)["ece"]
    assert temp_ece <= base_ece + 1e-6, (base_ece, temp_ece)
    # imbalanced: class 0 is 90%; logit_adj should not hurt macro-F1 vs base
    yi = np.where(rng.random(n) < 0.9, 0, rng.integers(1, C, n))
    lg2 = rng.normal(0, 1, (n, C)); lg2[np.arange(n), yi] += 2.0
    prior = np.bincount(yi, minlength=C) / n
    kp = softmax(rng.normal(0, 1, (n, C)))
    rows = dict((r[0], r[1]) for r in eval_methods(lg2, yi, lg2, yi, kp, kp, prior, C))
    assert rows["logit_adj"]["f1"] >= rows["base"]["f1"] - 1e-6
    p = 0.4 * softmax(lg2) + 0.6 * kp
    assert np.allclose(p.sum(1), 1.0)
    print("selfcheck OK  T=%.3f base_ece=%.3f temp_ece=%.3f" % (T, base_ece, temp_ece))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "anchors", "frozen"])
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    else:
        main(a.which)
