"""
mcnemar_test.py — paired significance of the patched method on defect-codebert.

Reuses cached outputs (out_defect_codebert_*.npz) + the bundled datastore, so it
runs in seconds. McNemar on discordant pairs, both continuity-corrected chi-square
and exact binomial (the latter is preferred when discordances are few).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import faiss
faiss.omp_set_num_threads(1)
from scipy.stats import chi2, binomtest
from knn_datastore import KNNDatastore
from knn_predictor import KNNPredictor
from calibration import TemperatureScaler, compute_entropy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
def _ds(task):  # datastore under results/CI_kNN_<task>_results/datastore/full
    return os.path.join(REPO, "results", f"CI_kNN_{task}_results", "datastore", "full")

CONFIGS = {
    "defect_codebert":      dict(ncls=4, dstore=_ds("defect_codebert")),
    "vuln_codebert":        dict(ncls=2, dstore=_ds("vuln_codebert")),
    "defect_graphcodebert": dict(ncls=4, dstore=_ds("defect_graphcodebert")),
    "vuln_graphcodebert":   dict(ncls=2, dstore=_ds("vuln_graphcodebert")),
}


def mcnemar(y, pa, pb):
    """H0: pa and pb have equal error rate. b/c = discordant counts."""
    ca, cb = (pa == y), (pb == y)
    b = int(np.sum(ca & ~cb))   # a right, b wrong
    c = int(np.sum(~ca & cb))   # a wrong, b right  (b improvements over a)
    n = b + c
    stat = (abs(b - c) - 1) ** 2 / n if n else 0.0
    p_chi = float(chi2.sf(stat, 1)) if n else 1.0
    p_exact = float(binomtest(min(b, c), n, 0.5).pvalue) if n else 1.0
    return b, c, p_chi, p_exact


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="defect_codebert", choices=list(CONFIGS))
    task = ap.parse_args().task
    cfg = CONFIGS[task]; NCLS = cfg["ncls"]
    dev_split = "dev" if task.startswith("defect") else "valid"

    test = np.load(os.path.join(HERE, f"out_{task}_test.npz"))
    dev = np.load(os.path.join(HERE, f"out_{task}_{dev_split}.npz"))
    y = test["labels"].astype(int)
    n = len(y)

    ts = TemperatureScaler(); ts.fit(dev["logits"], dev["labels"].astype(int))
    test_cal = ts.calibrate(test["logits"])
    ent = compute_entropy(test_cal)

    tr_cache = os.path.join(HERE, f"out_{task}_train.npz")
    if os.path.exists(tr_cache):   # match repro_patch --rebuild_datastore (consistent space)
        ztr = np.load(tr_cache); emb = ztr["emb"].copy(); faiss.normalize_L2(emb)
        ds = KNNDatastore(); ds.index = faiss.IndexFlatL2(emb.shape[1]); ds.index.add(emb)
        ds.labels = ztr["labels"].astype(int); ds.ids = np.arange(len(ds.labels))
        print(f"[datastore] rebuilt from {os.path.basename(tr_cache)}: {ds.index.ntotal} vectors")
    else:
        ds = KNNDatastore(); ds.load(cfg["dstore"])
    priors = np.bincount(ds.labels.astype(int), minlength=NCLS) / len(ds.labels)

    def P(temp, pr):
        return KNNPredictor(ds, num_classes=NCLS, k=8, lambda_val=0.3,
                            knn_temperature=temp, class_priors=priors if pr else None)

    base = test["probs"].argmax(1)
    m1_unpatched = P(10.0, False).predict(test["emb"], test_cal, uncertainty_gated=True, calibrated_entropy=ent)[0].argmax(1)
    m1_patched = P(0.1, True).predict(test["emb"], test_cal, uncertainty_gated=True, calibrated_entropy=ent)[0].argmax(1)

    print(f"{task} test N={n}")
    print(f"  acc  base={np.mean(base==y):.4f}  M1_unpatched={np.mean(m1_unpatched==y):.4f}  M1_patched={np.mean(m1_patched==y):.4f}\n")

    for name, pa, pb in [
        ("base            -> M1_patched (headline)", base, m1_patched),
        ("M1_unpatched    -> M1_patched (do patches help?)", m1_unpatched, m1_patched),
        ("base            -> M1_unpatched (orig method)", base, m1_unpatched),
    ]:
        b, c, p_chi, p_exact = mcnemar(y, pa, pb)
        sig = "***" if p_exact < 1e-3 else "**" if p_exact < 1e-2 else "*" if p_exact < 0.05 else "ns"
        print(f"{name}")
        print(f"    worsened={b}  improved={c}  net=+{c-b}   p_chi2={p_chi:.2e}  p_exact={p_exact:.2e}  [{sig}]")


if __name__ == "__main__":
    main()
