"""
Stage-0 pilot / GO-NO-GO. Uses ONLY existing cached artifacts (the 4 fine-tuned
combos: {defect,vuln} x {codebert,graphcodebert}). Zero recompute, zero credits.

For each combo it measures, with no datastore rebuild:
  - base_acc, base_ece, base_aurc  (cached test probs)
  - delta = retrieval gain = best-lambda kNN-blend test acc - base acc
            (lambda picked on the VAL split to avoid leakage; raw retrieval
             utility — NO confidence guard, NO class priors, NO P4)
  - S1-S4 separability on the VAL embeddings (a-priori, no datastore)

Then prints whether the separability scores rank the combos in the same order as
the retrieval gain. This is the cheapest possible test of the SENTRY hypothesis;
if S does not track delta even here, stop before spending anything.
"""
import os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import faiss; faiss.omp_set_num_threads(1)  # macOS libomp deadlock guard
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
import sys; sys.path.insert(0, HERE)

from knn_datastore import KNNDatastore
from knn_predictor import KNNPredictor
from calibration import compute_ece
from separability import all_scores, aurc

# combo -> (task, model, num_classes, val_split_name)
COMBOS = {
    "defect_codebert":      ("defect", "codebert",      4, "dev"),
    "defect_graphcodebert": ("defect", "graphcodebert", 4, "dev"),
    "vuln_codebert":        ("vuln",   "codebert",      2, "valid"),
    "vuln_graphcodebert":   ("vuln",   "graphcodebert", 2, "valid"),
}
LAMBDAS = [0.1, 0.3, 0.5, 0.7, 0.9]
K = 8


def load_split(task, model, split):
    f = os.path.join(HERE, f"out_{task}_{model}_{split}.npz")
    d = np.load(f, allow_pickle=True)
    return d["emb"], d["probs"], d["labels"].astype(int)


def blend_acc(predictor, query_emb, model_probs, labels, lam):
    predictor.lambda_val = lam
    final, preds, _, _ = predictor.predict(query_emb, model_probs)
    return (preds == labels).mean()


def run_combo(name):
    task, model, C, val_split = COMBOS[name]
    val_emb, val_probs, val_lab = load_split(task, model, val_split)
    test_emb, test_probs, test_lab = load_split(task, model, "test")

    ds = KNNDatastore()
    ds.load(os.path.join(REPO, "results", f"CI_kNN_{task}_{model}_results", "datastore", "full"))
    pred = KNNPredictor(ds, num_classes=C, k=K, knn_temperature="auto",
                        voting="distance_weighted")  # raw retrieval, no guards

    base_acc = float((test_probs.argmax(1) == test_lab).mean())

    # pick lambda on VAL, apply to TEST (no leakage)
    val_accs = {lam: blend_acc(pred, val_emb, val_probs, val_lab, lam) for lam in LAMBDAS}
    best_lam = max(val_accs, key=val_accs.get)
    test_blend_acc = float(blend_acc(pred, test_emb, test_probs, test_lab, best_lam))
    # also knn-only (lambda=0) test acc, for context
    knn_only_acc = float(blend_acc(pred, test_emb, test_probs, test_lab, 0.0))

    delta = test_blend_acc - base_acc
    scores = all_scores(val_emb, val_lab, k=K)
    base_ece, _ = compute_ece(test_probs, test_lab, n_bins=15)
    base_aurc = aurc(test_probs, test_lab)

    return {
        "combo": name, "C": C, "n_test": int(len(test_lab)),
        "base_acc": round(base_acc, 4),
        "best_lambda": best_lam,
        "knn_blend_acc": round(test_blend_acc, 4),
        "knn_only_acc": round(knn_only_acc, 4),
        "delta_retrieval_gain": round(delta, 4),
        "base_ece": round(float(base_ece), 4),
        "base_aurc": round(base_aurc, 4),
        **{k: round(v, 4) for k, v in scores.items()},
    }


def main():
    rows = [run_combo(c) for c in COMBOS]
    print("\n" + "=" * 100)
    cols = ["combo", "base_acc", "knn_blend_acc", "delta_retrieval_gain",
            "S1_local_label_consistency", "S2_fisher_ratio", "S3_silhouette",
            "S4_mahalanobis_margin", "base_ece", "base_aurc"]
    print(" | ".join(f"{c[:18]:>18}" for c in cols))
    for r in rows:
        print(" | ".join(f"{str(r[c]):>18}" for c in cols))

    # GO/NO-GO read: does each S rank-correlate with delta across the 4 points?
    from scipy.stats import spearmanr
    delta = np.array([r["delta_retrieval_gain"] for r in rows])
    print("\nSpearman rho(S, delta) over the 4 fine-tuned points (sign should be +):")
    for s in ["S1_local_label_consistency", "S2_fisher_ratio", "S3_silhouette", "S4_mahalanobis_margin"]:
        sv = np.array([r[s] for r in rows])
        rho, _ = spearmanr(sv, delta)
        print(f"  {s:30s} rho={rho:+.3f}   values={np.round(sv,3).tolist()}")
    print(f"  delta values = {np.round(delta,4).tolist()}")

    out = os.path.join(REPO, "results", "stage0_pilot.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {out}")
    print("\nNOTE: 4 points cannot do partial correlation; this only checks the SIGN/ranking. "
          "The base-accuracy control + regression need the frozen-probe grid (Stage 1).")


if __name__ == "__main__":
    main()
