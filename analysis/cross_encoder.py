"""
Fresh angle: 4 frozen code encoders embed the SAME code. Used one-at-a-time so far.
Test two training-free, multi-encoder hypotheses (no retraining):

  H1 (free accuracy): a soft-vote ensemble of per-encoder linear probes beats the best
      single encoder on acc/F1.
  H2 (free reliability): cross-encoder DISAGREEMENT (how many encoders dissent from the
      ensemble vote) is a better abstention signal than any single encoder's softmax
      confidence -> lower selective-prediction AURC.

Reads cached kNN-Prediction/grid_emb/{enc}__{ds}.npz (same test samples across encoders,
deterministic caps). Prints per-dataset + aggregate.
"""
import os, glob
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "kNN-Prediction"))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from calibration import compute_ece
from separability import aurc

EMB = os.path.join(HERE, "kNN-Prediction", "grid_emb")
ENCODERS = ["codebert", "graphcodebert", "unixcoder", "codet5p"]


def aurc_score(correct, score):
    order = np.argsort(-score); c = correct[order].astype(float)
    n = np.arange(1, len(c) + 1)
    return float(np.trapz(np.cumsum(1 - c) / n, n / len(c)))


def probe_probs(ds):
    """train a probe per encoder on its embeddings, return aligned test probs + labels."""
    out, labels, C = {}, None, None
    for enc in ENCODERS:
        f = os.path.join(EMB, f"{enc}__{ds}.npz")
        d = np.load(f)
        if "train_lab" not in d.files:
            return None
        tr, trl = d["train_emb"], d["train_lab"].astype(int)
        te, tel = d["test_emb"], d["test_lab"].astype(int)
        u = sorted(set(trl.tolist()) | set(tel.tolist()))
        m = {v: i for i, v in enumerate(u)}; C = len(u)
        trl = np.array([m[v] for v in trl.tolist()]); tel = np.array([m[v] for v in tel.tolist()])
        clf = LogisticRegression(max_iter=2000).fit(tr, trl)
        p = clf.predict_proba(te)
        full = np.zeros((len(te), C))
        for j, cl in enumerate(clf.classes_):
            full[:, int(cl)] = p[:, j]
        out[enc] = full
        if labels is None:
            labels = tel
        elif len(labels) != len(tel) or not np.array_equal(labels, tel):
            n = min(len(labels), len(tel))           # align if a cap differed
            labels = labels[:n]
            for k in out:
                out[k] = out[k][:n]
    return out, labels, C


def metrics(probs, labels):
    pred = probs.argmax(1)
    ece, _ = compute_ece(probs, labels, n_bins=15)
    return dict(acc=round(float((pred == labels).mean()), 4),
                f1=round(float(f1_score(labels, pred, average="macro")), 4),
                ece=round(float(ece), 4), aurc=round(float(aurc(probs, labels)), 4))


def run(ds):
    r = probe_probs(ds)
    if r is None:
        return None
    probs, y, C = r
    singles = {e: metrics(probs[e], y) for e in ENCODERS}
    best_single = max(singles.values(), key=lambda m: m["acc"])
    ens = np.mean([probs[e] for e in ENCODERS], axis=0)
    ens_m = metrics(ens, y)

    # cross-encoder disagreement: per sample, # encoders whose argmax != ensemble argmax
    ens_pred = ens.argmax(1)
    votes = np.stack([probs[e].argmax(1) for e in ENCODERS], 1)
    agree = (votes == ens_pred[:, None]).sum(1)          # 1..4, higher=more agreement
    correct = (ens_pred == y)
    # abstention AURC: disagreement score vs best single encoder's softmax confidence
    best_enc = max(ENCODERS, key=lambda e: singles[e]["acc"])
    sel_msp = aurc_score((probs[best_enc].argmax(1) == y), probs[best_enc].max(1))
    sel_ens_msp = aurc_score(correct, ens.max(1))
    sel_disagree = aurc_score(correct, agree + ens.max(1) * 0.0)   # pure agreement count
    sel_combo = aurc_score(correct, agree + ens.max(1))            # agreement + ens conf
    return dict(ds=ds, C=C, best_single=best_single, ens=ens_m,
                sel_msp=round(sel_msp, 4), sel_ens_msp=round(sel_ens_msp, 4),
                sel_disagree=round(sel_disagree, 4), sel_combo=round(sel_combo, 4))


def main():
    dss = sorted({os.path.basename(f).split("__")[1][:-4] for f in glob.glob(os.path.join(EMB, "*.npz"))})
    rows = [r for ds in dss if (r := run(ds)) is not None]
    print(f"{'dataset':<10}{'C':>4}{'best1_acc':>10}{'ens_acc':>9}{'Δacc':>8}{'best1_f1':>9}{'ens_f1':>8}{'ens_ece':>9}{'ens_aurc':>9}")
    dacc = []
    for r in rows:
        d = r["ens"]["acc"] - r["best_single"]["acc"]; dacc.append(d)
        print(f"{r['ds']:<10}{r['C']:>4}{r['best_single']['acc']:>10}{r['ens']['acc']:>9}{d:>+8.4f}"
              f"{r['best_single']['f1']:>9}{r['ens']['f1']:>8}{r['ens']['ece']:>9}{r['ens']['aurc']:>9}")
    print(f"\nH1 free-accuracy: ensemble - best single encoder, mean Δacc = {np.mean(dacc):+.4f} "
          f"(wins {sum(d>0 for d in dacc)}/{len(dacc)})")

    print(f"\n{'='*70}\nH2 selective AURC (lower=better): disagreement vs softmax confidence\n{'='*70}")
    print(f"{'dataset':<10}{'best1_msp':>11}{'ens_msp':>9}{'disagree':>10}{'combo':>8}")
    for r in rows:
        print(f"{r['ds']:<10}{r['sel_msp']:>11}{r['sel_ens_msp']:>9}{r['sel_disagree']:>10}{r['sel_combo']:>8}")
    for key, name in [("sel_msp", "best1_msp"), ("sel_ens_msp", "ens_msp"),
                      ("sel_disagree", "disagree"), ("sel_combo", "combo")]:
        print(f"  mean {name:<10} AURC = {np.mean([r[key] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
