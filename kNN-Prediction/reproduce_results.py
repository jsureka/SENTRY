"""
repro_patch.py — Reproduce baselines + measure patches for a (task, model) combo.

Usage: python repro_patch.py --task vuln_codebert | defect_codebert

Validation gate = B1 (pure model softmax). Accuracy is config-independent, so a
matching B1 accuracy proves the checkpoint loaded correctly and every downstream
number is trustworthy — INCLUDING ECE, which is what we are auditing here.
Patched vs unpatched is compared WITHIN one run (apples-to-apples).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.set_num_threads(4)
import faiss
faiss.omp_set_num_threads(1)
from torch.utils.data import DataLoader, SequentialSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from scipy.optimize import minimize
from scipy.special import softmax, log_softmax
from sklearn.metrics import f1_score, matthews_corrcoef

from knn_datastore import KNNDatastore, CodeDataset
from knn_predictor import KNNPredictor
from calibration import TemperatureScaler, compute_entropy, compute_ece, compute_brier_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

# Paths are relative to the SENTRY repo root. Model checkpoints live in models/
# (gitignored, ~500M each); datastores + tables live under results/CI_kNN_<task>_results/.
def _cfg(ncls, arch, tasktype, dev):
    task = f"{tasktype}_{arch}"                      # e.g. defect_codebert
    rdir = os.path.join(REPO, "results", f"CI_kNN_{task}_results")
    data = "Vulnerability-Detection" if tasktype == "vuln" else "Defect-Prediction"
    return dict(
        ncls=ncls, model=f"microsoft/{arch}-base",
        ckpt=os.path.join(REPO, "models", f"{arch}_{tasktype}_model.bin"),
        data=os.path.join(REPO, data, "dataset"), dev=dev,
        dstore=os.path.join(rdir, "datastore", "full"),
        table=os.path.join(rdir, f"final_table_{task}.json"))

CONFIGS = {
    "vuln_codebert":        _cfg(2, "codebert",      "vuln",   "valid"),
    "defect_codebert":      _cfg(4, "codebert",      "defect", "dev"),
    "defect_graphcodebert": _cfg(4, "graphcodebert", "defect", "dev"),
    "vuln_graphcodebert":   _cfg(2, "graphcodebert", "vuln",   "valid"),
}


def m(probs, labels, ncls):
    preds = probs.argmax(1)
    ece, _ = compute_ece(probs, labels)
    return dict(acc=float((preds == labels).mean()),
                f1=float(f1_score(labels, preds, average="macro", zero_division=0)),
                mcc=float(matthews_corrcoef(labels, preds)),
                ece=float(ece), brier=float(compute_brier_score(probs, labels, ncls)),
                meanconf=float(probs.max(1).mean()))


def temp_recalibrate(dev_blend, dev_labels, test_blend):
    eps = 1e-12
    dl, tl = np.log(dev_blend + eps), np.log(test_blend + eps)

    def nll(T):
        T = max(T[0], 0.01)
        lp = log_softmax(dl / T, axis=1)
        return -np.mean(lp[np.arange(len(dev_labels)), dev_labels])
    T = max(minimize(nll, x0=[1.0], method="Nelder-Mead").x[0], 0.01)
    return softmax(tl / T, axis=1), float(T)


def selective(score, correct):
    order = np.argsort(-score)
    c = correct[order].astype(float)
    acc_at = np.cumsum(c) / np.arange(1, len(c) + 1)
    at = lambda p: float(acc_at[max(int(p * len(c)) - 1, 0)])
    return {"acc@50%": round(at(0.5), 4), "acc@80%": round(at(0.8), 4), "acc@100%": round(float(c.mean()), 4)}


def load_model(ckpt, ncls, device, model_name):
    cfg = RobertaConfig.from_pretrained(model_name); cfg.num_labels = ncls
    tok = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, config=cfg)
    sd = torch.load(ckpt, map_location=device)
    sd = {k[len("encoder."):] if k.startswith("encoder.") else k: v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)} (expect 0/0)")
    return model.to(device).eval(), tok


def extract(model, tok, data_dir, split, task, device, batch=16):
    cache = os.path.join(HERE, f"out_{task}_{split}.npz")
    if os.path.exists(cache):
        print(f"[cache] {split} <- {os.path.basename(cache)}")
        z = np.load(cache); return {k: z[k] for k in z.files}
    from tqdm import tqdm
    ds = CodeDataset(os.path.join(data_dir, f"{split}.jsonl"), tok, 400)
    dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=batch)
    L, E, Y = [], [], []
    with torch.no_grad():
        for b in tqdm(dl, desc=f"extract {split}"):
            ids = b[0].to(device)
            out = model(ids, attention_mask=ids.ne(1), output_hidden_states=True)
            L.append(out.logits.cpu().numpy())
            E.append(out.hidden_states[-1][:, 0, :].cpu().numpy())
            Y.extend(b[1].numpy().tolist())
    o = {"logits": np.concatenate(L), "emb": np.concatenate(E).astype("float32"), "labels": np.array(Y)}
    o["probs"] = softmax(o["logits"], axis=1)
    np.savez(cache, **o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(CONFIGS))
    ap.add_argument("--rebuild_datastore", action="store_true",
                    help="build kNN datastore from THIS run's own train extraction "
                         "(use when the saved datastore is in a different embedding space)")
    args = ap.parse_args()
    task = args.task
    cfg = CONFIGS[task]
    ncls, dev = cfg["ncls"], cfg["dev"]
    device = torch.device("cpu")

    model, tok = load_model(cfg["ckpt"], ncls, device, cfg["model"])
    o_dev = extract(model, tok, cfg["data"], dev, task, device)
    o_test = extract(model, tok, cfg["data"], "test", task, device)
    test_probs, test_logits, test_emb, test_labels = o_test["probs"], o_test["logits"], o_test["emb"], o_test["labels"].astype(int)
    dev_logits, dev_emb, dev_labels = o_dev["logits"], o_dev["emb"], o_dev["labels"].astype(int)

    rows = []
    b1 = m(test_probs, test_labels, ncls); rows.append(("B1 model-only", b1))

    saved = {r["Method"]: r for r in json.load(open(cfg["table"]))}.get("B1: Model-Only", {})
    print(f"\n=== VALIDATION GATE: B1 vs saved  (mean_conf={b1['meanconf']:.3f}, acc={b1['acc']:.3f}) ===")
    for key, sk in [("acc", "Acc"), ("ece", "ECE"), ("brier", "Brier")]:
        sv = saved.get(sk, "--")
        d = abs(b1[key] - sv) if isinstance(sv, (int, float)) else float("nan")
        print(f"  {key:<6} mine={b1[key]:.4f}  saved={sv if isinstance(sv,(int,float)) else 'NA':<8}  Δ={d:.4f}")
    acc_ok = isinstance(saved.get("Acc"), (int, float)) and abs(b1["acc"] - saved["Acc"]) < 0.012
    print(f"  GATE (accuracy): {'PASS ✓ load correct' if acc_ok else 'FAIL ✗'}")

    ts = TemperatureScaler(); ts.fit(dev_logits, dev_labels)
    test_cal, dev_cal = ts.calibrate(test_logits), ts.calibrate(dev_logits)
    rows.append((f"B3 +temp T={ts.temperature:.2f}", m(test_cal, test_labels, ncls)))

    if args.rebuild_datastore:
        o_tr = extract(model, tok, cfg["data"], "train", task, device, batch=32)
        emb = o_tr["emb"].copy(); faiss.normalize_L2(emb)
        ds = KNNDatastore(); ds.index = faiss.IndexFlatL2(emb.shape[1]); ds.index.add(emb)
        ds.labels = o_tr["labels"].astype(int); ds.ids = np.arange(len(ds.labels))
        print(f"[datastore] REBUILT from own train extraction: {ds.index.ntotal} vectors")
    else:
        ds = KNNDatastore(); ds.load(cfg["dstore"])
    priors = np.bincount(ds.labels.astype(int), minlength=ncls) / len(ds.labels)
    print(f"[datastore] {len(ds.labels)} vectors, class freq={priors.round(3)}")

    def P(temp, use_priors):
        return KNNPredictor(ds, num_classes=ncls, k=8, lambda_val=0.3,
                            knn_temperature=temp, class_priors=priors if use_priors else None)

    rows.append(("B4 kNN (unpatched T=10)", m(P(10.0, False).predict(test_emb, test_probs)[0], test_labels, ncls)))
    rows.append(("B4 kNN (P1 T=0.1 +P3)", m(P(0.1, True).predict(test_emb, test_probs)[0], test_labels, ncls)))

    ent = compute_entropy(test_cal)
    rows.append(("M1 OURS (unpatched)", m(P(10.0, False).predict(test_emb, test_cal, uncertainty_gated=True, calibrated_entropy=ent)[0], test_labels, ncls)))
    pp = P(0.1, True)
    m1_test, _, _, det = pp.predict(test_emb, test_cal, uncertainty_gated=True, calibrated_entropy=ent)
    m1_dev = pp.predict(dev_emb, dev_cal, uncertainty_gated=True, calibrated_entropy=compute_entropy(dev_cal))[0]
    rows.append(("M1+ (P1+P3)", m(m1_test, test_labels, ncls)))
    m1_recal, T4 = temp_recalibrate(m1_dev, dev_labels, m1_test)
    rows.append((f"M1++ (P1+P3+P4 recal T={T4:.2f})", m(m1_recal, test_labels, ncls)))

    print("\n" + "=" * 86)
    print(f"{'Method':<34}{'Acc':>8}{'F1':>8}{'MCC':>8}{'ECE':>8}{'Brier':>9}{'Conf':>8}")
    print("-" * 86)
    for name, r in rows:
        print(f"{name:<34}{r['acc']:>8.4f}{r['f1']:>8.4f}{r['mcc']:>8.4f}{r['ece']:>8.4f}{r['brier']:>9.4f}{r['meanconf']:>8.3f}")
    print("=" * 86)

    correct = (m1_recal.argmax(1) == test_labels)
    print("\n=== P2 selective prediction (keep most-reliable first) ===")
    print("  by model confidence :", selective(test_cal.max(1), correct))
    print("  by retrieval reliab.:", selective(det["reliability"], correct))

    json.dump({n: r for n, r in rows}, open(os.path.join(HERE, f"patch_results_{task}.json"), "w"), indent=2)
    print(f"\nsaved patch_results_{task}.json")


if __name__ == "__main__":
    main()
