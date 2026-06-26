"""
Stage-1 frozen-probe grid. For each (encoder x dataset):
  - extract FROZEN mean-pooled embeddings (MPS), cached to grid_emb/*.npz
  - linear probe (logistic regression) on train -> the "model" probs
  - kNN datastore over train embeddings; retrieval gain delta = best-lambda
    (val-selected) blend test acc - probe test acc  (raw retrieval, no guards)
  - S1-S4 separability on val embeddings; base ECE, AURC
  -> one row appended to results/grid_results.csv (partial progress survives)

Frozen probes deliberately span a RANGE of accuracy/separability (some encoders
are weak on some tasks) -> decouples S from base-accuracy, which the 4 fine-tuned
points could not.
"""
import os, sys, json, csv, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import faiss; faiss.omp_set_num_threads(1)
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from knn_predictor import KNNPredictor
from calibration import compute_ece
from separability import all_scores, aurc

EMB_DIR = os.path.join(HERE, "grid_emb"); os.makedirs(EMB_DIR, exist_ok=True)
CSV_PATH = os.path.join(REPO, "results", "grid_results.csv")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# caps keep MPS extraction + faiss cheap; stratified subsample
CAP = {"train": 12000, "val": 2000, "test": 2000}
MAXLEN = 256
BATCH = 32

ENCODERS = {
    "codebert":      "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
    "unixcoder":     "microsoft/unixcoder-base",
    "codet5p":       "Salesforce/codet5p-110m-embedding",
}

# ---------- dataset loaders: return {split: (list[str], np.int array)} ----------

def _cap(texts, labels, n, seed=0):
    labels = np.asarray(labels)
    if len(texts) <= n:
        return texts, labels
    rng = np.random.default_rng(seed)
    # stratified-ish: sample proportionally per class
    idx = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        take = max(1, int(round(n * len(ci) / len(labels))))
        idx.extend(rng.choice(ci, min(take, len(ci)), replace=False).tolist())
    idx = np.array(idx[:n])
    return [texts[i] for i in idx], labels[idx]


def _relabel(labels):
    labels = np.asarray(labels)
    uniq = {v: i for i, v in enumerate(sorted(set(labels.tolist())))}
    return np.array([uniq[v] for v in labels.tolist()]), len(uniq)


def load_local_jsonl(task):
    """CodeChef (input/label) or Devign (func/target) local splits."""
    base = {"defect": os.path.join(REPO, "Defect-Prediction", "dataset"),
            "vuln":   os.path.join(REPO, "Vulnerability-Detection", "dataset")}[task]
    names = {"defect": {"train": "train.jsonl", "val": "dev.jsonl", "test": "test.jsonl"},
             "vuln":   {"train": "train.jsonl", "val": "valid.jsonl", "test": "test.jsonl"}}[task]
    out = {}
    for split, fn in names.items():
        texts, labels = [], []
        with open(os.path.join(base, fn)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                texts.append(js.get("func", js.get("input", "")))
                labels.append(int(js.get("target", js.get("label", 0))))
        out[split] = (texts, np.array(labels))
    return out


def load_hf(hf_id, code_field, label_field, split_map):
    from datasets import load_dataset
    ds = load_dataset(hf_id)
    out = {}
    for split, hfsplit in split_map.items():
        if hfsplit not in ds:
            continue
        d = ds[hfsplit]
        texts = [str(x) for x in d[code_field]]
        labels = [int(x) for x in d[label_field]]
        out[split] = (texts, np.array(labels))
    # if no val, carve from train
    if "val" not in out and "train" in out:
        t, l = out["train"]
        rng = np.random.default_rng(0); n = len(t); idx = rng.permutation(n)
        cut = int(0.85 * n)
        out["train"] = ([t[i] for i in idx[:cut]], l[idx[:cut]])
        out["val"] = ([t[i] for i in idx[cut:]], l[idx[cut:]])
    return out


def load_poj104_reslit():
    """POJ-104 CodeXGLUE splits have DISJOINT problem-classes (train 1-64, val
    65-80, test 81-104) for retrieval eval — unusable as classification. Pool all
    and stratified-resplit so every one of the 104 classes appears in each split.
    This is the high-separability, high-retrieval-gain anchor."""
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    ds = load_dataset("google/code_x_glue_cc_clone_detection_poj104")
    texts, labels = [], []
    for s in ds:
        texts += [str(x) for x in ds[s]["code"]]
        labels += [int(x) for x in ds[s]["label"]]
    labels = np.array(labels)
    idx = np.arange(len(texts))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=0, stratify=labels)
    va, te = train_test_split(tmp, test_size=0.50, random_state=0, stratify=labels[tmp])
    pick = lambda ix: ([texts[i] for i in ix], labels[ix])
    return {"train": pick(tr), "val": pick(va), "test": pick(te)}


def _cwe1(c):
    """PrimeVul cwe field may be a list or a string; take the first CWE id."""
    if isinstance(c, (list, tuple)):
        return str(c[0]) if len(c) else "NONE"
    return str(c)


def load_primevul_cwe(topk=15):
    """Recent (ICSE'24) MULTICLASS structured task: among VULNERABLE PrimeVul funcs,
    classify which CWE type. Keeps the top-k CWE classes. Tests whether the dichotomy
    is about representational STRUCTURE, not the security domain per se."""
    import collections
    from datasets import load_dataset
    ds = load_dataset("ASSERT-KTH/PrimeVul")
    tr = ds["train_unpaired"]
    cwes = [_cwe1(c) for c, v in zip(tr["cwe"], tr["is_vulnerable"]) if v]
    top = [c for c, _ in collections.Counter(cwes).most_common(topk) if c != "NONE"]
    idx = {c: i for i, c in enumerate(top)}
    out = {}
    for split, hf in {"train": "train_unpaired", "val": "valid_unpaired", "test": "test_unpaired"}.items():
        d = ds[hf]; texts, labs = [], []
        for func, v, c in zip(d["func"], d["is_vulnerable"], d["cwe"]):
            cc = _cwe1(c)
            if v and cc in idx:
                texts.append(str(func)); labs.append(idx[cc])
        out[split] = (texts, np.array(labs))
    return out


DATASETS = {
    "codechef":  lambda: load_local_jsonl("defect"),
    "devign":    lambda: load_local_jsonl("vuln"),
    "poj104":    load_poj104_reslit,
    "primevul":  lambda: load_hf("ASSERT-KTH/PrimeVul", "func", "is_vulnerable",
                                 {"train": "train_unpaired", "val": "valid_unpaired", "test": "test_unpaired"}),
    "reveal":    lambda: load_hf("claudios/reveal", "functionSource", "label",
                                 {"train": "train", "val": "validation", "test": "test"}),
    "diversevul": lambda: load_hf("bstee615/DiverseVul", "func", "target",
                                  {"train": "train", "val": "validation", "test": "test"}),
    # primevul_cwe (load_primevul_cwe) intentionally NOT registered: CWE classes are
    # long-tailed/degenerate (one class dominates, some test classes empty). Not a clean
    # multiclass testbed. Kept the loader for reference only.
}

# ---------- frozen embedding extraction ----------

def get_encoder(hf_id):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(DEVICE).eval()
    return tok, model


def embed(tok, model, texts):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = [" ".join(t.split()) for t in texts[i:i + BATCH]]
            enc = tok(batch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:  # codet5p embedding model returns a pooled tensor directly
                vec = out if torch.is_tensor(out) else out[0]
            embs.append(vec.float().cpu().numpy())
    return np.concatenate(embs, 0).astype(np.float32)


def cached_embed(enc_name, hf_id, ds_name, splits):
    cache = os.path.join(EMB_DIR, f"{enc_name}__{ds_name}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {s: (d[f"{s}_emb"], d[f"{s}_lab"]) for s in ["train", "val", "test"] if f"{s}_emb" in d}
    tok, model = get_encoder(hf_id)
    save, ret = {}, {}
    for s in ["train", "val", "test"]:
        if s not in splits:
            continue
        texts, labels = splits[s]
        texts, labels = _cap(texts, labels, CAP[s])
        e = embed(tok, model, texts)
        ret[s] = (e, labels)
        save[f"{s}_emb"] = e; save[f"{s}_lab"] = labels
    np.savez_compressed(cache, **save)
    del model; torch.mps.empty_cache() if DEVICE == "mps" else None
    return ret


# ---------- per-point evaluation ----------

def linear_probe(train_e, train_l, eval_e):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(train_e, train_l)
    # map decision to full class space (probe classes may miss a rare class)
    proba = clf.predict_proba(eval_e)
    return proba, clf.classes_


def knn_blend_acc(predictor, q_emb, model_probs, labels, lam):
    predictor.lambda_val = lam
    _, preds, _, _ = predictor.predict(q_emb.astype(np.float32), model_probs)
    return float((preds == labels).mean())


class _DS:  # minimal datastore wrapper compatible with KNNPredictor.search
    def __init__(self, emb, labels):
        emb = emb.astype(np.float32).copy(); faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatL2(emb.shape[1]); self.index.add(emb)
        self.labels = np.asarray(labels); self.ids = np.arange(len(labels))
    def search(self, q, k=8):
        q = q.astype(np.float32).copy(); faiss.normalize_L2(q)
        d, i = self.index.search(q, k)
        return d, self.labels[i], self.ids[i]


def eval_point(enc_name, ds_name, splits_emb):
    tr_e, tr_l0 = splits_emb["train"]
    va_e, va_l0 = splits_emb["val"]
    te_e, te_l0 = splits_emb["test"]
    # unify label space across splits
    alll = np.concatenate([tr_l0, va_l0, te_l0])
    _, C = _relabel(alll)
    remap = {v: i for i, v in enumerate(sorted(set(alll.tolist())))}
    tr_l = np.array([remap[v] for v in tr_l0.tolist()])
    va_l = np.array([remap[v] for v in va_l0.tolist()])
    te_l = np.array([remap[v] for v in te_l0.tolist()])

    # probe (model probs) on val + test, expanded to C columns
    def expand(proba, classes):
        full = np.zeros((proba.shape[0], C))
        for j, c in enumerate(classes):
            full[:, int(c)] = proba[:, j]
        s = full.sum(1, keepdims=True); return full / np.clip(s, 1e-12, None)
    va_proba, cl = linear_probe(tr_e, tr_l, va_e); va_probs = expand(va_proba, cl)
    te_proba, cl = linear_probe(tr_e, tr_l, te_e); te_probs = expand(te_proba, cl)

    base_acc = float((te_probs.argmax(1) == te_l).mean())
    ds = _DS(tr_e, tr_l)
    pred = KNNPredictor(ds, num_classes=C, k=8, knn_temperature="auto", voting="distance_weighted")
    LAM = [0.1, 0.3, 0.5, 0.7, 0.9]
    best = max(LAM, key=lambda lm: knn_blend_acc(pred, va_e, va_probs, va_l, lm))
    knn_acc = knn_blend_acc(pred, te_e, te_probs, te_l, best)
    knn_only = knn_blend_acc(pred, te_e, te_probs, te_l, 0.0)

    scores = all_scores(va_e, va_l, k=8)
    ece, _ = compute_ece(te_probs, te_l, n_bins=15)
    return {
        "point": f"{enc_name}/{ds_name}", "encoder": enc_name, "dataset": ds_name,
        "kind": "frozen_probe", "C": C, "n_train": len(tr_l), "n_test": len(te_l),
        "base_acc": round(base_acc, 4), "best_lambda": best,
        "knn_blend_acc": round(knn_acc, 4), "knn_only_acc": round(knn_only, 4),
        "delta_retrieval_gain": round(knn_acc - base_acc, 4),
        "base_ece": round(float(ece), 4), "base_aurc": round(aurc(te_probs, te_l), 4),
        **{k: round(v, 4) for k, v in scores.items()},
    }


FIELDS = ["point", "encoder", "dataset", "kind", "C", "n_train", "n_test", "base_acc",
          "best_lambda", "knn_blend_acc", "knn_only_acc", "delta_retrieval_gain",
          "base_ece", "base_aurc", "S1_local_label_consistency", "S2_fisher_ratio",
          "S3_silhouette", "S4_mahalanobis_margin"]


def append_row(row):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def done_points():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH) as f:
        return {r["point"] for r in csv.DictReader(f)}


def main(encoders=None, datasets=None):
    encoders = encoders or list(ENCODERS)
    datasets = datasets or list(DATASETS)
    have = done_points()
    for ds_name in datasets:
        try:
            splits = DATASETS[ds_name]()
        except Exception as e:
            print(f"[SKIP dataset {ds_name}] load failed: {e}"); continue
        for enc_name in encoders:
            point = f"{enc_name}/{ds_name}"
            if point in have:
                print(f"[skip cached row] {point}"); continue
            t0 = time.time()
            try:
                splits_emb = cached_embed(enc_name, ENCODERS[enc_name], ds_name, splits)
                if not all(s in splits_emb for s in ["train", "val", "test"]):
                    print(f"[SKIP {point}] missing split: {list(splits_emb)}"); continue
                row = eval_point(enc_name, ds_name, splits_emb)
                append_row(row)
                print(f"[OK {point}] acc={row['base_acc']} d={row['delta_retrieval_gain']} "
                      f"S2={row['S2_fisher_ratio']} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[FAIL {point}] {e}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    main(ap.parse_args().encoders, ap.parse_args().datasets)
