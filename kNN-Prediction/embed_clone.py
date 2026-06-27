"""
Clone detection (BigCloneBench) as a frozen-embedding point for the grid.

Clone detection is BINARY but HIGH-separability — the critical test of whether the
dichotomy axis is task-arity (multiclass vs binary) or REPRESENTABILITY. If retrieval
helps here (binary, separable), the failure on vuln is about separability, not arity.

Pair task: embed each function once with the frozen encoder, then build a pair
feature [e1, e2, |e1-e2|, e1*e2] (standard CodeXGLUE clone representation). Saves a
cache shaped exactly like the other grid datasets so methods_grid / reliability
harnesses read it unchanged.

Subsampled + balanced for tractability..
Run: python kNN-Prediction/embed_clone.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from run_grid import get_encoder, embed, ENCODERS, EMB_DIR

N = {"train": 8000, "val": 2000, "test": 2000}     # pairs per split
RNG = np.random.default_rng(0)


def balanced_pairs(split, n):
    """Return (func1[], func2[], label[]) balanced 50/50, n total."""
    f1, f2, y = [str(x) for x in split["func1"]], [str(x) for x in split["func2"]], np.array(split["label"]).astype(int)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    k = min(n // 2, len(pos), len(neg))
    idx = np.concatenate([RNG.choice(pos, k, replace=False), RNG.choice(neg, k, replace=False)])
    RNG.shuffle(idx)
    return [f1[i] for i in idx], [f2[i] for i in idx], y[idx]


def pair_feats(e1, e2):
    return np.concatenate([e1, e2, np.abs(e1 - e2), e1 * e2], axis=1).astype(np.float32)


def main():
    from datasets import load_dataset
    ds = load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench")
    smap = {"train": "train", "val": "validation" if "validation" in ds else "valid", "test": "test"}
    data = {s: balanced_pairs(ds[smap[s]], N[s]) for s in N}

    # unique function corpus across all splits -> embed once
    uniq = {}
    for s in data:
        for f in data[s][0] + data[s][1]:
            uniq.setdefault(f, len(uniq))
    corpus = [None] * len(uniq)
    for f, i in uniq.items():
        corpus[i] = f
    print(f"pairs: { {s: len(data[s][2]) for s in data} } | unique funcs: {len(corpus)}")

    for enc, hf_id in ENCODERS.items():
        cache = os.path.join(EMB_DIR, f"{enc}__bigclone.npz")
        if os.path.exists(cache):
            print(f"skip {enc} (cached)"); continue
        tok, model = get_encoder(hf_id)
        E = embed(tok, model, corpus)                      # (n_unique, d)
        save = {}
        for s in data:
            f1, f2, y = data[s]
            e1 = E[[uniq[f] for f in f1]]; e2 = E[[uniq[f] for f in f2]]
            save[f"{s}_emb"] = pair_feats(e1, e2); save[f"{s}_lab"] = y
        np.savez_compressed(cache, **save)
        print(f"saved {cache}  dim={save['train_emb'].shape[1]}")
        del model
        try:
            import torch; torch.mps.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
