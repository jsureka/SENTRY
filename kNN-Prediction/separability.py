"""
Separability scores: cheap, training-free, a-priori measures of how well a
fine-tuned code model's representation separates the task classes.

Hypothesis (SENTRY): retrieval-augmented inference (kNN over training embeddings)
improves classification accuracy IFF the representation separates the classes; a
separability score computed on a held-out labelled split predicts the realized
kNN accuracy gain BEFORE any datastore is built.

All scores take (embeddings, labels) and return a single float, higher = more
separable. Embeddings are L2-normalized internally to match the cosine space the
FAISS datastore uses (knn_datastore.py normalizes both keys and queries).

Scores:
  S1 local_label_consistency  — mean fraction of a point's k nearest *other*
                                 points that share its label (direct kNN-signal
                                 proxy; the only score that touches neighbours).
  S2 fisher_ratio             — trace(between-scatter)/trace(within-scatter);
                                 no neighbours, pure second moments.
  S3 silhouette               — sklearn silhouette on class labels (cosine).
  S4 mahalanobis_margin       — mean (dist to nearest other-class mean − dist to
                                 own-class mean) under a shared precision; the
                                 "deployable" score: needs no kNN, no datastore.

Also: aurc() — area under the risk-coverage curve, the reliability metric used
throughout the grid (lower = better selective behaviour).


"""
import numpy as np


def _l2norm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def local_label_consistency(emb, labels, k=8, normalize=True):
    """S1: mean fraction of k nearest *other* points sharing the label."""
    import faiss
    labels = np.asarray(labels)
    x = _l2norm(emb).astype(np.float32) if normalize else np.asarray(emb, np.float32)
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    # k+1 because the nearest neighbour of a point is itself; drop column 0.
    _, idx = index.search(x, k + 1)
    neigh = idx[:, 1:]
    same = (labels[neigh] == labels[:, None]).mean()
    return float(same)


def fisher_ratio(emb, labels, normalize=True):
    """S2: trace(S_between) / trace(S_within). Higher = classes more separated."""
    labels = np.asarray(labels)
    x = _l2norm(emb) if normalize else np.asarray(emb, np.float64)
    mu = x.mean(axis=0)
    sw = 0.0  # within-class scatter (trace)
    sb = 0.0  # between-class scatter (trace)
    for c in np.unique(labels):
        xc = x[labels == c]
        mc = xc.mean(axis=0)
        sw += ((xc - mc) ** 2).sum()
        sb += len(xc) * ((mc - mu) ** 2).sum()
    return float(sb / max(sw, 1e-12))


def silhouette(emb, labels, normalize=True, max_n=3000, seed=0):
    """S3: silhouette score on class labels (cosine). Sub-sampled for O(n^2) cost."""
    from sklearn.metrics import silhouette_score
    labels = np.asarray(labels)
    x = _l2norm(emb) if normalize else np.asarray(emb, np.float64)
    if len(x) > max_n:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(x), max_n, replace=False)
        x, labels = x[sel], labels[sel]
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(silhouette_score(x, labels, metric="cosine"))


def mahalanobis_margin(emb, labels, normalize=True, shrinkage=1e-3):
    """S4: mean(d_nearest_other_class − d_own_class) under shared precision.

    Deployable score — no neighbours, no datastore. Positive = own-class mean is
    closer than the nearest rival class mean on average (separated); ~0 or
    negative = overlapping.
    """
    labels = np.asarray(labels)
    x = _l2norm(emb) if normalize else np.asarray(emb, np.float64)
    classes = np.unique(labels)
    means = {int(c): x[labels == c].mean(axis=0) for c in classes}
    cov = np.cov(x, rowvar=False)
    cov += shrinkage * np.eye(cov.shape[0])  # regularize before inverting
    prec = np.linalg.inv(cov)

    def md(diff):  # squared Mahalanobis, vectorized over rows
        return np.einsum("ij,jk,ik->i", diff, prec, diff)

    d = {int(c): md(x - means[int(c)]) for c in classes}
    own = np.array([d[int(y)][i] for i, y in enumerate(labels)])
    other = np.full(len(x), np.inf)
    for c in classes:
        mask = labels != c
        other[mask] = np.minimum(other[mask], d[int(c)][mask])
    margin = (other - own)
    scale = np.median(own) + 1e-9  # scale-free: report margin relative to own spread
    return float(np.mean(margin) / scale)


def all_scores(emb, labels, k=8):
    """Compute S1–S4 as a dict."""
    return {
        "S1_local_label_consistency": local_label_consistency(emb, labels, k=k),
        "S2_fisher_ratio": fisher_ratio(emb, labels),
        "S3_silhouette": silhouette(emb, labels),
        "S4_mahalanobis_margin": mahalanobis_margin(emb, labels),
    }


def aurc(probs, labels):
    """Area under risk-coverage curve. Lower = better selective behaviour.

    Sort by confidence desc; risk(c) = error rate over the most-confident
    fraction c; integrate risk over coverage ∈ (0,1].
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    err = (probs.argmax(axis=1) != labels).astype(float)
    order = np.argsort(-conf)
    err = err[order]
    cum_risk = np.cumsum(err) / (np.arange(len(err)) + 1)
    coverage = (np.arange(len(err)) + 1) / len(err)
    return float(np.trapz(cum_risk, coverage))


def _selfcheck():
    """Separable Gaussians must score higher than overlapping ones on every S,
    and a confident-correct model must have lower AURC than a random one."""
    rng = np.random.default_rng(0)
    d, n = 16, 600
    # well-separated: class means 6 apart
    a = rng.normal(0, 1, (n, d)); a[:, 0] += 3
    b = rng.normal(0, 1, (n, d)); b[:, 0] -= 3
    sep_emb = np.vstack([a, b]); sep_lab = np.r_[np.zeros(n), np.ones(n)].astype(int)
    # overlapping: same mean
    a2 = rng.normal(0, 1, (n, d)); b2 = rng.normal(0, 1, (n, d))
    ov_emb = np.vstack([a2, b2]); ov_lab = sep_lab.copy()

    s_sep = all_scores(sep_emb, sep_lab)
    s_ov = all_scores(ov_emb, ov_lab)
    for key in s_sep:
        assert s_sep[key] > s_ov[key], f"{key}: sep {s_sep[key]:.3f} !> overlap {s_ov[key]:.3f}"

    # AURC: confident-correct vs random
    good = np.zeros((200, 2)); good[:, 0] = 0.95; good[:, 1] = 0.05
    lab = np.zeros(200, int)
    rand = rng.uniform(0, 1, (200, 2)); rand /= rand.sum(1, keepdims=True)
    assert aurc(good, lab) < aurc(rand, lab), "AURC: confident-correct should beat random"
    print("separability self-check OK")
    print("  separated:", {k: round(v, 3) for k, v in s_sep.items()})
    print("  overlap:  ", {k: round(v, 3) for k, v in s_ov.items()})


if __name__ == "__main__":
    _selfcheck()
