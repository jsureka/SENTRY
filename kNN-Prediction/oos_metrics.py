"""
OOS (Out-of-Scope) Detection Metrics — CVR / MVR / AUC
======================================================
Computes CodeImprove-compatible metrics from kNN pipeline outputs.

Definitions (matching CodeImprove paper):
- Out-of-scope (OOS): sample where model_pred != true_label
- In-scope (IS):  sample where model_pred == true_label
- CVR (Coverage Ratio ↑): fraction of OOS inputs correctly flagged
- MVR (Misclassification Verification Rate ↓): fraction of IS inputs wrongly flagged
- AUC: Area under ROC for OOS detection

Extensions (v2):
- Mahalanobis Distance OOD score (Lee et al., NeurIPS 2018)
- Energy Score        OOD score (Liu et al., NeurIPS 2020)
- Relative Mahalanobis (Ren et al., NeurIPS 2021)
These require the model's training embeddings and/or raw logits.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.special import softmax
import json, os


def compute_oos_labels(model_preds: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """Binary labels: 1 = out-of-scope (misclassified), 0 = in-scope."""
    return (model_preds != true_labels).astype(int)


def validity_score_knn_distance(distances: np.ndarray) -> np.ndarray:
    """Higher mean distance → more likely OOS. Normalize to [0,1]."""
    scores = distances.mean(axis=1) if distances.ndim > 1 else distances
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def validity_score_disagreement(model_preds: np.ndarray, knn_preds: np.ndarray,
                                 model_probs: np.ndarray, knn_probs: np.ndarray) -> np.ndarray:
    """Soft disagreement: 1 - overlap between model and kNN probability vectors."""
    # Bhattacharyya coefficient as agreement measure
    agreement = np.sum(np.sqrt(model_probs * knn_probs), axis=1)
    return 1.0 - agreement


def validity_score_entropy(probs: np.ndarray) -> np.ndarray:
    """Normalized entropy of calibrated probabilities."""
    eps = 1e-12
    k = probs.shape[1]
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    return ent / np.log(k)  # normalize to [0,1]


def validity_score_confidence_delta(model_probs: np.ndarray,
                                     knn_probs: np.ndarray) -> np.ndarray:
    """Absolute difference in max confidence between model and kNN."""
    model_conf = model_probs.max(axis=1)
    knn_conf = knn_probs.max(axis=1)
    delta = np.abs(model_conf - knn_conf)
    mn, mx = delta.min(), delta.max()
    if mx - mn < 1e-12:
        return np.zeros_like(delta)
    return (delta - mn) / (mx - mn)


def validity_score_composite(distances: np.ndarray, model_preds: np.ndarray,
                              knn_preds: np.ndarray, model_probs: np.ndarray,
                              knn_probs: np.ndarray,
                              weights=(0.3, 0.3, 0.2, 0.2)) -> np.ndarray:
    """Weighted combination of all validity signals."""
    s1 = validity_score_knn_distance(distances)
    s2 = validity_score_disagreement(model_preds, knn_preds, model_probs, knn_probs)
    s3 = validity_score_entropy(model_probs)
    s4 = validity_score_confidence_delta(model_probs, knn_probs)
    w = np.array(weights)
    w = w / w.sum()
    composite = w[0]*s1 + w[1]*s2 + w[2]*s3 + w[3]*s4
    return composite


def compute_cvr_mvr_at_threshold(oos_labels: np.ndarray, scores: np.ndarray,
                                  threshold: float) -> dict:
    """Compute CVR and MVR at a given threshold."""
    flagged = (scores >= threshold).astype(int)
    
    oos_mask = oos_labels == 1
    is_mask = oos_labels == 0
    
    n_oos = oos_mask.sum()
    n_is = is_mask.sum()
    
    tp = (flagged[oos_mask] == 1).sum() if n_oos > 0 else 0
    fp = (flagged[is_mask] == 1).sum() if n_is > 0 else 0
    
    cvr = tp / n_oos if n_oos > 0 else 0.0
    mvr = fp / n_is if n_is > 0 else 0.0
    
    return {'cvr': float(cvr), 'mvr': float(mvr), 'threshold': threshold,
            'n_oos': int(n_oos), 'n_is': int(n_is), 'tp': int(tp), 'fp': int(fp)}


def compute_oos_metrics(oos_labels: np.ndarray, scores: np.ndarray,
                        threshold: float = 0.2, method_name: str = '') -> dict:
    """Full OOS detection metrics: CVR, MVR, AUC, and ROC curve data."""
    # AUC
    try:
        auc = roc_auc_score(oos_labels, scores)
    except ValueError:
        auc = 0.5
    
    # CVR/MVR at threshold
    cm = compute_cvr_mvr_at_threshold(oos_labels, scores, threshold)
    
    # ROC curve for plotting
    fpr, tpr, thresholds = roc_curve(oos_labels, scores)
    
    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx])
    cm_optimal = compute_cvr_mvr_at_threshold(oos_labels, scores, optimal_threshold)
    
    return {
        'method': method_name,
        'auc': float(auc),
        'cvr_at_threshold': cm['cvr'],
        'mvr_at_threshold': cm['mvr'],
        'threshold': threshold,
        'cvr_optimal': cm_optimal['cvr'],
        'mvr_optimal': cm_optimal['mvr'],
        'optimal_threshold': optimal_threshold,
        'n_oos': cm['n_oos'],
        'n_is': cm['n_is'],
        'oos_ratio': cm['n_oos'] / (cm['n_oos'] + cm['n_is']),
        'roc_fpr': fpr.tolist(),
        'roc_tpr': tpr.tolist(),
    }


def compute_all_oos_metrics(
    model_logits: np.ndarray,
    model_preds: np.ndarray,
    true_labels: np.ndarray,
    knn_preds: np.ndarray,
    knn_probs: np.ndarray,
    distances: np.ndarray,
    temperature: float = 1.0,
    threshold: float = 0.2,
    # Optional: provide for Mahalanobis / RMD (need training set embeddings)
    train_embeddings: np.ndarray = None,
    train_labels: np.ndarray = None,
    test_embeddings: np.ndarray = None,
) -> dict:
    """
    Compute OOS detection metrics for all validity score strategies.

    Base strategies (always run):
      knn_distance, disagreement, entropy, confidence_delta, composite

    Extended strategies (run if train/test embeddings provided):
      mahalanobis, relative_mahalanobis, energy_score
    """
    # Calibrated model probabilities
    model_probs = softmax(model_logits / temperature, axis=1)
    oos_labels = compute_oos_labels(model_preds, true_labels)

    strategies = {
        'knn_distance':     validity_score_knn_distance(distances),
        'disagreement':     validity_score_disagreement(model_preds, knn_preds, model_probs, knn_probs),
        'entropy':          validity_score_entropy(model_probs),
        'confidence_delta': validity_score_confidence_delta(model_probs, knn_probs),
        'composite':        validity_score_composite(distances, model_preds, knn_preds, model_probs, knn_probs),
    }

    # -- Extended OOD detectors ----------------------------------------
    # Energy Score: cheap — just needs raw logits
    try:
        from knn_datastore import compute_energy_ood_scores
        energy_scores = compute_energy_ood_scores(model_logits)
        strategies['energy_score'] = energy_scores
        print("[OOD] Energy Score computed.")
    except Exception as e:
        print(f"[OOD] Energy Score skipped: {e}")

    # Mahalanobis Distance: needs training embeddings
    if train_embeddings is not None and test_embeddings is not None:
        try:
            from knn_datastore import compute_mahalanobis_ood_scores
            maha_scores = compute_mahalanobis_ood_scores(
                train_embeddings, train_labels, test_embeddings
            )
            strategies['mahalanobis'] = maha_scores
            print("[OOD] Mahalanobis Distance computed.")
        except Exception as e:
            print(f"[OOD] Mahalanobis skipped: {e}")

        # Relative Mahalanobis: needs training embeddings
        try:
            from knn_datastore import compute_relative_mahalanobis_ood_scores
            rmd_scores = compute_relative_mahalanobis_ood_scores(
                train_embeddings, train_labels, test_embeddings
            )
            strategies['relative_mahalanobis'] = rmd_scores
            print("[OOD] Relative Mahalanobis computed.")
        except Exception as e:
            print(f"[OOD] Relative Mahalanobis skipped: {e}")

    results = {}
    for name, scores in strategies.items():
        results[name] = compute_oos_metrics(oos_labels, scores, threshold, name)

    return results


def print_oos_comparison_table(knn_results: dict, codeimprove_results: dict = None):
    """
    Print a formatted OOD detector comparison table, including the new
    Mahalanobis / Energy / RMD detectors alongside the existing ones.
    """
    print("\n" + "="*90)
    print("OOD Detection Method Comparison: CVR / MVR / AUC")
    print("="*90)
    print(f"{'Method':<34} {'CVR(↑)':<10} {'MVR(↓)':<10} {'AUC':<10} {'Opt.CVR':<10} {'Opt.MVR':<10}")
    print("-"*90)

    # CodeImprove paper DSMG results (reported, not reproduced)
    print(f"{'CodeImprove DSMG (paper)':<34} {'57.9%':<10} {'3.0%':<10} {'0.911':<10} {'—':<10} {'—':<10}")

    if codeimprove_results:
        r = codeimprove_results
        print(f"{'CodeImprove DSMG (reproduced)':<34} "
              f"{r['cvr_at_threshold']*100:>5.1f}%     "
              f"{r['mvr_at_threshold']*100:>5.1f}%     "
              f"{r['auc']:.3f}       "
              f"{r['cvr_optimal']*100:>5.1f}%     "
              f"{r['mvr_optimal']*100:>5.1f}%")

    # Display order: original detectors first, then new ones
    display_order = [
        'knn_distance', 'disagreement', 'entropy', 'confidence_delta', 'composite',
        'energy_score', 'mahalanobis', 'relative_mahalanobis',
    ]
    label_map = {
        'knn_distance':         'kNN Distance (baseline)',
        'disagreement':         'Disagreement (baseline)',
        'entropy':              'Entropy (baseline)',
        'confidence_delta':     'Confidence Delta',
        'composite':            'Composite (weighted)',
        'energy_score':         'Energy Score [NEW]',
        'mahalanobis':          'Mahalanobis Distance [NEW]',
        'relative_mahalanobis': 'Relative Mahalanobis [NEW]',
    }

    for key in display_order:
        if key not in knn_results:
            continue
        r = knn_results[key]
        label = label_map.get(key, key)
        print(f"  {label:<32} "
              f"{r['cvr_at_threshold']*100:>5.1f}%     "
              f"{r['mvr_at_threshold']*100:>5.1f}%     "
              f"{r['auc']:.3f}       "
              f"{r['cvr_optimal']*100:>5.1f}%     "
              f"{r['mvr_optimal']*100:>5.1f}%")

    print("="*90)


if __name__ == '__main__':
    # Quick test with synthetic data
    np.random.seed(42)
    n = 100
    k_classes = 4
    
    true_labels = np.random.randint(0, k_classes, n)
    model_logits = np.random.randn(n, k_classes)
    model_preds = model_logits.argmax(axis=1)
    
    knn_probs = softmax(np.random.randn(n, k_classes), axis=1)
    knn_preds = knn_probs.argmax(axis=1)
    distances = np.random.rand(n, 8)
    
    results = compute_all_oos_metrics(model_logits, model_preds, true_labels,
                                       knn_preds, knn_probs, distances)
    print_oos_comparison_table(results)
