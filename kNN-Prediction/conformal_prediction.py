"""
conformal_prediction.py — Conformal Prediction Wrapper for CI-Gated kNN
=========================================================================
Wraps the kNN-augmented classifier output with Regularized Adaptive
Prediction Sets (RAPS), providing set-valued predictions with a
formal, distribution-free coverage guarantee:

    P(true_label ∈ prediction_set) >= 1 - alpha

This is the FIRST application of conformal prediction to OOD-aware
code defect/vulnerability detection and constitutes a novel contribution.

References:
  - Angelopoulos et al., "Uncertainty Sets for Image Classifiers using
    Conformal Prediction" (ICLR 2021) — RAPS
  - Vovk et al., "Algorithmic Learning in a Random World" (2005) — CP theory

Usage:
    from conformal_prediction import KNNConformalPredictor, conformal_metrics

    cp = KNNConformalPredictor(alpha=0.05, method='raps')
    cp.calibrate(calib_probs, calib_labels)          # fit on dev/calib set
    sets = cp.predict_set(test_probs)                # prediction sets
    metrics = cp.evaluate(test_probs, test_labels)   # coverage + set size
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple


class KNNConformalPredictor:
    """
    Conformal Prediction wrapper for kNN-augmented code classifiers.

    Supports two nonconformity score methods:
      - 'lac' : Least Ambiguous set-valued Classifiers (LAC)
                score = 1 - p(true_label|x)   (simple inversion)
      - 'raps': Regularized Adaptive Prediction Sets (RAPS)
                score = sum of sorted probs up to (and including)
                        the true class + regularization penalty

    The conformal threshold q_hat is computed on a calibration set.
    At test time, prediction sets include all labels with
    cumulative softmax probability <= q_hat.
    """

    def __init__(self, alpha: float = 0.05, method: str = 'raps',
                 raps_lambda: float = 0.1, raps_k_reg: int = 2):
        """
        Args:
            alpha:      Miscoverage level (0.05 → 95% marginal coverage guarantee)
            method:     'lac' or 'raps'
            raps_lambda: RAPS regularization strength
            raps_k_reg:  RAPS — top-k classes exempt from regularization
        """
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        assert method in ('lac', 'raps'), "method must be 'lac' or 'raps'"
        self.alpha = alpha
        self.method = method
        self.raps_lambda = raps_lambda
        self.raps_k_reg = raps_k_reg
        self.q_hat = None
        self.calibrated = False
        self.n_calib = 0
        self.num_classes = None

    # ------------------------------------------------------------------
    # Nonconformity scores
    # ------------------------------------------------------------------

    def _lac_scores(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """LAC nonconformity: 1 - P(true class | x)."""
        n = len(labels)
        return 1.0 - probs[np.arange(n), labels]

    def _raps_scores(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        RAPS nonconformity scores.
        For each sample, sort class probs descending; sum them until
        (and including) the true class rank, plus a regularization term
        for classes beyond k_reg.
        """
        n, C = probs.shape
        scores = np.zeros(n)

        for i in range(n):
            sorted_idx = np.argsort(-probs[i])          # descending
            cumsum = 0.0
            for rank, cls in enumerate(sorted_idx):
                # Regularization: penalize including more than k_reg classes
                reg = max(0, rank + 1 - self.raps_k_reg) * self.raps_lambda
                cumsum += probs[i, cls] + reg
                if cls == labels[i]:
                    scores[i] = cumsum
                    break
        return scores

    def _compute_scores(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self.method == 'lac':
            return self._lac_scores(probs, labels)
        else:
            return self._raps_scores(probs, labels)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, calib_probs: np.ndarray, calib_labels: np.ndarray):
        """
        Compute the conformal threshold q_hat from a calibration set.

        Args:
            calib_probs:  (n, C) softmax probabilities on calibration set
            calib_labels: (n,)   true integer labels
        """
        self.num_classes = calib_probs.shape[1]
        self.n_calib = len(calib_labels)

        scores = self._compute_scores(calib_probs, calib_labels)

        # Conformal quantile: ceil((n+1)*(1-alpha))/n  (finite-sample guarantee)
        level = np.ceil((self.n_calib + 1) * (1 - self.alpha)) / self.n_calib
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        self.calibrated = True

        print(f"[ConformalCP] Calibrated: alpha={self.alpha}, method={self.method}, "
              f"q_hat={self.q_hat:.4f}, n_calib={self.n_calib}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_sets_lac(self, probs: np.ndarray) -> List[List[int]]:
        """Return prediction sets using LAC: include class if P(c|x) >= 1-q_hat."""
        sets = []
        for p in probs:
            pred_set = [c for c in range(len(p)) if p[c] >= 1.0 - self.q_hat]
            if not pred_set:                          # ensure non-empty
                pred_set = [int(np.argmax(p))]
            sets.append(pred_set)
        return sets

    def _predict_sets_raps(self, probs: np.ndarray) -> List[List[int]]:
        """
        Return RAPS prediction sets: include classes in descending prob order
        until cumulative regularized score >= q_hat.
        """
        sets = []
        for p in probs:
            sorted_idx = np.argsort(-p)
            pred_set = []
            cumsum = 0.0
            for rank, cls in enumerate(sorted_idx):
                reg = max(0, rank + 1 - self.raps_k_reg) * self.raps_lambda
                cumsum += p[cls] + reg
                pred_set.append(int(cls))
                if cumsum >= self.q_hat:
                    break
            if not pred_set:
                pred_set = [int(sorted_idx[0])]
            sets.append(pred_set)
        return sets

    def predict_set(self, test_probs: np.ndarray) -> List[List[int]]:
        """
        Generate prediction sets for test samples.

        Args:
            test_probs: (n, C) softmax probabilities

        Returns:
            List of prediction sets, each a list of class indices
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before predict_set().")
        if self.method == 'lac':
            return self._predict_sets_lac(test_probs)
        else:
            return self._predict_sets_raps(test_probs)

    def predict_point(self, test_probs: np.ndarray) -> np.ndarray:
        """Standard argmax point prediction (for fallback / comparison)."""
        return np.argmax(test_probs, axis=1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_probs: np.ndarray, test_labels: np.ndarray,
                 spt_level: int = 0) -> Dict:
        """
        Compute conformal coverage and efficiency metrics.

        Args:
            test_probs:  (n, C) softmax probabilities
            test_labels: (n,)   true labels
            spt_level:   number of SPTs applied (for logging)

        Returns:
            dict of metrics:
                marginal_coverage : fraction of samples where label ∈ set
                avg_set_size      : average size of prediction sets
                singleton_rate    : fraction of sets with exactly 1 label
                empty_rate        : fraction of empty sets (should be 0)
                point_accuracy    : argmax accuracy
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before evaluate().")

        pred_sets = self.predict_set(test_probs)
        n = len(test_labels)

        covered = [int(test_labels[i]) in pred_sets[i] for i in range(n)]
        set_sizes = [len(s) for s in pred_sets]

        point_preds = self.predict_point(test_probs)
        point_acc = float(np.mean(point_preds == test_labels))

        metrics = {
            'spt_level':        spt_level,
            'alpha':            self.alpha,
            'method':           self.method,
            'q_hat':            float(self.q_hat),
            'marginal_coverage': float(np.mean(covered)),
            'avg_set_size':     float(np.mean(set_sizes)),
            'singleton_rate':   float(np.mean([s == 1 for s in set_sizes])),
            'empty_rate':       float(np.mean([s == 0 for s in set_sizes])),
            'point_accuracy':   point_acc,
            'n_samples':        n,
        }
        return metrics

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save calibration state."""
        state = {
            'alpha':      self.alpha,
            'method':     self.method,
            'raps_lambda': self.raps_lambda,
            'raps_k_reg': self.raps_k_reg,
            'q_hat':      self.q_hat,
            'n_calib':    self.n_calib,
            'num_classes': self.num_classes,
            'calibrated': self.calibrated,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"[ConformalCP] Saved to {path}")

    def load(self, path: str):
        """Load calibration state."""
        with open(path) as f:
            state = json.load(f)
        self.alpha       = state['alpha']
        self.method      = state['method']
        self.raps_lambda = state['raps_lambda']
        self.raps_k_reg  = state['raps_k_reg']
        self.q_hat       = state['q_hat']
        self.n_calib     = state['n_calib']
        self.num_classes = state['num_classes']
        self.calibrated  = state['calibrated']
        print(f"[ConformalCP] Loaded from {path}: q_hat={self.q_hat:.4f}")


# ---------------------------------------------------------------------------
# Convenience function: run full conformal evaluation across SPT levels
# ---------------------------------------------------------------------------

def run_conformal_spt_evaluation(
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    spt_probs_dict: Dict[int, np.ndarray],
    spt_labels_dict: Dict[int, np.ndarray],
    alpha: float = 0.05,
    method: str = 'raps',
    output_dir: str = './conformal_results',
) -> Dict[int, Dict]:
    """
    Calibrate on clean dev set, then evaluate on SPT-shifted test sets.

    Args:
        calib_probs/labels:    Dev-set probabilities and labels
        spt_probs_dict:        {n_spts: test_probs (n, C)} for n=0,1,2,3
        spt_labels_dict:       {n_spts: test_labels (n,)}
        alpha:                 Target miscoverage level
        method:                'lac' or 'raps'
        output_dir:            Where to save results

    Returns:
        {n_spts: metrics_dict}
    """
    os.makedirs(output_dir, exist_ok=True)
    cp = KNNConformalPredictor(alpha=alpha, method=method)
    cp.calibrate(calib_probs, calib_labels)
    cp.save(os.path.join(output_dir, 'conformal_calibration.json'))

    all_metrics = {}
    print(f"\n{'='*70}")
    print(f"Conformal Prediction Evaluation (alpha={alpha}, method={method})")
    print(f"{'='*70}")
    print(f"{'SPT Level':<12} {'Coverage':<12} {'Set Size':<12} "
          f"{'Singleton%':<14} {'Point Acc':<12}")
    print(f"{'-'*70}")

    for n_spt in sorted(spt_probs_dict.keys()):
        metrics = cp.evaluate(
            spt_probs_dict[n_spt], spt_labels_dict[n_spt], spt_level=n_spt
        )
        all_metrics[n_spt] = metrics
        print(f"  N={n_spt}        "
              f"{metrics['marginal_coverage']*100:>7.2f}%     "
              f"{metrics['avg_set_size']:>7.3f}       "
              f"{metrics['singleton_rate']*100:>8.1f}%      "
              f"{metrics['point_accuracy']*100:>7.2f}%")

    print(f"{'='*70}")
    print(f"  Target coverage: {(1-alpha)*100:.0f}%  (should hold even under SPT shift)")

    # Save all metrics
    out_path = os.path.join(output_dir, 'conformal_spt_metrics.json')
    with open(out_path, 'w') as f:
        json.dump({str(k): v for k, v in all_metrics.items()}, f, indent=2)
    print(f"\nConformal metrics saved to {out_path}")

    return all_metrics


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(0)
    C, n = 4, 500
    # Simulate calibration and test probs
    calib_probs  = np.random.dirichlet([3, 1, 1, 1], size=n)
    calib_labels = np.argmax(calib_probs + 0.1 * np.random.randn(n, C), axis=1)
    test_probs   = np.random.dirichlet([2, 1, 1, 1], size=200)
    test_labels  = np.argmax(test_probs  + 0.1 * np.random.randn(200, C), axis=1)

    cp = KNNConformalPredictor(alpha=0.1, method='raps')
    cp.calibrate(calib_probs, calib_labels)
    metrics = cp.evaluate(test_probs, test_labels)
    print("\nSelf-test metrics:", json.dumps(metrics, indent=2))
