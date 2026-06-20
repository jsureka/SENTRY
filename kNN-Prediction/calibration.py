"""
Calibration module: Temperature scaling and calibration metrics.

Implements post-hoc calibration for multiclass classification models,
including temperature scaling optimization on validation data,
ECE (Expected Calibration Error), and Brier score computation.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, log_softmax


class TemperatureScaler:
    """
    Temperature scaling for post-hoc calibration.
    
    Optimizes a single temperature parameter T such that:
        calibrated_probs = softmax(logits / T)
    
    T > 1 → softens predictions (fixes overconfidence)
    T < 1 → sharpens predictions (fixes underconfidence)
    T = 1 → no change
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.fitted = False
    
    def fit(self, logits, labels):
        """
        Optimize temperature on validation data.
        
        Args:
            logits: (n, num_classes) raw logits (pre-softmax)
            labels: (n,) true labels
        """
        def nll_loss(T):
            """Negative log-likelihood with temperature scaling."""
            T = max(T[0], 0.01)  # prevent division by zero
            scaled_logits = logits / T
            log_probs = log_softmax(scaled_logits, axis=1)
            # NLL = -mean(log_prob[true_class])
            nll = -np.mean(log_probs[np.arange(len(labels)), labels])
            return nll
        
        # Optimize T starting from 1.0
        result = minimize(nll_loss, x0=[1.0], method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-6})
        
        self.temperature = max(result.x[0], 0.01)
        self.fitted = True
        
        print(f"Temperature scaling fitted: T = {self.temperature:.4f}")
        print(f"  NLL before (T=1): {nll_loss([1.0]):.4f}")
        print(f"  NLL after  (T={self.temperature:.4f}): {result.fun:.4f}")
        
        return self.temperature
    
    def calibrate(self, logits):
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: (n, num_classes) raw logits
            
        Returns:
            calibrated_probs: (n, num_classes) calibrated probabilities
        """
        if not self.fitted:
            print("WARNING: Temperature not fitted. Using T=1.0 (no calibration)")
        
        scaled_logits = logits / self.temperature
        return softmax(scaled_logits, axis=1)
    
    def save(self, path):
        """Save temperature parameter."""
        np.save(path, np.array([self.temperature]))
        print(f"Temperature saved to {path}: T={self.temperature:.4f}")
    
    def load(self, path):
        """Load temperature parameter."""
        self.temperature = float(np.load(path)[0])
        self.fitted = True
        print(f"Temperature loaded from {path}: T={self.temperature:.4f}")


def compute_entropy(probs):
    """
    Compute entropy of probability distributions.
    
    Args:
        probs: (n, num_classes) probability array
        
    Returns:
        entropy: (n,) entropy values
    """
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (multiclass).
    
    Args:
        probs: (n, num_classes) predicted probabilities
        labels: (n,) true labels
        n_bins: number of bins for calibration
        
    Returns:
        ece: float, expected calibration error
        bin_info: dict with per-bin details
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_info = {'bins': [], 'accuracy': [], 'confidence': [], 'count': []}
    
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
            
            bin_info['bins'].append(f"[{bin_boundaries[i]:.2f}, {bin_boundaries[i+1]:.2f})")
            bin_info['accuracy'].append(float(avg_accuracy))
            bin_info['confidence'].append(float(avg_confidence))
            bin_info['count'].append(int(in_bin.sum()))
    
    return float(ece), bin_info


def compute_brier_score(probs, labels, num_classes=4):
    """
    Compute multiclass Brier score.
    
    Args:
        probs: (n, num_classes) predicted probabilities
        labels: (n,) true labels
        num_classes: number of classes
    
    Returns:
        brier: float, Brier score (lower is better)
    """
    n = len(labels)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), labels] = 1.0
    
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)


def compute_selective_risk(probs, labels, n_thresholds=100):
    """
    Compute selective risk curve: accuracy vs coverage at varying confidence thresholds.
    
    Args:
        probs: (n, num_classes) predicted probabilities
        labels: (n,) true labels
        n_thresholds: number of threshold points
        
    Returns:
        coverages: list of coverage values
        accuracies: list of accuracy values at each coverage
        risks: list of risk (1 - accuracy) values
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)
    
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    coverages = []
    accuracies = []
    risks = []
    
    for threshold in thresholds:
        mask = confidences >= threshold
        coverage = mask.mean()
        
        if coverage > 0:
            acc = correct[mask].mean()
        else:
            acc = 0.0
        
        coverages.append(float(coverage))
        accuracies.append(float(acc))
        risks.append(float(1.0 - acc))
    
    return coverages, accuracies, risks
