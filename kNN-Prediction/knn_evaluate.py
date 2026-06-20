"""
kNN Evaluation Pipeline: End-to-end evaluation with all metrics.

Runs baselines, kNN prediction, calibration, and generates comparison results.
"""
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report

from calibration import (
    TemperatureScaler, compute_entropy, compute_ece, 
    compute_brier_score, compute_selective_risk
)


class KNNEvaluator:
    """End-to-end evaluation comparing baselines vs kNN methods."""
    
    def __init__(self, num_classes=4, output_dir='./results'):
        self.num_classes = num_classes
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_metrics(self, probs, predictions, labels, method_name=''):
        """Compute all evaluation metrics."""
        accuracy = float(np.mean(predictions == labels))
        f1_macro = float(f1_score(labels, predictions, average='macro', zero_division=0))
        f1_weighted = float(f1_score(labels, predictions, average='weighted', zero_division=0))
        mcc = float(matthews_corrcoef(labels, predictions))
        ece, bin_info = compute_ece(probs, labels)
        brier = compute_brier_score(probs, labels, self.num_classes)
        coverages, accs, risks = compute_selective_risk(probs, labels)
        entropy = compute_entropy(probs)
        
        metrics = {
            'method': method_name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'mcc': mcc,
            'ece': ece,
            'brier_score': brier,
            'mean_entropy': float(entropy.mean()),
            'mean_confidence': float(np.max(probs, axis=1).mean()),
            'n_samples': len(labels),
            'selective_risk': {
                'coverages': coverages,
                'accuracies': accs,
                'risks': risks
            },
            'calibration_bins': bin_info
        }
        
        return metrics
    
    def run_evaluation(self, model_probs, model_logits, labels, 
                       knn_predictor, query_embeddings,
                       calibrator=None):
        """
        Run full evaluation suite.
        
        Args:
            model_probs: (n, num_classes) model softmax probabilities
            model_logits: (n, num_classes) raw logits (pre-softmax)
            labels: (n,) true labels
            knn_predictor: KNNPredictor instance
            query_embeddings: (n, embed_dim) test embeddings
            calibrator: TemperatureScaler instance (optional)
            
        Returns:
            all_results: dict of method -> metrics
        """
        all_results = {}
        
        # --- B1: Model only (no adaptation) ---
        print("\n=== B1: Model Only ===")
        t0 = time.time()
        b1_probs, b1_preds = knn_predictor.predict_model_only(model_probs)
        b1_time = time.time() - t0
        b1_metrics = self.compute_metrics(b1_probs, b1_preds, labels, 'B1_model_only')
        b1_metrics['latency_seconds'] = b1_time
        all_results['B1_model_only'] = b1_metrics
        print(f"  Accuracy: {b1_metrics['accuracy']:.4f}, F1: {b1_metrics['f1_macro']:.4f}, "
              f"MCC: {b1_metrics['mcc']:.4f}, ECE: {b1_metrics['ece']:.4f}")
        
        # --- B3: Model + Temperature Scaling only ---
        if calibrator is not None and calibrator.fitted:
            print("\n=== B3: Model + Temperature Scaling ===")
            t0 = time.time()
            cal_probs = calibrator.calibrate(model_logits)
            b3_preds = np.argmax(cal_probs, axis=1)
            b3_time = time.time() - t0
            b3_metrics = self.compute_metrics(cal_probs, b3_preds, labels, 'B3_calibrated_only')
            b3_metrics['latency_seconds'] = b3_time
            b3_metrics['temperature'] = calibrator.temperature
            all_results['B3_calibrated_only'] = b3_metrics
            print(f"  Accuracy: {b3_metrics['accuracy']:.4f}, F1: {b3_metrics['f1_macro']:.4f}, "
                  f"MCC: {b3_metrics['mcc']:.4f}, ECE: {b3_metrics['ece']:.4f}, T={calibrator.temperature:.4f}")
        
        # --- B4: Model + kNN only (no calibration) ---
        print("\n=== B4: Model + kNN (no calibration) ===")
        t0 = time.time()
        b4_probs, b4_preds, b4_knn_probs, b4_details = knn_predictor.predict(
            query_embeddings, model_probs
        )
        b4_time = time.time() - t0
        b4_metrics = self.compute_metrics(b4_probs, b4_preds, labels, 'B4_knn_only')
        b4_metrics['latency_seconds'] = b4_time
        b4_metrics['k'] = knn_predictor.k
        b4_metrics['lambda'] = knn_predictor.lambda_val
        all_results['B4_knn_only'] = b4_metrics
        print(f"  Accuracy: {b4_metrics['accuracy']:.4f}, F1: {b4_metrics['f1_macro']:.4f}, "
              f"MCC: {b4_metrics['mcc']:.4f}, ECE: {b4_metrics['ece']:.4f}")
        
        # --- M1: kNN + Calibrated Gating (full method) ---
        if calibrator is not None and calibrator.fitted:
            print("\n=== M1: kNN + Calibrated Gating (FULL METHOD) ===")
            cal_probs = calibrator.calibrate(model_logits)
            cal_entropy = compute_entropy(cal_probs)
            
            t0 = time.time()
            m1_probs, m1_preds, m1_knn_probs, m1_details = knn_predictor.predict(
                query_embeddings, cal_probs,
                uncertainty_gated=True,
                calibrated_entropy=cal_entropy
            )
            m1_time = time.time() - t0
            m1_metrics = self.compute_metrics(m1_probs, m1_preds, labels, 'M1_knn_calibrated_gated')
            m1_metrics['latency_seconds'] = m1_time
            m1_metrics['k'] = knn_predictor.k
            m1_metrics['temperature'] = calibrator.temperature
            m1_metrics['mean_lambda'] = float(m1_details['lambda_per_sample'].mean())
            all_results['M1_knn_calibrated_gated'] = m1_metrics
            print(f"  Accuracy: {m1_metrics['accuracy']:.4f}, F1: {m1_metrics['f1_macro']:.4f}, "
                  f"MCC: {m1_metrics['mcc']:.4f}, ECE: {m1_metrics['ece']:.4f}, "
                  f"mean_λ: {m1_metrics['mean_lambda']:.4f}")
        
        # --- Save results ---
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        # Remove non-serializable data for JSON output
        json_results = {}
        for key, val in all_results.items():
            json_results[key] = {k: v for k, v in val.items() 
                                if not isinstance(v, np.ndarray)}
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        # --- Print comparison table ---
        self._print_comparison_table(all_results)
        
        # --- Generate plots ---
        self._plot_selective_risk(all_results)
        self._plot_calibration(all_results)
        
        return all_results
    
    def _print_comparison_table(self, results):
        """Print a formatted comparison table."""
        print("\n" + "=" * 90)
        print(f"{'Method':<30} {'Acc':>8} {'F1-M':>8} {'MCC':>8} {'ECE':>8} {'Brier':>8} {'Latency':>10}")
        print("-" * 90)
        for method, metrics in results.items():
            latency = metrics.get('latency_seconds', 0)
            print(f"{method:<30} {metrics['accuracy']:>8.4f} {metrics['f1_macro']:>8.4f} "
                  f"{metrics['mcc']:>8.4f} {metrics['ece']:>8.4f} {metrics['brier_score']:>8.4f} "
                  f"{latency:>9.4f}s")
        print("=" * 90)
    
    def _plot_selective_risk(self, results):
        """Plot selective risk curves."""
        plt.figure(figsize=(8, 6))
        for method, metrics in results.items():
            sr = metrics['selective_risk']
            plt.plot(sr['coverages'], sr['accuracies'], label=method, linewidth=2)
        
        plt.xlabel('Coverage', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Selective Risk Curve: Accuracy vs Coverage', fontsize=14)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = os.path.join(self.output_dir, 'selective_risk_curve.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Selective risk curve saved to {path}")
    
    def _plot_calibration(self, results):
        """Plot calibration reliability diagrams."""
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
        if len(results) == 1:
            axes = [axes]
        
        for ax, (method, metrics) in zip(axes, results.items()):
            bins = metrics['calibration_bins']
            if bins['accuracy']:
                ax.bar(range(len(bins['accuracy'])), bins['accuracy'], 
                       alpha=0.7, label='Accuracy')
                ax.bar(range(len(bins['confidence'])), bins['confidence'], 
                       alpha=0.3, label='Confidence')
                ax.plot([0, len(bins['accuracy'])], [0, 1], 'r--', alpha=0.5)
                ax.set_title(f"{method}\nECE={metrics['ece']:.4f}", fontsize=10)
                ax.legend(fontsize=8)
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'calibration_diagrams.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Calibration diagrams saved to {path}")
    
    def save_predictions(self, predictions, ids, output_path):
        """Save predictions in CodeImprove-compatible format: idx\\tlabel\\n"""
        with open(output_path, 'w') as f:
            for idx, pred in zip(ids, predictions):
                f.write(f"{idx}\t{pred}\n")
        print(f"Predictions saved to {output_path}")
