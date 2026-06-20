"""
kNN Predictor: Interpolates model predictions with kNN voting for improved accuracy.

Supports multiple voting strategies, uncertainty-gated mixing,
and both all-input and uncertain-only modes.
"""
import numpy as np
from scipy.special import softmax


class KNNPredictor:
    """
    Combines model predictions with kNN-based predictions from a datastore.
    
    The final prediction is:
        p_final = λ · p_model + (1-λ) · p_kNN
    
    where p_kNN is computed from k nearest neighbors' labels,
    weighted by distance.
    """
    
    def __init__(self, datastore, num_classes=4, k=8, lambda_val=0.3,
                 knn_temperature=10.0, voting='distance_weighted',
                 confidence_guard_threshold=None):
        """
        Args:
            datastore: KNNDatastore instance (already built or loaded)
            num_classes: number of output classes
            k: number of nearest neighbors
            lambda_val: interpolation weight (1.0 = model only, 0.0 = kNN only)
            knn_temperature: temperature for distance-weighted voting softmax
            voting: one of 'uniform', 'distance_weighted', 'threshold_filtered'
            confidence_guard_threshold: if set (e.g. 0.70), samples where
                max(p_model) > threshold bypass kNN entirely (use model only).
                Critical for binary tasks (Devign) where the gate always fires
                because binary logits stay near 0.5.
        """
        self.datastore = datastore
        self.num_classes = num_classes
        self.k = k
        self.lambda_val = lambda_val
        self.knn_temperature = knn_temperature
        self.voting = voting
        self.confidence_guard_threshold = confidence_guard_threshold

        assert voting in ['uniform', 'distance_weighted', 'threshold_filtered'], \
            f"Unknown voting strategy: {voting}"
    
    def compute_knn_probs(self, distances, neighbor_labels):
        """
        Compute kNN probability distribution from neighbor distances and labels.
        
        Args:
            distances: (n_queries, k) array of distances
            neighbor_labels: (n_queries, k) array of neighbor labels
            
        Returns:
            knn_probs: (n_queries, num_classes) probability distribution
        """
        n_queries = distances.shape[0]
        knn_probs = np.zeros((n_queries, self.num_classes))
        
        if self.voting == 'uniform':
            # Each neighbor gets equal vote
            for i in range(n_queries):
                for j in range(self.k):
                    label = neighbor_labels[i, j]
                    if label < self.num_classes:
                        knn_probs[i, label] += 1.0
                # Normalize
                total = knn_probs[i].sum()
                if total > 0:
                    knn_probs[i] /= total
                else:
                    knn_probs[i] = 1.0 / self.num_classes
                    
        elif self.voting == 'distance_weighted':
            # Weight by softmax(-distance / temperature)
            # Smaller distance → higher weight
            weights = softmax(-distances / self.knn_temperature, axis=1)
            for i in range(n_queries):
                for j in range(self.k):
                    label = neighbor_labels[i, j]
                    if label < self.num_classes:
                        knn_probs[i, label] += weights[i, j]
                # Normalize
                total = knn_probs[i].sum()
                if total > 0:
                    knn_probs[i] /= total
                else:
                    knn_probs[i] = 1.0 / self.num_classes
                    
        elif self.voting == 'threshold_filtered':
            # Only neighbors within median distance vote (distance-weighted)
            for i in range(n_queries):
                median_dist = np.median(distances[i])
                mask = distances[i] <= median_dist
                if mask.sum() == 0:
                    mask = np.ones(self.k, dtype=bool)  # fallback: use all
                
                filtered_dists = distances[i][mask]
                filtered_labels = neighbor_labels[i][mask]
                weights = softmax(-filtered_dists / self.knn_temperature)
                
                for j, (label, w) in enumerate(zip(filtered_labels, weights)):
                    if label < self.num_classes:
                        knn_probs[i, label] += w
                        
                total = knn_probs[i].sum()
                if total > 0:
                    knn_probs[i] /= total
                else:
                    knn_probs[i] = 1.0 / self.num_classes
        
        return knn_probs
    
    def predict(self, query_embeddings, model_probs,
                uncertainty_gated=False, calibrated_entropy=None,
                gate_a=1.0, gate_b=-0.5):
        """
        Generate predictions by interpolating model and kNN probabilities.

        Args:
            query_embeddings: (n, embed_dim) numpy array of test embeddings
            model_probs: (n, num_classes) numpy array of model softmax outputs
            uncertainty_gated: if True, use adaptive λ based on entropy
            calibrated_entropy: (n,) array of calibrated entropy values (for gating)
            gate_a, gate_b: parameters for sigmoid gating function

        Returns:
            final_probs: (n, num_classes) interpolated probability distribution
            predictions: (n,) argmax predictions
            knn_probs: (n, num_classes) raw kNN probabilities
            details: dict with per-sample mixing info
        """
        # Confidence guard mask — samples where the model is already very
        # confident bypass kNN entirely (avoids noise injection on easy samples
        # and fixes the binary-task calibration-inversion problem).
        if self.confidence_guard_threshold is not None:
            guard_mask = model_probs.max(axis=1) > self.confidence_guard_threshold
        else:
            guard_mask = np.zeros(len(model_probs), dtype=bool)

        # kNN lookup (for all samples; guard applied after)
        distances, neighbor_labels, neighbor_ids = self.datastore.search(
            query_embeddings, k=self.k
        )

        # Compute kNN probabilities
        knn_probs = self.compute_knn_probs(distances, neighbor_labels)

        n = len(model_probs)

        if uncertainty_gated and calibrated_entropy is not None:
            # Adaptive λ: high entropy → low λ → more kNN weight
            lambda_per_sample = 1.0 / (1.0 + np.exp(-(gate_a * calibrated_entropy + gate_b)))
            lambda_per_sample = 1.0 - lambda_per_sample   # invert: high entropy → lower λ
            lambda_per_sample = lambda_per_sample.reshape(-1, 1)
        else:
            lambda_per_sample = np.full((n, 1), self.lambda_val)

        # Apply confidence guard: override λ → 1.0 for guarded samples
        if self.confidence_guard_threshold is not None:
            lambda_per_sample = lambda_per_sample.copy()
            lambda_per_sample[guard_mask] = 1.0

        # Interpolate
        final_probs = lambda_per_sample * model_probs + (1 - lambda_per_sample) * knn_probs

        # Normalize
        row_sums = final_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        final_probs = final_probs / row_sums

        predictions = np.argmax(final_probs, axis=1)

        n_guarded = int(guard_mask.sum())
        details = {
            'lambda_per_sample': lambda_per_sample.flatten(),
            'distances': distances,
            'neighbor_labels': neighbor_labels,
            'neighbor_ids': neighbor_ids,
            'n_guarded': n_guarded,
            'guard_ratio': n_guarded / n,
        }

        return final_probs, predictions, knn_probs, details
    
    def predict_model_only(self, model_probs):
        """Baseline: just use model predictions (λ=1.0)."""
        predictions = np.argmax(model_probs, axis=1)
        return model_probs, predictions
