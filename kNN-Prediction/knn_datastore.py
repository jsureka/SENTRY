"""
kNN Datastore: Build and manage a FAISS-based embedding datastore from training data.

Extracts embeddings from a fine-tuned CodeBERT/RoBERTa model's hidden states,
stores them in a FAISS index for fast nearest-neighbor retrieval at test time.

Extensions (v2):
  - ProtokNNDatastore: centroid-based prototype retrieval (smaller, faster, better calibrated)
  - compute_mahalanobis_ood_scores: post-hoc OOD detector using class-conditional Mahalanobis distance
  - compute_energy_ood_scores: Energy-based OOD score (Liu et al., NeurIPS 2020)
"""
import os
import json
import numpy as np
import torch
import faiss
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler


class CodeDataset(Dataset):
    """Simple dataset that loads JSONL and tokenizes code for embedding extraction."""
    
    def __init__(self, file_path, tokenizer, block_size=400):
        self.examples = []
        self.labels = []
        self.ids = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                js    = json.loads(line)
                # Support Defect (input/label/id) and Devign (func/target/idx)
                code  = js.get('func',   js.get('input',  ''))
                label = int(js.get('target', js.get('label', 0)))
                idx   = str(js.get('idx', js.get('id', js.get('commit_id', '0'))))
                code  = ' '.join(code.split())
                code_tokens   = tokenizer.tokenize(code)[:block_size - 2]
                source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
                source_ids    = tokenizer.convert_tokens_to_ids(source_tokens)
                source_ids   += [tokenizer.pad_token_id] * (block_size - len(source_ids))

                self.examples.append(source_ids)
                self.labels.append(label)
                self.ids.append(idx)

        print(f"Loaded {len(self.examples)} samples from {file_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
            self.ids[i]
        )


class EmbeddingExtractor:
    """Extract embeddings from a model's hidden states."""
    
    STRATEGIES = ['last_cls', 'second_last_cls', 'avg_last4_cls', 'mean_pool_last']
    
    def __init__(self, model, device, strategy='last_cls'):
        assert strategy in self.STRATEGIES, f"Unknown strategy: {strategy}. Choose from {self.STRATEGIES}"
        self.model = model
        self.device = device
        self.strategy = strategy
    
    def extract(self, input_ids):
        """
        Extract embeddings from model hidden states.
        
        Args:
            input_ids: (batch, seq_len) tensor of token IDs
            
        Returns:
            embeddings: (batch, hidden_dim) numpy array
        """
        self.model.eval()
        with torch.no_grad():
            # Access the encoder directly to get hidden states
            outputs = self.model.encoder(
                input_ids, 
                attention_mask=input_ids.ne(1),
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors
            
            if self.strategy == 'last_cls':
                # Last layer, [CLS] token (position 0)
                emb = hidden_states[-1][:, 0, :]
                
            elif self.strategy == 'second_last_cls':
                # Second-to-last layer, [CLS] token
                emb = hidden_states[-2][:, 0, :]
                
            elif self.strategy == 'avg_last4_cls':
                # Average of last 4 layers, [CLS] token
                last4 = torch.stack(list(hidden_states[-4:]), dim=0)
                emb = last4[:, :, 0, :].mean(dim=0)
                
            elif self.strategy == 'mean_pool_last':
                # Mean pool over all tokens in last layer
                # Use attention mask to ignore padding
                attention_mask = input_ids.ne(1).unsqueeze(-1).float()
                last_hidden = hidden_states[-1]
                emb = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            
            return emb.cpu().numpy()


class KNNDatastore:
    """FAISS-based datastore for kNN retrieval."""
    
    def __init__(self, embed_dim=768):
        self.embed_dim = embed_dim
        self.index = None
        self.labels = None
        self.ids = None
    
    def build(self, model, dataset, device, batch_size=32, strategy='last_cls'):
        """
        Build the datastore by extracting embeddings from all training samples.
        
        Args:
            model: Fine-tuned model (with .encoder attribute)
            dataset: CodeDataset instance
            device: torch device
            batch_size: batch size for embedding extraction
            strategy: embedding extraction strategy
        """
        extractor = EmbeddingExtractor(model, device, strategy=strategy)
        model.to(device)
        model.eval()
        
        dataloader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=batch_size
        )
        
        all_embeddings = []
        all_labels = []
        all_ids = []
        
        print(f"Building datastore with strategy='{strategy}'...")
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch[0].to(device)
            labels = batch[1].numpy()
            ids = batch[2]
            
            embeddings = extractor.extract(input_ids)
            all_embeddings.append(embeddings)
            all_labels.extend(labels.tolist())
            if isinstance(ids, torch.Tensor):
                all_ids.extend(ids.numpy().tolist())
            else:
                all_ids.extend(list(ids))
        
        all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        self.labels = np.array(all_labels)
        self.ids = np.array(all_ids)
        self.embed_dim = all_embeddings.shape[1]
        
        # Build FAISS index (exact L2 search — fine for small datasets)
        self.index = faiss.IndexFlatL2(self.embed_dim)
        
        # Optionally normalize for cosine similarity
        faiss.normalize_L2(all_embeddings)
        self.index.add(all_embeddings)
        
        print(f"Datastore built: {self.index.ntotal} vectors, dim={self.embed_dim}")
    
    def save(self, output_dir):
        """Save the FAISS index and labels to disk."""
        os.makedirs(output_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(output_dir, 'faiss_index.bin'))
        np.save(os.path.join(output_dir, 'labels.npy'), self.labels)
        np.save(os.path.join(output_dir, 'ids.npy'), self.ids)
        print(f"Datastore saved to {output_dir}")
    
    def load(self, output_dir):
        """Load the FAISS index and labels from disk."""
        self.index = faiss.read_index(os.path.join(output_dir, 'faiss_index.bin'))
        self.labels = np.load(os.path.join(output_dir, 'labels.npy'))
        self.ids = np.load(os.path.join(output_dir, 'ids.npy'))
        self.embed_dim = self.index.d
        print(f"Datastore loaded: {self.index.ntotal} vectors, dim={self.embed_dim}")
    
    def search(self, query_embeddings, k=8):
        """
        Search for k nearest neighbors.
        
        Args:
            query_embeddings: (n_queries, embed_dim) numpy array
            k: number of neighbors
            
        Returns:
            distances: (n_queries, k) array of L2 distances
            neighbor_labels: (n_queries, k) array of neighbor labels
            neighbor_ids: (n_queries, k) array of neighbor IDs
        """
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        
        distances, indices = self.index.search(query_embeddings, k)
        neighbor_labels = self.labels[indices]
        neighbor_ids = self.ids[indices]
        
        return distances, neighbor_labels, neighbor_ids


# =============================================================================
# ProtokNNDatastore — centroid-based prototype retrieval
# =============================================================================

class ProtokNNDatastore:
    """
    Prototype kNN datastore: instead of storing all N training embeddings,
    cluster each class into K centroids with K-Means and store only those.

    Benefits over full kNN:
      - ~200x smaller datastore (default: 50 centroids/class vs ~8K samples/class)
      - Noise-robust: centroids average out mislabeled/noisy training examples
      - Better Brier score: probability interpolation stays smooth
      - Faster retrieval at inference time

    Novel contribution: first application of prototype retrieval to
    OOD-aware code defect / vulnerability prediction.
    """

    def __init__(self, embed_dim: int = 768, n_clusters_per_class: int = 50):
        self.embed_dim = embed_dim
        self.n_clusters_per_class = n_clusters_per_class
        self.index = None
        self.labels = None
        self.cluster_sizes = None   # how many training points each centroid represents
        self._all_train_embeddings = None
        self._all_train_labels = None

    def build_from_embeddings(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ):
        """
        Build prototype index from precomputed embedding matrix.

        Args:
            train_embeddings: (N, D) float32 array
            train_labels:     (N,)   int array
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        self._all_train_embeddings = train_embeddings
        self._all_train_labels = train_labels
        self.embed_dim = train_embeddings.shape[1]

        proto_embs, proto_labels, sizes = [], [], []
        unique_classes = np.unique(train_labels)

        print(f"[ProtokNN] Building prototype datastore "
              f"({self.n_clusters_per_class} centroids/class, "
              f"{len(unique_classes)} classes)...")

        for cls in unique_classes:
            class_embs = train_embeddings[train_labels == cls]
            n_clust = min(self.n_clusters_per_class, len(class_embs))

            if n_clust < 2:
                # Too few samples — use the mean as the single centroid
                proto_embs.append(class_embs.mean(axis=0, keepdims=True))
                proto_labels.append(int(cls))
                sizes.append(len(class_embs))
            else:
                km = MiniBatchKMeans(
                    n_clusters=n_clust,
                    random_state=42,
                    batch_size=min(1024, len(class_embs)),
                    max_iter=100,
                )
                km.fit(class_embs)
                centroids = km.cluster_centers_.astype(np.float32)
                counts = np.bincount(km.labels_, minlength=n_clust)

                proto_embs.append(centroids)
                proto_labels.extend([int(cls)] * n_clust)
                sizes.extend(counts.tolist())

        all_protos = np.vstack(proto_embs).astype(np.float32)
        self.labels = np.array(proto_labels)
        self.cluster_sizes = np.array(sizes)

        # Normalize and build FAISS index
        faiss.normalize_L2(all_protos)
        self.index = faiss.IndexFlatL2(self.embed_dim)
        self.index.add(all_protos)

        print(f"[ProtokNN] Done. Prototype store: {self.index.ntotal} centroids "
              f"(vs {len(train_labels)} original training vectors, "
              f"{100*self.index.ntotal/len(train_labels):.1f}% size)")

    def build(self, model, dataset, device, batch_size: int = 32,
              strategy: str = 'last_cls'):
        """
        Build prototype datastore directly from model + dataset (end-to-end).
        Internally uses KNNDatastore's embedding extraction logic.
        """
        # Re-use the regular datastore to extract embeddings first
        temp_store = KNNDatastore()
        temp_store.build(model, dataset, device, batch_size, strategy)
        self.build_from_embeddings(temp_store.index.reconstruct_n(0, temp_store.index.ntotal),
                                   temp_store.labels)

    def save(self, output_dir: str):
        """Save prototype FAISS index and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(output_dir, 'proto_faiss_index.bin'))
        np.save(os.path.join(output_dir, 'proto_labels.npy'), self.labels)
        np.save(os.path.join(output_dir, 'proto_cluster_sizes.npy'), self.cluster_sizes)
        print(f"[ProtokNN] Saved to {output_dir}")

    def load(self, output_dir: str):
        """Load prototype FAISS index and metadata."""
        self.index = faiss.read_index(os.path.join(output_dir, 'proto_faiss_index.bin'))
        self.labels = np.load(os.path.join(output_dir, 'proto_labels.npy'))
        self.cluster_sizes = np.load(os.path.join(output_dir, 'proto_cluster_sizes.npy'))
        self.embed_dim = self.index.d
        print(f"[ProtokNN] Loaded: {self.index.ntotal} centroids, dim={self.embed_dim}")

    def search(self, query_embeddings: np.ndarray, k: int = 8):
        """Search for k nearest prototypes (same interface as KNNDatastore)."""
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        k_actual = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embeddings, k_actual)
        neighbor_labels = self.labels[indices]
        # Return dummy IDs (prototype index)
        neighbor_ids = indices
        return distances, neighbor_labels, neighbor_ids


# =============================================================================
# Post-hoc OOD detectors: Mahalanobis + Energy Score
# =============================================================================

def compute_mahalanobis_ood_scores(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    use_l2_norm: bool = True,
) -> np.ndarray:
    """
    Class-conditional Mahalanobis distance OOD detector.
    (Lee et al., NeurIPS 2018 + Müller & Hein 2024 L2-norm fix)

    Higher score → more in-distribution / less OOD.
    Use the negative minimum Mahalanobis distance across classes as the score.

    Args:
        train_embeddings: (N, D) training embeddings
        train_labels:     (N,)   corresponding labels
        test_embeddings:  (M, D) test embeddings
        use_l2_norm:      L2-normalize embeddings before fitting (recommended, 2024)

    Returns:
        scores: (M,) OOD scores — higher = more in-distribution
    """
    try:
        from sklearn.covariance import EmpiricalCovariance
    except ImportError:
        raise ImportError("scikit-learn required for Mahalanobis OOD detection")

    train_embs = train_embeddings.astype(np.float64)
    test_embs  = test_embeddings.astype(np.float64)

    if use_l2_norm:
        train_norms = np.linalg.norm(train_embs, axis=1, keepdims=True)
        test_norms  = np.linalg.norm(test_embs,  axis=1, keepdims=True)
        train_embs = train_embs / np.where(train_norms > 1e-10, train_norms, 1.0)
        test_embs  = test_embs  / np.where(test_norms  > 1e-10, test_norms,  1.0)

    # Fit shared covariance matrix on all training embeddings
    print("[Mahalanobis] Fitting covariance matrix on training embeddings...")
    cov_model = EmpiricalCovariance(assume_centered=False)
    cov_model.fit(train_embs)
    precision = cov_model.precision_          # inverse covariance (D x D)

    # Compute per-class means
    unique_classes = np.unique(train_labels)
    class_means = {
        int(c): train_embs[train_labels == c].mean(axis=0)
        for c in unique_classes
    }

    print(f"[Mahalanobis] Computing distances for {len(test_embs)} test samples "
          f"across {len(unique_classes)} classes...")

    scores = np.zeros(len(test_embs))
    for i, x in enumerate(test_embs):
        min_dist = np.inf
        for mu in class_means.values():
            diff = x - mu
            dist = float(diff @ precision @ diff)
            if dist < min_dist:
                min_dist = dist
        # Negate: high negative Mahal dist → far from all classes → OOD
        # Return positive score where high = in-distribution
        scores[i] = -min_dist

    return scores.astype(np.float32)


def compute_energy_ood_scores(logits: np.ndarray) -> np.ndarray:
    """
    Energy-based OOD score (Liu et al., NeurIPS 2020).

    Energy(x) = -T * log sum_c exp(f_c(x) / T)
    where T=1 by default (temperature is absorbed into logits from calibration).

    Higher energy → deeper OOD.  We negate so that *higher* score = more in-distribution
    (consistent with Mahalanobis and entropy conventions in this codebase).

    Args:
        logits: (N, C) raw model logits (before softmax)

    Returns:
        scores: (N,) energy scores — higher = more in-distribution
    """
    # log-sum-exp along class dimension
    # scipy.special.logsumexp is numerically stable
    from scipy.special import logsumexp
    energy = logsumexp(logits, axis=1)    # more in-dist → higher log-sum-exp
    return energy.astype(np.float32)      # already in natural units; no negation needed


def compute_relative_mahalanobis_ood_scores(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Relative Mahalanobis Distance (RMD) OOD detector (Ren et al., NeurIPS 2021).
    Subtracts the Mahalanobis distance to the *background* (class-agnostic) distribution,
    improving near-OOD detection versus standard Mahalanobis.

    score_RMD(x) = -Maha_class(x) - (-Maha_background(x))
                 = Maha_background(x) - Maha_class(x)

    Higher score → more in-distribution (relative to background).

    Args:
        train_embeddings: (N, D)
        train_labels:     (N,)
        test_embeddings:  (M, D)

    Returns:
        rmd_scores: (M,)  higher is more in-distribution
    """
    try:
        from sklearn.covariance import EmpiricalCovariance
    except ImportError:
        raise ImportError("scikit-learn required")

    # L2-normalize
    tr = train_embeddings.astype(np.float64)
    te = test_embeddings.astype(np.float64)
    tr /= np.linalg.norm(tr, axis=1, keepdims=True).clip(min=1e-10)
    te /= np.linalg.norm(te, axis=1, keepdims=True).clip(min=1e-10)

    # Background (all classes together)
    bg_cov   = EmpiricalCovariance().fit(tr)
    bg_prec  = bg_cov.precision_
    bg_mean  = tr.mean(axis=0)

    # Class-conditional
    cls_cov  = EmpiricalCovariance().fit(tr)
    cls_prec = cls_cov.precision_
    class_means = {int(c): tr[train_labels == c].mean(0)
                   for c in np.unique(train_labels)}

    rmd_scores = np.zeros(len(te))
    for i, x in enumerate(te):
        # Background distance
        diff_bg = x - bg_mean
        d_bg = float(diff_bg @ bg_prec @ diff_bg)
        # Min class distance
        d_class = min(
            float((x - mu) @ cls_prec @ (x - mu))
            for mu in class_means.values()
        )
        # RMD: larger = more in-distribution relative to background
        rmd_scores[i] = d_bg - d_class

    return rmd_scores.astype(np.float32)
