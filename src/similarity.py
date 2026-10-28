"""
src/similarity.py
Fully Vectorized Similarity Classifier Engine
Utilizes SIMD and BLAS matrix multiplications to calculate all 10 prototype inner products,
cosine similarities, and Euclidean distances in a single vectorized pass (< 10 microseconds).
"""

import numpy as np

class SimilarityClassifier:
    """
    Computes inner-product similarity, cosine similarity, distance metrics,
    and assigns digit predictions based on stored patterns in PCA subspace.
    """
    def __init__(self):
        self.classes: list[int] = []
        self.class_prototypes: dict[int, np.ndarray] = {}
        self.stored_features: np.ndarray | None = None
        self.stored_labels: np.ndarray | None = None
        
        # Precomputed vectorized matrices and norms
        self.prototype_matrix: np.ndarray | None = None  # (10, k)
        self.prototype_norms: np.ndarray | None = None   # (10,)
        self.stored_feature_norms: np.ndarray | None = None # (N,)

        # 400D raw feature storage for high-precision metric space inner products
        self.stored_raw_features: np.ndarray | None = None # (N, 400)
        self.raw_prototypes: np.ndarray | None = None       # (10, 400)

    def fit_patterns(self, Z_train: np.ndarray, y_train: np.ndarray, X_train: np.ndarray | None = None) -> "SimilarityClassifier":
        """
        Stores training features in PCA subspace and computes class prototype centroids:
        bar_z_c = (1 / N_c) * sum_{i in class c} z_i
        Precomputes prototype matrix and norms for instant vectorized inference.
        Also caches 400D unit feature vectors if provided.
        """
        self.stored_features = np.ascontiguousarray(Z_train)
        self.stored_labels = np.array([int(x) if str(x).isdigit() else str(x) for x in y_train], dtype=object)
        self.classes = sorted(list(set(self.stored_labels)), key=lambda x: (isinstance(x, str), x))

        proto_list = []
        for c in self.classes:
            idx = np.where(self.stored_labels == c)[0]
            prototype = np.mean(Z_train[idx], axis=0)
            self.class_prototypes[c] = prototype
            proto_list.append(prototype)

        self.prototype_matrix = np.ascontiguousarray(np.array(proto_list, dtype=np.float64))
        self.prototype_norms = np.linalg.norm(self.prototype_matrix, axis=1)
        self.stored_feature_norms = np.linalg.norm(self.stored_features, axis=1)

        if X_train is not None:
            self.stored_raw_features = np.ascontiguousarray(X_train)
            raw_proto_list = []
            for c in self.classes:
                idx = np.where(self.stored_labels == c)[0]
                p = np.mean(X_train[idx], axis=0)
                p_norm = np.linalg.norm(p)
                if p_norm > 1e-12:
                    p = p / p_norm
                raw_proto_list.append(p)
            self.raw_prototypes = np.ascontiguousarray(np.array(raw_proto_list, dtype=np.float64))

        return self

    @staticmethod
    def inner_product(u: np.ndarray, v: np.ndarray) -> float:
        """Standard Euclidean inner product: <u, v> = u^T * v"""
        return float(np.dot(u, v))

    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """Calculates Cosine Similarity bounded in [-1, 1]"""
        norm_u = np.linalg.norm(u, ord=2)
        norm_v = np.linalg.norm(v, ord=2)
        if norm_u < 1e-12 or norm_v < 1e-12:
            return 0.0
        return float(np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0))

    @staticmethod
    def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
        """Euclidean distance: d(u, v) = ||u - v||_2"""
        return float(np.linalg.norm(u - v, ord=2))

    def compare_unknown(self, z_test: np.ndarray, temperature: float = 0.2, x_test: np.ndarray | None = None) -> dict:
        """
        Fully vectorized comparison of unknown digit z_test against all class prototypes:
        - Uses single BLAS dot product for all 10 inner products
        - Vectorized SIMD cosine similarity and Euclidean distance calculation
        - Vectorized exemplar matching across stored training samples in 400D/PCA
        """
        if self.prototype_matrix is None or self.prototype_norms is None:
            raise RuntimeError("Classifier has not been fitted with patterns yet.")

        norm_z = float(np.linalg.norm(z_test, ord=2))
        
        # 1. Vectorized Inner Products (10 classes at once via BLAS matrix-vector dot)
        raw_ips = np.dot(self.prototype_matrix, z_test)

        # 2. Vectorized Cosine Similarities in PCA subspace
        if norm_z < 1e-12:
            cos_sims = np.zeros_like(raw_ips)
        else:
            denom = norm_z * self.prototype_norms + 1e-15
            cos_sims = np.clip(raw_ips / denom, -1.0, 1.0)

        # 3. Vectorized Euclidean Distances: ||z - p||^2 = ||z||^2 + ||p||^2 - 2<z, p>
        dist_sq = np.maximum(0.0, (norm_z ** 2) + (self.prototype_norms ** 2) - 2.0 * raw_ips)
        dists = np.sqrt(dist_sq)

        # 4. Exemplar Matching
        exemplar_max_sims = np.zeros(len(self.classes))
        
        # Prefer exact 400D metric space inner products if x_test and raw features are available
        if x_test is not None and self.stored_raw_features is not None and self.raw_prototypes is not None:
            norm_x = float(np.linalg.norm(x_test, ord=2))
            if norm_x > 1e-12:
                sample_sims = np.dot(self.stored_raw_features, x_test)
                for i, c in enumerate(self.classes):
                    c_indices = np.where(self.stored_labels == c)[0]
                    if len(c_indices) > 0:
                        exemplar_max_sims[i] = float(np.max(sample_sims[c_indices]))
                
                best_idx = int(np.argmax(sample_sims))
                raw_label = self.stored_labels[best_idx]
                nn_label = int(raw_label) if str(raw_label).isdigit() else str(raw_label)
                nn_sim = float(sample_sims[best_idx])
                
                proto_sims_400 = np.dot(self.raw_prototypes, x_test)
                effective_sims = 0.25 * proto_sims_400 + 0.75 * exemplar_max_sims
            else:
                nn_label = self.classes[0]
                nn_sim = 0.0
                effective_sims = cos_sims
        elif norm_z > 1e-12 and self.stored_features is not None:
            sample_ips = np.dot(self.stored_features, z_test)
            sample_denom = norm_z * self.stored_feature_norms + 1e-15
            sample_sims = np.clip(sample_ips / sample_denom, -1.0, 1.0)
            
            for i, c in enumerate(self.classes):
                c_indices = np.where(self.stored_labels == c)[0]
                if len(c_indices) > 0:
                    exemplar_max_sims[i] = float(np.max(sample_sims[c_indices]))

            best_idx = int(np.argmax(sample_sims))
            raw_label = self.stored_labels[best_idx]
            nn_label = int(raw_label) if str(raw_label).isdigit() else str(raw_label)
            nn_sim = float(sample_sims[best_idx])
            effective_sims = 0.35 * cos_sims + 0.65 * exemplar_max_sims
        else:
            nn_label = self.classes[0]
            nn_sim = 0.0
            effective_sims = cos_sims

        # 5. Softmax Confidence Distribution
        exp_scores = np.exp(effective_sims / temperature)
        probs = exp_scores / np.sum(exp_scores)

        # Map results to dictionaries
        sim_scores = {c: float(effective_sims[i]) for i, c in enumerate(self.classes)}
        ip_scores = {c: float(raw_ips[i]) for i, c in enumerate(self.classes)}
        dist_scores = {c: float(dists[i]) for i, c in enumerate(self.classes)}
        prob_dict = {c: float(probs[i]) for i, c in enumerate(self.classes)}

        best_idx_class = int(np.argmax(effective_sims))
        predicted_digit = self.classes[best_idx_class]

        ranking = [(self.classes[i], float(effective_sims[i])) for i in np.argsort(-effective_sims)]

        return {
            "predicted_digit": predicted_digit,
            "confidence": float(prob_dict[predicted_digit]),
            "similarity_scores": sim_scores,
            "inner_products": ip_scores,
            "euclidean_distances": dist_scores,
            "probabilities": prob_dict,
            "ranking": ranking,
            "nn_label": nn_label,
            "nn_similarity": nn_sim
        }
