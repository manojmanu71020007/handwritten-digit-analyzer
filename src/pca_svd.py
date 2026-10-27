"""
src/pca_svd.py
Optimized Mathematical Engine for SVD/PCA Dimensionality Reduction and Orthogonal Projection.
Features:
- Cached orthogonal projection operator P_k
- O(D) Pythagorean reconstruction error computation
- High-efficiency matrix operations
"""

import numpy as np

class PCASVDAnalyzer:
    """
    Mathematical engine for SVD/PCA dimensionality reduction and orthogonal projection.
    """
    def __init__(self, n_components: int = 15):
        self.k = n_components
        self.mean_vector: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.Vt: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.V_k: np.ndarray | None = None
        self._P_k: np.ndarray | None = None  # Cached projection matrix
        self.explained_variance: np.ndarray | None = None
        self.explained_variance_ratio: np.ndarray | None = None
        self.cumulative_variance_ratio: np.ndarray | None = None
        self.N: int = 0
        self.D: int = 0

    def fit(self, X: np.ndarray) -> "PCASVDAnalyzer":
        """
        Fits PCA/SVD on the dataset matrix X in R^(N x D).
        N: number of sample vectors
        D: dimension of each sample vector (H * W)
        """
        self.N, self.D = X.shape
        if self.k > min(self.N, self.D):
            self.k = min(self.N, self.D)

        # 1. Compute empirical mean vector mu in R^D
        self.mean_vector = np.mean(X, axis=0)

        # 2. Mean-center data: X_c in R^(N x D)
        X_c = X - self.mean_vector

        # 3. Full Singular Value Decomposition: X_c = U * Sigma * V^T
        # U in R^(N x N), singular_values in R^(min(N, D)), Vt in R^(D x D)
        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)

        self.U = U
        self.singular_values = s
        self.Vt = Vt
        self.V = Vt.T  # Columns of V are right singular vectors (principal component directions)

        # Truncate to top k components
        self.V_k = np.ascontiguousarray(self.V[:, :self.k])  # R^(D x k)

        # Precompute and cache projection operator P_k = V_k * V_k^T (O(1) access later)
        self._P_k = np.dot(self.V_k, self.V_k.T)

        # 4. Eigenvalues of Covariance Matrix C = (1 / (N - 1)) * X_c^T * X_c
        dof = max(1, self.N - 1)
        self.explained_variance = (s ** 2) / dof
        total_var = np.sum(self.explained_variance)
        if total_var > 0:
            self.explained_variance_ratio = self.explained_variance / total_var
        else:
            self.explained_variance_ratio = np.zeros_like(self.explained_variance)
        self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)

        return self

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Projects centered vector into the k-dimensional subspace spanned by V_k:
        z = (x - mu) * V_k
        Returns coordinate vector z in R^k.
        Supports 1D vector (D,) or 2D batch (M, D).
        """
        if self.mean_vector is None or self.V_k is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        x_centered = x - self.mean_vector
        return np.dot(x_centered, self.V_k)

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstructs high-dimensional vector from subspace coordinates z:
        x_recon = mu + z * V_k^T
        """
        if self.mean_vector is None or self.V_k is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        return self.mean_vector + np.dot(z, self.V_k.T)

    def get_projection_matrix(self) -> np.ndarray:
        """
        Returns cached D x D orthogonal projection matrix P_k = V_k * V_k^T.
        Satisfies P_k^2 = P_k and P_k^T = P_k.
        O(1) complexity because it was pre-calculated during fit().
        """
        if self._P_k is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._P_k

    def reconstruction_error(self, x: np.ndarray, z: np.ndarray | None = None) -> float:
        """
        Computes L2 reconstruction error ||x - x_recon||_2 in O(D)
        using the Pythagorean identity for orthogonal projectors:
        ||x_c - P_k x_c||^2 = ||x_c||^2 - ||z||^2
        """
        if self.mean_vector is None:
            raise RuntimeError("Model has not been fitted yet.")
        if z is None:
            z = self.project(x)
        x_c = x - self.mean_vector
        norm_sq = float(np.dot(x_c, x_c))
        z_sq = float(np.dot(z, z))
        res_sq = max(0.0, norm_sq - z_sq)
        return float(np.sqrt(res_sq))

    def get_eigen_digit(self, component_idx: int, target_size: tuple[int, int]) -> np.ndarray:
        """
        Reshapes the i-th principal component vector (eigen-digit) into 2D matrix (H, W).
        """
        if self.V is None:
            raise RuntimeError("Model has not been fitted yet.")
        if component_idx >= self.V.shape[1]:
            raise IndexError(f"Component index {component_idx} out of range (max {self.V.shape[1]-1}).")
        col = self.V[:, component_idx]
        return col.reshape(target_size)
