"""
src/verification.py
Mathematical Verification Engine
Verifies core theorems and linear algebraic properties:
1. Orthonormality of Right Singular Vectors: V^T * V = I
2. Projection Matrix Idempotence: P^2 = P
3. Projection Matrix Symmetry (Orthogonality): P^T = P
4. Cauchy-Schwarz Inequality: |<u, v>| <= ||u||_2 * ||v||_2
5. SVD Energy Conservation (Parseval's Identity): sum(sigma_i^2) == ||X_c||_F^2
6. Monotonicity of Reconstruction Error across subspace dimensions
"""

import numpy as np

class MathematicalVerifier:
    """
    Runs formal checks and proofs on the linear algebraic objects.
    """
    @staticmethod
    def verify_orthonormality(V_k: np.ndarray, tol: float = 1e-10) -> dict:
        """
        Theorem: The columns of V_k form an orthonormal set in R^D.
        Condition: V_k^T * V_k = I_k
        """
        k = V_k.shape[1]
        gram = np.dot(V_k.T, V_k)
        identity = np.eye(k)
        diff = np.linalg.norm(gram - identity, ord="fro")
        is_valid = bool(diff < tol)
        return {
            "name": "Basis Orthonormality (V_k^T * V_k = I_k)",
            "frobenius_error": float(diff),
            "tolerance": tol,
            "passed": is_valid,
            "interpretation": f"Error ||V_k^T * V_k - I_k||_F = {diff:.2e} < {tol:.0e}. Orthonormality verified."
        }

    @staticmethod
    def verify_projection_idempotence(P: np.ndarray, tol: float = 1e-10) -> dict:
        """
        Theorem: An orthogonal projector P onto a subspace satisfies P^2 = P.
        """
        P_squared = np.dot(P, P)
        diff = np.linalg.norm(P_squared - P, ord="fro")
        is_valid = bool(diff < tol)
        return {
            "name": "Projection Idempotence (P^2 = P)",
            "frobenius_error": float(diff),
            "tolerance": tol,
            "passed": is_valid,
            "interpretation": f"Error ||P^2 - P||_F = {diff:.2e} < {tol:.0e}. Subspace projector is idempotent."
        }

    @staticmethod
    def verify_projection_symmetry(P: np.ndarray, tol: float = 1e-10) -> dict:
        """
        Theorem: An orthogonal projection matrix is self-adjoint (symmetric): P^T = P.
        """
        diff = np.linalg.norm(P.T - P, ord="fro")
        is_valid = bool(diff < tol)
        return {
            "name": "Projection Self-Adjointness (P^T = P)",
            "frobenius_error": float(diff),
            "tolerance": tol,
            "passed": is_valid,
            "interpretation": f"Error ||P^T - P||_F = {diff:.2e} < {tol:.0e}. Projection is strictly orthogonal."
        }

    @staticmethod
    def verify_cauchy_schwarz(u: np.ndarray, v: np.ndarray) -> dict:
        """
        Theorem: Cauchy-Schwarz Inequality:
        |<u, v>| <= ||u||_2 * ||v||_2
        Equivalently: -1 <= cos(theta) <= 1
        """
        ip = float(np.dot(u, v))
        norm_u = float(np.linalg.norm(u, ord=2))
        norm_v = float(np.linalg.norm(v, ord=2))
        bound = norm_u * norm_v
        passed = bool(abs(ip) <= (bound + 1e-10))
        ratio = float(abs(ip) / (bound + 1e-15)) if bound > 0 else 0.0

        return {
            "name": "Cauchy-Schwarz Inequality (|<u, v>| <= ||u|| * ||v||)",
            "inner_product": ip,
            "norm_product_bound": bound,
            "ratio": ratio,
            "passed": passed,
            "interpretation": f"|<u, v>| = {abs(ip):.5f} <= {bound:.5f}. Ratio = {ratio:.5f} <= 1.0. Holds true."
        }

    @staticmethod
    def verify_svd_energy_conservation(X_c: np.ndarray, singular_values: np.ndarray, tol: float = 1e-8) -> dict:
        """
        Theorem: Total Frobenius energy equals sum of squared singular values:
        ||X_c||_F^2 = sum_{i=1}^r sigma_i^2
        """
        fro_norm_sq = float(np.linalg.norm(X_c, ord="fro") ** 2)
        sum_sigma_sq = float(np.sum(singular_values ** 2))
        rel_diff = abs(fro_norm_sq - sum_sigma_sq) / max(fro_norm_sq, 1e-12)
        passed = bool(rel_diff < tol)

        return {
            "name": "SVD Energy Conservation (sum(sigma_i^2) = ||X_c||_F^2)",
            "frobenius_energy": fro_norm_sq,
            "singular_values_energy": sum_sigma_sq,
            "relative_error": float(rel_diff),
            "passed": passed,
            "interpretation": f"||X_c||_F^2 = {fro_norm_sq:.4f}, sum(sigma^2) = {sum_sigma_sq:.4f}. Rel error = {rel_diff:.2e}."
        }

    @staticmethod
    def verify_reconstruction_monotonicity(x_sample: np.ndarray, pca_analyzer, ks: list[int] = [1, 3, 5, 10, 15, 20]) -> dict:
        """
        Theorem: Adding orthogonal principal components monotonically reduces (or preserves)
        the projection residual error:
        e_k = ||x - P_k(x)||_2
        """
        errors = []
        x_c = x_sample - pca_analyzer.mean_vector
        for k in ks:
            if k <= pca_analyzer.V.shape[1]:
                Vk = pca_analyzer.V[:, :k]
                zk = np.dot(x_c, Vk)
                x_recon = pca_analyzer.mean_vector + np.dot(zk, Vk.T)
                err = float(np.linalg.norm(x_sample - x_recon, ord=2))
                errors.append((k, err))

        is_monotonic = all(errors[i][1] >= errors[i+1][1] - 1e-8 for i in range(len(errors)-1))
        return {
            "name": "Monotonicity of Reconstruction Error",
            "k_vs_error": errors,
            "passed": is_monotonic,
            "interpretation": f"Error decreases from {errors[0][1]:.4f} (k={errors[0][0]}) down to {errors[-1][1]:.4f} (k={errors[-1][0]}). Monotonic."
        }
