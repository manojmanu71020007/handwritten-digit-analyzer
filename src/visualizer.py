"""
src/visualizer.py
Visualizer module for generating clear, academic-grade plots:
1. Digit Matrix & Heatmap
2. Top Eigen-Digits (Orthonormal Principal Axes)
3. Scree Plot & Cumulative Explained Variance
4. 2D / 3D PCA Subspace Projection Scatter
5. Inner-Product Similarity Scores Bar Chart
6. Multi-resolution Subspace Reconstruction Comparison
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# Set clean matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

class DigitVisualizer:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_matrix_representation(self, matrix: np.ndarray, digit_label: str = "Unknown", filename: str = "1_digit_matrix.png"):
        """
        Visualizes the original digit image alongside its raw numerical matrix values.
        """
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 2, width_ratios=[1, 1.4])

        # Subplot 1: Image Heatmap
        ax1 = fig.add_subplot(gs[0])
        im = ax1.imshow(matrix, cmap="Blues", interpolation="nearest")
        ax1.set_title(f"Stage 1 & 2: Digit Image Matrix ({matrix.shape[0]}x{matrix.shape[1]})\nDigit: {digit_label}", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Column Index j (Width)")
        ax1.set_ylabel("Row Index i (Height)")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Pixel Intensity (0-255)")

        # Subplot 2: Numerical Matrix Grid (downsampled or cropped for clarity if needed)
        ax2 = fig.add_subplot(gs[1])
        # Display central 10x10 submatrix to show exact numerical values
        sub = matrix[2:12, 2:12]
        ax2.imshow(sub, cmap="YlGnBu", alpha=0.3)
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                val = int(sub[i, j])
                ax2.text(j, i, str(val), ha="center", va="center", fontsize=8, color="black" if val < 160 else "white")
        ax2.set_xticks(range(sub.shape[1]))
        ax2.set_yticks(range(sub.shape[0]))
        ax2.set_xticklabels([f"j={j+2}" for j in range(sub.shape[1])])
        ax2.set_yticklabels([f"i={i+2}" for i in range(sub.shape[0])])
        ax2.set_title("Numerical Submatrix A[2:12, 2:12] in R^(10x10)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_scree_analysis(self, singular_values: np.ndarray, explained_variance_ratio: np.ndarray, k: int, filename: str = "2_scree_plot.png"):
        """
        Plots singular value spectrum (scree plot) and cumulative explained variance.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        n_comp = len(singular_values)
        indices = np.arange(1, n_comp + 1)

        # Singular values plot
        ax1.plot(indices, singular_values, "o-", color="#1f77b4", linewidth=2, markersize=5)
        ax1.axvline(x=k, color="red", linestyle="--", label=f"Selected Cutoff k = {k}")
        ax1.set_title("SVD Singular Value Spectrum (Scree Plot)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Singular Value Index i")
        ax1.set_ylabel("Singular Value σ_i")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Cumulative explained variance
        cum_var = np.cumsum(explained_variance_ratio) * 100
        ax2.plot(indices, cum_var, "s-", color="#2ca02c", linewidth=2, markersize=5)
        ax2.axvline(x=k, color="red", linestyle="--", label=f"Cutoff k = {k} ({cum_var[k-1]:.1f}% Var)")
        ax2.axhline(y=cum_var[k-1], color="gray", linestyle=":")
        ax2.set_title("Cumulative Explained Variance", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Principal Components k")
        ax2.set_ylabel("Explained Variance (%)")
        ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_eigen_digits(self, V: np.ndarray, target_size: tuple[int, int], n_digits: int = 10, filename: str = "3_eigen_digits.png"):
        """
        Visualizes the first n_digits right singular vectors (Eigen-digits) reshaped as images.
        """
        fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
        fig.suptitle("Orthonormal Basis Vectors: Top 10 Eigen-Digits (Columns of V_k)", fontsize=14, fontweight="bold")
        for i, ax in enumerate(axes.flat):
            if i < n_digits:
                eigen_img = V[:, i].reshape(target_size)
                # Diverging colormap because basis vectors can have positive and negative weights
                vmax = np.max(np.abs(eigen_img))
                im = ax.imshow(eigen_img, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                ax.set_title(f"PC #{i+1} (v_{i+1})", fontsize=11)
                ax.axis("off")
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_pca_subspace_2d(self, Z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, predicted_class: int, filename: str = "4_pca_subspace_2d.png"):
        """
        Plots a 2D scatter of the training dataset projected onto the first 2 principal components,
        with the unknown digit highlighted.
        """
        fig, ax = plt.subplots(figsize=(9, 7))
        cmap = plt.get_cmap("tab10")

        classes = sorted(list(set(y_train)))
        for c in classes:
            idx = np.where(y_train == c)[0]
            ax.scatter(Z_train[idx, 0], Z_train[idx, 1], label=f"Digit {c}", s=45, color=cmap(c), alpha=0.7)

        # Plot unknown test digit
        ax.scatter(z_test[0], z_test[1], color="black", edgecolor="yellow", linewidth=2.5, s=200, marker="*", zorder=10, label=f"Unknown Digit (Pred: {predicted_class})")

        ax.set_title("Stage 6: Orthogonal Projection onto 2D PCA Subspace", fontsize=13, fontweight="bold")
        ax.set_xlabel("Principal Axis 1 (z_1)")
        ax.set_ylabel("Principal Axis 2 (z_2)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_pca_subspace_3d(self, Z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, predicted_class: int, filename: str = "4_pca_subspace_3d.png"):
        """
        Plots a high-definition 3D vector representation and scatter of the dataset
        projected onto the first 3 principal components (R^3), with the test digit
        highlighted as a 3D vector.
        """
        fig = plt.figure(figsize=(9, 7.5))
        ax = fig.add_subplot(111, projection="3d")
        cmap = plt.get_cmap("tab10")

        classes = sorted(list(set(y_train)))
        for c in classes:
            idx = np.where(y_train == c)[0]
            ax.scatter(Z_train[idx, 0], Z_train[idx, 1], Z_train[idx, 2],
                       label=f"Digit {c}", s=38, color=cmap(c), alpha=0.6, depthshade=True)

        # Plot 3D Vector from origin (0,0,0) to test digit (z1, z2, z3)
        ax.plot([0, z_test[0]], [0, z_test[1]], [0, z_test[2]],
                color="#e11d48", linewidth=2.5, linestyle="-", label="Vector $\mathbf{z}_{3D}$", zorder=15)

        # Plot unknown test digit marker
        ax.scatter([z_test[0]], [z_test[1]], [z_test[2]],
                   color="#e11d48", edgecolor="white", linewidth=2.0, s=220, marker="*",
                   zorder=20, label=f"Test Digit (Pred: {predicted_class})")

        # Visual drop-down stem for 3D depth perception
        z_floor = ax.get_zlim()[0] if ax.get_zlim() else np.min(Z_train[:, 2])
        ax.plot([z_test[0], z_test[0]], [z_test[1], z_test[1]], [z_floor, z_test[2]],
                color="#64748b", linestyle=":", linewidth=1.2)

        norm_3d = float(np.linalg.norm(z_test[:3]))
        ax.set_title(f"Stage 6: 3D Vector Representation in Principal Subspace ($\mathbf{{z}} \in \mathbb{{R}}^3$)\n"
                     f"$\mathbf{{z}}_{{3D}} = [{z_test[0]:+.3f}, {z_test[1]:+.3f}, {z_test[2]:+.3f}]^T$,  $\|\mathbf{{z}}_{{3D}}\|_2 = {norm_3d:.3f}$",
                     fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Principal Axis 1 ($z_1$)", fontsize=9, labelpad=8)
        ax.set_ylabel("Principal Axis 2 ($z_2$)", fontsize=9, labelpad=8)
        ax.set_zlabel("Principal Axis 3 ($z_3$)", fontsize=9, labelpad=8)

        ax.view_init(elev=22, azim=40)
        ax.legend(bbox_to_anchor=(1.05, 0.95), loc="upper left", fontsize=8.5, frameon=True)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_pca_subspace(self, Z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, predicted_class: int, filename: str = "4_pca_subspace.png"):
        """
        Generates both the 3D vector subspace representation and the 2D version.
        Saves the 3D plot as filename (e.g. 4_pca_subspace.png) and 4_pca_subspace_3d.png,
        and saves the 2D version as 4_pca_subspace_2d.png.
        """
        self.plot_pca_subspace_2d(Z_train, y_train, z_test, predicted_class, filename="4_pca_subspace_2d.png")
        self.plot_pca_subspace_3d(Z_train, y_train, z_test, predicted_class, filename="4_pca_subspace_3d.png")
        return self.plot_pca_subspace_3d(Z_train, y_train, z_test, predicted_class, filename=filename)

    def plot_similarity_scores(self, sim_scores: dict[int, float], predicted_digit: int, true_label: str = "Unknown", filename: str = "5_similarity_scores.png"):
        """
        Plots a horizontal bar chart of Inner-Product Cosine Similarity across classes 0 to 9.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        digits = list(sim_scores.keys())
        scores = [sim_scores[d] for d in digits]

        colors = ["#2ca02c" if d == predicted_digit else "#1f77b4" for d in digits]
        bars = ax.barh(digits, scores, color=colors, edgecolor="black", height=0.6)

        ax.set_yticks(digits)
        ax.set_yticklabels([f"Digit {d}" for d in digits], fontsize=11)
        ax.set_xlabel("Cosine Similarity: S(z_test, z_proto) = <z_test, z_proto> / (||z_test|| ||z_proto||)", fontsize=11)
        ax.set_title(f"Stage 7 & 8: Similarity Scores & Prediction\nPredicted: Digit {predicted_digit} | Ground Truth: {true_label}", fontsize=13, fontweight="bold")
        ax.set_xlim(min(0.0, min(scores) - 0.1), 1.05)

        # Add text labels on bars
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=10, fontweight="bold" if bar.get_facecolor()[:3] == (0.172, 0.627, 0.172) else "normal")

        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_reconstruction_progression(self, original_matrix: np.ndarray, pca_analyzer, ks: list[int] = [1, 3, 5, 10, 15], filename: str = "6_reconstruction.png"):
        """
        Shows original image vs progressive reconstruction at varying subspace dimensions k.
        """
        fig, axes = plt.subplots(1, len(ks) + 1, figsize=(15, 3.2))
        x_orig = original_matrix.flatten()

        # Original
        axes[0].imshow(original_matrix, cmap="Blues")
        axes[0].set_title("Original Image\n(D = 400)", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        x_c = x_orig - pca_analyzer.mean_vector
        for idx, k in enumerate(ks):
            Vk = pca_analyzer.V[:, :k]
            zk = np.dot(x_c, Vk)
            x_recon = pca_analyzer.mean_vector + np.dot(zk, Vk.T)
            img_recon = x_recon.reshape(original_matrix.shape)
            err = np.linalg.norm(x_orig - x_recon, ord=2)

            axes[idx + 1].imshow(img_recon, cmap="Blues")
            axes[idx + 1].set_title(f"k = {k} dims\nErr: {err:.1f}", fontsize=10)
            axes[idx + 1].axis("off")

        fig.suptitle("Subspace Orthogonal Projection & Reconstruction: x_tilde = mu + P_k (x - mu)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path
