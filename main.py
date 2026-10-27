"""
main.py
End-to-End Execution of Handwritten Digit Feature and Similarity Analyzer
Fulfills all sequential stages:
Digit image -> Matrix representation -> Vectorization -> Normalization -> PCA/SVD -> Projection -> Inner-product similarity -> Prediction
"""

import os
import sys
import argparse
import numpy as np

from src.preprocessor import DigitPreprocessor
from src.pca_svd import PCASVDAnalyzer
from src.similarity import SimilarityClassifier
from src.verification import MathematicalVerifier
from src.visualizer import DigitVisualizer

def load_training_dataset(train_dir: str, preprocessor: DigitPreprocessor):
    """
    Loads all training digit images, vectors, and labels.
    """
    X_list = []
    y_list = []
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory '{train_dir}' not found. Please run generate_dataset.py first.")

    files = sorted([f for f in os.listdir(train_dir) if f.endswith(".png")])
    for fname in files:
        # Expected naming: digit_{c}_sample_{var}.png
        parts = fname.split("_")
        digit_label = int(parts[1])
        fpath = os.path.join(train_dir, fname)
        mat = preprocessor.image_to_matrix(fpath)
        vec = preprocessor.vectorize(mat)
        vec_norm, _ = preprocessor.normalize_l2(vec)
        X_list.append(vec_norm)
        y_list.append(digit_label)

    return np.array(X_list), np.array(y_list)

def run_pipeline(test_image_path: str = "dataset/test/test_7_sample_6.png", n_components: int = 15):
    print("=" * 80)
    print("      HANDWRITTEN DIGIT FEATURE AND SIMILARITY ANALYZER")
    print("   Sequential Computational Process: Modules 1, 2, 3")
    print("=" * 80)

    # Instantiate modules
    preprocessor = DigitPreprocessor(target_size=(20, 20))
    visualizer = DigitVisualizer(output_dir="outputs")
    verifier = MathematicalVerifier()

    # Step 0: Load Stored Training Patterns
    print("\n[*] Loading Stored Digit Patterns (Training Set)...")
    X_train, y_train = load_training_dataset("dataset/train", preprocessor)
    print(f"    Loaded {X_train.shape[0]} training samples across classes 0-9.")
    print(f"    Vector space dimension: D = {X_train.shape[1]} (20x20 pixels).")

    # Step 1: Fit SVD / PCA on Stored Patterns
    print(f"\n[*] Stage 5: Computing PCA/SVD on Mean-Centered Training Set (k = {n_components})...")
    pca = PCASVDAnalyzer(n_components=n_components).fit(X_train)
    cum_var = pca.cumulative_variance_ratio[pca.k - 1] * 100
    print(f"    Top {pca.k} singular values: {np.round(pca.singular_values[:pca.k], 3)}")
    print(f"    Cumulative Explained Variance retained: {cum_var:.2f}%")

    # Step 2: Project Training Patterns and Initialize Similarity Classifier
    Z_train = pca.project(X_train)
    classifier = SimilarityClassifier().fit_patterns(Z_train, y_train)

    # -------------------------------------------------------------------------
    # STAGE 1: DIGIT IMAGE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 1: DIGIT IMAGE INPUT")
    print("-" * 80)
    if not os.path.exists(test_image_path):
        # Fallback to first test image available
        test_files = [f for f in os.listdir("dataset/test") if f.endswith(".png")]
        test_image_path = os.path.join("dataset/test", test_files[0])
    
    print(f"Input image path: '{test_image_path}'")
    # Extract true label if filename conforms
    true_label = "Unknown"
    base = os.path.basename(test_image_path)
    if "test_" in base:
        true_label = base.split("_")[1]

    # -------------------------------------------------------------------------
    # STAGE 2: MATRIX REPRESENTATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 2: MATRIX REPRESENTATION (A in R^(20 x 20))")
    print("-" * 80)
    digit_matrix = preprocessor.image_to_matrix(test_image_path)
    print(f"Matrix Dimension: {digit_matrix.shape[0]} rows x {digit_matrix.shape[1]} columns")
    print(f"Data Type: {digit_matrix.dtype}, Value Range: [{np.min(digit_matrix):.1f}, {np.max(digit_matrix):.1f}]")
    print("\nVisual ASCII Matrix Representation:")
    print(preprocessor.format_ascii_matrix(digit_matrix))
    print("\nCentral Submatrix A[6:12, 6:12]:")
    print(np.round(digit_matrix[6:12, 6:12], 1))

    # -------------------------------------------------------------------------
    # STAGE 3: VECTORIZATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 3: VECTORIZATION (Linear Isomorphism vec(A): R^(20x20) -> R^400)")
    print("-" * 80)
    digit_vector = preprocessor.vectorize(digit_matrix)
    print(f"Vector Space: R^{len(digit_vector)}")
    print(f"First 10 Coordinate Entries of x: {np.round(digit_vector[:10], 1)}")
    print(f"Middle 10 Coordinate Entries of x: {np.round(digit_vector[195:205], 1)}")

    # -------------------------------------------------------------------------
    # STAGE 4: NORMALIZATION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 4: NORMALIZATION (Euclidean L2 Norm Sphere: x_hat = x / ||x||_2)")
    print("-" * 80)
    normalized_vec, l2_norm = preprocessor.normalize_l2(digit_vector)
    print(f"Computed L2 Norm: ||x||_2 = sqrt(sum x_i^2) = {l2_norm:.4f}")
    print(f"Verification ||x_hat||_2 = {np.linalg.norm(normalized_vec, 2):.6f} (Unit sphere in R^400)")
    print(f"First 5 entries of normalized vector x_hat: {np.round(normalized_vec[np.nonzero(normalized_vec)[0][:5]], 4)}")

    # -------------------------------------------------------------------------
    # STAGE 5 & 6: PCA/SVD AND ORTHOGONAL PROJECTION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"STAGE 5 & 6: PCA/SVD ORTHOGONAL PROJECTION (z_test = (x_hat - mu) * V_k in R^{pca.k})")
    print("-" * 80)
    z_test = pca.project(normalized_vec)
    print(f"Projected Feature Vector Dimension: R^{len(z_test)}")
    
    # Explicit 3D Vector Representation
    z_3d = z_test[:3]
    norm_3d = np.linalg.norm(z_3d)
    print("\n" + "=" * 50)
    print(">>> 3D VECTOR REPRESENTATION (z_3D in R^3) <<<")
    print("=" * 50)
    print(f"  Coordinates   : z_1 = {z_3d[0]:+.6f}")
    print(f"                  z_2 = {z_3d[1]:+.6f}")
    print(f"                  z_3 = {z_3d[2]:+.6f}")
    print(f"  Vector Form   : z_3D = [{z_3d[0]:+.4f}, {z_3d[1]:+.4f}, {z_3d[2]:+.4f}]^T")
    print(f"  Euclidean Norm: ||z_3D||_2 = {norm_3d:.6f}")
    print("=" * 50)

    print(f"\nAll Low-dimensional Feature Coordinates (z_1 to z_{pca.k}):")
    for idx, val in enumerate(z_test):
        print(f"  z_{idx+1:02d} = {val:+.5f}")

    recon_err = pca.reconstruction_error(normalized_vec)
    print(f"Subspace Reconstruction Error ||x_hat - x_recon||_2 = {recon_err:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 7: INNER-PRODUCT SIMILARITY
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 7: INNER-PRODUCT SIMILARITY CALCULATION")
    print("-" * 80)
    results = classifier.compare_unknown(z_test)
    print(f"{'Class (Digit)':<15} | {'Raw Inner Product':<20} | {'Cosine Similarity S':<20} | {'Euclidean Dist':<15} | {'Probability':<12}")
    print("-" * 90)
    for c in classifier.classes:
        ip = results["inner_products"][c]
        cs = results["similarity_scores"][c]
        ed = results["euclidean_distances"][c]
        pr = results["probabilities"][c] * 100
        marker = " <== [MAX]" if c == results["predicted_digit"] else ""
        print(f"Digit {c:<8} | {ip:>+18.4f} | {cs:>+18.4f} | {ed:>14.4f} | {pr:>10.2f}% {marker}")

    # -------------------------------------------------------------------------
    # STAGE 8: PREDICTION
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STAGE 8: PREDICTION DECISION")
    print("-" * 80)
    print(f"  >>> PREDICTED DIGIT       : {results['predicted_digit']} <<<")
    print(f"  >>> CONFIDENCE SCORE      : {results['confidence'] * 100:.2f}%")
    print(f"  >>> GROUND TRUTH LABEL    : {true_label}")
    print(f"  >>> 1-NN MATCH IN DATASET : Digit {results['nn_label']} (Similarity: {results['nn_similarity']:.4f})")
    match_status = "CORRECT [PASS]" if str(results['predicted_digit']) == str(true_label) else "MISMATCH / NOVEL"
    print(f"  >>> CLASSIFICATION STATUS : {match_status}")

    # -------------------------------------------------------------------------
    # MATHEMATICAL VERIFICATION OF LINEAR ALGEBRA PROPERTIES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("         MATHEMATICAL VERIFICATION OF PRESCRIBED PROPERTIES")
    print("=" * 80)

    # 1. Orthonormality of Basis: V_k^T * V_k = I_k
    v1 = verifier.verify_orthonormality(pca.V_k)
    print(f"[1] {v1['name']}")
    print(f"    Result: {v1['interpretation']} Passed: {v1['passed']}")

    # 2. Idempotence of Projector: P^2 = P
    P_k = pca.get_projection_matrix()
    v2 = verifier.verify_projection_idempotence(P_k)
    print(f"\n[2] {v2['name']}")
    print(f"    Result: {v2['interpretation']} Passed: {v2['passed']}")

    # 3. Symmetry of Projector: P^T = P
    v3 = verifier.verify_projection_symmetry(P_k)
    print(f"\n[3] {v3['name']}")
    print(f"    Result: {v3['interpretation']} Passed: {v3['passed']}")

    # 4. Cauchy-Schwarz Inequality
    proto_pred = classifier.class_prototypes[results["predicted_digit"]]
    v4 = verifier.verify_cauchy_schwarz(z_test, proto_pred)
    print(f"\n[4] {v4['name']}")
    print(f"    Result: {v4['interpretation']} Passed: {v4['passed']}")

    # 5. SVD Parseval Energy Conservation
    X_c = X_train - pca.mean_vector
    v5 = verifier.verify_svd_energy_conservation(X_c, pca.singular_values)
    print(f"\n[5] {v5['name']}")
    print(f"    Result: {v5['interpretation']} Passed: {v5['passed']}")

    # 6. Monotonicity of Error
    v6 = verifier.verify_reconstruction_monotonicity(normalized_vec, pca, ks=[1, 2, 5, 10, 15, 20])
    print(f"\n[6] {v6['name']}")
    print(f"    Result: {v6['interpretation']} Passed: {v6['passed']}")

    # -------------------------------------------------------------------------
    # GENERATE VISUAL OUTPUT PLOTS
    # -------------------------------------------------------------------------
    print("\n[*] Generating high-resolution academic visualization figures in 'outputs/'...")
    p1 = visualizer.plot_matrix_representation(digit_matrix, digit_label=str(true_label))
    p2 = visualizer.plot_scree_analysis(pca.singular_values, pca.explained_variance_ratio, pca.k)
    p3 = visualizer.plot_eigen_digits(pca.V, target_size=(20, 20), n_digits=10)
    p4 = visualizer.plot_pca_subspace(Z_train, y_train, z_test, results["predicted_digit"])
    p5 = visualizer.plot_similarity_scores(results["similarity_scores"], results["predicted_digit"], true_label=str(true_label))
    p6 = visualizer.plot_reconstruction_progression(digit_matrix, pca, ks=[1, 3, 5, 10, 15])
    print("    [OK] Generated 1_digit_matrix.png")
    print("    [OK] Generated 2_scree_plot.png")
    print("    [OK] Generated 3_eigen_digits.png")
    print("    [OK] Generated 4_pca_subspace.png")
    print("    [OK] Generated 5_similarity_scores.png")
    print("    [OK] Generated 6_reconstruction.png")
    print("\nExecution completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwritten Digit Feature and Similarity Analyzer")
    parser.add_argument("--image", type=str, default="dataset/test/test_7_sample_6.png", help="Path to test digit image")
    parser.add_argument("--k", type=int, default=15, help="Number of principal components for PCA/SVD subspace")
    args = parser.parse_args()
    run_pipeline(test_image_path=args.image, n_components=args.k)
