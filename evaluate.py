"""
evaluate.py
Evaluates the mathematical analyzer over the entire test dataset (30 samples: 3 per digit 0-9).
Generates an accuracy report and confusion matrix.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessor import DigitPreprocessor
from src.pca_svd import PCASVDAnalyzer
from src.similarity import SimilarityClassifier

def run_evaluation(k=15):
    prep = DigitPreprocessor(target_size=(20, 20))
    
    # 1. Load train
    X_train, y_train = [], []
    for f in sorted(os.listdir("dataset/train")):
        if f.endswith(".png"):
            c = int(f.split("_")[1])
            mat = prep.image_to_matrix(os.path.join("dataset/train", f))
            vec = prep.vectorize(mat)
            vec_n, _ = prep.normalize_l2(vec)
            X_train.append(vec_n)
            y_train.append(c)
    X_train, y_train = np.array(X_train), np.array(y_train)

    # 2. Fit PCA/SVD
    pca = PCASVDAnalyzer(n_components=k).fit(X_train)
    Z_train = pca.project(X_train)
    clf = SimilarityClassifier().fit_patterns(Z_train, y_train, X_train=X_train)

    # 3. Evaluate on test set
    test_files = sorted([f for f in os.listdir("dataset/test") if f.endswith(".png")])
    y_true, y_pred = [], []
    
    print("=" * 70)
    print(f"{'Test Image':<25} | {'True':<5} | {'Predicted':<10} | {'Cosine Sim':<12} | {'Result':<8}")
    print("-" * 70)

    for f in test_files:
        c_true = int(f.split("_")[1])
        mat = prep.image_to_matrix(os.path.join("dataset/test", f))
        vec = prep.vectorize(mat)
        vec_n, _ = prep.normalize_l2(vec)
        z = pca.project(vec_n)
        res = clf.compare_unknown(z, x_test=vec_n)
        c_pred = res["predicted_digit"]
        sim = res["similarity_scores"][c_pred]
        
        y_true.append(c_true)
        y_pred.append(c_pred)
        
        status = "MATCH" if c_true == c_pred else "MISMATCH"
        print(f"{f:<25} | {c_true:<5} | {c_pred:<10} | {sim:>+10.4f} | {status:<8}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_true == y_pred) * 100
    print("=" * 70)
    print(f"Overall Accuracy: {accuracy:.2f}% ({np.sum(y_true == y_pred)} / {len(y_true)} correctly classified)")

    # Compute confusion matrix
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix on Test Set (Accuracy = {accuracy:.1f}%, k = {k})", fontsize=12, fontweight="bold")
    plt.colorbar(label="Sample Count")
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Digit Class", fontsize=11)
    plt.ylabel("True Digit Class", fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    cm_path = os.path.join("outputs", "7_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[OK] Saved Confusion Matrix plot to '{cm_path}'")

    return accuracy, cm

if __name__ == "__main__":
    run_evaluation(k=15)
