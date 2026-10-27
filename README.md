# Handwritten Digit Feature and Similarity Analyzer
### Mini-Project 48 | Modules: 1, 2, 3 (Linear Algebra & Metric Spaces)

A complete, software-based solution implementing the prescribed sequential linear algebra computational pipeline:
$$\text{Digit Image} \longrightarrow \text{Matrix Representation} \longrightarrow \text{Vectorization} \longrightarrow \text{Normalization} \longrightarrow \text{PCA/SVD} \longrightarrow \text{Projection} \longrightarrow \text{Inner-Product Similarity} \longrightarrow \text{Prediction}$$

---

## 🌟 Key Features

1. **Pure Linear Algebra Engine**: Implemented from mathematical first principles using NumPy & SciPy without black-box machine learning libraries.
2. **Sequential Computational Process**:
   - **Stage 1 & 2**: Digit Image $\to$ Discrete Grayscale Matrix $A \in \mathbb{R}^{20 \times 20}$.
   - **Stage 3**: Linear Isomorphism Vectorization $\text{vec}(A) \in \mathbb{R}^{400}$.
   - **Stage 4**: Euclidean $L_2$ Hypersphere Normalization $\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$.
   - **Stage 5**: Singular Value Decomposition (SVD) $X_c = U \Sigma V^T$ and Covariance Spectrum Analysis.
   - **Stage 6**: Orthogonal Projection onto $k$-dimensional subspace ($P_k = V_k V_k^T$).
   - **Stage 7**: Inner products and Cauchy-Schwarz bounded Cosine Similarity $S(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$.
   - **Stage 8**: Maximum similarity pattern matching and calibrated Softmax prediction.
3. **Formal Mathematical Verification Engine**:
   - Basis Orthonormality ($V_k^T V_k = I_k$)
   - Projector Idempotence ($P^2 = P$)
   - Projector Self-Adjointness ($P^T = P$)
   - Cauchy-Schwarz Inequality ($|\langle \mathbf{u}, \mathbf{v} \rangle| \le \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$)
   - SVD Parseval Energy Conservation ($\sum \sigma_i^2 = \|X_c\|_F^2$)
   - Monotonic Residual Error Reduction ($e_{k+1} \le e_k$)
4. **Three Interactive Demonstration Interfaces**:
   - **CLI Pipeline (`main.py`)**: Detailed terminal breakdown with ASCII matrix view, coordinates, and verification proofs.
   - **Desktop GUI (`gui.py`)**: Tkinter mouse canvas allowing live digit sketching, test sample loading, and real-time linear algebra inspection.
   - **Web Application (`app.py`)**: Flask + Chart.js responsive dashboard for interactive drawing and chart visualization.
5. **Academic Deliverables**:
   - `REPORT.md`: Comprehensive theoretical documentation with equations, derivations, and 8 detailed viva Q&As.
   - `PRESENTATION.md`: Slide-by-slide viva presentation deck.
   - `Handwritten_Digit_Analyzer_Report.pdf`: Ready-to-submit compiled academic PDF report.

---

## 🚀 Quickstart Guide

### 1. Run the Terminal Pipeline
Executes the entire 8-stage mathematical sequence on a test digit, prints matrices, coordinates, and verifies theorems:
```bash
python main.py --image dataset/test/test_7_sample_6.png --k 15
```

### 2. Launch the Interactive Desktop GUI (Tkinter)
Draw digits on a canvas with your mouse or select test samples:
```bash
python gui.py
```

### 3. Launch the Interactive Web Dashboard (Flask)
Start the web server and open `http://127.0.0.1:5000` in your browser:
```bash
python app.py
```

### 4. Run the Automated Mathematical Unit Tests
Verifies that all linear algebra properties pass with errors $< 10^{-14}$:
```bash
python -m pytest tests/test_math_pipeline.py -v
```

### 5. Evaluate Accuracy across the Entire Test Dataset
Calculates classification metrics and saves the confusion matrix:
```bash
python evaluate.py
```

### 6. Recompile the Academic PDF Report
```bash
python generate_report_pdf.py
```

---

## 📂 Project Structure

```
handwriting/
├── dataset/
│   ├── train/                 # 60 images (10 digits x 6 variations)
│   └── test/                  # 30 novel test images (10 digits x 3 variations)
├── outputs/                   # High-resolution generated plots & charts
│   ├── 1_digit_matrix.png     # Digit heatmap and numerical matrix
│   ├── 2_scree_plot.png       # Singular values and explained variance
│   ├── 3_eigen_digits.png     # Orthonormal basis vectors (Eigen-digits)
│   ├── 4_pca_subspace.png     # 2D projection scatter plot
│   ├── 5_similarity_scores.png# Inner-product cosine similarity rankings
│   ├── 6_reconstruction.png   # Multi-resolution orthogonal reconstruction
│   └── 7_confusion_matrix.png # Confusion matrix on unseen test set
├── src/
│   ├── preprocessor.py        # Matrix rep, vectorization, L2 normalization
│   ├── pca_svd.py             # SVD, covariance, orthogonal projection
│   ├── similarity.py          # Inner products, cosine similarity, predictions
│   ├── verification.py        # Mathematical proofs & invariant verifiers
│   └── visualizer.py          # Academic figure generation
├── tests/
│   └── test_math_pipeline.py  # 9 Pytest unit tests
├── templates/
│   └── index.html             # Web interface template
├── app.py                     # Flask web server
├── gui.py                     # Tkinter desktop GUI
├── main.py                    # Terminal pipeline driver
├── evaluate.py                # Dataset accuracy evaluator
├── generate_dataset.py        # Handwritten digit generator
├── generate_report_pdf.py     # PDF compiler using fpdf2
├── PRESENTATION.md            # Viva presentation slide script
├── REPORT.md                  # Comprehensive academic report
└── Handwritten_Digit_Analyzer_Report.pdf # Compiled PDF report
```

---

## 📊 Summary of Mathematical Verification

| Theorem | Mathematical Invariant | Observed Error | Status |
| :--- | :--- | :--- | :---: |
| **Orthonormality** | $V_k^T V_k = I_k$ | $\|V_k^T V_k - I_k\|_F = 5.11 \times 10^{-15}$ | **PASSED** |
| **Idempotence** | $P_k^2 = P_k$ | $\|P_k^2 - P_k\|_F = 5.10 \times 10^{-15}$ | **PASSED** |
| **Self-Adjointness** | $P_k^T = P_k$ | $\|P_k^T - P_k\|_F = 0.00$ | **PASSED** |
| **Cauchy-Schwarz** | $|\langle \mathbf{u}, \mathbf{v} \rangle| \le \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$ | $|\langle \mathbf{u}, \mathbf{v} \rangle| = 0.2017 \le 0.3730$ | **PASSED** |
| **Parseval's Identity** | $\|X_c\|_F^2 = \sum \sigma_i^2$ | Relative Error $= 1.48 \times 10^{-15}$ | **PASSED** |
| **Error Monotonicity** | $e_{k+1} \le e_k$ | $0.6148 \to 0.2679$ | **PASSED** |
