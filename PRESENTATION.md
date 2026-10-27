# VIVA / PRESENTATION SLIDE DECK
## 48. Handwritten Digit Feature and Similarity Analyzer

**Course Modules:** Modules 1, 2, 3 (Linear Algebra, Matrix Decompositions & Metric Spaces)  
**Speaker Notes & Slide-by-Slide Defense Guide**

---

### Slide 1: Title & Overview
- **Title**: Handwritten Digit Feature and Similarity Analyzer
- **Domain**: Applied Linear Algebra & Pattern Recognition
- **Sequential Mathematical Process**:
  $$\text{Digit Image} \to \text{Matrix} \to \text{Vectorization} \to \text{Normalization} \to \text{PCA/SVD} \to \text{Projection} \to \text{Inner-Product Similarity} \to \text{Prediction}$$
- **Key Concepts Covered**:
  1. Discrete Image Matrices ($A \in \mathbb{R}^{20 \times 20}$)
  2. Euclidean Vector Spaces ($\mathbb{R}^{400}$) & Vectorization
  3. Euclidean Norms & $L_2$ Hypersphere Normalization
  4. Singular Value Decomposition (SVD) & Covariance Eigendecomposition
  5. Orthogonal Subspace Projection Operators ($P_k = V_k V_k^T$)
  6. Inner Products, Cauchy-Schwarz Inequality & Cosine Similarity

---

### Slide 2: Problem Statement & Objectives
- **Aim**: Represent handwritten digits mathematically, reduce their dimensionality from $400 \to 15$, and recognize unknown digits using inner-product metric similarity.
- **Why Linear Algebra?**
  - Eliminates reliance on "black-box" machine learning.
  - Every step is transparent, deterministic, and verifiable through linear algebraic theorems.
- **Dataset**:
  - Training Set: 60 handwritten digit images (10 classes $\times$ 6 variations).
  - Test Set: 30 novel handwritten digit images (10 classes $\times$ 3 unseen variations).

---

### Slide 3: Stage 1 & 2 - Image Sampling & Matrix Representation
- **Mathematical Formulation**:
  $$I: \{1, \dots, H\} \times \{1, \dots, W\} \to [0, 255]$$
  $$A \in \mathbb{R}^{20 \times 20}$$
- **Explanation**:
  - Grayscale intensity sampled on a $20 \times 20$ grid.
  - Values bounded in $[0, 255]$ (0 = background black, 255 = foreground stroke).
  - Represented in memory as a discrete real matrix.

---

### Slide 4: Stage 3 & 4 - Vectorization and $L_2$ Normalization
- **Vector Space Isomorphism**:
  $$\text{vec}: \mathbb{R}^{H \times W} \cong \mathbb{R}^D, \quad D = 20 \times 20 = 400$$
  - Preserves vector addition and scalar multiplication.
- **Norm Normalization**:
  $$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{400} x_i^2}, \quad \hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} \in S^{399}$$
- **Significance**:
  - Stroke brightness varies across handwriting instruments.
  - Mapping to the unit sphere ensures that distance is purely angular:
    $$\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2 = 2(1 - \cos \theta)$$

---

### Slide 5: Stage 5 - SVD & Principal Component Analysis
- **Centering Data Matrix**:
  $$\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \hat{\mathbf{x}}_i, \quad X_c = X - \mathbf{1}_N \boldsymbol{\mu}^T \in \mathbb{R}^{60 \times 400}$$
- **SVD Theorem**:
  $$X_c = U \Sigma V^T$$
  - $U \in \mathbb{R}^{60 \times 60}$: Left singular vectors (sample space).
  - $\Sigma \in \mathbb{R}^{60 \times 400}$: Singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
  - $V \in \mathbb{R}^{400 \times 400}$: Right singular vectors (principal component axes).
- **Variance Retention**:
  - At $k = 15$, retained variance $= \mathbf{88.42\%}$.
  - Dimensionality reduced by **$96.25\%$** ($400 \to 15$).

---

### Slide 6: Stage 6 - Orthogonal Projection & "Eigen-Digits"
- **Subspace Basis**:
  $$V_k = [\mathbf{v}_1, \dots, \mathbf{v}_k] \in \mathbb{R}^{400 \times 15}$$
- **Orthogonal Projection Operator**:
  $$P_k = V_k V_k^T \in \mathbb{R}^{400 \times 400}$$
  - **Idempotence**: $P_k^2 = P_k$
  - **Self-Adjoint**: $P_k^T = P_k$
- **Feature Vector**:
  $$\mathbf{z} = (\hat{\mathbf{x}} - \boldsymbol{\mu}) V_k \in \mathbb{R}^{15}$$
- **Eigen-Digits**:
  - Reshaping columns $\mathbf{v}_i$ back into $20 \times 20$ matrices visualizes the orthogonal basis modes of handwritten digits.

---

### Slide 7: Stage 7 & 8 - Inner-Product Similarity & Prediction
- **Inner Product in Subspace**:
  $$\langle \mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c \rangle = \mathbf{z}_{\text{test}}^T \bar{\mathbf{z}}_c$$
- **Cosine Similarity Metric**:
  $$S(\mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c) = \frac{\langle \mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c \rangle}{\|\mathbf{z}_{\text{test}}\|_2 \|\bar{\mathbf{z}}_c\|_2} \in [-1, 1]$$
- **Classification Rule**:
  $$\hat{y} = \arg\max_{c \in \{0, \dots, 9\}} S(\mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c)$$
- **Softmax Probability**:
  $$p(c \mid \mathbf{z}) = \frac{\exp(S_c / \tau)}{\sum_j \exp(S_j / \tau)}, \quad \tau = 0.2$$

---

### Slide 8: Experimental Results
- **Test Set Accuracy**: **90.00%** on novel handwritten digits.
- **Top Class Performance**: Digits 0, 1, 2, 3, 4, 5, 6 achieved **100% precision**.
- **Case Study (Digit 7)**:
  - Cosine Similarity for Digit 7: **+0.5407** (Highest).
  - Next closest (Digit 2): +0.4636.
  - Softmax Confidence: **44.46%** (Clear decision boundary).

---

### Slide 9: Mathematical Verification Table
| Invariant | Property | Measured Error | Verdict |
| :--- | :--- | :--- | :---: |
| Orthonormality | $V_k^T V_k = I_k$ | $\|V_k^T V_k - I_k\|_F = 5.11 \times 10^{-15}$ | **PASSED** |
| Idempotence | $P_k^2 = P_k$ | $\|P_k^2 - P_k\|_F = 5.10 \times 10^{-15}$ | **PASSED** |
| Self-Adjointness | $P_k^T = P_k$ | $\|P_k^T - P_k\|_F = 0.00$ | **PASSED** |
| Cauchy-Schwarz | $\|\langle \mathbf{u}, \mathbf{v} \rangle\| \le \|\mathbf{u}\|\|\mathbf{v}\|$ | Ratio $= 0.5407 \le 1.0$ | **PASSED** |
| SVD Energy | $\sum \sigma_i^2 = \|X_c\|_F^2$ | Rel Error $= 1.48 \times 10^{-15}$ | **PASSED** |
| Monotonic Residual | $e_k \le e_{k-1}$ | $0.6148 \to 0.2679$ | **PASSED** |

---

### Slide 10: Demonstration Platforms & Conclusion
- **Interactive Demonstrations Developed**:
  1. **Terminal Driver (`main.py`)**: Full step-by-step mathematical trace with ASCII matrix and verification checks.
  2. **Desktop GUI (`gui.py`)**: Tkinter mouse canvas allowing live digit sketching and instant classification.
  3. **Web Dashboard (`app.py`)**: Flask + Chart.js modern UI displaying live charts, eigen-coordinates, and matrices.
  4. **Automated Unit Tests (`pytest`)**: 9 rigorous unit tests verifying mathematical axioms.
- **Key Takeaway**: High-dimensional image classification can be cleanly solved using core linear algebra—matrix representation, norm-preserving maps, SVD basis projection, and inner-product metrics.
