# MINI-PROJECT REPORT
## 48. Handwritten Digit Feature and Similarity Analyzer

**Course Modules:** Modules 1, 2, 3 (Linear Algebra, Vector Spaces, Matrix Decompositions, Metric Spaces)  
**Date:** Academic Year 2026  
**Implementation:** Pure NumPy / SciPy Linear Algebra Engine (from scratch)

---

## 1. Aim and Objectives

### 1.1 Aim
To mathematically represent handwritten digits as high-dimensional vectors, reduce their dimensionality using **Singular Value Decomposition (SVD)** and **Principal Component Analysis (PCA)**, and compare an unknown digit against stored patterns through **Inner-Product Cosine Similarity** in the reduced orthogonal subspace to achieve accurate pattern recognition.

### 1.2 Objectives
1. **Matrix Representation**: Formulate grayscale digit images as discrete real matrices $A \in \mathbb{R}^{H \times W}$.
2. **Vector Space Isomorphism**: Flatten matrix representations into high-dimensional coordinate vectors $\mathbf{x} = \text{vec}(A) \in \mathbb{R}^D$ where $D = H \cdot W$.
3. **Norms & Normalization**: Compute the Euclidean $L_2$ norm $\|\mathbf{x}\|_2$ and map vectors onto the unit sphere $S^{D-1}$ to eliminate illumination and stroke-intensity bias.
4. **Subspace Modeling via SVD/PCA**: Compute the empirical covariance matrix and perform full SVD ($X_c = U \Sigma V^T$) to extract principal eigen-directions (eigen-digits).
5. **Orthogonal Projection**: Derive the subspace projection operator $P_k = V_k V_k^T$ and project centered digit vectors into a $k$-dimensional orthogonal subspace $\mathbb{R}^k$.
6. **Inner-Product Metric Similarity**: Compute inner products and cosine similarity scores $S(\mathbf{u}, \mathbf{v}) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ against stored pattern centroids.
7. **Mathematical Verification**: Formally verify the mathematical invariants: basis orthonormality ($V_k^T V_k = I_k$), projection idempotence ($P^2 = P$), self-adjointness ($P^T = P$), Cauchy-Schwarz inequality ($|\langle \mathbf{u}, \mathbf{v} \rangle| \le \|\mathbf{u}\| \|\mathbf{v}\|$), and Parseval energy conservation.

---

## 2. Sequential Mathematical Computational Process

The project strictly follows the prescribed 8-stage sequential process where the output of each stage serves directly as the input to the subsequent stage:

$$\begin{aligned}
\text{Digit Image} &\xrightarrow{\text{Stage 1}} \text{Grayscale Sampling} \\
&\xrightarrow{\text{Stage 2}} \text{Matrix Representation } A \in \mathbb{R}^{H \times W} \\
&\xrightarrow{\text{Stage 3}} \text{Vectorization } \mathbf{x} = \text{vec}(A) \in \mathbb{R}^D \\
&\xrightarrow{\text{Stage 4}} \text{Normalization } \hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} \in S^{D-1} \\
&\xrightarrow{\text{Stage 5}} \text{PCA / SVD Analysis } X_c = U \Sigma V^T \\
&\xrightarrow{\text{Stage 6}} \text{Orthogonal Projection } \mathbf{z} = (\hat{\mathbf{x}} - \boldsymbol{\mu}) V_k \in \mathbb{R}^k \\
&\xrightarrow{\text{Stage 7}} \text{Inner-Product Similarity } S(\mathbf{z}, \bar{\mathbf{z}}_c) = \frac{\langle \mathbf{z}, \bar{\mathbf{z}}_c \rangle}{\|\mathbf{z}\|_2 \|\bar{\mathbf{z}}_c\|_2} \\
&\xrightarrow{\text{Stage 8}} \text{Prediction } \hat{y} = \arg\max_c S(\mathbf{z}, \bar{\mathbf{z}}_c)
\end{aligned}$$

---

## 3. Detailed Mathematical Formulation

### 3.1 Stage 1 & 2: Matrices and Digit Representation
A handwritten digit image of dimension $H \times W$ (here $20 \times 20$) is mathematically defined as a 2D discrete intensity field:
$$I: \{1, \dots, H\} \times \{1, \dots, W\} \to [0, 255]$$
This induces the image matrix $A \in \mathbb{R}^{H \times W}$:
$$A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1W} \\
a_{21} & a_{22} & \cdots & a_{2W} \\
\vdots & \vdots & \ddots & \vdots \\
a_{H1} & a_{H2} & \cdots & a_{HW}
\end{bmatrix}$$
where $a_{ij} \in \mathbb{R}_{\ge 0}$ represents the grayscale light intensity at row $i$ and column $j$.

### 3.2 Stage 3: Vector Spaces and Vectorization
The vector space of real matrices $M_{H \times W}(\mathbb{R})$ is isomorphic to the Euclidean space $\mathbb{R}^D$ where $D = H \cdot W = 20 \times 20 = 400$.  
The vectorization operator $\text{vec}: \mathbb{R}^{H \times W} \to \mathbb{R}^D$ is a linear isomorphism defined by:
$$\mathbf{x} = \text{vec}(A) = \left[ a_{11}, a_{12}, \dots, a_{1W}, a_{21}, \dots, a_{HW} \right]^T \in \mathbb{R}^{400}$$
Linearity ensures:
1. $\text{vec}(A + B) = \text{vec}(A) + \text{vec}(B)$
2. $\text{vec}(\alpha A) = \alpha \text{vec}(A), \quad \forall \alpha \in \mathbb{R}$

### 3.3 Stage 4: Vector Norms and Normalization
To prevent classifier sensitivity to overall stroke intensity (brightness), vectors are mapped to the unit hypersphere $S^{D-1} \subset \mathbb{R}^D$.
The Euclidean ($L_2$) norm on $\mathbb{R}^D$ is induced by the standard inner product:
$$\|\mathbf{x}\|_2 = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle} = \sqrt{\sum_{i=1}^D x_i^2}$$
The normalized unit vector $\hat{\mathbf{x}}$ is:
$$\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}, \quad \text{such that } \|\hat{\mathbf{x}}\|_2 = 1$$
**Key Geometric Property**: When vectors have unit norm, the Euclidean distance squared is directly proportional to their inner product:
$$\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2 = \|\hat{\mathbf{u}}\|_2^2 + \|\hat{\mathbf{v}}\|_2^2 - 2 \langle \hat{\mathbf{u}}, \hat{\mathbf{v}} \rangle = 2 - 2 \langle \hat{\mathbf{u}}, \hat{\mathbf{v}} \rangle$$

### 3.4 Stage 5: SVD and Principal Component Analysis (PCA)
Let $X \in \mathbb{R}^{N \times D}$ be the matrix of $N = 60$ normalized training samples.
1. **Centering**: Compute the empirical sample mean vector:
   $$\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \hat{\mathbf{x}}_i \in \mathbb{R}^D$$
   The mean-centered data matrix is:
   $$X_c = X - \mathbf{1}_N \boldsymbol{\mu}^T \in \mathbb{R}^{N \times D}$$

2. **Singular Value Decomposition (SVD)**:
   By the Fundamental Theorem of Linear Algebra, any real matrix $X_c$ admits a factorization:
   $$X_c = U \Sigma V^T$$
   where:
   - $U \in \mathbb{R}^{N \times N}$ is an orthogonal matrix of left singular vectors ($U^T U = I_N$).
   - $\Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_{\min(N, D)}) \in \mathbb{R}^{N \times D}$ contains singular values ordered as $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
   - $V \in \mathbb{R}^{D \times D}$ is an orthogonal matrix of right singular vectors ($V^T V = I_D$).

3. **Covariance Relation & Eigenvalues**:
   The sample covariance matrix $C \in \mathbb{R}^{D \times D}$ is:
   $$C = \frac{1}{N - 1} X_c^T X_c = \frac{1}{N - 1} (V \Sigma^T U^T)(U \Sigma V^T) = V \left( \frac{\Sigma^T \Sigma}{N - 1} \right) V^T$$
   Thus, the columns of $V$ are precisely the orthonormal eigenvectors of the covariance matrix, and the sample variances (eigenvalues) along each principal direction are:
   $$\lambda_i = \frac{\sigma_i^2}{N - 1}$$
   The explained variance ratio for the $i$-th component is:
   $$\eta_i = \frac{\lambda_i}{\sum_{j=1}^r \lambda_j} = \frac{\sigma_i^2}{\sum_{j=1}^r \sigma_j^2}$$

### 3.5 Stage 6: Orthogonal Subspace Projection
We retain the top $k = 15$ principal axes forming the truncated basis matrix $V_k \in \mathbb{R}^{D \times k}$:
$$V_k = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k]$$
1. **Subspace Coordinate Representation**:
   For any centered digit vector $\mathbf{x}_c = \hat{\mathbf{x}} - \boldsymbol{\mu} \in \mathbb{R}^D$, its low-dimensional coordinate vector $\mathbf{z} \in \mathbb{R}^k$ is obtained via orthogonal projection:
   $$\mathbf{z} = \mathbf{x}_c V_k = [ \langle \mathbf{x}_c, \mathbf{v}_1 \rangle, \langle \mathbf{x}_c, \mathbf{v}_2 \rangle, \dots, \langle \mathbf{x}_c, \mathbf{v}_k \rangle ]$$

2. **Orthogonal Projection Operator**:
   The operator that orthogonally projects any high-dimensional vector onto the subspace $\mathcal{S}_k = \text{span}\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ is:
   $$P_k = V_k V_k^T \in \mathbb{R}^{D \times D}$$
   Properties of $P_k$:
   - **Idempotent**: $P_k^2 = (V_k V_k^T)(V_k V_k^T) = V_k (V_k^T V_k) V_k^T = V_k I_k V_k^T = P_k$.
   - **Self-Adjoint (Symmetric)**: $P_k^T = (V_k V_k^T)^T = V_k V_k^T = P_k$.

3. **Reconstruction & Projection Residual**:
   The reconstructed digit in high-dimensional space is:
   $$\tilde{\mathbf{x}} = \boldsymbol{\mu} + \mathbf{z} V_k^T = \boldsymbol{\mu} + P_k \mathbf{x}_c$$
   By the Orthogonal Projection Theorem, the residual vector $(\mathbf{x}_c - P_k \mathbf{x}_c)$ is strictly orthogonal to the subspace $\mathcal{S}_k$:
   $$\langle P_k \mathbf{x}_c, \mathbf{x}_c - P_k \mathbf{x}_c \rangle = 0$$

### 3.6 Stage 7: Inner-Product Similarity
In the $k$-dimensional feature space, each digit class $c \in \{0, 1, \dots, 9\}$ is represented by its class prototype centroid $\bar{\mathbf{z}}_c$:
$$\bar{\mathbf{z}}_c = \frac{1}{N_c} \sum_{i \in \text{class } c} \mathbf{z}_i \in \mathbb{R}^k$$

Given an unknown test digit with feature vector $\mathbf{z}_{\text{test}} \in \mathbb{R}^k$:
1. **Raw Inner Product**:
   $$\langle \mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c \rangle = \mathbf{z}_{\text{test}}^T \bar{\mathbf{z}}_c = \sum_{j=1}^k z_{\text{test}, j} \cdot \bar{z}_{c, j}$$
2. **Cosine Similarity**:
   $$S(\mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c) = \frac{\langle \mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c \rangle}{\|\mathbf{z}_{\text{test}}\|_2 \|\bar{\mathbf{z}}_c\|_2} = \cos \theta_c \in [-1, 1]$$
   By the **Cauchy-Schwarz Inequality**:
   $$|\langle \mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c \rangle| \le \|\mathbf{z}_{\text{test}}\|_2 \|\bar{\mathbf{z}}_c\|_2 \implies -1 \le S(\mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c) \le +1$$

### 3.7 Stage 8: Prediction Decision
The unknown digit is assigned to the class exhibiting maximum cosine similarity:
$$\hat{y} = \arg\max_{c \in \{0, 1, \dots, 9\}} S(\mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c)$$
A calibrated confidence distribution is computed using the Gibbs-Boltzmann / Softmax distribution with temperature $\tau = 0.2$:
$$p(c \mid \mathbf{z}_{\text{test}}) = \frac{\exp\left( \frac{S(\mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_c)}{\tau} \right)}{\sum_{j=0}^9 \exp\left( \frac{S(\mathbf{z}_{\text{test}}, \bar{\mathbf{z}}_j)}{\tau} \right)}$$

---

## 4. Program Architecture and Implementation

```
handwriting/
├── dataset/
│   ├── train/                 # 60 images (10 digits x 6 variations)
│   └── test/                  # 30 novel test images (10 digits x 3 variations)
├── outputs/                   # High-resolution generated plots & charts
│   ├── 1_digit_matrix.png
│   ├── 2_scree_plot.png
│   ├── 3_eigen_digits.png
│   ├── 4_pca_subspace.png
│   ├── 5_similarity_scores.png
│   ├── 6_reconstruction.png
│   └── 7_confusion_matrix.png
├── src/
│   ├── __init__.py
│   ├── preprocessor.py        # Matrix rep, vectorization, L2 normalization
│   ├── pca_svd.py             # SVD, covariance, orthogonal projection
│   ├── similarity.py          # Inner products, cosine similarity, predictions
│   ├── verification.py        # Mathematical proofs & invariant verifiers
│   └── visualizer.py          # Publication-grade plotting routines
├── tests/
│   └── test_math_pipeline.py  # 9 Pytest unit tests verifying linear algebra
├── templates/
│   └── index.html             # Interactive HTML5/Bootstrap web interface
├── app.py                     # Flask web server with live canvas
├── gui.py                     # Tkinter desktop GUI for drawing & analysis
├── main.py                    # Terminal pipeline driver with complete math trace
├── evaluate.py                # Full test-set evaluator and confusion matrix generator
├── generate_dataset.py        # Handwritten digit generator (PIL vector curves)
└── REPORT.md                  # Comprehensive academic report
```

---

## 5. Experimental Results and Analysis

### 5.1 Dimensionality Reduction and Explained Variance
With $N = 60$ training images of size $20 \times 20$ ($D = 400$):
- **Top 5 Singular Values**: $\sigma_1 = 2.431, \sigma_2 = 1.892, \sigma_3 = 1.762, \sigma_4 = 1.393, \sigma_5 = 1.364$.
- **At $k = 15$ Components**: Retains **88.42%** of total dataset variance while compressing the dimension from $400 \to 15$ (a **96.25% dimensionality reduction**).

### 5.2 Step-by-Step Computational Trace on Test Digit '7'
- **Stage 1 (Image Input)**: `dataset/test/test_7_sample_6.png`
- **Stage 2 (Matrix)**: $A \in \mathbb{R}^{20 \times 20}$, intensity values in $[0, 245.0]$.
- **Stage 3 (Vectorization)**: $\mathbf{x} = \text{vec}(A) \in \mathbb{R}^{400}$.
- **Stage 4 (Normalization)**: Computed norm $\|\mathbf{x}\|_2 = 1336.3106 \implies \|\hat{\mathbf{x}}\|_2 = 1.000000$.
- **Stage 5 & 6 (Projection)**: Low-dimensional coordinates in $\mathbb{R}^{15}$:
  $$z = [-0.314, +0.026, -0.086, +0.233, +0.261, -0.169, -0.081, -0.110, -0.200, +0.033, +0.219, -0.083, +0.013, +0.007, -0.024]^T$$
- **Stage 7 & 8 (Inner Product & Cosine Similarity Ranking)**:

| Digit Class | Raw Inner Product $\langle \mathbf{z}, \bar{\mathbf{z}}_c \rangle$ | Cosine Similarity $S(\mathbf{z}, \bar{\mathbf{z}}_c)$ | Euclidean Distance $d$ | Confidence | Decision |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Digit 7** | **+0.2017** | **+0.5407** | **0.5854** | **44.46%** | **PREDICTED [CORRECT]** |
| Digit 2 | +0.1448 | +0.4636 | 0.5875 | 30.23% | Runner-up |
| Digit 3 | +0.0985 | +0.3195 | 0.6564 | 14.71% | Third |
| Digit 1 | +0.0248 | +0.0727 | 0.7978 | 4.28% | - |
| Digit 5 | -0.0111 | -0.0361 | 0.8045 | 2.49% | - |
| Digit 9 | -0.0289 | -0.1227 | 0.7614 | 1.61% | - |
| Digit 0 | -0.0789 | -0.2657 | 0.8759 | 0.79% | - |
| Digit 6 | -0.1175 | -0.3153 | 0.9899 | 0.62% | - |
| Digit 4 | -0.1273 | -0.3793 | 0.9642 | 0.45% | - |
| Digit 8 | -0.1063 | -0.4151 | 0.8727 | 0.37% | - |

**1-NN Stored Prototype Match**: Correctly matched with training sample of Digit 7 with similarity $S = 0.9479$.

### 5.3 Overall Classification Accuracy
Evaluated over 30 unseen handwritten test digits (3 novel variations per class):
- **Overall Accuracy**: **90.00%** (27 / 30 correctly classified).
- Digits 0, 1, 2, 3, 4, 5, 6 achieved **100% accuracy**.
- High similarity score margins confirm robust separation in the reduced eigenspace.

---

## 6. Computational Efficiency & Optimization Analysis

A dedicated performance profiling benchmark (`benchmark.py`) was executed over 500 test trials to evaluate computational speed, algorithmic complexity, and throughput:

### 6.1 Algorithmic Complexity Reductions
1. **Dimensionality Reduction**: SVD reduces vector space from $D = 400 \to k = 15$ dimensions, achieving a **96.25% reduction** in storage and bandwidth (120 bytes vs 3,200 bytes per digit).
2. **Inner-Product Complexity**:
   - High-dimensional space: $400 \times 10 = 4,000$ multiplications.
   - Reduced eigenspace: $15 \times 10 = 150$ multiplications (**26.7x FLOPs speedup**).
3. **Pythagorean Orthogonal Residual Calculation**:
   Instead of performing a full $O(D \cdot k)$ reconstruction matrix product $\tilde{\mathbf{x}} = \boldsymbol{\mu} + \mathbf{z} V_k^T$ followed by an $O(D)$ norm, we compute the residual using the Pythagorean theorem for orthogonal projectors in $O(D)$:
   $$\|\mathbf{x}_c - P_k \mathbf{x}_c\|_2 = \sqrt{\|\mathbf{x}_c\|_2^2 - \|\mathbf{z}\|_2^2}$$
4. **Zero-Overhead Static Theorem Verification**:
   The basis orthonormality ($V_k^T V_k = I_k$) and projection matrix idempotence ($P_k^2 = P_k$) depend solely on $V_k$. They are precomputed once at startup, eliminating over 64,000,000 floating-point matrix operations from every real-time query.

### 6.2 Empirical Benchmark Results (`benchmark.py`)
| Performance Metric | Measured Value | Architectural Benefit |
| :--- | :--- | :--- |
| **Average Inference Latency** | **0.097 ms** (&lt; 0.1 ms) | Sub-millisecond real-time response |
| **Median Inference Latency** | **0.072 ms** | Deterministic linear algebra performance |
| **Throughput (Single CPU Core)** | **10,360 digits / second** | Massive scalability under high query load |
| **Vectorized BLAS Dot Product** | **5.66 µs** (vs 67.78 µs loop) | **12.0x acceleration** via SIMD instructions |
| **RAM Cache Sample Access** | **&lt; 0.01 ms** | $O(1)$ memory lookup with zero disk I/O |

---

## 7. Formal Mathematical Verifications

The software executes formal linear algebraic proofs and invariant verifications:

| # | Mathematical Theorem / Invariant | Governing Equation | Observed Numerical Error | Status |
| :--- | :--- | :--- | :--- | :---: |
| 1 | **Right Singular Vectors Orthonormality** | $V_k^T V_k = I_k$ | $\|V_k^T V_k - I_k\|_F = 5.11 \times 10^{-15}$ | **PASSED** |
| 2 | **Projection Matrix Idempotence** | $P_k^2 = P_k$ | $\|P_k^2 - P_k\|_F = 5.10 \times 10^{-15}$ | **PASSED** |
| 3 | **Projection Matrix Self-Adjointness** | $P_k^T = P_k$ | $\|P_k^T - P_k\|_F = 0.00$ | **PASSED** |
| 4 | **Cauchy-Schwarz Inequality** | $|\langle \mathbf{u}, \mathbf{v} \rangle| \le \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$ | $|\langle \mathbf{u}, \mathbf{v} \rangle| = 0.2017 \le 0.3730$ | **PASSED** |
| 5 | **SVD Parseval Energy Conservation** | $\|X_c\|_F^2 = \sum_{i=1}^r \sigma_i^2$ | Relative Error $= 1.48 \times 10^{-15}$ | **PASSED** |
| 6 | **Monotonic Error Reduction** | $e_{k+1} \le e_k$ | $e(1)=0.6148 \to e(20)=0.2679$ | **PASSED** |

---

## 7. Viva Voce Questions & Answers (Comprehensive Preparation)

### Q1: What is the mathematical justification for vectorizing a 2D image matrix?
**Answer:** The set of real matrices $M_{H \times W}(\mathbb{R})$ under standard matrix addition and scalar multiplication forms a real vector space of dimension $D = H \cdot W$. By the fundamental theorem of linear algebra, any finite-dimensional real vector space of dimension $D$ is linearly isomorphic to the Euclidean space $\mathbb{R}^D$ ($\mathbb{R}^{H \times W} \cong \mathbb{R}^D$). The mapping $\text{vec}(\cdot)$ preserves vector addition ($\text{vec}(A+B) = \text{vec}(A) + \text{vec}(B)$) and scalar multiplication ($\text{vec}(\alpha A) = \alpha \text{vec}(A)$), enabling all linear algebraic operations (inner products, orthogonal projections, SVD) to be performed.

### Q2: Why is Euclidean $L_2$ normalization performed before applying PCA?
**Answer:** Without normalization, the magnitude of a vectorized digit depends on ink intensity, stroke width, and overall illumination ($a_{ij} \in [0, 255]$). Two identical digits drawn with different pressures would have large Euclidean distance. By projecting onto the unit hypersphere $S^{D-1}$ via $\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$, the distance between two vectors becomes purely a function of the angle between them: $\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2 = 2 - 2 \cos \theta$.

### Q3: How is Singular Value Decomposition (SVD) connected to Principal Component Analysis (PCA)?
**Answer:** Let $X_c \in \mathbb{R}^{N \times D}$ be the mean-centered data matrix. The sample covariance matrix is $C = \frac{1}{N-1} X_c^T X_c$.  
Using the SVD factorization $X_c = U \Sigma V^T$:
$$C = \frac{1}{N-1} (V \Sigma^T U^T)(U \Sigma V^T) = V \left( \frac{\Sigma^T \Sigma}{N-1} \right) V^T$$
Because $V$ is orthogonal and $\Sigma^T \Sigma$ is diagonal, this is the exact eigendecomposition of $C$. The right singular vectors (columns of $V$) are the eigenvectors of $C$ (principal component axes), and the sample variances are $\lambda_i = \frac{\sigma_i^2}{N-1}$. Thus, computing SVD on $X_c$ directly yields PCA without having to explicitly form the large $D \times D$ covariance matrix.

### Q4: Prove that the subspace projection matrix $P_k = V_k V_k^T$ is idempotent and symmetric.
**Answer:**
1. **Idempotence**:
   $$P_k^2 = (V_k V_k^T)(V_k V_k^T) = V_k (V_k^T V_k) V_k^T$$
   Since the columns of $V_k$ are orthonormal, $V_k^T V_k = I_k$. Therefore:
   $$P_k^2 = V_k I_k V_k^T = V_k V_k^T = P_k$$
2. **Symmetry**:
   $$P_k^T = (V_k V_k^T)^T = (V_k^T)^T V_k^T = V_k V_k^T = P_k$$
An operator that is both idempotent and self-adjoint is by definition an **orthogonal projector**.

### Q5: What is the geometric meaning of the Cauchy-Schwarz inequality in digit matching?
**Answer:** The Cauchy-Schwarz inequality states that for any two vectors $\mathbf{u}, \mathbf{v}$ in an inner product space:
$$|\langle \mathbf{u}, \mathbf{v} \rangle| \le \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$$
Dividing by the product of norms gives:
$$-1 \le \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \le +1$$
This guarantees that the cosine of the angle $\cos \theta = S(\mathbf{u}, \mathbf{v})$ is well-defined and strictly bounded in $[-1, 1]$. A score of $+1$ indicates identical feature directions, $0$ indicates orthogonal features (disjoint digit patterns), and negative values indicate opposite spatial distributions.

### Q6: Why do we project into a low-dimensional subspace ($k = 15$) instead of comparing directly in $\mathbb{R}^{400}$?
**Answer:**
1. **Curse of Dimensionality**: High-dimensional spaces suffer from distance concentration, where the ratio of distances between the nearest and farthest neighbors approaches 1.
2. **Noise and Redundancy Filtering**: The raw 400 dimensions contain high-frequency stroke jitter, translation variations, and background pixels. SVD concentrates the genuine structural variance of digit shapes into the leading $k$ singular vectors, discarding noise in the null/tail space.
3. **Computational Efficiency**: Comparing 15-dimensional vectors requires $15$ multiplications versus $400$, representing a $26.7\times$ speedup.

### Q7: What are "Eigen-digits"?
**Answer:** An eigen-digit is the 2D visual representation of a right singular vector $\mathbf{v}_i \in \mathbb{R}^D$ reshaped back into an $H \times W$ matrix. They form an orthonormal basis for the digit image space. Any digit can be expressed as a linear combination of these eigen-digits plus the mean digit:
$$\tilde{\mathbf{x}} = \boldsymbol{\mu} + \sum_{i=1}^k z_i \mathbf{v}_i$$

### Q8: What is Parseval's Identity in this context?
**Answer:** In linear algebra, Parseval's identity states that the Frobenius norm of a matrix equals the square root of the sum of the squares of its singular values:
$$\|X_c\|_F^2 = \text{Tr}(X_c^T X_c) = \sum_{i=1}^{\min(N, D)} \sigma_i^2$$
This proves that SVD completely accounts for the total energy / variance in the dataset.

---

## 8. Conclusion
The mini-project successfully implemented an end-to-end mathematical analyzer for handwritten digit classification based strictly on foundational linear algebra concepts:
- Converted discrete grayscale image matrices into Euclidean vector spaces.
- Normalized vector representations to eliminate stroke-intensity variability.
- Applied SVD to discover orthogonal principal axes and achieved 88.42% explained variance compression at $k = 15$.
- Used orthogonal projection and inner-product cosine similarity to classify unseen handwritten digits with **90.00% accuracy**.
- Verified all underlying theorems (Orthonormality, Idempotence, Self-Adjointness, Cauchy-Schwarz, and Energy Conservation) with numerical errors bounded below $10^{-14}$.
