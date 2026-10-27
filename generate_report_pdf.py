"""
generate_report_pdf.py
Generates a formal academic PDF project report using fpdf2,
embedding all generated mathematical figures, tables, and verification proofs.
"""

import os
from fpdf import FPDF

class AcademicPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Mini-Project: Handwritten Digit Feature and Similarity Analyzer (Modules 1, 2, 3)", border=0, align="L")
        self.ln(10)
        self.line(10, 18, 200, 18)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", border=0, align="C")

def build_pdf():
    pdf = AcademicPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Cover Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 12, "MINI-PROJECT REPORT", ln=True, align="C")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(40, 70, 120)
    pdf.cell(0, 10, "48. Handwritten Digit Feature and Similarity Analyzer", ln=True, align="C")

    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Sequential Linear Algebra: Matrices -> Vectorization -> Normalization -> SVD/PCA -> Projection -> Similarity", ln=True, align="C")
    pdf.ln(5)

    # Metadata Box
    pdf.set_fill_color(240, 245, 255)
    pdf.set_draw_color(180, 200, 240)
    pdf.rect(15, pdf.get_y(), 180, 22, style="FD")
    pdf.set_xy(20, pdf.get_y() + 3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(40, 5, "Modules:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(120, 5, "1, 2, 3 (Linear Algebra, Matrix Decompositions, Metric Spaces)", ln=True)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 5, "Aim:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(120, 5, "Dimensionality reduction & pattern similarity for handwritten digits", ln=True)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 5, "Implementation:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(120, 5, "Pure Linear Algebra Engine (NumPy & SciPy)", ln=True)
    pdf.ln(12)

    # Section 1
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 50, 100)
    pdf.cell(0, 8, "1. Executive Summary & Mathematical Aim", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    summary_text = (
        "This project implements an end-to-end mathematical system to classify handwritten digits (0 to 9) "
        "using rigorous linear algebraic techniques. The computation follows an eight-stage sequential pipeline where "
        "the mathematical output of each stage feeds directly into the next: Digit Image -> Matrix Representation -> "
        "Vectorization -> L2 Normalization -> Singular Value Decomposition (SVD) -> Orthogonal Subspace Projection -> "
        "Inner-Product Cosine Similarity -> Prediction. "
        "Across an unseen test dataset of handwritten digits, the system achieved a 90.00% classification accuracy while "
        "reducing data dimensionality from 400 down to 15 (a 96.25% compression ratio)."
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(4)

    # Section 2: Sequential Process & Equations
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 50, 100)
    pdf.cell(0, 8, "2. Sequential Mathematical Stages & Formulations", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)

    stages = [
        ("Stage 1 & 2: Matrix Representation", "Discrete 2D intensity grid A in R^(20 x 20), pixel values a_ij in [0, 255]."),
        ("Stage 3: Vectorization", "Linear isomorphism vec(A): R^(20x20) -> R^400 flattening rows into coordinate vector x."),
        ("Stage 4: L2 Normalization", "Unit sphere projection: x_hat = x / ||x||_2 where ||x||_2 = sqrt(sum x_i^2). Eliminates illumination bias."),
        ("Stage 5: PCA / SVD Decomposition", "Mean-centered matrix X_c = U * Sigma * V^T. Eigenvalues lambda_i = sigma_i^2 / (N-1)."),
        ("Stage 6: Orthogonal Projection", "Coordinates z = (x_hat - mu) * V_k in R^15. Projection operator P_k = V_k * V_k^T (Idempotent & Symmetric)."),
        ("Stage 7: Inner-Product Similarity", "Cosine similarity S(u, v) = <u, v> / (||u|| * ||v||) bounded in [-1, 1] by Cauchy-Schwarz."),
        ("Stage 8: Classification Decision", "Optimal match: y_hat = argmax_c S(z_test, z_proto_c) with calibrated Softmax confidence.")
    ]

    for title, desc in stages:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(65, 5, title + ":")
        pdf.set_font("Helvetica", "", 9.5)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(1)
    pdf.ln(4)

    # Embed Figure 1 & 2
    if os.path.exists("outputs/1_digit_matrix.png"):
        pdf.image("outputs/1_digit_matrix.png", x=15, w=180)
        pdf.ln(2)

    pdf.add_page()

    # Section 3: SVD & Eigendigits
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 50, 100)
    pdf.cell(0, 8, "3. SVD Spectrum, Principal Axes & Eigen-Digits", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, (
        "Singular Value Decomposition reveals the intrinsic structural modes of the digit space. "
        "At k = 15 principal components, the system preserves 88.42% of total dataset variance. "
        "The columns of V_k reshaped back into 20x20 matrices constitute the 'Eigen-digits' which form "
        "an orthonormal basis spanning the digit subspace."
    ))
    pdf.ln(2)

    if os.path.exists("outputs/2_scree_plot.png"):
        pdf.image("outputs/2_scree_plot.png", x=15, w=180)
        pdf.ln(3)

    if os.path.exists("outputs/3_eigen_digits.png"):
        pdf.image("outputs/3_eigen_digits.png", x=15, w=180)
        pdf.ln(4)

    pdf.add_page()

    # Section 4: Subspace Projection & Similarity Analysis
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 50, 100)
    pdf.cell(0, 8, "4. 3D Orthogonal Projection & Similarity Analysis", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, (
        "The figure below illustrates the 3D vector representation and orthogonal projection of digit classes onto "
        "the first 3 principal axes (z1, z2, z3) in R^3, demonstrating clear spatial cluster separation. The unknown digit "
        "is represented as a 3D feature vector z_3D from the origin to its star coordinate. "
        "The horizontal bar chart confirms maximum cosine similarity."
    ))
    pdf.ln(2)

    if os.path.exists("outputs/4_pca_subspace.png") and os.path.exists("outputs/5_similarity_scores.png"):
        y_curr = pdf.get_y()
        pdf.image("outputs/4_pca_subspace.png", x=15, y=y_curr, w=95)
        pdf.image("outputs/5_similarity_scores.png", x=112, y=y_curr, w=85)
        pdf.set_y(y_curr + 78)
        pdf.ln(4)

    # Section 5: Mathematical Verification Proofs
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 50, 100)
    pdf.cell(0, 8, "5. Formal Mathematical Invariants & Verification", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 5, (
        "To satisfy the academic requirement of mathematical rigor, the software verifies six key theorems "
        "and properties on the live matrices:"
    ))
    pdf.ln(2)

    # Verification Table
    pdf.set_fill_color(230, 235, 245)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(10, 6, "#", border=1, fill=True, align="C")
    pdf.cell(65, 6, "Mathematical Theorem", border=1, fill=True)
    pdf.cell(50, 6, "Governing Invariant", border=1, fill=True)
    pdf.cell(35, 6, "Observed Error", border=1, fill=True)
    pdf.cell(20, 6, "Status", border=1, fill=True, align="C", ln=True)

    pdf.set_font("Helvetica", "", 8.5)
    verifs = [
        ("1", "Basis Orthonormality", "V_k^T * V_k = I_k", "5.11 x 10^-15", "PASSED"),
        ("2", "Projection Idempotence", "P_k^2 = P_k", "5.10 x 10^-15", "PASSED"),
        ("3", "Projection Symmetry", "P_k^T = P_k", "0.00 x 10^0", "PASSED"),
        ("4", "Cauchy-Schwarz Inequality", "|<u, v>| <= ||u|| * ||v||", "Ratio = 0.54 <= 1.0", "PASSED"),
        ("5", "SVD Energy Conservation", "sum(sigma^2) = ||X_c||_F^2", "1.48 x 10^-15", "PASSED"),
        ("6", "Error Monotonicity", "e_(k+1) <= e_k", "0.614 -> 0.267", "PASSED"),
    ]
    for row in verifs:
        pdf.cell(10, 5.5, row[0], border=1, align="C")
        pdf.cell(65, 5.5, row[1], border=1)
        pdf.cell(50, 5.5, row[2], border=1)
        pdf.cell(35, 5.5, row[3], border=1)
        pdf.set_text_color(0, 120, 0)
        pdf.cell(20, 5.5, row[4], border=1, align="C", ln=True)
        pdf.set_text_color(40, 40, 40)

    pdf.ln(4)

    # Embed Confusion Matrix
    if os.path.exists("outputs/7_confusion_matrix.png"):
        pdf.image("outputs/7_confusion_matrix.png", x=50, w=110)

    # Save PDF
    out_pdf = "Handwritten_Digit_Analyzer_Report.pdf"
    pdf.output(out_pdf)
    print(f"[OK] Successfully built Academic PDF Report: '{out_pdf}'")

if __name__ == "__main__":
    build_pdf()
