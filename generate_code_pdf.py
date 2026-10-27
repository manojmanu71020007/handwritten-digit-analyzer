"""
generate_code_pdf.py
Generates a clean, simple Black & White PDF containing the complete Python source code
used to build and run the Handwritten Digit Analyzer website.
Design: Pure monochrome (Black text on White background), standard monospaced layout.
"""

import os
from fpdf import FPDF, XPos, YPos

class BlackWhiteCodePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(0, 0, 0)
        self.cell(0, 7, "Handwritten Digit Analyzer - Python Website Source Code", border="B", align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(0, 0, 0)
        self.cell(0, 6, f"Page {self.page_no()} of {{nb}}", border="T", align="C")

def create_code_pdf(output_filename="Website_Python_Code_BW.pdf"):
    pdf = BlackWhiteCodePDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()

    files_to_include = [
        ("1. app.py (Flask Web Application & API Server)", "app.py"),
        ("2. src/preprocessor.py (Image Preprocessor, Vectorization & L2 Normalization)", "src/preprocessor.py"),
        ("3. src/pca_svd.py (SVD/PCA Engine & Orthogonal Subspace Projection)", "src/pca_svd.py"),
        ("4. src/similarity.py (Inner Product & Cosine Similarity Classifier)", "src/similarity.py"),
    ]

    # Title Page / Header
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "PYTHON SOURCE CODE DOCUMENTATION", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, "Handwritten Digit Feature and Similarity Analyzer Website", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, "Pure Black & White Code Archive", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Included Python Modules:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    for title, path in files_to_include:
        size = os.path.getsize(path) if os.path.exists(path) else 0
        pdf.cell(0, 5, f" - {title} ({size:,} bytes)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # Append Each File
    for title, path in files_to_include:
        if not os.path.exists(path):
            continue

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, title, border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

        with open(path, "r", encoding="utf-8") as f:
            code_lines = f.readlines()

        pdf.set_font("Courier", "", 7.5)
        line_num = 1
        for line in code_lines:
            clean_line = line.rstrip("\r\n").replace("\t", "    ")
            # Replace non-latin1 unicode characters with ascii equivalents for standard Courier font
            safe_line = (
                clean_line
                .replace("ℝ", "R")
                .replace("⁴⁰⁰", "^400")
                .replace("¹⁵", "^15")
                .replace("²⁰", "^20")
                .replace("²", "^2")
                .replace("ᵀ", "^T")
                .replace("₁", "_1")
                .replace("₂", "_2")
                .replace("₃", "_3")
                .replace("₃D", "_3D")
                .replace("—", "-")
                .replace("–", "-")
                .replace("“", '"')
                .replace("”", '"')
                .replace("’", "'")
                .replace("≤", "<=")
                .replace("≥", ">=")
                .replace("µ", "u")
                .replace("σ", "sigma")
                .replace("μ", "mu")
                .replace("θ", "theta")
                .replace("Σ", "Sigma")
                .replace("⚡", "")
            )
            prefix = f"{line_num:4d} | "
            # Wrap long lines cleanly
            if len(safe_line) > 105:
                # Print in chunks
                pdf.cell(0, 3.6, prefix + safe_line[:105], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                indent = "     |   "
                pdf.cell(0, 3.6, indent + safe_line[105:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(0, 3.6, prefix + safe_line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            line_num += 1

        pdf.ln(6)

    pdf.output(output_filename)
    print(f"[OK] Generated Black & White Code PDF: {output_filename}")

if __name__ == "__main__":
    create_code_pdf()
