"""
gui.py
Interactive Desktop Graphical User Interface (Tkinter)
Allows drawing a digit or loading test samples, and executes the 8-stage mathematical pipeline.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

from src.preprocessor import DigitPreprocessor
from src.pca_svd import PCASVDAnalyzer
from src.similarity import SimilarityClassifier

class DigitAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Feature and Similarity Analyzer (Linear Algebra)")
        self.root.geometry("1100x750")
        self.root.minsize(1000, 700)

        # Mathematical Engine
        self.target_size = (20, 20)
        self.preprocessor = DigitPreprocessor(target_size=self.target_size)
        self.pca = None
        self.classifier = None
        self.k = 15

        # Drawing canvas state
        self.canvas_size = 200
        self.drawing_img = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.drawing_draw = ImageDraw.Draw(self.drawing_img)
        self.current_matrix = np.zeros(self.target_size)

        # Train PCA on startup
        self.init_math_engine()

        # Build UI Layout
        self.build_ui()

    def init_math_engine(self):
        try:
            X_train, y_train = [], []
            train_dir = "dataset/train"
            if not os.path.exists(train_dir):
                return
            for f in sorted(os.listdir(train_dir)):
                if f.endswith(".png"):
                    c = int(f.split("_")[1])
                    mat = self.preprocessor.image_to_matrix(os.path.join(train_dir, f))
                    vec = self.preprocessor.vectorize(mat)
                    vec_n, _ = self.preprocessor.normalize_l2(vec)
                    X_train.append(vec_n)
                    y_train.append(c)
            X_train, y_train = np.array(X_train), np.array(y_train)

            self.pca = PCASVDAnalyzer(n_components=self.k).fit(X_train)
            Z_train = self.pca.project(X_train)
            self.Z_train = Z_train
            self.y_train = y_train
            self.classifier = SimilarityClassifier().fit_patterns(Z_train, y_train)
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to train PCA engine: {e}")

    def build_ui(self):
        # Top Header
        header = tk.Frame(self.root, bg="#1E293B", pady=12)
        header.pack(fill=tk.X)
        title_label = tk.Label(header, text="Handwritten Digit Feature & Similarity Analyzer", font=("Helvetica", 16, "bold"), fg="white", bg="#1E293B")
        title_label.pack()
        subtitle = tk.Label(header, text="Sequential Linear Algebra: Matrices -> Vectorization -> L2 Norm -> SVD/PCA -> Projection -> Inner-Product Similarity", font=("Helvetica", 9), fg="#94A3B8", bg="#1E293B")
        subtitle.pack()

        # Main Split Container
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Input (Drawing / File Selector)
        left_frame = tk.LabelFrame(main_frame, text=" 1. Input Digit (Draw or Load) ", font=("Helvetica", 11, "bold"), padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Canvas for Drawing
        self.canvas = tk.Canvas(left_frame, width=self.canvas_size, height=self.canvas_size, bg="black", cursor="cross")
        self.canvas.pack(pady=5)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # Controls under Canvas
        btn_frame1 = tk.Frame(left_frame)
        btn_frame1.pack(fill=tk.X, pady=4)
        clear_btn = tk.Button(btn_frame1, text="Clear Canvas", command=self.clear_canvas, width=12, bg="#EF4444", fg="white", font=("Helvetica", 9, "bold"))
        clear_btn.pack(side=tk.LEFT, padx=2)

        analyze_draw_btn = tk.Button(btn_frame1, text="Analyze Drawing", command=self.analyze_drawing, width=14, bg="#10B981", fg="white", font=("Helvetica", 9, "bold"))
        analyze_draw_btn.pack(side=tk.RIGHT, padx=2)

        # File Chooser
        sep = ttk.Separator(left_frame, orient="horizontal")
        sep.pack(fill=tk.X, pady=10)

        load_label = tk.Label(left_frame, text="Or select from Test Dataset:", font=("Helvetica", 9, "bold"))
        load_label.pack(anchor="w")

        test_dir = "dataset/test"
        sample_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")]) if os.path.exists(test_dir) else []
        self.selected_file_var = tk.StringVar(value=sample_files[0] if sample_files else "")
        self.file_combo = ttk.Combobox(left_frame, textvariable=self.selected_file_var, values=sample_files, state="readonly", width=25)
        self.file_combo.pack(fill=tk.X, pady=4)
        self.file_combo.bind("<<ComboboxSelected>>", self.on_select_sample)

        browse_btn = tk.Button(left_frame, text="Browse Custom Image...", command=self.browse_custom, font=("Helvetica", 9))
        browse_btn.pack(fill=tk.X, pady=3)

        # Right Column: Stages 2-8 Outputs
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Prediction Highlight Box
        self.pred_card = tk.Frame(right_frame, bg="#3B82F6", pady=8, padx=15)
        self.pred_card.pack(fill=tk.X, pady=(0, 10))
        self.pred_label = tk.Label(self.pred_card, text="PREDICTED DIGIT: ?", font=("Helvetica", 20, "bold"), fg="white", bg="#3B82F6")
        self.pred_label.pack(side=tk.LEFT)
        self.conf_label = tk.Label(self.pred_card, text="Confidence: --%", font=("Helvetica", 14), fg="#E0E7FF", bg="#3B82F6")
        self.conf_label.pack(side=tk.RIGHT)

        # Notebook for detailed stages
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Similarity Scores & Ranking
        tab_sim = ttk.Frame(notebook)
        notebook.add(tab_sim, text="Stage 7 & 8: Similarity Ranking")
        self.sim_tree = ttk.Treeview(tab_sim, columns=("digit", "cosine", "inner_prod", "dist", "prob"), show="headings", height=10)
        self.sim_tree.heading("digit", text="Digit Pattern")
        self.sim_tree.heading("cosine", text="Cosine Similarity S(u,v)")
        self.sim_tree.heading("inner_prod", text="Raw Inner Product <u,v>")
        self.sim_tree.heading("dist", text="Euclidean Distance")
        self.sim_tree.heading("prob", text="Softmax Confidence")
        for col in ("digit", "cosine", "inner_prod", "dist", "prob"):
            self.sim_tree.column(col, anchor="center")
        self.sim_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 2: Mathematical Vectors & PCA Coordinates
        tab_vec = ttk.Frame(notebook)
        notebook.add(tab_vec, text="Stage 3, 4, 5: Mathematical Vectors")
        self.vec_text = tk.Text(tab_vec, wrap="none", font=("Consolas", 9), padx=8, pady=8)
        self.vec_text.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Matrix Representation (ASCII / Grid)
        tab_mat = ttk.Frame(notebook)
        notebook.add(tab_mat, text="Stage 2: Matrix Representation")
        self.mat_text = tk.Text(tab_mat, wrap="none", font=("Courier New", 8), padx=8, pady=8)
        self.mat_text.pack(fill=tk.BOTH, expand=True)

        # Tab 4: Stage 6: 3D Vector Graph (R^3)
        self.tab_3d = ttk.Frame(notebook)
        notebook.add(self.tab_3d, text="Stage 6: 3D Vector Graph (R³)")
        self.fig_3d = Figure(figsize=(6, 5), dpi=95, facecolor="#F8FAFC")
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Trigger analysis on initial selection
        if sample_files:
            self.on_select_sample(None)

    def paint(self, event):
        brush_r = 9
        x1, y1 = event.x - brush_r, event.y - brush_r
        x2, y2 = event.x + brush_r, event.y + brush_r
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.drawing_draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing_img = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.drawing_draw = ImageDraw.Draw(self.drawing_img)
        self.pred_label.config(text="PREDICTED DIGIT: ?")
        self.conf_label.config(text="Confidence: --%")

    def analyze_drawing(self):
        # Resize drawn PIL image to 20x20
        img_20 = self.drawing_img.resize(self.target_size, Image.Resampling.BILINEAR)
        matrix = np.array(img_20, dtype=np.float64)
        if np.max(matrix) < 10.0:
            messagebox.showwarning("Empty Canvas", "Please draw a digit (0-9) on the canvas first.")
            return
        self.execute_pipeline(matrix, label_title="User Hand-Drawn Digit")

    def on_select_sample(self, event):
        fname = self.selected_file_var.get()
        if not fname:
            return
        path = os.path.join("dataset/test", fname)
        self.load_image_path(path)

    def browse_custom(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.load_image_path(path)

    def load_image_path(self, path):
        img = Image.open(path).convert("L")
        # Display on canvas
        img_disp = img.resize((self.canvas_size, self.canvas_size), Image.Resampling.NEAREST)
        self.tk_img = ImageTk.PhotoImage(img_disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.drawing_img = img_disp
        self.drawing_draw = ImageDraw.Draw(self.drawing_img)

        matrix = self.preprocessor.image_to_matrix(img)
        self.execute_pipeline(matrix, label_title=os.path.basename(path))

    def execute_pipeline(self, matrix: np.ndarray, label_title: str = ""):
        if self.pca is None or self.classifier is None:
            return

        # Stage 3: Vectorization
        x_vec = self.preprocessor.vectorize(matrix)
        # Stage 4: Normalization
        x_norm, l2_norm = self.preprocessor.normalize_l2(x_vec)
        # Stage 5 & 6: PCA Projection
        z = self.pca.project(x_norm)
        # Stage 7 & 8: Similarity & Prediction
        res = self.classifier.compare_unknown(z)

        # Update Prediction Card
        pred_digit = res["predicted_digit"]
        conf = res["confidence"] * 100
        self.pred_label.config(text=f"PREDICTED DIGIT: {pred_digit}")
        self.conf_label.config(text=f"Confidence: {conf:.1f}%")

        # Update Similarity Table
        for row in self.sim_tree.get_children():
            self.sim_tree.delete(row)

        for c in self.classifier.classes:
            tag = "max" if c == pred_digit else ""
            self.sim_tree.insert("", "end", values=(
                f"Digit {c}" + (" [TOP MATCH]" if c == pred_digit else ""),
                f"{res['similarity_scores'][c]:+.4f}",
                f"{res['inner_products'][c]:+.4f}",
                f"{res['euclidean_distances'][c]:.4f}",
                f"{res['probabilities'][c]*100:.2f}%"
            ), tags=(tag,))
        self.sim_tree.tag_configure("max", background="#D1FAE5", font=("Helvetica", 9, "bold"))

        # Update Vectors Tab
        self.vec_text.delete("1.0", tk.END)
        v_lines = [
            f"INPUT SOURCE: {label_title}",
            "=" * 70,
            f"STAGE 3: VECTORIZATION (Isomorphism R^(20x20) -> R^400)",
            f"  Vector Length: D = {len(x_vec)}",
            f"  First 10 Unnormalized Coordinates:\n  {np.round(x_vec[:10], 1)}",
            "",
            f"STAGE 4: NORMALIZATION (Unit Sphere)",
            f"  Computed Euclidean L2 Norm ||x||_2 = {l2_norm:.4f}",
            f"  Normalized Vector Norm ||x_hat||_2 = {np.linalg.norm(x_norm, 2):.6f}",
            "",
            "=" * 50,
            f"3D VECTOR REPRESENTATION (z_3D in R^3):",
            f"  z_1 = {z[0]:+.6f}",
            f"  z_2 = {z[1]:+.6f}",
            f"  z_3 = {z[2]:+.6f}",
            f"  z_3D = [{z[0]:+.4f}, {z[1]:+.4f}, {z[2]:+.4f}]^T",
            f"  3D Magnitude ||z_3D||_2 = {np.linalg.norm(z[:3]):.6f}",
            "=" * 50,
            "",
            f"STAGE 5 & 6: PCA / SVD SUBSPACE PROJECTION (k = {self.k})",
            f"  Reduced Coordinate Vector z in R^{self.k}:",
        ]
        for idx, val in enumerate(z):
            v_lines.append(f"    z_{idx+1:02d} = {val:+.6f}")
        v_lines.append(f"\nSubspace Reconstruction Residual: {self.pca.reconstruction_error(x_norm):.4f}")
        self.vec_text.insert(tk.END, "\n".join(v_lines))

        # Update Matrix Tab
        self.mat_text.delete("1.0", tk.END)
        m_lines = [
            f"STAGE 2: MATRIX REPRESENTATION A in R^(20 x 20)",
            "=" * 50,
            f"Dimensions: {matrix.shape[0]} rows x {matrix.shape[1]} columns",
            f"Intensity Min: {np.min(matrix):.1f}, Max: {np.max(matrix):.1f}",
            "",
            "Visual Shading Map:",
            self.preprocessor.format_ascii_matrix(matrix),
            "",
            "Central 6x6 Matrix Values A[7:13, 7:13]:",
            str(np.round(matrix[7:13, 7:13], 1))
        ]
        self.mat_text.insert(tk.END, "\n".join(m_lines))

        # Update 3D Vector Graph
        self.update_3d_plot(z, pred_digit)

    def update_3d_plot(self, z_test: np.ndarray, pred_digit: int):
        try:
            if not hasattr(self, "Z_train") or self.Z_train is None:
                return
            self.ax_3d.clear()
            cmap = plt.get_cmap("tab10")
            classes = sorted(list(set(self.y_train)))

            # 1. Plot training clusters in 3D
            for c in classes:
                idx = np.where(self.y_train == c)[0]
                self.ax_3d.scatter(
                    self.Z_train[idx, 0], self.Z_train[idx, 1], self.Z_train[idx, 2],
                    label=f"Digit {c}", s=32, color=cmap(c), alpha=0.55
                )

            # 2. Draw 3D Vector Line from Origin (0,0,0) to (z1, z2, z3)
            self.ax_3d.plot(
                [0, z_test[0]], [0, z_test[1]], [0, z_test[2]],
                color="#E11D48", linewidth=2.5, linestyle="-", label="Vector z_3D"
            )

            # 3. Draw Test Digit Marker
            self.ax_3d.scatter(
                [z_test[0]], [z_test[1]], [z_test[2]],
                color="#E11D48", edgecolor="white", linewidth=1.8, s=220, marker="*",
                label=f"Test Digit (Pred: {pred_digit})", zorder=20
            )

            # Vertical guide line
            z_min = float(np.min(self.Z_train[:, 2]))
            self.ax_3d.plot(
                [z_test[0], z_test[0]], [z_test[1], z_test[1]], [z_min, z_test[2]],
                color="#64748B", linestyle=":", linewidth=1.2
            )

            norm_3d = float(np.linalg.norm(z_test[:3]))
            self.ax_3d.set_title(
                f"3D Vector Space: z_3D = [{z_test[0]:+.3f}, {z_test[1]:+.3f}, {z_test[2]:+.3f}]ᵀ  (||z_3D||₂ = {norm_3d:.3f})",
                fontsize=9.5, fontweight="bold", pad=6
            )
            self.ax_3d.set_xlabel("z₁ (PC 1)", fontsize=8)
            self.ax_3d.set_ylabel("z₂ (PC 2)", fontsize=8)
            self.ax_3d.set_zlabel("z₃ (PC 3)", fontsize=8)
            self.ax_3d.view_init(elev=22, azim=40)
            self.ax_3d.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7.5)
            self.fig_3d.tight_layout()
            self.canvas_3d.draw()
        except Exception as e:
            print(f"[Warning] Failed to update 3D plot: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitAnalyzerGUI(root)
    root.mainloop()
