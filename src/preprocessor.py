"""
src/preprocessor.py
High-Performance Digit Preprocessor
Implements Stages 1 to 4 of the Mathematical Pipeline:
1. Digit image reading
2. Matrix representation (H x W)
3. Vectorization (Linear Isomorphism R^(H x W) -> R^D)
4. Normalization (L2 norm and Min-Max scaling)
Includes C-speed PIL bounding box cropping and auto-centering.
"""

import numpy as np
from PIL import Image

class DigitPreprocessor:
    """
    Handles image reading, grayscale matrix formation, vectorization, and normalization.
    """
    def __init__(self, target_size=(20, 20)):
        self.target_size = target_size
        self.height, self.width = target_size
        self.dim = self.height * self.width

    def image_to_matrix(self, image_path_or_pil) -> np.ndarray:
        """
        Stage 1 & 2: Reads image, converts to grayscale, resizes, and yields
        Matrix Representation A in R^(H x W) with pixel values in [0, 255].
        """
        if isinstance(image_path_or_pil, str):
            with Image.open(image_path_or_pil) as img_file:
                img = img_file.convert("L")
        elif isinstance(image_path_or_pil, Image.Image):
            img = image_path_or_pil.convert("L")
        else:
            raise TypeError("Expected file path or PIL Image object.")

        img_resized = img.resize(self.target_size, Image.Resampling.BILINEAR)
        return np.array(img_resized, dtype=np.float64)

    def preprocess_drawn_image(self, pil_img) -> np.ndarray:
        """
        Special high-speed preprocessor for freehand drawn digits:
        1. Uses PIL C-level getbbox() with threshold > 50 to isolate true strokes
        2. Fits and centers digit bounding box symmetrically within a 15x15 box on a 20x20 canvas
        3. Preserves aspect ratio and prevents any edge clipping
        """
        img = pil_img.convert("L")
        
        # Fast C-level bounding box search on thresholded mask (> 50 eliminates faint UI artifacts)
        mask = img.point(lambda p: 255 if p > 50 else 0)
        bbox = mask.getbbox()
        if bbox is None:
            # Blank canvas
            return np.zeros(self.target_size, dtype=np.float64)

        crop = img.crop(bbox)
        inner_size = 15
        w, h = crop.size
        if w > h:
            new_w = inner_size
            new_h = max(1, int(round((h / w) * inner_size)))
        else:
            new_h = inner_size
            new_w = max(1, int(round((w / h) * inner_size)))

        crop_resized = crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
        arr = np.array(crop_resized, dtype=np.float64)

        # Symmetrical bounding box center placement on canvas (MNIST geometric alignment)
        paste_x = (self.width - new_w) // 2
        paste_y = (self.height - new_h) // 2

        canvas = np.zeros(self.target_size, dtype=np.float64)
        canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = arr
        return canvas

    def vectorize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Stage 3: Vectorization operator vec(A): R^(H x W) -> R^D
        Flattens 2D matrix into 1D coordinate vector in R^(H*W).
        """
        if matrix.shape != self.target_size:
            raise ValueError(f"Matrix shape {matrix.shape} does not match target size {self.target_size}")
        return matrix.ravel()

    def unvectorize(self, vector: np.ndarray) -> np.ndarray:
        """
        Inverse operator vec^(-1)(x): R^D -> R^(H x W)
        Reshapes 1D coordinate vector back into 2D image matrix.
        """
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} does not match {self.dim}")
        return vector.reshape(self.target_size)

    def normalize_l2(self, vector: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Stage 4: L2 Norm Normalization:
        x_hat = x / ||x||_2
        where ||x||_2 = sqrt(sum(x_i^2))
        Returns normalized unit vector and computed L2 norm.
        """
        norm_val = float(np.linalg.norm(vector, ord=2))
        if norm_val < 1e-12:
            return np.zeros_like(vector), 0.0
        normalized_vector = vector / norm_val
        return normalized_vector, norm_val

    def normalize_minmax(self, vector: np.ndarray) -> np.ndarray:
        """
        Min-Max normalization mapping entries to [0, 1]:
        x'_i = (x_i - min(x)) / (max(x) - min(x))
        """
        min_val = np.min(vector)
        max_val = np.max(vector)
        if max_val - min_val < 1e-12:
            return np.zeros_like(vector)
        return (vector - min_val) / (max_val - min_val)

    def format_ascii_matrix(self, matrix: np.ndarray, threshold: float = 80.0) -> str:
        """
        Formats the digit matrix as a readable ASCII representation for console display.
        Uses shading characters: ' ' < '.' < ':' < '+' < '#' < '@'.
        """
        shades = " .:-=+*#%@"
        lines = []
        h, w = matrix.shape
        lines.append("+" + "-" * (w * 2) + "+")
        for i in range(h):
            row_str = "|"
            for j in range(w):
                val = matrix[i, j]
                idx = int((val / 255.0) * (len(shades) - 1))
                char = shades[idx]
                row_str += char + " "
            row_str += "|"
            lines.append(row_str)
        lines.append("+" + "-" * (w * 2) + "+")
        return "\n".join(lines)
