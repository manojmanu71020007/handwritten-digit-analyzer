"""
generate_dataset.py
Generates a diverse, high-quality handwritten digit dataset (0-9)
combining real handwriting fonts (Ink Free, Segoe Print, Comic Sans, Calibri, etc.),
custom vector strokes (alternative European & American handwriting styles),
random slants/rotations (-12 deg to +12 deg), and Center-of-Mass standardization.
Saves images into dataset/train/ and dataset/test/.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def standardize_matrix(arr_2d: np.ndarray, target_size=(20, 20), inner_size=15) -> np.ndarray:
    """
    Standardizes a digit image using Center-of-Mass (Centroid) alignment (MNIST standard):
    1. Bounding box cropping
    2. Aspect-ratio preserving resize to inner_size
    3. Center-of-Mass placement at (10, 10) on a 20x20 canvas
    """
    img = Image.fromarray(arr_2d.astype(np.uint8))
    mask = img.point(lambda p: 255 if p > 25 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return np.zeros(target_size, dtype=np.float64)

    crop = img.crop(bbox)
    w, h = crop.size
    if w > h:
        new_w = inner_size
        new_h = max(1, int(round((h / w) * inner_size)))
    else:
        new_h = inner_size
        new_w = max(1, int(round((w / h) * inner_size)))

    crop_resized = crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
    arr = np.array(crop_resized, dtype=np.float64)

    total_mass = np.sum(arr)
    if total_mass < 1e-5:
        return np.zeros(target_size, dtype=np.float64)

    y_coords, x_coords = np.indices(arr.shape)
    cy = np.sum(y_coords * arr) / total_mass
    cx = np.sum(x_coords * arr) / total_mass

    paste_x = int(round(10.0 - cx))
    paste_y = int(round(10.0 - cy))

    canvas = np.zeros(target_size, dtype=np.float64)
    src_x1 = max(0, -paste_x)
    src_y1 = max(0, -paste_y)
    src_x2 = min(new_w, target_size[1] - paste_x)
    src_y2 = min(new_h, target_size[0] - paste_y)

    dst_x1 = max(0, paste_x)
    dst_y1 = max(0, paste_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]

    return canvas

def create_font_digit(digit: int, font_name: str, font_size=65, angle=0.0):
    """Renders a digit using a specified font and rotation."""
    canvas_size = 100
    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)

    font_path = os.path.join("C:/Windows/Fonts", font_name)
    if os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # Draw centered
    bbox = draw.textbbox((0, 0), str(digit), font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (canvas_size - tw) // 2 - bbox[0]
    ty = (canvas_size - th) // 2 - bbox[1]
    draw.text((tx, ty), str(digit), fill=255, font=font)

    if abs(angle) > 0.5:
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR)

    # Slight blur to simulate natural ink diffusion
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    arr = np.array(img, dtype=np.float64)
    return standardize_matrix(arr)

def create_stroke_digit(digit: int, style=0, angle=0.0):
    """Renders procedural strokes mimicking authentic handwriting variations."""
    canvas_size = 100
    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = 50, 50
    r = 30
    sw = random.randint(6, 10)

    if digit == 0:
        # Oval or round 0
        w_fac = 0.6 if style % 2 == 0 else 0.8
        draw.ellipse([cx - r * w_fac, cy - r, cx + r * w_fac, cy + r], outline=255, width=sw)
    elif digit == 1:
        # Plain stick or stick with serif & base
        draw.line([(cx, cy - r), (cx, cy + r)], fill=255, width=sw)
        if style % 2 == 1:
            draw.line([(cx - 12, cy - r + 10), (cx, cy - r)], fill=255, width=sw - 1)
            draw.line([(cx - 15, cy + r), (cx + 15, cy + r)], fill=255, width=sw - 1)
    elif digit == 2:
        # Loopy or sharp bottom 2
        pts = [(cx - 18, cy - 16), (cx - 10, cy - r), (cx + 14, cy - r + 4), (cx + 18, cy - 10), (cx - 16, cy + r), (cx + 20, cy + r)]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i+1]], fill=255, width=sw)
    elif digit == 3:
        # Rounded 3 or flat-top 3
        if style % 2 == 0:
            pts = [(cx - 18, cy - r), (cx + 16, cy - r), (cx + 12, cy - 6), (cx, cy), (cx + 16, cy + 8), (cx + 14, cy + r), (cx - 18, cy + r)]
        else:
            # Flat top
            draw.line([(cx - 18, cy - r), (cx + 16, cy - r)], fill=255, width=sw)
            draw.line([(cx + 16, cy - r), (cx + 2, cy - 4)], fill=255, width=sw)
            pts = [(cx + 2, cy - 4), (cx + 16, cy + 8), (cx + 14, cy + r), (cx - 18, cy + r)]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i+1]], fill=255, width=sw)
    elif digit == 4:
        # Open 4 or closed 4
        if style % 2 == 0:
            draw.line([(cx + 4, cy - r), (cx - 18, cy + 6)], fill=255, width=sw)
            draw.line([(cx - 20, cy + 6), (cx + 20, cy + 6)], fill=255, width=sw)
            draw.line([(cx + 6, cy - r + 4), (cx + 6, cy + r)], fill=255, width=sw)
        else:
            # Closed triangle top
            draw.line([(cx - 16, cy + 6), (cx + 8, cy - r)], fill=255, width=sw)
            draw.line([(cx - 18, cy + 6), (cx + 18, cy + 6)], fill=255, width=sw)
            draw.line([(cx + 8, cy - r), (cx + 8, cy + r)], fill=255, width=sw)
    elif digit == 5:
        # Top bar, stem, bottom curve
        draw.line([(cx + 16, cy - r), (cx - 16, cy - r)], fill=255, width=sw)
        draw.line([(cx - 16, cy - r), (cx - 16, cy - 4)], fill=255, width=sw)
        draw.line([(cx - 16, cy - 4), (cx + 14, cy - 4)], fill=255, width=sw)
        draw.line([(cx + 14, cy - 4), (cx + 18, cy + 10)], fill=255, width=sw)
        draw.line([(cx + 18, cy + 10), (cx - 16, cy + r)], fill=255, width=sw)
    elif digit == 6:
        # Arch down into circle
        pts = [(cx + 14, cy - r), (cx - 14, cy - 8), (cx - 18, cy + 14), (cx + 14, cy + 14), (cx + 14, cy - 4), (cx - 14, cy + 2)]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i+1]], fill=255, width=sw)
    elif digit == 7:
        # Plain 7 or crossed 7
        draw.line([(cx - 20, cy - r), (cx + 20, cy - r)], fill=255, width=sw)
        draw.line([(cx + 20, cy - r), (cx - 8, cy + r)], fill=255, width=sw)
        if style % 2 == 1:
            draw.line([(cx - 10, cy), (cx + 10, cy)], fill=255, width=sw - 1)
    elif digit == 8:
        # Figure 8
        draw.ellipse([cx - 16, cy - r, cx + 16, cy - 2], outline=255, width=sw)
        draw.ellipse([cx - 18, cy - 4, cx + 18, cy + r], outline=255, width=sw)
    elif digit == 9:
        # Circle top and stem
        draw.ellipse([cx - 16, cy - r, cx + 16, cy + 4], outline=255, width=sw)
        if style % 2 == 0:
            draw.line([(cx + 14, cy - 6), (cx + 10, cy + r)], fill=255, width=sw)
        else:
            # Curved hook stem
            draw.line([(cx + 14, cy - 6), (cx + 8, cy + r - 4)], fill=255, width=sw)
            draw.line([(cx + 8, cy + r - 4), (cx - 8, cy + r)], fill=255, width=sw)

    if abs(angle) > 0.5:
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    arr = np.array(img, dtype=np.float64)
    return standardize_matrix(arr)

def generate_all():
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Handwriting and natural fonts
    available_fonts = ["inkfree.ttf", "segoepr.ttf", "comic.ttf", "calibri.ttf", "arial.ttf", "trebuc.ttf"]
    valid_fonts = [f for f in available_fonts if os.path.exists(os.path.join("C:/Windows/Fonts", f))]
    if not valid_fonts:
        valid_fonts = ["arial.ttf"]

    print(f"Generating diverse handwritten dataset with fonts: {valid_fonts}...")

    # For each digit, generate 15 diverse training variations
    for digit in range(10):
        sample_idx = 0
        # 1. Font-based natural handwriting with different angles
        for font in valid_fonts:
            for angle in [-10.0, 0.0, 10.0]:
                mat = create_font_digit(digit, font, font_size=random.randint(60, 75), angle=angle)
                img = Image.fromarray(mat.astype(np.uint8))
                img.save(os.path.join(train_dir, f"digit_{digit}_sample_{sample_idx}.png"))
                sample_idx += 1
                if sample_idx >= 12:
                    break
            if sample_idx >= 12:
                break

        # 2. Procedural stroke handwriting (open/closed, crossed/uncrossed)
        for st in range(4):
            mat = create_stroke_digit(digit, style=st, angle=random.choice([-8.0, 0.0, 8.0]))
            img = Image.fromarray(mat.astype(np.uint8))
            img.save(os.path.join(train_dir, f"digit_{digit}_sample_{sample_idx}.png"))
            sample_idx += 1

        print(f"  Digit {digit}: Generated {sample_idx} diverse training variations.")

    # Generate novel test set (5 unseen samples per digit = 50 test samples)
    print("Generating novel unseen test variations...")
    for digit in range(10):
        for test_idx in range(5):
            # Pick a novel rotation and stroke style
            angle = random.uniform(-14.0, 14.0)
            if test_idx % 2 == 0 and valid_fonts:
                font = random.choice(valid_fonts)
                mat = create_font_digit(digit, font, font_size=random.randint(55, 78), angle=angle)
            else:
                mat = create_stroke_digit(digit, style=test_idx + 4, angle=angle)

            img = Image.fromarray(mat.astype(np.uint8))
            img.save(os.path.join(test_dir, f"test_{digit}_sample_{test_idx}.png"))

    print("[OK] Digit dataset generation complete! Saved to dataset/train and dataset/test.")

if __name__ == "__main__":
    generate_all()
