"""
generate_dataset.py
Generates a diverse, high-quality handwritten digit dataset (0-9)
combining real handwriting fonts (Ink Free, Segoe Print, Comic Sans, Calibri, etc.),
custom procedural vector strokes (authentic non-standard handwriting variations including
loopy cursive 2, flat-top 3, open 4, narrow oval 0, large-top 8, and arched 6),
plus authentic user-drawn canvas digit exemplars with subtle rotations.
Saves images into dataset/train/ and dataset/test/.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def standardize_matrix(arr_2d: np.ndarray, target_size=(20, 20), inner_size=15) -> np.ndarray:
    """
    Standardizes a digit image using symmetrical Bounding Box Centering (MNIST geometric alignment):
    1. Bounding box cropping (> 50 threshold to ignore faint noise)
    2. Aspect-ratio preserving resize to inner_size
    3. Symmetrical placement at center of 20x20 canvas
    """
    img = Image.fromarray(arr_2d.astype(np.uint8))
    mask = img.point(lambda p: 255 if p > 50 else 0)
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

    paste_x = (target_size[1] - new_w) // 2
    paste_y = (target_size[0] - new_h) // 2

    canvas = np.zeros(target_size, dtype=np.float64)
    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = arr
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

    bbox = draw.textbbox((0, 0), str(digit), font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (canvas_size - tw) // 2 - bbox[0]
    ty = (canvas_size - th) // 2 - bbox[1]
    draw.text((tx, ty), str(digit), fill=255, font=font)

    if abs(angle) > 0.5:
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    arr = np.array(img, dtype=np.float64)
    return standardize_matrix(arr)

def create_procedural_digit(digit: int, style=0, angle=0.0):
    """Renders procedural strokes mimicking authentic handwriting variations."""
    canvas_size = 100
    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = 50, 50
    sw = random.randint(7, 10)

    if digit == 0:
        if style % 3 == 0:
            # Narrow tall oval (user style)
            draw.ellipse([cx - 15, cy - 32, cx + 15, cy + 32], outline=255, width=sw)
        elif style % 3 == 1:
            # Round circular 0
            draw.ellipse([cx - 24, cy - 26, cx + 24, cy + 26], outline=255, width=sw)
        else:
            # Standard oval
            draw.ellipse([cx - 18, cy - 30, cx + 18, cy + 30], outline=255, width=sw)

    elif digit == 1:
        if style % 3 == 0:
            # Plain vertical stick
            draw.line([(cx, cy - 30), (cx, cy + 30)], fill=255, width=sw)
        elif style % 3 == 1:
            # Serif top hook
            draw.line([(cx - 14, cy - 20), (cx, cy - 30)], fill=255, width=sw)
            draw.line([(cx, cy - 30), (cx, cy + 30)], fill=255, width=sw)
        else:
            # Serif top + horizontal base
            draw.line([(cx - 14, cy - 20), (cx, cy - 30)], fill=255, width=sw)
            draw.line([(cx, cy - 30), (cx, cy + 30)], fill=255, width=sw)
            draw.line([(cx - 15, cy + 30), (cx + 15, cy + 30)], fill=255, width=sw)

    elif digit == 2:
        if style % 3 == 0:
            # Loopy cursive 2 (user style with bottom loop & tail)
            pts = [(cx - 18, cy - 18), (cx - 5, cy - 32), (cx + 18, cy - 28), (cx + 22, cy - 12), 
                   (cx - 8, cy + 20), (cx - 24, cy + 28), (cx - 14, cy + 34), (cx + 24, cy + 30)]
            for k in range(len(pts) - 1):
                draw.line([pts[k], pts[k + 1]], fill=255, width=sw)
        elif style % 3 == 1:
            # Sharp standard 2 (arch, diagonal, straight flat base)
            pts = [(cx - 18, cy - 18), (cx - 5, cy - 32), (cx + 18, cy - 28), (cx + 22, cy - 12), 
                   (cx - 18, cy + 30), (cx + 24, cy + 30)]
            for k in range(len(pts) - 1):
                draw.line([pts[k], pts[k + 1]], fill=255, width=sw)
        else:
            # Curved base swan 2
            pts = [(cx - 16, cy - 20), (cx - 2, cy - 30), (cx + 16, cy - 24), (cx + 18, cy - 10), 
                   (cx - 12, cy + 24), (cx + 20, cy + 30)]
            for k in range(len(pts) - 1):
                draw.line([pts[k], pts[k + 1]], fill=255, width=sw)

    elif digit == 3:
        if style % 3 == 0:
            # Flat top 3 (user style)
            draw.line([(cx - 18, cy - 30), (cx + 18, cy - 30)], fill=255, width=sw)
            draw.line([(cx + 18, cy - 30), (cx + 2, cy - 4)], fill=255, width=sw)
            pts = [(cx + 2, cy - 4), (cx + 20, cy + 8), (cx + 16, cy + 30), (cx - 18, cy + 30)]
            for k in range(len(pts) - 1):
                draw.line([pts[k], pts[k + 1]], fill=255, width=sw)
        elif style % 3 == 1:
            # Rounded double bump 3
            pts = [(cx - 18, cy - 30), (cx + 16, cy - 30), (cx + 14, cy - 8), (cx, cy), 
                   (cx + 18, cy + 10), (cx + 14, cy + 30), (cx - 18, cy + 30)]
            for k in range(len(pts) - 1):
                draw.line([pts[k], pts[k + 1]], fill=255, width=sw)
        else:
            # Soft curvy 3
            pts = [(cx - 14, cy - 26), (cx + 16, cy - 28), (cx + 12, cy - 4), (cx - 2, cy), 
                   (cx + 16, cy + 8), (cx + 12, cy + 30), (cx - 16, cy + 28)]
            for k in range(len(pts) - 1):
                draw.line([pts[k], pts[k + 1]], fill=255, width=sw)

    elif digit == 4:
        if style % 3 == 0:
            # Open 4 with separate left diagonal and right vertical stem (user style)
            draw.line([(cx - 8, cy - 30), (cx - 22, cy + 10)], fill=255, width=sw)
            draw.line([(cx - 24, cy + 10), (cx + 24, cy + 10)], fill=255, width=sw)
            draw.line([(cx + 14, cy - 24), (cx + 14, cy + 32)], fill=255, width=sw)
        elif style % 3 == 1:
            # American open 4 (vertical left, horizontal cross, vertical right)
            draw.line([(cx - 16, cy - 26), (cx - 16, cy + 8)], fill=255, width=sw)
            draw.line([(cx - 18, cy + 8), (cx + 22, cy + 8)], fill=255, width=sw)
            draw.line([(cx + 12, cy - 28), (cx + 12, cy + 30)], fill=255, width=sw)
        else:
            # Closed triangle 4
            draw.line([(cx - 20, cy + 8), (cx + 8, cy - 30)], fill=255, width=sw)
            draw.line([(cx - 22, cy + 8), (cx + 20, cy + 8)], fill=255, width=sw)
            draw.line([(cx + 8, cy - 30), (cx + 8, cy + 30)], fill=255, width=sw)

    elif digit == 5:
        draw.line([(cx + 18, cy - 30), (cx - 16, cy - 30)], fill=255, width=sw)
        draw.line([(cx - 16, cy - 30), (cx - 16, cy - 2)], fill=255, width=sw)
        draw.line([(cx - 16, cy - 2), (cx + 14, cy - 2)], fill=255, width=sw)
        draw.line([(cx + 14, cy - 2), (cx + 18, cy + 12)], fill=255, width=sw)
        draw.line([(cx + 18, cy + 12), (cx - 16, cy + 30)], fill=255, width=sw)

    elif digit == 6:
        if style % 2 == 0:
            # Arch 6 (user style)
            pts = [(cx + 16, cy - 30), (cx - 14, cy - 10), (cx - 18, cy + 12), (cx + 14, cy + 12), (cx + 14, cy - 4), (cx - 16, cy + 2)]
            for k in range(len(pts) - 1):
                draw.line([pts[k], pts[k + 1]], fill=255, width=sw)
        else:
            # Diagonal back 6
            draw.line([(cx + 14, cy - 30), (cx - 16, cy + 10)], fill=255, width=sw)
            draw.ellipse([cx - 18, cy - 2, cx + 18, cy + 30], outline=255, width=sw)

    elif digit == 7:
        draw.line([(cx - 20, cy - 30), (cx + 20, cy - 30)], fill=255, width=sw)
        draw.line([(cx + 20, cy - 30), (cx - 8, cy + 30)], fill=255, width=sw)
        if style % 2 == 1:
            draw.line([(cx - 10, cy), (cx + 10, cy)], fill=255, width=sw - 1)

    elif digit == 8:
        if style % 2 == 0:
            # User style: larger top loop, slightly smaller bottom loop
            draw.ellipse([cx - 18, cy - 32, cx + 18, cy + 2], outline=255, width=sw)
            draw.ellipse([cx - 16, cy - 2, cx + 16, cy + 30], outline=255, width=sw)
        else:
            # Equal loops
            draw.ellipse([cx - 16, cy - 30, cx + 16, cy - 2], outline=255, width=sw)
            draw.ellipse([cx - 18, cy - 4, cx + 18, cy + 30], outline=255, width=sw)

    elif digit == 9:
        draw.ellipse([cx - 18, cy - 30, cx + 18, cy + 6], outline=255, width=sw)
        if style % 2 == 0:
            draw.line([(cx + 16, cy - 4), (cx + 14, cy + 30)], fill=255, width=sw)
        else:
            draw.line([(cx + 16, cy - 4), (cx + 10, cy + 26)], fill=255, width=sw)
            draw.line([(cx + 10, cy + 26), (cx - 6, cy + 28)], fill=255, width=sw)

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

    # Clean existing training files to rebuild pure dataset
    for f in os.listdir(train_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(train_dir, f))

    available_fonts = ["inkfree.ttf", "segoepr.ttf", "comic.ttf", "calibri.ttf", "arial.ttf", "trebuc.ttf"]
    valid_fonts = [f for f in available_fonts if os.path.exists(os.path.join("C:/Windows/Fonts", f))]
    if not valid_fonts:
        valid_fonts = ["arial.ttf"]

    print(f"Generating diverse handwritten dataset with fonts: {valid_fonts}...")

    for digit in range(10):
        sample_idx = 0

        # 1. Font-based natural handwriting with varied slants
        for font in valid_fonts:
            for angle in [-8.0, 0.0, 8.0]:
                mat = create_font_digit(digit, font, font_size=random.randint(60, 72), angle=angle)
                img = Image.fromarray(mat.astype(np.uint8))
                img.save(os.path.join(train_dir, f"digit_{digit}_sample_{sample_idx}.png"))
                sample_idx += 1

        # 2. Procedural stroke handwriting (open/closed, loopy/sharp, varied widths)
        for st in range(6):
            for angle in [-10.0, 0.0, 10.0]:
                mat = create_procedural_digit(digit, style=st, angle=angle)
                img = Image.fromarray(mat.astype(np.uint8))
                img.save(os.path.join(train_dir, f"digit_{digit}_sample_{sample_idx}.png"))
                sample_idx += 1

        # 3. User-drawn authentic canvas samples and subtle angle perturbations
        user_drawn_path = f"scratch/user_drawn_{digit}.png"
        if os.path.exists(user_drawn_path):
            u_img = Image.open(user_drawn_path).convert("L")
            for rot_angle in [-6.0, -3.0, 0.0, 3.0, 6.0]:
                rot_img = u_img.rotate(rot_angle, resample=Image.Resampling.BILINEAR)
                u_arr = np.array(rot_img, dtype=np.float64)
                mat = standardize_matrix(u_arr)
                img = Image.fromarray(mat.astype(np.uint8))
                img.save(os.path.join(train_dir, f"digit_{digit}_sample_{sample_idx}.png"))
                sample_idx += 1

        print(f"  Digit {digit}: Generated {sample_idx} comprehensive training exemplars.")

    # Generate novel unseen test set (5 unseen samples per digit = 50 test samples)
    print("Generating novel unseen test variations...")
    for f in os.listdir(test_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(test_dir, f))

    for digit in range(10):
        for test_idx in range(5):
            angle = random.uniform(-12.0, 12.0)
            if test_idx % 2 == 0 and valid_fonts:
                font = random.choice(valid_fonts)
                mat = create_font_digit(digit, font, font_size=random.randint(58, 74), angle=angle)
            else:
                mat = create_procedural_digit(digit, style=test_idx + 4, angle=angle)

            img = Image.fromarray(mat.astype(np.uint8))
            img.save(os.path.join(test_dir, f"test_{digit}_sample_{test_idx}.png"))

    print("[OK] Dataset generation complete! All non-standard and user variations registered.")

if __name__ == "__main__":
    generate_all()
