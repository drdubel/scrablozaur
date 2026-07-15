"""Generate a training dataset for letter_classifier.py's CNN.

Two sources, both normalised exactly like glyph_normalizer.py's extractor
(ink centred in a 64px canvas with a 12% margin, aspect preserved):

1. font renderings -- every alphabet letter drawn with all usable system
   fonts (bold + regular), then augmented;
2. real harvested glyphs -- masks under src/data/real_templates/ (reviewed
   crops from actual photos, see harvest_templates.py + review_templates.py),
   heavily augmented.

Augmentations model what the real extractor produces: small rotations and
shears, stroke thickness changes (erode/dilate), blur, sensor noise, partial
strokes, and occasional grid-line bar artefacts near the canvas border.

Usage:
    python scripts/generate_synthetic_dataset.py --out src/data_train --per-letter 400

Ported from ocr/scripts/generate_synthetic_dataset.py.
"""

import argparse
import glob
import os
import random
import sys
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from letter_classifier import POLISH_ALPHABET, normalize_template  # noqa: E402

SIZE = 64

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def render_letter(ch, font, squeeze):
    img = Image.new("L", (300, 300), 0)
    ImageDraw.Draw(img).text((150, 150), ch, fill=255, font=font, anchor="mm")
    mask = np.asarray(img)
    if mask.max() == 0:
        return None
    if squeeze != 1.0:
        mask = cv2.resize(mask, None, fx=1.0, fy=squeeze, interpolation=cv2.INTER_LINEAR)
    return (mask > 127).astype(np.uint8) * 255


def augment(mask, rng):
    """Binary glyph mask -> augmented grayscale sample (ink dark on white)."""
    m = mask.copy()
    h, w = m.shape

    # Geometric jitter (rotation, shear, scale) around the centre.
    ang = rng.uniform(-6, 6)
    scale = rng.uniform(0.9, 1.08)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[0, 1] += rng.uniform(-0.06, 0.06)  # shear
    M[:, 2] += (rng.uniform(-2, 2), rng.uniform(-2, 2))
    m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR)

    # Stroke thickness.
    r = rng.random()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if r < 0.30:
        m = cv2.erode(m, k)
    elif r > 0.70:
        m = cv2.dilate(m, k)

    # Partial stroke damage (worn print).
    if rng.random() < 0.20:
        for _ in range(rng.randint(1, 3)):
            x, y = rng.randrange(w), rng.randrange(h)
            cv2.circle(m, (x, y), rng.randint(1, 3), 0, -1)

    # Occasional grid-line bar artefact near a border (extractor leftovers).
    if rng.random() < 0.12:
        edge = rng.choice(["top", "bottom", "left", "right"])
        t = rng.randint(1, 3)
        pos = rng.randint(0, 6)
        if edge == "top":
            m[pos : pos + t, rng.randint(0, 8) : w - rng.randint(0, 8)] = 255
        elif edge == "bottom":
            m[h - pos - t : h - pos, rng.randint(0, 8) : w - rng.randint(0, 8)] = 255
        elif edge == "left":
            m[rng.randint(0, 8) : h - rng.randint(0, 8), pos : pos + t] = 255
        else:
            m[rng.randint(0, 8) : h - rng.randint(0, 8), h - pos - t : h - pos] = 255

    norm = normalize_template(m, SIZE)
    if norm is None:
        norm = np.zeros((SIZE, SIZE), np.uint8)

    # Mask -> grayscale appearance: variable ink darkness / paper tone.
    ink = rng.uniform(20, 90)
    paper = rng.uniform(200, 255)
    g = np.where(norm > 127, ink, paper).astype(np.float32)
    # Resolution jitter: teach the classifier soft, upsampled strokes too.
    if rng.random() < 0.5:
        small = rng.randint(30, 56)
        g = cv2.resize(
            cv2.resize(g, (small, small), interpolation=cv2.INTER_AREA), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR
        )
    if rng.random() < 0.7:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.4, 1.3))
    g += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 10), g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "src", "data_train"))
    ap.add_argument("--per-letter", type=int, default=400)
    ap.add_argument(
        "--real-templates", default=os.path.join(os.path.dirname(__file__), "..", "src", "data", "real_templates")
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    fonts = []
    for p in FONT_CANDIDATES:
        if os.path.isfile(p):
            try:
                fonts.append(ImageFont.truetype(p, 180))
            except OSError:
                pass
    if not fonts:
        raise SystemExit("no usable fonts found -- edit FONT_CANDIDATES")
    print(f"{len(fonts)} fonts")

    total = 0
    for ch in POLISH_ALPHABET:
        out_dir = os.path.join(args.out, ch)
        os.makedirs(out_dir, exist_ok=True)

        bases = []
        for fi, font in enumerate(fonts):
            for squeeze in (1.0, 1.15):
                m = render_letter(ch, font, squeeze)
                if m is not None:
                    bases.append((f"f{fi}s{int(squeeze * 100)}", m))
        real_dir = os.path.join(args.real_templates, ch)
        if os.path.isdir(real_dir):
            for f in sorted(glob.glob(os.path.join(real_dir, "*.png"))):
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is not None and img.max() > 0:
                    # Real glyphs get 3x the sampling weight of font glyphs:
                    # they carry the actual tile typeface.
                    for _ in range(3):
                        bases.append((f"r{os.path.basename(f)[:-4]}", img))
        if not bases:
            print(f"warning: no sources for {ch!r}")
            continue

        for i in range(args.per_letter):
            tag, base = bases[i % len(bases)]
            sample = augment((base > 127).astype(np.uint8) * 255, rng)
            name = unicodedata.normalize("NFC", f"{tag}_{i:04d}.png")
            cv2.imwrite(os.path.join(out_dir, name), sample)
            total += 1
    print(f"wrote {total} samples to {args.out}")


if __name__ == "__main__":
    main()
