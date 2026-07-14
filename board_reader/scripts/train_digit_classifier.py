"""Train letter_classifier.py's digit CNN -- reads the tile's own printed
point-value digit, used to help disambiguate the letter (see
letter_classifier.py's classify_digit_cnn_batch() / points_distribution()).

Real digit crops come from scripts/harvest_digit_templates.py + review_templates.py
--digits (src/data/real_digit_templates/<digit>/*.png), the same
harvest-then-review split letters use -- the digit's *label* needs no
review (it's pinned down for free by ground truth + LETTER_POINTS), but
the *crop* can still be chopped or grab the wrong region, so a bad
extraction shouldn't go straight into training unreviewed.

Why a CNN at all: a first cut at point-value reading used pure Dice
template matching (letter_classifier.py's classify_digit_templates(),
still kept as a fallback) and measured only 82.9% accurate even on
cleanly-extracted crops -- simple blocky 0-9 digit shapes don't separate
well with a handful of synthetic templates alone, the same reason letters
needed a CNN rather than template matching alone.

Usage:
    python scripts/harvest_digit_templates.py && python scripts/review_templates.py --digits   # once, or after a fresh harvest
    python scripts/train_digit_classifier.py [--epochs 12] [--per-digit 400]
"""

import argparse
import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import glyph_normalizer as gn  # noqa: E402
from letter_classifier import LetterCNN, normalize_template  # noqa: E402

DIGITS = "0123456789"
SIZE = gn.DIGIT_SIZE  # 32 -- much smaller than letters' 64, digits are simpler shapes

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

REAL_DIGITS_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "data", "real_digit_templates")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "data_train_digits")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "src", "models", "digit_cnn.pt")


def render_digit(d, font, squeeze):
    img = Image.new("L", (140, 140), 0)
    ImageDraw.Draw(img).text((70, 70), d, fill=255, font=font, anchor="mm")
    mask = np.asarray(img)
    if mask.max() == 0:
        return None
    if squeeze != 1.0:
        mask = cv2.resize(mask, None, fx=1.0, fy=squeeze, interpolation=cv2.INTER_LINEAR)
    return (mask > 127).astype(np.uint8) * 255


def augment(mask, rng):
    """Binary digit mask -> augmented grayscale sample (ink dark on white).
    Same idea as generate_synthetic_dataset.py's augment(), scaled down for
    the smaller digit canvas."""
    m = mask.copy()
    h, w = m.shape

    ang = rng.uniform(-8, 8)
    scale = rng.uniform(0.88, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[0, 1] += rng.uniform(-0.08, 0.08)
    M[:, 2] += (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
    m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR)

    r = rng.random()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if r < 0.30:
        m = cv2.erode(m, k)
    elif r > 0.70:
        m = cv2.dilate(m, k)

    norm = normalize_template(m, out_size=SIZE)
    if norm is None:
        norm = np.zeros((SIZE, SIZE), np.uint8)

    ink = rng.uniform(20, 90)
    paper = rng.uniform(200, 255)
    g = np.where(norm > 127, ink, paper).astype(np.float32)
    if rng.random() < 0.5:
        small = rng.randint(14, 26)
        g = cv2.resize(cv2.resize(g, (small, small), interpolation=cv2.INTER_AREA), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.7:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.4, 1.1))
    g += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 10), g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def build_dataset(per_digit, seed, real_digits_dir):
    rng = random.Random(seed)
    fonts = []
    for p in FONT_CANDIDATES:
        if os.path.isfile(p):
            try:
                fonts.append(ImageFont.truetype(p, 100))
            except OSError:
                pass
    if not fonts:
        raise SystemExit("no usable fonts found -- edit FONT_CANDIDATES")
    print(f"{len(fonts)} fonts")

    total = 0
    for d in DIGITS:
        out_dir = os.path.join(DATASET_DIR, d)
        os.makedirs(out_dir, exist_ok=True)
        bases = []
        for font in fonts:
            for squeeze in (1.0, 1.15):
                m = render_digit(d, font, squeeze)
                if m is not None:
                    bases.append(m)
        # Real, reviewed crops get 3x the sampling weight -- they carry the
        # actual tile typeface, same convention as generate_synthetic_dataset.py.
        real_dir = os.path.join(real_digits_dir, d)
        n_real = 0
        if os.path.isdir(real_dir):
            for f in sorted(glob.glob(os.path.join(real_dir, "*.png"))):
                crop = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if crop is None:
                    continue
                n_real += 1
                for _ in range(3):
                    bases.append((crop < 128).astype(np.uint8) * 255)  # gray -> binary ink mask
        if not bases:
            print(f"warning: no sources for digit {d!r}")
            continue
        print(f"  {d}: {n_real} real crops")
        for i in range(per_digit):
            base = bases[i % len(bases)]
            sample = augment(base, rng)
            cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), sample)
            total += 1
    print(f"wrote {total} samples to {DATASET_DIR}")


class DigitDataset(Dataset):
    def __init__(self, root):
        self.classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.samples = []
        for idx, d in enumerate(self.classes):
            folder = os.path.join(root, d)
            for f in os.listdir(folder):
                if f.endswith(".png"):
                    self.samples.append((os.path.join(folder, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = (255.0 - img.astype(np.float32)) / 255.0
        return torch.from_numpy(x).unsqueeze(0), label


def train(epochs, batch, lr):
    ds = DigitDataset(DATASET_DIR)
    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch, num_workers=0)
    print(f"{len(ds)} samples, {len(ds.classes)} classes: {''.join(ds.classes)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = LetterCNN(len(ds.classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    for epoch in range(epochs):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(dim=1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
        acc = correct / max(1, total)
        print(f"epoch {epoch + 1:2d}/{epochs}  loss {loss_sum / seen:.4f}  val acc {acc:.4f}  ({time.time() - t0:.0f}s)")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "classes": ds.classes}, MODEL_OUT)
    print(f"best val acc {best_acc:.4f} -> {MODEL_OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-digits", default=REAL_DIGITS_DIR, help="reviewed real digit crops (default: %(default)s)")
    ap.add_argument("--per-digit", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    build_dataset(args.per_digit, args.seed, args.real_digits)
    train(args.epochs, args.batch, args.lr)


if __name__ == "__main__":
    main()
