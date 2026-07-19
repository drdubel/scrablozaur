"""Train letter_classifier.py's CNN on a dataset produced by
generate_synthetic_dataset.py.

    python scripts/train_classifier.py --data src/data_train \
        --out src/models/letter_cnn.pt --epochs 12

The checkpoint stores the class list next to the weights, so the
classifier can never mis-map indices to letters.

Ported from ocr/scripts/train_classifier.py.
"""

import argparse
import os
import sys
import time
import unicodedata

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from letter_classifier import LetterCNN  # noqa: E402


class GlyphDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.classes = sorted(
            unicodedata.normalize("NFC", d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        for idx, ch in enumerate(self.classes):
            d = os.path.join(root, ch)
            for f in os.listdir(d):
                if f.endswith(".png"):
                    self.samples.append((os.path.join(d, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = (255.0 - img.astype(np.float32)) / 255.0  # ink -> 1
        return torch.from_numpy(x).unsqueeze(0), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "src", "data_train"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "src", "models", "letter_cnn.pt"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    ds = GlyphDataset(args.data)
    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    train = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val = DataLoader(val_ds, batch_size=args.batch, num_workers=0)
    print(f"{len(ds)} samples, {len(ds.classes)} classes: {''.join(ds.classes)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = LetterCNN(len(ds.classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        for x, y in train:
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
            for x, y in val:
                pred = model(x.to(device)).argmax(dim=1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
        acc = correct / max(1, total)
        print(
            f"epoch {epoch + 1:2d}/{args.epochs}  loss {loss_sum / seen:.4f}  val acc {acc:.4f}  ({time.time() - t0:.0f}s)"
        )
        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "classes": ds.classes}, args.out
            )
    print(f"best val acc {best_acc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
