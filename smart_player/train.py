"""Train model.py's LeaveValueNet on data from generate_data.py.

    python smart_player/generate_data.py 20000
    python smart_player/train.py

The checkpoint stores the tile alphabet next to the weights, so
model.get_model() can never load it against a mismatched encoding.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import ALPHABET, LeaveValueNet, encode_leave


class LeaveDataset(Dataset):
    def __init__(self, path: str) -> None:
        data = np.load(path)
        self.leaves = data["leaves"]
        self.unseen = data["unseen"]
        self.margins = data["margins"]

    def __len__(self) -> int:
        return len(self.leaves)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = encode_leave(str(self.leaves[i]), int(self.unseen[i]))
        y = torch.tensor(float(self.margins[i]))
        return x, y


def train(
    data_path: str,
    out_path: str,
    epochs: int = 30,
    batch: int = 256,
    lr: float = 1e-3,
    quiet: bool = False,
) -> float:
    """Train a fresh LeaveValueNet on `data_path`, saving the best-val-MSE
    checkpoint to `out_path`. Returns that best val MSE. Used both by this
    file's CLI and directly by iterate.py (one call per round)."""
    ds = LeaveDataset(data_path)
    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)
    if not quiet:
        print(f"{len(ds)} samples, {len(ALPHABET)} tile types: {''.join(ALPHABET)}")

    # No mps/cpu device dance here (unlike board_reader's CNNs): a
    # ~3.5k-parameter MLP is dominated by Python/dispatch overhead either
    # way, and this also has to run cheaply on CPU at inference time inside
    # SmartPlayer.evaluate_word, called dozens of times per move.
    model = LeaveValueNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
        sched.step()

        model.eval()
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += loss_fn(model(x), y).item() * len(y)
                val_seen += len(y)
        val_mse = val_loss / val_seen
        if not quiet:
            print(
                f"epoch {epoch + 1:2d}/{epochs}  train mse {loss_sum / seen:.3f}  "
                f"val mse {val_mse:.3f}  ({time.time() - t0:.1f}s)"
            )
        if val_mse <= best_val:
            best_val = val_mse
            torch.save({"state_dict": model.state_dict(), "alphabet": ALPHABET}, out_path)
    if not quiet:
        print(f"best val mse {best_val:.3f} -> {out_path}")
    return best_val


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "_leave_dataset.npz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "models", "leave_value.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    train(args.data, args.out, args.epochs, args.batch, args.lr)
