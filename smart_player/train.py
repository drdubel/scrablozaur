"""Train model.py's LeaveValueNet on data from generate_data.py.

    python smart_player/generate_data.py 20000
    python smart_player/train.py

The checkpoint stores the tile alphabet and input feature width next to the
weights, so model.get_model() can never load it against a mismatched
encoding.

Uses CUDA/MPS automatically when available. The whole dataset is encoded
and kept resident on-device as plain tensors rather than going through
torch.utils.data's Dataset/DataLoader machinery: at the tens-of-millions-
of-samples scale this trains on, per-sample Python-level __getitem__ calls
(and the small-batch-256 default that went with them) were the actual
bottleneck, not compute -- a GPU sitting idle, starved by a slow CPU-side
data pipeline, trains no faster than CPU. See model.py's encode_leaves()
for the vectorized encoding this replaced.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from model import ALPHABET, INPUT_DIM, LeaveValueNet, encode_leaves


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    data_path: str,
    out_path: str,
    epochs: int = 30,
    batch: int = 8192,
    lr: float = 1e-3,
    hidden1: int = 128,
    hidden2: int = 64,
    device: str | None = None,
    quiet: bool = False,
    warm_start: str | None = None,
    center_by_decision: bool = False,
) -> float:
    """Train a LeaveValueNet on `data_path`, saving the best-val-MSE checkpoint
    to `out_path`. Returns that best val MSE. Used both by this file's CLI and
    directly by iterate.py (one call per round).

    `warm_start` initialises from an existing checkpoint instead of from random.
    Without it a policy-iteration round is "retrain an equally-powerful model on
    similarly-distributed data", which is exactly what iterate.py measured: 0/5
    rounds promoted, every candidate landing in a 48.8-50.8% band. Nothing
    carried over between rounds, so nothing compounded.
    """
    device = device or _pick_device()

    raw = np.load(data_path)
    board_features = np.stack([raw["tw_open"], raw["dw_open"], raw["tl_open"], raw["dl_open"], raw["board_fill"]], axis=1)
    X = encode_leaves(raw["leaves"], raw["unseen"], board_features).to(device)
    targets = raw["margins"].astype(np.float32)

    if center_by_decision:
        # Only differences *within* a decision can change which move gets
        # picked: every candidate in one decision shares the same board and the
        # same unseen pool, so anything the position explains is a constant
        # offset that cannot reorder them.
        #
        # This matters because a high global R^2 is not evidence of a better
        # ranker. The first distilled checkpoint reached R^2 = 0.665 against
        # 0.068 for the n-step target and still lost by 8 points a game -- it
        # was predicting how good the *position* was, which is exactly the part
        # that cancels. Centring here makes the loss measure the part that does
        # not cancel.
        if "decision" not in raw:
            raise ValueError(
                f"{data_path} has no `decision` column; regenerate it with distill.py "
                "(generate_data.py output cannot be centred this way -- it has one "
                "sample per decision, so there is nothing to centre against)"
            )
        groups = raw["decision"].astype(np.int64)
        sums = np.bincount(groups, weights=targets.astype(np.float64))
        counts = np.bincount(groups)
        targets = (targets - (sums / np.maximum(counts, 1))[groups]).astype(np.float32)
        if not quiet:
            print(f"centred within {counts.size} decisions; target std "
                  f"{raw['margins'].std():.2f} -> {targets.std():.2f}")

    y = torch.from_numpy(targets).to(device)
    n = len(X)
    if not quiet:
        print(
            f"{n} samples, {len(ALPHABET)} tile types: {''.join(ALPHABET)}  "
            f"device={device}  hidden={hidden1}/{hidden2}"
        )

    n_val = max(1, int(0.1 * n))
    split = torch.randperm(n, generator=torch.Generator().manual_seed(0), device="cpu").to(device)
    val_idx, train_idx = split[:n_val], split[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    n_train = len(X_train)
    # Denominator for R^2: the MSE a model that only ever predicts the mean
    # would score. Computed on the validation split so it matches val_mse.
    val_var = max(float(y_val.var(unbiased=False).item()), 1e-9)

    model = LeaveValueNet(hidden1, hidden2).to(device)
    if warm_start:
        ckpt = torch.load(warm_start, map_location="cpu", weights_only=True)
        if (ckpt.get("hidden1"), ckpt.get("hidden2")) != (hidden1, hidden2):
            raise ValueError(
                f"{warm_start} is {ckpt.get('hidden1')}/{ckpt.get('hidden2')} but training at "
                f"{hidden1}/{hidden2}; pass matching --hidden1/--hidden2 to warm-start from it"
            )
        model.load_state_dict(ckpt["state_dict"])
        if not quiet:
            print(f"warm-started from {warm_start}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        for start in range(0, n_train, batch):
            idx = perm[start : start + batch]
            xb, yb = X_train[idx], y_train[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(X_val), batch):
                xb, yb = X_val[start : start + batch], y_val[start : start + batch]
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_mse = val_loss / len(X_val)
        if not quiet:
            # R^2 against predicting the mean. Raw MSE is unreadable on its own:
            # ~2100 sounds catastrophic but the target's own variance is ~2223,
            # so it is really "explains 5% of the target". Printing the ratio
            # would have made the noise ceiling obvious on the very first run
            # instead of after several rounds of ablation.
            print(
                f"epoch {epoch + 1:2d}/{epochs}  train mse {loss_sum / n_train:.3f}  "
                f"val mse {val_mse:.3f}  val R^2 {1.0 - val_mse / val_var:.4f}  "
                f"({time.time() - t0:.1f}s)"
            )
        if val_mse <= best_val:
            best_val = val_mse
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": state_dict,
                    "alphabet": ALPHABET,
                    "hidden1": hidden1,
                    "hidden2": hidden2,
                    "input_dim": INPUT_DIM,
                },
                out_path,
            )
    if not quiet:
        print(f"best val mse {best_val:.3f}  (R^2 {1.0 - best_val / val_var:.4f}) -> {out_path}")
    return best_val


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "_leave_dataset.npz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "models", "leave_value.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden1", type=int, default=128, help="First hidden layer width (default: 128)")
    ap.add_argument("--hidden2", type=int, default=64, help="Second hidden layer width (default: 64)")
    ap.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto-detect, preferring cuda then mps)")
    ap.add_argument(
        "--center-by-decision",
        action="store_true",
        help="Subtract each decision's mean target before training, so the loss measures only "
        "what can actually reorder candidates. Requires distill.py's `decision` column.",
    )
    ap.add_argument(
        "--warm-start",
        default=None,
        help="Initialise from this checkpoint instead of from random, so a round of policy "
        "iteration refines the champion rather than re-rolling an equally-good model.",
    )
    args = ap.parse_args()

    train(
        args.data,
        args.out,
        args.epochs,
        args.batch,
        args.lr,
        args.hidden1,
        args.hidden2,
        args.device,
        warm_start=args.warm_start,
        center_by_decision=args.center_by_decision,
    )
