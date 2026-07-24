"""Learned rack-leave value: replaces StrategicPlayer's leave heuristic
(src/strategy.py's `evaluate_word`) with a small MLP trained on self-play
outcomes -- see generate_data.py and train.py. The existing heuristic sums
`letter_points` over `get_letters_left()`, which is the tiles not yet on the
board or in the player's own hand (i.e. still in the bag or the opponent's
rack) -- identical for every candidate word in a single decision, so it
never actually discriminates between candidates, only raw move score does.
This model instead scores the rack a candidate would actually leave behind.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from scrablozaur import Board

# Single source of truth for which tiles exist -- derived from the engine's
# own tile distribution rather than hardcoded, so it can never drift from it.
ALPHABET = sorted(set(Board.fresh_tile_bag()))
_INDEX = {ch: i for i, ch in enumerate(ALPHABET)}
# + 1 scalar for game phase (unseen tiles) + 5 board-context scalars from
# board_features.encode_board (tw/dw/tl/dl open fraction, board fill).
N_BOARD_FEATURES = 5
INPUT_DIM = len(ALPHABET) + 1 + N_BOARD_FEATURES

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "models", "leave_value.pt")


def encode_leave(leave: str, unseen_tiles: int, board_features: tuple[float, ...]) -> torch.Tensor:
    """Feature vector for a rack leave: one count per tile type, a scalar
    for how many tiles are still unseen (not on the board or in this
    player's hand -- i.e. in the bag or the opponent's rack), and the 5
    board-context scalars from board_features.encode_board(). A leave's
    value depends on both game phase (tiles you'll happily draw past early
    on can become a liability once few unseen tiles remain to fix them)
    and board context (e.g. a leftover S is worth more when bingo-enabling
    premium squares are still open) that the leave alone can't express.
    """
    x = torch.zeros(INPUT_DIM)
    for ch in leave:
        x[_INDEX[ch]] += 1.0
    x[len(ALPHABET)] = unseen_tiles / 100.0
    x[len(ALPHABET) + 1 :] = torch.tensor(board_features)
    return x


def encode_leaves(leaves: np.ndarray, unseen_tiles: np.ndarray, board_features: np.ndarray) -> torch.Tensor:
    """Vectorized batch equivalent of calling encode_leave() once per
    sample -- same feature semantics, see its docstring. `board_features`
    is an (N, 5) array (already plain floats -- see generate_data.py's
    npz columns -- so no per-sample Python loop is needed here, unlike the
    letter counts). Building the whole (N, INPUT_DIM) matrix via one numpy
    pass per letter (len(ALPHABET) of them) instead of a Python loop per
    sample turns dataset construction from O(N) individual tensor
    mutations into O(len(ALPHABET)) vectorized calls -- the difference
    that matters once N is in the tens of millions (see train.py)."""
    x = np.zeros((len(leaves), INPUT_DIM), dtype=np.float32)
    for j, ch in enumerate(ALPHABET):
        x[:, j] = np.char.count(leaves, ch)
    x[:, len(ALPHABET)] = np.asarray(unseen_tiles, dtype=np.float32) / 100.0
    x[:, len(ALPHABET) + 1 :] = np.asarray(board_features, dtype=np.float32)
    return torch.from_numpy(x)


# Defaults for checkpoints saved before hidden sizes were stored alongside
# the weights -- i.e. the original architecture, so old checkpoints
# (missing these keys) still load with the exact shape they were trained
# at. train.py's own CLI defaults are larger (see there): with datasets now
# in the tens of millions of samples, this size was chosen back when
# training was CPU-bound and slow, not because it's the right capacity for
# the data available.
_LEGACY_HIDDEN1 = 64
_LEGACY_HIDDEN2 = 32


class LeaveValueNet(nn.Module):
    """MLP over rack-leave features. No convolution: a leave is an
    unordered multiset of tiles, with no spatial structure to exploit."""

    def __init__(self, hidden1: int = _LEGACY_HIDDEN1, hidden2: int = _LEGACY_HIDDEN2) -> None:
        super().__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


DEFAULT_WEIGHTS_PATH = _WEIGHTS_PATH

_models: dict[str, LeaveValueNet] = {}


def get_model(path: str = DEFAULT_WEIGHTS_PATH) -> LeaveValueNet:
    """Lazily load a checkpoint once per (process, path) -- keyed by path
    rather than a single global slot so a champion and a candidate
    checkpoint can be loaded side by side (see player.py, evaluate.py's
    --candidate mode, and iterate.py). Reconstructs whatever hidden layer
    sizes the checkpoint was actually trained with (train.py saves them),
    falling back to the original architecture for older checkpoints that
    predate that (see _LEGACY_HIDDEN1/2 above)."""
    if path not in _models:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt["alphabet"] != ALPHABET:
            raise ValueError(f"{path} was trained against a different tile alphabet")
        # Older checkpoints (34-dim, no board features) predate this key --
        # assume their own dim so they still load, rather than mask a real
        # mismatch against today's INPUT_DIM with a silent default.
        ckpt_input_dim = ckpt.get("input_dim", len(ALPHABET) + 1)
        if ckpt_input_dim != INPUT_DIM:
            raise ValueError(
                f"{path} was trained with input_dim={ckpt_input_dim}, but the current feature set is "
                f"INPUT_DIM={INPUT_DIM} -- regenerate the dataset and retrain against the current encoding."
            )
        model = LeaveValueNet(ckpt.get("hidden1", _LEGACY_HIDDEN1), ckpt.get("hidden2", _LEGACY_HIDDEN2))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _models[path] = model
    return _models[path]
