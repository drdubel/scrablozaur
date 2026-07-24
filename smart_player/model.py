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
# + 1 scalar feature for game phase (see encode_leave).
INPUT_DIM = len(ALPHABET) + 1

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "models", "leave_value.pt")


def encode_leave(leave: str, unseen_tiles: int) -> torch.Tensor:
    """Feature vector for a rack leave: one count per tile type, plus a
    scalar for how many tiles are still unseen (not on the board or in this
    player's hand -- i.e. in the bag or the opponent's rack). A leave's
    value depends on game phase: tiles you'll happily draw past early on
    can become a liability once few unseen tiles remain to fix them.
    """
    x = torch.zeros(INPUT_DIM)
    for ch in leave:
        x[_INDEX[ch]] += 1.0
    x[-1] = unseen_tiles / 100.0
    return x


def encode_leaves(leaves: np.ndarray, unseen_tiles: np.ndarray) -> torch.Tensor:
    """Vectorized batch equivalent of calling encode_leave() once per
    sample -- same feature semantics, see its docstring. Building the
    whole (N, INPUT_DIM) matrix via one numpy pass per letter
    (len(ALPHABET) of them) instead of a Python loop per sample turns
    dataset construction from O(N) individual tensor mutations into
    O(len(ALPHABET)) vectorized calls -- the difference that matters once
    N is in the tens of millions (see train.py)."""
    x = np.zeros((len(leaves), INPUT_DIM), dtype=np.float32)
    for j, ch in enumerate(ALPHABET):
        x[:, j] = np.char.count(leaves, ch)
    x[:, -1] = np.asarray(unseen_tiles, dtype=np.float32) / 100.0
    return torch.from_numpy(x)


class LeaveValueNet(nn.Module):
    """Tiny MLP (~3.5k params). No convolution: a leave is an unordered
    multiset of tiles, with no spatial structure to exploit."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


DEFAULT_WEIGHTS_PATH = _WEIGHTS_PATH

_models: dict[str, LeaveValueNet] = {}


def get_model(path: str = DEFAULT_WEIGHTS_PATH) -> LeaveValueNet:
    """Lazily load a checkpoint once per (process, path) -- keyed by path
    rather than a single global slot so a champion and a candidate
    checkpoint can be loaded side by side (see player.py, evaluate.py's
    --candidate mode, and iterate.py)."""
    if path not in _models:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt["alphabet"] != ALPHABET:
            raise ValueError(f"{path} was trained against a different tile alphabet")
        model = LeaveValueNet()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _models[path] = model
    return _models[path]
