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
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from languages import engine_language, load as load_language  # noqa: E402

# Which language this net is for. Levels 9-10 are the only consumers, and a net
# is only ever valid for the tile alphabet it was trained on -- see the guard in
# `get_model`.
#
# Set through the environment rather than an argument because the whole
# pipeline (generate_data -> train -> distill -> export_weights, plus arena and
# evaluate) fans out across process pools whose workers re-import this module;
# an env var is inherited by them for free, while a `--lang` flag would have to
# be threaded through every worker entry point.
#
#     SCRABLOZAUR_LANGUAGE=en uv run python smart_player/generate_data.py ...
LANGUAGE = os.environ.get("SCRABLOZAUR_LANGUAGE", "pl")

# Single source of truth for which tiles exist -- derived from the engine's
# own tile distribution rather than hardcoded, so it can never drift from it.
ALPHABET = sorted(set(engine_language(LANGUAGE).fresh_tile_bag()))
_INDEX = {ch: i for i, ch in enumerate(ALPHABET)}
# + 1 scalar for game phase (unseen tiles) + 5 board-context scalars from
# board_features.encode_board (tw/dw/tl/dl open fraction, board fill).
N_BOARD_FEATURES = 5
INPUT_DIM = len(ALPHABET) + 1 + N_BOARD_FEATURES

# Per-language, like the dictionaries and the OCR models: a net is only ever
# valid for the tile alphabet it was trained on, so they cannot share a
# directory without inviting exactly the mix-up `get_model` guards against.
_SPEC = load_language(LANGUAGE)
_WEIGHTS_PATH = (
    str(_SPEC.leave_net.checkpoint)
    if _SPEC.leave_net
    else os.path.join(os.path.dirname(__file__), "models", LANGUAGE, "leave_value.pt")
)


def encode_leave(
    leave: str, unseen_tiles: int, board_features: tuple[float, ...], alphabet: list[str] | None = None
) -> torch.Tensor:
    """Feature vector for a rack leave: one count per tile type, a scalar
    for how many tiles are still unseen (not on the board or in this
    player's hand -- i.e. in the bag or the opponent's rack), and the 5
    board-context scalars from board_features.encode_board(). A leave's
    value depends on both game phase (tiles you'll happily draw past early
    on can become a liability once few unseen tiles remain to fix them)
    and board context (e.g. a leftover S is worth more when bingo-enabling
    premium squares are still open) that the leave alone can't express.
    """
    alphabet = alphabet if alphabet is not None else ALPHABET
    index = _INDEX if alphabet is ALPHABET else {ch: i for i, ch in enumerate(alphabet)}
    x = torch.zeros(len(alphabet) + 1 + N_BOARD_FEATURES)
    for ch in leave:
        x[index[ch]] += 1.0
    x[len(alphabet)] = unseen_tiles / 100.0
    x[len(alphabet) + 1 :] = torch.tensor(board_features)
    return x


def encode_leaves(
    leaves: np.ndarray,
    unseen_tiles: np.ndarray,
    board_features: np.ndarray,
    alphabet: list[str] | None = None,
) -> torch.Tensor:
    """Vectorized batch equivalent of calling encode_leave() once per
    sample -- same feature semantics, see its docstring. `board_features`
    is an (N, 5) array (already plain floats -- see generate_data.py's
    npz columns -- so no per-sample Python loop is needed here, unlike the
    letter counts). Building the whole (N, INPUT_DIM) matrix via one numpy
    pass per letter (len(ALPHABET) of them) instead of a Python loop per
    sample turns dataset construction from O(N) individual tensor
    mutations into O(len(ALPHABET)) vectorized calls -- the difference
    that matters once N is in the tens of millions (see train.py)."""
    alphabet = alphabet if alphabet is not None else ALPHABET
    x = np.zeros((len(leaves), len(alphabet) + 1 + N_BOARD_FEATURES), dtype=np.float32)
    for j, ch in enumerate(alphabet):
        x[:, j] = np.char.count(leaves, ch)
    x[:, len(alphabet)] = np.asarray(unseen_tiles, dtype=np.float32) / 100.0
    x[:, len(alphabet) + 1 :] = np.asarray(board_features, dtype=np.float32)
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

    def __init__(
        self,
        hidden1: int = _LEGACY_HIDDEN1,
        hidden2: int = _LEGACY_HIDDEN2,
        input_dim: int = INPUT_DIM,
    ) -> None:
        super().__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        # Defaults to this process's language, but a checkpoint for another one
        # carries its own width -- one process serves every installed language.
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


DEFAULT_WEIGHTS_PATH = _WEIGHTS_PATH

_models: dict[str, LeaveValueNet] = {}
# Each loaded checkpoint's own tile order, for callers to encode against.
_alphabets: dict[str, list[str]] = {}


def model_alphabet(path: str = DEFAULT_WEIGHTS_PATH) -> list[str]:
    """The tile order `path`'s net was trained with. Callers must encode
    leaves against this, not against the module-level ALPHABET, or a net for
    another language silently reads the wrong features."""
    get_model(path)
    return _alphabets[path]


def get_model(path: str = DEFAULT_WEIGHTS_PATH) -> LeaveValueNet:
    """Lazily load a checkpoint once per (process, path) -- keyed by path
    rather than a single global slot so a champion and a candidate
    checkpoint can be loaded side by side (see player.py, evaluate.py's
    --candidate mode, and iterate.py). Reconstructs whatever hidden layer
    sizes the checkpoint was actually trained with (train.py saves them),
    falling back to the original architecture for older checkpoints that
    predate that (see _LEGACY_HIDDEN1/2 above)."""
    if path not in _models:
        # This tiny MLP is only ever run on small batches (one decision's ~50
        # candidates or ~127 exchange subsets); Torch's intra-op threading gives
        # no measurable speedup there (flat across 1..8 threads) and oversubscribes
        # when many pool workers each spin up all-core Torch. Pin to one thread.
        torch.set_num_threads(1)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        # The checkpoint's own alphabet is the authority, not this process's
        # LANGUAGE: one web process serves games in every installed language.
        # A net fed another language's leaves does not fail, it reads a
        # permuted feature vector and returns confident nonsense -- so callers
        # encode with `model_alphabet(path)` rather than a module global.
        ckpt_alphabet = list(ckpt["alphabet"])
        # Older checkpoints (34-dim, no board features) predate this key --
        # assume their own dim so they still load, rather than mask a real
        # mismatch with a silent default.
        ckpt_input_dim = ckpt.get("input_dim", len(ckpt_alphabet) + 1)
        expected_dim = len(ckpt_alphabet) + 1 + N_BOARD_FEATURES
        if ckpt_input_dim != expected_dim:
            raise ValueError(
                f"{path} was trained with input_dim={ckpt_input_dim}, but its {len(ckpt_alphabet)}-tile "
                f"alphabet encodes {expected_dim} -- regenerate the dataset and retrain."
            )
        _alphabets[path] = ckpt_alphabet
        model = LeaveValueNet(
            ckpt.get("hidden1", _LEGACY_HIDDEN1),
            ckpt.get("hidden2", _LEGACY_HIDDEN2),
            input_dim=ckpt_input_dim,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _models[path] = model
    return _models[path]
