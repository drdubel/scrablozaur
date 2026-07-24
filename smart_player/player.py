"""SmartPlayer: a StrategicPlayer whose `evaluate_word` ranks candidates by
the predicted value of the rack they would actually leave behind (see
model.py), instead of the constant, non-discriminating heuristic described
there.
"""

import os
import sys

import torch

from scrablozaur import Board, Dawg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strategy import StrategicPlayer  # noqa: E402

from board_features import encode_board  # noqa: E402
from model import DEFAULT_WEIGHTS_PATH, encode_leave, get_model  # noqa: E402


def remove_used(letters: str, used: list[str]) -> str:
    """Remove the tiles a move consumed from a rack; a letter not held
    literally is assumed to come from a blank, mirroring how the engine
    reports blank usage in a move's `used` list (see Board.get_best_words)."""
    for ch in used:
        if ch in letters:
            letters = letters.replace(ch, "", 1)
        elif "?" in letters:
            letters = letters.replace("?", "", 1)
        else:
            raise ValueError(f"Letter '{ch}' not found in rack '{letters}'.")
    return letters


class SmartPlayer(StrategicPlayer):
    """StrategicPlayer with a learned leave evaluator. `model_path` defaults
    to the current champion checkpoint; pass an explicit path to play a
    specific candidate/older checkpoint instead (used by evaluate.py's
    --candidate mode and iterate.py to pit checkpoints against each other)."""

    def __init__(self, board: Board, model_path: str = DEFAULT_WEIGHTS_PATH) -> None:
        super().__init__(board)
        self.model_path = model_path

    def get_best_word(self, dawg: Dawg, parallel: bool) -> tuple[str, int, tuple[int, int, bool], list[str]]:
        # The real board/rack don't change across the ~50 candidates being
        # compared in one decision -- only each candidate's hypothetical
        # placement does. Computing board features (and unseen-tile count)
        # once per turn here, instead of once per evaluate_word() call,
        # avoids redundantly recomputing the same value up to 50x per move.
        self._board_features = encode_board(self.board)
        self._unseen_tiles = len(self.get_letters_left())
        return super().get_best_word(dawg, parallel)

    def evaluate_word(
        self, dawg: Dawg, word: str, points: int, position: tuple[int, int, bool], used: list[str]
    ) -> int:
        leave = remove_used(self.letters, used)
        with torch.inference_mode():
            x = encode_leave(leave, self._unseen_tiles, self._board_features)
            leave_value = get_model(self.model_path)(x.unsqueeze(0)).item()
        return points + round(leave_value)
