"""SmartPlayer: a StrategicPlayer whose `evaluate_word` ranks candidates by
the predicted value of the rack they would actually leave behind (see
model.py), instead of the constant, non-discriminating heuristic described
there. Also overrides `play_word` to make the play-vs-exchange decision (and
which tiles to exchange) with the same learned value model, instead of
StrategicPlayer's fixed `points < 6` threshold and vowel/consonant-balance
heuristic (`get_letters_to_exchange`, src/strategy.py).
"""

import itertools
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
        return points + round(self._leave_value(leave))

    def _leave_value(self, leave: str) -> float:
        """Predicted value of holding `leave`, in the current turn's board
        context (self._board_features/_unseen_tiles, cached by
        get_best_word()). Also the natural way to score an exchange: the
        tiles you *keep* when exchanging are exactly a "leave" in the same
        sense a leave-after-a-move is -- what you draw to replace the
        discarded tiles is random either way, so the model's prediction for
        the kept subset is the best available signal for ranking exchanges,
        same as it ranks candidate moves."""
        with torch.inference_mode():
            x = encode_leave(leave, self._unseen_tiles, self._board_features)
            return get_model(self.model_path)(x.unsqueeze(0)).item()

    def _best_exchange(self) -> tuple[str, float]:
        """Brute-force every non-empty subset of the rack to discard (i.e.
        every non-full subset to keep) and return (letters_to_exchange,
        predicted_value_of_the_kept_leave) for whichever keeps the
        highest-valued leave. At most 2**7 - 1 = 127 subsets for a full
        rack -- cheap enough to search exactly rather than guess via a
        hand-tuned heuristic."""
        letters = self.letters
        n = len(letters)
        best_value = float("-inf")
        best_discard = ""
        for discard_size in range(1, n + 1):
            for discard_idx in itertools.combinations(range(n), discard_size):
                idx_set = set(discard_idx)
                keep = "".join(ch for i, ch in enumerate(letters) if i not in idx_set)
                value = self._leave_value(keep)
                if value > best_value:
                    best_value = value
                    best_discard = "".join(letters[i] for i in discard_idx)
        return best_discard, best_value

    def play_word(self, dawg: Dawg, parallel: bool = False) -> str | None:
        """Returns the played word, `None` if letters were exchanged instead
        (a real, repeatable action -- not "no play"), or `""` only when
        genuinely no action is available (no legal word and can't exchange)."""
        word, points, position, used = self.get_best_word(dawg, parallel)
        word_value = points + round(self._leave_value(remove_used(self.letters, used))) if word else float("-inf")

        if self.board.can_exchange():
            discard, exchange_value = self._best_exchange()
            if round(exchange_value) > word_value:
                self.exchange_letters(discard)
                self.last_exchanged = True
                return None

        self.last_exchanged = False

        if not word:
            return ""

        self.score += points
        self.board.place_word(word, position[0], position[1], position[2])
        self.letters = remove_used(self.letters, used)
        self.draw_letters()
        return word
