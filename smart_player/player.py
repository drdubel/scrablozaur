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

import numpy as np
import torch

from scrablozaur import Board, Dawg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strategy import StrategicPlayer  # noqa: E402

from board_features import encode_board  # noqa: E402
from model import ALPHABET, DEFAULT_WEIGHTS_PATH, encode_leave, encode_leaves, get_model  # noqa: E402


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


# How heavily the learned leave value counts against a move's own score.
#
# **Retune this whenever the checkpoint changes.** The optimum is a property of
# the model, not a constant: it measures how far the estimate deserves to be
# trusted, and a better-trained net earns a higher weight. Measured at 1500
# seeded pairs per point, against the previous champion (`models/leave_v1.pt`):
#
#   w                      0.80    0.90    1.00    1.10    1.25    1.50
#   leave_v2 (2M games)    +0.7    +3.8    +5.4    +4.9    +1.2   -13.6
#   leave_k4 (200k games)    --    -2.0    -1.1     --     -3.1     --
#
# Same architecture, same lookahead, same target -- only the data volume
# differs, and the peaks land in different places. `leave_v1` peaked at 0.8
# because its predictions were noisy enough that leaning harder on them cost
# points; the current checkpoint (2M games) peaks at 1.0 and is worth
# +4.8 +/- 1.7 points/game over it once both play at their own optimum.
#
# The failure mode this guards against: benchmarking a new checkpoint at the
# incumbent's weight. Doing exactly that scored `leave_v2` at +0.7 +/- 1.5 -- a
# tie -- and nearly discarded a real five-point improvement.
DEFAULT_LEAVE_WEIGHT = 1.0


class SmartPlayer(StrategicPlayer):
    """StrategicPlayer with a learned leave evaluator. `model_path` defaults
    to the current champion checkpoint; pass an explicit path to play a
    specific candidate/older checkpoint instead (used by evaluate.py's
    --candidate mode and iterate.py to pit checkpoints against each other)."""

    def __init__(
        self,
        board: Board,
        model_path: str = DEFAULT_WEIGHTS_PATH,
        leave_weight: float = DEFAULT_LEAVE_WEIGHT,
        use_endgame: bool = True,
    ) -> None:
        super().__init__(board)
        self.model_path = model_path
        self.leave_weight = leave_weight
        # Off only for A/B measurement -- there is no reason to estimate a
        # position that can be searched exactly.
        self.use_endgame = use_endgame
        self.last_endgame_diff: int | None = None
        self.last_endgame_exact: bool | None = None

    def _cache_turn_context(self) -> None:
        """Board features and unseen-tile count for this decision.

        Neither changes across the candidates being compared -- only each
        candidate's hypothetical placement does -- so they are computed once and
        reused by every leave valuation this turn, including `play_word`'s
        play-vs-exchange comparison.
        """
        self._board_features = encode_board(self.board)
        self._unseen_tiles = len(self.get_letters_left())

    def _endgame_move(self, dawg: Dawg) -> tuple[str, int, tuple[int, int, bool], list[str]] | None:
        """Searched best play once the bag is empty, or `None` while it isn't.

        With no tiles left to draw the game stops being a game of chance: the
        opponent holds exactly the tiles that are neither on the board nor in
        our own rack, so the position is fully known and worth searching
        instead of guessing at. Estimating is at its weakest here and the
        margins are at their most decisive.
        """
        if not self.use_endgame or self.board.bag_remaining() > 0:
            return None
        counts = self.board.unseen_tile_counts(self.letters)
        opponent_rack = "".join(ALPHABET[i] * n for i, n in enumerate(counts))
        if not opponent_rack:
            # Opponent is already out; there is no game left to search.
            return None
        word, score, position, used, diff, _nodes, exact = self.board.solve_endgame(
            dawg, self.letters, opponent_rack
        )
        self.last_endgame_diff, self.last_endgame_exact = diff, exact
        # An empty word means the search preferred to pass, which the game loop
        # reads as a no-play -- exactly what a pass is, and the search already
        # accounted for the fact that two in a row end the game.
        return (word, score, position, used)

    def get_best_word(
        self, dawg: Dawg, parallel: bool
    ) -> tuple[str, int, tuple[int, int, bool], list[str]]:
        self._cache_turn_context()
        endgame = self._endgame_move(dawg)
        if endgame is not None:
            return endgame
        # Score every candidate's leave in a *single* batched forward pass,
        # rather than one batch-size-1 forward (and its per-call dispatch
        # overhead) per candidate.
        words = self.get_best_words(dawg, self.letters, parallel)
        if not words:
            return ("", 0, (0, 0, True), [])
        leaves = [
            remove_used(self.letters, used) for (_word, _pts, _pos, used) in words
        ]
        values = self._leave_values(leaves)
        # Highest move score plus the weighted leave value; `max` keeps the
        # first at the maximum, so an exact tie still falls to the
        # highest-scoring candidate as it always did.
        best_word, _ = max(zip(words, values), key=lambda wv: self._rank(wv[0][1], wv[1]))
        return (best_word[0], best_word[1], best_word[2], best_word[3])

    def _rank(self, points: int, leave_value: float) -> float:
        """A candidate's ranking value: its score plus what it leaves behind.

        No longer rounds the leave term. Rounding was there to reproduce an
        older integer tie-break, but it quantises `leave_weight` away entirely
        for weights below ~0.5 -- and the whole point of the weight is to find
        out whether the leave term is currently over-counted.
        """
        return points + self.leave_weight * leave_value

    def evaluate_word(
        self,
        dawg: Dawg,
        word: str,
        points: int,
        position: tuple[int, int, bool],
        used: list[str],
    ) -> float:
        """Single-candidate ranking value (move score + weighted leave value).
        `get_best_word` ranks the real candidate set in one batched pass
        instead of calling this per candidate; kept for direct/external use."""
        leave = remove_used(self.letters, used)
        return self._rank(points, self._leave_value(leave))

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

    def _leave_values(self, leaves: list[str]) -> list[float]:
        """Batched `_leave_value`: predicted value of each leave in `leaves`,
        in the current turn's cached board context, via one forward pass.
        Reuses model.encode_leaves (the vectorized encoder already used for
        dataset construction) so the whole (N, INPUT_DIM) batch is built with
        a handful of numpy passes rather than N per-sample tensor builds."""
        if not leaves:
            return []
        leaves_arr = np.asarray(leaves)
        unseen = np.full(len(leaves), self._unseen_tiles, dtype=np.float32)
        board_feats = np.tile(
            np.asarray(self._board_features, dtype=np.float32), (len(leaves), 1)
        )
        x = encode_leaves(leaves_arr, unseen, board_feats)
        with torch.inference_mode():
            return get_model(self.model_path)(x).tolist()

    def _best_exchange(self) -> tuple[str, float]:
        """Brute-force every non-empty subset of the rack to discard (i.e.
        every non-full subset to keep) and return (letters_to_exchange,
        predicted_value_of_the_kept_leave) for whichever keeps the
        highest-valued leave. At most 2**7 - 1 = 127 subsets for a full
        rack -- cheap enough to search exactly rather than guess via a
        hand-tuned heuristic."""
        letters = self.letters
        n = len(letters)
        keeps: list[str] = []
        discards: list[str] = []
        for discard_size in range(1, n + 1):
            for discard_idx in itertools.combinations(range(n), discard_size):
                idx_set = set(discard_idx)
                keeps.append(
                    "".join(ch for i, ch in enumerate(letters) if i not in idx_set)
                )
                discards.append("".join(letters[i] for i in discard_idx))
        if not keeps:
            return "", float("-inf")
        # One batched forward over all subsets instead of 127 batch-size-1 calls.
        values = self._leave_values(keeps)
        # `max` returns the first index at the maximum -- the same subset the
        # original strict `value > best_value` scan (which kept the earliest
        # max) would have chosen.
        best_i = max(range(len(values)), key=values.__getitem__)
        return discards[best_i], values[best_i]

    def play_word(self, dawg: Dawg, parallel: bool = False) -> str | None:
        """Returns the played word, `None` if letters were exchanged instead
        (a real, repeatable action -- not "no play"), or `""` only when
        genuinely no action is available (no legal word and can't exchange)."""
        word, points, position, used = self.get_best_word(dawg, parallel)
        word_value = (
            self._rank(points, self._leave_value(remove_used(self.letters, used)))
            if word
            else float("-inf")
        )

        if self.board.can_exchange():
            discard, exchange_value = self._best_exchange()
            # An exchange scores nothing, so its value is the kept leave alone --
            # weighted the same way, which is why the weight genuinely moves the
            # play-vs-exchange threshold instead of cancelling out.
            if self._rank(0, exchange_value) > word_value:
                self.exchange_letters(discard)
                self.last_exchanged = True
                return None

        self.last_exchanged = False

        if not word:
            return ""

        self.score += points
        self.board.place_word(word, position[0], position[1], position[2], used)
        self.letters = remove_used(self.letters, used)
        self.draw_letters()
        return word
