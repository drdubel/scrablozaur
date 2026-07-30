"""Shared self-play game loop.

Plays players against each other to completion and applies the standard
end-of-game adjustment. The rules themselves live in `src/rules.py` -- this is
only the turn order -- so that batch self-play, the CLI benchmark and the
interactive web session all end games the same way rather than each carrying
their own slightly different copy.

Used by `generate_data.py` and `distill.py` (to harvest training samples),
`evaluate.py` and `arena.py` (to benchmark), and `src/main.py`.
"""

import os
import sys
import time
from collections.abc import Callable
from typing import Protocol

from scrablozaur import Dawg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rules import Streaks, TurnResult, apply_end_of_game_scoring, classify_turn  # noqa: E402


class GamePlayer(Protocol):
    score: int
    letters: str

    def play_word(self, dawg: Dawg, parallel: bool = ...) -> str | None: ...


def play_game(
    players: list[GamePlayer],
    dawg: Dawg,
    parallel: bool = False,
    on_turn: Callable[[int, GamePlayer, str | None, TurnResult, float], None] | None = None,
) -> None:
    """Play `players` (already dealt onto a shared board) to completion,
    mutating their `.score`/`.letters` in place. Turn order is simply
    `players[0], players[1], players[0], ...`.

    `on_turn(idx, player, word, result, elapsed)` fires after each turn. It
    exists so callers that need a running commentary -- `src/main.py` builds a
    transcript and per-move timings -- can have one without keeping a second
    copy of the loop, which is how that file drifted to a different end
    condition in the first place.
    """
    turn = 0
    streaks = Streaks()
    went_out_idx: int | None = None
    n = len(players)

    while True:
        idx = turn % n
        player = players[idx]
        started = time.perf_counter()
        word = player.play_word(dawg, parallel=parallel)
        elapsed = time.perf_counter() - started
        result = classify_turn(word)
        streaks.record(result)
        if on_turn is not None:
            on_turn(idx, player, word, result, elapsed)

        # An empty rack after a play means the bag was empty too: `draw_letters`
        # always tops back up to seven while anything remains.
        if result is TurnResult.PLAYED and not player.letters:
            went_out_idx = idx
            break
        if streaks.exhausted:
            break
        turn += 1

    apply_end_of_game_scoring(players, went_out_idx)
