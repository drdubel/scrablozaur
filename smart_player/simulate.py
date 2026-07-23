"""Shared self-play game loop: plays two players against each other to
completion, applying the standard end-of-game rack-value adjustment (see
Board.rack_value). Used by both generate_data.py (to harvest leave-value
training samples) and evaluate.py (to benchmark win rate) so the two share
one definition of what "a finished game" means, ported from web/game.py's
`_check_game_over`/`_apply_end_of_game_scoring` (the fullest, most correct
implementation of these rules already in the repo).
"""

from typing import Protocol

from scrablozaur import Board, Dawg

# Standard rule: the game ends once nobody has played a word for this many
# consecutive turns in a row (pass/exchange both count as "no play").
_NO_PLAY_LIMIT = 2


class GamePlayer(Protocol):
    score: int
    letters: str

    def play_word(self, dawg: Dawg, parallel: bool = ...) -> str: ...


def play_game(players: list[GamePlayer], dawg: Dawg, parallel: bool = False) -> None:
    """Play `players` (already dealt onto a shared board) to completion,
    mutating their `.score`/`.letters` in place. Turn order is simply
    `players[0], players[1], players[0], ...`."""
    turn = 0
    no_play_streak = 0
    went_out_idx: int | None = None
    n = len(players)

    while True:
        idx = turn % n
        player = players[idx]
        word = player.play_word(dawg, parallel=parallel)
        if word:
            no_play_streak = 0
            if not player.letters:
                went_out_idx = idx
                break
        else:
            no_play_streak += 1
            if no_play_streak >= _NO_PLAY_LIMIT:
                break
        turn += 1

    if went_out_idx is not None:
        others_value = sum(Board.rack_value(p.letters) for i, p in enumerate(players) if i != went_out_idx)
        players[went_out_idx].score += others_value
        for i, p in enumerate(players):
            if i != went_out_idx:
                p.score -= Board.rack_value(p.letters)
    else:
        for p in players:
            p.score -= Board.rack_value(p.letters)
