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

# Standard rule: the game ends once nobody has taken a genuine no-action
# turn (no legal word and can't exchange -- play_word() returns "" for
# this) for this many consecutive turns in a row. Exchanging (play_word()
# returns None) is a real, repeatable action a player can take as many
# times as they want and never counts toward this on its own.
_NO_PLAY_LIMIT = 2

# Exchanging never counts toward _NO_PLAY_LIMIT (it's a legal, repeatable
# action), which leaves a hole: StrategicPlayer exchanges whenever its best
# move is worth < 6 points, so two of them can reach a state -- blocked/closed
# board, or a rack pool that keeps producing junk while the bag still holds a
# full rack -- where both sides exchange forever and play_game never returns.
# Measured rate in self-play: ~2 games in 700_000, which is exactly the kind of
# thing that silently wedges one worker (and therefore the whole batch) during
# a multi-million-game generate_data.py run. The standard tournament rule caps
# *scoreless* turns instead, and exchanges are scoreless, so mirror that here.
_NO_SCORE_LIMIT = 6


class GamePlayer(Protocol):
    score: int
    letters: str

    def play_word(self, dawg: Dawg, parallel: bool = ...) -> str | None: ...


def play_game(players: list[GamePlayer], dawg: Dawg, parallel: bool = False) -> None:
    """Play `players` (already dealt onto a shared board) to completion,
    mutating their `.score`/`.letters` in place. Turn order is simply
    `players[0], players[1], players[0], ...`."""
    turn = 0
    no_play_streak = 0
    no_score_streak = 0
    went_out_idx: int | None = None
    n = len(players)

    while True:
        idx = turn % n
        player = players[idx]
        word = player.play_word(dawg, parallel=parallel)
        if word:
            no_play_streak = 0
            no_score_streak = 0
            if not player.letters:
                went_out_idx = idx
                break
        elif word is None:
            # Exchanged -- a real action, so it doesn't count toward the
            # no-play streak, but it is scoreless and so must be bounded.
            no_score_streak += 1
            if no_score_streak >= _NO_SCORE_LIMIT:
                break
        else:
            no_play_streak += 1
            no_score_streak += 1
            if no_play_streak >= _NO_PLAY_LIMIT or no_score_streak >= _NO_SCORE_LIMIT:
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
