"""The rules that decide when a game ends and how it is finally scored.

There were three copies of this: `smart_player/simulate.py`, `src/main.py` and
`web/game.py`. They had drifted -- `src/main.py` ended a game after **four**
consecutive no-plays where the others used two, and only `simulate.py` had the
scoreless-turn cap that stops two exchanging players wedging forever. A game
played through one harness was therefore not the same game played through
another, which makes benchmark numbers incomparable for no good reason.

This module is the single definition. It deliberately holds no board, no bag and
no turn order, only the rules, so the batch self-play loop, the CLI benchmark
and the interactive web session can all obey them without sharing a control
flow they cannot share.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from scrablozaur import Board

# A game ends once this many consecutive turns pass with nobody playing a word.
# Only a *genuine* no-action turn counts: no legal play and no exchange
# available. Exchanging is a real, repeatable action with no limit of its own.
NO_PLAY_LIMIT = 2

# ...but exchanging being unlimited leaves a hole. `StrategicPlayer` exchanges
# whenever its best move is worth under 6 points, so two of them can reach a
# closed board or a rack pool that keeps producing junk and exchange at each
# other indefinitely. Measured in self-play at roughly 2 games in 700,000 --
# rare, and exactly the kind of thing that silently wedges one worker and with
# it a whole multi-million-game generation run. The standard tournament rule
# caps *scoreless* turns, and an exchange is scoreless, so cap those.
NO_SCORE_LIMIT = 6


class Scoring(Protocol):
    """Anything with a score and a rack: a bot player, or a web seat."""

    score: int
    letters: str


class TurnResult(Enum):
    """What a player's turn actually was.

    `play_word` encodes this in its return value -- the word, `None` for an
    exchange, `""` for genuinely stuck -- which is easy to misread. Naming the
    three cases is what stopped "exchanged" and "stuck" being conflated, a bug
    that used to end games after a single round.
    """

    PLAYED = "played"
    EXCHANGED = "exchanged"
    NO_ACTION = "no_action"


def classify_turn(word: str | None) -> TurnResult:
    """Read `play_word`'s three-valued return."""
    if word:
        return TurnResult.PLAYED
    if word is None:
        return TurnResult.EXCHANGED
    return TurnResult.NO_ACTION


@dataclass
class Streaks:
    """The two counters that end a game, and the rule for reading them."""

    no_play: int = 0
    no_score: int = 0

    def record(self, result: TurnResult) -> None:
        if result is TurnResult.PLAYED:
            self.no_play = 0
            self.no_score = 0
            return
        # An exchange is a real action, so it never counts toward the no-play
        # streak -- but it scores nothing, so it does count toward that one.
        self.no_score += 1
        if result is TurnResult.NO_ACTION:
            self.no_play += 1

    @property
    def exhausted(self) -> bool:
        """True once the game should stop, nobody having gone out."""
        return self.no_play >= NO_PLAY_LIMIT or self.no_score >= NO_SCORE_LIMIT


def went_out(player: Scoring, bag_remaining: int) -> bool:
    """A player goes out by playing their last tile with the bag empty."""
    return not player.letters and bag_remaining == 0


def apply_end_of_game_scoring(players: list[Scoring], went_out_idx: int | None) -> None:
    """The standard final adjustment, applied in place.

    Whoever goes out gains the summed rack value of everyone else and keeps
    their own score (they hold nothing). If instead the game ran out of turns,
    every player simply loses the value of what they are still holding.
    """
    if went_out_idx is not None:
        others = sum(Board.rack_value(p.letters) for i, p in enumerate(players) if i != went_out_idx)
        players[went_out_idx].score += others
        for i, p in enumerate(players):
            if i != went_out_idx:
                p.score -= Board.rack_value(p.letters)
    else:
        for p in players:
            p.score -= Board.rack_value(p.letters)
