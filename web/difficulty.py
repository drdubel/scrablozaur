"""Difficulty as a single custom level (1-10) instead of named tiers.

The web app used to offer six fixed buttons (easy/medium/hard/impossible/
smart/sim). Two problems with that: the gaps between them were large and
unadjustable -- "medium" is beatable by anyone, "hard" beats most people --
and the names said nothing about *what the bot would actually do*, so picking
one was guesswork.

Now there is one dial. `src/strategy.py` owns the part a simulator needs (a
level -> rank window, so `RankedPlayer` and the web bot pick moves with the
same code), and this module owns the part a *player* needs: which decision
procedure a level runs, and a description of it honest enough to set
expectations before the game starts. `GET /api/game/difficulty-levels` serves
these to the setup dialog, so the slider's feedback text is derived from the
real windows rather than a hand-written copy that can drift.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strategy import (  # noqa: E402
    MAX_LEVEL,
    MAX_RANKED_LEVEL,
    MIN_LEVEL,
    clamp_level,
    rank_window,
)

__all__ = [
    "MAX_LEVEL",
    "MIN_LEVEL",
    "DEFAULT_LEVEL",
    "SMART_LEVEL",
    "SIM_LEVEL",
    "EngineMode",
    "LevelInfo",
    "clamp_level",
    "engine_mode",
    "describe_level",
    "all_levels",
]

# The two top levels stop reaching down a ranked list and change the decision
# itself. They are the last two steps of the dial rather than a separate
# control, because from a player's point of view they are simply "even better
# opponent" -- the mechanism is an implementation detail the description names.
SMART_LEVEL = MAX_RANKED_LEVEL + 1  # 9: weighs the rack it leaves behind
SIM_LEVEL = MAX_LEVEL               # 10: simulates the opponent's reply

# Middle of the dial: something a decent club player has to work for, and an
# obvious direction to move in either way. The old default was "hard", which
# started most people off losing.
DEFAULT_LEVEL = 5


class EngineMode(str, Enum):
    """Which decision procedure a level runs."""

    RANKED = "ranked"  # best-first list, pick from a rank window
    SMART = "smart"    # score + learned leave value (smart_player.choose_move)
    SIM = "sim"        # Monte-Carlo rollouts (smart_player.choose_move_sim)


def engine_mode(level: int) -> EngineMode:
    level = clamp_level(level)
    if level >= SIM_LEVEL:
        return EngineMode.SIM
    if level >= SMART_LEVEL:
        return EngineMode.SMART
    return EngineMode.RANKED


@dataclass(frozen=True)
class LevelInfo:
    level: int
    name: str
    emoji: str
    #: What the bot does, mechanically -- derived from the real rank window.
    summary: str
    #: What the person on the other side of the board should expect.
    expect: str
    engine: EngineMode
    #: Rank window into the best-first list; None once the level stops using one.
    rank_best: int | None
    rank_worst: int | None
    #: True for levels that spend noticeable time thinking (~0.1 s/move).
    slow: bool


# Name + expectation per level. Written in a neutral, descriptive register:
# each entry states the competitive level of the opponent and its practical
# consequence for the game; the mechanical half of the description is generated.
_LEVELS: dict[int, tuple[str, str, str]] = {
    1: ("Wprowadzający", "🌱", "Poziom szkoleniowy z wyraźną przewagą gracza. Zalecany do nauki zasad."),
    2: ("Początkujący", "🍀", "Zdecydowana przewaga gracza; przeciwnik znajduje jedynie pojedyncze wartościowe słowa."),
    3: ("Rekreacyjny", "🎈", "Przeciwnik gra poprawnie, ale rzadko wykorzystuje pola premiowane."),
    4: ("Towarzyski", "🎲", "Wyrównana rywalizacja na poziomie gry towarzyskiej."),
    5: ("Klubowy", "🎯", "Poziom klubowy: przeciwnik systematycznie sięga po pola premiowane i wymaga uważnej gry."),
    6: ("Zaawansowany", "🔥", "Wysoki poziom gry — każdy słabszy ruch szybko przekłada się na stratę punktową."),
    7: ("Ekspert", "🧗", "Przeciwnik niemal zawsze wybiera jeden z dwóch najlepiej punktujących ruchów."),
    8: ("Maksymalizator", "💀", "Przeciwnik zawsze zagrywa ruch o najwyższej możliwej punktacji na planszy."),
    9: ("Strateg", "🧠", "Poziom turniejowy: punktacja bilansowana z jakością liter pozostawianych na stojaku."),
    10: ("Analityk", "🔮", "Najwyższy dostępny poziom: przed wyborem ruchu analizowane są możliwe odpowiedzi gracza."),
}


def _summary(level: int) -> str:
    mode = engine_mode(level)
    if mode is EngineMode.SIM:
        return "Symuluje odpowiedzi przeciwnika metodą Monte-Carlo i wybiera ruch o najlepszym bilansie punktowym."
    if mode is EngineMode.SMART:
        return "Ocenia ruch łącznie z wartością liter pozostających na stojaku."
    best, worst = rank_window(level)
    if best == worst == 1:
        return "Zawsze wybiera ruch nr 1 w rankingu najlepiej punktujących."
    if best == worst:
        return f"Zawsze wybiera ruch nr {best} w rankingu najlepiej punktujących."
    return f"Losuje spośród ruchów nr {best}–{worst} w rankingu najlepiej punktujących."


def describe_level(level: int) -> LevelInfo:
    level = clamp_level(level)
    name, emoji, expect = _LEVELS[level]
    mode = engine_mode(level)
    best, worst = rank_window(level) if mode is EngineMode.RANKED else (None, None)
    return LevelInfo(
        level=level,
        name=name,
        emoji=emoji,
        summary=_summary(level),
        expect=expect,
        engine=mode,
        rank_best=best,
        rank_worst=worst,
        slow=mode is EngineMode.SIM,
    )


def all_levels() -> list[LevelInfo]:
    return [describe_level(level) for level in range(MIN_LEVEL, MAX_LEVEL + 1)]
