import os
import sys
from typing import Counter

from scrablozaur import Board

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strategy import (MAX_LEVEL, MAX_RANKED_LEVEL, MIN_LEVEL, StrategicPlayer, clamp_level,
                      pick_by_rank, rank_window)

from web.difficulty import DEFAULT_LEVEL, EngineMode, all_levels, engine_mode


def test_save_letters_left():
    board = Board()
    player = StrategicPlayer(board)
    player.letters = "abc"
    tile_bag = player.board.fresh_tile_bag()
    assert sorted(player.get_letters_left()) == sorted(list((Counter(tile_bag) - Counter(["a", "b", "c"])).elements()))


def test_save_letters_left_with_duplicates():
    board = Board()
    player = StrategicPlayer(board)
    player.letters = "aab"
    tile_bag = player.board.fresh_tile_bag()
    assert sorted(player.get_letters_left()) == sorted(list((Counter(tile_bag) - Counter(["a", "a", "b"])).elements()))


def test_exchange_letters():
    board = Board()
    player = StrategicPlayer(board)
    player.letters = "abcdefg"
    player.exchange_letters("abg")
    assert len(player.letters) == 7


# ── Custom difficulty level (the slider) ─────────────────────────────────────


def test_rank_window_gets_stricter_with_level():
    """The whole point of a dial is that every notch is at least as strong as
    the one below it -- a non-monotone window would make the slider lie."""
    windows = [rank_window(level) for level in range(MIN_LEVEL, MAX_LEVEL + 1)]
    for (prev_best, prev_worst), (best, worst) in zip(windows, windows[1:]):
        assert best <= prev_best and worst <= prev_worst
    assert windows[0] == (20, 40)
    assert rank_window(MAX_RANKED_LEVEL) == (1, 1)
    # Levels past the ranked range always want the best move on the list.
    assert rank_window(MAX_LEVEL) == (1, 1)


def test_clamp_level():
    assert clamp_level(-5) == MIN_LEVEL
    assert clamp_level(99) == MAX_LEVEL
    assert clamp_level(DEFAULT_LEVEL) == DEFAULT_LEVEL


def test_pick_by_rank_stays_in_window():
    ranked = [(f"w{i}", 100 - i, (0, i, True), ["a"]) for i in range(50)]
    for _ in range(50):
        chosen = pick_by_rank(ranked, 3)
        best, worst = rank_window(3)
        assert best <= ranked.index(chosen) + 1 <= worst


def test_pick_by_rank_clamps_to_short_lists():
    """A position offering two legal plays cannot honour a window starting at
    rank 20 -- the weakest level must still return a move."""
    ranked = [("a", 10, (0, 0, True), ["a"]), ("b", 5, (0, 1, True), ["b"])]
    assert pick_by_rank(ranked, MIN_LEVEL) in ranked
    assert pick_by_rank([], MIN_LEVEL) is None


def test_every_level_is_described_once():
    levels = all_levels()
    assert [info.level for info in levels] == list(range(MIN_LEVEL, MAX_LEVEL + 1))
    assert all(info.name and info.summary and info.expect for info in levels)
    assert len({info.name for info in levels}) == len(levels)


def test_engine_mode_per_level():
    assert engine_mode(MIN_LEVEL) is EngineMode.RANKED
    assert engine_mode(MAX_RANKED_LEVEL) is EngineMode.RANKED
    assert engine_mode(MAX_RANKED_LEVEL + 1) is EngineMode.SMART
    assert engine_mode(MAX_LEVEL) is EngineMode.SIM


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All tests passed.")
