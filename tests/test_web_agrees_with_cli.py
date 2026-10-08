"""The web app and the CLI players must make the same move.

They used not to. `web/game.py` carried its own copy of the bot's decision, and
it drifted: it generated candidates with the legacy per-span search (measured
1.8 points/move worse than the engine's generator), ignored `leave_weight`
entirely, and never searched the endgame. Nothing caught that, because nothing
compared the two.

This is that comparison. It is deliberately an end-to-end equality check rather
than a test of any one piece: whatever the shared decision function does, both
callers must do the same thing, so a future divergence fails here rather than
quietly shipping a weaker bot.
"""

import os
import sys

from scrablozaur import Board, Dawg

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# `web` is a real package rooted at the repo; `src` and `smart_player` are
# script-style and import their siblings directly, so both need to be on the
# path (same arrangement web/game.py itself uses).
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "smart_player"))

from languages import engine_language, load as load_language  # noqa: E402
from player import SmartPlayer, choose_move  # noqa: E402
from sim_player import choose_move_sim  # noqa: E402
from strategy import StrategicPlayer  # noqa: E402

from web.game import _engine_suggestions  # noqa: E402

_spec = load_language("pl")
_lang = engine_language(_spec)
_dawg = Dawg(_lang, str(_spec.dawg), str(_spec.gaddag))


def _positions(n_seeds: int = 12, moves: int = 8):
    """Real mid-game positions, reached by letting greedy players fill a board."""
    for seed in range(n_seeds):
        board = Board.seeded(_lang, seed)
        a, b = StrategicPlayer(board), StrategicPlayer(board)
        for _ in range(moves):
            for player in (a, b):
                if player.letters:
                    yield board, player.letters
                player.play_word(_dawg)


def test_web_and_cli_choose_the_same_move():
    """`choose_move` is the single implementation, so a SmartPlayer and the web
    path must agree given the same board and rack."""
    checked = 0
    for board, rack in _positions():
        player = SmartPlayer(board)
        player.letters = rack
        cli = player.get_best_word(_dawg, False)

        web = choose_move(board, _dawg, rack, board.bag_remaining())

        assert cli[0] == web.word, f"rack {rack!r}: cli {cli[0]!r} vs web {web.word!r}"
        assert cli[1] == web.score
        assert cli[2] == web.position
        checked += 1
    assert checked > 50, f"only {checked} positions exercised"


def test_web_suggestions_match_the_engine():
    """The web's suggestion list is the engine's top-n, in the engine's order --
    it is what the difficulty tiers sample from and what human hints show."""
    checked = 0
    for board, rack in _positions(n_seeds=6):
        engine = board.get_best_words(_dawg, rack, 10)
        web = _engine_suggestions(board, _dawg, rack, 10)
        assert len(web) == len(engine)
        for got, (word, score, (row, col, horizontal), _used) in zip(web, engine):
            assert (got["word"], got["score"]) == (word, score)
            assert (got["row"], got["col"], got["horizontal"]) == (row, col, horizontal)
            # `cells` drives the board highlight, so it must cover the whole word.
            assert len(got["cells"]) == len(word)
            assert got["cells"][0] == (row, col)
        checked += 1
    assert checked > 20


def test_endgame_is_skipped_when_the_bag_count_contradicts_the_board():
    """`bag_remaining` comes from the caller, so it can disagree with the board.

    The web app maintains its own `TileBag` separately from the engine's grid,
    so a stale or scanned position can claim an empty bag while the board still
    has 90 unseen tiles. The deduced 'opponent rack' is then far larger than a
    rack, and `solve_endgame` does not fail on that -- it simply never returns.
    It has to fall through to the static decision instead.
    """
    board = Board.seeded(_lang, 3)
    rack = board.give_letters("")
    # Nothing is on the board, so ~93 tiles are unseen -- not an endgame,
    # whatever the caller claims.
    assert sum(board.unseen_tile_counts(rack)) > 7

    choice = choose_move(board, _dawg, rack, bag_remaining=0)
    assert choice.endgame_diff is None, "ran an endgame search on an impossible rack"
    assert choice.word, "should still return a normal opening move"


def test_sim_and_static_agree_in_the_endgame():
    """With the bag empty there is nothing to sample, so the simulating path
    must defer to the same search the static path uses."""
    for seed in (11, 12, 13):
        board = Board.seeded(_lang, seed)
        a, b = StrategicPlayer(board), StrategicPlayer(board)
        while board.bag_remaining() > 0 and (a.letters or b.letters):
            a.play_word(_dawg)
            if board.bag_remaining() == 0:
                break
            b.play_word(_dawg)
        if board.bag_remaining() != 0 or not a.letters:
            continue
        static = choose_move(board, _dawg, a.letters, 0)
        simmed = choose_move_sim(board, _dawg, a.letters, 0)
        assert static.word == simmed.word
        assert simmed.endgame_diff == static.endgame_diff


if __name__ == "__main__":
    test_web_and_cli_choose_the_same_move()
    test_web_suggestions_match_the_engine()
    test_endgame_is_skipped_when_the_bag_count_contradicts_the_board()
    test_sim_and_static_agree_in_the_endgame()
    print("All tests passed.")
