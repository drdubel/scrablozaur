import os
import sys

from scrablozaur import Board, Dawg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strategy import StrategicPlayer


def test_save_letters_left():
    board = Board()
    player = StrategicPlayer(board)
    player.letters = "abc"
    player.tile_bag = ["a", "b", "c", "d", "e"]
    assert sorted(player.get_letters_left()) == sorted(["d", "e"])


def test_save_letters_left_with_duplicates():
    board = Board()
    player = StrategicPlayer(board)
    player.letters = "aab"
    player.tile_bag = ["a", "a", "b", "c", "d", "e"]
    assert sorted(player.get_letters_left()) == sorted(["c", "d", "e"])


if __name__ == "__main__":
    test_save_letters_left()
    test_save_letters_left_with_duplicates()
    print("All tests passed.")
