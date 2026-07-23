import os
import sys
from typing import Counter

from scrablozaur import Board

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strategy import StrategicPlayer


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


if __name__ == "__main__":
    test_save_letters_left()
    test_save_letters_left_with_duplicates()
    test_exchange_letters()
    print("All tests passed.")
