from collections import Counter

from scrablozaur import Board, Dawg


class SimplePlayer:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.letters = ""
        self.draw_letters()
        self.score = 0

    def draw_letters(self) -> None:
        """Draw letters from the bag to fill the player's hand up to 7 letters."""
        self.letters += self.board.give_letters(self.letters)

    def play_word(self, dawg: Dawg) -> str:
        """Find and play the best word from the player's letters on the board.

        This method should:
          - Analyze the board to find valid placement patterns.
          - Use the DAWG to find the best scoring word that can be formed with
            the player's letters and fits one of the patterns.
          - Place the word on the board and update the player's letters.
        """
        w = self.board.get_best_word(dawg, self.letters, parallel=False)
        self.score += w[1]
        self.board.place_word(w[0], w[2][0], w[2][1], w[2][2])
        for ch in w[3]:
            # `ch` is the literal letter placed on the board, but if a blank
            # stood in for it, the rack only has '?' -- not `ch` -- so fall
            # back to removing the blank instead. Without this, `replace()`
            # silently no-ops on a literal it can't find, the blank never
            # actually leaves the rack, and it gets "reused" as a fresh
            # wildcard every subsequent turn for the rest of the game.
            if ch in self.letters:
                self.letters = self.letters.replace(ch, "", 1)
            elif "?" in self.letters:
                self.letters = self.letters.replace("?", "", 1)
            else:
                raise ValueError(f"Letter '{ch}' not found in player's letters.")

        self.draw_letters()
        return w[0]


class StrategicPlayer:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.letters = ""
        self.tile_bag = board.fresh_tile_bag()
        self.draw_letters()
        self.score = 0

    def exchange_letters(self, letters_to_exchange: str) -> None:
        """Exchange letters from the player's hand with new letters from the bag."""
        self.letters = self.board.exchange_letters(self.letters, letters_to_exchange)

    def draw_letters(self) -> None:
        """Draw letters from the bag to fill the player's hand up to 7 letters."""
        self.letters += self.board.give_letters(self.letters)

    def get_letters_left(self) -> list[str]:
        """Return the letters used in the last played word for scoring purposes."""
        used_letters = [ch for ch in self.board.__str__().split() if ch != "-"] + [ch for ch in self.letters]
        return list((Counter(self.tile_bag) - Counter(used_letters)).elements())

    def get_best_words(self, dawg: Dawg, letters: str) -> list[tuple[str, int, tuple[int, int, bool], list[str]]]:
        """Find the best scoring words that can be placed on the board with the given letters."""
        words = self.board.get_best_words(dawg, letters, n=50, parallel=True)

        return words

    def evaluate_word(
        self, dawg: Dawg, word: str, points: int, position: tuple[int, int, bool], used: list[str]
    ) -> int:
        """Evaluate the score of placing a word on the board at the given position and orientation."""
        used_points = sum(self.board.letter_points(ch) for ch in used)
        score = points - used_points

        return score

    def get_best_word(self, dawg: Dawg) -> tuple[str, tuple[int, int, bool], list[str]]:
        """Find the best scoring word from the player's letters on the board."""
        words = self.get_best_words(dawg, self.letters)
        best_word = max(words, key=lambda w: self.evaluate_word(dawg, *w), default=None)

        return (best_word[0], best_word[2], best_word[3]) if best_word else ("", (0, 0, True), [])

    def play_word(self, dawg: Dawg) -> str:
        """Find and play the best word from the player's letters on the board.

        This method should:
          - Analyze the board to find valid placement patterns.
          - Use the DAWG to find the best scoring word that can be formed with
            the player's letters and fits one of the patterns.
          - Place the word on the board and update the player's letters.
        """
        word, position, used = self.get_best_word(dawg)
        if not word:
            return ""

        points = self.board.calculate_word_points(word, position[0], position[1], position[2], self.letters)
        self.score += points
        self.board.place_word(word, position[0], position[1], position[2])
        for ch in used:
            # `ch` is the literal letter placed on the board, but if a blank
            # stood in for it, the rack only has '?' -- not `ch` -- so fall
            # back to removing the blank instead.
            if ch in self.letters:
                self.letters = self.letters.replace(ch, "", 1)
            elif "?" in self.letters:
                self.letters = self.letters.replace("?", "", 1)
            else:
                raise ValueError(f"Letter '{ch}' not found in player's letters.")

        self.draw_letters()
        return word
