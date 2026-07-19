from scrablozaur import Board, Dawg


class StrategicPlayer:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.letters = ""
        self.draw_letters()
        self.score = 0

    def draw_letters(self) -> None:
        """Draw letters from the bag to fill the player's hand up to 7 letters."""
        self.letters += self.board.give_letters(self.letters)

    def get_best_words(
        self, dawg: Dawg, letters: str, first: bool = False
    ) -> list[tuple[str, int, tuple[int, int, bool], list[str]]]:
        """Find the best scoring words that can be placed on the board with the given letters."""
        words = self.board.get_best_words(dawg, letters, first, n=50, parallel=True)

        return words

    def evaluate_word(
        self, dawg: Dawg, word: str, points: int, position: tuple[int, int, bool], used: list[str]
    ) -> int:
        """Evaluate the score of placing a word on the board at the given position and orientation."""
        used_points = sum(self.board.letter_points(ch) for ch in used)
        score = points - used_points

        return score

    def get_best_word(self, dawg: Dawg, first: bool = False) -> tuple[str, tuple[int, int, bool]]:
        """Find the best scoring word from the player's letters on the board."""
        words = self.get_best_words(dawg, self.letters, first)
        best_word = max(words, key=lambda w: self.evaluate_word(dawg, *w), default=None)

        return (best_word[0], best_word[2]) if best_word else ("", (0, 0, True))

    def play_word(self, dawg: Dawg, first: bool = False) -> str:
        """Find and play the best word from the player's letters on the board.

        This method should:
          - Analyze the board to find valid placement patterns.
          - Use the DAWG to find the best scoring word that can be formed with
            the player's letters and fits one of the patterns.
          - Place the word on the board and update the player's letters.
        """
        word, position = self.get_best_word(dawg, first)
        if not word:
            return ""

        points = self.board.calculate_word_points(word, position[0], position[1], position[2], self.letters)
        self.score += points
        self.board.place_word(word, position[0], position[1], position[2])
        for ch in word:
            if ch in self.letters:
                self.letters = self.letters.replace(ch, "", 1)
            else:
                self.letters = self.letters.replace("?", "", 1)
        self.draw_letters()
        return word
