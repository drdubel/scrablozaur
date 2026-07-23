from collections import Counter

from scrablozaur import Board, Dawg

VOWELS = "aąeęioóuy"
CONSONANTS = "bcćdfghjklłmnńprsśtwzżź"


class SimplePlayer:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.letters = ""
        self.draw_letters()
        self.score = 0
        self.last_exchanged = False

    def draw_letters(self) -> None:
        """Draw letters from the bag to fill the player's hand up to 7 letters."""
        self.letters += self.board.give_letters(self.letters)

    def play_word(self, dawg: Dawg, parallel: bool = False) -> str:
        """Find and play the best word from the player's letters on the board.

        This method should:
          - Analyze the board to find valid placement patterns.
          - Use the DAWG to find the best scoring word that can be formed with
            the player's letters and fits one of the patterns.
          - Place the word on the board and update the player's letters.
        """
        word, points, position, used = self.board.get_best_word(dawg, self.letters, parallel)
        self.score += points
        self.board.place_word(word, position[0], position[1], position[2])
        for ch in used:
            if ch in self.letters:
                self.letters = self.letters.replace(ch, "", 1)
            elif "?" in self.letters:
                self.letters = self.letters.replace("?", "", 1)
            else:
                raise ValueError(f"Letter '{ch}' not found in player's letters.")

        self.draw_letters()
        return word


class StrategicPlayer:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.letters = ""
        self.draw_letters()
        self.score = 0
        self.last_exchanged = False

    def exchange_letters(self, letters_to_exchange: str) -> None:
        """Exchange letters from the player's hand with new letters from the bag."""
        self.letters = self.board.exchange_letters(self.letters, letters_to_exchange)

    def get_letter_balance(self) -> tuple[list[str], list[str]]:
        """Return the count of vowels and consonants in the player's letters."""
        vowels = [ch for ch in self.letters if ch in VOWELS]
        consonants = [ch for ch in self.letters if ch in CONSONANTS]

        return vowels, consonants

    def draw_letters(self) -> None:
        """Draw letters from the bag to fill the player's hand up to 7 letters."""
        self.letters += self.board.give_letters(self.letters)

    def get_letters_left(self) -> list[str]:
        """Return the letters used in the last played word for scoring purposes."""
        used_letters = [ch for ch in self.board.__str__().split() if ch != "-"] + [ch for ch in self.letters]

        return list((Counter(self.board.fresh_tile_bag()) - Counter(used_letters)).elements())

    def get_best_words(
        self, dawg: Dawg, letters: str, parallel: bool
    ) -> list[tuple[str, int, tuple[int, int, bool], list[str]]]:
        """Find the best scoring words that can be placed on the board with the given letters."""
        words = self.board.get_best_words(dawg, letters, n=50, parallel=parallel)

        return words

    def get_best_word(self, dawg: Dawg, parallel: bool) -> tuple[str, int, tuple[int, int, bool], list[str]]:
        """Find the best scoring word from the player's letters on the board."""
        words = self.get_best_words(dawg, self.letters, parallel)
        best_word = max(words, key=lambda w: self.evaluate_word(dawg, *w), default=None)

        return (best_word[0], best_word[1], best_word[2], best_word[3]) if best_word else ("", 0, (0, 0, True), [])

    def evaluate_word(
        self, dawg: Dawg, word: str, points: int, position: tuple[int, int, bool], used: list[str]
    ) -> int:
        """Evaluate the score of placing a word on the board at the given position and orientation."""
        left_points = sum(self.board.letter_points(ch) for ch in self.get_letters_left())
        score = points + left_points

        return score

    def get_letters_to_exchange(self) -> str:
        """Determine which letters to exchange based on the current hand and letter balance."""
        vowels, consonants = self.get_letter_balance()
        vowels = sorted(vowels, key=lambda ch: self.board.letter_points(ch))
        consonants = sorted(consonants, key=lambda ch: self.board.letter_points(ch))
        letters_to_exchange = ""

        if len(vowels) < 3:
            letters_to_exchange += "".join(consonants[:3])
        elif len(consonants) < 3:
            letters_to_exchange += "".join(vowels[:3])
        else:
            min_vowel = 1
            min_consonant = 3
            letters_to_exchange += "".join(vowels[::-1][min_vowel:]) + "".join(consonants[::-1][min_consonant:])

        return letters_to_exchange

    def play_word(self, dawg: Dawg, parallel: bool = False) -> str:
        """Find and play the best word from the player's letters on the board.

        This method should:
          - Analyze the board to find valid placement patterns.
          - Use the DAWG to find the best scoring word that can be formed with
            the player's letters and fits one of the patterns.
          - Place the word on the board and update the player's letters.
        """
        word, points, position, used = self.get_best_word(dawg, parallel)

        if self.board.can_exchange() and points < 10 and not self.last_exchanged:
            self.exchange_letters(self.get_letters_to_exchange())
            self.last_exchanged = True
            return ""

        self.last_exchanged = False

        if not word:
            return ""

        self.score += points
        self.board.place_word(word, position[0], position[1], position[2])
        for ch in used:
            if ch in self.letters:
                self.letters = self.letters.replace(ch, "", 1)
            elif "?" in self.letters:
                self.letters = self.letters.replace("?", "", 1)
            else:
                raise ValueError(f"Letter '{ch}' not found in player's letters.")

        self.draw_letters()
        return word
