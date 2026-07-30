import random
from collections import Counter

from scrablozaur import Board, Dawg

VOWELS = "aąeęioóuy"
CONSONANTS = "bcćdfghjklłmnńprsśtwzżź"

# Rank windows for the beatable difficulty tiers, as (best, worst) 1-based ranks
# into the candidate list ordered by StrategicPlayer's own evaluation.
#
# A weaker opponent should play a *worse move*, not a differently-chosen one, so
# every tier ranks candidates identically and only differs in how far down the
# list it reaches. Picking uniformly inside a window rather than at a fixed rank
# keeps games from repeating themselves; clamping to the list length means a
# position offering three legal plays still works.
RANK_WINDOWS: dict[str, tuple[int, int]] = {
    "impossible": (1, 1),
    "hard": (1, 3),
    "medium": (5, 12),
    "easy": (15, 30),
}


def pick_by_rank(
    ranked: list[tuple[str, int, tuple[int, int, bool], list[str]]],
    difficulty: str,
) -> tuple[str, int, tuple[int, int, bool], list[str]] | None:
    """Choose from a best-first candidate list by `difficulty`'s rank window.

    A free function rather than a method so the web app picks its bot's move
    with exactly this code instead of a lookalike -- the same arrangement
    `smart_player.player.choose_move` uses, and for the same reason: the last
    time the web had its own copy of a decision, it silently drifted.
    """
    if not ranked:
        return None
    best, worst = RANK_WINDOWS[difficulty]
    # Clamp into range: a position offering two legal plays cannot honour a
    # window starting at rank 15, and should fall back to the worst it has
    # rather than failing or silently playing the best.
    hi = min(worst, len(ranked))
    lo = min(best, hi)
    return ranked[random.randint(lo, hi) - 1]


class SimplePlayer:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.letters = ""
        self.draw_letters()
        self.score = 0
        self.last_exchanged = False

    def exchange_letters(self, letters_to_exchange: str) -> None:
        """Exchange letters from the player's hand with new letters from the bag."""
        self.letters = self.board.exchange_letters(self.letters, letters_to_exchange)

    def draw_letters(self) -> None:
        """Draw letters from the bag to fill the player's hand up to 7 letters."""
        self.letters += self.board.give_letters(self.letters)

    def get_letter_balance(self) -> tuple[list[str], list[str]]:
        """Return the count of vowels and consonants in the player's letters."""
        vowels = [ch for ch in self.letters if ch in VOWELS]
        consonants = [ch for ch in self.letters if ch in CONSONANTS]

        return vowels, consonants

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
            min_consonant = 2
            letters_to_exchange += "".join(vowels[::-1][min_vowel:]) + "".join(
                consonants[::-1][min_consonant:]
            )

        return letters_to_exchange

    def play_word(self, dawg: Dawg, parallel: bool = False) -> str | None:
        """Find and play the best word from the player's letters on the board.

        This method should:
          - Analyze the board to find valid placement patterns.
          - Use the DAWG to find the best scoring word that can be formed with
            the player's letters and fits one of the patterns.
          - Place the word on the board and update the player's letters.

        Returns the played word, `None` if letters were exchanged instead
        (a real, repeatable action -- not "no play"), or `""` only when
        genuinely no action is available (no legal word and can't exchange).
        """
        word, points, position, used = self.board.get_best_word(
            dawg, self.letters, parallel
        )
        if not word:
            if self.board.can_exchange():
                self.exchange_letters(self.get_letters_to_exchange())
                self.last_exchanged = True
                return None
            self.last_exchanged = False
            return ""

        self.last_exchanged = False
        self.score += points
        self.board.place_word(word, position[0], position[1], position[2], used)
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
        used_letters = [ch for ch in self.board.__str__().split() if ch != "-"] + [
            ch for ch in self.letters
        ]

        return list(
            (Counter(self.board.fresh_tile_bag()) - Counter(used_letters)).elements()
        )

    def get_best_words(
        self, dawg: Dawg, letters: str, parallel: bool
    ) -> list[tuple[str, int, tuple[int, int, bool], list[str]]]:
        """Find the best scoring words that can be placed on the board with the given letters."""
        words = self.board.get_best_words(dawg, letters, n=50, parallel=parallel)

        return words

    def get_best_word(
        self, dawg: Dawg, parallel: bool
    ) -> tuple[str, int, tuple[int, int, bool], list[str]]:
        """Find the best scoring word from the player's letters on the board."""
        words = self.get_best_words(dawg, self.letters, parallel)
        # The leave heuristic (sum of letter_points over the unseen tiles) is
        # identical for every candidate in this decision -- it depends only on
        # the board and rack, not the candidate -- so compute it once here
        # instead of re-deriving it inside evaluate_word for all ~50 candidates.
        self._leave_points = sum(
            self.board.letter_points(ch) for ch in self.get_letters_left()
        )
        best_word = max(words, key=lambda w: self.evaluate_word(dawg, *w), default=None)

        return (
            (best_word[0], best_word[1], best_word[2], best_word[3])
            if best_word
            else ("", 0, (0, 0, True), [])
        )

    def evaluate_word(
        self,
        dawg: Dawg,
        word: str,
        points: int,
        position: tuple[int, int, bool],
        used: list[str],
    ) -> int:
        """Rank a candidate by its move score plus the value of the leave. The
        leave term is constant across a decision's candidates, so get_best_word
        precomputes it into self._leave_points once per move; this stays
        argmax-equivalent to computing it per candidate."""
        return points + self._leave_points

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
            min_consonant = 2
            letters_to_exchange += "".join(vowels[::-1][min_vowel:]) + "".join(
                consonants[::-1][min_consonant:]
            )

        return letters_to_exchange

    def play_word(self, dawg: Dawg, parallel: bool = False) -> str | None:
        """Find and play the best word from the player's letters on the board.

        This method should:
          - Analyze the board to find valid placement patterns.
          - Use the DAWG to find the best scoring word that can be formed with
            the player's letters and fits one of the patterns.
          - Place the word on the board and update the player's letters.

        Returns the played word, `None` if letters were exchanged instead
        (a real, repeatable action -- not "no play"), or `""` only when
        genuinely no action is available (no legal word and can't exchange).
        """
        word, points, position, used = self.get_best_word(dawg, parallel)

        if self.board.can_exchange() and points < 6:
            self.exchange_letters(self.get_letters_to_exchange())
            self.last_exchanged = True
            return None

        self.last_exchanged = False

        if not word:
            return ""

        self.score += points
        self.board.place_word(word, position[0], position[1], position[2], used)
        for ch in used:
            if ch in self.letters:
                self.letters = self.letters.replace(ch, "", 1)
            elif "?" in self.letters:
                self.letters = self.letters.replace("?", "", 1)
            else:
                raise ValueError(f"Letter '{ch}' not found in player's letters.")

        self.draw_letters()
        return word


class RankedPlayer(StrategicPlayer):
    """A StrategicPlayer that deliberately plays a worse move.

    The difficulty tiers used to live in `web/game.py` as a weighted-random
    sample over raw scores, which meant the web had a notion of "a weaker
    opponent" that no simulator could reproduce -- so tier strength could not be
    benchmarked, only guessed at. As a player class it runs anywhere the others
    do, including `arena.py` and self-play.

    It ranks candidates exactly as `StrategicPlayer` does and then reaches
    further down the list; `difficulty` names a window in `RANK_WINDOWS`.
    """

    def __init__(self, board: Board, difficulty: str = "hard") -> None:
        super().__init__(board)
        if difficulty not in RANK_WINDOWS:
            raise ValueError(f"unknown difficulty {difficulty!r} (expected one of {sorted(RANK_WINDOWS)})")
        self.difficulty = difficulty

    def get_best_word(
        self, dawg: Dawg, parallel: bool
    ) -> tuple[str, int, tuple[int, int, bool], list[str]]:
        words = self.get_best_words(dawg, self.letters, parallel)
        if not words:
            return ("", 0, (0, 0, True), [])

        # Same ordering StrategicPlayer's argmax would produce, kept whole so a
        # rank can be taken from it.
        self._leave_points = sum(self.board.letter_points(ch) for ch in self.get_letters_left())
        ranked = sorted(words, key=lambda w: self.evaluate_word(dawg, *w), reverse=True)
        chosen = pick_by_rank(ranked, self.difficulty)
        if chosen is None:
            return ("", 0, (0, 0, True), [])
        return (chosen[0], chosen[1], chosen[2], chosen[3])
