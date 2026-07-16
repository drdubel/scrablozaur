class Dawg:
    """DAWG (Directed Acyclic Word Graph) dictionary loaded from a binary file."""

    def __init__(self, path: str) -> None:
        """Load a DAWG from a .bin file built with the `build` command."""

    def contains(self, word: str) -> bool:
        """Return True if the word exists in the dictionary."""

    def __contains__(self, word: str) -> bool:
        """Support for the `in` operator: `"word" in dawg`."""

    def node_count(self) -> int:
        """Return the number of nodes in the DAWG after minimization."""

    def search(self, pattern: str, letters: str) -> list[str]:
        """Find all words matching the pattern, using letters from the bag.

        Pattern syntax:
          - regular letter  – must appear exactly at this position
          - '-'             – exactly one letter drawn from the bag
          - '*'             – zero or more letters drawn from the bag

        The letters argument is a bag with multiplicity: "aab" allows 'a' twice
        and 'b' once across all '-' and '*' positions combined.

        '?' in the bag acts as a wildcard tile (blank in Scrabble) that can
        substitute for any letter, but is consumed like a regular tile.

        Examples:
            dawg.search("l--y", "oadn")   # 4-letter words: l + 2 from bag + y
            dawg.search("l*y",  "oadn")   # l + any number from bag + y
            dawg.search("k-t",  "oar?")   # k + one from bag + t, '?' as wildcard
            dawg.search("*",    "abcd?")  # all words buildable from these tiles
        """

class Board:
    """Scrabble board with a DAWG dictionary and a bag of letters."""

    def __init__(self, board: list[list[str]]) -> None:
        """Initialize the board with a 15x15 grid of characters.

        Each cell can be:
          - a letter (e.g. 'a', 'b', ..., 'z') – fixed letter on the board
          - '-' – empty cell where a letter can be placed

        The board must be exactly 15 rows of 15 columns each.
        """

    def __str__(self) -> str:
        """Return a string representation of the board for printing."""

    def calculate_word_points(self, word: str, row: int, col: int, horizontal: bool, letters: str) -> int:
        """Calculate the points for placing a word at the given position and orientation.

        The word must fit on the board and can only be placed on empty cells ('-').
        The points are calculated based on letter values and board bonuses.
        """

    def place_word(self, word: str, row: int, col: int, horizontal: bool) -> None:
        """Place a word on the board at the given position and orientation.

        The word must fit on the board and can only be placed on empty cells ('-').
        This method modifies the board state by filling in the letters of the word.
        """

    def get_row_patterns(self, row_idx: int) -> list[tuple[int, int]]:
        """Return a list of valid horizontal patterns in the specified row.

        Each pattern is represented as a tuple (start, end) indicating the
        starting and ending column indices of the pattern.
        """

    def get_col_patterns(self, col_idx: int) -> list[tuple[int, int]]:
        """Return a list of valid vertical patterns in the specified column.

        Each pattern is represented as a tuple (start, end) indicating the
        starting and ending row indices of the pattern.
        """

    def get_all_patterns(self) -> list[tuple[int, int, int, bool]]:
        """Return a list of all valid patterns on the board for word placement.

        Each pattern is represented as a tuple:
          (index, start, end, horizontal)
        where:
          - index: row index if horizontal, column index if vertical
          - start: starting position of the pattern
          - end: ending position of the pattern
          - horizontal: True if it's a horizontal pattern, False if vertical
        """

    def best_word_from_pattern(self, dawg: Dawg, row: int, start: int, end: int, horizontal: bool, letters: str) -> str:
        """Find the best scoring word that can be placed in the specified pattern.

        This method generates a pattern string based on the current board state
        and uses the DAWG to find all matching words that can be formed with the
        provided letters. It then calculates the points for each valid word and
        returns the one with the highest score.
        """

    def get_best_word(
        self, dawg: Dawg, letters: str, first: bool, parallel: bool
    ) -> tuple[str, int, tuple[int, int, bool], list[str]]:
        """Find the best scoring word that can be placed on the board with the given letters.

        This method searches through all valid patterns on the board and uses the DAWG
        to find matching words that can be formed with the provided letters. It returns
        the best scoring word along with its position and orientation.
        The `first` parameter indicates whether this is the first move of the game, which affects the validity of placements (the first word must cover the center cell).
        The returned list contains the letters used from the player's hand.
        """

    def check_word_placement(self, dawg: Dawg, word: str, row: int, col: int, horizontal: bool) -> None:
        """Check if a word can be placed at the given position and orientation.

        This method raises an exception if the word cannot be placed due to:
          - Out of bounds
          - Overlapping with existing letters that do not match
          - Not connecting to any existing words (except for the first move)
        """

    def give_letters(self, letters: str) -> str:
        """Simulate drawing letters from the bag to fill the player's hand to 7 tiles.

        The `letters` argument represents the current letters in the player's hand.
        This method returns a string of new letters drawn from the bag, ensuring
        that the total number of letters (current + drawn) does not exceed 7.
        """

    @staticmethod
    def can_exchange(bag_remaining: int) -> bool:
        """Whether exchanging tiles for new ones is currently allowed.

        Standard Scrabble rule: exchanging is only permitted while at least 7
        tiles remain in the bag, regardless of how many tiles are exchanged.
        """

    @staticmethod
    def rack_value(letters: str) -> int:
        """Sum of face point values of the given rack (blanks score 0).

        Used for the standard end-of-game scoring adjustment: the player who
        goes out gains this value from each opponent's rack; everyone else
        loses it from their own.
        """

    @staticmethod
    def first_draw_winner(draws: list[str]) -> int:
        """Index into `draws` of who goes first: each player draws one tile,
        closest to 'A' in alphabet order wins, a blank ('?') beats every
        letter, first index wins ties. Does not consume/mutate any bag —
        the caller returns the drawn tiles before dealing real racks.
        """
