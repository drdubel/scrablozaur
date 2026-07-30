def set_num_threads(n: int) -> None:
    """Set how many threads parallel move generation uses.

    Applies to `Board.get_best_words(..., parallel=True)`. Must be called before
    the first generation; raises `RuntimeError` if the pool is already built.
    Overrides the `RAYON_NUM_THREADS` environment variable and the default of
    `min(8, cores)` — the engine caps its own pool below the full core count
    because a single move is too cheap to parallelise across many threads.
    """

def num_threads() -> int:
    """Number of threads parallel move generation will use (builds the pool on
    the first call if it does not exist yet)."""

class Dawg:
    """DAWG (Directed Acyclic Word Graph) dictionary loaded from a binary file.

    Also carries a GADDAG for the same lexicon (loaded from a sibling
    `gaddag.bin`, or an explicit path) that powers `Board.get_best_words`
    move generation. Word lookups (`contains`) always use the compact DAWG.
    """

    def __init__(self, path: str, gaddag_path: str | None = None) -> None:
        """Load a DAWG from a .bin file built with the `build` command.

        If `gaddag_path` is omitted, a sibling GADDAG is auto-loaded when
        present: same directory, with `dawg` replaced by `gaddag` in the
        filename (e.g. `words/dawg.bin` -> `words/gaddag.bin`, built with the
        `build-gaddag` command). When a GADDAG is loaded, `Board.get_best_words`
        uses fast anchor-based generation; otherwise it falls back to the legacy
        DAWG pattern search. Pass an explicit `gaddag_path` to override, or leave
        both absent to run purely on the DAWG.
        """

    def contains(self, word: str) -> bool:
        """Return True if the word exists in the dictionary."""

    def __contains__(self, word: str) -> bool:
        """Support for the `in` operator: `"word" in dawg`."""

    def node_count(self) -> int:
        """Return the number of nodes in the DAWG after minimization."""

    def has_gaddag(self) -> bool:
        """Whether a GADDAG is loaded, i.e. whether `Board.get_best_words`
        uses the fast anchor-based generator rather than the legacy fallback."""

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

    def __init__(self) -> None:
        """Initialize an empty 15x15 board with a full standard tile bag.

        The bag's draw order is seeded from the clock (mixed with a
        per-process counter). Use `Board.seeded` when you need it repeatable.
        """

    @staticmethod
    def seeded(seed: int) -> Board:
        """An empty board whose bag draws deterministically from `seed`.

        Two boards built with the same seed and driven through the same
        sequence of draws and exchanges deal identical tiles. This is what
        lets a benchmark play one bag twice with the seats swapped and compare
        the two results as a matched pair.
        """

    def copy(self) -> Board:
        """Deep copy of the whole position: grid, blank mask, bag, opening-move
        flag, and the draw RNG's stream position.

        The only faithful way to branch a position -- `str()` + `from_grid()`
        loses the bag and, without a blank mask, which tiles were blanks.
        """

    @staticmethod
    def from_grid(board: list[list[str]], blanks: list[list[bool]] | None = None) -> Board:
        """Construct a board pre-filled from a 15x15 grid of characters.

        Each cell can be:
          - a letter (e.g. 'a', 'b', ..., 'z') – fixed letter on the board
          - '-' – empty cell where a letter can be placed

        `blanks`, if given, is a 15x15 mask marking which occupied squares hold
        a blank tile playing as that letter; pair it with `Board.blank_mask()`
        to round-trip a position without losing scoring information.

        The board must be exactly 15 rows of 15 columns each. Tiles already on
        the grid are subtracted from the bag (a blank consuming a `'?'` rather
        than a copy of the letter it stands in for), and the call raises if the
        grid needs more copies of a tile than the distribution has.
        """

    def __str__(self) -> str:
        """Return a string representation of the board for printing."""

    def calculate_word_points(self, word: str, row: int, col: int, horizontal: bool, letters: str) -> int:
        """Calculate the points for placing a word at the given position and orientation.

        The word must fit on the board and can only be placed on empty cells ('-').
        The points are calculated based on letter values and board bonuses.
        """

    def place_word(
        self,
        word: str,
        row: int,
        col: int,
        horizontal: bool,
        used: list[str] | None = None,
    ) -> list[tuple[int, int]]:
        """Place a word on the board at the given position and orientation.

        Returns the cells the play actually filled, in placement order -- pass
        that straight to `unplace_word` to undo it.

        `used` is the move's tile list exactly as `get_best_words` returns it:
        one entry per newly covered square, `'?'` where a blank stood in for a
        letter. Passing it is what lets the board remember that a square holds
        a blank, so that square keeps scoring 0 in every later word that runs
        through it. Omitting it treats every placed tile as a real one, which
        silently over-scores the rest of the game if the play used a blank.
        """

    def unplace_word(self, cells: list[tuple[int, int]], was_first: bool = False) -> None:
        """Undo a `place_word`, given the cells it returned.

        `was_first` restores the opening-move flag, which a search unwinding
        the first play of the game has to put back. Cheaper than cloning the
        board at every node.
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

    def get_best_words(
        self, dawg: Dawg, letters: str, n: int, parallel: bool = True
    ) -> list[tuple[str, int, tuple[int, int, bool], list[str]]]:
        """Find the `n` best scoring words that can be placed on the board with the given letters.

        Returns up to `n` candidates, best scoring first, each with its position and
        orientation. Each returned list contains the letters used from the player's hand
        for that candidate, one entry per newly placed tile. Real tiles are allocated before
        blanks, so a letter used more times than the hand has real copies of it reports `'?'`
        (a blank standing in for it, scoring 0) for the extra occurrences rather than the
        literal letter.

        When the `Dawg` carries a GADDAG (see `Dawg.__init__`), this enumerates every legal
        play via anchor-based bidirectional generation and returns the true top `n`. The
        legacy DAWG fallback keeps only the single best word per board span before ranking,
        so for `n > 1` it can under-report a span's other high-scoring plays; the best move
        (`n == 1`, `get_best_word`) is identical for both. Scores are computed identically
        either way.
        """

    def get_best_word(
        self, dawg: Dawg, letters: str, parallel: bool = True
    ) -> tuple[str, int, tuple[int, int, bool], list[str]]:
        """Find the best scoring word that can be placed on the board with the given letters.

        Equivalent to `get_best_words(dawg, letters, n=1, parallel=parallel)[0]`,
        or `("", 0, (0, 0, True), [])` if no valid word can be placed.
        The returned list contains the letters used from the player's hand, one entry per newly
        placed tile, with `'?'` for any tile a blank had to stand in for (see `get_best_words`).
        """

    def all_moves(
        self, dawg: Dawg, letters: str, parallel: bool = True
    ) -> list[tuple[str, int, tuple[int, int, bool], list[str]]]:
        """Every legal play, unsorted and untruncated.

        Same tuple shape as `get_best_words`. Use this when you rank candidates
        by something other than raw score: `get_best_words` keeps the top `n`
        *by score*, and a tile-dumping play, a blocking play, or the play that
        goes out first in an endgame can each be worth more than its score and
        fall outside that cut. Skips the sort, so it is also cheaper than
        asking `get_best_words` for a huge `n`.

        Requires a GADDAG (see `Dawg.__init__`) and raises `ValueError` without
        one -- the legacy pattern search only yields the best word per board
        span, so it cannot enumerate.
        """

    def check_word_placement(self, dawg: Dawg, word: str, row: int, col: int, horizontal: bool) -> None:
        """Check the cross-words a placement would form.

        Raises if the play is shorter than 2 letters, runs out of bounds, or
        forms a perpendicular cross-word that is not in the dictionary.

        It does *not* check the main word against the dictionary, and it does
        not check connectivity to existing tiles -- callers that accept
        arbitrary (e.g. human-entered) moves must do both themselves.
        """

    def give_letters(self, letters: str) -> str:
        """Simulate drawing letters from the bag to fill the player's hand to 7 tiles.

        The `letters` argument represents the current letters in the player's hand.
        This method returns a string of new letters drawn from the bag, ensuring
        that the total number of letters (current + drawn) does not exceed 7.
        Advances the board's draw RNG (see `Board.seeded`).
        """

    def bag_remaining(self) -> int:
        """How many tiles are left in the bag.

        Public information in Scrabble, so this is safe for a player to
        consult -- unlike `bag_tiles`.
        """

    def bag_tiles(self) -> list[str]:
        """The bag's actual contents.

        Hidden information: for benchmarks, tests and exact position setup, not
        for a player's own decision-making.
        """

    def set_bag(self, tiles: list[str]) -> None:
        """Replace the bag wholesale.

        For constructing exact positions (endgame tests, pre-endgame
        enumeration) without replaying a whole game up to them.
        """

    def blank_mask(self) -> list[list[bool]]:
        """15x15 mask of which occupied squares hold a blank tile.

        Pairs with `from_grid(grid, blanks)` to round-trip a position.
        """

    def board_feature_scalars(self) -> tuple[float, float, float, float, float]:
        """`(tw_open, dw_open, tl_open, dl_open, board_fill)`.

        The fraction of each premium-square type still unclaimed, and how full
        the board is. The same five scalars
        `smart_player/board_features.encode_board` computes, but read from the
        engine's own bonus layout rather than a transcription of it.
        """

    def unseen_tile_counts(self, rack: str) -> list[int]:
        """Counts of the tiles neither on the board nor in `rack` -- the bag
        plus every opponent's rack -- indexed by `LeaveNet.alphabet()`.

        Legitimate information: a player can see the board and their own rack.
        It is only *correct* because the board remembers which squares hold
        blanks, so a played blank counts against the `'?'` supply instead of
        against the letter it is standing in for.
        """

    def simulate(
        self,
        dawg: Dawg,
        net: LeaveNet,
        letters: str,
        candidates: int = 20,
        iterations: int = 200,
        plies: int = 1,
        batch: int = 25,
        root_eval_cap: int = 150,
        seed: int = 0,
    ) -> list[tuple[str, int, tuple[int, int, bool], list[str], float, float, int]]:
        """Rank this rack's plays by Monte-Carlo simulation.

        Returns `(word, score, (row, col, horizontal), used, equity, stderr,
        iterations)` per surviving candidate, best equity first.

        `equity` is the simulated score differential plus the difference in what
        each side is left holding, so it reads on the same points scale as a
        move's score -- but unlike `score` it accounts for what the move hands
        the opponent, which no function of (own score, own leave) can express.

        Candidates are seeded from the *whole* legal move list ranked by static
        equity, not the top-n by score. Every candidate within one iteration is
        rolled out against the same sampled tiles (common random numbers), and
        candidates whose confidence interval falls clear of the leader's stop
        being sampled -- so the nominal `candidates * iterations` cost is rarely
        paid in full.

        `plies` counts half-moves after the candidate, opponent first. Prefer
        odd values: they leave both sides having played equally often, so the
        differential compares like with like. Requires a GADDAG.

        `seed` of 0 picks one from the clock. Pass a real seed for a
        reproducible decision.
        """

    def exchange_letters(self, letters: str, letters_to_exchange: str) -> str:
        """Exchange letters from the player's hand with new letters from the bag.

        The `letters` argument represents the current letters in the player's hand.
        The `letters_to_exchange` argument specifies which letters the player wants to exchange.
        This method returns a new string of letters representing the player's hand after the exchange.
        """

    def can_exchange(self) -> bool:
        """Whether exchanging tiles for new ones is currently allowed.

        Standard Scrabble rule: exchanging is only permitted while at least 7
        tiles remain in the bag, regardless of how many tiles are exchanged.
        """

    @staticmethod
    def fresh_tile_bag() -> list[str]:
        """The standard Polish Scrabble tile distribution (100 tiles) that
        `Board()` and `Board.from_grid()` each start with.
        """

    @staticmethod
    def rack_value(letters: str) -> int:
        """Sum of face point values of the given rack (blanks score 0).

        Used for the standard end-of-game scoring adjustment: the player who
        goes out gains this value from each opponent's rack; everyone else
        loses it from their own.
        """

    @staticmethod
    def letter_points(letter: str) -> int:
        """Face point value of a single letter.

        `letter` must be exactly one character. Blanks ('?') score 0, the
        same fixed value they always score in-play (see
        `calculate_word_points`).
        """

    @staticmethod
    def first_draw_winner(draws: list[str]) -> int:
        """Index into `draws` of who goes first: each player draws one tile,
        closest to 'A' in alphabet order wins, a blank ('?') beats every
        letter, first index wins ties. Does not consume/mutate any bag —
        the caller returns the drawn tiles before dealing real racks.
        """


class LeaveNet:
    """The trained leave-value MLP, run natively by the engine.

    Simulation evaluates a leave at every rollout ply -- hundreds of thousands
    of times per move -- so crossing back into PyTorch for a 13k-parameter net
    would cost far more in FFI and GIL traffic than the ~10k multiply-adds it
    is asking for. `smart_player/export_weights.py` writes a checkpoint into
    the format loaded here; the `.pt` stays the source of truth.
    """

    def __init__(self, path: str) -> None:
        """Load an exported net. Raises `OSError` if the file is not a
        SCRBNET1 export, is truncated, or was trained against a different
        number of input features than the engine encodes.
        """

    def raw(self, x: list[float]) -> float:
        """Forward pass on an already-encoded feature vector.

        Exists so a test can check the engine's arithmetic against PyTorch's on
        identical input; ordinary callers want `value`.
        """

    def value(self, board: Board, leave: str, rack: str) -> float:
        """Predicted value of holding `leave` in `board`'s context, for a player
        whose full rack is `rack` (which fixes the unseen-tile count).
        """

    @staticmethod
    def alphabet() -> list[str]:
        """The tile-symbol order the feature vector's first 33 slots use.

        Must equal `smart_player.model.ALPHABET`; if it ever diverges, every
        prediction is silently wrong, so it is asserted in the tests.
        """
