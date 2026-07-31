"""SimPlayer: picks its move by Monte-Carlo simulation instead of static
evaluation.

`SmartPlayer` ranks candidates by `score + leave_value`. That is a good static
equity function, but it is structurally blind to the one thing that decides
close games: what the move hands the opponent. A play that scores four more
points while opening the triple-word lane is a bad play, and no function of
(my score, my leave) can tell you so.

`SimPlayer` asks the question directly. For each leading candidate it plays the
move, deals the opponent a plausible rack from the tiles neither on the board
nor in its own rack, lets both sides reply with the static player, and scores
the resulting position. The engine does the whole rollout natively
(`Board.simulate`) -- see src/lib.rs for the common-random-numbers and pruning
details, and `export_weights.py` for how the leave net gets there.

The exchange decision is deliberately left as `SmartPlayer` computes it,
statically: a simulated equity and a static leave value are not on the same
scale, and mixing them to decide play-vs-exchange would compare two different
quantities. Simulation only chooses *which word*.
"""

import os
import sys
from typing import NamedTuple

from scrablozaur import Board, Dawg, LeaveNet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export_weights import DEFAULT_BIN_PATH, export  # noqa: E402
from languages import engine_language  # noqa: E402
from model import DEFAULT_WEIGHTS_PATH, LANGUAGE  # noqa: E402
from player import DEFAULT_ENDGAME_NODES, SmartPlayer, choose_move  # noqa: E402

DEFAULT_ITERATIONS = 200
DEFAULT_CANDIDATES = 20
# Kept separate from `DEFAULT_LEAVE_WEIGHT` because the two are not the same
# quantity: the sim's leaf already contains a *realised* score differential, so
# its leave term corrects a different thing than the static player's does. They
# happen to agree at 1.0 for the current checkpoint. Retune independently if
# either the checkpoint or the rollout depth changes.
SIM_LEAVE_WEIGHT = 1.0
# One opponent reply, so both sides have played the same number of moves when
# the position is scored. Measured against the alternatives (see
# smart_player/README.md): the balanced 2v2 window (plies=3) is worth the same
# within error at more than twice the cost, and the unbalanced 2v1 window
# (plies=2) is worse -- it hands us an extra scoring turn the opponent never
# gets, which rewards setting up our own follow-up over noticing what the
# candidate concedes.
DEFAULT_PLIES = 1

_nets: dict[str, LeaveNet] = {}


class SimChoice(NamedTuple):
    """A simulated play. First four fields match the engine's move tuple.

    `equity` is on the same points scale as a move's score but accounts for
    what the move concedes; `stderr` says how well the simulation resolved it.
    Both are `None` when the move came from the endgame search instead, which
    reports a proven differential rather than a sampled estimate.
    """

    word: str
    score: int
    position: tuple[int, int, bool]
    used: list[str]
    equity: float | None = None
    stderr: float | None = None
    endgame_diff: int | None = None
    endgame_exact: bool | None = None


def choose_move_sim(
    board: Board,
    dawg: Dawg,
    rack: str,
    bag_remaining: int,
    *,
    model_path: str = DEFAULT_WEIGHTS_PATH,
    iterations: int = DEFAULT_ITERATIONS,
    candidates: int = DEFAULT_CANDIDATES,
    plies: int = DEFAULT_PLIES,
    leave_weight: float = SIM_LEAVE_WEIGHT,
    use_endgame: bool = True,
    endgame_max_nodes: int = DEFAULT_ENDGAME_NODES,
    seed: int = 0,
) -> SimChoice:
    """Pick a move by simulation, as a function of the position.

    Shared by `SimPlayer` and the web app so there is one implementation. See
    `player.choose_move` for why `bag_remaining` is a parameter rather than
    read off the board.
    """
    # Nothing left to sample once the bag is empty -- search it instead.
    if use_endgame and bag_remaining == 0:
        endgame = choose_move(
            board,
            dawg,
            rack,
            bag_remaining,
            model_path=model_path,
            leave_weight=leave_weight,
            use_endgame=True,
            endgame_max_nodes=endgame_max_nodes,
        )
        return SimChoice(
            endgame.word,
            endgame.score,
            endgame.position,
            endgame.used,
            endgame_diff=endgame.endgame_diff,
            endgame_exact=endgame.endgame_exact,
        )

    results = board.simulate(
        dawg,
        get_net(model_path),
        rack,
        candidates=candidates,
        iterations=iterations,
        plies=plies,
        leave_weight=leave_weight,
        seed=seed,
    )
    if not results:
        return SimChoice("", 0, (0, 0, True), [])
    word, score, position, used, equity, stderr, _n = results[0]
    return SimChoice(word, score, position, used, equity, stderr)


def get_net(pt_path: str = DEFAULT_WEIGHTS_PATH, bin_path: str | None = None) -> LeaveNet:
    """Load the engine-side net for a checkpoint, exporting it first if the
    binary is missing or older than the `.pt` it came from.

    The `.pt` stays the source of truth, so a freshly trained checkpoint can
    never be silently simulated with the previous run's weights.
    """
    bin_path = bin_path or (
        DEFAULT_BIN_PATH if pt_path == DEFAULT_WEIGHTS_PATH else os.path.splitext(pt_path)[0] + ".bin"
    )
    if bin_path not in _nets:
        stale = not os.path.exists(bin_path) or os.path.getmtime(bin_path) < os.path.getmtime(pt_path)
        if stale:
            export(pt_path, bin_path)
        _nets[bin_path] = LeaveNet(engine_language(LANGUAGE), bin_path)
    return _nets[bin_path]


class SimPlayer(SmartPlayer):
    """SmartPlayer that chooses its word by simulation.

    `seed` makes a game reproducible: each decision mixes it with the move
    number, so the same position in a replayed game samples the same tiles
    without every position in the game sampling identically.
    """

    def __init__(
        self,
        board: Board,
        model_path: str = DEFAULT_WEIGHTS_PATH,
        iterations: int = DEFAULT_ITERATIONS,
        candidates: int = DEFAULT_CANDIDATES,
        plies: int = DEFAULT_PLIES,
        leave_weight: float = SIM_LEAVE_WEIGHT,
        use_endgame: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(board, model_path, leave_weight, use_endgame)
        self.net = get_net(model_path)
        self.iterations = iterations
        self.candidates = candidates
        self.plies = plies
        self.seed = seed or 0x5EED
        self._decision = 0
        self.last_equity: float | None = None
        self.last_stderr: float | None = None

    def get_best_word(
        self, dawg: Dawg, parallel: bool
    ) -> tuple[str, int, tuple[int, int, bool], list[str]]:
        # SmartPlayer.play_word reads these back when it compares playing
        # against exchanging, so they have to be set even though the simulation
        # computes its own copies engine-side.
        self._cache_turn_context()
        self._decision += 1

        choice = choose_move_sim(
            self.board,
            dawg,
            self.letters,
            self.board.bag_remaining(),
            model_path=self.model_path,
            iterations=self.iterations,
            candidates=self.candidates,
            plies=self.plies,
            leave_weight=self.leave_weight,
            use_endgame=self.use_endgame,
            # Vary per decision, but reproducibly for a given player seed.
            seed=(self.seed * 0x9E3779B1 + self._decision) & 0xFFFFFFFFFFFFFFFF,
        )
        self.last_equity, self.last_stderr = choice.equity, choice.stderr
        self.last_endgame_diff = choice.endgame_diff
        self.last_endgame_exact = choice.endgame_exact
        return (choice.word, choice.score, choice.position, choice.used)
