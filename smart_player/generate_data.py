"""Self-play data generator for the leave-value model (see model.py).

Plays many self-play games to completion (via simulate.play_game) and
records every (leave, unseen_tiles) a player held during the game, labelled
with an n-step score-differential return: the change in (this player's
score - opponent's score) between the moment the leave was held and
`lookahead` of that player's own turns later (or the game's end, whichever
comes first). See README.md's "Credit assignment" section for why this
beats labelling every leave with the whole game's final margin.

By default self-play is StrategicPlayer vs StrategicPlayer (the fixed
greedy baseline). Pass `--player smart` to instead self-play with
SmartPlayer itself (optionally `--model-path` to pick which checkpoint) --
i.e. generate the *next* round of training data with the current best
learned player rather than the static baseline. That's the data-generation
half of policy iteration; see iterate.py for the full generate -> train ->
gate -> promote loop built on top of this.

    python smart_player/generate_data.py 200000 --lookahead 4
    python smart_player/generate_data.py 200000 --player smart
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm  # type: ignore

from scrablozaur import Board, Dawg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strategy import StrategicPlayer  # noqa: E402

from board_features import encode_board  # noqa: E402
from model import DEFAULT_WEIGHTS_PATH  # noqa: E402
from player import SmartPlayer  # noqa: E402
from simulate import play_game  # noqa: E402

_dawg = Dawg(os.path.join(os.path.dirname(__file__), "..", "words", "dawg.bin"))

# (leave, unseen_tiles, board_features, score - opponent.score at that moment).
_LogEntry = tuple[str, int, tuple[float, ...], int]


class _RecordingMixin:
    """Instruments any StrategicPlayer subclass to capture the leave -- the
    rack right after removing a move's tiles but before refilling from the
    bag, which is exactly when a leave's value is realized -- every time it
    actually plays a word, along with the board state and the score
    differential at that moment (needed to compute an n-step return
    afterwards). `opponent` is wired up by `_play_one_game` once both
    players exist.

    Board features are captured from get_best_word() -- i.e. the board
    *before* this move is placed -- not from draw_letters() where self.board
    already reflects the move. That matters: SmartPlayer.evaluate_word (see
    player.py) ranks candidates using the pre-move board, since the real
    board is identical across all candidates in one decision and only their
    hypothetical placements differ. Recording post-move board state here
    would train on a systematically different (more-filled) distribution
    than what inference ever sees, so this mirrors SmartPlayer's own
    caching independently rather than relying on it (StrategicPlayer, the
    default self-play baseline, has no such caching of its own)."""

    opponent: "_RecordingMixin"

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._dealt = False
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]
        self.leave_log: list[_LogEntry] = []

    def get_best_word(self, dawg: Dawg, parallel: bool) -> tuple[str, int, tuple[int, int, bool], list[str]]:
        self._pre_move_board_features = encode_board(self.board)  # type: ignore[attr-defined]
        return super().get_best_word(dawg, parallel)  # type: ignore[misc]

    def draw_letters(self) -> None:
        if self._dealt:
            diff = self.score - self.opponent.score  # type: ignore[attr-defined]
            unseen = len(self.get_letters_left())  # type: ignore[attr-defined]
            self.leave_log.append((self.letters, unseen, self._pre_move_board_features, diff))  # type: ignore[attr-defined]
        super().draw_letters()  # type: ignore[misc]
        self._dealt = True


class _RecordingStrategic(_RecordingMixin, StrategicPlayer):
    pass


class _RecordingSmart(_RecordingMixin, SmartPlayer):
    pass


_PLAYER_CLASSES = {"strategic": _RecordingStrategic, "smart": _RecordingSmart}


# (leave, unseen_tiles, board_features, target).
_Sample = tuple[str, int, tuple[float, ...], int]


def _n_step_returns(log: list[_LogEntry], final_diff: int, lookahead: int) -> list[_Sample]:
    """Turn a player's chronological leave_log into (leave, unseen,
    board_features, target) samples, where target is the change in score
    differential between each leave and `lookahead` of that player's own
    turns later -- or the game's actual final differential, for leaves
    within `lookahead` turns of the end (there's nothing further to look
    ahead to). `lookahead <= 0` means unbounded: every leave is credited
    with the final differential, exactly like the original whole-game-
    margin approach."""
    n = len(log)
    samples = []
    for i, (leave, unseen, board_features, diff_now) in enumerate(log):
        j = i + lookahead
        diff_future = log[j][3] if 0 < lookahead and j < n else final_diff
        samples.append((leave, unseen, board_features, diff_future - diff_now))
    return samples


def _play_one_game(parallel: bool, lookahead: int, player: str, model_path: str) -> tuple[list[_Sample], list[_Sample]]:
    """Play one self-play game to completion and return each player's
    (leave, unseen_tiles, board_features, n_step_return) samples."""
    board = Board()
    cls = _PLAYER_CLASSES[player]
    p1 = cls(board, model_path) if player == "smart" else cls(board)
    p2 = cls(board, model_path) if player == "smart" else cls(board)
    p1.opponent, p2.opponent = p2, p1
    play_game([p1, p2], _dawg, parallel=parallel)

    final_diff = p1.score - p2.score
    samples1 = _n_step_returns(p1.leave_log, final_diff, lookahead)
    samples2 = _n_step_returns(p2.leave_log, -final_diff, lookahead)
    return samples1, samples2


def generate(
    n_games: int,
    out_path: str,
    lookahead: int,
    player: str = "strategic",
    model_path: str = DEFAULT_WEIGHTS_PATH,
    n_workers: int | None = None,
    quiet: bool = False,
) -> None:
    all_leaves: list[str] = []
    all_unseen: list[int] = []
    all_board_features: list[tuple[float, ...]] = []
    all_targets: list[int] = []

    # Mirrors src/main.py's benchmark(): only let the Rust side parallelize
    # a single move's search when there's just one Python worker process --
    # otherwise the two parallelism strategies fight over the same cores.
    parallel = n_workers == 1
    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        n_workers = executor._max_workers  # type: ignore
        batch = n_workers * 50
        with tqdm(total=n_games, desc="Self-play games", disable=quiet) as pbar:
            for i in range(0, n_games, batch):
                futures = [
                    executor.submit(_play_one_game, parallel, lookahead, player, model_path)
                    for _ in range(min(batch, n_games - i))
                ]
                for future in as_completed(futures):
                    for samples in future.result():
                        for leave, unseen, board_features, target in samples:
                            all_leaves.append(leave)
                            all_unseen.append(unseen)
                            all_board_features.append(board_features)
                            all_targets.append(target)
                    pbar.update(1)

    elapsed = time.perf_counter() - wall_start
    if not quiet:
        print(f"{n_games} games -> {len(all_leaves)} samples in {elapsed:.1f}s")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    board_arr = np.array(all_board_features, dtype=np.float32)
    np.savez(
        out_path,
        leaves=np.array(all_leaves),
        unseen=np.array(all_unseen, dtype=np.int32),
        margins=np.array(all_targets, dtype=np.int32),
        tw_open=board_arr[:, 0],
        dw_open=board_arr[:, 1],
        tl_open=board_arr[:, 2],
        dl_open=board_arr[:, 3],
        board_fill=board_arr[:, 4],
    )
    if not quiet:
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", type=int, nargs="?", default=20000, help="Number of self-play games (default: 20000)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "_leave_dataset.npz"))
    ap.add_argument(
        "--lookahead",
        type=int,
        default=4,
        help="Credit each leave with the score-differential swing over this many of the "
        "player's own future turns (default: 4). 0 means unbounded (whole-game margin).",
    )
    ap.add_argument(
        "--player",
        choices=list(_PLAYER_CLASSES),
        default="strategic",
        help="Self-play with the fixed StrategicPlayer baseline, or with SmartPlayer itself "
        "(policy iteration -- see iterate.py). Default: strategic.",
    )
    ap.add_argument("--model-path", default=DEFAULT_WEIGHTS_PATH, help="Checkpoint to use when --player smart.")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    generate(args.games, args.out, args.lookahead, args.player, args.model_path, args.workers)
