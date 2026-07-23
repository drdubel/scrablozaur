"""Self-play data generator for the leave-value model (see model.py).

Plays many StrategicPlayer-vs-StrategicPlayer games to completion (via
simulate.play_game) and records every (leave, unseen_tiles) a player held
during the game, labelled with an n-step score-differential return: the
change in (this player's score - opponent's score) between the moment the
leave was held and `lookahead` of that player's own turns later (or the
game's end, whichever comes first).

This is a bounded-horizon alternative to the simpler "label every leave
with the final game margin" approach (still available via
`--lookahead 0`, treated as unbounded): a single leave is only one of
~15-20 decisions in a game, so crediting it with the *whole* game's
outcome buries its effect in a lot of irrelevant late-game variance (see
smart_player/README.md's "Current status" section for the measured
impact -- 50k games of the unbounded version only explained ~5.6% of
final-margin variance). Truncating the return window is the standard
n-step-return way to trade some of that noise for bias, without needing
full TD bootstrapping off the (still-training) value network itself.

    python smart_player/generate_data.py 200000 --lookahead 4
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

from simulate import play_game  # noqa: E402

_dawg = Dawg(os.path.join(os.path.dirname(__file__), "..", "words", "dawg.bin"))


class _RecordingPlayer(StrategicPlayer):
    """StrategicPlayer instrumented to capture the leave -- the rack right
    after removing a move's tiles but before refilling from the bag, which
    is exactly when a leave's value is realized -- every time it actually
    plays a word, along with the score differential at that moment (needed
    to compute an n-step return afterwards). `opponent` is wired up by
    `_play_one_game` once both players exist."""

    opponent: "_RecordingPlayer"

    def __init__(self, board: Board) -> None:
        self._dealt = False
        super().__init__(board)
        # (leave, unseen_tiles, score - opponent.score at that moment).
        self.leave_log: list[tuple[str, int, int]] = []

    def draw_letters(self) -> None:
        if self._dealt:
            diff = self.score - self.opponent.score
            self.leave_log.append((self.letters, len(self.get_letters_left()), diff))
        super().draw_letters()
        self._dealt = True


def _n_step_returns(log: list[tuple[str, int, int]], final_diff: int, lookahead: int) -> list[tuple[str, int, int]]:
    """Turn a player's chronological leave_log into (leave, unseen, target)
    samples, where target is the change in score differential between each
    leave and `lookahead` of that player's own turns later -- or the game's
    actual final differential, for leaves within `lookahead` turns of the
    end (there's nothing further to look ahead to). `lookahead <= 0` means
    unbounded: every leave is credited with the final differential, exactly
    like the original whole-game-margin approach."""
    n = len(log)
    samples = []
    for i, (leave, unseen, diff_now) in enumerate(log):
        j = i + lookahead
        diff_future = log[j][2] if 0 < lookahead and j < n else final_diff
        samples.append((leave, unseen, diff_future - diff_now))
    return samples


def _play_one_game(parallel: bool, lookahead: int) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """Play one self-play game to completion and return each player's
    (leave, unseen_tiles, n_step_return) samples."""
    board = Board()
    p1, p2 = _RecordingPlayer(board), _RecordingPlayer(board)
    p1.opponent, p2.opponent = p2, p1
    play_game([p1, p2], _dawg, parallel=parallel)

    final_diff = p1.score - p2.score
    samples1 = _n_step_returns(p1.leave_log, final_diff, lookahead)
    samples2 = _n_step_returns(p2.leave_log, -final_diff, lookahead)
    return samples1, samples2


def generate(n_games: int, out_path: str, lookahead: int, n_workers: int | None = None) -> None:
    all_leaves: list[str] = []
    all_unseen: list[int] = []
    all_targets: list[int] = []

    # Mirrors src/main.py's benchmark(): only let the Rust side parallelize
    # a single move's search when there's just one Python worker process --
    # otherwise the two parallelism strategies fight over the same cores.
    parallel = n_workers == 1
    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        n_workers = executor._max_workers  # type: ignore
        batch = n_workers * 50
        with tqdm(total=n_games, desc="Self-play games") as pbar:
            for i in range(0, n_games, batch):
                futures = [executor.submit(_play_one_game, parallel, lookahead) for _ in range(min(batch, n_games - i))]
                for future in as_completed(futures):
                    for samples in future.result():
                        for leave, unseen, target in samples:
                            all_leaves.append(leave)
                            all_unseen.append(unseen)
                            all_targets.append(target)
                    pbar.update(1)

    elapsed = time.perf_counter() - wall_start
    print(f"{n_games} games -> {len(all_leaves)} samples in {elapsed:.1f}s")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        leaves=np.array(all_leaves),
        unseen=np.array(all_unseen, dtype=np.int32),
        margins=np.array(all_targets, dtype=np.int32),
    )
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
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    generate(args.games, args.out, args.lookahead, args.workers)
