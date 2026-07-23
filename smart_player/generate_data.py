"""Self-play data generator for the leave-value model (see model.py).

Plays many StrategicPlayer-vs-StrategicPlayer games to completion (via
simulate.play_game) and records every (leave, unseen_tiles) a player held
during the game, labelled with that player's final score margin. This is
how classical Scrabble engines' leave tables are built (e.g. Maven's),
just automated end-to-end instead of hand-computed.

    python smart_player/generate_data.py 20000
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
    plays a word."""

    def __init__(self, board: Board) -> None:
        self._dealt = False
        super().__init__(board)
        self.leave_log: list[tuple[str, int]] = []

    def draw_letters(self) -> None:
        if self._dealt:
            self.leave_log.append((self.letters, len(self.get_letters_left())))
        super().draw_letters()
        self._dealt = True


def _play_one_game(parallel: bool) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """Play one self-play game to completion and return each player's
    (leave, unseen_tiles, final_margin) samples."""
    board = Board()
    players = [_RecordingPlayer(board), _RecordingPlayer(board)]
    play_game(players, _dawg, parallel=parallel)

    p1, p2 = players
    margin = p1.score - p2.score
    samples1 = [(leave, unseen, margin) for leave, unseen in p1.leave_log]
    samples2 = [(leave, unseen, -margin) for leave, unseen in p2.leave_log]
    return samples1, samples2


def generate(n_games: int, out_path: str, n_workers: int | None = None) -> None:
    all_leaves: list[str] = []
    all_unseen: list[int] = []
    all_margins: list[int] = []

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
                futures = [executor.submit(_play_one_game, parallel) for _ in range(min(batch, n_games - i))]
                for future in as_completed(futures):
                    for samples in future.result():
                        for leave, unseen, margin in samples:
                            all_leaves.append(leave)
                            all_unseen.append(unseen)
                            all_margins.append(margin)
                    pbar.update(1)

    elapsed = time.perf_counter() - wall_start
    print(f"{n_games} games -> {len(all_leaves)} samples in {elapsed:.1f}s")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        leaves=np.array(all_leaves),
        unseen=np.array(all_unseen, dtype=np.int32),
        margins=np.array(all_margins, dtype=np.int32),
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", type=int, nargs="?", default=20000, help="Number of self-play games (default: 20000)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "_leave_dataset.npz"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    generate(args.games, args.out, args.workers)
