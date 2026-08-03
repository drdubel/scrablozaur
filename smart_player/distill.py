"""Distil the simulator into the leave-value model.

    python smart_player/distill.py 5000 --iterations 100 --candidates 12
    python smart_player/train.py --data smart_player/_distill.npz \\
        --out smart_player/models/pl/leave_v3.pt --warm-start smart_player/models/pl/leave_value.pt

Why this and not more self-play. `generate_data.py` labels a leave with an
n-step score-differential return, and that target is mostly noise: measured on
fresh data, position explains 1.58% of its variance and the leave 2.66%, against
a total variance of ~2223. The trained net tops out around R^2 = 6.75%, and
every downstream gain -- simulation, the endgame search, the leave weight --
runs into that ceiling. Ten times the games moved it by ~3 points; it is the
label, not the volume.

A simulation equity is the same quantity measured far more precisely: standard
error ~1.8 points at 100 iterations, against the n-step return's standard
deviation of 47.5. That is roughly **25x the signal-to-noise per sample**, which
is what actually raises the ceiling.

Target is `sim_equity - raw_score`. The move's own score needs no learning --
the engine reports it exactly -- so subtracting it leaves precisely the part the
model is for: what the position and the leave are worth after the move.

Two things worth knowing:

- **Every simulated candidate is kept, not just the chosen one.** `Board.simulate`
  already returns them all with equity and standard error, so one call yields
  ~12 labels instead of 1. `--batch` is pinned to `--iterations` so the
  simulator skips its final pruning pass and reports every candidate rather than
  only the survivors, which would bias the set towards near-optimal leaves.
- **The labels bootstrap.** A rollout's leaf is evaluated with the *current*
  net, so the target partly reflects what the model already believes. That can
  drift as easily as improve, which is why a candidate is promoted on a measured
  arena win and never on a lower training loss.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm  # type: ignore

from scrablozaur import Board, Dawg, set_num_threads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strategy import StrategicPlayer  # noqa: E402

from board_features import encode_board  # noqa: E402
from languages import engine_language, load as load_language  # noqa: E402
from model import DEFAULT_WEIGHTS_PATH, LANGUAGE  # noqa: E402
from player import remove_used  # noqa: E402
from sim_player import get_net  # noqa: E402
from simulate import play_game  # noqa: E402

_spec = load_language(LANGUAGE)
_lang = engine_language(_spec)
_dawg = Dawg(_lang, str(_spec.dawg), str(_spec.gaddag))

# (leave, unseen, board_features, target, decision_id)
_Sample = tuple[str, int, tuple[float, ...], float, int]

# Namespace stride for per-seat decision ids. A game never reaches this many
# turns for one player; the cap on moves per game is far below it.
_MAX_DECISIONS_PER_PLAYER = 1000


class _LabellingPlayer(StrategicPlayer):
    """Plays the simulator's choice, and keeps every candidate it scored.

    Subclasses `StrategicPlayer` rather than `SimPlayer` because the exchange
    logic is irrelevant here and the goal is a labelled candidate set, not the
    strongest possible play.
    """

    def __init__(self, board: Board, model_path: str, iterations: int, candidates: int, plies: int, seed: int):
        super().__init__(board)
        self.net = get_net(model_path)
        self.iterations = iterations
        self.candidates = candidates
        self.plies = plies
        self.seed = seed
        self._decision = 0
        self.samples: list[_Sample] = []

    def get_best_word(self, dawg: Dawg, parallel: bool):
        # Captured before the move, matching what the model sees at inference.
        board_features = encode_board(self.board)
        unseen = sum(self.board.unseen_tile_counts(self.letters))
        self._decision += 1

        results = self.board.simulate(
            dawg,
            self.net,
            self.letters,
            candidates=self.candidates,
            iterations=self.iterations,
            plies=self.plies,
            # batch == iterations: one pass, no pruning, every candidate scored.
            batch=self.iterations,
            seed=(self.seed * 0x9E3779B1 + self._decision) & 0xFFFFFFFFFFFFFFFF,
        )
        if not results:
            return ("", 0, (0, 0, True), [])

        for word, score, _pos, used, equity, _stderr, n in results:
            if n == 0:
                continue  # never actually simulated (single-candidate decision)
            leave = remove_used(self.letters, used)
            # `self._decision` groups the candidates that were competing against
            # each other. Only differences *within* a group can change which
            # move gets picked, so training needs to be able to isolate them --
            # see train.py's --center-by-decision.
            self.samples.append((leave, unseen, board_features, equity - score, self._decision))

        word, score, pos, used, _eq, _se, _n = results[0]
        return (word, score, pos, used)


def _play_one_game(args: tuple[str, int, int, int, int]) -> list[_Sample]:
    model_path, iterations, candidates, plies, seed = args
    board = Board.seeded(_lang, seed)
    players = [
        _LabellingPlayer(board, model_path, iterations, candidates, plies, seed * 2 + i)
        for i in range(2)
    ]
    play_game(players, _dawg, parallel=False)
    # Both players number their decisions from 1, so tag with the seat before
    # merging or the two would collide and two unrelated candidate sets would
    # be treated as one decision.
    return [
        (leave, unseen, feats, target, seat * _MAX_DECISIONS_PER_PLAYER + local)
        for seat, p in enumerate(players)
        for (leave, unseen, feats, target, local) in p.samples
    ]


def _init_worker() -> None:
    """One engine thread per worker -- the pool already fans out across cores,
    and `Board.simulate` spinning up its own 8 threads inside each of them turns
    a one-minute run into ten of thrashing."""
    set_num_threads(1)


def generate(
    n_games: int,
    out_path: str,
    model_path: str = DEFAULT_WEIGHTS_PATH,
    iterations: int = 100,
    candidates: int = 12,
    plies: int = 1,
    seed0: int = 0,
    n_workers: int | None = None,
    quiet: bool = False,
) -> int:
    leaves: list[str] = []
    unseen: list[int] = []
    feats: list[tuple[float, ...]] = []
    targets: list[float] = []
    decisions: list[int] = []
    next_decision = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
        jobs = [
            (model_path, iterations, candidates, plies, seed0 + i) for i in range(n_games)
        ]
        futures = [executor.submit(_play_one_game, j) for j in jobs]
        for future in tqdm(as_completed(futures), total=n_games, desc="Sim-labelled games", disable=quiet):
            # Decision ids are per-player-per-game, so offset them into a global
            # namespace as each game's results arrive.
            local_to_global: dict[int, int] = {}
            for leave, u, f, target, local_id in future.result():
                if local_id not in local_to_global:
                    local_to_global[local_id] = next_decision
                    next_decision += 1
                leaves.append(leave)
                unseen.append(u)
                feats.append(f)
                targets.append(target)
                decisions.append(local_to_global[local_id])

    elapsed = time.perf_counter() - t0
    if not quiet:
        print(f"{n_games} games -> {len(leaves)} labels in {elapsed:.1f}s "
              f"({len(leaves) / max(n_games, 1):.1f} per game)")
        if targets:
            arr = np.asarray(targets)
            print(f"target: mean {arr.mean():+.2f}  std {arr.std():.2f}  "
                  f"range [{arr.min():+.1f}, {arr.max():+.1f}]")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    board_arr = np.array(feats, dtype=np.float32)
    # Same column layout generate_data.py writes, so train.py reads either.
    np.savez(
        out_path,
        leaves=np.array(leaves),
        unseen=np.array(unseen, dtype=np.int32),
        margins=np.array(targets, dtype=np.float32),
        tw_open=board_arr[:, 0],
        dw_open=board_arr[:, 1],
        tl_open=board_arr[:, 2],
        dl_open=board_arr[:, 3],
        board_fill=board_arr[:, 4],
        decision=np.array(decisions, dtype=np.int32),
    )
    if not quiet:
        print(f"Saved to {out_path}")
    return len(leaves)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", type=int, nargs="?", default=5000, help="Self-play games to label (default: 5000)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "_distill.npz"))
    ap.add_argument("--model-path", default=DEFAULT_WEIGHTS_PATH, help="Checkpoint the rollouts evaluate with.")
    ap.add_argument("--iterations", type=int, default=100,
                    help="Rollouts per candidate. 100 gives a label std error of ~1.8 points, "
                         "still ~25x better than the n-step return (default: 100).")
    ap.add_argument("--candidates", type=int, default=12, help="Candidates scored per decision (default: 12)")
    ap.add_argument("--plies", type=int, default=1, help="Half-moves simulated after the candidate (default: 1)")
    ap.add_argument("--seed", type=int, default=0, help="Base seed, so a run is reproducible (default: 0)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    generate(args.games, args.out, args.model_path, args.iterations,
             args.candidates, args.plies, args.seed, args.workers)
