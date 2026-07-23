"""Benchmark SmartPlayer's win rate against the existing StrategicPlayer and
SimplePlayer baselines by playing many games via simulate.play_game (mirrors
src/main.py's benchmark(), swapping in SmartPlayer for one side).

    python smart_player/evaluate.py 2000
    python smart_player/evaluate.py 2000 --opponent simple
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import random

from tqdm import tqdm  # type: ignore

from scrablozaur import Board, Dawg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strategy import SimplePlayer, StrategicPlayer  # noqa: E402

from player import SmartPlayer  # noqa: E402
from simulate import play_game  # noqa: E402

_dawg = Dawg(os.path.join(os.path.dirname(__file__), "..", "words", "dawg.bin"))

_OPPONENTS = {"strategic": StrategicPlayer, "simple": SimplePlayer}


def _play(opponent_name: str) -> tuple[int, int]:
    """Play one game, SmartPlayer vs. the named opponent (random seat
    assignment so first-move advantage evens out). Returns (smart_score,
    opponent_score)."""
    board = Board()
    smart = SmartPlayer(board)
    other = _OPPONENTS[opponent_name](board)
    players = [smart, other] if random() < 0.5 else [other, smart]

    play_game(players, _dawg, parallel=False)

    smart_idx = 0 if players[0] is smart else 1
    return players[smart_idx].score, players[1 - smart_idx].score


def run(n_games: int, opponent: str, n_workers: int | None = None) -> None:
    wins = losses = ties = 0
    smart_total = opp_total = 0
    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_play, opponent) for _ in range(n_games)]
        for future in tqdm(as_completed(futures), total=n_games, desc=f"SmartPlayer vs {opponent}"):
            s, o = future.result()
            smart_total += s
            opp_total += o
            if s > o:
                wins += 1
            elif o > s:
                losses += 1
            else:
                ties += 1

    elapsed = time.perf_counter() - wall_start
    decisive = wins + losses
    win_rate = wins / decisive * 100 if decisive else float("nan")

    print(f"\n{n_games} games vs {opponent} in {elapsed:.1f}s")
    print(f"SmartPlayer: {wins}W {losses}L {ties}T  win rate {win_rate:.1f}%")
    print(f"Avg score -- SmartPlayer: {smart_total / n_games:.1f}  {opponent}: {opp_total / n_games:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", type=int, nargs="?", default=1000, help="Number of games to play (default: 1000)")
    ap.add_argument("--opponent", choices=list(_OPPONENTS), default="strategic")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    run(args.games, args.opponent, args.workers)
