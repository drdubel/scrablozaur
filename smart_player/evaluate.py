"""Benchmark SmartPlayer's win rate via simulate.play_game (mirrors
src/main.py's benchmark(), swapping in SmartPlayer for one side). Two modes:

    python smart_player/evaluate.py 2000                       # vs StrategicPlayer baseline
    python smart_player/evaluate.py 2000 --opponent simple      # vs SimplePlayer baseline
    python smart_player/evaluate.py 2000 --candidate PATH.pt    # champion vs a candidate checkpoint

The --candidate mode is what iterate.py uses to decide whether a freshly
trained checkpoint is actually better than the current champion before
promoting it.
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

from languages import engine_language, load as load_language  # noqa: E402
from model import DEFAULT_WEIGHTS_PATH, LANGUAGE  # noqa: E402
from player import SmartPlayer  # noqa: E402
from simulate import play_game  # noqa: E402

_spec = load_language(LANGUAGE)
_lang = engine_language(_spec)
_dawg = Dawg(_lang, str(_spec.dawg), str(_spec.gaddag))

_OPPONENTS = {"strategic": StrategicPlayer, "simple": SimplePlayer}


def _play_vs_baseline(opponent_name: str, model_path: str) -> tuple[int, int]:
    """Play one game, SmartPlayer(model_path) vs. the named baseline
    (random seat assignment so first-move advantage evens out). Returns
    (smart_score, opponent_score)."""
    board = Board(_lang)
    smart = SmartPlayer(board, model_path)
    other = _OPPONENTS[opponent_name](board)
    players = [smart, other] if random() < 0.5 else [other, smart]

    play_game(players, _dawg, parallel=False)

    smart_idx = 0 if players[0] is smart else 1
    return players[smart_idx].score, players[1 - smart_idx].score


def _play_candidate(champion_path: str, candidate_path: str) -> tuple[int, int]:
    """Play one game, candidate checkpoint vs. champion checkpoint (random
    seat assignment). Returns (candidate_score, champion_score)."""
    board = Board(_lang)
    champion = SmartPlayer(board, champion_path)
    candidate = SmartPlayer(board, candidate_path)
    players = [candidate, champion] if random() < 0.5 else [champion, candidate]

    play_game(players, _dawg, parallel=False)

    candidate_idx = 0 if players[0] is candidate else 1
    return players[candidate_idx].score, players[1 - candidate_idx].score


def run(
    n_games: int,
    opponent: str = "strategic",
    model_path: str = DEFAULT_WEIGHTS_PATH,
    candidate_path: str | None = None,
    n_workers: int | None = None,
    quiet: bool = False,
) -> dict[str, float]:
    """Play `n_games` and return {wins, losses, ties, win_rate, avg_self,
    avg_opponent} for the "subject" side (SmartPlayer, or the candidate
    checkpoint when `candidate_path` is given)."""
    wins = losses = ties = 0
    self_total = opp_total = 0
    label = "candidate vs champion" if candidate_path else f"SmartPlayer vs {opponent}"
    wall_start = time.perf_counter()

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            if candidate_path:
                futures = [executor.submit(_play_candidate, model_path, candidate_path) for _ in range(n_games)]
            else:
                futures = [executor.submit(_play_vs_baseline, opponent, model_path) for _ in range(n_games)]
            for future in tqdm(as_completed(futures), total=n_games, desc=label, disable=quiet):
                s, o = future.result()
                self_total += s
                opp_total += o
                if s > o:
                    wins += 1
                elif o > s:
                    losses += 1
                else:
                    ties += 1
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping early...")
        for future in futures:
            future.cancel()

    elapsed = time.perf_counter() - wall_start
    decisive = wins + losses
    win_rate = wins / decisive if decisive else float("nan")

    if not quiet:
        print(f"\n{n_games} games ({label}) in {elapsed:.1f}s")
        print(f"{wins}W {losses}L {ties}T  win rate {win_rate * 100:.1f}%")
        print(f"Avg score -- subject: {self_total / n_games:.1f}  other: {opp_total / n_games:.1f}")

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "avg_self": self_total / n_games,
        "avg_opponent": opp_total / n_games,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", type=int, nargs="?", default=1000, help="Number of games to play (default: 1000)")
    ap.add_argument("--opponent", choices=list(_OPPONENTS), default="strategic")
    ap.add_argument("--model-path", default=DEFAULT_WEIGHTS_PATH, help="Champion checkpoint to evaluate.")
    ap.add_argument("--candidate", default=None, help="If given, play this checkpoint against --model-path instead.")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    run(args.games, args.opponent, args.model_path, args.candidate, args.workers)
