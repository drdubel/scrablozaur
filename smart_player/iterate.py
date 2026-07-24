"""Policy iteration: repeatedly generate self-play data with the current
best SmartPlayer, train a candidate on it, and promote the candidate to
champion only if it actually beats the current champion.

Each round:
  1. generate_data.generate(..., player="smart", model_path=champion) --
     self-play with the CURRENT champion instead of the fixed
     StrategicPlayer baseline generate_data.py otherwise defaults to. This
     is what makes it policy iteration rather than one static regression:
     the data-generating policy improves round over round.
  2. train.train(...) a candidate checkpoint on that round's data.
  3. evaluate.run(..., candidate=...) the candidate against the current
     champion over --eval-games games.
  4. Promote (overwrite the champion checkpoint) only if the candidate's
     win rate clears --promote-threshold -- champion/challenger gating,
     the same guard AlphaZero-style self-play uses to stop a single bad
     training round from silently regressing the deployed model.

Each round's scratch dataset is deleted after training (they're ~150MB+ at
200k games); only the round's metrics and, if promoted, the checkpoint
persist.

    python smart_player/iterate.py --rounds 3 --games 100000
"""

import argparse
import os
import shutil
import time

from evaluate import run as evaluate_run
from generate_data import generate
from model import DEFAULT_WEIGHTS_PATH
from train import train as train_model

_SCRATCH_DIR = os.path.dirname(__file__)


def iterate(
    rounds: int,
    games: int,
    lookahead: int = 4,
    eval_games: int = 3000,
    promote_threshold: float = 0.52,
    epochs: int = 30,
    hidden1: int = 128,
    hidden2: int = 64,
    n_workers: int | None = None,
) -> None:
    champion_path = DEFAULT_WEIGHTS_PATH
    candidate_path = os.path.join(_SCRATCH_DIR, "models", "_candidate.pt")
    history: list[dict[str, object]] = []

    for r in range(1, rounds + 1):
        round_start = time.perf_counter()
        print(f"\n=== Round {r}/{rounds} (champion: {champion_path}) ===")

        dataset_path = os.path.join(_SCRATCH_DIR, f"_iter_round{r}.npz")
        print(f"[{r}] generating {games} self-play games with the current champion...")
        generate(
            games,
            dataset_path,
            lookahead,
            player="smart",
            model_path=champion_path,
            n_workers=n_workers,
            quiet=True,
        )

        print(f"[{r}] training candidate...")
        val_mse = train_model(dataset_path, candidate_path, epochs=epochs, hidden1=hidden1, hidden2=hidden2, quiet=True)
        os.remove(dataset_path)

        print(f"[{r}] evaluating candidate vs champion over {eval_games} games...")
        result = evaluate_run(
            eval_games,
            model_path=champion_path,
            candidate_path=candidate_path,
            n_workers=n_workers,
            quiet=True,
        )

        promoted = result["win_rate"] >= promote_threshold
        if promoted:
            shutil.copyfile(candidate_path, champion_path)

        elapsed = time.perf_counter() - round_start
        status = "PROMOTED" if promoted else "rejected"
        print(
            f"[{r}] candidate: {result['wins']:.0f}W {result['losses']:.0f}L {result['ties']:.0f}T "
            f"win rate {result['win_rate'] * 100:.1f}% (avg {result['avg_self']:.1f} vs "
            f"{result['avg_opponent']:.1f})  val mse {val_mse:.1f}  -> {status}  ({elapsed / 60:.1f} min)"
        )
        history.append({"round": r, "val_mse": val_mse, **result, "promoted": promoted})

    if history and not history[-1]["promoted"] and os.path.exists(candidate_path):
        os.remove(candidate_path)

    print("\n=== Summary ===")
    for h in history:
        status = "PROMOTED" if h["promoted"] else "rejected"
        print(f"round {h['round']}: win rate {h['win_rate'] * 100:.1f}%  val mse {h['val_mse']:.1f}  {status}")
    n_promoted = sum(1 for h in history if h["promoted"])
    print(f"{n_promoted}/{rounds} rounds promoted. Final champion: {champion_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3, help="Number of generate -> train -> gate rounds (default: 3)")
    ap.add_argument("--games", type=int, default=100000, help="Self-play games per round (default: 100000)")
    ap.add_argument("--lookahead", type=int, default=4)
    ap.add_argument("--eval-games", type=int, default=3000, help="Games to gate each candidate over (default: 3000)")
    ap.add_argument(
        "--promote-threshold",
        type=float,
        default=0.52,
        help="Minimum candidate win rate to replace the champion (default: 0.52)",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden1", type=int, default=128)
    ap.add_argument("--hidden2", type=int, default=64)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    iterate(
        args.rounds,
        args.games,
        args.lookahead,
        args.eval_games,
        args.promote_threshold,
        args.epochs,
        args.hidden1,
        args.hidden2,
        args.workers,
    )
