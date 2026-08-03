"""Paired-seed benchmark: play the same bag twice with the seats swapped.

    python smart_player/arena.py --a smart --b strategic --pairs 1500
    python smart_player/arena.py --a smart:models/pl/cand.pt --b smart --pairs 1500

Why pairs. The dominant variance in a Scrabble result is which tiles came out
of the bag, not which player is better -- so `evaluate.py`'s independent games
need enormous samples to resolve a small edge (~0.5pp of std error at
n=10000). Playing one seeded bag order twice, once with each player in seat 1,
cancels most of that: both players face the same opening racks and the same
early bag, and the seat advantage cancels exactly. What is left is closer to
the thing being measured.

The cancellation is partial, not total -- once the two players choose
different moves the positions diverge and so do the later draws. It still buys
roughly the precision of `evaluate.py` at 4-5x the game count, which matters a
great deal once a move costs simulation time rather than microseconds.

Reported numbers:

  match score   mean over pairs of A's share of the pair (win=1, tie=0.5),
                with a paired standard error. This is the statistic to trust
                and the one Elo is derived from -- its error bar accounts for
                the pairing.
  win rate      wins / (wins + losses) over all 2*pairs games, ties dropped.
                Not paired, and reported only because it is what
                `smart_player/README.md`'s historical numbers use, so the two
                stay comparable.
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm  # type: ignore

from scrablozaur import Board, Dawg, set_num_threads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from strategy import SimplePlayer, StrategicPlayer  # noqa: E402

from languages import engine_language, load as load_language  # noqa: E402
from model import DEFAULT_WEIGHTS_PATH, LANGUAGE  # noqa: E402
from player import DEFAULT_LEAVE_WEIGHT, SmartPlayer  # noqa: E402
from sim_player import DEFAULT_CANDIDATES, DEFAULT_ITERATIONS, DEFAULT_PLIES, SimPlayer  # noqa: E402
from simulate import GamePlayer, play_game  # noqa: E402

_spec = load_language(LANGUAGE)
_lang = engine_language(_spec)
_dawg = Dawg(_lang, str(_spec.dawg), str(_spec.gaddag))

_MASK64 = (1 << 64) - 1


def _init_worker() -> None:
    """Pin the engine to one thread inside each worker process.

    `Board.simulate` fans its iterations out over the engine's own pool (8
    threads by default). Combined with one worker process per core that is an
    8x oversubscription, and it does not merely fail to help -- a sim benchmark
    that should take a minute spends ten thrashing instead. The arena already
    parallelises at the process level, so the engine should stay serial here.
    """
    set_num_threads(1)


def _splitmix64(x: int) -> int:
    """Scatter consecutive pair indices into unrelated 64-bit seeds.

    Board.seeded feeds its argument straight into xorshift64, and seeds that
    differ in one low bit produce streams that are visibly related for the
    first few draws -- which would correlate the pairs we are averaging over.
    """
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def make_player(spec: str, board: Board, seed: int = 0) -> GamePlayer:
    """Build a player from a spec string.

        simple | strategic
        smart[:checkpoint.pt][~leave_weight][!noeg]
        sim[:checkpoint.pt][~leave_weight][!noeg][@iterations[/candidates[/plies]]]

    e.g. `smart`, `smart~0.5`, `smart!noeg`, `sim@500`, `sim~0.4@200/30/1`,
    `sim:models/cand.pt@200`.

    `!noeg` turns the endgame search off. It exists to measure what the search
    is worth, not because anyone should play without it.
    """
    head, _, tuning = spec.partition("@")
    head, _, flags = head.partition("!")
    head, _, weight = head.partition("~")
    name, _, path = head.partition(":")
    leave_weight = float(weight) if weight else DEFAULT_LEAVE_WEIGHT
    use_endgame = "noeg" not in flags

    if name == "simple":
        return SimplePlayer(board)
    if name == "strategic":
        return StrategicPlayer(board)
    if name == "smart":
        return SmartPlayer(board, path or DEFAULT_WEIGHTS_PATH, leave_weight, use_endgame)
    if name == "sim":
        parts = [p for p in tuning.split("/") if p]
        iterations = int(parts[0]) if len(parts) > 0 else DEFAULT_ITERATIONS
        candidates = int(parts[1]) if len(parts) > 1 else DEFAULT_CANDIDATES
        plies = int(parts[2]) if len(parts) > 2 else DEFAULT_PLIES
        return SimPlayer(
            board,
            path or DEFAULT_WEIGHTS_PATH,
            iterations=iterations,
            candidates=candidates,
            plies=plies,
            leave_weight=leave_weight,
            use_endgame=use_endgame,
            seed=seed,
        )
    raise ValueError(f"unknown player spec {spec!r} (expected simple, strategic, smart, or sim)")


def play_pair(spec_a: str, spec_b: str, seed: int) -> tuple[int, int]:
    """Play one seeded bag twice, swapping seats, and return A's score margin
    in each game.

    Both games deal from the same seeded board, and players are constructed in
    seat order (each draws its opening rack in its constructor) -- so seat 1
    gets the same opening rack in both games, and each player plays each side
    of the same deal.
    """
    margins = []
    for a_first in (True, False):
        board = Board.seeded(_lang, seed)
        # Same simulation seed in both games of the pair, so a simulating
        # player samples the same tile orders on each side of the deal.
        if a_first:
            a = make_player(spec_a, board, seed)
            b = make_player(spec_b, board, seed)
            players = [a, b]
        else:
            b = make_player(spec_b, board, seed)
            a = make_player(spec_a, board, seed)
            players = [b, a]
        play_game(players, _dawg, parallel=False)
        margins.append(a.score - b.score)
    return margins[0], margins[1]


def _summarise(pair_margins: list[tuple[int, int]]) -> dict[str, float]:
    n = len(pair_margins)
    if n == 0:
        raise ValueError("no completed pairs")

    # Per-pair mean margin: seat advantage cancels within the pair.
    means = [(m1 + m2) / 2 for m1, m2 in pair_margins]
    mean_margin = sum(means) / n
    if n > 1:
        var = sum((m - mean_margin) ** 2 for m in means) / (n - 1)
        margin_se = math.sqrt(var / n)
    else:
        margin_se = float("nan")

    # Per-pair match score for A: win 1, tie 0.5, loss 0, averaged over the
    # pair's two games. Paired, so its error bar is the honest one.
    def game_score(m: int) -> float:
        return 1.0 if m > 0 else (0.0 if m < 0 else 0.5)

    scores = [(game_score(m1) + game_score(m2)) / 2 for m1, m2 in pair_margins]
    match_score = sum(scores) / n
    if n > 1:
        svar = sum((s - match_score) ** 2 for s in scores) / (n - 1)
        score_se = math.sqrt(svar / n)
    else:
        score_se = float("nan")

    wins = sum(1 for pair in pair_margins for m in pair if m > 0)
    losses = sum(1 for pair in pair_margins for m in pair if m < 0)
    ties = 2 * n - wins - losses
    decisive = wins + losses
    win_rate = wins / decisive if decisive else float("nan")

    # Unpaired std error on the same margin data, purely to show what the
    # pairing bought.
    flat = [m for pair in pair_margins for m in pair]
    fmean = sum(flat) / len(flat)
    fvar = sum((m - fmean) ** 2 for m in flat) / (len(flat) - 1) if len(flat) > 1 else float("nan")
    unpaired_se = math.sqrt(fvar / len(flat))

    return {
        "pairs": n,
        "games": 2 * n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "match_score": match_score,
        "match_score_se": score_se,
        "elo": _elo(match_score),
        "mean_margin": mean_margin,
        "margin_se": margin_se,
        "unpaired_margin_se": unpaired_se,
    }


def _elo(p: float) -> float:
    """Elo difference implied by a match score. Clamped so a clean sweep
    reports a large finite number instead of an infinity."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(p / (1 - p))


def run(
    spec_a: str,
    spec_b: str,
    pairs: int,
    seed0: int = 0,
    n_workers: int | None = None,
    quiet: bool = False,
) -> dict[str, float]:
    """Play `pairs` seeded pairs of A vs B and return the summary statistics."""
    results: list[tuple[int, int]] = []
    label = f"{spec_a} vs {spec_b}"
    t0 = time.perf_counter()
    futures = []

    try:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
            futures = [executor.submit(play_pair, spec_a, spec_b, _splitmix64(seed0 + i)) for i in range(pairs)]
            for future in tqdm(as_completed(futures), total=pairs, desc=label, disable=quiet):
                results.append(future.result())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, summarising what finished...")
        for future in futures:
            future.cancel()

    stats = _summarise(results)
    stats["elapsed_s"] = time.perf_counter() - t0

    if not quiet:
        _print_summary(spec_a, spec_b, stats)
    return stats


def _print_summary(spec_a: str, spec_b: str, s: dict[str, float]) -> None:
    ms, se = s["match_score"], s["match_score_se"]
    print(f"\n{int(s['pairs'])} pairs ({int(s['games'])} games) in {s['elapsed_s']:.1f}s")
    print(f"  A = {spec_a}")
    print(f"  B = {spec_b}")
    print(f"  {int(s['wins'])}W {int(s['losses'])}L {int(s['ties'])}T")
    print(
        f"  match score : {ms * 100:.2f}% +/- {se * 100:.2f}pp (95% CI {(ms - 1.96 * se) * 100:.2f} - {(ms + 1.96 * se) * 100:.2f})"
    )
    print(f"  elo         : {s['elo']:+.0f}  (95% CI {_elo(ms - 1.96 * se):+.0f} to {_elo(ms + 1.96 * se):+.0f})")
    print(f"  win rate    : {s['win_rate'] * 100:.2f}%  (ties dropped; comparable to evaluate.py)")
    print(f"  mean margin : {s['mean_margin']:+.1f} +/- {s['margin_se']:.1f} pts")
    if s["margin_se"] > 0:
        print(
            f"  pairing cut the margin std error {s['unpaired_margin_se'] / s['margin_se']:.2f}x "
            f"(unpaired would be {s['unpaired_margin_se']:.1f})"
        )
    else:
        # Two deterministic, identical players on a mirrored bag play the same
        # game twice with the labels swapped, so every pair cancels exactly.
        # Worth saying out loud: it means the pairing is wired up correctly.
        print("  pairing cancelled the bag entirely (every pair mirrored exactly)")
    if abs(ms - 0.5) < 1.96 * se:
        print("  -> not distinguishable from a tie at this sample size.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--a", required=True, help="Player A spec: simple | strategic | smart[:checkpoint.pt] | sim[:checkpoint.pt]"
    )
    ap.add_argument("--b", required=True, help="Player B spec (same syntax)")
    ap.add_argument("--pairs", type=int, default=500, help="Seeded bag pairs; each is 2 games (default: 500)")
    ap.add_argument("--seed", type=int, default=0, help="Base seed, so a run is reproducible (default: 0)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    run(args.a, args.b, args.pairs, args.seed, args.workers)
