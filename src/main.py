import argparse
import os
import resource
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import random

from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "smart_player"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scrablozaur import Board, Dawg
from smart_player.player import SmartPlayer
from smart_player.sim_player import SimPlayer
from strategy import SimplePlayer, StrategicPlayer

d = Dawg("words/dawg.bin", "words/gaddag.bin")


def _rusage_self_now() -> float:
    """This worker's own cumulative CPU seconds since it started.

    Self-reported rather than measured by the parent via RUSAGE_CHILDREN:
    under the `forkserver` start method (Python 3.14's new POSIX default),
    the actual worker is a grandchild spawned by a long-lived forkserver
    helper, so the parent's RUSAGE_CHILDREN never sees its usage -- the
    helper hasn't exited (and so hasn't been reaped/aggregated) by the time
    we'd check. Self-reporting works the same under fork, spawn, and
    forkserver alike.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _weighted_average(counts: Counter[int]) -> float:
    """Average of a distribution given as {score: occurrences}."""
    total_games = sum(counts.values())
    return sum(score * count for score, count in counts.items()) / total_games


def _weighted_median(counts: Counter[int]) -> float:
    """Median of a distribution given as {score: occurrences}, without expanding it."""
    total_games = sum(counts.values())
    sorted_items = sorted(counts.items())

    def value_at(position: int) -> int:
        """The score at the given 1-indexed position in the sorted, expanded multiset."""
        cumulative = 0
        for score, count in sorted_items:
            cumulative += count
            if cumulative >= position:
                return score
        raise ValueError("position out of range")

    if total_games % 2:
        return float(value_at((total_games + 1) // 2))

    lo = value_at(total_games // 2)
    hi = value_at(total_games // 2 + 1)
    return (lo + hi) / 2


def _render_table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    """Render rows as a plain ASCII table.

    `align` is one 'l'/'r' per column; defaults to left-aligning the first
    column (labels) and right-aligning the rest (numbers).
    """
    if align is None:
        align = "l" + "r" * (len(headers) - 1)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(cells: list[str]) -> str:
        aligned = [cell.rjust(w) if a == "r" else cell.ljust(w) for cell, w, a in zip(cells, widths, align)]
        return "| " + " | ".join(aligned) + " |"

    separator = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    lines = [separator, format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def graj(
    p1_type: str, p2_type: str, parallel: bool = False, debug: bool = False
) -> tuple[int, int, str, float, Counter[str], float, int]:
    cpu_start = _rusage_self_now()

    log: list[str] = []
    words_played: Counter[str] = Counter()
    move_time_total = 0.0
    move_count = 0

    def emit(*parts: object) -> None:
        """Record a line to the game transcript, and print it too if debug is on."""
        line = " ".join(str(p) for p in parts)
        log.append(line)
        if debug:
            print(line)

    def play(player: SimplePlayer | StrategicPlayer | SmartPlayer) -> str | None:
        nonlocal move_time_total, move_count
        move_start = time.perf_counter()
        word = player.play_word(d, parallel=parallel)
        move_time_total += time.perf_counter() - move_start
        move_count += 1
        if word:
            words_played[word] += 1
        return word

    b = Board()

    match p1_type:
        case "1":
            p1 = SimplePlayer(b)
        case "2":
            p1 = StrategicPlayer(b)
        case "3":
            p1 = SmartPlayer(b)
        case "4":
            p1 = SimPlayer(b)
        case _:
            raise ValueError(f"Unknown player type: {p1_type}")

    match p2_type:
        case "1":
            p2 = SimplePlayer(b)
        case "2":
            p2 = StrategicPlayer(b)
        case "3":
            p2 = SmartPlayer(b)
        case "4":
            p2 = SimPlayer(b)
        case _:
            raise ValueError(f"Unknown player type: {p2_type}")

    opener = p1 if random() < 0.5 else p2
    second = p2 if opener is p1 else p1
    players = [opener, second]

    # Only a genuine no-action turn (no legal word AND can't exchange --
    # play_word() returns "" for this, vs None for an exchange) counts
    # toward ending the game. Exchanging is a real, repeatable action a
    # player can take as many times as they want (see strategy.py) and
    # never signals a stuck/deadlocked game on its own.
    no_play_streak = 0
    went_out_idx: int | None = None
    turn = 0

    while True:
        idx = turn % 2
        player = players[idx]
        name = "Player 1" if player is p1 else "Player 2"
        w = play(player)
        if w:
            no_play_streak = 0
            emit(f"{name} plays: {w}")
            emit(b)
            if not player.letters:
                went_out_idx = idx
                break
        elif w is None:
            emit(f"{name} exchanged letters")
            emit(b)
        else:
            no_play_streak += 1
            emit(f"{name} cannot play.")
            emit(b)
            if no_play_streak >= 4:
                break
        turn += 1

    if went_out_idx is not None:
        others_value = sum(Board.rack_value(pl.letters) for i, pl in enumerate(players) if i != went_out_idx)
        players[went_out_idx].score += others_value
        for i, pl in enumerate(players):
            if i != went_out_idx:
                pl.score -= Board.rack_value(pl.letters)
    else:
        for pl in players:
            pl.score -= Board.rack_value(pl.letters)

    emit(f"Final Scores: Player 1: {p1.score}, Player 2: {p2.score}")
    emit(b)

    cpu_end = _rusage_self_now()
    return p1.score, p2.score, "\n".join(log), cpu_end - cpu_start, words_played, move_time_total, move_count


def _print_benchmark_results(
    n_workers: int,
    games_played: int,
    wall_elapsed: float,
    cpu_total: float,
    avg_cpu_per_core: float,
    avg_move_time_ms: float,
    wins: list[int],
    ties: int,
    win_rate_p1: str,
    win_rate_p2: str,
    word_counts: Counter[str],
    best_score: int,
    best_game_path: str,
    p1_scores: Counter[int],
    p2_scores: Counter[int],
) -> None:
    print()
    print(
        _render_table(
            ["Run", "Value"],
            [
                ["Workers", str(n_workers)],
                ["Games played", str(games_played)],
                ["Wall time", f"{wall_elapsed:.2f}s"],
                ["Throughput", f"{games_played / wall_elapsed:.1f} games/s"],
                ["CPU time (total)", f"{cpu_total:.2f}s"],
                ["CPU time (avg/core)", f"{avg_cpu_per_core:.2f}s"],
                ["CPU utilization (avg/core)", f"{cpu_total / (wall_elapsed * n_workers) * 100:.1f}%"],
                ["Avg time per game", f"{wall_elapsed / games_played * 1000:.2f} ms"],
                ["Avg time per move (avg per game)", f"{avg_move_time_ms:.2f} ms"],
                ["Avg moves per game", f"{sum(word_counts.values()) / games_played:.2f}"],
                ["Moves played", str(sum(word_counts.values()))],
                ["Distinct words played", str(len(word_counts))],
                ["Best single-player score", f"{best_score}"],
            ],
        )
    )

    print()
    print(
        _render_table(
            ["Score", "Player 1", "Player 2"],
            [
                ["Average", f"{_weighted_average(p1_scores):.2f}", f"{_weighted_average(p2_scores):.2f}"],
                ["Median", f"{_weighted_median(p1_scores):.1f}", f"{_weighted_median(p2_scores):.1f}"],
                ["Max", str(max(p1_scores)), str(max(p2_scores))],
                ["Min", str(min(p1_scores)), str(min(p2_scores))],
                ["Wins", str(wins[0]), str(wins[1])],
                ["Win rate", win_rate_p1, win_rate_p2],
                ["Ties", str(ties), str(ties)],
            ],
        )
    )

    print()
    print("Best game transcript written to:", best_game_path)

    plt.hist(list(p1_scores.keys()), weights=list(p1_scores.values()), bins=20, label="Player 1")
    plt.hist(list(p2_scores.keys()), weights=list(p2_scores.values()), bins=20, label="Player 2")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores")
    plt.legend()
    # plt.show()


def benchmark(N: int, p1_type: str, p2_type: str, n_workers: int | None = None, debug: bool = False) -> None:
    # Scores are heavily repeated across thousands of games, so track
    # {score: occurrences} per player instead of one entry per game -- keeps
    # memory bounded by the number of distinct scores rather than N.
    p1_scores: Counter[int] = Counter()
    p2_scores: Counter[int] = Counter()
    wins = [0, 0]
    best_score = -1
    best_transcript = ""
    cpu_total = 0.0
    total_move_time = 0.0
    total_moves = 0
    word_counts: Counter[str] = Counter()
    games_played = 0

    parallel = True if n_workers == 1 else False
    if n_workers is None:
        n_workers = os.cpu_count() - 1 or 1

    # Per-move rayon parallelism is only engaged with a single worker; with
    # several workers each game runs single-threaded (process-level parallelism).
    if parallel:
        engine_threads = int(os.environ.get("RAYON_NUM_THREADS") or min(8, os.cpu_count() or 1))
        print(f"Engine move generation: parallel, {engine_threads} rayon thread(s) per move")
    else:
        print("Engine move generation: single-threaded per move (parallelism is across workers)")

    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        n_workers = executor._max_workers
        print(f"Running {N} games with {n_workers} worker process(es)...")
        try:
            with tqdm(total=N, desc="Games played") as pbar:
                batch_size = n_workers * 1000
                for i in range(0, N, batch_size):
                    futures = [
                        executor.submit(graj, p1_type, p2_type, parallel, debug) for _ in range(min(batch_size, N - i))
                    ]

                    for future in as_completed(futures):
                        p1, p2, transcript, cpu_time, game_words, move_time, move_count = future.result()
                        games_played += 1
                        p1_scores[p1] += 1
                        p2_scores[p2] += 1
                        cpu_total += cpu_time
                        total_move_time += move_time
                        total_moves += move_count
                        pbar.update(1)

                        word_counts.update(game_words)
                        if p1 > p2:
                            wins[0] += 1
                        elif p2 > p1:
                            wins[1] += 1

                        if p1 > best_score or p2 > best_score:
                            best_score = max(p1, p2)
                            best_transcript = transcript

        except KeyboardInterrupt:
            # Drop not-yet-started games so shutdown doesn't run through the
            # rest of the queue; already-running games are left to finish.
            print(f"\nInterrupted -- stopping after {games_played}/{N} games.")
            executor.shutdown(wait=True, cancel_futures=True)

    if games_played == 0:
        print("No games completed.")
        return

    best_game_path = "_best_game.txt"
    with open(best_game_path, "w") as f:
        f.write(best_transcript + "\n")

    wall_elapsed = time.perf_counter() - wall_start
    avg_cpu_per_core = cpu_total / n_workers
    avg_move_time_ms = total_move_time / total_moves * 1000

    ties = games_played - wins[0] - wins[1]
    decisive_games = games_played - ties
    win_rate_p1 = f"{wins[0] / decisive_games * 100:.2f}%" if decisive_games else "N/A"
    win_rate_p2 = f"{wins[1] / decisive_games * 100:.2f}%" if decisive_games else "N/A"

    _print_benchmark_results(
        n_workers=n_workers,
        games_played=games_played,
        wall_elapsed=wall_elapsed,
        cpu_total=cpu_total,
        avg_cpu_per_core=avg_cpu_per_core,
        avg_move_time_ms=avg_move_time_ms,
        wins=wins,
        ties=ties,
        win_rate_p1=win_rate_p1,
        win_rate_p2=win_rate_p2,
        word_counts=word_counts,
        best_score=best_score,
        best_game_path=best_game_path,
        p1_scores=p1_scores,
        p2_scores=p2_scores,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the engine by playing simulated games.")
    parser.add_argument("games", type=int, nargs="?", default=10000, help="Number of games to play (default: 10000)")
    parser.add_argument(
        "p1",
        type=str,
        nargs="?",
        default="1",
        help="Player 1 type: 1=Simple, 2=Strategic, 3=Smart, 4=Sim (default: 1)",
    )
    parser.add_argument(
        "p2",
        type=str,
        nargs="?",
        default="1",
        help="Player 2 type: 1=Simple, 2=Strategic, 3=Smart, 4=Sim (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes to use (default: all available cores)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Rayon threads for the engine's parallel move generation "
        "(default: min(8, cores)); only engaged when a single worker runs move-gen in parallel",
    )
    parser.add_argument("--debug", action="store_true", help="Print detailed game logs for debugging purposes")
    args = parser.parse_args()

    if args.threads is not None:
        # Set before any worker is spawned so each inherits it; the Rust engine
        # reads RAYON_NUM_THREADS when it builds its move-generation pool.
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)

    benchmark(args.games, args.p1, args.p2, args.workers, args.debug)
