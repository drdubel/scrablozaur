import argparse
import resource
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import random

from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm  # type: ignore

from scrablozaur import Board, Dawg
from strategy import SimplePlayer, StrategicPlayer

d = Dawg("words/dawg.bin")


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


def graj(debug: bool = False) -> tuple[int, int, str, float, Counter[str]]:
    cpu_start = _rusage_self_now()

    log: list[str] = []
    words_played: Counter[str] = Counter()

    def emit(*parts: object) -> None:
        """Record a line to the game transcript, and print it too if debug is on."""
        line = " ".join(str(p) for p in parts)
        log.append(line)
        if debug:
            print(line)

    def play(player: SimplePlayer | StrategicPlayer) -> str:
        word = player.play_word(d)
        if word:
            words_played[word] += 1
        return word

    b = Board()

    p1 = StrategicPlayer(b)
    p2 = SimplePlayer(b)

    opener = p1 if random() < 0.5 else p2
    second = p2 if opener is p1 else p1

    w = play(opener)
    emit(f"Player 1 plays: {w}")
    emit(b)

    if not w:
        # The opener's rack couldn't form any word through the centre (rare,
        # but happens -- e.g. an all-consonant draw). Give the other player a
        # shot at the opening instead of ending the game 0-0 before it starts.
        opener, second = second, opener
        w = play(opener)
        emit("Player 1 cannot open -- Player 2 plays:", w)
        emit(b)
        if not w:
            # Neither player's opening rack is playable -- genuinely stuck.
            cpu_end = _rusage_self_now()
            return p1.score, p2.score, "\n".join(log), cpu_end - cpu_start, words_played

    while w:
        w = play(second)
        if w:
            emit(f"{'Player 2' if second is p2 else 'Player 1'} plays: {w}")
            emit(b)
        else:
            emit(f"{'Player 2' if second is p2 else 'Player 1'} cannot play.")

        w = play(opener)
        if w:
            emit(f"{'Player 1' if opener is p1 else 'Player 2'} plays: {w}")
            emit(b)
        else:
            emit(f"{'Player 1' if opener is p1 else 'Player 2'} cannot play.")
            emit(b)
            break

    emit(f"Final Scores: Player 1: {p1.score}, Player 2: {p2.score}")
    emit(b)

    cpu_end = _rusage_self_now()
    return p1.score, p2.score, "\n".join(log), cpu_end - cpu_start, words_played


def _print_benchmark_results(
    n_workers: int,
    games_played: int,
    wall_elapsed: float,
    cpu_total: float,
    avg_cpu_per_core: float,
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

    most_placed = word_counts.most_common(10)
    least_placed = sorted(word_counts.items(), key=lambda item: item[1])[:10]
    print()
    print(
        _render_table(
            ["#", "Most placed", "Count", "Least placed", "Count"],
            [
                [str(i + 1), most_word, str(most_count), least_word, str(least_count)]
                for i, ((most_word, most_count), (least_word, least_count)) in enumerate(zip(most_placed, least_placed))
            ],
            align="llrlr",
        )
    )

    print("Best game transcript written to:", best_game_path)

    plt.hist(list(p1_scores.keys()), weights=list(p1_scores.values()), bins=20, label="Player 1")
    plt.hist(list(p2_scores.keys()), weights=list(p2_scores.values()), bins=20, label="Player 2")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores")
    plt.legend()
    # plt.show()


def benchmark(N: int) -> None:
    # Scores are heavily repeated across thousands of games, so track
    # {score: occurrences} per player instead of one entry per game -- keeps
    # memory bounded by the number of distinct scores rather than N.
    p1_scores: Counter[int] = Counter()
    p2_scores: Counter[int] = Counter()
    wins = [0, 0]
    best_score = -1
    best_transcript = ""
    cpu_total = 0.0
    word_counts: Counter[str] = Counter()
    games_played = 0

    wall_start = time.perf_counter()

    with ProcessPoolExecutor() as executor:
        n_workers = executor._max_workers
        try:
            with tqdm(total=N, desc="Games played") as pbar:
                batch_size = n_workers * 100
                for i in range(0, N, batch_size):
                    futures = [executor.submit(graj, False) for _ in range(min(batch_size, N - i))]

                    for future in as_completed(futures):
                        p1, p2, transcript, cpu_time, game_words = future.result()
                        games_played += 1
                        p1_scores[p1] += 1
                        p2_scores[p2] += 1
                        cpu_total += cpu_time
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
    args = parser.parse_args()

    # graj(debug=True)
    benchmark(args.games)
