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

from scrablozaur import Board, Dawg
from strategy import StrategicPlayer

d = Dawg("words/dawg.bin")


class Player:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.letters = ""
        self.draw_letters()
        self.score = 0

    def draw_letters(self) -> None:
        """Draw letters from the bag to fill the player's hand up to 7 letters."""
        self.letters += self.board.give_letters(self.letters)

    def play_word(self, dawg: Dawg) -> str:
        """Find and play the best word from the player's letters on the board.

        This method should:
          - Analyze the board to find valid placement patterns.
          - Use the DAWG to find the best scoring word that can be formed with
            the player's letters and fits one of the patterns.
          - Place the word on the board and update the player's letters.
        """
        w = self.board.get_best_word(dawg, self.letters, parallel=False)
        self.score += w[1]
        self.board.place_word(w[0], w[2][0], w[2][1], w[2][2])
        for ch in w[3]:
            # `ch` is the literal letter placed on the board, but if a blank
            # stood in for it, the rack only has '?' -- not `ch` -- so fall
            # back to removing the blank instead. Without this, `replace()`
            # silently no-ops on a literal it can't find, the blank never
            # actually leaves the rack, and it gets "reused" as a fresh
            # wildcard every subsequent turn for the rest of the game.
            if ch in self.letters:
                self.letters = self.letters.replace(ch, "", 1)
            elif "?" in self.letters:
                self.letters = self.letters.replace("?", "", 1)
            else:
                raise ValueError(f"Letter '{ch}' not found in player's letters.")

        self.draw_letters()
        return w[0]


def _rusage_self_now() -> tuple[float, float]:
    """This worker's own (cpu_seconds, peak_rss_mb) since it started.

    Self-reported rather than measured by the parent via RUSAGE_CHILDREN:
    under the `forkserver` start method (Python 3.14's new POSIX default),
    the actual worker is a grandchild spawned by a long-lived forkserver
    helper, so the parent's RUSAGE_CHILDREN never sees its usage -- the
    helper hasn't exited (and so hasn't been reaped/aggregated) by the time
    we'd check. Self-reporting works the same under fork, spawn, and
    forkserver alike.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = usage.ru_utime + usage.ru_stime
    # ru_maxrss is bytes on macOS/BSD but kilobytes on Linux.
    peak_rss_mb = usage.ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return cpu_seconds, peak_rss_mb


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


def graj(debug: bool = False) -> tuple[int, int, str, float, float, int, Counter[str]]:
    cpu_start, _ = _rusage_self_now()
    pid = os.getpid()

    log: list[str] = []
    words_played: Counter[str] = Counter()

    def emit(*parts: object) -> None:
        """Record a line to the game transcript, and print it too if debug is on."""
        line = " ".join(str(p) for p in parts)
        log.append(line)
        if debug:
            print(line)

    def play(player: Player) -> str:
        word = player.play_word(d)
        if word:
            words_played[word] += 1
        return word

    b = Board()

    p1 = Player(b)
    p2 = Player(b)

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
            cpu_end, peak_rss_mb = _rusage_self_now()
            return p1.score, p2.score, "\n".join(log), cpu_end - cpu_start, peak_rss_mb, pid, words_played

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

    cpu_end, peak_rss_mb = _rusage_self_now()
    return p1.score, p2.score, "\n".join(log), cpu_end - cpu_start, peak_rss_mb, pid, words_played


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
    # ru_maxrss is a high-water mark for the whole worker process, so keep
    # only the latest (i.e. largest) figure reported per pid rather than
    # summing across every game that worker has played.
    worker_peak_rss_mb: dict[int, float] = {}
    word_counts: Counter[str] = Counter()
    games_played = 0

    wall_start = time.perf_counter()

    with ProcessPoolExecutor() as executor:
        n_workers = executor._max_workers
        futures = [executor.submit(graj, False) for _ in range(N)]
        try:
            for future in tqdm(as_completed(futures), total=N):
                p1, p2, transcript, cpu_time, peak_rss_mb, pid, game_words = future.result()
                games_played += 1
                p1_scores[p1] += 1
                p2_scores[p2] += 1
                cpu_total += cpu_time
                worker_peak_rss_mb[pid] = max(worker_peak_rss_mb.get(pid, 0.0), peak_rss_mb)
                word_counts.update(game_words)
                if p1 > p2:
                    wins[0] += 1
                elif p2 > p1:
                    wins[1] += 1

                if p1 > best_score or p2 > best_score:
                    best_score = max(p1, p2)
                    best_transcript = transcript
                    print(f"New best score: {best_score} (P1: {p1}, P2: {p2})")
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
    avg_peak_rss_per_core = sum(worker_peak_rss_mb.values()) / len(worker_peak_rss_mb)

    ties = games_played - wins[0] - wins[1]
    decisive_games = games_played - ties
    win_rate_p1 = f"{wins[0] / decisive_games * 100:.2f}%" if decisive_games else "N/A"
    win_rate_p2 = f"{wins[1] / decisive_games * 100:.2f}%" if decisive_games else "N/A"

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
                ["Peak RSS (avg/core)", f"{avg_peak_rss_per_core:.1f} MB"],
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
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the engine by playing simulated games.")
    parser.add_argument("games", type=int, nargs="?", default=10000, help="Number of games to play (default: 10000)")
    args = parser.parse_args()

    # graj(debug=True)
    benchmark(args.games)
