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


def benchmark() -> None:
    N = 10000
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

    print(f"Workers: {n_workers}")
    print(f"Games: {games_played}")
    print(f"Wall time: {wall_elapsed:.2f}s ({games_played / wall_elapsed:.1f} games/s)")
    print(f"CPU time (total): {cpu_total:.2f}s")
    print(f"CPU time (avg per core): {avg_cpu_per_core:.2f}s")
    print(f"CPU utilization (avg per core): {cpu_total / (wall_elapsed * n_workers) * 100:.1f}%")
    print(f"Peak RSS (avg per core): {avg_peak_rss_per_core:.1f} MB")
    print(f"Average score P1: {_weighted_average(p1_scores):.2f}")
    print(f"Average score P2: {_weighted_average(p2_scores):.2f}")
    print(f"Median score P1: {_weighted_median(p1_scores)}")
    print(f"Median score P2: {_weighted_median(p2_scores)}")
    print(f"Max score P1: {max(p1_scores)}")
    print(f"Max score P2: {max(p2_scores)}")
    print(f"Min score P1: {min(p1_scores)}")
    print(f"Min score P2: {min(p2_scores)}")
    ties = games_played - wins[0] - wins[1]
    decisive_games = games_played - ties
    print(f"Wins P1: {wins[0]}")
    print(f"Wins P2: {wins[1]}")
    if decisive_games:
        print(f"Win rate P1: {wins[0] / decisive_games * 100:.2f}%")
        print(f"Win rate P2: {wins[1] / decisive_games * 100:.2f}%")
    print(f"Ties: {ties}")
    print(f"Distinct words played: {len(word_counts)}")
    print("Most placed words:")
    for word, count in word_counts.most_common(10):
        print(f"  {word}: {count}")
    print("Least placed words:")
    for word, count in sorted(word_counts.items(), key=lambda item: item[1])[:10]:
        print(f"  {word}: {count}")
    print(f"Best game (Best score for one player: {best_score}) saved to {best_game_path}")

    plt.hist(list(p1_scores.keys()), weights=list(p1_scores.values()), bins=20, label="Player 1")
    plt.hist(list(p2_scores.keys()), weights=list(p2_scores.values()), bins=20, label="Player 2")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # graj(debug=True)
    benchmark()
