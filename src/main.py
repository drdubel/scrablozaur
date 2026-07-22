import os
import resource
import statistics
import sys
import time
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


def graj(debug: bool = False) -> tuple[int, int, str, float, float, int]:
    cpu_start, _ = _rusage_self_now()
    pid = os.getpid()

    log: list[str] = []

    def emit(*parts: object) -> None:
        """Record a line to the game transcript, and print it too if debug is on."""
        line = " ".join(str(p) for p in parts)
        log.append(line)
        if debug:
            print(line)

    b = Board()

    p1 = Player(b)
    p2 = Player(b)

    opener = p1 if random() < 0.5 else p2
    second = p2 if opener is p1 else p1

    w = opener.play_word(d)
    emit(f"Player 1 plays: {w}")
    emit(b)

    if not w:
        # The opener's rack couldn't form any word through the centre (rare,
        # but happens -- e.g. an all-consonant draw). Give the other player a
        # shot at the opening instead of ending the game 0-0 before it starts.
        opener, second = second, opener
        w = opener.play_word(d)
        emit("Player 1 cannot open -- Player 2 plays:", w)
        emit(b)
        if not w:
            # Neither player's opening rack is playable -- genuinely stuck.
            cpu_end, peak_rss_mb = _rusage_self_now()
            return p1.score, p2.score, "\n".join(log), cpu_end - cpu_start, peak_rss_mb, pid

    while w:
        w = second.play_word(d)
        if w:
            emit(f"{'Player 2' if second is p2 else 'Player 1'} plays: {w}")
            emit(b)
        else:
            emit(f"{'Player 2' if second is p2 else 'Player 1'} cannot play.")

        w = opener.play_word(d)
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
    return p1.score, p2.score, "\n".join(log), cpu_end - cpu_start, peak_rss_mb, pid


def benchmark() -> None:
    N = 10000
    scores = []
    wins = [0, 0]
    best_score = -1
    best_transcript = ""
    cpu_total = 0.0
    # ru_maxrss is a high-water mark for the whole worker process, so keep
    # only the latest (i.e. largest) figure reported per pid rather than
    # summing across every game that worker has played.
    worker_peak_rss_mb: dict[int, float] = {}

    wall_start = time.perf_counter()

    with ProcessPoolExecutor() as executor:
        n_workers = executor._max_workers
        futures = [executor.submit(graj, False) for _ in range(N)]
        for future in tqdm(as_completed(futures), total=N):
            p1, p2, transcript, cpu_time, peak_rss_mb, pid = future.result()
            scores.append((p1, p2))
            cpu_total += cpu_time
            worker_peak_rss_mb[pid] = max(worker_peak_rss_mb.get(pid, 0.0), peak_rss_mb)
            if p1 > p2:
                wins[0] += 1
            elif p2 > p1:
                wins[1] += 1

            if p1 > best_score or p2 > best_score:
                best_score = max(p1, p2)
                best_transcript = transcript

    best_game_path = "_best_game.txt"
    with open(best_game_path, "w") as f:
        f.write(best_transcript + "\n")

    wall_elapsed = time.perf_counter() - wall_start
    avg_cpu_per_core = cpu_total / n_workers
    avg_peak_rss_per_core = sum(worker_peak_rss_mb.values()) / len(worker_peak_rss_mb)

    print(f"Workers: {n_workers}")
    print(f"Games: {N}")
    print(f"Wall time: {wall_elapsed:.2f}s ({N / wall_elapsed:.1f} games/s)")
    print(f"CPU time (total): {cpu_total:.2f}s")
    print(f"CPU time (avg per core): {avg_cpu_per_core:.2f}s")
    print(f"CPU utilization (avg per core): {cpu_total / (wall_elapsed * n_workers) * 100:.1f}%")
    print(f"Peak RSS (avg per core): {avg_peak_rss_per_core:.1f} MB")
    print(f"Average score P1: {sum(score[0] for score in scores) / len(scores):.2f}")
    print(f"Average score P2: {sum(score[1] for score in scores) / len(scores):.2f}")
    print(f"Median score P1: {statistics.median(score[0] for score in scores)}")
    print(f"Median score P2: {statistics.median(score[1] for score in scores)}")
    print(f"Max score P1: {max(score[0] for score in scores)}")
    print(f"Max score P2: {max(score[1] for score in scores)}")
    print(f"Min score P1: {min(score[0] for score in scores)}")
    print(f"Min score P2: {min(score[1] for score in scores)}")
    print(f"Wins P1: {wins[0]}")
    print(f"Wins P2: {wins[1]}")
    print(f"Win rate P1: {wins[0] / N * 100:.2f}%")
    print(f"Win rate P2: {wins[1] / N * 100:.2f}%")
    print(f"Ties: {N - wins[0] - wins[1]}")
    print(f"Best game (Best score for one player: {best_score}) saved to {best_game_path}")

    plt.hist([score[0] for score in scores], bins=20, label="Player 1")
    plt.hist([score[1] for score in scores], bins=20, label="Player 2")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # graj(debug=True)
    benchmark()
