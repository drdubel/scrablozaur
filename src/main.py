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


def graj(debug: bool = False) -> tuple[int, int]:
    b = Board()

    p1 = Player(b)
    p2 = Player(b)

    opener = p1 if random() < 0.5 else p2
    second = p2 if opener is p1 else p1

    w = opener.play_word(d)
    if debug:
        print(f"Player 1 plays: {w}")
        print(b)

    if not w:
        # The opener's rack couldn't form any word through the centre (rare,
        # but happens -- e.g. an all-consonant draw). Give the other player a
        # shot at the opening instead of ending the game 0-0 before it starts.
        opener, second = second, opener
        w = opener.play_word(d)
        if debug:
            print("Player 1 cannot open -- Player 2 plays:", w)
            print(b)
        if not w:
            # Neither player's opening rack is playable -- genuinely stuck.
            return p1.score, p2.score

    while w:
        w = second.play_word(d)
        if w:
            if debug:
                print(f"{'Player 2' if second is p2 else 'Player 1'} plays: {w}")
                print(b)
        else:
            if debug:
                print(f"{'Player 2' if second is p2 else 'Player 1'} cannot play.")

        w = opener.play_word(d)
        if w:
            if debug:
                print(f"{'Player 1' if opener is p1 else 'Player 2'} plays: {w}")
                print(b)
        else:
            if debug:
                print(f"{'Player 1' if opener is p1 else 'Player 2'} cannot play.")
                print(b)
            break

    if debug:
        print(f"Final Scores: Player 1: {p1.score}, Player 2: {p2.score}")
        print(b)

    return p1.score, p2.score


def speed_test() -> None:
    N = 10000
    scores = []
    wins = [0, 0]

    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_start = time.perf_counter()

    with ProcessPoolExecutor() as executor:
        n_workers = executor._max_workers
        futures = [executor.submit(graj, False) for _ in range(N)]
        for future in tqdm(as_completed(futures), total=N):
            p1, p2 = future.result()
            scores.append((p1, p2))
            if p1 > p2:
                wins[0] += 1
            elif p2 > p1:
                wins[1] += 1

    wall_elapsed = time.perf_counter() - wall_start
    cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    # RUSAGE_CHILDREN aggregates usage of the pool's worker processes once
    # they've exited (which they have -- the `with` block above already
    # joined them), so this is total CPU time spent playing games, not
    # wall time.
    cpu_user = cpu_after.ru_utime - cpu_before.ru_utime
    cpu_sys = cpu_after.ru_stime - cpu_before.ru_stime
    cpu_total = cpu_user + cpu_sys
    # ru_maxrss is bytes on macOS/BSD but kilobytes on Linux.
    peak_rss_mb = cpu_after.ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)

    print(f"Workers: {n_workers}")
    print(f"Games: {N}")
    print(f"Wall time: {wall_elapsed:.2f}s ({N / wall_elapsed:.1f} games/s)")
    print(f"CPU time: {cpu_total:.2f}s (user {cpu_user:.2f}s, sys {cpu_sys:.2f}s)")
    print(f"CPU utilization: {cpu_total / (wall_elapsed * n_workers) * 100:.1f}%")
    print(f"Peak worker RSS: {peak_rss_mb:.1f} MB")
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

    plt.hist([score[0] for score in scores], bins=20, label="Player 1")
    plt.hist([score[1] for score in scores], bins=20, label="Player 2")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # graj(debug=True)
    speed_test()
