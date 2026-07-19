import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm

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

    def play_word(self, dawg: Dawg, first: bool = False) -> str:
        """Find and play the best word from the player's letters on the board.

        This method should:
          - Analyze the board to find valid placement patterns.
          - Use the DAWG to find the best scoring word that can be formed with
            the player's letters and fits one of the patterns.
          - Place the word on the board and update the player's letters.
        """
        w = self.board.get_best_word(dawg, self.letters, first, parallel=False)
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
            else:
                self.letters = self.letters.replace("?", "", 1)
        self.draw_letters()
        return w[0]


def graj(debug: bool = False) -> tuple[int, int]:
    b = Board([["-" for _ in range(15)] for _ in range(15)])

    p1 = StrategicPlayer(b)
    p2 = Player(b)
    opener: Player | StrategicPlayer = p1
    second: Player | StrategicPlayer = p2
    w = opener.play_word(d, first=True)
    if debug:
        print(f"Player 1 plays: {w}")
        print(b)

    if not w:
        # The opener's rack couldn't form any word through the centre (rare,
        # but happens -- e.g. an all-consonant draw). Give the other player a
        # shot at the opening instead of ending the game 0-0 before it starts.
        opener, second = p2, p1
        w = opener.play_word(d, first=True)
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

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(graj, False) for _ in range(N)]
        for future in tqdm(as_completed(futures), total=N):
            p1, p2 = future.result()
            scores.append((p1, p2))
            if p1 > p2:
                wins[0] += 1
            elif p2 > p1:
                wins[1] += 1

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
