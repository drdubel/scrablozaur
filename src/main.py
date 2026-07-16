import statistics

from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm

from scrablozaur import Board, Dawg

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
        w = self.board.get_best_word(dawg, self.letters, first, parallel=True)
        self.score += w[1]
        self.board.place_word(w[0], w[2][0], w[2][1], w[2][2])
        for ch in w[3]:
            self.letters = self.letters.replace(ch, "", 1)
        self.draw_letters()
        return w[0]


def graj(debug: bool = False) -> tuple[int, int]:
    b = Board([["-" for _ in range(15)] for _ in range(15)])

    p1 = Player(b)
    p2 = Player(b)
    w = p1.play_word(d, first=True)
    if debug:
        print(f"Player 1 plays: {w}")
        print(b)

    while w:
        w = p2.play_word(d)
        if w:
            if debug:
                print(f"Player 2 plays: {w}")
                print(b)
        else:
            if debug:
                print("Player 2 cannot play.")

        w = p1.play_word(d)
        if w:
            if debug:
                print(f"Player 1 plays: {w}")
                print(b)
        else:
            if debug:
                print("Player 1 cannot play.")
                print(b)
            break

    if debug:
        print(f"Final Scores: Player 1: {p1.score}, Player 2: {p2.score}")
        print(b)

    return p1.score, p2.score


def speed_test() -> None:
    N = 1000
    scores = []

    with tqdm(total=N) as pbar:
        for _ in range(N):
            p1, p2 = graj(debug=False)
            scores.extend([p1, p2])
            pbar.update(1)

    print(f"Average score: {sum(scores) / len(scores):.2f}")
    print(f"Median score: {statistics.median(scores)}")
    print(f"Max score: {max(scores)}")
    print(f"Min score: {min(scores)}")

    plt.hist(scores, bins=20)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores")
    plt.show()


if __name__ == "__main__":
    # graj(debug=True)
    speed_test()
