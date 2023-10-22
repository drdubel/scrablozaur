import pickle
import re
from operator import itemgetter

from src.create_dawg import Node
from src.points import bonuses, letter_points

dawg = pickle.loads(open("words/dawg.pickle", "rb").read())


class Game:
    def __init__(self):
        self.letters = input()
        self.board = [input().split() for _ in range(15)]
        self.possible_words = []
        self.best_word = ()

    def insert_best_word(self):
        if self.best_word[0]:
            self.board = list(zip(*self.board))

    def validate_word(self, node: Node, word: str, x: int = 0):
        if x == len(word):
            if node.is_terminal:
                return word
            return False
        if word[x] in node.children:
            return self.validate_word(node.children[word[x]], word, x + 1)
        return False

    def check_crossword(self, column, new_letter, y, x) -> tuple:
        points = 0
        for result in re.finditer(r"\w+", column):
            if result.start() <= y and y <= result.end():
                if self.validate_word(dawg, column[result.start() : result.end()]):
                    points += sum(
                        map(
                            lambda letter: letter_points[letter],
                            column[result.start() : result.end()],
                        )
                    )
                    if (y, x) in bonuses:
                        points += letter_points[new_letter] * (bonuses[(y, x)][0] - 1)
                        points *= bonuses[(y, x)][1]
                    return column[result.start() : result.end()], points
        return False, 0

    def find_words(
        self,
        node: Node,
        av_letters: tuple,
        words: list,
        y: int,
        addit_words: list,
        orientation: int = 0,
        word: str = "",
        can_be: tuple = (False, False),
        points: tuple = (0, 0, 1),
        x: int = 0,
    ) -> list:
        if (
            node.is_terminal
            and can_be[0]
            and can_be[1]
            and (
                x == 15
                or (x <= 14 and self.board[y][x] == "-")
                or (x <= 13 and self.board[y][x] == "-" and self.board[y][x + 1] == "-")
            )
        ):
            pos = (y, x - len(word))
            score = points[0] * points[2] + points[1]
            if orientation:
                pos = pos[::-1]

            if av_letters[0] == 7:
                score += 50

            words.append((orientation, pos, word, score))

        if x == 15:
            return words

        if self.board[y][x] in node.children:
            words = self.find_words(
                node.children[self.board[y][x]],
                av_letters,
                words,
                y,
                addit_words,
                orientation=orientation,
                word=word + self.board[y][x],
                can_be=(True, True),
                points=(
                    points[0] + letter_points[self.board[y][x]],
                    points[1],
                    points[2],
                ),
                x=x + 1,
            )

        elif not word and x > 0 and self.board[y][x - 1] != "-":
            pass

        elif self.board[y][x] == "-":
            for letter, child in node.children.items():
                new_points = 0
                new_addit_word = ""
                if letter not in av_letters[1]:
                    continue

                if (y > 0 and self.board[y - 1][x] != "-") or (
                    y < 14 and self.board[y + 1][x] != "-"
                ):
                    column = list(list(zip(*self.board))[x])
                    column[y] = letter
                    new_addit_word, new_points = self.check_crossword(
                        "".join(column), letter, y, x
                    )
                    if not new_addit_word:
                        continue

                bonus = bonuses[(y, x)] if (y, x) in bonuses else (1, 1)

                words = self.find_words(
                    child,
                    (av_letters[0] + 1, av_letters[1].replace(letter, "", 1)),
                    words,
                    y,
                    addit_words + [new_addit_word] if new_addit_word else addit_words,
                    orientation=orientation,
                    word=word + letter,
                    can_be=(True, True) if new_addit_word else (can_be[0], True),
                    points=(
                        points[0] + letter_points[letter] * bonus[0],
                        points[1] + new_points,
                        points[2] * bonus[1],
                    ),
                    x=x + 1,
                )

        if not word:
            words = self.find_words(
                node,
                av_letters,
                words,
                y,
                addit_words,
                orientation=orientation,
                x=x + 1,
            )

        return words

    def find_all_words(self):
        self.possible_words = []

        for i in range(15):
            if (
                self.board[i].count("-") == 15
                and (i == 0 or self.board[i - 1].count("-") == 15)
                and (i == 14 or self.board[i + 1].count("-") == 15)
            ):
                continue
            self.possible_words.extend(
                self.find_words(
                    dawg,
                    (0, self.letters),
                    [],
                    i,
                    [],
                )
            )
        self.board = list(zip(*self.board))
        for i in range(15):
            if (
                self.board[i].count("-") == 15
                and (i == 0 or self.board[i - 1].count("-") == 15)
                and (i == 14 or self.board[i + 1].count("-") == 15)
            ):
                continue
            self.possible_words.extend(
                self.find_words(dawg, (0, self.letters), [], i, [], orientation=1)
            )
        self.board = list(zip(*self.board))
        self.best_word = max(self.possible_words, key=itemgetter(3))
        print(self.best_word)


def main():
    game = Game()
    game.find_all_words()


if __name__ == "__main__":
    main()
