# TODO
# alternative move

import pickle
import re
from operator import itemgetter
from random import sample

from tqdm import tqdm

from src.create_dawg import Node
from src.data import bonuses, letter_points, tile_bag

with open("words/dawg.pickle", "rb") as f:
    dawg = pickle.loads(f.read())


class NoPossibleWordError(Exception):
    pass


class Game:
    def __init__(self, board=[["-" for _ in range(15)] for _ in range(15)]):
        self.letters = ""
        self.board = board
        self.possible_words = []
        self.best_word = ()
        self.tile_bag = tile_bag.copy()
        self.score = 0

    def __str__(self) -> str:
        pretty_board = "\n".join([" ".join(x) for x in self.board])
        return pretty_board.upper()

    def exchange_letters(self, n):
        letters_to_rem = sample(self.letters, n)
        self.tile_bag.extend(letters_to_rem)
        new_letters = sample(self.tile_bag, n)

        for letter in letters_to_rem:
            self.letters = self.letters.replace(letter, "", 1)

        for letter in new_letters:
            self.tile_bag.remove(letter)

        self.letters += "".join(new_letters)

    def get_new_letters(self):
        new_letters = sample(
            self.tile_bag,
            min(len(self.tile_bag), 7 - len(self.letters)),
        )
        for letter in new_letters:
            self.tile_bag.remove(letter)
        return "".join(new_letters)

    def insert_word(self, orientation, pos, word):
        if orientation:
            self.board = list(list(x) for x in zip(*self.board))
            pos = pos[::-1]
        self.board[pos[0]][pos[1] : pos[1] + len(word)] = word
        if orientation:
            self.board = list(list(x) for x in zip(*self.board))

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

    def find_first_words(
        self,
        node: Node,
        av_letters: tuple,
        words: list,
        can_be=False,
        word: str = "",
        points: tuple = (0, 1),
        x: int = 0,
    ) -> list:
        if node.is_terminal and can_be:
            pos = (7, x - len(word))
            score = points[0] * points[1]

            if av_letters[0] == 7:
                score += 50

            words.append((pos, word, score, av_letters[1]))

        if x == 15:
            return words

        for letter, child in node.children.items():
            if letter not in av_letters[1]:
                continue

            if x == 7:
                can_be = True

            bonus = bonuses[(7, x)] if (7, x) in bonuses else (1, 1)

            words = self.find_first_words(
                child,
                (av_letters[0] + 1, av_letters[1].replace(letter, "", 1)),
                words,
                can_be=can_be,
                word=word + letter,
                points=(
                    points[0] + letter_points[letter] * bonus[0],
                    points[1] * bonus[1],
                ),
                x=x + 1,
            )

        if not word:
            words = self.find_first_words(
                node,
                av_letters,
                words,
                x=x + 1,
            )

        return words

    def find_words(
        self,
        node: Node,
        av_letters: tuple,
        words: list,
        y: int,
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

            words.append((orientation, pos, word, score, av_letters[1]))

        if x == 15:
            return words

        if (
            self.board[y][x] != "-"
            and self.board[y][x] in node.children
            and not (not word and self.board[y][x] != "-")
        ):
            words = self.find_words(
                node.children[self.board[y][x]],
                av_letters,
                words,
                y,
                orientation=orientation,
                word=word + self.board[y][x],
                can_be=(True, can_be[1]),
                points=(
                    points[0] + letter_points[self.board[y][x]],
                    points[1],
                    points[2],
                ),
                x=x + 1,
            )

        elif self.board[y][x] == "-" and not (not word and self.board[y][x - 1] != "-"):
            for letter, child in node.children.items():
                new_points = 0
                new_addit_word = ""
                if letter not in av_letters[1]:
                    continue

                if (y > 0 and self.board[y - 1][x] != "-") or (
                    y < 14 and self.board[y + 1][x] != "-"
                ):
                    column = list(list(list(x) for x in zip(*self.board))[x])
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
                orientation=orientation,
                x=x + 1,
            )

        return words

    def place_best_first_word(self):
        self.possible_words = self.find_first_words(dawg, (0, self.letters), [])

        if not self.possible_words:
            self.exchange_letters(7)
            self.place_best_first_word()

        self.best_word = max(self.possible_words, key=itemgetter(2))
        self.insert_word(0, *self.best_word[:2])
        self.letters = self.best_word[3]
        self.score += self.best_word[2]
        return self.best_word

    def place_best_word(self):
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
                )
            )

        self.board = list(list(x) for x in zip(*self.board))
        for i in range(15):
            if (
                self.board[i].count("-") == 15
                and (i == 0 or self.board[i - 1].count("-") == 15)
                and (i == 14 or self.board[i + 1].count("-") == 15)
            ):
                continue
            self.possible_words.extend(
                self.find_words(dawg, (0, self.letters), [], i, orientation=1)
            )

        if not self.possible_words:
            if not self.tile_bag:
                return False
            self.exchange_letters(7)
            self.place_best_first_word()

        self.board = list(list(x) for x in zip(*self.board))
        self.best_word = max(self.possible_words, key=itemgetter(3))
        self.insert_word(*self.best_word[:3])
        self.letters = self.best_word[4]
        self.score += self.best_word[3]
        return self.best_word


def play_game():
    game = Game()
    game.letters += game.get_new_letters()
    game.place_best_first_word()
    # print(game.letters)
    # print(game.place_best_first_word())
    # print(game.score)
    # print(game)
    while True:
        try:
            # print(game.tile_bag)
            game.letters += game.get_new_letters()
            game.place_best_word()
            # print(game.letters)
            # print(game.place_best_word())
            # print(game.score)
            # print(game)
        except NoPossibleWordError:
            if not game.tile_bag:
                # print("KONIEC!")
                break
            litery = game.exchange_letters(7)
            print(litery)
            game.letters = litery
    return game.score


def main():
    n = 1000
    points = 0
    for _ in tqdm(range(n)):
        points += play_game()
    print(points / n)


if __name__ == "__main__":
    main()
