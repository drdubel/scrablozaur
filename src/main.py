# TODO
# clever alternative move

import pickle
import re
from multiprocessing import Pool
from operator import itemgetter
from random import shuffle, sample

from tqdm import tqdm

from src.create_dawg import Node
from src.data import bonuses, letter_points, tile_bag

with open("words/dawg.pickle", "rb") as f:
    dawg = pickle.loads(f.read())


class NoPossibleWords(Exception):
    pass


class Game:
    def __init__(self, board=[["-" for _ in range(15)] for _ in range(15)]):
        self.board = board
        self.tile_bag = tile_bag.copy()
        self.end = False

    def __repr__(self) -> str:
        pretty_board = "\n".join([" ".join(x) for x in self.board])
        return pretty_board.upper()

    def insert_word(self, orientation, pos, word):
        if orientation:
            self.board = list(list(x) for x in zip(*self.board))
            pos = pos[::-1]
        self.board[pos[0]][pos[1] : pos[1] + len(word)] = word
        if orientation:
            self.board = list(list(x) for x in zip(*self.board))

    def give_new_letters(self, letters):
        new_letters = sample(
            self.tile_bag,
            min(len(self.tile_bag), 7 - len(letters)),
        )
        for letter in new_letters:
            self.tile_bag.remove(letter)
        return new_letters


class Player:
    def __init__(self, game):
        self.letters = []
        self.score = 0
        self.game = game
        self.get_new_letters()

    def exchange_letters(self, n):
        shuffle(self.letters)
        for _ in range(min(len(letters), n)):
            letter = self.letters.pop()
            self.tile_bag.append(letter)
        self.get_new_letters()

    def get_new_letters(self):
        self.letters.extend(self.game.give_new_letters(self.letters))

    def validate_word(self, node: Node, word: str, x: int = 0):
        if x == len(word):
            if node.is_terminal:
                return word
            return False
        if word[x] in node.children:
            return self.validate_word(node.children[word[x]], word, x + 1)
        return False

    def check_crossword(self, column, new_letter, y, x) -> tuple:
        score = 0
        for result in re.finditer(r"\w+", column):
            if result.start() <= y and y <= result.end():
                if self.validate_word(dawg, column[result.start() : result.end()]):
                    score += sum(
                        map(
                            lambda letter: letter_points[letter],
                            column[result.start() : result.end()],
                        )
                    )
                    if (y, x) in bonuses:
                        score += letter_points[new_letter] * (bonuses[(y, x)][0] - 1)
                        score *= bonuses[(y, x)][1]
                    return True, score
        return False, 0

    def find_first_words(
        self,
        node: Node,
        av_letters: tuple,
        best_word: tuple,
        can_be: bool = False,
        word: str = "",
        points: tuple = (0, 1),
        x: int = 0,
    ) -> tuple:
        if node.is_terminal and can_be:
            pos = (7, x - len(word))
            score = points[0] * points[1]

            if av_letters[0] == 7:
                score += 50

            if best_word:
                best_word = max(
                    best_word, (pos, word, score, av_letters[1]), key=itemgetter(2)
                )
            else:
                best_word = (pos, word, score, av_letters[1])

        if x == 15:
            return best_word

        for letter, child in node.children.items():
            if letter not in av_letters[1]:
                continue

            if x == 7:
                can_be = True

            bonus = bonuses[(7, x)] if (7, x) in bonuses else (1, 1)

            new_av_letters = av_letters[1].copy()
            new_av_letters.remove(letter)
            best_word = self.find_first_words(
                child,
                (av_letters[0] + 1, new_av_letters),
                best_word,
                can_be=can_be,
                word=word + letter,
                points=(
                    points[0] + letter_points[letter] * bonus[0],
                    points[1] * bonus[1],
                ),
                x=x + 1,
            )

        if not word:
            best_word = self.find_first_words(
                node,
                av_letters,
                best_word,
                x=x + 1,
            )

        return best_word

    def find_words(
        self,
        node: Node,
        av_letters: tuple,
        best_word: tuple,
        y: int,
        orientation: int = 0,
        word: str = "",
        can_be: tuple = (False, False),
        points: tuple = (0, 0, 1),
        x: int = 0,
    ) -> tuple:
        if (
            node.is_terminal
            and can_be[0]
            and can_be[1]
            and (
                x == 15
                or (x <= 14 and self.game.board[y][x] == "-")
                or (
                    x <= 13
                    and self.game.board[y][x] == "-"
                    and self.game.board[y][x + 1] == "-"
                )
            )
        ):
            pos = (y, x - len(word))
            score = points[0] * points[2] + points[1]
            if orientation:
                pos = pos[::-1]

            if av_letters[0] == 7:
                score += 50

            if best_word:
                best_word = max(
                    best_word,
                    (orientation, pos, word, score, av_letters[1]),
                    key=itemgetter(3),
                )
            else:
                best_word = (orientation, pos, word, score, av_letters[1])

        if x == 15:
            return best_word

        if (
            self.game.board[y][x] != "-"
            and self.game.board[y][x] in node.children
            and not (not word and self.game.board[y][x] != "-")
        ):
            best_word = self.find_words(
                node.children[self.game.board[y][x]],
                av_letters,
                best_word,
                y,
                orientation=orientation,
                word=word + self.game.board[y][x],
                can_be=(True, can_be[1]),
                points=(
                    points[0] + letter_points[self.game.board[y][x]],
                    points[1],
                    points[2],
                ),
                x=x + 1,
            )

        elif self.game.board[y][x] == "-" and not (
            not word and self.game.board[y][x - 1] != "-"
        ):
            for letter, child in node.children.items():
                new_points = 0
                new_addit_word = False
                if letter not in av_letters[1]:
                    continue

                if (y > 0 and self.game.board[y - 1][x] != "-") or (
                    y < 14 and self.game.board[y + 1][x] != "-"
                ):
                    column = list(list(zip(*self.game.board))[x])
                    column[y] = letter
                    new_addit_word, new_points = self.check_crossword(
                        "".join(column), letter, y, x
                    )
                    if not new_addit_word:
                        continue

                bonus = bonuses[(y, x)] if (y, x) in bonuses else (1, 1)

                new_av_letters = av_letters[1].copy()
                new_av_letters.remove(letter)
                best_word = self.find_words(
                    child,
                    (av_letters[0] + 1, new_av_letters),
                    best_word,
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
            best_word = self.find_words(
                node,
                av_letters,
                best_word,
                y,
                orientation=orientation,
                x=x + 1,
            )

        return best_word

    def place_best_first_word(self):
        best_word = self.find_first_words(dawg, (0, self.letters), [])

        if not best_word:
            raise NoPossibleWords()

        self.game.insert_word(0, *best_word[:2])
        self.letters = best_word[3]
        self.score += best_word[2]
        self.get_new_letters()
        return best_word

    def place_best_word(self):
        best_word = (0, (0, 0), "", 0, "")
        for i in range(15):
            if (
                self.game.board[i].count("-") == 15
                and (i == 0 or self.game.board[i - 1].count("-") == 15)
                and (i == 14 or self.game.board[i + 1].count("-") == 15)
            ):
                continue
            possible_word = self.find_words(
                dawg,
                (0, self.letters),
                [],
                i,
            )
            if possible_word:
                best_word = max(best_word, possible_word, key=itemgetter(3))

        self.game.board = list(list(x) for x in zip(*self.game.board))
        for i in range(15):
            if (
                self.game.board[i].count("-") == 15
                and (i == 0 or self.game.board[i - 1].count("-") == 15)
                and (i == 14 or self.game.board[i + 1].count("-") == 15)
            ):
                continue
            possible_word = self.find_words(
                dawg, (0, self.letters), [], i, orientation=1
            )
            if possible_word:
                best_word = max(best_word, possible_word, key=itemgetter(3))
        self.game.board = list(list(x) for x in zip(*self.game.board))

        if not best_word[3]:
            raise NoPossibleWords()
        self.game.insert_word(*best_word[:3])
        self.letters = best_word[4]
        self.score += best_word[3]
        self.get_new_letters()
        return best_word

    def move(self, first=False):
        try:
            if first:
                word = self.place_best_first_word()
            else:
                word = self.place_best_word()
            return word

        except NoPossibleWords:
            if self.game.tile_bag:
                self.exchange_letters(2)
            else:
                self.game.end = True
                return False


def play_game(i=0):
    game = Game()
    player1 = Player(game)
    player2 = Player(game)
    player2.move(first=True)
    while not game.end:
        player1.move()
        print("Player1 move: ", player1.move())
        if game.end:
            break

        player2.move()
        print("Player2 move: ", player2.move(), "\n")
        print(game)

    for letter in player1.letters:
        player1.score -= letter_points[letter]
    for letter in player2.letters:
        player2.score -= letter_points[letter]

    if player1.score > player2.score:
        print("Player1 Won!")
    else:
        print("Player2 Won!")

    return player1.score, player2.score


def main():
    print(play_game())


if __name__ == "__main__":
    main()
