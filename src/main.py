import pickle
from pprint import pprint

from create_dawg import Node

dawg = pickle.loads(open("words/dawg.pickle", "rb").read())


def get_word_end(node: Node, word: str, i: int = 0):
    if i == len(word):
        return node
    if word[i] in node.children:
        return get_word_end(node.children[word[i]], word, i + 1)
    return False


def find_words_from(
    node: Node,
    board: tuple,
    av_letters: str,
    words: list,
    orientation: int,
    line: int,
    addit_words: list,
    word: str = "",
    nwords: int = 0,
    can_be: bool = False,
    i: int = 0,
):
    if i == 15:
        if node.is_terminal and can_be:
            nwords += 1
            words.append(((line, i - len(word)), word, addit_words))
        return nwords, words
    wrong_letters = ""
    new_addit_words = []
    above = list(
        filter(lambda x: x[0] + len(x[1]) == line, board[orientation ^ 1][i].items())
    )
    below = (
        board[orientation ^ 1][i][line + 1]
        if line + 1 in board[orientation ^ 1][i]
        else False
    )

    if i in board[orientation][line]:
        pass

    elif above and below:
        for letter in node.children.keys():
            if letter not in av_letters:
                continue

            new_node = get_word_end(dawg, above[0][1] + letter + below)
            if not new_node:
                wrong_letters += letter
            elif not new_node.is_terminal:
                wrong_letters += letter
            else:
                new_addit_words.append(
                    ((line, i), above[0][1] + letter + below, letter)
                )

    elif above:
        for letter in node.children.keys():
            if letter not in av_letters:
                continue

            new_node = get_word_end(dawg, above[0][1] + letter)
            if not new_node:
                wrong_letters += letter
            elif not new_node.is_terminal:
                wrong_letters += letter
            else:
                new_addit_words.append(((line, i), above[0][1] + letter, letter))

    elif below:
        for letter in node.children.keys():
            if letter not in av_letters:
                continue

            new_node = get_word_end(dawg, letter + below)
            if not new_node:
                wrong_letters += letter
            elif not new_node.is_terminal:
                wrong_letters += letter
            else:
                new_addit_words.append(((line, i), letter + below, letter))

    if i in board[orientation][line]:
        new_node = get_word_end(node, board[orientation][line][i])
        if new_node:
            nwords, words = find_words_from(
                new_node,
                board,
                av_letters,
                words,
                orientation,
                line,
                addit_words + new_addit_words,
                word=word + board[orientation][line][i],
                nwords=nwords,
                can_be=True,
                i=i + len(board[orientation][line][i]),
            )

        if not word:
            nwords, words = find_words_from(
                node,
                board,
                av_letters,
                words,
                orientation,
                line,
                addit_words + new_addit_words,
                word=word,
                nwords=nwords,
                i=i + 1 + len(board[orientation][line][i]),
            )

    elif i + 1 in board[orientation][line]:
        for letter, child in node.children.items():
            if letter not in av_letters or letter in wrong_letters:
                continue

            new_node = get_word_end(child, board[orientation][line][i + 1])
            if not new_node:
                continue
            nwords, words = find_words_from(
                new_node,
                board,
                av_letters.replace(letter, "", 1),
                words,
                orientation,
                line,
                addit_words + new_addit_words,
                word=word + letter + board[orientation][line][i + 1],
                nwords=nwords,
                can_be=True,
                i=i + 1 + len(board[orientation][line][i + 1]),
            )

        if not word:
            nwords, words = find_words_from(
                node,
                board,
                av_letters,
                words,
                orientation,
                line,
                addit_words + new_addit_words,
                word=word,
                nwords=nwords,
                i=i + 1,
            )

    else:
        if node.is_terminal and can_be:
            nwords += 1
            words.append(((line, i - len(word)), word, addit_words))
        for letter, child in node.children.items():
            if letter not in av_letters or letter in wrong_letters:
                continue

            nwords, words = find_words_from(
                child,
                board,
                av_letters.replace(letter, "", 1),
                words,
                orientation,
                line,
                addit_words + new_addit_words,
                word=word + letter,
                nwords=nwords,
                can_be=can_be,
                i=i + 1,
            )

        if not word:
            nwords, words = find_words_from(
                node,
                board,
                av_letters,
                words,
                orientation,
                line,
                addit_words + new_addit_words,
                word=word,
                nwords=nwords,
                can_be=can_be,
                i=i + 1,
            )

    return nwords, words


def main():
    for i in range(15):
        pprint(
            find_words_from(
                dawg,
                (
                    (
                        {},
                        {},
                        {},
                        {},
                        {4: "t", 6: "m"},
                        {4: "c", 6: "a"},
                        {4: "hamulec"},
                        {4: "ó", 6: "a", 8: "o", 10: "i"},
                        {1: "gwar", 8: "d", 10: "a"},
                        {4: "z", 7: "mydło"},
                        {10: "o"},
                        {},
                        {},
                        {},
                        {},
                    ),
                    (
                        {},
                        {8: "g"},
                        {8: "w"},
                        {8: "a"},
                        {4: "tchórz"},
                        {6: "a"},
                        {4: "mama"},
                        {6: "u", 9: "m"},
                        {6: "lody"},
                        {6: "e", 9: "d"},
                        {6: "ciało"},
                        {9: "m"},
                        {},
                        {},
                        {},
                    ),
                ),
                "qwertyu",
                [],
                0,
                i,
                [],
            )
        )


if __name__ == "__main__":
    main()
