import pickle


class Node:
    next_id = 0

    def __init__(self):
        self.is_terminal = False
        self.id = Node.next_id
        Node.next_id += 1
        self.children = {}

    def __repr__(self):
        out = []
        if self.is_terminal:
            out.append("1")
        else:
            out.append("0")
        for key, val in self.children.items():
            out.append(key)
            out.append(str(val.id))
        return "_".join(out)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()


def get_word_end(node: Node, word: str, i: int = 0):
    if i == len(word):
        return node
    if word[i] in node.children:
        return get_word_end(node.children[word[i]], word, i + 1)
    return False


def find_words_from(
    node: Node,
    pattern: dict,
    av_letters: str,
    words: list,
    word: str = "",
    nwords: int = 0,
    liczone: bool = False,
    i: int = 0,
):
    if i == 15:
        if node.is_terminal and liczone:
            nwords += 1
            words.append((i - len(word), word))
        return nwords, words
    if i in pattern[1]:
        new_node = get_word_end(node, pattern[1][i])
        if new_node:
            nwords, words = find_words_from(
                new_node,
                pattern,
                av_letters,
                words=words,
                word=word + pattern[1][i],
                nwords=nwords,
                liczone=True,
                i=i + len(pattern[1][i]),
            )
        if not word:
            nwords, words = find_words_from(
                node,
                pattern,
                av_letters,
                words=words,
                word=word,
                nwords=nwords,
                i=i + 1 + len(pattern[1][i]),
            )
    elif i + 1 in pattern[1]:
        for letter, child in node.children.items():
            if letter not in av_letters:
                continue

            new_node = get_word_end(child, pattern[1][i + 1])
            if not new_node:
                continue
            nwords, words = find_words_from(
                new_node,
                pattern,
                av_letters.replace(letter, "", 1),
                words=words,
                word=word + letter + pattern[1][i + 1],
                nwords=nwords,
                liczone=True,
                i=i + 1 + len(pattern[1][i + 1]),
            )
        if not word:
            nwords, words = find_words_from(
                node,
                pattern,
                av_letters,
                words=words,
                word=word,
                nwords=nwords,
                i=i + 1,
            )

    else:
        if node.is_terminal and liczone:
            nwords += 1
            words.append((i - len(word), word))
        for letter, child in node.children.items():
            if letter not in av_letters:
                continue

            nwords, words = find_words_from(
                child,
                pattern,
                av_letters.replace(letter, "", 1),
                words=words,
                word=word + letter,
                nwords=nwords,
                liczone=liczone,
                i=i + 1,
            )
        if not word:
            nwords, words = find_words_from(
                node,
                pattern,
                av_letters,
                words=words,
                word=word,
                nwords=nwords,
                liczone=liczone,
                i=i + 1,
            )
    return nwords, words


def main():
    dawg = pickle.loads(open("words/dawg.pickle", "rb").read())
    print(
        find_words_from(
            dawg,
            [{}, {0: "t", 5: "w", 10: "a"}, {}],
            "abcdefg",
            [],
        )
    )


if __name__ == "__main__":
    main()
