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


def count_words(
    node: Node,
    substr: str,
    av_letters: str,
    words: list = [],
    word: str = "",
    nwords: int = 0,
):
    if node.is_terminal and substr in word:
        nwords += 1
        words.append(word)
    for letter, child in node.children.items():
        if letter in av_letters:
            nwords, words = count_words(
                child,
                substr,
                av_letters.replace(letter, "", 1),
                words,
                word + letter,
                nwords,
            )
    return nwords, words


def main():
    dawg = pickle.loads(open("words/dawg.pickle", "rb").read())
    print(count_words(dawg, "tak", "takakakta"))


if __name__ == "__main__":
    main()
