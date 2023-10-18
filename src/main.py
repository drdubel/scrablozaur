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
        if node.is_terminal:
            return node
        return False
    if word[i] in node.children:
        return get_word_end(node.children[word[i]], word, i + 1)
    return False


def count_words(
    node: Node,
    pattern: dict,
    av_letters: str,
    words: list,
    word: str = "",
    nwords: int = 0,
    i: int = 0,
):
    if i + 1 in pattern:
        for letter, child in node.children.items():
            if letter in av_letters:
                new_node = get_word_end(child, pattern[i + 1])
                if new_node:
                    nwords, words = count_words(
                        new_node,
                        pattern,
                        av_letters.replace(letter, "", 1),
                        words,
                        word + letter + pattern[i + 1],
                        nwords,
                        i + 1 + len(pattern[i + 1]),
                    )
    else:
        if node.is_terminal and i >= max(pattern.keys()):
            nwords += 1
            words.append(word)
        for letter, child in node.children.items():
            if letter in av_letters:
                nwords, words = count_words(
                    child,
                    pattern,
                    av_letters.replace(letter, "", 1),
                    words,
                    word + letter,
                    nwords,
                    i + 1,
                )
    return nwords, words


def main():
    dawg = pickle.loads(open("words/dawg.pickle", "rb").read())
    print(count_words(dawg, {2: "tak"}, "takakakta"))


if __name__ == "__main__":
    main()
