from dataclasses import dataclass

from src.consts import CODE


class TrieNode:
    next_id = 0

    def __init__(self):
        self.is_terminal = False
        self.id = TrieNode.next_id
        TrieNode.next_id += 1
        self.children = {}


@dataclass
class DoubleArrayNode:
    base: int = 0
    check: str = ""


def build_trie(words: list) -> TrieNode:
    root = TrieNode()

    for word in words:
        node = root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()

            node = node.children[char]
        node.is_terminal = True

    return root


def build_double_array(node: TrieNode, base=1):
    nodes = [DoubleArrayNode(check=char) for char in node.children.keys()]

    for char, child_node in node.children.items():
        pass

    return nodes


def main():
    words = open("words/slowa2.txt", "r").read().split()
    trie = build_trie(words)
    print(trie)


if __name__ == "__main__":
    main()
