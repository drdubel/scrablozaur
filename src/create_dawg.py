import os
import pickle

from tqdm import tqdm


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


def get_pref_len(prev_word, word):
    pref_len = 0
    for char1, char2 in zip(prev_word, word):
        if char1 != char2:
            return pref_len
        pref_len += 1
    return pref_len


def minimize(curr_node, pref_len, minimized_nodes, non_minimized_nodes):
    for _ in range(len(non_minimized_nodes) - pref_len):
        parent, letter, child = non_minimized_nodes.pop()
        if child in minimized_nodes:
            parent.children[letter] = minimized_nodes[child]
        else:
            minimized_nodes[child] = child
        curr_node = parent
    return curr_node


def build_dawg(word_list):
    Node.next_id = 0
    root = Node()
    minimized_nodes = {root: root}
    non_minimized_nodes = []
    curr_node = root
    prev_word = ""
    for word in tqdm(word_list):
        pref_len = get_pref_len(prev_word, word)

        if non_minimized_nodes:
            curr_node = minimize(
                curr_node, pref_len, minimized_nodes, non_minimized_nodes
            )

        for letter in word[pref_len:]:
            next_node = Node()
            curr_node.children[letter] = next_node
            non_minimized_nodes.append((curr_node, letter, next_node))
            curr_node = next_node

        curr_node.is_terminal = True
        prev_word = word

    minimize(curr_node, 0, minimized_nodes, non_minimized_nodes)
    print(len(minimized_nodes))
    return root


def main():
    word_list = open("words/slowa2.txt").read().split()
    dawg = build_dawg(word_list)
    with open("words/dawg.pickle", "wb") as f:
        pickle.dump(dawg, f)
    print(round(os.path.getsize("words/dawg.pickle") / 1024**2, 3), "MiB")


if __name__ == "__main__":
    main()
