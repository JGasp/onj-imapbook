
class Graph:
    def __init__(self, nodes):
        self.nodes = nodes

        self.similarity_0 = {}
        self.avg_similarity_0 = 0

        self.similarity_05 = {}
        self.avg_similarity_05 = 0

        self.similarity_1 = {}
        self.avg_similarity_1 = 0

        self.node_index = {}
        for i, n in enumerate(nodes):
            if n.word not in self.node_index:
                self.node_index[n.word] = []
            self.node_index[n.word].append(i)

    def get_nodes_with_word(self, word):
        if word in self.node_index:
            return self.node_index[word]
        else:
            return []


class Node:
    def __init__(self, word: str, prev_link, next_link):
        self.word = word

        self.prev: Link = prev_link
        self.next: Link = next_link


class Link:
    def __init__(self):
        self.words = []

    def add(self, word: str):
        self.words.append(word)
