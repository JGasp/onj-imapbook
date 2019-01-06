from typing import List

from model.dto.Mark import Mark


class Graph:
    def __init__(self, nodes):
        self.nodes: List[Node] = nodes

        self.similarity = {Mark.M0: {}, Mark.M05: {}, Mark.M1: {}}
        self.avg_similarity = {Mark.M0: 0, Mark.M05: 0, Mark.M1: 0}

        self.node_index = {}
        for i, n in enumerate(nodes):
            if n.word not in self.node_index:
                self.node_index[n.word] = []
            self.node_index[n.word].append(i)


class Node:
    def __init__(self, word: str, prev_link, next_link):
        self.word = word

        self.prev: Link = prev_link
        self.next: Link = next_link

    def __str__(self):
        return self.word


class Link:
    def __init__(self):
        self.words = []

    def add(self, word: str):
        self.words.append(word)

    def __str__(self):
        return ",".join(self.words)
