
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
