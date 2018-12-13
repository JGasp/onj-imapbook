from typing import List
from qaModel.dataStructure.answer import Answer


class Question:
    def __init__(self, value, graph):
        self.value: str = value
        self.answers: List[Answer] = []
        self.graph = graph

    def add_answer(self, answer: Answer):
        self.answers.append(answer)
