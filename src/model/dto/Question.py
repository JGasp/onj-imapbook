from typing import List, Dict

from model.dto import Graph
from model.dto.Answer import Answer
from model.dto.Answer import Mark


class Question:
    def __init__(self, raw_text, context_text=None, text_graph=None, context_graph=None):
        self.raw_text: str = raw_text
        self.context_text = context_text

        self.text_graph: Graph = text_graph
        self.context_graph: Graph = context_graph

        self.answers: List[Answer] = []
        self.answers_by_mark: Dict[Mark, List[Answer]] = {Mark.M0: [], Mark.M05: [], Mark.M1: []}

        self.linear_regression = None

    def add_answer(self, answer: Answer):
        self.answers.append(answer)
        self.answers_by_mark[answer.final_mark].append(answer)

    def build_graph(self, fun_graph_build):
        self.text_graph = fun_graph_build(self.raw_text)
        self.context_graph = fun_graph_build(self.context_text)

    def copy(self):
        # answer_copy = []
        # answer_by_mark_copy = {"0": [], "0.5": [], "1": []}
        #
        # for a in self.answers:
        #     a_copy = a.copy()
        #     answer_copy.append(a_copy)
        #     answer_by_mark_copy[a_copy.final_mark].append(a_copy)

        question_copy = Question(self.raw_text, self.context_text)
        # question_copy.answers = answer_copy
        # question_copy.answers_by_mark = answer_by_mark_copy

        return question_copy
