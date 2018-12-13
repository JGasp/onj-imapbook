from qaModel.dataStructure.text import Text
from qaModel.dataStructure.question import Question
from qaModel.dataStructure.answer import Answer
from qaModel.dataStructure.node import Node
from qaModel.dataStructure.node import Link

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from typing import Dict

import re


class QAModel:
    def __init__(self):
        self.text = None
        self.questions: Dict[Question] = {}

        self.Stemmer = SnowballStemmer("english")
        self.Stop_words = set(stopwords.words('english'))

    def set_text(self, raw_text: str):
        graph = self.build_graph(raw_text)
        self.text = Text(raw_text, graph)

    def set_questions(self, questions):
        for q in questions:
            answers = questions[q]

            q_graph = self.build_graph(q)
            q_data = Question(q, q_graph)

            for a in answers:
                q_data.answers.append(Answer(a["res"], self.build_graph(a["res"]), a["gr"], a["ar"], a["fr"]))

            self.questions[q] = q_data

    def build_graph(self, raw_text: str):
        tokenize_text = self.steam(self.tokenize(raw_text))

        prev = Link()

        graph = []
        for word in tokenize_text:
            if word in self.Stop_words:
                prev.add(word)
            else:
                n = Link()
                node = Node(word, prev, n)
                prev = n
                graph.append(node)

        return graph

    def build(self):
        print("asd")

    def make_prediction(self, question, answer):
        return 1

    def tokenize(self, raw_text):
        tokens = []

        for word in nltk.sent_tokenize(raw_text):
            for token in nltk.word_tokenize(word):
                tokens.append(token.lower())

        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        return filtered_tokens

    def steam(self, tokens):
        stems = []

        for token in tokens:
            stems.append(self.Stemmer.stem(token))

        return stems