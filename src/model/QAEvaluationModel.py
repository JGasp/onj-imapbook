from model.dto.Question import Question
from model.dto.Answer import Answer, Mark
from model.dto.Graph import Node
from model.dto.Graph import Link
from model.dto.Graph import Graph

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from typing import Dict

import re


class QAEvaluationModel:
    def __init__(self, word_len=5, stop_word_len=1, len_lambda=0.2):
        self.questions: Dict[str, Question] = {}

        self.word_len = word_len
        self.stop_word_len = stop_word_len
        self.len_lambda = len_lambda

        self.find_first_similar = True

        self.Stemmer = SnowballStemmer("english")
        self.Stop_words = set(stopwords.words('english'))

    def set_questions(self, questions: Dict[str, Question]):
        self.questions = {}

        for key, q in questions.items():
            q_copy = q.copy()
            q_copy.build_graph(self.build_graph)

            for a in q.answers:
                a_copy = a.copy()
                a_copy.build_graph(self.build_graph)
                q_copy.add_answer(a_copy)

            self.questions[key] = q_copy

    def build_graph(self, raw_text: str):
        tokenize_text = self.steam(self.tokenize(raw_text))

        prev = Link()
        nodes = []
        for word in tokenize_text:
            if word in self.Stop_words:
                prev.add(word)
            else:
                n = Link()
                node = Node(word, prev, n)
                prev = n
                nodes.append(node)

        return Graph(nodes)

    def graph_contains(self, val: str, graph):
        for n in graph:
            if n.word == val:
                return True
        return False

    def get_graph_direction(self, ni1, ni2):
        if ni1 > ni2:
            return ni2, ni1
        return ni1, ni2

    def calculate_node_distance(self, graph, start, end):
        distance = 1
        for i in range(start, end):
            n = graph.nodes[i]
            distance += len(n.next.words) * self.stop_word_len

            if i != start:
                distance += self.word_len

        return distance

    def calculate_answer_similarity(self, answer: Answer, graph: Graph):
        if answer.text_graph.nodes == 0:
            print("break")

        similarity = 0
        sim_values = graph.similarity[answer.final_mark]

        prev_word_index = []
        stop_words = {}
        for an in answer.text_graph.nodes:
            word_index = graph.get_nodes_with_word(an.word)

            # Increase node score
            if len(word_index) > 0:
                for wi in word_index:
                    n = graph.nodes[wi]
                    if n in sim_values:
                        similarity += sim_values[n]

                if len(prev_word_index) > 0:
                    # Normalize score based on number of paths
                    paths = len(prev_word_index) * len(word_index)
                    for pwi in prev_word_index:
                        for wi in word_index:
                            start, end = self.get_graph_direction(pwi, wi)
                            distance = self.calculate_node_distance(graph, start, end)

                            for i in range(start, end):
                                n = graph.nodes[i]

                                stop_words_sim = 0  # TODO: Evaluate stop_word similarity based on multiple matching
                                for w in n.next.words:
                                    if w in stop_words:
                                        stop_words_sim += sim_values[n.next]

                                if self.len_lambda > 0:
                                    stop_words_sim = stop_words_sim / (self.len_lambda * distance)

                                similarity += stop_words_sim / paths

            if not(self.find_first_similar and len(word_index) == 0):
                prev_word_index = word_index
                stop_words = {}

            for w in an.next.words:
                stop_words[w] = True

        # Normalize similarity based on answer length
        similarity /= len(answer.text_graph.nodes)

        return similarity

    def update_context_similarity(self, answer: Answer, question: Question):  # TODO jump reference
        graph = question.context_graph
        sim_values = graph.similarity[answer.final_mark]

        prev_word_index = []
        stop_words = {}

        for an in answer.text_graph.nodes:
            node_index = graph.get_nodes_with_word(an.word)

            if len(node_index) > 0:
                # Increase node score TODO: decrease with number of multiple appearances
                for wi in node_index:
                    n = graph.nodes[wi]
                    if n not in sim_values:
                        sim_values[n] = 0
                    sim_values[n] += self.word_len

                if len(prev_word_index) > 0:
                    # Normalize score based on number of paths
                    paths = len(prev_word_index) * len(node_index)
                    for pwi in prev_word_index:
                        for wi in node_index:
                            start, end = self.get_graph_direction(pwi, wi)
                            distance = self.calculate_node_distance(graph, start, end)

                            for i in range(start, end):
                                n = graph.nodes[i]

                                stop_word_similarity = 0.0
                                for w in n.next.words:
                                    if w in stop_words:
                                        stop_word_similarity += self.stop_word_len

                                if stop_word_similarity > 0:
                                    # Penalize long references
                                    if self.len_lambda > 0:
                                        stop_word_similarity = stop_word_similarity / (self.len_lambda * distance)

                                    # Normalize value based on number of paths between different matching nodes
                                    stop_word_similarity /= paths

                                    if n.next not in sim_values:
                                        sim_values[n.next] = 0
                                    sim_values[n.next] += stop_word_similarity

            if not (self.find_first_similar and len(node_index) == 0):  # Find first similar node (skip not found ones)
                prev_word_index = node_index
                stop_words = {}

            for w in an.next.words:  # TODO: Evaluate usability of keeping stop_words from not found node
                stop_words[w] = True

    def build_graph_similarity(self):
        for key, q in self.questions.items():
            for a in q.answers:
                self.update_context_similarity(a, q)

        for key, q in self.questions.items():
            graph = q.context_graph

            avg_similarity = {Mark.M0: 0, Mark.M05: 0, Mark.M1: 0}
            count = {Mark.M0: 0, Mark.M05: 0, Mark.M1: 0}

            for a in q.answers:
                sim = self.calculate_answer_similarity(a, graph)
                avg_similarity[a.final_mark] += sim
                count[a.final_mark] += 1

            for m in avg_similarity:
                graph.avg_similarity[m] /= count[m]

    def build(self):
        self.build_graph_similarity()

    def make_prediction(self, question, answer):

        q = self.questions[question]

        a = Answer(answer)
        a.build_graph(self.build_graph)

        answers_sim = {Mark.M0: 0, Mark.M05: 0, Mark.M1: 0}
        for m in [Mark.M0, Mark.M05, Mark.M1]:
            for ans in q.answers_by_mark[m]:
                answers_sim[m] += self.calculate_answer_similarity(a, ans.text_graph)
            answers_sim[m] /= len(q.answers_by_mark[m])

        # TODO add answer similarity to other answers (cosine distance + graph coverage)
        # (avg_sim - pred_sim) / avg_sim => lowest value represent the most similar cluster
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
