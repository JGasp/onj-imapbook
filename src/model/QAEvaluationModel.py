from enum import Enum

from sklearn import linear_model
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
import pickle
import os.path


class AnswerSimParam(Enum):
    AVG = "Average"
    MAX = "Maximum"


class QAEvaluationModel:
    def __init__(self, word_len=5, stop_word_len=5, len_lambda=0.2, ans_sim=AnswerSimParam.AVG):
        self.questions: Dict[str, Question] = {}

        self.word_len = word_len
        self.stop_word_len = stop_word_len
        self.len_lambda = len_lambda

        # Regularization
        self.context_base_similarity = self.word_len * 1
        self.answers_base_similarity = self.word_len * 1  # + self.stop_word_len * 1

        self.find_first_similar = True
        self.answer_similarity: AnswerSimParam = ans_sim

        # Functions to override for word2vec
        self.is_link_similarity = self.link_similarity
        self.get_similar_nodes = self.filter_similar_nodes

        self.Stemmer = SnowballStemmer("english")
        self.Stop_words = set(stopwords.words('english'))

    def persist(self, file_name):
        with open(file_name, 'wb') as output:
            pickle.dump(self.questions, output, pickle.HIGHEST_PROTOCOL)

    def load(self, file_name):
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as pickle_file:
                self.questions = pickle.load(pickle_file)

    def add_questions(self, questions: Dict[str, Question]):
        for key, q in questions.items():
            q_copy = q.copy()
            q_copy.build_graph(self.build_graph)

            for a in q.answers:
                a_copy = a.copy()
                a_copy.build_graph(self.build_graph)

                if len(a_copy.text_graph.nodes) > 0:
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

    @staticmethod
    def link_similarity(link: Link, stop_words):
        for w in link.words:
            if w in stop_words:
                return 1
        return 0

    @staticmethod
    def filter_similar_nodes(word, graph: Graph):
        nodes_index = []
        similarity = {}

        if word in graph.node_index:
            index_list = graph.node_index[word]
            for i in index_list:
                nodes_index.append(i)
                similarity[i] = 1

        return nodes_index, similarity

    def calculate_answer_similarity(self, answer: Answer, graph: Graph, mark: Mark):
        similarity = 0
        sim_values = graph.similarity[mark]

        prev_word_index = []
        stop_words = {}
        for an in answer.text_graph.nodes:
            node_index, node_similarity = self.get_similar_nodes(an.word, graph)

            # Increase node score
            if len(node_index) > 0:
                for wi in node_index:
                    n = graph.nodes[wi]
                    if n in sim_values:
                        similarity += sim_values[n] * node_similarity[wi]

                if len(prev_word_index) > 0:
                    # Normalize score based on number of paths
                    paths = len(prev_word_index) * len(node_index)
                    for pwi in prev_word_index:
                        for wi in node_index:
                            start, end = self.get_graph_direction(pwi, wi)
                            distance = self.calculate_node_distance(graph, start, end)

                            for i in range(start, end):
                                n = graph.nodes[i]

                                stop_words_sim = 0
                                if n.next in sim_values:
                                    sim_sw = self.is_link_similarity(n.next, stop_words)
                                    if sim_sw > 0:
                                        stop_words_sim += sim_values[n.next] * sim_sw
                                        break  # Disregard multiple stop_words similarity

                                if self.len_lambda > 0:
                                    stop_words_sim = stop_words_sim / (self.len_lambda * distance)

                                similarity += stop_words_sim / paths

            if not(self.find_first_similar and len(node_index) == 0):
                prev_word_index = node_index
                stop_words = {}

            for w in an.next.words:
                stop_words[w] = True

        if len(answer.text_graph.nodes) == 0:
            return 0.0

        # Normalize similarity based on answer length
        similarity /= len(answer.text_graph.nodes)

        return similarity

    def update_context_similarity(self, answer: Answer, question: Question):
        self.update_graph_similarity(answer, question.context_graph)

    def update_graph_similarity(self, answer: Answer, graph: Graph):
        sim_values = graph.similarity[answer.final_mark]

        prev_word_index = []
        stop_words = {}

        for an in answer.text_graph.nodes:
            node_index, node_similarity = self.get_similar_nodes(an.word, graph)

            if len(node_index) > 0:
                # Increase node score TODO: try decrease with number of multiple appearances
                for wi in node_index:
                    n = graph.nodes[wi]
                    if n not in sim_values:
                        sim_values[n] = 0
                    sim_values[n] += self.word_len * node_similarity[wi]

                if len(prev_word_index) > 0:
                    # Normalize score based on number of paths
                    paths = len(prev_word_index) * len(node_index)
                    for pwi in prev_word_index:
                        for wi in node_index:
                            start, end = self.get_graph_direction(pwi, wi)
                            distance = self.calculate_node_distance(graph, start, end)

                            for i in range(start, end):
                                n = graph.nodes[i]

                                sim_sw = self.is_link_similarity(n.next, stop_words)
                                stop_word_similarity = self.stop_word_len * sim_sw

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

            for w in an.next.words:
                stop_words[w] = True

    def build_context_graph_similarity(self):
        for key, q in self.questions.items():
            for a in q.answers:
                self.update_context_similarity(a, q)

        for key, q in self.questions.items():
            graph = q.context_graph

            sum_similarity = {m: 0 for m in Mark.values()}
            count = {m: 0 for m in Mark.values()}

            for a in q.answers:
                sim = self.calculate_answer_similarity(a, graph, mark=a.final_mark)
                sum_similarity[a.final_mark] += sim
                count[a.final_mark] += 1

            for m in Mark.values():
                graph.avg_similarity[m] = self.context_base_similarity
                if count[m] > 0:
                    graph.avg_similarity[m] += sum_similarity[m] / count[m]

    def build_answers_graph_similarity(self):
        for key, q in self.questions.items():
            for a in q.answers:
                for aup in q.answers_by_mark[a.final_mark]:
                    self.update_graph_similarity(aup, a.text_graph)

        for key, q in self.questions.items():
            for a in q.answers:

                sum_similarity = {m: 0 for m in Mark.values()}
                count = {m: 0 for m in Mark.values()}

                for aup in q.answers_by_mark[a.final_mark]:
                    sim = self.calculate_answer_similarity(aup, a.text_graph, mark=a.final_mark)
                    sum_similarity[a.final_mark] += sim
                    count[a.final_mark] += 1

                for m in Mark.values():
                    a.text_graph.avg_similarity[m] = self.answers_base_similarity
                    if count[m] > 0:
                        a.text_graph.avg_similarity[m] += sum_similarity[m] / count[m]

    def transform_parameters(self, answers_sim, context_sim):
        x = []
        for m in Mark.values():
            x.append(answers_sim[m])
            x.append(context_sim[m])

        return x

    def train_linear_regression(self):
        for k, q in self.questions.items():
            X = []
            y = []

            for a in q.answers:
                answers_sim, context_sim = self.calculate_prediction_ratio(q.raw_text, a.raw_text)

                x_i = self.transform_parameters(answers_sim, context_sim)

                mark = a.final_mark.value.replace(",", ".")
                y_i = float(mark)

                X.append(x_i)
                y.append(y_i)

            # TODO: Handle over fitting
            lm = linear_model.LinearRegression()
            model = lm.fit(X, y)

            q.linear_regression = lm

    def build(self):
        self.build_context_graph_similarity()
        self.build_answers_graph_similarity()
        self.train_linear_regression()

    def calculate_prediction_ratio(self, question, answer):
        q = self.questions[question]

        a = Answer(answer)
        a.build_graph(self.build_graph)

        answers_sim = {m: 0 for m in Mark.values()}
        for m in Mark.values():
            for ans in q.answers_by_mark[m]:

                if answer == 'd':
                    answer = answer

                sim = self.calculate_answer_similarity(a, ans.text_graph, mark=m) / ans.text_graph.avg_similarity[m]

                if self.answer_similarity == AnswerSimParam.AVG:
                    # Average similarity to answers with same marks
                    answers_sim[m] += sim / len(q.answers_by_mark[m])
                elif self.answer_similarity == AnswerSimParam.MAX:
                    # Most similar answer with same mark
                    if answers_sim[m] < sim:
                        answers_sim[m] = sim

        context_sim = {m: 0 for m in Mark.values()}
        for m in Mark.values():
            sim = self.calculate_answer_similarity(a, q.context_graph, mark=m)
            context_sim[m] = sim / q.context_graph.avg_similarity[m]

        return answers_sim, context_sim

    # def calculate_combined_prediction_ratio(self, question, answer):
    #     answers_sim, context_sim = self.calculate_prediction_ratio(question, answer)
    #
    #     answers_sim_total = 0
    #     for key, value in answers_sim.items():
    #         answers_sim_total += value
    #
    #     context_sim_total = 0
    #     for key, value in context_sim.items():
    #         context_sim_total += value
    #
    #     combined_sim = {m: 0 for m in Mark.values()}
    #     for m in Mark.values():
    #         sim = 0
    #         if answers_sim_total > 0:
    #             sim += answers_sim[m] / answers_sim_total
    #
    #         if context_sim_total > 0:
    #             sim += (context_sim[m] / context_sim_total) * 0.3
    #
    #         combined_sim[m] = sim
    #
    #     return combined_sim

    def make_regression_prediction(self, question, answer):
        answers_sim, context_sim = self.calculate_prediction_ratio(question, answer)
        q = self.questions[question]

        x_i = self.transform_parameters(answers_sim, context_sim)
        p = q.linear_regression.predict([x_i])

        if p >= 0.75:
            return 1
        elif p < 0.25:
            return 0
        else:
            return 0.5

    # def make_prediction(self, question, answer):
    #     sim = self.calculate_combined_prediction_ratio(question, answer)
    #
    #     max_sim = 0
    #     final_mark = None
    #
    #     for m in Mark.values():
    #         if sim[m] > max_sim:
    #             final_mark = m
    #             max_sim = sim[m]
    #
    #     return final_mark

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
