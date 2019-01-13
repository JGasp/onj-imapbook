from enum import Enum

from sklearn import linear_model
from model.dto.Question import Question
from model.dto.Answer import Answer, Mark
from model.dto.Graph import Node
from model.dto.Graph import Link
from model.dto.Graph import Graph

import math
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from typing import Dict

import re
import pickle
import os.path
import random


class AnswerSimParam(Enum):
    AVG = "Average"
    MAX = "Maximum"


class QAEvaluationModel:
    def __init__(self, word_len=1, stop_word_len=1, len_lambda=0.2, max_ans_graphs=None, ans_sim=AnswerSimParam.MAX):
        self.questions: Dict[str, Question] = {}

        self.word_weight = word_len
        self.stop_word_weight = stop_word_len
        self.len_lambda = len_lambda

        self.find_first_similar = True
        self.answer_similarity: AnswerSimParam = ans_sim
        self.max_ans_graphs = max_ans_graphs

        # Functions to override for ModelC
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
            distance += len(n.next.words)  # * self.stop_word_weight

            if i != start:
                distance += 1  # self.word_weight

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
        total_similarity = 0
        sim_values = graph.similarity[mark]

        prev_word_index = []
        node_stop_words = {}
        for an in answer.text_graph.nodes:
            node_index, node_similarity = self.get_similar_nodes(an.word, graph)

            # Increase node score
            if len(node_index) > 0:
                for wi in node_index:
                    n = graph.nodes[wi]

                    best_sim = 0
                    if n in sim_values:
                        sim = sim_values[n] * node_similarity[wi]
                        if sim > best_sim:
                            best_sim = sim

                    total_similarity += best_sim

                if len(prev_word_index) > 0:
                    best_sim = 0
                    t_pwi = None
                    t_wi = None
                    t_link = None

                    for pwi in prev_word_index:
                        for wi in node_index:
                            start, end = self.get_graph_direction(pwi, wi)
                            distance = self.calculate_node_distance(graph, start, end)

                            if distance == 1 and len(node_stop_words) == 0:
                                best_sim = 1
                                t_pwi = graph.nodes[pwi]
                                t_wi = graph.nodes[wi]
                            else:
                                for i in range(start, end):
                                    n = graph.nodes[i]

                                    if n.next in sim_values:
                                        sim_sw = self.is_link_similarity(n.next, node_stop_words)
                                        if sim_sw > 0:
                                            sim_sw += sim_values[n.next] * sim_sw

                                            if self.len_lambda > 0:
                                                sim_sw = sim_sw / (self.len_lambda * distance)

                                            if sim_sw > best_sim:
                                                best_sim = sim_sw
                                                t_pwi = graph.nodes[pwi]
                                                t_wi = graph.nodes[wi]
                                                t_link = n.next

                    if best_sim > 0:
                        if t_link is None:
                            total_similarity += 1
                        else:
                            total_similarity += sim_values[t_link]

                        # total_similarity += sim_values[t_pwi] / 2
                        # total_similarity += sim_values[t_wi] / 2

            if not(self.find_first_similar and len(node_index) == 0):
                prev_word_index = node_index
                node_stop_words = {}

            for w in an.next.words:
                node_stop_words[w] = True

        if len(answer.text_graph.nodes) == 0:
            return 0.0

        # Normalize similarity based on answer length
        total_similarity /= len(answer.text_graph.nodes)

        return total_similarity

    def update_graph_similarity(self, answer: Answer, graph: Graph):
        sim_values = graph.similarity[answer.final_mark]

        prev_word_index = []
        node_stop_words = {}

        for an in answer.text_graph.nodes:
            node_index, node_similarity = self.get_similar_nodes(an.word, graph)

            if len(node_index) > 0:
                for wi in node_index:
                    n = graph.nodes[wi]
                    if n not in sim_values:
                        sim_values[n] = 0
                    sim_values[n] += self.word_weight * node_similarity[wi]

                if len(prev_word_index) > 0:
                    best_sim = 0
                    t_pwi = None
                    t_wi = None
                    t_link = None

                    for pwi in prev_word_index:
                        for wi in node_index:
                            start, end = self.get_graph_direction(pwi, wi)
                            distance = self.calculate_node_distance(graph, start, end)

                            if len(node_stop_words) == 0:
                                if distance == 1:
                                    best_sim = 1
                                    t_pwi = graph.nodes[pwi]
                                    t_wi = graph.nodes[wi]
                            else:
                                for i in range(start, end):
                                    n = graph.nodes[i]

                                    if len(n.next.words) > 0:
                                        sw_sim = self.is_link_similarity(n.next, node_stop_words)

                                        if sw_sim > 0:  # Penalize long references
                                            sw_sim = self.stop_word_weight * sw_sim
                                            if self.len_lambda > 0:
                                                sw_sim = sw_sim / (self.len_lambda * distance)

                                            if sw_sim > best_sim:
                                                best_sim = sw_sim
                                                t_pwi = graph.nodes[pwi]
                                                t_wi = graph.nodes[wi]
                                                t_link = n.next

                    if best_sim > 0:
                        if t_link is not None:
                            if t_link not in sim_values:
                                sim_values[t_link] = 0
                            sim_values[t_link] += best_sim

                        sim_values[t_pwi] += best_sim / 2
                        sim_values[t_wi] += best_sim / 2

            if not (self.find_first_similar and len(node_index) == 0):  # Find first similar node (skip not found ones)
                prev_word_index = node_index
                node_stop_words = {}

            for w in an.next.words:
                node_stop_words[w] = True

    def build_context_graph_similarity(self):
        for key, q in self.questions.items():
            for a in q.answers:
                self.update_graph_similarity(a, q.context_graph)

        for key, q in self.questions.items():
            graph = q.context_graph

            for m in Mark.values():
                for k in graph.similarity[m]:
                    graph.similarity[m][k] = math.log(1 + graph.similarity[m][k])

            sum_similarity = {m: 0 for m in Mark.values()}
            count = {m: 0 for m in Mark.values()}

            for a in q.answers:
                sim = self.calculate_answer_similarity(a, graph, mark=a.final_mark)
                sum_similarity[a.final_mark] += sim
                count[a.final_mark] += 1

            for m in Mark.values():
                if count[m] > 0:
                    graph.avg_similarity[m] = (sum_similarity[m] + 1) / count[m]
                else:
                    graph.avg_similarity[m] = 1

    def build_answers_graph_similarity(self):
        for key, q in self.questions.items():
            for m in Mark.values():
                answers = q.answers_by_mark[m]

                graph_answers = answers
                if self.max_ans_graphs is not None:
                    random.shuffle(graph_answers)
                    graph_answers = graph_answers[:self.max_ans_graphs]

                for ag in graph_answers:
                    for aup in answers:
                        self.update_graph_similarity(aup, ag.text_graph)

        for key, q in self.questions.items():
            for m in Mark.values():
                graph_answers = q.answers_by_mark[m]
                if self.max_ans_graphs is not None:
                    graph_answers = graph_answers[:self.max_ans_graphs]

                for ag in graph_answers:

                    sum_similarity = 0
                    count = 0

                    for aup in q.answers_by_mark[m]:
                        sim = self.calculate_answer_similarity(aup, ag.text_graph, mark=ag.final_mark)
                        sum_similarity += sim
                        count += 1

                    if count > 0:
                        ag.text_graph.avg_similarity[m] = (sum_similarity + 1) / count
                    else:
                        ag.text_graph.avg_similarity[m] = 1

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

            lm = linear_model.LinearRegression()
            # lm = linear_model.LogisticRegression(solver='lbfgs')
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
            graph_answers = q.answers_by_mark[m]
            if self.max_ans_graphs is not None:
                graph_answers = graph_answers[:self.max_ans_graphs]

            for ag in graph_answers:
                if ag.text_graph.avg_similarity[m] > 0:
                    sim = self.calculate_answer_similarity(a, ag.text_graph, mark=m) / ag.text_graph.avg_similarity[m]

                    if self.answer_similarity == AnswerSimParam.AVG:
                        # Average similarity to answers with same marks
                        answers_sim[m] += sim / len(graph_answers)
                    elif self.answer_similarity == AnswerSimParam.MAX:
                        # Most similar answer with same mark
                        if answers_sim[m] < sim:
                            answers_sim[m] = sim

        context_sim = {m: 0 for m in Mark.values()}
        for m in Mark.values():
            sim = self.calculate_answer_similarity(a, q.context_graph, mark=m)
            context_sim[m] = sim / q.context_graph.avg_similarity[m]

        return answers_sim, context_sim

    def make_prediction(self, question, answer):
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
