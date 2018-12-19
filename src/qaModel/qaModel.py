from qaModel.dataStructure.text import Text
from qaModel.dataStructure.question import Question
from qaModel.dataStructure.answer import Answer
from qaModel.dataStructure.graph import Node
from qaModel.dataStructure.graph import Link
from qaModel.dataStructure.graph import Graph

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from typing import Dict

import re


class QAModel:
    def __init__(self):
        self.text = None
        self.questions: Dict[Question] = {}
        self.question_context = {}
        self.final_question_context = {"0":{}, "0.5":{}, "1":{}}

        self.max_contexts = 10
        self.neighbours = 10

        self.word_len = 5
        self.stop_word_len = 1
        self.len_lambda = 0.2

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

    def graph_contains(self, val: str, graph):
        for n in graph:
            if n.word == val:
                return True
        return False

    def init_contexts(self):
        for key in self.questions:
            q = self.questions[key]
            context = []

            count = 0
            sub_context = []
            prev_neighbours = []

            for n in self.text.graph:
                if self.graph_contains(n.word, q.graph):
                    count = self.neighbours

                if count > 0:
                    if count == self.neighbours:
                        sub_context.extend(prev_neighbours)

                    count -= 1
                    sub_context.append(n)
                elif len(sub_context) > 0:
                    context.append(Graph(sub_context))
                    sub_context = []

                prev_neighbours.append(n)
                if len(prev_neighbours) > self.neighbours:
                    prev_neighbours.pop(0)

            self.question_context[key] = context

    def get_graph_direction(self, ni1, ni2):
        if ni1 > ni2:
            return ni2, ni1
        return ni1, ni2

    def calculate_node_distance(self, graph, start, end):
        distance = 0
        for i in range(start, end):
            n = graph.nodes[i]
            distance += len(n.next.words) * self.stop_word_len

            if i != start:
                distance += self.word_len

        return distance

    def calculate_answer_similarity(self, answer: Answer, graph: Graph):
        similarity = 0

        sim_values = {}
        if answer.final_mark == "0":
            sim_values = graph.similarity_0
        elif answer.final_mark == "0.5":
            sim_values = graph.similarity_05
        elif answer.final_mark == "1":
            sim_values = graph.similarity_1

        prev_word_index = []
        stop_words = {}
        for an in answer.graph:
            word_index = graph.get_nodes_with_word(an.word)

            # Increase node score
            for wi in word_index:
                n = graph.nodes[wi]
                if n in sim_values:
                    similarity += sim_values[n]

            for w in an.next.words:
                stop_words[w] = True
            for w in an.prev.words:
                stop_words[w] = True

            if len(prev_word_index) > 0:
                # Normalize score based on number of paths
                paths = len(prev_word_index) * len(word_index)
                for pwi in prev_word_index:
                    for wi in word_index:
                        start, end = self.get_graph_direction(pwi, wi)
                        distance = self.calculate_node_distance(graph, start, end)

                        for i in range(start, end):
                            n = graph.nodes[i]

                            for w in n.next.words:
                                if w in stop_words:
                                    sim = sim_values[n]
                                    if self.len_lambda > 0:
                                        sim = sim / (self.len_lambda * distance)
                                    similarity += sim / paths

            if len(word_index) > 0:
                prev_word_index = word_index
                stop_words = {}

        return similarity

    def update_graph_similarity(self, answer: Answer, graph: Graph):
        sim_values = {}
        if answer.final_mark == "0":
            sim_values = graph.similarity_0
        elif answer.final_mark == "0.5":
            sim_values = graph.similarity_05
        elif answer.final_mark == "1":
            sim_values = graph.similarity_1

        prev_word_index = []
        stop_words = {}
        for an in answer.graph:
            word_index = graph.get_nodes_with_word(an.word)

            # Increase node score
            for wi in word_index:
                n = graph.nodes[wi]
                if n not in sim_values:
                    sim_values[n] = 0
                sim_values[n] += self.word_len

            for w in an.next.words:
                stop_words[w] = True
            for w in an.prev.words:
                stop_words[w] = True

            if len(prev_word_index) > 0:
                # Normalize score based on number of paths
                paths = len(prev_word_index) * len(word_index)
                for pwi in prev_word_index:
                    for wi in word_index:
                        start, end = self.get_graph_direction(pwi, wi)
                        distance = self.calculate_node_distance(graph, start, end)

                        for i in range(start, end):
                            n = graph.nodes[i]

                            stop_word_similarity = 0
                            for w in n.next.words:
                                if w in stop_words:
                                    stop_word_similarity = 1
                                    break

                            if self.len_lambda > 0:
                                stop_word_similarity = stop_word_similarity / (self.len_lambda * distance)

                            if n not in sim_values:
                                sim_values[n] = 0
                            sim_values[n] += (stop_word_similarity / paths)

            if len(word_index) > 0:
                prev_word_index = word_index
                stop_words = {}

    def calculate_answer_similarity(self):
        for q in self.questions:
            question = self.questions[q]
            context = self.question_context[q]

            for graph in context:
                for a in question.answers:
                    self.update_graph_similarity(a, graph)

        for q in self.questions:
            question = self.questions[q]
            context = self.question_context[q]

            for graph in context:

                similarity = {"0": 0, "05": 0, "1": 0}
                count = {"0": 0, "05": 0, "1": 0}

                for a in question.answers:
                    sim = self.calculate_answer_similarity(a, graph)
                    similarity[a.final_mark] += sim
                    count[a.final_mark] += 1

                for key in similarity:
                    sim = similarity[key]
                    c = count[key]

                    avg = sim / c
                    if key == "0":
                        graph.avg_similarity_0 = avg
                    elif key == "05":
                        graph.avg_similarity_05 = avg
                    elif key == "1":
                        graph.avg_similarity_1 = avg

    def trim_context(self):

        for q in self.questions:
            context = self.question_context[q]

            context.sort(key=lambda x: -x.avg_similarity_0)
            self.final_question_context["0"][q] = context[0:self.max_contexts]

            context.sort(key=lambda x: -x.avg_similarity_05)
            self.final_question_context["0.5"][q] = context[0:self.max_contexts]

            context.sort(key=lambda x: -x.avg_similarity_1)
            self.final_question_context["1"][q] = context[0:self.max_contexts]

    def build(self):
        self.init_contexts()
        self.calculate_answer_similarity()
        self.trim_context()

    def make_prediction(self, question, answer):

        q = self.questions[question]

        c0 = self.final_question_context["0"][question]
        c05 = self.final_question_context["0.5"][question]
        c1 = self.final_question_context["1"][question]

        a = Answer(answer, self.build_graph(answer))

        sim0 = 0
        for c in c0:
            sim0 += self.calculate_answer_similarity(a, c)
        sim0 /= len(c0)

        sim05 = 0
        for c in c05:
            sim05 += self.calculate_answer_similarity(a, c)
        sim05 /= len(c05)

        sim1 = 0
        for c in c1:
            sim1 += self.calculate_answer_similarity(a, c)
        sim1 /= len(c1)

        # TODO add answer similarity to other answers (cosin distance + graph coverage)
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


    # def trim_contexts(self):
    #
    #     for q in self.question_context:
    #         question = self.questions[q]
    #         contexts = self.question_context[q]
    #
    #         for graph in contexts:
    #             sim = self.calculate_question_graph_similarity(question, graph)
    #             graph.question_similarity = sim
    #
    #         contexts.sort(key=lambda x: -x.question_similarity)
    #         self.question_context[q] = contexts[0:self.max_contexts]
    #
    # def calculate_question_graph_similarity(self, question, graph):
    #
    #     best_similarity = 0.0
    #
    #     prev_word_index = []
    #     stop_words = {}
    #     for qn in question.graph:
    #         word_index = graph.get_nodes_with_word(qn.word)
    #
    #         for w in qn.next.words:
    #             stop_words[w] = True
    #
    #         if len(prev_word_index) > 0:
    #             for pwi in prev_word_index:
    #                 for wi in word_index:
    #                     similarity = self.calculate_node_similarity(graph, wi, pwi, stop_words)
    #
    #                     if similarity > best_similarity:
    #                         best_similarity = similarity
    #
    #         if len(word_index) > 0:
    #             prev_word_index = word_index
    #             stop_words = {}
    #
    #     return best_similarity