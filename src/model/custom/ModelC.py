from typing import Dict

from model import Data
from model.QAEvaluationModel import QAEvaluationModel
from model.dto import Question
from model.dto.Graph import Link, Graph
import gensim


class LearnerModelC:
    def __init__(self):
        self.questions: Dict[str, Question] = Data.get_questions()
        self.generated_questions: Dict[str, Question] = Data.get_generated_answers()

        self.model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
        self.similarity_threshold = 0.7

        # Thresholds
        # buy <> purchase => 0.7639905
        # space <> gravity => 0.19068718

    def link_similarity(self, link: Link, stop_words):
        for w in link.words:
            for sw in stop_words:

                sim = self.model.similarity(w, sw)
                if sim > self.similarity_threshold:
                    return True

        return False

    def filter_similar_nodes(self, word, graph: Graph):
        nodes = []
        for key, ni in graph.node_index.items():
            sim = self.model.similarity(word, key)
            if sim > self.similarity_threshold:
                nodes.extend(ni)

        return nodes

    def build(self):
        for key, gq in self.generated_questions.items():
            for ga in gq.answers:
                self.questions[key].add_answer(ga)

        model = QAEvaluationModel()
        model.is_link_similarity = self.link_similarity
        model.get_similar_nodes = self.filter_similar_nodes

        model.set_questions(self.questions)
        model.build()

        return ClassifierModelC(model)


class ClassifierModelC:
    def __init__(self, model: QAEvaluationModel):
        self.model = model

    def make_prediction(self, question, answer):
        self.model.make_regression_prediction(question, answer)
