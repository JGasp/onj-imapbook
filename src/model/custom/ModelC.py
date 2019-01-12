from typing import Dict

from model import Data
from model.ConceptNetCore import ConceptNetCore
from model.QAEvaluationModel import QAEvaluationModel
from model.dto import Question
from model.dto.Graph import Link, Graph
import gensim


class ModelC:
    def __init__(self, use_concept_net=True, similarity_threshold=0.6):

        self.questions: Dict[str, Question] = Data.get_questions()
        self.generated_questions: Dict[str, Question] = Data.get_generated_answers()

        for key, gq in self.generated_questions.items():
            for ga in gq.answers:
                self.questions[key].add_answer(ga)

        # Download from https://github.com/mmihaltz/word2vec-GoogleNews-vectors
        self.word_to_vec = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
        self.similarity_threshold = similarity_threshold

        self.use_concept_net = use_concept_net
        if self.use_concept_net:
            self.concept_net_core = ConceptNetCore()

        self.model: QAEvaluationModel = None
        self.model_file_name = './model/model_c.data'

        # Thresholds
        # buy <> purchase => 0.7639905
        # space <> gravity => 0.19068718
        # confident <> brave = > 0.19375078
        # confident <> happy = > 0.5893531
        # confident <> car = > 0.06907264
        # confident <> animal = > -0.010198391
        # brave <> courageous = > -0.010198391

        # word_to_vec.similar_by_word('brave')
        # [('courageous', 0.7414764761924744), ('bravely', 0.648894190788269), ('bravest', 0.6485252380371094),
        # ('valiant', 0.5948496460914612), ('gallant', 0.5745828151702881), ('courageously', 0.572306752204895),
        # ('courage', 0.5651585459709167), ('brave_souls', 0.5543081760406494), ('fearless', 0.553043782711029),
        # ('heroic', 0.5513851046562195)]

        # word_to_vec.similar_by_word('buy')
        # [('sell', 0.8308461904525757), ('purchase', 0.7639904618263245), ('buying', 0.7209186553955078),
        # ('bought', 0.7087081670761108), ('buys', 0.6617437601089478), ('Buy', 0.5850198864936829),
        # ('tobuy', 0.5843992829322815), ('purchased', 0.5826954245567322), ('Buying', 0.5780205130577087),
        # ('acquire', 0.5730166435241699)]

    def link_similarity(self, link: Link, stop_words):
        max_sim = 0
        for w in link.words:
            if w in stop_words:
                return 1
            else:
                for sw in stop_words:
                    if w in self.word_to_vec and sw in self.word_to_vec:
                        sim = self.word_to_vec.similarity(w, sw)
                        if sim > self.similarity_threshold:
                            if sim > max_sim:
                                max_sim = sim

        return max_sim

    def filter_similar_nodes(self, word, graph: Graph):
        nodes = []
        similarity = {}
        for key, ni in graph.node_index.items():
            if key == word:
                for i in ni:
                    similarity[i] = 1
                    nodes.append(i)
            else:
                if word in self.word_to_vec:
                    words = [key]

                    if self.use_concept_net:
                        data = self.concept_net_core.get_data(key)
                        for d in data:
                            words.append(d.label)

                    for w in words:
                        if w in self.word_to_vec:
                            sim = self.word_to_vec.similarity(word, w)
                            if sim > self.similarity_threshold:
                                for i in ni:
                                    similarity[i] = sim
                                    nodes.append(i)

        return nodes, similarity

    def build(self):
        self.build_custom(self.questions)

    def build_custom(self, questions):

        self.model = QAEvaluationModel()
        self.model.is_link_similarity = self.link_similarity
        self.model.get_similar_nodes = self.filter_similar_nodes

        self.model.add_questions(questions)
        self.model.build()

        return self.model

    def persist_concept_net_data(self):
        self.concept_net_core.persist()

    def make_prediction(self, question, answer):
        self.model.make_prediction(question, answer)

    def persist(self):
        self.model.persist(self.model_file_name)

    def load(self):
        self.model = QAEvaluationModel()
        self.model.is_link_similarity = self.link_similarity
        self.model.get_similar_nodes = self.filter_similar_nodes
        self.model.load(self.model_file_name)
