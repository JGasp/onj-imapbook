from typing import Dict

from model import Data
from model.dto.Question import Question
from model.QAEvaluationModel import QAEvaluationModel


class ModelB:
    def __init__(self):
        self.questions: Dict[str, Question] = Data.get_questions()
        self.generated_questions: Dict[str, Question] = Data.get_generated_answers()
        self.model = None

        self.model_file_name = './model/model_b.data'

    def build(self):
        for key, gq in self.generated_questions.items():
            for ga in gq.answers:
                self.questions[key].add_answer(ga)

        self.model = QAEvaluationModel()
        self.model.add_questions(self.questions)
        self.model.build()

    def make_prediction(self, question, answer):
        self.model.make_regression_prediction(question, answer)

    def persist(self):
        self.model.persist(self.model_file_name)

    def load(self):
        self.model = QAEvaluationModel()
        self.model.load(self.model_file_name)
