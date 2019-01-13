from typing import Dict

from model.QAEvaluationModel import QAEvaluationModel
from model import Data
from model.dto import Question


class ModelA:
    def __init__(self, include_generated_questions=True):
        self.questions: Dict[str, Question] = Data.get_single_answer_questions()

        self.generated_questions = {}
        if include_generated_questions:
            self.generated_questions: Dict[str, Question] = Data.get_generated_answers()

        self.model: QAEvaluationModel = None
        self.model_file_name = './model/model_a.data'

    def build(self):
        for key, gq in self.generated_questions.items():
            for ga in gq.answers:
                self.questions[key].add_answer(ga)

        self.model = QAEvaluationModel()
        self.model.add_questions(self.questions)
        self.model.build()

    def make_prediction(self, question, answer):
        return self.model.make_prediction(question, answer)

    def persist(self):
        self.model.persist(self.model_file_name)

    def load(self):
        self.model = QAEvaluationModel()
        self.model.load(self.model_file_name)
