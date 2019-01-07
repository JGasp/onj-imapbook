from typing import Dict

from model import Data
from model.dto.Question import Question
from model.QAEvaluationModel import QAEvaluationModel


class LearnerModelB:
    def __init__(self):
        self.questions: Dict[str, Question] = Data.get_questions()
        self.generated_questions: Dict[str, Question] = Data.get_generated_answers()

    def build(self):
        for key, gq in self.generated_questions.items():
            for ga in gq.answers:
                self.questions[key].add_answer(ga)

        model = QAEvaluationModel()
        model.add_questions(self.questions)
        model.build()

        return ClassifierModelB(model)


class ClassifierModelB:
    def __init__(self, model: QAEvaluationModel):
        self.model = model

    def make_prediction(self, question, answer):
        self.model.make_regression_prediction(question, answer)

