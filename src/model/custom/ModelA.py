from model.QAEvaluationModel import QAEvaluationModel
from model import Data


class LearnerModelA:
    def __init__(self):
        self.raw_text = Data.get_text()
        self.questions = Data.get_questions()

    def build(self):
        qa_model = QAEvaluationModel()
        qa_model.set_questions(self.questions)
        qa_model.build()

        return ClassifierModelA(qa_model)


class ClassifierModelA:
    def __init__(self, model: QAEvaluationModel):
        self.model = model

    def make_prediction(self, question, answer):
        self.model.make_prediction(question, answer)
