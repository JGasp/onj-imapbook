from qaModel.qaModel import QAModel
import data


class LearnerModelA:
    def __init__(self):
        self.raw_text = data.get_text()
        self.questions = data.get_questions()

    def build(self):
        qa_model = QAModel()
        qa_model.set_text(self.raw_text)
        qa_model.set_questions(self.questions)
        qa_model.build()

        return ClassifierModelA(qa_model)


class ClassifierModelA:
    def __init__(self, model: QAModel):
        self.model = model

    def make_prediction(self, question, answer):
        self.model.make_prediction(question, answer)
