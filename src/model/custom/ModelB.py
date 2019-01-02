from typing import Dict

from model.dto.Question import Question


class LearnerModelB:
    def __init__(self):
        self.questions: Dict[str, Question] = {}

    def build(self):
        print("TODO")


class ClassifierModelB:
    def __init__(self):
        print("TODO")

    def make_prediction(self, question, answer):
        print("TODO")

    @staticmethod
    def get_build_model():
        # TODO pre-build model
        return ClassifierModelB()
