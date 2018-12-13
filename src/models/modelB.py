from typing import Dict

from qaModel.dataStructure.question import Question

import pandas


class LearnerModelB:
    def __init__(self):
        self.questions: Dict[Question] = {}

    def build(self):
        self.load_data()





class ClassifierModelB:
    def __init__(self):
        print("asd")

    def make_prediction(self, question, answer):
        print("asd")

    @staticmethod
    def get_build_model():
        # TODO pre-build model
        return ClassifierModelB
