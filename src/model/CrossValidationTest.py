from typing import Dict

from model import Data
import random

from model.dto.Question import Question
from model.QAEvaluationModel import QAEvaluationModel


class CrossValidationTest:
    def __init__(self):
        self.questions: Dict[str, Question] = Data.get_questions()

    def run_test(self, k=5):
        question_batch = {}
        results = {self.questions[key].raw_text: [] for key in self.questions}

        for key, q in self.questions.items():
            num_of_answers = len(q.answers)
            random_index = [i for i in range(num_of_answers)]
            random.shuffle(random_index)

            question_batch[q] = [[] for _ in range(k)]

            for i in range(num_of_answers):
                a = q.answers[i]
                ri = random_index[i]

                question_batch[q][ri % k].append(a)

        for run in range(k):
            train = {}
            test = {}

            for key, q in self.questions.items():
                q_train_copy = q.copy()
                q_test_copy = q.copy()

                for i in range(k):
                    if i == run:
                        for a in question_batch[q][i]:
                            q_test_copy.add_answer(a)
                    else:
                        for a in question_batch[q][i]:
                            q_train_copy.add_answer(a)

                train[q.raw_text] = q_train_copy
                test[q.raw_text] = q_test_copy

            qa_model = QAEvaluationModel()
            qa_model.set_questions(train)
            qa_model.build()

            for key in test:
                q = test[key]

                res = {"response": [], "actual": [], "predicted": []}

                for a in q.answers:
                    ri = qa_model.make_prediction(q.raw_text, a)
                    res["response"].append(a)
                    res["predicted"].append(ri)
                    res["actual"].append(a["fr"])

                results[q] = res

        mae = 0
        counter = 0

        for q in results:
            answers = results[q]

            print("Question %s" % q)

            for a in answers:
                mae += abs(a["predicted"] - a["actual"])
                counter += 1

                print("%s \t %f %f" % (a, a["predicted"], a["actual"]))
