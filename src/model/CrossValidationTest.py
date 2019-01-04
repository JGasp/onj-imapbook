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

                for a in q.answers:
                    ri = qa_model.make_regression_prediction(q.raw_text, a.raw_text)
                    actual_mark = float(a.final_mark.value.replace(",", "."))
                    res = {"response": a.raw_text, "actual": actual_mark, "predicted": ri}
                    results[q.raw_text].append(res)

        for q in results:
            answers = results[q]
            mae = 0
            counter = 0

            print(">>>> Question %s" % q)

            for a in answers:
                mae += abs(a["predicted"] - a["actual"])
                counter += 1

                print("%f %f \t %s" % (a["predicted"], a["actual"], a["response"]))

            mae /= counter
            print("#### MAE: %f" % mae)
