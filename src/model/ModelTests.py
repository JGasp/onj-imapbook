from typing import Dict
from sklearn.metrics import f1_score
from model import Data
import random
import numpy as np

from model.dto.Answer import Answer
from model.dto.Question import Question
from model.QAEvaluationModel import QAEvaluationModel


class CrossValidationModelBTest:
    def __init__(self, include_generated_answers=True):
        self.questions: Dict[str, Question] = Data.get_questions()

        self.generated_answers = {}
        if include_generated_answers:
            self.generated_answers = Data.get_generated_answers()

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

                if self.generated_answers is not None:
                    if q.raw_text in self.generated_answers:
                        for ga in self.generated_answers[q.raw_text].answers[:1]:
                            q_train_copy.add_answer(ga)

                train[q.raw_text] = q_train_copy
                test[q.raw_text] = q_test_copy

            qa_model = QAEvaluationModel()
            qa_model.add_questions(train)
            qa_model.build()

            for key in test:
                q = test[key]

                for a in q.answers:
                    ri = qa_model.make_prediction(q.raw_text, a.raw_text)
                    actual_mark = float(a.final_mark.value.replace(",", "."))
                    res = {"response": a.raw_text, "actual": actual_mark, "predicted": ri}
                    results[q.raw_text].append(res)

        global_confusion_matrix = np.zeros((3, 3))
        global_mae = 0
        global_counter = 0

        for q in results:
            answers = results[q]

            confusion_matrix = np.zeros((3, 3))
            mae = 0
            counter = 0

            print(">>>> Question %s" % q)

            for a in answers:
                diff = abs(a["predicted"] - a["actual"])

                mae += diff
                counter += 1

                global_mae += diff
                global_counter += 1

                pred = int(a["predicted"] * 2)
                actual = int(a["actual"] * 2)
                confusion_matrix[pred, actual] += 1
                global_confusion_matrix[pred, actual] += 1

                print("%02f %02f \t %s" % (a["predicted"], a["actual"], a["response"]))

            mae /= counter
            print("#### MAE: %f" % mae)
            print(confusion_matrix)

        global_mae /= global_counter
        print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("#### MAE: %f" % global_mae)
        print(global_confusion_matrix)

        true_scores = []
        pred_scores = []
        for k, answers in results.items():
            for a in answers:
                true_scores.append(int(float(a['actual']) * 10))
                pred_scores.append(int(float(a['predicted']) * 10))

        print("F1 (macro): %f" % f1_score(true_scores, pred_scores, average='macro'))
        print("F1 (micro): %f" % f1_score(true_scores, pred_scores, average='micro'))
        print("F1 (weighted): %f" % f1_score(true_scores, pred_scores, average='weighted'))


class ModelATester:

    def __init__(self, include_generated_answers=True):
        self.questions: Dict[str, Question] = Data.get_single_answer_questions()

        self.generated_answers = {}
        if include_generated_answers:
            self.generated_answers = Data.get_generated_answers()

            for key, gq in self.generated_answers.items():
                q = self.questions[key]

                for ga in gq.answers[:4]:
                    q.add_answer(ga)

        # question_list = [q for k, q in self.questions.items()]
        # size = len(question_list)
        #
        # for i in range(size):
        #     for j in range(i+1, size):
        #         q_i = question_list[i]
        #         q_j = question_list[j]
        #
        #         a_i = q_i.answers[0]
        #         a_j = q_j.answers[0]
        #
        #         q_i.add_answer(Answer(a_j.raw_text))
        #         q_j.add_answer(Answer(a_i.raw_text))

        self.test_questions: Dict[str, Question] = Data.get_questions()

        for key, q in self.questions.items():
            for a in q.answers:
                self.test_questions[key].remove_answer(a.raw_text)

    def run_test(self):

        results = {self.questions[key].raw_text: [] for key in self.questions}

        model = QAEvaluationModel()
        model.add_questions(self.questions)
        model.build()

        for key in self.test_questions:
            q = self.test_questions[key]

            for a in q.answers:
                ri = model.make_prediction(q.raw_text, a.raw_text)
                actual_mark = float(a.final_mark.value.replace(",", "."))
                res = {"response": a.raw_text, "actual": actual_mark, "predicted": ri}
                results[q.raw_text].append(res)

        global_confusion_matrix = np.zeros((3, 3))
        global_mae = 0
        global_counter = 0

        for k, answers in results.items():

            confusion_matrix = np.zeros((3, 3))
            mae = 0
            counter = 0

            print(">>>> Question %s" % k)

            for a in answers:
                diff = abs(a["predicted"] - a["actual"])

                mae += diff
                counter += 1

                global_mae += diff
                global_counter += 1

                pred = int(a["predicted"] * 2)
                actual = int(a["actual"] * 2)
                confusion_matrix[pred, actual] += 1
                global_confusion_matrix[pred, actual] += 1

                print("%02f %02f \t %s" % (a["predicted"], a["actual"], a["response"]))

            mae /= counter
            print("#### MAE: %f" % mae)
            print(confusion_matrix)

        global_mae /= global_counter
        print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("#### MAE: %f" % global_mae)
        print(global_confusion_matrix)

        true_scores = []
        pred_scores = []
        for k, answers in results.items():
            for a in answers:
                true_scores.append(int(float(a['actual']) * 10))
                pred_scores.append(int(float(a['predicted']) * 10))

        print("F1 (macro): %f" % f1_score(true_scores, pred_scores, average='macro'))
        print("F1 (micro): %f" % f1_score(true_scores, pred_scores, average='micro'))
        print("F1 (weighted): %f" % f1_score(true_scores, pred_scores, average='weighted'))
