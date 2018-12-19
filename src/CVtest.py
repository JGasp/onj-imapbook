import data
import random
from qaModel.qaModel import QAModel


class CVTest:
    def __init__(self):
        self.text = data.get_text()
        self.questions = data.get_questions()

    def run_test(self, k=5):

        question_batch = [{} for _ in range(k)]

        results = {q:[] for q in self.questions}

        for q in self.questions:
            answers = self.questions[q]
            num_of_answers = len(answers)
            permut = range(num_of_answers)
            random.shuffle(permut)

            for i in range(k):
                question_batch[i][q] = []

            for i in range(num_of_answers):
                a = answers[i]
                p = permut[i]

                question_batch[p][q].append(a)

        for run in range(k):
            train = {q:[] for q in self.questions}
            test = None

            for i in range(k):
                if i == run:
                    test = question_batch[i]
                else:
                    for q in question_batch[i]:
                        train[q].extend(question_batch[i][q])

            qa_model = QAModel()
            qa_model.set_text(self.text)
            qa_model.set_questions(train)
            qa_model.build()

            for q in test:
                answers = test[q]

                res = {"response": [], "actual": [], "predicted": []}

                for a in answers:
                    p = qa_model.make_prediction(q, a)
                    res["response"].append(a)
                    res["predicted"].append(p)
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
