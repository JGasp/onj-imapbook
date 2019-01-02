import pandas
from model.dto.Answer import Answer
from model.dto.Answer import Mark
from model.dto.Question import Question


def get_text():
    f = open("./res/Weightless.txt")
    return f.read()


def get_questions():
    df = pandas.read_csv('./res/Weightless_dataset_train.csv')

    questions = {}
    for i, row in df.iterrows():
        q = row['Question']

        if q in questions:
            q_data = questions[q]
        else:
            context = row['Text.used.to.make.inference']
            q_data = Question(q, context_text=context)
            questions[q] = q_data

        res = row['Response']
        gr = row['Glenn.s.rating']
        ar = row['Amber.s.rating']
        fr = row['Final.rating']

        ans = Answer(res, Mark.by_value(gr), Mark.by_value(ar), Mark.by_value(fr))

        q_data.add_answer(ans)

    return questions
