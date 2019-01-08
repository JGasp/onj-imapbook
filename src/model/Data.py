import pandas
import random
from model.dto.Answer import Answer
from model.dto.Answer import Mark
from model.dto.Question import Question


def get_text():
    f = open("./res/Weightless.txt")
    return f.read()


def get_single_answer_questions():
    df = pandas.read_csv('./res/Weightless_dataset_train_A.csv', encoding='utf-8')

    questions = {}
    for i, row in df.iterrows():
        q = row['Question']

        if q in questions:
            q_data = questions[q]
        else:
            context = row['Text.used.to.make.inference'].replace("...", "")
            q_data = Question(q, context_text=context)
            questions[q] = q_data

        res = row['Response']
        gr = row['Glenn.s.rating']
        ar = row['Amber.s.rating']
        fr = row['Final.rating']

        ans = Answer(res, Mark.by_value(gr), Mark.by_value(ar), Mark.by_value(fr))

        q_data.add_answer(ans)

    return questions


def get_questions():
    df = pandas.read_csv('./res/Weightless_dataset_train.csv', encoding='utf-8')

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


def get_generated_answers():
    df = pandas.read_csv('./res/Generated_answers.csv', encoding='utf-8')

    questions = {}
    for i, row in df.iterrows():
        q = row['Question']

        if q not in questions:
            questions[q] = Question(q)

        q_data = questions[q]

        ans = Answer(row['Answer'], Mark.M0, Mark.M0, Mark.M0)
        q_data.add_answer(ans)

    return questions


def generate_answers():
    text = get_text()
    text = text.split(" ")
    text_len = len(text)

    questions = get_questions()

    f = open("./res/Generated_answers.csv", "w", encoding="utf-8")
    f.write("Question,Answer\n")

    for key, q in questions.items():
        q: Question = q

        num = int((len(q.answers_by_mark[Mark.M05]) + len(q.answers_by_mark[Mark.M1])) / 2 - len(q.answers_by_mark[Mark.M0]))

        avg_len = 0
        count = 0

        for m in [Mark.M05, Mark.M1]:
            for a in q.answers_by_mark[m]:
                avg_len += len(a.raw_text.split(" "))
                count += 1

        avg_len = int(avg_len / count)

        for i in range(num):
            sentance = []

            for j in range(avg_len):
                word = text[random.randint(0, text_len - 1)].replace("\"", "").replace("\n",  " ")
                sentance.append(word)

            gen_answer = " ".join(sentance)

            f.write("\"%s\",\"%s\"\n" % (q.raw_text, gen_answer))

    f.close()
