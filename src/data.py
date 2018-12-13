import pandas


def get_text():
    f = open("./res/Weightless.txt")
    return f.read()


def get_questions():
    df = pandas.read_csv('./res/Weightless_dataset_train.csv')

    questions = {}
    for i, row in df.iterrows():
        q = row['Question']

        if q in questions:
            q_set = questions[q]
        else:
            q_set = []
            questions[q] = q_set

        res = row['Response']
        gr = row['Glenn.s.rating']
        ar = row['Amber.s.rating']
        fr = row['Final.rating']

        ans = {"res": res, "gr": gr, "ar": ar, "fr": fr}

        q_set.append(ans)

    return questions
