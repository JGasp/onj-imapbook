from model import Data
from model.Data import generate_answers
from model.ModelTests import CrossValidationModelBTest, ModelATester, CrossValidationModelCTest, ModelTester
from model.custom.ModelA import ModelA
from model.custom.ModelB import ModelB
from model.custom.ModelC import ModelC
from sklearn.metrics import f1_score
import numpy as np
import gensim


def train_models():
    test = ModelTester()

    model_a = ModelA()
    model_a.build()
    model_a.persist()
    test.run_test(model_a)

    model_a = ModelA()
    model_a.load()
    test.run_test(model_a)

    model_b = ModelB()
    model_b.build()
    model_b.persist()
    test.run_test(model_b)

    model_b = ModelB()
    model_b.load()
    test.run_test(model_b)

    word2vec = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

    model_c = ModelC(preloaded_word2vec=word2vec)
    model_c.build(max_ans_graphs=5)
    model_c.persist_concept_net_data()
    model_c.persist()
    test.run_test(model_c)

    model_c = ModelC(preloaded_word2vec=word2vec)
    model_c.load(max_ans_graphs=5)
    test.run_test(model_c)


def calculate_human_prediction_F1():
    questions = Data.get_questions()

    gleen = []
    gleen_matrix = np.zeros((3, 3))

    amber = []
    amber_matrix = np.zeros((3, 3))

    final = []

    for k, q in questions.items():
        for a in q.answers:
            gl_mark = float(a.gleen_mark.value.replace(",", "."))
            fn_mark = float(a.final_mark.value.replace(",", "."))
            am_mark = float(a.amber_mark.value.replace(",", "."))

            gleen_matrix[int(gl_mark * 2), int(fn_mark * 2)] += 1
            amber_matrix[int(am_mark * 2), int(fn_mark * 2)] += 1

            final.append(int(fn_mark * 10))
            gleen.append(int(gl_mark * 10))
            amber.append(int(am_mark * 10))

    print("## GLEEN")
    print(gleen_matrix)
    print("F1 (macro): %f" % f1_score(gleen, final, average='macro'))
    print("F1 (micro): %f" % f1_score(gleen, final, average='micro'))
    print("F1 (weighted): %f" % f1_score(gleen, final, average='weighted'))

    print("## Amber")
    print(amber_matrix)
    print("F1 (macro): %f" % f1_score(amber, final, average='macro'))
    print("F1 (micro): %f" % f1_score(amber, final, average='micro'))
    print("F1 (weighted): %f" % f1_score(amber, final, average='weighted'))


if __name__ == "__main__":
    #  calculate_human_prediction_F1()

    train_models()

    # a_test = ModelATester()
    # a_test.run_test()

    # b_test = CrossValidationModelBTest()
    # b_test.run_test()

    # word2vec = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
    #
    # c_test = CrossValidationModelCTest(preloaded_word2vec=word2vec, include_generated_answers=False)
    # c_test.run_test()

    # generate_answers()
