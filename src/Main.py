import gensim
from model.CrossValidationTest import CrossValidationTest

if __name__ == "__main__":
    cv_test = CrossValidationTest()
    cv_test.run_test()

    # generate_answers()

    # model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
    #
    # buy_vec = model['buy']
    # purchase_vec = model['purchase']
    #
    # gravity_vec = model['gravity']
    # space_vec = model['space']
