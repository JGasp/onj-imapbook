from model import Data
from model.Data import generate_answers
from model.ModelTests import CrossValidationModelBTest, ModelATester, CrossValidationModelCTest
from model.custom.ModelA import ModelA
from model.custom.ModelB import ModelB
from model.custom.ModelC import ModelC


def train_models():
    model_a = ModelA()
    model_a.build()
    model_a.persist()

    model_b = ModelB()
    model_b.build()
    model_b.persist()

    model_c = ModelC()
    model_c.build()
    model_c.persist_concept_net_data()
    model_c.persist()


def calculate_human_prediction_F1():
    questions = Data.get_questions()


if __name__ == "__main__":
    # a_test = ModelATester()
    # a_test.run_test()

    # b_test = CrossValidationModelBTest()
    # b_test.run_test()

    model_c = ModelC()
    model_c.build()
    model_c.persist_concept_net_data()

    # c_test = CrossValidationModelCTest()
    # c_test.run_test()

    # generate_answers()
