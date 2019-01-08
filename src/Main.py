from model.CrossValidationTest import CrossValidationTest
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


if __name__ == "__main__":
    cv_test = CrossValidationTest()
    cv_test.run_test()

    # generate_answers()
