import json

from model.custom.ModelA import ModelA
from model.custom.ModelB import ModelB
from model.custom.ModelC import ModelC

from flask import Flask, url_for
from flask import json
from flask import request
from flask import Response


# Initialize all models
model_a = ModelA()
model_a.load()

model_b = ModelB()
model_b.load()

model_c = ModelC()
model_c.load(max_ans_graphs=5)


def evaluate_request(model_id, question, answer):
    if model_id == 'A':
        return model_a.make_prediction(question, answer)
    elif model_id == 'B':
        return model_b.make_prediction(question, answer)
    elif model_id == 'C':
        return model_c.make_prediction(question, answer)
    else:
        return None


app = Flask(__name__)


@app.route('/predict', methods = ["POST"])
def api_articles():
    content = json.dumps(request.json)
    print("Request: " + content)

    json_content = request.json

    if 'modelId' in json_content:
        predicted = evaluate_request(json_content['modelId'], json_content['question'], json_content['questionResponse'])

        data = {
            "score": predicted,
            "probability": None
        }
        js = json.dumps(data)

        return Response(js, status=200, mimetype='application/json')
    else:
        return Response(status=400, mimetype='application/json')


if __name__ == '__main__':
    app.run(port=8080)
