import socketserver
import json

from model.custom.ModelA import ModelA
from model.custom.ModelB import ModelB
from model.custom.ModelC import ModelC


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


class ModelRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = self.request.recv(2048).strip()
        request = json.load(data)

        predicted = evaluate_request(request['modelId'], request['question'], request['questionResponse'])

        response = json.dump({"score": predicted}, fp=lambda o: o.__dict__)
        self.request.sendall(response.encode())


if __name__ == "__main__":
    HOST, PORT = "localhost", 8080

    tcp_server = socketserver.TCPServer((HOST, PORT), ModelRequestHandler)
    tcp_server.serve_forever()
