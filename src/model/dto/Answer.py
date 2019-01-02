from model.dto.Graph import Graph
from model.dto.Mark import Mark


class Answer:
    def __init__(self, raw_text: str, gl: Mark=Mark.M0, am=Mark.M0, fm=Mark.M0):
        self.raw_text: str = raw_text
        self.text_graph: Graph = None

        self.gleen_mark: Mark = gl
        self.amber_mark: Mark = am
        self.final_mark: Mark = fm

    def build_graph(self, fun_graph_build):
        self.text_graph = fun_graph_build(self.raw_text)

    def copy(self):
        return Answer(self.raw_text, self.gleen_mark, self.amber_mark, self.final_mark)
