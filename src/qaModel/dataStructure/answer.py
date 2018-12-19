
class Answer:
    def __init__(self, value, graph, gl=0, am=0, fm=0):
        self.value: str = value
        self.graph = graph

        self.gleen_mark = gl
        self.amber_mark = am
        self.final_mark = fm
