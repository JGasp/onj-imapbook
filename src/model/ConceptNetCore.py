from typing import Dict, List

import requests
import pickle
import os.path


class ConceptNetEdge:
    def __init__(self, _id, label, depth, weight):
        self._id = _id
        self.label = label
        self.depth = depth
        self.weight = weight


class ConceptNetClient:
    def __init__(self):
        self.uri = 'http://api.conceptnet.io'

    def query(self, word, max_depth=3, max_items=1000):
        edges = []

        words_to_resolve = [word]

        for depth in range(max_depth):

            next_batch = []

            for w in words_to_resolve:
                req = requests.get(self.uri + '/c/en/' + w).json()

                for e in req['edges']:
                    if e['start']['label'] == word:
                        node = e['end']
                    else:
                        node = e['start']

                    next_batch.append(node['label'])
                    edges.append(ConceptNetEdge(node['@id'], node['label'], depth, e['weight']))

            depth += 1
            words_to_resolve = next_batch
            if len(edges) >= max_items:
                break

        return edges

class ConceptNetStorage:
    def __init__(self):
        self.file_name = 'concept_net.data'

    def persist(self, concept_net_data: Dict[str, List[ConceptNetEdge]]):
        with open(self.file_name, 'wb') as output:
            pickle.dump(concept_net_data, output, pickle.HIGHEST_PROTOCOL)

    def load(self):
        if os.path.isfile(self.file_name):
            return pickle.load(self.file_name)
        else:
            return {}


class ConceptNetCore:
    def __init__(self):
        self.storage = ConceptNetStorage()
        self.client = ConceptNetClient()

        self.concept_net_data = self.storage.load()

    def get_data(self, word):
        if word in self.concept_net_data:
            return self.concept_net_data[word]
        else:
            data = self.client.query(word)
            self.concept_net_data[word] = data
            return data
