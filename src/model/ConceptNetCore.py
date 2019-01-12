from typing import Dict, List

import requests
import pickle
import os.path
import re


class ConceptNetEdge:
    def __init__(self, _id, label, depth, weight):
        self._id = _id
        self.label = label
        self.depth = depth
        self.weight = weight

    def __str__(self):
        return self.label


class ConceptNetClient:
    def __init__(self):
        self.uri = 'http://api.conceptnet.io'

    def query(self, word, max_depth=1, max_items=100):
        edges = []

        words_to_resolve = [word]

        for depth in range(max_depth):

            next_batch = []

            for w in words_to_resolve:
                print("ConceptNetClient: Resolving word [%s]" % w)
                req = requests.get(self.uri + '/c/en/' + w).json()

                for e in req['edges']:
                    if e['start']['label'] == word:
                        node = e['end']
                    else:
                        node = e['start']

                    if 'language' in node:
                        if node['language'] == 'en':

                            if re.match(r'\A[\w-]+\Z', node['label']):
                                next_batch.append(node['label'])
                                edges.append(ConceptNetEdge(node['@id'], node['label'], depth, e['weight']))
                                if len(edges) >= max_items:
                                    break

            depth += 1
            words_to_resolve = next_batch
            if len(edges) >= max_items:
                break

        return edges


class ConceptNetStorage:
    def __init__(self):
        self.file_name = './model/concept_net.data'

    def persist(self, concept_net_data: Dict[str, List[ConceptNetEdge]]):
        with open(self.file_name, 'wb') as output:
            pickle.dump(concept_net_data, output, pickle.HIGHEST_PROTOCOL)

    def load(self):
        if os.path.isfile(self.file_name):
            with open(self.file_name, 'rb') as pickle_file:
                return pickle.load(pickle_file)
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

    def persist(self):
        self.storage.persist(self.concept_net_data)
