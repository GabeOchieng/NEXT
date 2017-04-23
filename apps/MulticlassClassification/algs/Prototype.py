import time
import random
import numpy as np
from next.utils import debug_print
import os
import json
from ..lib import SVM

VERBOSE = 1


class MulticlassClassificationPrototype:
    def __init__(self):
        self.alg_label = 'Prototype'

    def initExp(self, butler, batch_size, n):
        t0 = time.time()
        butler.algorithms.set(key='num_reported_answers', value=0)
        butler.algorithms.set(key='batch_size', value=batch_size)
        butler.algorithms.set(key='query_batch', value=[])
        butler.algorithms.set(key='test_accuracy', value=[])
        butler.algorithms.set(key='n', value=n)

        return True

    def getQuery(self, butler, participant_uid):

        labels = self._get_labels(butler)
        test_indices = butler.experiment.get(key='args')['test_indices']
        unlabeled_test = [i for i in test_indices if i not in labels]

        if unlabeled_test:
            return random.choice(unlabeled_test)

    def getQueryCache(self, butler, args):
        raise NotImplementedError

    def processAnswer(self, butler, index, label):

        # Increment the number of reported answers by one
        num_answers = butler.algorithms.increment(key='num_reported_answers')

        return True

    def _get_labels(self, butler):
        """
        Turns labels into what the sklearn classifiers expect
        """
        queries = butler.queries.get(pattern={'exp_uid': butler.exp_uid, 'alg_label': self.alg_label,
                                              'label': {'$exists': True}})

        labels = {}
        for q in queries:
            if q['index'] not in labels:
                labels[q['index']] = [q['label']]
            else:
                labels[q['index']].append(q['label'])

        processed_labels = {}

        exp_args = butler.experiment.get(key='args')
        label_mode = exp_args['label_mode']
        classes = exp_args['classes']

        for i, label in labels.items():
            if label_mode == 'onehot':
                max_count = max([label.count(j) for j in xrange(len(classes))])
                max_labels = [j for j in xrange(len(classes)) if label.count(j) == max_count]
                if len(max_labels) == 1:
                    processed_labels[i] = max_labels[0]
            elif label_mode == 'multilabel':
                label = np.array(label)
                raw_label = np.mean(label, axis=0)
                multilabel = [1 if l > .5 else 0 for l in raw_label]
                processed_labels[i] = multilabel

        return processed_labels

    def getModel(self, butler):
        return True
