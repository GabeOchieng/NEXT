import time
import random
import numpy as np
from next.utils import debug_print
import os
import json
from ..lib import SVM


class MulticlassClassificationPrototype:
    def __init__(self):
        self.alg_label = 'Prototype'

    def initExp(self, butler, cache_size):
        t0 = time.time()
        butler.algorithms.set(key='num_reported_answers', value=0)
        butler.algorithms.set(key='cache_size', value=cache_size)
        butler.algorithms.set(key='query_cache', value=[])
        butler.algorithms.set(key='test_accuracy', value=[])
        butler.algorithms.set(key='labels', value=[])
        butler.algorithms.set(key='scores', value=[])

        return True

    def getQuery(self, butler, participant_uid):

        butler.job('getQueryCache', {})

        labels = self._get_labels(butler)
        test_indices = butler.experiment.get(key='test_indices')
        unlabeled_test = [i for i in test_indices if i not in labels]

        if unlabeled_test:
            return random.choice(unlabeled_test)
        else:
            wait = 0
            index = None
            while True:
                try:
                    index = butler.algorithms.pop(key='query_cache')
                except:
                    pass
                if index is not None:
                    break
                butler.job('getQueryCache', {})
                wait += 1
                time.sleep(wait)

        return index

    def getQueryCache(self, butler, args):
        debug_print("Attempting to call getQueryCache but it's not implemented.")
        raise NotImplementedError

    def processAnswer(self, butler, index, label):

        # Increment the number of reported answers by one
        num_answers = butler.algorithms.increment(key='num_reported_answers')

        test_indices = butler.experiment.get(key='test_indices')
        if index in test_indices:
            butler.experiment.append(key='test_labels', value=(index, label))
        else:
            butler.algorithms.append(key='labels', value=(index, label))
            butler.job('score', {})

        debug_print(butler.algorithms.get(key='scores'))

        return True

    def score(self, butler, args):

        labels = self._get_labels(butler)
        test_indices = set(butler.experiment.get(key='test_indices'))
        n = butler.experiment.get(key='n')

        train_indices = [i for i in labels if i not in test_indices]

        X = self._get_X(butler)

        test_indices = list(test_indices)
        X_test = X[test_indices]
        X_train = X[train_indices]

        classes = butler.experiment.get(key='args')['classes']

        for i in xrange(len(classes)):
            y_train = self._get_y(butler, labels, train_indices, i)
            y_test = self._get_y(butler, labels, test_indices, i)
            svm = SVM().fit(X_train, y_train)
            if svm:
                accuracy = svm.score(X_test, y_test)
                butler.algorithms.append(key='scores', value=(len(train_indices), i, accuracy))

    def _get_labels(self, butler):

        exp_args = butler.experiment.get(key='args')
        label_mode = exp_args['label_mode']
        classes = exp_args['classes']

        raw_labels = butler.algorithms.get(key='labels')
        raw_labels += butler.experiment.get(key='test_labels')

        labels = {}

        for index, label in raw_labels:
            labels[index] = labels.get(index, []) + [label]

        processed_labels = {}

        for i, label in labels.items():
            if label_mode == 'onehot':
                label = [l.index(1) for l in label]
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

    def _get_y(self, butler, labels, indices, class_):

        label_mode = butler.experiment.get(key='args')['label_mode']
        if label_mode == 'onehot':
            return np.array([1 if labels[i] == class_ else 0 for i in indices])
        else:
            return np.array([labels[i][class_] for i in indices])

    def _get_X(self, butler):
        targets = butler.targets.get_targetset(butler.exp_uid)
        features = [t['features'] for t in targets]
        return np.array(features)

    def getModel(self, butler):
        return True
