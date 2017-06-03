import time
import random
import numpy as np
from next.utils import debug_print
import os
import json
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
import cPickle as pickle


class MulticlassClassificationPrototype:
    def __init__(self):
        self.alg_label = 'Prototype'

    def initExp(self, butler, cache_size, n_classes):
        butler.algorithms.set(key='num_reported_answers', value=0)
        butler.algorithms.set(key='cache_size', value=cache_size)
        butler.algorithms.set(key='query_cache', value=[])
        butler.algorithms.set(key='responses', value=[])
        for i in range(n_classes):
            butler.algorithms.set(key='scores_class{}'.format(i), value=[])
        butler.algorithms.set(key='processed_train_labels', value=[])
        butler.algorithms.set(key='processed_test_labels', value=[])

        # to cache the features
        self.get_X(butler)

        return True

    def getQuery(self, butler):

        butler.job('getQueryCache', {})

        y_train, y_test, labeled_train, labeled_test = self.get_y(butler)
        train_labels = list(dict(zip(labeled_train, y_train.tolist())).items())
        test_labels = list(dict(zip(labeled_test, y_test.tolist())).items())
        butler.experiment.set(key='train_labels', value=train_labels)
        butler.experiment.set(key='test_labels', value=test_labels)

        labeled_test = set(labeled_test)
        test_indices = butler.experiment.get(key='test_indices')
        unlabeled_test = [i for i in test_indices if i not in labeled_test]

        if unlabeled_test:
            return random.choice(unlabeled_test)
        else:
            wait = .5
            index = None
            while True:
                try:
                    index = butler.algorithms.pop(key='query_cache')
                except:
                    pass
                if index is not None:
                    break
                butler.job('getQueryCache', {})
                time.sleep(wait)
                if wait < 10:
                    wait += .5

        return index

    def getQueryCache(self, butler, args):
        debug_print("Attempting to call getQueryCache but it's not implemented.")
        raise NotImplementedError

    def processAnswer(self, butler, index, label):

        # Increment the number of reported answers by one
        num_answers = butler.algorithms.increment(key='num_reported_answers')

        test_indices = set(butler.experiment.get(key='test_indices'))
        if index in test_indices:
            butler.experiment.append(key='test_responses', value=(index, label))
        else:
            butler.algorithms.append(key='responses', value=(index, label))
            #  score every 5 responses
            if num_answers % 5 == 4:
                butler.job('score', {})

        return True

    def score(self, butler, args):

        lock = butler.algorithms.memory.lock('scoring')
        lock.acquire()

        y_train, y_test, labeled_train, labeled_test = self.get_y(butler)
        if y_train is None or len(labeled_train) < 2:
            lock.release()
            return

        old_scores = butler.algorithms.get(key='scores_class0')
        if old_scores and old_scores[-1][0] == len(labeled_train):
            lock.release()
            return

        X = self.get_X(butler)

        X_test = X[labeled_test]
        X_train = X[labeled_train]

        classes = butler.experiment.get(key='args')['classes']

        for i in xrange(len(classes)):
            if len(set(y_train[:, i].tolist())) == 2:
                svm = LinearSVC(class_weight='balanced').fit(X_train, y_train[:, i])
                precision = average_precision_score(y_test[:, i], svm.decision_function(X_test))
                # checks for nan
                if precision != precision:
                    precision = 0

                butler.algorithms.append(key='scores_class{}'.format(i), value=(len(labeled_train), precision))

        lock.release()

    def get_y(self, butler):
        """
        returns
        
        y: an (# labeled) x (# classes) binary array indicating labels
        if label_mode == 'onehot', each row will have exactly one 1
        else each row can have any number of 1s
        
        labeled_ids: the indices that each row of y corresponds to
        """

        exp_args = butler.experiment.get(key='args')
        label_mode = exp_args['label_mode']
        classes = exp_args['classes']

        responses = butler.algorithms.get(key='responses')
        test_responses = butler.experiment.get(key='test_responses')

        train_labels = {}
        train_ids = []
        test_labels = {}
        test_ids = []

        for index, label in responses:
            train_labels[index] = train_labels.get(index, []) + [label]
            train_ids.append(index)

        for index, label in test_responses:
            test_labels[index] = test_labels.get(index, []) + [label]
            test_ids.append(index)

        y_train_raw = [train_labels[i] for i in train_ids]
        y_test_raw = [test_labels[i] for i in test_ids]

        if label_mode == 'onehot':
            def process_label(label):
                label = np.array(label).sum(axis=0)
                out_label = np.zeros(len(classes))
                out_label[label.argmax()] = 1
                return out_label
        else:  # label_mode == 'multilabel'
            def process_label(label):
                return (np.array(label).mean(axis=0) >= .5).astype(int)

        y_train = np.array(list(map(process_label, y_train_raw)))
        y_test = np.array(list(map(process_label, y_test_raw)))

        assert len(y_train) == 0 or y_train.shape == (len(train_ids), len(classes))
        assert len(y_test) == 0 or y_test.shape == (len(test_ids), len(classes))

        return y_train, y_test, train_ids, test_ids

    def get_X(self, butler):
        """
        Gets the feature matrix for all targets.
        """
        lock = butler.experiment.memory.lock('features_lock')
        lock.acquire()
        if butler.experiment.memory.exists(key='features'):
            features = pickle.loads(str(butler.experiment.memory.get(key='features')))
        else:
            targets = butler.targets.get_targetset(butler.exp_uid)
            features = [t['features'] for t in targets]
            features = np.array(features)
            butler.experiment.memory.set(key='features', value=pickle.dumps(features))
        lock.release()
        return features

    def getModel(self, butler):
        return True
