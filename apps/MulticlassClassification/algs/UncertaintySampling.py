import time
import random
import numpy as np
from Prototype import MulticlassClassificationPrototype
from next.utils import debug_print
from sklearn.svm import LinearSVC


class MyAlg(MulticlassClassificationPrototype):
    def __init__(self):
        self.alg_label = 'UncertaintySampling'

    def getQueryCache(self, butler, args):

        lock = butler.algorithms.memory.lock('getting_query_cache')
        lock.acquire()

        # This is in case many jobs get submitted
        cache_size = butler.algorithms.get(key='cache_size')
        if len(butler.algorithms.get(key='query_cache')) >= cache_size:
            lock.release()
            return

        X = self.get_X(butler)
        test_indices = set(butler.experiment.get(key='test_indices'))
        n = butler.experiment.get(key='n')

        while len(butler.algorithms.get(key='query_cache')) < cache_size:

            y_train, _, labeled_train, _ = self.get_y(butler)

            if len(labeled_train) > 1:
                y_train = random.choice(y_train.T)
            else:
                y_train = np.array([])  # not enough labeled, set to empty

            X_train = X[labeled_train]

            labeled_train = set(labeled_train)
            current_cache = set(butler.algorithms.get(key='query_cache'))

            unlabeled_train = [i for i in range(n) if (i not in labeled_train and
                                                       i not in test_indices and
                                                       i not in current_cache)]

            X_unlabeled = X[unlabeled_train]

            if len(set(y_train.tolist())) == 2:
                svm = LinearSVC(class_weight='balanced').fit(X_train, y_train)
                scores = svm.decision_function(X_unlabeled)
                n_to_append = int((cache_size - len(butler.algorithms.get(key='query_cache'))))
                scores_sorted = np.argsort(np.abs(scores))[:n_to_append]
                best = np.array(unlabeled_train)[scores_sorted]
                for b in best:
                    butler.algorithms.append(key='query_cache', value=b)
            else:
                # unsuccessful training, random
                butler.algorithms.append(key='query_cache', value=random.choice(unlabeled_train))

        lock.release()
