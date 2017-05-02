import time
import random
import numpy as np
from Prototype import MulticlassClassificationPrototype
from next.utils import debug_print
from ..lib import SVM


class MyAlg(MulticlassClassificationPrototype):
    def __init__(self):
        self.alg_label = 'UncertaintySampling'

    def getQueryCache(self, butler, args):

        # This is in case many jobs get submitted
        cache_size = butler.algorithms.get(key='cache_size')
        if len(butler.algorithms.get(key='query_cache')) >= cache_size:
            return

        labels = self._get_labels(butler)

        test_indices = set(butler.experiment.get(key='test_indices'))
        n = butler.experiment.get(key='n')

        train_indices = [i for i in xrange(n) if i not in test_indices]

        train_labeled = []
        train_unlabeled = []

        for i in train_indices:
            if i in labels:
                train_labeled.append(i)
            else:
                train_unlabeled.append(i)

        X = self._get_X(butler)
        X_labeled = X[train_labeled]
        X_unlabeled = X[train_unlabeled]

        exp_args = butler.experiment.get(key='args')
        label_mode = exp_args['label_mode']
        classes = exp_args['classes']

        active_class = np.random.choice(len(classes))

        y = self._get_y(butler, labels, train_labeled, active_class)

        svm = SVM().fit(X_labeled, y)

        if svm:
            scores = svm.decision_function(X_unlabeled)
            best = train_unlabeled[np.argmin(np.abs(scores))]
            butler.algorithms.append(key='query_cache', value=best)
        else:
            # unsuccessful training, random
            butler.algorithms.append(key='query_cache', value=random.choice(train_unlabeled))
