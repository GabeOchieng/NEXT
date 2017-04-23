import time
import random
import numpy as np
from rocket.RocketClassification.algs.Prototype import RocketClassificationPrototype
from next.utils import debug_print
from rocket.constants import VERBOSE


class MyAlg(RocketClassificationPrototype):
    def __init__(self):
        self.alg_label = 'UncertaintySampling'

    def getQueryCache(self, butler, args):

        # This is in case many jobs get submitted
        query_cache = butler.algorithms.get(key='query_cache')
        if query_cache:
            if VERBOSE:
                debug_print("Duplicate getQueryCache() jobs submitted, exiting this one...")
            return query_cache

        t0 = time.time()
        train_n = butler.algorithms.get(key='train_n')
        batch_size = butler.algorithms.get(key='batch_size')
        labels = self.get_labels_(butler, 'train')
        unlabeled = [i for i in xrange(train_n) if i not in labels]

        if len(labels) < 20:
            if VERBOSE:
                debug_print("Not enough labels to fit classifier - grabbing a random batch.")
            query_cache = random.sample(unlabeled, batch_size)
        elif len(unlabeled) <= batch_size:
            query_cache = unlabeled + random.sample(xrange(train_n), batch_size - len(unlabeled))
        elif train_n <= batch_size:
            query_cache = np.random.permutation(xrange(train_n)).tolist()
        else:
            labeled = labels.keys()
            if self.can_fit_(butler, labels):
                active_classes = set(butler.experiment.get(key='args')['active_classes'])
                active_i = [i for i, c in enumerate(butler.experiment.get(key='args')['classes']) if c in active_classes]
                X = butler.targets.get_features(butler.exp_uid, 'train')
                y = np.array(labels.values())
                labeled_classes = sorted(list(set(y)))
                active_labeled = [i for i in active_i if i in labeled_classes]
                if len(active_labeled) == 0:
                    query_cache = random.sample(unlabeled, batch_size)
                    if VERBOSE:
                        debug_print("No active classes are labeled yet - picking a random batch.")
                else:
                    clf = self.get_clf_(butler)
                    clf.fit(X[labeled], y)

                    dists = clf.decision_function(X[unlabeled])
                    active_labeled = [clf.classes_.tolist().index(i) for i in active_labeled]

                    if len(dists.shape) > 1 and dists.shape[1] > 1:
                        dists = dists[:, active_labeled]
                    dists = np.abs(dists)
                    dists[dists == 0] = 2
                    dists[dists == 1] = 2
                    # TODO min-min strategy - try different ones?
                    if len(dists.shape) > 1:
                        dists = np.min(dists, axis=1)
                    best = dists.argsort()
                    last_cache = set(butler.algorithms.get(key='last_cache'))
                    best = [i for i in best if unlabeled[i] not in last_cache]
                    best = best[:batch_size]
                    if dists[best[0]] == 2:
                        query_cache = random.sample(unlabeled, batch_size)
                    else:
                        query_cache = [unlabeled[i] for i in best]
            else:
                if VERBOSE:
                    debug_print("Not enough labels to fit classifier - grabbing a random batch.")
                query_cache = random.sample(unlabeled, batch_size)

        # Again, check if another job finished while this was running...
        query_cache_old = butler.algorithms.get(key='query_cache')
        if query_cache_old:
            if VERBOSE:
                debug_print("Duplicate getQueryCache() jobs submitted, exiting this one...")
            return query_cache_old

        butler.algorithms.set(key='last_cache', value=query_cache)
        butler.algorithms.set(key='query_cache', value=query_cache)

        if VERBOSE:
            debug_print("getQueryCache() successful: got cache: {}. Took {}s".format(query_cache, time.time() - t0))

        return query_cache
