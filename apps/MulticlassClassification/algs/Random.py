import time
import random
import numpy as np
from Prototype import MulticlassClassificationPrototype
from next.utils import debug_print


class MyAlg(MulticlassClassificationPrototype):
    def __init__(self):
        self.alg_label = 'Random'

    def getQueryCache(self, butler, args):

        # This is in case many jobs get submitted
        cache_size = butler.algorithms.get(key='cache_size')
        if len(butler.algorithms.get(key='query_cache')) >= cache_size:
            return

        n = butler.experiment.get(key='n')

        labels = self._get_labels(butler)
        unlabeled = [i for i in xrange(n) if i not in labels]

        butler.algorithms.append(key='query_cache', value=random.choice(unlabeled))
