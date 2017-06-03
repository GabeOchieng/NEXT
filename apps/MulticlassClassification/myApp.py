from next.utils import debug_print
from next.apps.SimpleTargetManager import SimpleTargetManager
import numpy as np
import random
import time
import json
import pandas as pd


class MyApp(object):
    def __init__(self, db):
        self.app_id = 'MulticlassClassification'
        self.TargetManager = SimpleTargetManager(db)

    def initExp(self, butler, init_algs, args):

        target_set = args['targets']

        if isinstance(target_set, (str, unicode)):
            try:
                with open(target_set) as f:
                    target_set = json.load(f)
            except ValueError:
                target_set = pd.read_json(target_set, lines=True)
                target_set = target_set.to_dict(orient='records')

        self.TargetManager.set_targetset(butler.exp_uid, target_set)

        n = len(target_set)

        test_indices = np.random.choice(n, args['test_size'])

        butler.experiment.set(key='test_indices', value=test_indices)
        butler.experiment.set(key='test_responses', value=[])
        butler.experiment.set(key='test_labels', value=[])
        butler.experiment.set(key='train_labels', value=[])
        butler.experiment.set(key='n', value=n)

        init_algs({'cache_size': args['cache_size'], 'n_classes': len(args['classes'])})

        del args['targets']
        return args

    def getQuery(self, butler, alg, args):

        index = alg()
        target = self.TargetManager.get_target_item(butler.exp_uid, index)

        # default to text
        if 'type' not in target:
            target['type'] = 'text'

        if target['type'] == 'text':
            # Fix newlines and tabs with the appropriate html
            target['data'] = target['text'].replace("\n", "<br>").replace("\t", "&emsp;")

        exp_args = butler.experiment.get(key='args')
        target['label_mode'] = exp_args['label_mode']
        target['classes'] = exp_args['classes']

        return target

    def processAnswer(self, butler, alg, args):

        query = butler.queries.get(uid=args['query_uid'])
        alg_args = {'index': query['target_id'], 'label': args['label']}
        alg(alg_args)

        return alg_args 

    def getModel(self, butler, alg, args):
        return alg({})

    def chooseAlg(self, butler, algs, getQueryArgs):
        return random.choice(algs)

