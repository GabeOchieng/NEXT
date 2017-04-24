from next.utils import debug_print
from next.apps.SimpleTargetManager import SimpleTargetManager
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import random
import time
import json


class MyApp(object):
    def __init__(self, db):
        self.app_id = 'RocketClassification'
        self.TargetManager = SimpleTargetManager(db)

    def initExp(self, butler, init_algs, args):

        target_set = args['target_set']

        if isinstance(target_set, (str, unicode)):
            with open(target_set) as f:
                target_set = json.load(f)

        self.TargetManager.set_targetset(butler.exp_uid, target_set)

        n = len(target_set)

        test_indices = np.random.choice(n, args['test_size'])

        butler.experiment.set(key='test_indices', value=test_indices)
        butler.experiment.set(key='test_labels', value=[])

        # Defaults loaded from myApp.yaml are floats, which can cause errors
        init_algs({'cache_size': int(args['cache_size'])})

        return args

    def getQuery(self, butler, alg, args):

        index = alg({'participant_uid': args['participant_uid']})
        target = self.TargetManager.get_target_item(butler.exp_uid, index)

        if target['type'] == 'text':
            # Fix newlines and tabs with the appropriate html
            target['data'] = target['data'].replace("\n", "<br>").replace("\t", "&emsp;")

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
