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

        args['test_indices'] = np.random.choice(n, args['test_size'])

        # Defaults loaded from myApp.yaml are floats, which can cause errors
        args['batch_size'] = int(args['batch_size'])

        alg_data = {'batch_size': args['batch_size'],
                    'n': n}

        init_algs(alg_data)

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
        label = args['label']
        butler.experiment.increment(key='num_reported_answers_for_' + query['alg_label'])

        if query['label_mode'] == 'onehot':
            label = label.index(1)

        test_indices = butler.experiment.get(key='args')['test_indices']
        if query['target_id'] in test_indices:
            butler.experiment.append(key='test_labels', value=('target_id', label))

        alg_args = {'index': query['target_id'], 'label': label}
        alg(alg_args)
        return alg_args 

    def getModel(self, butler, alg, args):
        return alg({})
