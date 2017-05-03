import json
import next.utils as utils
import next.apps.SimpleTargetManager
import numpy as np
import time


class MyApp:
    def __init__(self, db):
        self.app_id = 'Tests'
        self.TargetManager = next.apps.SimpleTargetManager.SimpleTargetManager(db)

    def initExp(self, butler, init_algs, args):

        butler.experiment.set(key='experiment_foo', value='experiment_bar')
        assert np.all(np.equal(butler.namespace['namespace_ndarray'], np.arange(16).reshape(4, 4)))
        assert butler.namespace['namespace_foo'] == 'namespace_bar'

        init_algs({})

        return args

    def getQuery(self, butler, alg, args):

        assert butler.experiment.get(key='experiment_foo') == 'experiment_bar'
        assert np.all(np.equal(butler.namespace['namespace_ndarray'], np.arange(16).reshape(4, 4)))
        assert butler.namespace['namespace_foo'] == 'namespace_bar'

        assert alg()

        return {}

    def processAnswer(self, butler, alg, args):

        assert alg({})

        assert butler.experiment.get(key='experiment_foo') == 'experiment_bar'
        assert np.all(np.equal(butler.namespace['namespace_foo'], np.arange(16).reshape(4, 4)))

        return {}

    def getModel(self, butler, alg, args):

        assert alg()
        assert np.all(np.equal(butler.namespace['namespace_foo'], np.arange(16).reshape(4, 4)))

        return True

    def setupNamespace(self, namespace):

        worker_type = utils.get_worker_type()

        utils.debug_print("worker_type is {}".format(worker_type))
        if worker_type != 'dashboard':
            utils.debug_print("async or sync worker, loading namespace")
            utils.debug_print('running myApp.setupNamespace')
            namespace['namespace_foo'] = 'namespace_bar'
            namespace['namespace_ndarray'] = np.arange(16).reshape(4, 4)
        else:
            utils.debug_print("dashboard worker, not loading namespace")

        time.sleep(1)

        return namespace
