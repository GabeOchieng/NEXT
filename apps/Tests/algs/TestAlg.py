import numpy as np


class MyAlg:
    def __init__(self):
        self.alg_label = 'TestAlg'

    def initExp(self, butler, dummy):

        assert butler.experiment.get(key='experiment_foo') == 'experiment_bar'
        butler.algorithms.set(key='algorithms_foo', value='algorithms_bar')
        assert np.all(np.equal(butler.namespace['namespace_ndarray'], np.arange(16).reshape(4, 4)))
        assert butler.namespace['namespace_foo'] == 'namespace_bar'
        butler.job('background_task', {})

        return True

    def getQuery(self, butler):

        assert butler.experiment.get(key='experiment_foo') == 'experiment_bar'
        assert butler.algorithms.get(key='algorithms_foo') == 'algorithms_bar'
        assert np.all(np.equal(butler.namespace['namespace_ndarray'], np.arange(16).reshape(4, 4)))
        assert butler.namespace['namespace_foo'] == 'namespace_bar'
        butler.job('background_task', {})

        return True

    def processAnswer(self, butler):

        assert butler.experiment.get(key='experiment_foo') == 'experiment_bar'
        assert butler.algorithms.get(key='algorithms_foo') == 'algorithms_bar'
        assert np.all(np.equal(butler.namespace['namespace_ndarray'], np.arange(16).reshape(4, 4)))
        assert butler.namespace['namespace_foo'] == 'namespace_bar'
        butler.job('background_task', {})

        return True

    def getModel(self, butler):

        return True

    def background_task(self, butler, args):

        assert np.all(np.equal(butler.namespace['namespace_ndarray'], np.arange(16).reshape(4, 4)))
        assert butler.namespace['namespace_foo'] == 'namespace_bar'
