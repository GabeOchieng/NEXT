import time
import random
import numpy as np
from sklearn.multiclass import OneVsRestClassifier as ovr
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from next.utils import debug_print
import os
import json

VERBOSE = 1


class MulticlassClassificationPrototype:
    def __init__(self):
        self.alg_label = 'Prototype'

    def initExp(self, butler, batch_size, n):
        t0 = time.time()
        butler.algorithms.set(key='num_reported_answers', value=0)
        butler.algorithms.set(key='batch_size', value=batch_size)
        butler.algorithms.set(key='query_batch', value=[])
        butler.algorithms.set(key='test_auc', value=[])
        butler.algorithms.set(key='past_auc', value=[])
        butler.algorithms.set(key='last_cache', value=[])
        butler.algorithms.set(key='n', value=n)

        return True

    def getQuery(self, butler, participant_uid):



        clf = self.get_clf_(butler)

        t0 = time.time()

        test_n = butler.algorithms.get(key='test_n')
        query_cache = butler.algorithms.get(key='query_cache')
        query_reserve = butler.algorithms.get(key='query_reserve')

        if not query_cache:
            butler.job('getQueryCache', {})
        if not query_reserve:
            butler.job('getQueryReserve', {})

        i = 1
        subset = None
        query = None
        while subset is None or query is None:
            while not (query_reserve or query_cache):
                query_reserve = butler.algorithms.get(key='query_reserve')
                query_cache = butler.algorithms.get(key='query_cache')
                time.sleep(.5 * i)
                if VERBOSE:
                    debug_print("Waiting for queries" + "." * i)
                i += 1
            if query_cache and query_reserve:
                test_labels = self.get_labels_(butler, subset='test')
                if len(test_labels) == test_n:
                    target = .995
                else:
                    # TODO tune this parameter?
                    target_n = 50.
                    target = 1 / (1 + np.exp(-len(test_labels) / target_n))
                if VERBOSE > 1:
                    debug_print("Probability of randomly grabbing a test over train query: {}".format(1 - target))
                if np.random.rand() > target:
                    subset = 'test'
                else:
                    subset = 'train'
            elif query_cache:
                subset = 'train'
            elif query_reserve:
                subset = 'test'

            if subset == 'train':
                query = self.grab_cache_(butler)
            elif subset == 'test':
                query = self.grab_reserve_(butler)

        if VERBOSE:
            query_cache = butler.algorithms.get(key='query_cache')
            query_reserve = butler.algorithms.get(key='query_reserve')
            debug_print("getQuery() successful: got a {} query. Took {}s.\nStatus: {} in cache, "
                        "{} in reserve".format(subset, time.time() - t0, len(query_cache), len(query_reserve)))

        return subset, query

    def grab_cache_(self, butler):

        exp_args = butler.experiment.get(key='args')
        train_n = butler.algorithms.get(key='train_n')
        batch_size = exp_args['batch_size']

        query_cache = butler.algorithms.get(key='query_cache')
        if not query_cache:
            # This should only happen if two users try to grab last query at same time... very rare!
            query_cache = self.getQueryCache(butler, {'train_n': train_n})
        if len(query_cache) == batch_size:
            butler.job('run_analysis', {})
        query = query_cache.pop(0)
        butler.algorithms.set(key='query_cache', value=query_cache)
        if not query_cache:
            if VERBOSE > 1:
                debug_print("  query_cache empty, submitting job for getQueryCache()")
            butler.job('getQueryCache', {})
        return query

    def grab_reserve_(self, butler):

        query_reserve = butler.algorithms.get(key='query_reserve')
        test_n = butler.algorithms.get(key='test_n')
        exp_args = butler.experiment.get(key='args')
        reserve_size = exp_args['reserve_size']

        if not query_reserve:
            query_reserve = self.getQueryReserve(butler, {})
        query = query_reserve.pop(0)
        butler.algorithms.set(key='query_reserve', value=query_reserve)

        if len(query_reserve) < (min(reserve_size, test_n) // 2):
            if VERBOSE > 1:
                debug_print("  query_reserve at half capacity, submitting job for getQueryReserve()")
            butler.job('getQueryReserve', {})

        return query

    def getQueryCache(self, butler, args):
        raise NotImplementedError

    def getQueryReserve(self, butler, args):
        t0 = time.time()
        query_reserve = butler.algorithms.get(key='query_reserve')
        reserve_size = butler.algorithms.get(key='reserve_size')
        test_n = butler.algorithms.get(key='test_n')
        # Check if this is a duplicate job
        if len(query_reserve) > (min(reserve_size, test_n) // 2):
            return query_reserve
        test_labels = self.get_labels_(butler, subset='test')
        unlabeled = [i for i in xrange(test_n) if i not in test_labels]
        if len(unlabeled) >= reserve_size:
            query_reserve = random.sample(unlabeled, reserve_size)
        elif test_n > reserve_size:
            query_reserve = unlabeled + random.sample(xrange(test_n), reserve_size - len(unlabeled))
        else:
            query_reserve = np.random.permutation(xrange(test_n)).tolist()

        query_reserve_old = butler.algorithms.get(key='query_reserve')
        if len(query_reserve_old) > (min(reserve_size, test_n) // 2):
            return query_reserve_old

        butler.algorithms.set(key='query_reserve', value=query_reserve)
        if VERBOSE:
            debug_print("getQueryReserve() successful: got {} test queries for reserve."
                        " Took {}s".format(len(query_reserve), time.time() - t0))

        return query_reserve

    def processAnswer(self, butler, index, subset, label):
        t0 = time.time()

        # Increment the number of reported answers by one
        num_answers = butler.algorithms.increment(key='num_reported_answers')

        # Save labels every 5 answers?
        if (num_answers + 1) % 5 == 0:
            butler.job('save_labels', {})

        if VERBOSE:
            exp_args = butler.experiment.get(key='args')
            classes = exp_args['classes']
            label_mode = exp_args['label_mode']
            if label_mode == 'multilabel':
                pretty_label = '(' + ', '.join(k + ': ' + str(v) for k, v in zip(classes, label)) + ')'
            else:
                pretty_label = '({})'.format(classes[label])

            debug_print("processAnswer() successful: saved label {} for index {} of {} subset."
                        " Took {}s".format(pretty_label, index, subset, time.time() - t0))

        return True

    def get_labels_(self, butler, subset=None, threshold=.5):
        """
        Turns labels into what the sklearn classifiers expect
        """
        t0 = time.time()

        queries = butler.queries.get(pattern={'exp_uid': butler.exp_uid, 'alg_label': self.alg_label,
                                              'subset': subset, 'label': {'$exists': True}})

        labels = {}
        for q in queries:
            if q['index'] not in labels:
                labels[q['index']] = [q['label']]
            else:
                labels[q['index']].append(q['label'])

        processed_labels = {}

        exp_args = butler.experiment.get(key='args')
        label_mode = exp_args['label_mode']
        classes = exp_args['classes']

        for i, label in labels.items():
            if label_mode == 'onehot':
                max_count = max([label.count(j) for j in xrange(len(classes))])
                max_labels = [j for j in xrange(len(classes)) if label.count(j) == max_count]
                if len(max_labels) == 1:
                    processed_labels[i] = max_labels[0]
            elif label_mode == 'multilabel':
                label = np.array(label)
                raw_label = np.mean(label, axis=0)
                multilabel = [1 if l > threshold else 0 for l in raw_label]
                processed_labels[i] = multilabel

        if VERBOSE > 1:
            if processed_labels:
                sample = random.sample(processed_labels, 1)[0]
                sample_label = processed_labels[sample]
                if subset is None:
                    debug_print("get_labels_() successful: processed {} labels, labels look like: {}. "
                                "Took {}s".format(len(processed_labels), sample_label, time.time() - t0))
                else:
                    debug_print("get_labels_() successful: processed {} {} labels, labels look like: {}. "
                                "Took {}s".format(len(processed_labels), subset, sample_label, time.time() - t0))
            else:
                debug_print("get_labels_() unsuccessful: no labels found. Took {}s".format(time.time() - t0))

        return processed_labels

    def run_analysis(self, butler, args):
        t0 = time.time()
        exp_args = butler.experiment.get(key='args')
        X_train = butler.targets.get_features(butler.exp_uid, 'train')
        train_labels = self.get_labels_(butler, subset='train')
        X_test = butler.targets.get_features(butler.exp_uid, 'test')
        test_labels = self.get_labels_(butler, subset='test')

        n_train_labels = len(train_labels)

        labeled = train_labels.keys()
        if not self.can_fit_(butler, train_labels=train_labels, label_mode=exp_args['label_mode']):
            if VERBOSE:
                debug_print("run_analysis() unsuccessful: not enough labels to fit classifier yet."
                            " Took {}s".format(time.time() - t0))
            return
        clf = self.get_clf_(butler)
        clf.fit(X_train[labeled], np.array(train_labels.values()))

        # optimize classifier
        # TODO do grid search to optimize classifier

        # get prediction stability
        t1 = time.time()
        proba_hist = butler.algorithms.get(key='proba_hist')
        proba_now = clf.predict_proba(X_test)
        butler.algorithms.set(key='proba_hist', value=proba_now)

        # TODO need to deal with changing `decision_function` shapes as more labels come in
        # Right now just kind of ignores it... might not be too bad?
        if proba_hist:
            proba_hist = np.array(proba_hist)
            if proba_hist.shape == proba_now.shape:
                proba_hist = np.abs(proba_hist - proba_now)
                proba_hist = np.max(proba_hist, axis=1)
                bin_count = 4
                bins = np.logspace(np.log(.001), np.log(.5), bin_count + 1, base=np.e)
                hist, _ = np.histogram(proba_hist, bins)

                butler.algorithms.append(key='proba_stability', value=(n_train_labels, hist))

                if VERBOSE > 1:
                    debug_print("proba_stability successful: looks like: {}."
                                " Took {}s".format(hist, time.time() - t1))
                del proba_hist
            else:
                if VERBOSE > 1:
                    debug_print("proba_stability unsuccessful: proba changed shape because of new labels."
                                " Took {}s".format(time.time() - t1))
        else:
            if VERBOSE > 1:
                debug_print("proba_stability unsuccessful: no proba_hist yet."
                            " Took {}s".format(time.time() - t1))

        clf.fit(X_train[labeled], np.array(train_labels.values()))

        dist_now = clf.decision_function(X_test)
        dist_now = self.fix_df(dist_now)

        # test confidence
        # TODO need to deal with changing shape of test confidence, this is broken as is
        t1 = time.time()
        test_confidence = np.mean(np.abs(dist_now), axis=0).tolist()

        butler.algorithms.append(key='test_confidence', value=(n_train_labels, test_confidence))
        if VERBOSE > 1:
            debug_print("test_confidence successful: looks like {}. Took {}s".format(test_confidence, time.time() - t1))
        del test_confidence

        # test auc
        t1 = time.time()
        if len(set(test_labels.values())) > 1:
            test_auc = self.get_auc_(clf, exp_args, dist_now[test_labels.keys()], test_labels)
            butler.algorithms.append(key='test_auc', value=(n_train_labels, test_auc))
            if VERBOSE > 1:
                debug_print("test_auc successful: looks like: {}. Took {}s".format(test_auc, time.time() - t1))
            del test_auc
        else:
            if VERBOSE > 1:
                debug_print("test_auc unsuccessful: not enough labels. Took {}s".format(time.time() - t1))
        del dist_now

        # past auc
        t1 = time.time()
        n_past = 100
        query_history = self.get_query_history_(butler, 'train')
        n_take = min(len(query_history) // 2, n_past)
        past_queries = set(query_history[-n_take:])
        past_train_labels = {}
        past_test_labels = {}
        for k, v in train_labels.items():
            if k in past_queries:
                past_train_labels[k] = v
            else:
                past_test_labels[k] = v
        # only get past_auc if (1) can fit and (2) more than one kind of label in holdout set
        if self.can_fit_(butler, train_labels=past_train_labels, label_mode=exp_args['label_mode']) \
                and len(set(past_test_labels.values())) > 1:
            clf.fit(X_train[past_train_labels.keys()], np.array(past_train_labels.values()))
            dist_past = self.fix_df(clf.decision_function(X_train[past_test_labels.keys()]))
            past_auc = self.get_auc_(clf, exp_args, dist_past, past_test_labels)
            butler.algorithms.append(key='past_auc', value=(n_train_labels, past_auc))
            if VERBOSE > 1:
                debug_print("past_auc successful: looks like: {}. Took {}s".format(past_auc, time.time() - t1))
            del dist_past
            del past_auc
        else:
            if VERBOSE > 1:
                debug_print("past_auc unsuccessful: not enough labels.  Took {}s".format(time.time() - t1))

        if VERBOSE:
            debug_print("run_analysis() successful. Took {}s".format(time.time() - t0))

    def fix_df(self, dist):
        """
        Puts clf.decision_function() in a consistent format, to deal with new classes emerging in training set.
        E.g. If only two classes, makes it 2 columns instead of 1.

        :param dist: np.ndarray
        :return: dist: np.ndarray
        """
        if len(dist.shape) == 1:
            dist = np.vstack([-dist, dist]).T
        elif dist.shape[1] == 1:
            dist = np.hstack([-dist, dist])

        return dist

    def get_auc_(self, clf, exp_args, proba, labels):
        shared_labels = set([c for c in clf.classes_ if c in set(labels.values())])
        trained_classes = clf.classes_.tolist()
        n_classes = len(exp_args['classes'])
        auc = []
        for c in xrange(n_classes):
            if c in shared_labels:
                if exp_args['label_mode'] == 'onehot':
                    labels_c = [1 if l == c else 0 for l in labels.values()]
                    i = trained_classes.index(c)
                else:
                    # multilabel mode
                    labels_c = [l[c] for l in labels.values()]
                if len(set(labels_c)) == 1:
                    auc.append(0)
                else:
                    score = roc_auc_score(labels_c, proba[:, i])
                    auc.append(score)
            else:
                auc.append(0)
        return auc

    def get_query_history_(self, butler, subset, sort=True):
        queries = butler.queries.get(pattern={'exp_uid': butler.exp_uid, 'alg_label': self.alg_label,
                                              'subset': subset, 'label': {'$exists': True}})

        # make sure only unique ones
        unique_queries = set()
        queries_filtered = []
        for q in queries:
            if q['index'] not in unique_queries:
                unique_queries.add(q['index'])
                queries_filtered.append(q)

        # TODO this sorts by query generated, not label received... need to do some timedelta operations to fix this
        if sort:
            queries = sorted(queries_filtered, key=lambda x: x['timestamp_query_generated'])

        queries = [q['index'] for q in queries]

        return queries

    def getModel(self, butler):
        return True

    def get_clf_(self, butler):
        """
        label_mode = butler.experiment.get(key='args')['label_mode']
        if label_mode == 'onehot':
            clf = LinearSVC()
        elif label_mode == 'multilabel':
            clf = ovr(LinearSVC())
        """
        clf = ovr(SVC(probability=True, decision_function_shape='ovr'))
        clf_params = butler.algorithms.get(key='clf_params')
        clf_params = dict([('estimator__' + k, v) for k, v in clf_params])
        clf.set_params(**clf_params)
        return clf

    def can_fit_(self, butler, train_labels=None, label_mode=None):
        if label_mode is None:
            label_mode = butler.experiment.get(key='args')['label_mode']
        if train_labels is None:
            train_labels = self.get_labels_(butler, subset='train')
        unique_labels = set()
        if label_mode == 'multilabel':
            # just need two different kinds of labels
            for v in train_labels.values():
                unique_labels.add(tuple(v))
                if len(unique_labels) > 1:
                    return True
        elif label_mode == 'onehot':
            for v in train_labels.values():
                unique_labels.add(v)
                if len(unique_labels) > 1:
                    return True
        return False

    def save_labels(self, butler, args):
        t0 = time.time()
        output_dir = '/rocket/data/output/{}/{}/'.format(butler.exp_uid, self.alg_label)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_labels = self.get_labels_(butler, subset='train')
        test_labels = self.get_labels_(butler, subset='test')

        train_label_file = os.path.join(output_dir, 'train_labels.json')
        test_label_file = os.path.join(output_dir, 'test_labels.json')
        all_label_file = os.path.join(output_dir, 'all_labels.json')

        targets = butler.targets.get_targets_(butler.exp_uid)
        train_target_map = {t['train_index']:t for t in targets if t['subset'] == 'train'}
        test_target_map = {t['test_index']:t for t in targets if t['subset'] == 'test'}
        train_output = {int(train_target_map[i]['raw_index']): l for i, l in train_labels.items()}
        test_output = {int(test_target_map[i]['raw_index']): l for i, l in test_labels.items()}
        all_output = train_output.copy()
        all_output.update(test_output)

        with open(train_label_file, 'w') as f:
            json.dump(train_output, f)
        with open(test_label_file, 'w') as f:
            json.dump(test_output, f)
        with open(all_label_file, 'w') as f:
            json.dump(all_output, f)

        if VERBOSE:
            debug_print("save_labels() successful: Saved {} train labels and {} test labels."
                        " Took {}s".format(len(train_output), len(test_output), time.time() - t0))
