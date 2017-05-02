import numpy as np


class SVM(object):
    def __init__(self, C=.001, learning_rate=.05, max_iter=10000, tolerance=.05):
        self.C = C
        self.W = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X, y):
        if len(set(y)) != 2:
            return None
        X, y = self._fix_Xy(X, y)
        if min(np.count_nonzero(y - 1), np.count_nonzero(y + 1)) < 5:
            return None
        self.W = np.random.rand(X.shape[1])

        last_loss = np.inf

        for i in xrange(self.max_iter):
            j = np.random.choice(X.shape[0])
            self._iter(X[j], y[j])
            if i % 1000 == 0:
                this_loss = self._loss(X, y)
                print(this_loss)
                if np.abs(this_loss - last_loss) < self.tolerance:
                    break
                last_loss = this_loss

        return self

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def decision_function(self, X):
        X, _ = self._fix_Xy(X=X)
        return np.dot(X, self.W)

    def score(self, X, y):
        _, y = self._fix_Xy(y=y)
        y_hat = self.predict(X)
        return (y == y_hat).astype(int).mean()

    def _iter(self, x, y):
        update = np.zeros(self.W.shape)
        if (1 - y * np.dot(self.W, x)) > 0:
            update += -y * x
        update += 2 * self.C * self.W
        self.W -= self.learning_rate * update

    def _loss(self, X, y):
        return np.sum(np.max(np.vstack([np.zeros(X.shape[0]), 1 - np.multiply(y, np.dot(X, self.W))]), axis=0)) +\
               self.C * np.linalg.norm(self.W)**2

    def _fix_Xy(self, X=None, y=None):
        if y is not None:
            self._classes = sorted(list(set(y)))
            y = np.vectorize(lambda x: {self._classes[0]: -1, self._classes[1]: 1}[x])(y)
        if X is not None:
            X -= np.mean(X, axis=0)
            stdX = np.std(X, axis=0)
            stdX[stdX == 0] = 1
            X /= stdX
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        return X, y


def test_svm():
    X = np.random.rand(100, 3) - .5
    W_hat = np.random.rand(3) - .5
    y = np.sign(np.dot(X, W_hat))
    svm = SVM()
    svm.fit(X, y)
    print(svm.score(X, y))

    X = np.random.rand(9, 3) - .5
    W_hat = np.random.rand(3) - .5
    y = np.sign(np.dot(X, W_hat))
    svm = SVM()
    assert svm.fit(X, y) is None


if __name__ == '__main__':
    test_svm()
