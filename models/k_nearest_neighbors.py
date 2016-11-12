import numpy as np


class KNearestNeighbors:

    def __init__(self, k=1):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # compute distances
        dists = np.sqrt(np.sum(X**2, axis=1, keepdims=True)
                        - 2 * X.dot(self.X_train.T)
                        + np.sum(self.X_train**2, axis=1))

        num_X = X.shape[0]
        y_pred = np.zeros(num_X)
        for i in xrange(num_X):
            closest_y = self.y_train[np.argsort(dists[i])[:self.k]]
            counts = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts)

        return y_pred
