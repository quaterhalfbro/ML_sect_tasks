import numpy as np

class LogisticalRegression:
    def __init__(self):
        self.weights = []

    def fit(self, x, y, epochs, lr=0.001):
        x_b = np.c_[np.ones(x.shape[0]), x]
        self.weights = np.zeros(x_b.shape[1])
        for i in range(epochs):
            pred = self.predict(x)
            rs = y * (1 - pred) + (y - 1) * pred
            rs = rs.reshape(rs.shape[0], 1).repeat(x_b.shape[1], axis=1)
            gradient = (x_b * rs).sum(axis=0)
            self.weights += gradient * lr

    def predict(self, x):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        x = np.c_[np.ones(x.shape[0]), x]
        return sigmoid(x @ self.weights)

    def score(self, x, y, metric='r2'):
        pred = self.predict(x)
        tp = len(np.where((np.round(pred) == 1) & (y == 1))[0])
        tn = len(np.where((np.round(pred) == 0) & (y == 0))[0])
        fn = len(np.where((np.round(pred) == 1) & (y == 0))[0])
        fp = len(np.where((np.round(pred) == 0) & (y == 1))[0])
        if metric == 'precision':
            return tp / (tp + fp)
        if metric == 'recall':
            return tp / (tp + fn)
        if metric == 'f1':
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * precision * recall / (precision + recall)
        if metric == 'acc':
            return (tp + tn) / (tp + tn + fn + fp)
        return 1 - np.sum(np.power(pred - y, 2)) / (y.var() * len(y))
