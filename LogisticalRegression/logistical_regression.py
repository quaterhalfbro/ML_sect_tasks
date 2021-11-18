import numpy as np

class LogisticalRegression:
    def __init__(self):
        self.weights = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: int = 0.001):
        x_b = np.c_[np.ones(x.shape[0]), x]
        self.weights = np.zeros(x_b.shape[1])
        for i in range(epochs):
            pred = self.predict(x)
            rs = y * (1 - pred) + (y - 1) * pred
            rs = rs.reshape(rs.shape[0], 1).repeat(x_b.shape[1], axis=1)
            gradient = (x_b * rs).sum(axis=0)
            self.weights += gradient * lr

    def predict(self, x: np.ndarray) -> np.ndarray:
        try:
            x = np.c_[np.ones(x.shape[0]), x]
            return self.sigmoid(x @ self.weights)
        except IndexError:
            return np.random.rand(x.shape[0])

    def score(self, x: np.ndarray, y: np.ndarray) -> dict:
        pred = self.predict(x)
        metrics = {'acc': 0, 'r2': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        tp = len(np.where((np.round(pred) == 1) & (y == 1))[0])
        tn = len(np.where((np.round(pred) == 0) & (y == 0))[0])
        fn = len(np.where((np.round(pred) == 1) & (y == 0))[0])
        fp = len(np.where((np.round(pred) == 0) & (y == 1))[0])
        metrics['precision'] = tp / (tp + fp)
        metrics['recall'] = tp / (tp + fn)
        metrics['f1'] = (2 * metrics['precision'] * metrics['recall'] /
                        (metrics['precision'] + metrics['recall']))
        metrics['acc'] = (tp + tn) / (tp + tn + fn + fp)
        metrics['r2'] = 1 - np.sum(np.power(pred - y, 2)) / (y.var() * len(y))
        return metrics
