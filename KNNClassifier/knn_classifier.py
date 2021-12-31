import numpy as np
from typing import Dict


class KNNClassifier:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = x, y
        self.n_neighbors = min(self.n_neighbors, self.x.shape[0])

    def predict(self, x: np.ndarray) -> np.ndarray:
        distances = np.array([np.sum((self.x - i) ** 2, axis=1) for i in x])
        nearest_neighbors = self.y[np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]]
        y_pred = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            unique, counts = np.unique(nearest_neighbors[i], return_counts=True)
            y_pred[i] = unique[np.argmax(counts)]
        return y_pred

    def predict_proba(self, x:np.ndarray) -> np.ndarray:
        distances = np.array([np.sum((self.x - i) ** 2, axis=1) for i in x])
        nearest_neighbors = self.y[np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]]
        classes = np.unique(self.y)
        y_pred = np.zeros((x.shape[0], classes.shape[0])).astype('float64')
        classes_ind = dict()
        for i in range(classes.shape[0]):
            classes_ind[classes[i]] = i
        for i in range(x.shape[0]):
            unique, counts = np.unique(nearest_neighbors[i], return_counts=True)
            for obj, count in zip(unique, counts):
                y_pred[i, classes_ind[obj]] = count / self.n_neighbors
        return y_pred

    def score(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
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
