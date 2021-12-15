import numpy as np
from typing import List, Dict, Union


def entropy(y: np.ndarray) -> float:
    counts = np.unique(y, return_counts=True)[1] / len(y)
    return -np.sum(np.nan_to_num(counts * np.log2(counts), 0))


class NotFittedError(Exception):
    pass


class DecisionTree:
    def __init__(self, max_depth: int = 10, min_entropy: float = 0.1):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.splits = []
        self.split_number = 0

    def search_best_splits(self, x: np.ndarray, y: np.ndarray, cur_depth=0) -> List[Union[tuple, list]]:
        if cur_depth == self.max_depth or entropy(y) <= self.min_entropy:
            values, counts = np.unique(y, return_counts=True)
            return values[counts.argmax()]
        best_entropy = 2
        for feature in range(x.shape[1]):
            for split in np.unique(x[:, feature]):
                first_class = np.where(x[:, feature] >= split)[0]
                second_class = np.where(x[:, feature] < split)[0]
                y1 = y[first_class]
                y2 = y[second_class]
                entropy_value = (entropy(y1) * len(y1) + entropy(y2) * len(y2)) / len(y)
                if entropy_value >= best_entropy:
                    continue
                best_entropy = entropy_value
                class_1 = first_class
                class_2 = second_class
                best_split = (feature, split)
        return [best_split, self.search_best_splits(x[class_1], y[class_1], cur_depth + 1),
                self.search_best_splits(x[class_2], y[class_2], cur_depth + 1)]

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.splits = self.search_best_splits(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        def tree_predict(split, x):
            if type(split) is not list:
                return split
            if x[split[0][0]] >= split[0][1]:
                return tree_predict(split[1], x)
            return tree_predict(split[2], x)

        if self.splits == []:
            raise NotFittedError
        return np.array([tree_predict(self.splits, i) for i in x])

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
