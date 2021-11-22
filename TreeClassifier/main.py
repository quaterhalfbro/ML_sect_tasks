import numpy as np


class DecisionTree:
    def __init__(self, max_depth: int = 10, min_entropy: float = 0.1):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.splits = []
        self.split_number = 0

    def entropy(self, y: np.ndarray) -> float:
        counts = np.unique(y, return_counts=True)[1] / len(y)
        return -np.sum(np.nan_to_num(counts * np.log2(counts), 0))

    def search_best_splits(self, x: np.ndarray, y: np.ndarray, cur_depth=0) -> list:
        if cur_depth == self.max_depth or self.entropy(y) <= self.min_entropy:
            values, counts = np.unique(y, return_counts=True)
            return values[counts.argmax()]
        else:
            best_entropy = 2
            for feature in range(x.shape[1]):
                for split in np.unique(x[:, feature]):
                    first_class = np.where(x[:, feature] >= split)[0]
                    second_class = np.where(x[:, feature] < split)[0]
                    y1 = y[first_class]
                    y2 = y[second_class]
                    entropy = (self.entropy(y1) * len(y1) + self.entropy(y2) * len(y2)) / len(y)
                    if entropy < best_entropy:
                        best_entropy = entropy
                        class_1 = first_class
                        class_2 = second_class
                        best_split = lambda x: x[feature] >= split
            return [best_split, self.search_best_splits(x[class_1], y[class_1], cur_depth + 1),
                    self.search_best_splits(x[class_2], y[class_2], cur_depth + 1)]

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.splits = self.search_best_splits(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        def tree_predict(split, x):
            if type(split) is not list:
                return split
            if split[0](x):
                return tree_predict(split[1], x)
            return tree_predict(split[2], x)

        if self.splits == []:
            raise Exception('NotFittedError')
        return tree_predict(self.splits, x)


model = DecisionTree(10)
data_x = np.array([[1, 1, 2],
                  [1, 1, 1],
                  [1, 4, 3],
                  [2, 1, 1],
                  [1, 2, 2],
                  [3, 1, 1],
                  [2, 2, 1],
                  [2, 3, 1],
                  [2, 2, 2],
                  [3, 3, 1],
                  [3, 2, 2]])
data_y = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1])
model.fit(data_x, data_y)
print(model.predict(np.array([1, 1, 2])))
