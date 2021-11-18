import numpy as np


class LinearRegression:
    def __init__(self):
        self.weights = []

    def fit(self, x: np.ndarray, y: np.ndarray, l2_regulation: int = 0):
        x_with_b = np.c_[np.ones(x.shape[0]), x]
        x_t = np.transpose(x_with_b)
        r_matrix = np.zeros([x_t.shape[0], x_t.shape[0]], int)
        np.fill_diagonal(r_matrix, l2_regulation)
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(x_t, x_with_b) + r_matrix), x_t), y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.c_[np.ones(x.shape[0]), x]
        return np.dot(x, self.weights)

    def square_error(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum((self.predict(x) - y) ** 2)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        y_mean = y.mean()
        return 1 - self.square_error(x, y) / np.sum((y - y_mean) ** 2)

    def cross_val_score(self, x: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
        mean_score = 0
        for i in range(cv):
            train_x = np.delete(x, range(cv * i, cv * i + cv), axis=0)
            train_y = np.delete(y, range(cv * i, cv * i + cv))
            test_x = x[cv * i:cv * i + cv]
            test_y = y[cv * i:cv * i + cv]
            result = self.fit(train_x, train_y)
            mean_score += self.score(test_x, test_y)
        mean_score /= cv
        return mean_score
