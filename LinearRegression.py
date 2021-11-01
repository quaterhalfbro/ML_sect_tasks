import numpy as np

class LinearRegression:    
    def __init__(self):
        self.weights = []

    def fit(self, x: np.array, y: np.array, l2_regulation=0):
        x_with_b = np.c_[np.ones(x.shape[0]), x]
        x_t = np.transpose(x_with_b)
        r_matrix = np.zeros([x_t.shape[0], x_t.shape[0]], int)
        np.fill_diagonal(r_matrix, l2_regulation)
        try:
            self.weights = np.dot(np.dot(np.linalg.inv(np.dot(x_t, x_with_b) + r_matrix), x_t), y)
        except np.linalg.LinAlgError:
            return -1
        return self.score(x, y)

    '''def gradient_fit(self, x: np.array, y: np.array, learning_rate, epochs_count):
        self.weights = np.ones(x.shape[1])
        for i in range(epochs_count):
            gradient = 2 * np.dot(np.transpose(x), (np.dot(x, self.weights) - y))
            self.weights -= learning_rate * gradient
        return self.score(x, y)'''

    def predict(self, x: np.array):
        x = np.c_[np.ones(x.shape[0]), x]
        return np.dot(x, self.weights)

    def r(self, x: np.array, y: np.array):
        return np.sum((self.predict(x) - y) ** 2)

    def score(self, x: np.array, y: np.array):
        y_mean = y.mean()
        return 1 - self.r(x, y) / np.sum((y - y_mean) ** 2)

    def cross_val_score(self, x: np.array, y: np.array, cv=5):
        mean_score = 0
        for i in range(cv):
            train_x = np.delete(x, range(cv * i, cv * i + cv), axis=0)
            train_y = np.delete(y, range(cv * i, cv * i + cv))
            test_x = x[cv * i:cv * i + cv]
            test_y = y[cv * i:cv * i + cv]
            result = self.fit(train_x, train_y)
            if result == -1:
                return -1
            mean_score += self.score(test_x, test_y)
        mean_score /= cv
        return mean_score

    def ransac_fit(self, x: np.array, y: np.array, radius=1):
        self.fit(x, y)
        points = abs(self.predict(x) - y) < radius
        inlaers_x = x[points]
        inlaers_y = y[points]
        score = self.fit(inlaers_x, inlaers_y)
        return (score, points)
