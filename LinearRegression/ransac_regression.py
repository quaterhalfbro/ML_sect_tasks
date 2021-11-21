import numpy as np
from linear_regression import LinearRegression


class RansacRegression(LinearRegression):
    def fit(self, x: np.ndarray, y: np.ndarray, radius: float = 1) -> (float, np.ndarray):
        super().fit(x, y)
        points = abs(self.predict(x) - y) < radius
        inlaers_x = x[points]
        inlaers_y = y[points]
        super().fit(inlaers_x, inlaers_y)
