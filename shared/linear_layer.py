from shared.layer import Layer
import numpy as np

class LinearLayer(Layer):
    def __init__(self, name, weights, bias) -> None:
        super().__init__(name)
        self.weights: np.ndarray = weights
        self.bias: np.ndarray = bias

    def activate(self, x, y):
        return [x * self.weights[0][0] + y * self.weights[1][0] + self.bias[0], x * self.weights[0][1] + y * self.weights[1][1] + self.bias[1]] 