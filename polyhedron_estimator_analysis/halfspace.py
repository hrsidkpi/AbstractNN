import numpy as np

class Halfspace(object):

    def __init__(self, weights: np.ndarray, offset: float) -> None:
        self.weights = weights
        self.offset = offset

    def contains_vertex(self, v):
        return self.weights.T @ v + self.offset < 0

    def seperator_contains_vertex(self, v):
        return self.weights.T @ v + self.offset == 0
    
    def apply_transformation(self, transformation_linear, transformation_offset):
        self.weights = self.weights.T @ np.linalg.inv(transformation_linear)
        self.offset = self.offset - self.weights @ np.linalg.inv(transformation_linear) @ transformation_offset

    def __repr__(self) -> str:
        return str(self.weights) + " + " + str(self.offset)

