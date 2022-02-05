import numpy as np
import matplotlib.pyplot as plt
from shared.linear_layer import LinearLayer
from utils.math_util import *


class BoxEstimator(object):

    def __init__(self, bounds) -> None:
        self.vertices = np.array([v for v in vertices_from_bounds(bounds)])
        self.inflation_ratio = 1

    def _boxify(self):
        old_area = poly_area(self.vertices)
        self.vertices = np.array(vertices_from_bounds(bounds_from_vertices(self.vertices)))
        new_area = poly_area(self.vertices)

        self.inflation_ratio *= new_area / old_area

    def apply_linear_transform(self, layer: LinearLayer):
        temp_inflation_ratio = self.inflation_ratio

        self.vertices = self.vertices @ layer.weights
        new_verts = []
        for v in self.vertices:
            new_verts.append((v.reshape(2,1)+layer.bias).reshape(2,))
        self.vertices = np.array(new_verts)

        self._boxify()

        print("Calculated completeness loss: ", self.inflation_ratio / temp_inflation_ratio)

    def apply_relu(self):
        temp_inflation_ratio = self.inflation_ratio

        new_verts = []
        for v in self.vertices:
            new_verts.append([relu(x) for x in v])
        self.vertices = np.array(new_verts)
        self._boxify()

        print("Calculated completeness loss: ", self.inflation_ratio / temp_inflation_ratio)


    def get_bounds(self):
        return bounds_from_vertices(self.vertices)

    def draw(self):
        bounds = self.get_bounds()
        print(bounds)
        min_x = bounds[0][0]
        min_y = bounds[1][0]
        max_x = bounds[0][1]
        max_y = bounds[1][1]
        plt.plot([min_x, max_x], [min_y, min_y], c='red', label='output domain')
        plt.plot([min_x, max_x], [max_y, max_y], c='red')
        plt.plot([min_x, min_x], [min_y, max_y], c='red')
        plt.plot([max_x, max_x], [min_y, max_y], c='red')

    def get_area(self):
        bounds = self.get_bounds()
        min_x = bounds[0][0]
        min_y = bounds[1][0]
        max_x = bounds[0][1]
        max_y = bounds[1][1] 
        return (max_x - min_x) * (max_y - min_y)





