import numpy as np
import matplotlib.pyplot as plt
from shared.linear_layer import LinearLayer
from utils.math_util import *


def boxify(vertices):    

    bounds = bounds_from_vertices(vertices)
    min_x = bounds[0][0]
    max_x = bounds[0][1]
    min_y = bounds[1][0]
    max_y = bounds[1][1]

    if min_x < 0 and max_x >= 0:
        min_x = 0
    if min_y < 0 and max_y >= 0:
        min_y = 0

    grid = []

    i = 0
    polygon = sort_clockwise(vertices)
    for x in np.linspace(min_x, max_x, 100):
        grid.append([])
        j = 0
        for y in np.linspace(min_y, max_y, 100):
            if point_in_polygon([x, y], polygon):
                corner_x = x
                corner_y = y
                if j != 0 and grid[i][j-1][3] != None:
                    corner_y = grid[i][j-1][3]
                if i != 0 and grid[i-1][j][2] != None:
                    corner_x = grid[i-1][j][2]
                grid[i].append([x, y, corner_x, corner_y])
            else:
                grid[i].append([x, y, None, None])
            j += 1
        i += 1

    max_area = 0
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j][2] == None:
                continue
            area = (grid[i][j][2] - grid[i][j][0]) * (grid[i][j][3] - grid[i][j][1])
            if area >= max_area:
                max_area = area
                min_x = grid[i][j][2]
                min_y = grid[i][j][3]
                max_x = grid[i][j][0]
                max_y = grid[i][j][1]            

    new_bounds = [[min_x, max_x], [min_y, max_y]]
    return np.array(vertices_from_bounds(new_bounds))

class LowerBoxEstimator(object):

    def __init__(self, bounds) -> None:
        self.vertices = np.array([v for v in vertices_from_bounds(bounds)])

    def _boxify(self):
        self.vertices = boxify(self.vertices)

    def apply_linear_transform(self, layer: LinearLayer):
        self.vertices = self.vertices @ layer.weights + layer.bias
        self._boxify()

    def apply_relu(self):
        new_verts = []
        for v in self.vertices:
            new_verts.append([relu(x) for x in v])
        self.vertices = np.array(new_verts)
        self._boxify()


    def get_bounds(self):
        return bounds_from_vertices(self.vertices)

    def draw(self):
        bounds = self.get_bounds()
        min_x = bounds[0][0]
        min_y = bounds[1][0]
        max_x = bounds[0][1]
        max_y = bounds[1][1]
        plt.plot([min_x, max_x], [min_y, min_y], c='yellow', label='output domain')
        plt.plot([min_x, max_x], [max_y, max_y], c='yellow')
        plt.plot([min_x, min_x], [min_y, max_y], c='yellow')
        plt.plot([max_x, max_x], [min_y, max_y], c='yellow')

    def get_area(self):
        bounds = self.get_bounds()
        min_x = bounds[0][0]
        min_y = bounds[1][0]
        max_x = bounds[0][1]
        max_y = bounds[1][1] 
        return (max_x - min_x) * (max_y - min_y)





