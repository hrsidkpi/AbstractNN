import numpy as np
import matplotlib.pyplot as plt
from polyhedron_estimator_analysis.halfspace import Halfspace
from shared.linear_layer import LinearLayer
from utils.math_util import *
from typing import List
from tqdm import tqdm

class PolyhedronEstimator(object):

    def __init__(self, halfspaces: List[Halfspace]) -> None:
        self.halfspaces: List[Halfspace] = halfspaces
        self.dimensions = len(self.halfspaces[0].weights)

    def apply_linear_transform(self, layer: LinearLayer):
        for hs in self.halfspaces:
            hs.apply_transformation(layer.weights, layer.bias)
        self.dimensions = len(layer.bias)

    def apply_relu(self):
        for i in range(self.dimensions):
            weights = [0 for _ in range(self.dimensions)]
            weights[i] = -1
            self.halfspaces.append(Halfspace(np.array(weights), 0))

    def get_bounds(self):
        raise Exception("Not implemented")

    def is_point_in_polyhedron(self, v):
        add = True
        for hs in self.halfspaces:
            if not hs.contains_vertex(v):
                add = False
        return add

    def draw2d(self):
        plt.figure()
        points = []
        for x in np.linspace(-3, 3, 50):
            for y in np.linspace(-3, 3, 50):
                v = np.array([x,y]).T
                add = True
                for hs in self.halfspaces:
                    if not hs.contains_vertex(v):
                        add = False
                if add:
                    points.append(v)
        points = np.array(points)
        plt.scatter(points[:,0], points[:,1])

    def draw3d(self):
        points = []
        for x in np.linspace(-3, 3, 50):
            for y in np.linspace(-3, 3, 50):
                for z in np.linspace(-3, 3, 50):
                    v = np.array([x,y, z]).T
                    if self.is_point_in_polyhedron(v):
                        points.append(v)
        print(np.array(points).shape)
        points = np.array(points)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.scatter(points[:,0], points[:,1], points[:,2])

    def get_hypervolume(self):
        points = get_points_in_hypercube(self.dimensions, [[-6, 6] for _ in range(self.dimensions)], 200)
        differential_cube_hypervolume = (12**self.dimensions) / (200**self.dimensions)
        hypervolume = 0
        for p in tqdm(points):
            if self.is_point_in_polyhedron(p):
                hypervolume += differential_cube_hypervolume
        return hypervolume

        





