import numpy as np
import matplotlib.pyplot as plt
from shared.linear_layer import LinearLayer
from utils.math_util import *

class PolygonEstimator(object):

    def __init__(self, vertices) -> None:
        self.vertices = vertices
        self.inflation_ratio = 1

    def apply_linear_transform(self, layer: LinearLayer):
        self.vertices = self.vertices @ layer.weights
        new_verts = []
        for v in self.vertices:
            new_verts.append((v.reshape(2,1)+layer.bias).reshape(2,))
        self.vertices = np.array(new_verts)

    def apply_relu(self):
        print("applying ReLu...")
        new_verts = []
        
        last_positive = None
        points = sort_clockwise(self.vertices)
        for v in points:
            if v[0] >= 0 and v[1] >= 0:
                last_positive = v
        if last_positive is None:
            self.vertices = [[0,0]]
            return
        print("first positive vertex: ", last_positive)

        points = sort_clockwise(self.vertices, last_positive)
        last_positive = None
        first_negative = None
        last_negative = None
        is_in_negative = False
        print("\n\n\n\n")
        print("initial vertices: " , self.vertices)
        print("initial vertices sorted: \n" , points)
        print("\n")
        for v in points:

            print("\nPoint", v, ":")
            print("is_in_negative: ", is_in_negative)
            print("is_positive: ", v[0]>=0 and v[1] >= 0)
            print("current verts: ", new_verts)

            if is_in_negative:
                if v[0] >= 0 and v[1] >= 0:
                    if last_negative is None:
                        last_negative = first_negative
                    new_verts.append(midpoint_on_axis(last_positive, first_negative))
                    new_verts.append(midpoint_on_axis(last_negative, v))
                    new_verts.append(v)
                    last_positive = v
                    is_in_negative = False
                else:
                    last_negative = v
            else:
                if v[0] >= 0 and v[1] >= 0:
                    new_verts.append(v)
                    last_positive = v
                else:
                    is_in_negative = True
                    first_negative = v
        
        if is_in_negative:
            if last_negative is None:
                last_negative = first_negative
            print(last_positive, first_negative, last_negative, points[0])
            new_verts.append(midpoint_on_axis(last_positive, first_negative))
            new_verts.append(midpoint_on_axis(last_negative, points[0]))

        print("new verts: ", new_verts)

        self.vertices = np.array(new_verts)
        print("\n\n")

    def get_bounds(self):
        return bounds_from_vertices(self.vertices)

    def draw(self):
        print(self.vertices)
        poly_closed = list(self.vertices) + [self.vertices[0]]
        draw_xs, draw_ys = zip(*poly_closed)
        plt.plot(draw_xs, draw_ys, c='red', linewidth=4)

    def get_area(self):
        return poly_area(self.vertices)





