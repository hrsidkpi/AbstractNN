import numpy as np
import matplotlib.pyplot as plt
from shared.linear_layer import LinearLayer
from utils.math_util import *

class ZonotopeEstimator(object):

    def __init__(self, generators, bias) -> None:
        self.generators = generators
        self.bias = bias

    def apply_linear_transform(self, layer: LinearLayer):
        self.bias = layer.weights @ self.bias + layer.bias
        new_gens = []
        for g in self.generators:
            new_gens.append(g)
        self.generators = new_gens

    def apply_relu(self):
        new_gens = []
        bias_sum = 0
        for j in range(len(self.generators)):
            alpha_j = None
            d_star = None
            for d in range(len(self.generators[0])):
                sum_i = 0
                for i in range(len(self.generators)):
                    sum_i += abs(self.generators[i][d])
                t_j_d = self.bias[0][d] + 2 * abs(self.generators[j][d]) - sum_i
                alpha_j_d = 1 - abs(t_j_d)/2*abs(self.generators[j][d]) if t_j_d > 0 else 1
                if alpha_j is None or alpha_j_d < alpha_j:
                    alpha_j = alpha_j_d
                    d_star = d
            o_j = self.generators[j][d_star] / abs(self.generators[j][d_star])
            s_j = -1
            g_j = alpha_j * self.generators[j]
            new_gens.append(g_j)
            bias_sum += s_j * (1-alpha_j) * o_j * g_j
        self.generators = new_gens
        self.bias = self.bias + bias_sum

    def get_bounds(self):
        bounds = []
        for v in range(len(self.bias)):
            lower_v = 0
            upper_v = 0
            for g in self.generators:
                upper_v += abs(g[v])
                lower_v -= abs(g[v])
            bounds.append([lower_v, upper_v])
        return bounds

    def draw(self):
        xs = []
        ys = []
        epsilons = get_points_in_hypercube(len(self.generators), [[-1, 1] for _ in range(len(self.generators))], 10)
        for eps in epsilons:
            p = self.bias
            for i in range(len(eps)):
                p = p + np.array(self.generators[i] * eps[i])
            xs.append(p[0])
            ys.append(p[1])
        plt.scatter(xs, ys)

                

    def get_area(self):
        return None





