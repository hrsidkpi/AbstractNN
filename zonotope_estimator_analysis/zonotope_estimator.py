from hashlib import new
import numpy as np
import matplotlib.pyplot as plt
from rsa import sign
from torch import argmin
from shared.linear_layer import LinearLayer
from utils.math_util import *
from scipy.optimize import linprog

class ZonotopeEstimator(object):

    def __init__(self, generators, bias) -> None:
        self.generators = generators.astype(float)
        self.bias = bias.astype(float)

    def apply_linear_transform(self, layer: LinearLayer):
        self.bias = layer.weights @ self.bias + layer.bias
        new_gens = []
        for g in self.generators:
            new_gens.append(g)
        self.generators = new_gens

    def change_eps_bounds(self, i: int, lower: float, upper: float):
        if(lower > upper):
            raise Exception("Invalid bounds for zonotope epsilon")
        if(lower == upper):
            self.bias += lower * self.generators[i]
            self.generators = np.delete(self.generators, i, axis=0)
        else:
            self.bias += (lower + upper) / 2 * self.generators[i]
            self.generators[i] =  self.generators[i] * ((upper - lower) / 2)




    def biggest_negative_point_lb(self, i, j):
        all_points = self.get_vertices_with_epsilons()
        closest = None
        closest_val = None
        for p in all_points:
            if p[0][i] >= 0:
                continue
            if p[1][j] >= 0:
                continue
            if closest_val == None:
                closest = p
                closest_val = p[0][i]
            else:
                if p[0][i] > closest_val:
                    closest = p
                    closest_val = p[0][i]
        return closest

    def biggest_negative_point_ub(self, i, j):
        all_points = self.get_vertices_with_epsilons()
        closest = None
        closest_val = None
        for p in all_points:
            if p[0][i] >= 0:
                continue
            if p[1][j] <= 0:
                continue
            if closest_val == None:
                closest = p
                closest_val = p[0][i]
            else:
                if p[0][i] > closest_val:
                    closest = p
                    closest_val = p[0][i]
        return closest

    def smallest_positive_point_lb(self, i, j):
        all_points = self.get_vertices_with_epsilons()
        closest = None
        closest_val = None
        for p in all_points:
            if p[0][i] <= 0:
                continue
            if p[1][j] >= 0:
                continue
            if closest_val == None:
                closest = p
                closest_val = p[0][i]
            else:
                if p[0][i] < closest_val:
                    closest = p
                    closest_val = p[0][i]
        return closest

    def smallest_positive_point_ub(self, i, j):
        all_points = self.get_vertices_with_epsilons()
        closest = None
        closest_val = None
        for p in all_points:
            if p[0][i] <= 0:
                continue
            if p[1][j] <= 0:
                continue
            if closest_val == None:
                closest = p
                closest_val = p[0][i]
            else:
                if p[0][i] < closest_val:
                    closest = p
                    closest_val = p[0][i]
        return closest


    def meet_gt_0(self, i):
        for j in range(len(self.generators)):

            verts = self.get_vertices_with_epsilons()
            can_change_ub = True
            can_change_lb = True
            for v in verts:
                if v[0][i] >= 0 and v[1][j] > 0:
                    can_change_ub = False
                if v[0][i] >= 0 and v[1][j] < 0:
                    can_change_lb = False
            if not (can_change_ub or can_change_lb):
                continue

            
            if can_change_lb:
                closest_res = self.biggest_negative_point_lb(i, j)
                if closest_res is None:
                    continue
                closest = closest_res[0]
                dist = closest[i]
                new_lb = -1 - dist / self.generators[j][i]
                if(new_lb <= 1):
                    self.change_eps_bounds(j, new_lb, 1)
                    return
            if can_change_ub:
                closest_res = self.biggest_negative_point_ub(i, j)
                if closest_res is None:
                    continue
                closest = closest_res[0]
                dist = closest[i]
                new_ub = 1 - dist / self.generators[j][i]
                if(new_ub >= -1):
                    self.change_eps_bounds(j, -1, new_ub)
                    return

    def meet_lt_0(self, i):
        for j in range(len(self.generators)):

            verts = self.get_vertices_with_epsilons()
            can_change_ub = True
            can_change_lb = True
            for v in verts:
                if v[0][i] <= 0 and v[1][j] > 0:
                    can_change_ub = False
                if v[0][i] <= 0 and v[1][j] < 0:
                    can_change_lb = False
            if not (can_change_ub or can_change_lb):
                continue

            
            if can_change_lb:
                closest_res = self.smallest_positive_point_lb(i, j)
                if closest_res is None:
                    continue
                closest = closest_res[0]
                dist = closest[i]
                new_lb = -1 - dist / self.generators[j][i]
                if(new_lb <= 1):
                    self.change_eps_bounds(j, new_lb, 1)
                    return
            if can_change_ub:
                closest_res = self.smallest_positive_point_ub(i, j)
                if closest_res is None:
                    continue
                closest = closest_res[0]
                dist = closest[i]
                new_ub = 1 - dist / self.generators[j][i]
                if(new_ub >= -1):
                    self.change_eps_bounds(j, -1, new_ub)
                    return


    def snap_to_0(self, dim: int):
        for g in self.generators:
            g[dim] = 0
        self.bias[dim] = 0

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



    def get_all_possible_eps_values(self, dim):
        if dim == len(self.generators)-1:
            return [[1], [-1]]
        prev_eps_values = self.get_all_possible_eps_values(dim+1)
        new_values = []
        for v in prev_eps_values:
            new_values.append([1] + v)
            new_values.append([-1] + v)
        return new_values

    def is_extreme_point(self, point):
        dir = point - self.bias
        
        c = [-1] + [0 for _ in range(len(self.generators))]
        
        A_ineq = [[0 for _ in range(len(self.generators)+1)]]
        b_ineq = [0]
        
        A_eq = []
        b_eq = []
        for i in range(len(self.bias)):
            row = [dir[i]]
            for g in self.generators:
                row.append(g[i])
            A_eq.append(row)
            b_eq.append(0)

        bounds = [(None, None)] + [(-1, 1) for _ in range(len(self.generators))]

        res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        val = res.x[0]

        MAX_ALLOWED_DIFF = 0.0001
        if(abs(val - 1) < MAX_ALLOWED_DIFF):
            return True
        return False
        

    def get_vertices(self):
        eps_values = self.get_all_possible_eps_values(0)
        verts = []
        for epsilons in eps_values:
            vert = np.copy(self.bias)
            for i in range(len(self.generators)):
                vert += epsilons[i] * self.generators[i]
            verts.append(vert)
        verts = [v for v in verts if self.is_extreme_point(v)]
        return verts

    def get_vertices_with_epsilons(self):
        eps_values = self.get_all_possible_eps_values(0)
        verts = []
        for epsilons in eps_values:
            vert = np.copy(self.bias)
            for i in range(len(self.generators)):
                vert += epsilons[i] * self.generators[i]
            verts.append([vert, epsilons])
        verts = [v for v in verts if self.is_extreme_point(v[0])]
        return verts

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

    def draw_3d(self):
        xs = []
        ys = []
        zs = []
        epsilons = get_points_in_hypercube(len(self.generators), [[-1, 1] for _ in range(len(self.generators))], 10)
        for eps in epsilons:
            p = self.bias
            for i in range(len(eps)):
                p = p + np.array(self.generators[i] * eps[i])
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        xx, yy = np.meshgrid(np.linspace(min(xs) - 1, max(xs) + 1, 10), np.linspace(min(ys) - 1, max(ys) + 1, 10))
        zz = 0 * xx
        ax.plot_surface(xx, yy, zz, alpha=0.2)

        ax.scatter(xs, ys, zs)

    def draw_lines(self, c='red'):
        vertices = sort_clockwise(self.get_vertices())
        poly_closed = list(vertices) + [vertices[0]]
        draw_xs, draw_ys = zip(*poly_closed)
        plt.plot(draw_xs, draw_ys, c, linewidth=4)





    def get_area(self):
        return None





