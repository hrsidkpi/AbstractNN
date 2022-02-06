from zonotope_estimator_analysis.zonotope_estimator import ZonotopeEstimator
import numpy as np
from scipy.optimize import linprog



def zonotope_bound_for_points(points, vectors):
    dim = len(points[0])
    num_points = len(points)
    num_vecs = len(vectors)

    c = [1 for _ in range(num_vecs)] + [0 for _ in range(num_points * num_vecs)] + [0 for _ in range(dim)]

    A_eq = []
    b_eq = []
    for j, p in enumerate(points): 
        for d in range(dim):
            row = [0 for _ in range(len(c))]
            for i, g in enumerate(vectors): 
                row[num_vecs + j * num_vecs + i] = g[d]
            row[num_vecs + num_vecs * num_points + d] = 1
            A_eq.append(row)
            b_eq.append(p[d])

    A_ineq = []
    b_ineq = []
    for i, v in enumerate(vectors):
        for j, p in enumerate(points):
            v_lb = [0 for _ in range(len(c))]
            v_lb[i] = -1
            v_lb[num_vecs + j * num_vecs + i] = -1
            A_ineq.append(v_lb)
            b_ineq.append(0)
            v_ub = [0 for _ in range(len(c))]
            v_ub[i] = -1
            v_ub[num_vecs + j * num_vecs + i] = 1
            A_ineq.append(v_ub)
            b_ineq.append(0)

    bounds = [(0, None) for _ in range(num_vecs)] + [(None, None) for _ in range(num_vecs * num_points)] + [(None, None) for _ in range(dim)]
    res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    
    generators = []
    for i, g in enumerate(vectors):
        generators.append(res.x[i] * np.array(g))
    center = []
    for d in range(dim):
        center.append(res.x[num_vecs + num_vecs * num_points + d])
    return ZonotopeEstimator(np.array(generators), np.array(center))

    



def join_zonotopes(z1: ZonotopeEstimator, z2: ZonotopeEstimator):
    points = list(z1.get_vertices()) + list(z2.get_vertices())
    centers_vec = z2.bias - z1.bias

    vectors = [centers_vec] + list(z1.generators)

    extreme_points = []
    center_guess = (z1.bias + z2.bias) / 2
    for v in vectors:
        np_v = np.array(v)
        max_p = points[0]
        min_p = points[0]
        max_p_val = np_v @ (np.array(points[0]) - center_guess)
        min_p_val = np_v @ (np.array(points[0]) - center_guess)
        for p in points:
            p_val = np_v @ (np.array(p) - center_guess)
            if p_val < min_p_val:
                min_p_val = p_val
                min_p = p
            if p_val > max_p_val:
                max_p_val = p_val
                max_p = p
        extreme_points.append(max_p)
        extreme_points.append(min_p)

    ############ TODO I'm currently using points and not extreme points because extreme points aren't working.
    # The idea to use extreme points is in the following article:
    # https://graphics.stanford.edu/~anguyen/papers/zonotope.pdf
    # however, they do not specifiy what point to use as the center to compare the points in relation to.
    # I tried using a guess of the center (middle of the two zonotopes) and (0,0) and none of them work.
    return zonotope_bound_for_points(points, vectors)
    

def apply_relu_on_zonotope(z: ZonotopeEstimator, dim):
    z_pos = copy_zonotope(z)
    z_neg = copy_zonotope(z)

    z_pos.meet_gt_0(dim)
    z_neg.meet_lt_0(dim)
    z_neg.snap_to_0(dim)

    return join_zonotopes(z_pos, z_neg)

def copy_zonotope(z: ZonotopeEstimator) -> ZonotopeEstimator:
    res = ZonotopeEstimator(np.copy(z.generators), np.copy(z.bias))
    return res