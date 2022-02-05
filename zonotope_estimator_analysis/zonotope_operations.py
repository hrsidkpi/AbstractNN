from zonotope_estimator_analysis.zonotope_estimator import ZonotopeEstimator
import numpy as np


def join_zonotopes_1d(z1: ZonotopeEstimator, z2: ZonotopeEstimator, d: int):
    bounds_1 = z1.get_bounds()
    bounds_2 = z2.get_bounds()
    bias = (z1.bias + z2.bias) / 2
    generators = [[] for _ in range(len(z1.generators)+1)]
    for v in range(len(z1.bias)):
        if v == d:
            sum_alphas = 0
            for i in range(len(generators)-1):
                min_a = min(z1.generators[i][v], z2.generators[i][v])
                max_a = max(z1.generators[i][v], z2.generators[i][v])

                alpha = None
                if(min_a * max_a < 0):
                    alpha = 0
                elif(min_a < 0):
                    alpha = max_a
                else:
                    alpha = min_a

                generators[i].append(alpha)
                sum_alphas += abs(alpha)
            print(bounds_1[v][1] + bounds_2[v][1])
            print(bias[v])
            print(sum_alphas)
            generators[len(generators)-1].append(bounds_1[v][1] + bounds_2[v][1] - bias[v] - sum_alphas)
        else:
            for i in range(len(generators)-1):
                generators[i].append(z1.generators[i][v])
            generators[len(generators)-1].append(0)

    return ZonotopeEstimator(np.array(generators), np.array(bias))


def join_zonotopes(z1: ZonotopeEstimator, z2: ZonotopeEstimator):
    bounds_1 = z1.get_bounds()
    bounds_2 = z2.get_bounds()
    bias = (z1.bias + z2.bias) / 2
    generators = [[] for _ in range(len(z1.generators)+1)]
    for v in range(len(z1.bias)):
        sum_alphas = 0
        for i in range(len(generators)-1):
            min_a = min(z1.generators[i][v], z2.generators[i][v])
            max_a = max(z1.generators[i][v], z2.generators[i][v])

            alpha = None
            if(min_a * max_a < 0):
                alpha = 0
            elif(min_a < 0):
                alpha = max_a
            else:
                alpha = min_a

            generators[i].append(alpha)
            sum_alphas += abs(alpha)
        print(bounds_1[v][1] + bounds_2[v][1])
        print(bias[v])
        print(sum_alphas)
        generators[len(generators)-1].append(bounds_1[v][1] + bounds_2[v][1] - bias[v] - sum_alphas)

    return ZonotopeEstimator(np.array(generators), np.array(bias))