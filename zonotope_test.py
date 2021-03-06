import numpy as np
import matplotlib.pyplot as plt
from zonotope_estimator_analysis.zonotope_estimator import ZonotopeEstimator
from zonotope_estimator_analysis.zonotope_operations import *






vectors1 = np.array([
    [1, 2],
    [-4, -1],
    [1, -1],
    [1, 1],
    #[-1,1],
])
bias1 = np.array([2, 0])
z1 = ZonotopeEstimator(vectors1, bias1)

z1.draw_lines(c='blue')
z2 = apply_relu_on_zonotope(z1, 0)
z2.draw_lines(c='orange')

plt.show()
"""

vectors2 = np.array([
    [-2,1],
    [1,2]
])
bias2 = np.array([1, 4])
z2 = ZonotopeEstimator(vectors2, bias2)

z1.draw_lines(c='blue')
z2.draw_lines(c='blue')
z_join = join_zonotopes(z1, z2)
z_join.draw_lines(c='red')


z3 = join_zonotopes_1d(z1, z2, 0)
z3 = join_zonotopes_1d(z3, z2, 1)


z3.draw()
z2.draw()
z1.draw()

plt.show()

"""