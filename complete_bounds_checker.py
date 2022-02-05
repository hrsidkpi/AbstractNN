import matplotlib.pyplot as plt
import numpy as np

from typing import List

from numpy.core.fromnumeric import searchsorted, sort

from box_estimator_analysis.box_estimator import BoxEstimator
from box_estimator_analysis.lower_box_estimator import LowerBoxEstimator
from polygon_estimator_analysis.polygon_estimator import PolygonEstimator
from polyhedron_estimator_analysis.polyhedron_estimator import PolyhedronEstimator

from shared.layer import Layer
from shared.linear_layer import LinearLayer
from shared.relu_layer import ReluLayer

from utils.math_util import *

from random_network.random_network_generator import generate_random_network
from random_polygon_generator import generate_random_polygon
from zonotope_estimator_analysis.zonotope_estimator import ZonotopeEstimator

layers = generate_random_network(2, -2, 5, -1, 6)

############## box estimator ###############
# start_x_lower = np.random.randint(low=3, high=13)
# start_y_lower = np.random.randint(low=3, high=13)
# START_X_BOUND = [start_x_lower, start_x_lower + np.random.randint(3, 8)]
# START_Y_BOUND = [start_y_lower, start_y_lower + np.random.randint(3, 8)]
# START_VERTICES_UNSORTED = vertices_from_bounds([START_X_BOUND, START_Y_BOUND])

############# poly estimator ##########
# START_VERTICES_UNSORTED = generate_random_polygon(10, -3, 13, -3, 13)
# START_X_BOUND, START_Y_BOUND = bounds_from_vertices(START_VERTICES_UNSORTED)
# 
# START_VERTICES = sort_clockwise(START_VERTICES_UNSORTED)
# 
# real_points = []
# for x in np.linspace(START_X_BOUND[0], START_X_BOUND[1], 100):
#         for y in np.linspace(START_Y_BOUND[0], START_Y_BOUND[1], 100):
#             real_points.append([x, y])
# real_points = np.array(real_points)


########### zonotope estimator ########
generators = np.random.randint(low=-3, high=3, size=(3, 2))
bias = np.random.randint(low=-3, high=3, size=(1,2))

START_X_BOUND = [0,0]
START_Y_BOUND = [0,0]


lower_ai = LowerBoxEstimator([START_X_BOUND, START_Y_BOUND])

upper_ai = ZonotopeEstimator(generators, bias)

# plt.figure()
# plt.title("initial bounds")
upper_ai.draw()
# lower_ai.draw()
plt.figure()
upper_ai.apply_relu()
upper_ai.draw()
plt.show()


"""


for layer in layers:

    print("==================")
    print(layer.name)
    print("==================")

    plt.figure()

    # Draw complete domain on previous output
    previous_bounds = upper_ai.get_bounds()
    xs_bound = np.linspace(previous_bounds[0][0], previous_bounds[0][1], 100)
    ys_bound = np.linspace(previous_bounds[1][0], previous_bounds[1][1], 100)
    res_xs = []
    res_ys = []
    for x in xs_bound:
        for y in ys_bound:
            res_x, res_y = layer.activate(x ,y)
            res_xs.append(res_x)
            res_ys.append(res_y)
    
    plt.scatter(res_xs, res_ys, s=1, label='complete domain on previous output')


    # Draw output domain
    if isinstance(layer, ReluLayer):
        upper_ai.apply_relu()
        lower_ai.apply_relu()
    if isinstance(layer, LinearLayer):
        upper_ai.apply_linear_transform(layer)
        lower_ai.apply_linear_transform(layer)
    upper_ai.draw()
    lower_ai.draw()

    # Draw real complete domain
    new_points = []
    for point in real_points:
        new_points.append(layer.activate(point[0], point[1]))
    new_points = np.array(new_points)
    xs = new_points[:,0]
    ys = new_points[:,1]
    real_points = new_points
    

    completeness_loss = upper_ai.get_area() / lower_ai.get_area()


    plt.scatter(xs, ys, s=1, label='real complete domain')

    plt.title(layer.name + ", completeness loss: " + str(completeness_loss))
    plt.legend()

plt.show()
input()

"""