import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
from utils.math_util import *


poly1=[
    [1,5],
    [-1,4],
    [1,3],
    [-1,2],
    [1,1],
    [-1,0],
    [1,-1],
    [-1,-2],
    [1,-3],
    [-1,-4],
    [1,-5],
    [2, 0],
]


def transform(point):
    x = point[0]
    y = point[1]
    return [x, x - 2 * y]


# print(point_in_polygon([2.4, 0.01], poly1))
# input()

xs = list(np.linspace(-5, 5, 60))
ys = list(np.linspace(-5, 5, 60))

points = []
for x in xs:
    for y in ys:
        p = [x, y]
        if point_in_polygon(p, poly1):
            points.append(p)

points = np.array(points)

plt.scatter(points[:,0], points[:,1], c='r')

poly1_closed = poly1 + [poly1[0]]
draw_xs, draw_ys = zip(*poly1_closed)
plt.plot(draw_xs, draw_ys, c='m', linewidth=4)


new_poly = []
for p in poly1:
    new_poly.append(transform(p))

new_points = []
for p in points:
    new_points.append(transform(p))
new_points = np.array(new_points)


plt.scatter(new_points[:,0], new_points[:,1], c='b')

new_poly_closed = new_poly + [new_poly[0]]
draw_xs, draw_ys = zip(*new_poly_closed)
plt.plot(draw_xs, draw_ys, c='c', linewidth=4)

plt.show()
