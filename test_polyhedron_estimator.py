import numpy as np
import matplotlib.pyplot as plt
from polyhedron_estimator_analysis.polyhedron_estimator import PolyhedronEstimator
from polyhedron_estimator_analysis.halfspace import Halfspace
from shared.linear_layer import LinearLayer


hfs = [
    Halfspace(np.array([1, 0, 0]), -1),
    Halfspace(np.array([-1, 0, 0]), 0),
    Halfspace(np.array([0, 1, 0]), -2),
    Halfspace(np.array([0, -1, 0]), 0),
    Halfspace(np.array([0, 0, 1]), -2),
    Halfspace(np.array([0, 0, -1]), 0),
]


theta = np.pi / 4
layer = LinearLayer("layer 1a", np.array([[1, 0, 0], [0,np.cos(theta),-np.sin(theta)], [0,np.sin(theta),np.cos(theta)]]), np.array([0, 0, 1]))


p = PolyhedronEstimator(hfs)
p.draw3d()
plt.title("area = " + str(p.get_hypervolume()))

# 3.84

p.apply_linear_transform(layer)
p.draw3d()
plt.title("area = " + str(p.get_hypervolume()))

p.apply_relu()
p.draw3d()
plt.title("area = " + str(p.get_hypervolume()))

plt.show()