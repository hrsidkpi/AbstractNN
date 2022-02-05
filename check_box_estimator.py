from box_estimator_analysis.box_estimator import BoxEstimator
import numpy as np

layer_1 = np.array([
    [1, 3],
    [-3, 1]
])

layer_2 = np.array([
    [1, -1],
    [1, 1]
])



box = BoxEstimator([[0, 3], [0, 2]])
print(box.inflation_ratio)
box.apply_linear_transform(layer_1)
print(box.inflation_ratio)
box.apply_relu()
print(box.inflation_ratio)
box.apply_linear_transform(layer_2)
print(box.inflation_ratio)
box.apply_relu()
print(box.inflation_ratio)
print(box.get_bounds())