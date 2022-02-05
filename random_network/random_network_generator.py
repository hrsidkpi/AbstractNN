from shared.layer import Layer
from shared.linear_layer import LinearLayer
from shared.relu_layer import ReluLayer
import numpy as np


def generate_random_network(num_of_layers, min_weight, max_weight, min_bias, max_bias):
    layers = []
    for i in range(num_of_layers):
        weights = np.random.randint(low=min_weight, high=max_weight, size=(2,2))
        bias = np.random.randint(low=min_bias, high=max_bias, size=(2,1))
        layers.append(LinearLayer("layer"+str(i)+"a", weights, bias))
        layers.append(ReluLayer("layer"+str(i)+"b"))
    return layers