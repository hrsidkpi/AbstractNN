import numpy as np


def generate_random_polygon(num_of_vertices, min_x, max_x, min_y, max_y):
    verts = []
    for _ in range(num_of_vertices):
        x = np.random.randint(low=min_x, high=max_x)
        y = np.random.randint(low=min_y, high=max_y)
        verts.append([x,y])
    return np.array(verts)