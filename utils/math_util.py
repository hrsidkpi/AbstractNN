USE_SHAPELY = False

if(USE_SHAPELY):
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
import numpy as np

def point_crosses_segment(point, seg_p1, seg_p2):
    x1 = seg_p1[0] - point[0]
    y1 = seg_p1[1] - point[1]
    x2 = seg_p2[0] - point[0]
    y2 = seg_p2[1] - point[1]

    if y1 < 0 and y2 < 0:
        return False
    if y1 > 0 and y2 > 0:
        return False

    x_per_y = (x2-x1) / (y2-y1)
    x_hit = x1 - x_per_y * y1

    return x_hit > 0
    

if(USE_SHAPELY):
    def point_in_polygon(point, polygon):
        poly = Polygon([(x,y) for [x,y] in polygon])
        p = Point(point[0], point[1])
        return poly.contains(p)


def vertices_from_bounds(bounds):
    if len(bounds) == 0:
        return []
    curr_bound = bounds[0]
    next_vertices = vertices_from_bounds(bounds[1:])
    if len(next_vertices) != 0:
        vertices_lower = [[curr_bound[0]] + v for v in next_vertices]
        vertices_upper = [[curr_bound[1]] + v for v in next_vertices]
    else:
        vertices_lower = [[curr_bound[0]]]
        vertices_upper = [[curr_bound[1]]]
    return vertices_lower + vertices_upper


def bounds_from_vertices(vertices):
    bounds = []
    for var in range(len(vertices[0])):
        curr_min = vertices[0][var]
        curr_max = vertices[0][var]
        for v in vertices:
            if v[var] < curr_min:
                curr_min = v[var]
            if v[var] > curr_max:
                curr_max = v[var]
        bounds.append([curr_min, curr_max])
    return bounds

def midpoint_on_axis(p1, p2):
    if p1[0] == p2[0]: # vertical line
        if (p1[0] < 0 and p2[0] < 0) or (p1[0] > 0 and p2[0] > 0):
            return None
        return [p1[0], 0]
    
    h = p2[1] - p1[1]
    w = p2[0] - p1[0]
    d = h / w
    y_offset = p1[1] - d * p1[0]
    x_offset = -y_offset / d
    if y_offset >= 0 and ((p1[0] <= 0 and p2[0] >= 0) or (p1[0] >= 0 and p2[0] <= 0)):
        return [0, y_offset]
    if x_offset >= 0 and ((p1[1] <= 0 and p2[1] >= 0) or (p1[1] >= 0 and p2[1] <= 0)):
        return [x_offset, 0]
    return None



def angle_with_start(coord, start):
    vec = np.array(coord) - np.array(start)
    return np.angle(np.complex(vec[0], vec[1]))

def points_equal(p1, p2):
    if len(p1) != len(p2):
        return False
    for i in range(len(p1)):
        if p1[i] != p2[i]:
            return False
    return True

def find_center(points):
    x = 0
    y = 0
    for p in points:
        x += p[0]
        y += p[1]
    center = [x / len(points), y / len(points)]
    return center


def roll_to_start(array: np.ndarray, start):
    found_i = -1
    for i in range(len(array)):
        if points_equal(array[i], start):
            found_i = i
            break
    if found_i == -1:
        raise Exception("Roll to start given point not in the array")
    print("roll to start done with i=", found_i)
    return np.roll(array, found_i-1, axis=0)

def sort_clockwise(points, starting_point = None):
    points = np.array(points)
    center = find_center(points)
    res = np.array(sorted(points, key=lambda coord: angle_with_start(coord, center), reverse=True))
    if starting_point is None:
        return res
    else:
        return roll_to_start(res, starting_point)

def poly_area(vertices):
    verts_sorted = sort_clockwise(vertices)
    x = verts_sorted[:,0]
    y = verts_sorted[:,1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def relu(x):
    if x < 0:
        return 0
    return x


def get_points_in_hypercube(dimensions, bounds, points_per_dimention):
    if dimensions == 1:
        return list([[v] for v in np.linspace(bounds[0][0], bounds[0][1], points_per_dimention)])
    bound_for_this_dimention = bounds[0]
    bounds_next = bounds[1:]
    next_points = get_points_in_hypercube(dimensions-1, bounds_next, points_per_dimention)
    points = []
    for v in list(np.linspace(bound_for_this_dimention[0], bound_for_this_dimention[1], points_per_dimention)):
        for p in next_points:
            points.append([v] + p)
    return points