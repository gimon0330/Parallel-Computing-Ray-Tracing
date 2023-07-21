from math import acos, tan
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def get_box(i, objects, camera, screen, width, height):
    ratio = float(width) / height

    x, y, z = objects[i]['center']
    a = np.linalg.norm(objects[i]['center'] - camera)
    ay = np.linalg.norm(np.array([0, y, z]) - camera)
    ax = np.linalg.norm(np.array([x, 0, z]) - camera)
    r = objects[i]['radius']

    def rescale(inp, mode):
        if mode == 'x':
            ret = round(inp * (width / 2) + (width / 2))
            ret = max(0, ret)
            ret = min(ret, width - 1)
            return ret
        elif mode == 'y':
            ret = round(inp * (height / 2 * ratio) + (height / 2))
            ret = max(0, ret)
            ret = min(ret, height - 1)
            return ret
        else:
            raise

    def method1():
        # x1, x2
        b = np.sqrt(abs(ax * ax - r * r))
        st = (abs(z) + 1) / ax
        if x == 0:
            x1 = -r / b
            x2 = r / b
        else:
            tt = st / np.sqrt(1 - st * st)
            x1 = (b - r * tt) / (b * tt + r)
            x2 = x1 + 2 * r * b / (ax * ax * st * st - r * r)
        sign = 1 if x >= 0 else -1
        x1 *= sign
        x2 *= sign
        x1, x2 = rescale(x1, 'x'), rescale(x2, 'x')
        x1, x2 = min(x1, x2), max(x1, x2)

        # y1, y2
        b = np.sqrt(abs(ay * ay - r * r))
        st = (abs(z) + 1) / ay
        if y == 0:
            y1 = -r / b
            y2 = r / b
        else:
            tt = st / np.sqrt(1 - st * st)
            y1 = (b - r * tt) / (b * tt + r)
            y2 = y1 + 2 * r * b / (ay * ay * st * st - r * r)
        sign = -1 if y >= 0 else 1
        y1 *= sign
        y2 *= sign
        y1, y2 = rescale(y1, 'y'), rescale(y2, 'y')
        y1, y2 = min(y1, y2), max(y1, y2)

        return x1, x2, y1, y2

    def method2():
        st = (abs(z) + 1) / a
        ct = np.sqrt(1 - st * st)
        tt = st / ct
        cphi = r / a
        sphi = np.sqrt(1 - cphi * cphi)
        tphi = sphi / cphi
        sxy = np.sqrt(x * x + y * y)

        l = (tphi - tt) / (1 + tphi * tt)
        pts = []
        rects = [(screen[0], screen[1]), (screen[0], screen[3]), (screen[2], screen[1]),
                 (screen[2], screen[3])]
        for (_x, _y) in rects:
            if x * _x + y * _y >= l * sxy:
                pts.append((_x, _y))
        if y != 0:
            for _x in [screen[0], screen[2]]:
                cand_y = (l * sxy - x * _x) / y
                if abs(cand_y) <= 1 / ratio:
                    pts.append((_x, cand_y))
        if x != 0:
            for _y in [screen[1], screen[3]]:
                cand_x = (l * sxy - y * _y) / x
                if abs(cand_x) <= 1:
                    pts.append((cand_x, _y))
        pts = np.array(pts)

        pts_x, pts_y = pts[:, 0], -pts[:, 1]
        return rescale(pts_x.min(), 'x'), rescale(pts_x.max(), 'x'), \
            rescale(pts_y.min(), 'y'), rescale(pts_x.max(), 'y')

    if ax >= r and ay >= r:
        return method1()
    else:
        return method2()


def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    oc = ray_origin - center
    a = np.dot(ray_direction, ray_direction)
    b = 2 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - radius ** 2
    delta = b ** 2 - 4 * a * c
   
    if delta > 0:
        sqrt_delta = np.sqrt(delta)
        t1 = (-b + sqrt_delta) / (2 * a)
        t2 = (-b - sqrt_delta) / (2 * a)
        return min(t1, t2) if t1 > 0 and t2 > 0 else None
   
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


def ray_tracing(x, y):
    # screen is on origin
    pixel = np.array([x, y, 0])
    origin = camera
    direction = normalize(pixel - origin)
    color = np.zeros((3))
    reflection = 1
    for k in range(max_depth):
        # check for intersections
        nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
        if nearest_object is None:
            break
        intersection = origin + min_distance * direction
        normal_to_surface = normalize(intersection - nearest_object['center'])
        shifted_point = intersection + 1e-5 * normal_to_surface
        intersection_to_light = normalize(light['position'] - shifted_point)
        _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
        intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
        is_shadowed = min_distance < intersection_to_light_distance
        if is_shadowed: break
        illumination = np.zeros((3))
        # ambiant
        illumination += nearest_object['ambient'] * light['ambient']
        # diffuse
        illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)
        # specular
        intersection_to_camera = normalize(camera - intersection)
        H = normalize(intersection_to_light + intersection_to_camera)
        illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)
        # reflection
        color += reflection * illumination
        reflection *= nearest_object['reflection']
        origin = shifted_point
        direction = reflected(direction, normal_to_surface)
    return color

def list_chuck(arr, n):
    length = len(arr)
    chunk_size = length // n
    mod = length % n

    chunks = []
    st = 0
    for i in range(n):
        en = st + chunk_size + (i < mod)
        chunks.append(arr[st:en])
        st = en

    return chunks

   

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start_time = MPI.Wtime()
#########################################################################################################
width = 300
height = 200

max_depth = 3
camera = np.array([0, 0, 1])
light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }
objects = [
    { 'center': np.array([-0.2, 0, -1]), 'radius': 0.2, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 1, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.1 },
    { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0.5, 0, -1]), 'radius': 0.5, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
]
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

Y = np.linspace(screen[1], screen[3], height)
X = np.linspace(screen[0], screen[2], width)


d = set()
for i in range(len(objects)):
    x1, x2, y1, y2 = get_box(i, objects, camera, screen, width, height)
    for j in range(x1, x2+1):
        for k in range(y1, y2+1):
            d.add((k, j))
d = list(d)


sctd = list_chuck(d, size)
sctd = set(sctd[rank])


dd = [(t[0], t[1], *np.clip(ray_tracing(X[t[1]], Y[t[0]]), 0, 1)) for t in sctd]
dd = np.array(dd)
   

gd = np.array(comm.gather(dd, root=0), dtype=object)


if rank == 0:
    image = np.zeros((height, width, 3))

    for dt in gd:
        image[tuple(dt[:, 0:2].T.astype('int'))] = np.stack(dt[:, 2:])
    plt.imsave('image.png', image)
   
#########################################################################################################
end_time = MPI.Wtime()

if rank==0:
    print("time: " + str(end_time-start_time))
