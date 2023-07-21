from ray_moon2 import make_image
import numpy as np
from random import random
from math import pi, sin, cos
import sys

objects = [
    {'center': np.array([0, 0, -8]), 'radius': 1.2, 'ambient': np.array([0.1, 0, 0]),
     'diffuse': np.array([1, 0.7, 0.1]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.5},
     
    {'center': np.array([0, 0, -5.5]), 'radius': 0.2, 'ambient': np.array([0.1, 0, 0]),
     'diffuse': np.array([0.7, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.3},
     
    {'center': np.array([0, 0, -3.5]), 'radius': 0.4, 'ambient': np.array([0.1, 0, 0]),
     'diffuse': np.array([0.7, 1, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.3},
     
    {'center': np.array([0, 0, -2.5]), 'radius': 0.6, 'ambient': np.array([0.1, 0, 0]),
     'diffuse': np.array([0.1, 1, 0.3]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.3},
     
    {'center': np.array([0, 0, -1]), 'radius': 0.35, 'ambient': np.array([0.1, 0, 0]),
     'diffuse': np.array([1, 1, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.3},
     
    {'center': np.array([0, -9000, 0]), 'radius': 9000 - 1.5, 'ambient': np.array([0.1, 0.1, 0.1]),
    'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.6}
]

sun = objects[0]['center']
zzz = 0.3* 2 * pi
n1 = np.array([sin(zzz), cos(zzz), 0])
n = np.array([2, 3, 3.0])
n /= np.linalg.norm(n)
n2 = np.cross(n1, n)

thetas = np.array([0.2 * 2 * pi, 0.5*2*pi, 0.8*2*pi, 0.6*2*pi])
omegas = np.array([pi/8, pi/10, pi/20, pi/15])
objects[1]['center'] = sun + 2.5 * (n1*sin(thetas[0]) + n2*cos(thetas[0]))
objects[2]['center'] = sun + 4.5 * (n1*sin(thetas[1]) + n2*cos(thetas[1]))
objects[3]['center'] = sun + 5.5 * (n1*sin(thetas[2]) + n2*cos(thetas[2]))
objects[4]['center'] = sun + 7 * (n1*sin(thetas[3]) + n2*cos(thetas[3]))

cur = 0

REVOL_FRAMES = 200
STILL_FRAMES = 20
FALL_FRAMES = 40

for i in range(REVOL_FRAMES):
    thetas += omegas
    objects[1]['center'] = sun + 2.5 * (n1*sin(thetas[0]) + n2*cos(thetas[0]))
    objects[2]['center'] = sun + 4.5 * (n1*sin(thetas[1]) + n2*cos(thetas[1]))
    objects[3]['center'] = sun + 5.5 * (n1*sin(thetas[2]) + n2*cos(thetas[2]))
    objects[4]['center'] = sun + 7 * (n1*sin(thetas[3]) + n2*cos(thetas[3]))
    # objects[1]['center'] +=
    make_image(cur, objects)
    cur += 1

for i in range(STILL_FRAMES):
    make_image(cur, objects)
    cur += 1

mass = np.array([1, 2, 4, 3])
velo = np.zeros(4)
force = 20
accel = force / mass
dt = 0.05

for i in range(FALL_FRAMES):
    velo += accel * dt
    dy = velo*dt + accel*dt*dt/2
    for j in range(1, 1+4):
        objects[j]['center'][1] -= dy[j-1]
    make_image(cur, objects)
    cur += 1