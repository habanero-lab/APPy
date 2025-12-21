import numpy as np

def vec3(a, b, c):
    return np.array([a, b, c])

def vec2(a, b):
    return np.array([a, b])

def length(vec):
    return np.linalg.norm(vec)

def normalize(vec):
    return vec / np.linalg.norm(vec)

def dot(a, b):
    return a @ b

def distance(a, b):
    return np.linalg.norm(a - b)

def step(edge, x):
    if x < edge:
        return 0
    else:
        return 1
    
def clamp(x, minval, maxval):
    return max(min(x, maxval), minval)
    
def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)