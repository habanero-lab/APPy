import numpy as np
import math

# -------------------------------------------------
# Vector constructors
# -------------------------------------------------

def vec3(a, b, c):
    return np.array([a, b, c])

def vec2(a, b):
    return np.array([a, b])

# -------------------------------------------------
# Basic vector ops
# -------------------------------------------------

def length(vec):
    return np.linalg.norm(vec)

def normalize(vec):
    return vec / np.linalg.norm(vec)

def dot(a, b):
    return a @ b

def distance(a, b):
    return np.linalg.norm(a - b)

# -------------------------------------------------
# Scalar utility functions (GLSL / Metal style)
# -------------------------------------------------

def step(edge, x):
    return np.where(x < edge, 0, 1)

def clamp(x, minval, maxval):
    return np.minimum(np.maximum(x, minval), maxval)

def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# -------------------------------------------------
# Logarithms
# -------------------------------------------------

def log(x):
    return np.log(x)

def log2(x):
    return np.log2(x)

def log10(x):
    return np.log10(x)

# -------------------------------------------------
# Power functions
# -------------------------------------------------

def pow(x, y):
    return np.power(x, y)

def powr(x, y):
    # Metal: x >= 0 required (no check enforced here)
    return np.power(x, y)

# -------------------------------------------------
# Rounding
# -------------------------------------------------

def rint(x):
    # Round to nearest even (IEEE-754)
    return np.rint(x)

def round(x):
    # Round half away from zero (Metal semantics)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def trunc(x):
    return np.trunc(x)

# -------------------------------------------------
# Roots
# -------------------------------------------------

def sqrt(x):
    return np.sqrt(x)

def rsqrt(x):
    return 1.0 / np.sqrt(x)

# -------------------------------------------------
# Trigonometry
# -------------------------------------------------

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def sincos(x):
    """
    Metal-style sincos:
    returns (sin(x), cos(x))
    """
    return np.sin(x), np.cos(x)

def tan(x):
    return np.tan(x)

# -------------------------------------------------
# Hyperbolic
# -------------------------------------------------

def sinh(x):
    return np.sinh(x)

def tanh(x):
    return np.tanh(x)

# -------------------------------------------------
# Pi-scaled trigonometry
# -------------------------------------------------

def sinpi(x):
    return np.sin(np.pi * x)

def tanpi(x):
    return np.tan(np.pi * x)

# -------------------------------------------------
# Inverse trigonometry
# -------------------------------------------------

def acos(x):
    return np.arccos(x)

def asin(x):
    return np.arcsin(x)

def atan(x):
    return np.arctan(x)

def atan2(y, x):
    return np.arctan2(y, x)

# -------------------------------------------------
# Hyperbolic / inverse hyperbolic
# -------------------------------------------------

def cosh(x):
    return np.cosh(x)

def acosh(x):
    return np.arccosh(x)

def asinh(x):
    return np.arcsinh(x)

def atanh(x):
    return np.arctanh(x)

# -------------------------------------------------
# Rounding
# -------------------------------------------------

def ceil(x):
    return np.ceil(x)

def floor(x):
    return np.floor(x)

def copysign(x, y):
    return np.copysign(x, y)

# -------------------------------------------------
# Exponentials
# -------------------------------------------------

def exp(x):
    return np.exp(x)

def exp2(x):
    return np.exp2(x)

def exp10(x):
    return np.power(10.0, x)

# -------------------------------------------------
# Absolute value
# -------------------------------------------------

def fabs(x):
    return np.fabs(x)

def abs(x):
    # Shadowing built-in abs intentionally (Metal-style)
    return np.fabs(x)

def fdim(x, y):
    # x - y if x > y else +0
    return np.maximum(x - y, 0.0)

# -------------------------------------------------
# Arithmetic helpers
# -------------------------------------------------

def divide(x, y):
    return x / y

def fma(a, b, c):
    """
    Fused multiply-add.
    NumPy provides IEEE-compliant fma where supported.
    """
    return a * b + c

# -------------------------------------------------
# Min / Max (Metal semantics)
# -------------------------------------------------

def fmax(x, y):
    return np.where(np.isnan(x), y,
           np.where(np.isnan(y), x,
           np.maximum(x, y)))

def fmin(x, y):
    return np.where(np.isnan(x), y,
           np.where(np.isnan(y), x,
           np.minimum(x, y)))

def max(x, y):
    return fmax(x, y)

def min(x, y):
    return fmin(x, y)

def fmax3(x, y, z):
    return fmax(x, fmax(y, z))

def max3(x, y, z):
    return fmax3(x, y, z)

def fmin3(x, y, z):
    return fmin(x, fmin(y, z))

def min3(x, y, z):
    return fmin3(x, y, z)

# -------------------------------------------------
# Median
# -------------------------------------------------

def fmedian3(x, y, z):
    return x + y + z - fmin3(x, y, z) - fmax3(x, y, z)

def median3(x, y, z):
    return fmedian3(x, y, z)

# -------------------------------------------------
# Modulo / fractional
# -------------------------------------------------

def fmod(x, y):
    return x - y * np.trunc(x / y)

def fract(x):
    return x - np.floor(x)

def cospi(x):
    return np.cos(np.pi * x)
