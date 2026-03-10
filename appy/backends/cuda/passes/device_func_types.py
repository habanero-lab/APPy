# Type stubs for CUDA device functions.
# Each function returns the inferred return type given argument type strings.

# --- Math Functions ---
def acos(x): return x
def acosh(x): return x
def asin(x): return x
def asinh(x): return x
def atan(x): return x
def atan2(x, y): return x
def atanh(x): return x
def ceil(x): return x
def copysign(x, y): return x
def cos(x): return x
def cosh(x): return x
def exp(x): return x
def exp2(x): return x
def fabs(x): return x
def floor(x): return x
def fma(a, b, c): return a
def fmax(x, y): return x
def fmin(x, y): return x
def fmod(x, y): return x
def log(x): return x
def log2(x): return x
def log10(x): return x
def pow(x, y): return x
def rint(x): return x
def round(x): return x
def rsqrt(x): return x
def sin(x): return x
def sinh(x): return x
def sqrt(x): return x
def tan(x): return x
def tanh(x): return x
def trunc(x): return x

# --- Common Functions ---
def clamp(x, minval, maxval): return x
def max(x, y): return x
def min(x, y): return x
def abs(x): return x

# numpy ufunc aliases
def minimum(x, y): return x
def maximum(x, y): return x

# --- Type Conversion ---
def int(x): return 'int'
def float(x): return 'float'

# --- Random ---
def random(): return 'float'
