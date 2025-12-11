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
def cospi(x): return x
def divide(x, y): return x
def exp(x): return x
def exp2(x): return x
def exp10(x): return x
def fabs(x): return x
def fdim(x, y): return x
def floor(x): return x
def fma(a, b, c): return a
def fmax(x, y): return x
def fmax3(x, y, z): return x
def fmedian3(x, y, z): return x
def fmin(x, y): return x
def fmin3(x, y, z): return x
def fmod(x, y): return x
def fract(x): return x
def frexp(x, exponent): return x
def ilogb(x): return x
def ldexp(x, k): return x
def log(x): return x
def log2(x): return x
def log10(x): return x
def modf(x, intval): return x
def nextafter(x, y): return x
def pow(x, y): return x
def powr(x, y): return x
def rint(x): return x
def round(x): return x
def rsqrt(x): return x
def sin(x): return x
def sincos(x, cosval): return x
def sinh(x): return x
def sinpi(x): return x
def sqrt(x): return x
def tan(x): return x
def tanh(x): return x
def tanpi(x): return x
def trunc(x): return x

# --- Common Functions ---
def clamp(x, minval, maxval): return x
def mix(x, y, a): return x
def saturate(x): return x
def sign(x): return x
def smoothstep(edge0, edge1, x): return x
def step(edge, x): return x

# --- Integer Functions ---
def abs(x): return x  # renamed to avoid clash with fabs
def absdiff(x, y): return x
def addsat(x, y): return x
def clz(x): return x
def ctz(x): return x
def extract_bits(x, offset, bits): return x
def hadd(x, y): return x
def insert_bits(base, insert, offset, bits): return base
def mad24(x, y, z): return x
def madhi(a, b, c): return a
def madsat(a, b, c): return a
def max(x, y): return x
def max3(x, y, z): return x
def median3_int(x, y, z): return x
def min(x, y): return x
def min3(x, y, z): return x
def mul24(x, y): return x
def mulhi(x, y): return x
def popcount(x): return x
def reverse_bits(x): return x
def rhadd(x, y): return x
def rotate(v, i): return v
def subsat(x, y): return x

# --- Relational Functions ---
def all(x): return "bool"
def any(x): return "bool"
def isfinite(x): return x
def isinf(x): return x
def isnan(x): return x
def isnormal(x): return x
def isordered(x, y): return x
def isunordered(x, y): return x
def not_fn(x): return x  # renamed because `not` is a keyword
def select(a, b, c): return a
def signbit(x): return x

# --- Geometric Functions ---
def normalize(x): return x
def dot(x, y):
    assert x[-1] in ['2', '3', '4'] and y[-1] in ['2', '3', '4']
    return x[:-1]

def distance(x, y):
    assert x[-1] in ['2', '3', '4'] and y[-1] in ['2', '3', '4']
    return x[:-1]

# --- Type Conversion Functions ---
def int(x): return "int"
def float(x): return "float"
def float3(x, y, z): return "float3"


# --- My Functions ---
def random(): return "float"