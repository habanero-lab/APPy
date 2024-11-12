import appy
import inspect

def add_two_vec(a, b):
    c = appy.empty_like(a)
    #pragma parallel for simd
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    return c

src = inspect.getsource(add_two_vec)
newcode = appy.compile_from_src(src)
print(newcode)