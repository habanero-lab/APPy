import numpy as np
import appy


def test1():
    @appy.jit
    def foo(x):
        return x
    
    assert foo([1, 2, 3]) == [1, 2, 3]


def test2():
    @appy.jit
    def foo(x):
        for i in range(x.shape[0]):
            x[i] += 1
        return x
    
    assert all(foo(np.zeros(10)) == np.ones(10))


def test3():
    @appy.jit
    def foo(x):
        s = 0
        for i in range(x.shape[0]):
            s += x[i]
        return s
    
    assert foo(np.arange(10)) == 45
