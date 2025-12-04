import appy.np_shared as nps
import numpy as np

def test():
    a = nps.empty((100, 200))
    assert np.array_equal(a + 1, a.arr + 1) 

if __name__ == '__main__':
    test()