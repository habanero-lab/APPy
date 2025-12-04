import appy.np_shared as nps
import numpy as np

def test():
    a = nps.zeros((100, 200))
    assert np.array_equal(np.exp(a * 2), np.ones((100, 200))) 

if __name__ == '__main__':
    test()