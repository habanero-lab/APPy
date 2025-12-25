import appy
import numpy as np



@appy.cpu_shader(dump_code=1)
def test():
    height = 100
    width = 100
    out = np.empty((height, width, 4), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            out[i,j] = 255, 255, 255, 255
    assert np.all(out == 255)


if __name__ == '__main__':
    test()
