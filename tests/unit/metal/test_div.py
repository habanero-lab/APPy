import appy.np_shared as nps
import numpy as np
import appy

@appy.jit(dump_code=True)
def div_kernel(out, alpha, beta):
    for i in appy.prange(out.shape[0]):
        out[i] = alpha / beta

    
def test_uint8_div_uint8():
    out = nps.empty((10,), dtype=np.float32)
    alpha = np.uint8(1)
    beta = np.uint8(2)
    div_kernel(out, alpha, beta)
    
    assert np.allclose(out, 0.5)