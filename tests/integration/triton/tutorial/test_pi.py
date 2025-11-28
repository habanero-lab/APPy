import appy
import numpy as np

# Set the number of steps
num_steps = 1_000_000
step = 1.0 / num_steps

@appy.jit(dump_code=True)
def compute_pi_appy(num_steps, step):
    sum = 0.0
    #pragma parallel for simd shared(sum,step)
    for i in range(1, num_steps + 1):
        x = (i - 0.5) * step
        sum += 4.0 / (1.0 + x * x)
    return step * sum

def compute_pi(num_steps, step):
    sum = 0.0
    for i in range(1, num_steps + 1):
        x = (i - 0.5) * step
        sum += 4.0 / (1.0 + x * x)
    return step * sum


def test():
    assert np.isclose(compute_pi(num_steps, step), compute_pi_appy(num_steps, step), rtol=1e-06, atol=1e-08)
