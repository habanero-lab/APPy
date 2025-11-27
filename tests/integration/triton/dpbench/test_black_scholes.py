# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


def initialize(nopt, seed, types_dict):
    import numpy as np
    import numpy.random as default_rng

    dtype: np.dtype = types_dict["float"]
    S0L = dtype.type(10.0)
    S0H = dtype.type(50.0)
    XL = dtype.type(10.0)
    XH = dtype.type(50.0)
    TL = dtype.type(1.0)
    TH = dtype.type(2.0)
    RISK_FREE = dtype.type(0.1)
    VOLATILITY = dtype.type(0.2)

    default_rng.seed(seed)
    price = default_rng.uniform(S0L, S0H, nopt).astype(dtype)
    strike = default_rng.uniform(XL, XH, nopt).astype(dtype)
    t = default_rng.uniform(TL, TH, nopt).astype(dtype)
    rate = RISK_FREE
    volatility = VOLATILITY
    call = np.zeros(nopt, dtype=dtype)
    put = -np.ones(nopt, dtype=dtype)

    return (price, strike, t, rate, volatility, call, put)


import numpy as np
from scipy.special import erf

def black_scholes_np(nopt, price, strike, t, rate, volatility, call, put):
    mr = -rate
    sig_sig_two = volatility * volatility * 2

    P = price
    S = strike
    T = t

    a = np.log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = np.true_divide(1.0, np.sqrt(z))

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * erf(w1)
    d2 = 0.5 + 0.5 * erf(w2)

    Se = np.exp(b) * S

    call[:] = P * d1 - Se * d2
    put[:] = call - P + Se


import appy

@appy.jit(backend="triton", dump_code=True)
def black_scholes_appy(nopt, price, strike, t, rate, volatility, call, put):
    mr = -rate
    sig_sig_two = volatility * volatility * 2
    #pragma parallel for simd
    for i in range(nopt):
        P = price[i]
        S = strike[i]
        T = t[i]

        a = np.log(P / S)
        b = T * mr

        z = T * sig_sig_two
        c = 0.25 * z
        y = 1.0 / np.sqrt(z)

        w1 = (a - b + c) * y
        w2 = (a - b - c) * y

        d1 = 0.5 + 0.5 * erf(w1)
        d2 = 0.5 + 0.5 * erf(w2)

        Se = np.exp(b) * S

        r = P * d1 - Se * d2
        call[i] = r
        put[i] = r - P + Se


def test():
    import numpy as np

    nopt = 1000000

    price, strike, t, rate, volatility, call, put = initialize(nopt, 0, {"float": np.dtype(np.float64)})

    call1, put1 = call.copy(), put.copy()
    call2, put2 = call.copy(), put.copy()

    black_scholes_np(nopt, price, strike, t, rate, volatility, call1, put1)
    black_scholes_appy(nopt, price, strike, t, rate, volatility, call2, put2)

    assert np.allclose(call1, call2)
    assert np.allclose(put1, put2)
