import numpy as np
import scipy.stats

from test_kuiper import seed, double_check, check_uniform, check_fpp
import zm2

def test_null():
    for N in 100, 1000, 10000:
        for m in [1,2,5]:
            yield check_uniform, lambda x: zm2.zm2(x,m)[1], N

@seed()
@double_check
def test_non_null():
    Z, fpp = zm2.zm2(np.random.random(1000)/2,5)
    assert fpp<0.01

def test_fpp():
    for N in 10, 100, 500:
        for m in [1,2,5]:
            yield check_fpp, lambda x: zm2.zm2(x,m)[1], N, 1000, 0.05

