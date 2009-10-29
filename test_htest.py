import numpy as np
import scipy.stats

from test_kuiper import seed, double_check, check_uniform, check_fpp
import htest
import zm2

def F(x):
    return htest.h_test(x)[2]

def test_uniform():
    for N in 20, 100, 1000:
        yield check_uniform, F, N

@seed()
@double_check
def test_non_null():
    H, M, fpp = htest.h_test(np.random.random(1000)/2)
    assert fpp<0.01


def test_fpp():
    for N in 10, 50, 100:
        yield check_fpp, F, N, 1000, 0.05

@seed()
@double_check
def test_mean_stddev():
    Hs = []
    for i in range(1000):
        H, M, fpp = htest.h_test(np.random.random(100))
        Hs.append(H)
    print np.mean(Hs), np.std(Hs)
    assert np.abs(np.mean(Hs)-2.51)<0.1
    assert np.abs(np.std(Hs)-2.51)<0.3

@seed()
def test_compare_zm2():
    data = np.random.random(1000)
    H, M, fpp = htest.h_test(data)
    Z, zm2fpp = zm2.zm2(data, M)
    assert np.abs(Z - (H+4*M-4))<1e-8
    assert fpp>zm2fpp
    
