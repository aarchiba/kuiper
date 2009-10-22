import numpy as np
import scipy.stats

import zm2

def test_null():
    for N in 100, 1000, 10000:
        yield check_null, N

def check_null(N):
    Z, fpp = zm2.Zm2(np.random.random(N),5)
    assert fpp>0.01

def test_non_null():
    Z, fpp = zm2.Zm2(np.random.random(1000)/2,5)
    assert fpp<0.01

def test_fpp():
    for N in 100, 200:
        yield check_fpp, N, 1000, 0.05

def check_fpp(N,M,thresh):
    fp = 0
    for i in range(M):
        Z, fpp = zm2.Zm2(np.random.random(N),5)
        if fpp<thresh:
            fp += 1
    print thresh, float(fp)/M
    assert scipy.stats.binom(M,thresh).sf(fp-1)>0.005
    assert scipy.stats.binom(M,thresh).cdf(fp-1)>0.005

