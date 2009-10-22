import numpy as np
import scipy.stats

import htest
import zm2

def test_null():
    for N in 100, 1000, 10000:
        yield check_null, N

def check_null(N):
    H, M, fpp = htest.h_test(np.random.random(N))
    assert fpp>0.01

def test_non_null():
    H, M, fpp = htest.h_test(np.random.random(1000)/2)
    assert fpp<0.01

def test_fpp():
    for N in 100, 200:
        yield check_fpp, N, 1000, 0.05

def check_fpp(N,M,thresh):
    fp = 0
    for i in range(M):
        H, m, fpp = htest.h_test(np.random.random(N))
        if fpp<thresh:
            fp += 1
    print thresh, float(fp)/M
    assert scipy.stats.binom(M,thresh).sf(fp-1)>0.005
    assert scipy.stats.binom(M,thresh).cdf(fp-1)>0.005

def test_mean_stddev():
    Hs = []
    for i in range(1000):
        H, M, fpp = htest.h_test(np.random.random(1000))
        Hs.append(H)
    print np.mean(Hs), np.std(Hs)
    assert np.abs(np.mean(Hs)-2.51)<0.1
    assert np.abs(np.std(Hs)-2.51)<0.3

def test_compare_zm2():
    data = np.random.random(1000)
    H, M, fpp = htest.h_test(data)
    Z, zm2fpp = zm2.Zm2(data, M)
    assert np.abs(Z - (H+4*M-4))<1e-8
    assert fpp>zm2fpp
    
