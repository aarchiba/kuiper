import numpy as np
import scipy.stats

from test_kuiper import seed, double_check, check_uniform, check_fpp, check_fpp_kuiper
import anderson_darling

def F(x):
    return anderson_darling.anderson_darling(x)[-1]


def test_fpp_values():
    for (f, p)  in [(1.9329578, 0.9),
                   (2.492367, 0.95), 
                   (3.878125, 0.99),
                   (9, 0.999960466)]:
        assert np.abs(anderson_darling.anderson_darling_fpp(f)-(1-p))<0.0001

def test_uniform():
    for N in 20, 100, 1000:
        yield check_uniform, F, N

@seed()
@double_check
def test_non_null():
    A2, fpp = anderson_darling.anderson_darling(np.random.random(1000)/2)
    assert fpp<0.01


def test_fpp():
    for N in 10, 50, 100:
        yield check_fpp, F, N, 1000, 0.05
        yield check_fpp_kuiper, F, N, 1000


@seed()
@double_check
def check_fpp_ad_nonuniform(N, M, cdf, cdfinv):
    r = []
    for i in range(M):
        s = cdfinv(np.random.random(N))
        A2, fpp = anderson_darling.anderson_darling(s,cdf)
        r.append(fpp)
    assert anderson_darling.anderson_darling(r)>0.01

def test_fpp_nonuniform():
    for N in 10, 50, 100:
        yield check_fpp_ad_nonuniform, N, 1000, np.exp, np.log
        yield check_fpp_ad_nonuniform, N, 1000, lambda x: x**3, lambda x: x**(1./3)

        yield check_fpp_kuiper, F, N, 1000
