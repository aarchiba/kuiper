
import numpy as np
import scipy.stats

import kuiper

# FIXME
def double_check(f):
    """Run a probabilistic test again if it fails"""
    def double(*args,**kwargs):
        try:
            f(*args,**kwargs)
        except AssertionError:
            f(*args,**kwargs)


def test_uniform():
    for N in [100,1000,10000]:
        yield check_uniform, N

#@double_check
def check_uniform(N):
    assert kuiper.kuiper(np.random.random(N))[1]>0.01

def test_fpp():
    for N in [100,1000]:
        yield check_fpp, N, 100, 0.05

#@double_check
def check_fpp(N,M,fpp):
    fps = 0
    for i in range(M):
        D, f = kuiper.kuiper(np.random.random(N))
        if f<fpp:
            fps += 1
    assert scipy.stats.binom(M,fpp).sf(fps-1)>0.005
    assert scipy.stats.binom(M,fpp).cdf(fps-1)>0.005

def test_detect_nonuniform():
    D, f = kuiper.kuiper(np.random.random(500)*0.5)
    assert f<0.01


def test_weighted():
    a = (np.random.random(100) * 3.4 + 0.8)%1
    i = (0.8,4.2,1)
    b, t = kuiper.fold_intervals([i])
    cdf = kuiper.cdf_from_intervals(b,t)
    assert kuiper.kuiper(a,cdf)[1]>0.01
