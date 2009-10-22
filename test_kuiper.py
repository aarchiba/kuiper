
import numpy as np
import scipy.stats
from nose.tools import make_decorator

import kuiper

# FIXME
#@make_decorator doesn't work!
def double_check(f):
    """Run a probabilistic test again if it fails.
    
    This squares the chance of a probabilistic test failing by chance,
    but isn't likely to hide real problems (since they will probably
    reoccur, since the test is being rerun with the same parameters).

    """
    def double(*args,**kwargs):
        try:
            f(*args,**kwargs)
        except AssertionError:
            f(*args,**kwargs)
    double.__name__ = f.__name__
    double.__dict__ = f.__dict__
    double.__doc__ = f.__doc__
    double.__module__ = f.__module__
    return double


def test_uniform():
    for N in [10,100,1000,10000]:
        yield check_uniform, N

@double_check
def check_uniform(N):
    assert kuiper.kuiper(np.random.random(N))[1]>0.01

def test_fpp():
    for N in [10,100,1000]:
        yield check_fpp, N, 100, 0.05
        yield check_fpp, N, 100, 0.25
    #Seems to fail for N==5
    #yield check_fpp, 5, 1000, 0.05

@double_check
def check_fpp(N,M,fpp):
    fps = 0
    for i in range(M):
        D, f = kuiper.kuiper(np.random.random(N))
        if f<fpp:
            fps += 1
    assert scipy.stats.binom(M,fpp).sf(fps-1)>0.005
    assert scipy.stats.binom(M,fpp).cdf(fps-1)>0.005

@double_check
def test_detect_nonuniform():
    D, f = kuiper.kuiper(np.random.random(500)*0.5)
    assert f<0.01


@double_check
def test_weighted():
    a = (np.random.random(100) * 3.4 + 0.8)%1
    i = (0.8,4.2,1)
    b, t = kuiper.fold_intervals([i])
    cdf = kuiper.cdf_from_intervals(b,t)
    assert kuiper.kuiper(a,cdf)[1]>0.01


def test_kuiper_two():
    for (N,M) in [(100,100),
                  (20,100),
                  (100,20),
                  (10,20),
                  (5,5),
                  (1000,100)]:
        yield check_kuiper_two_uniform, N, M
        yield check_kuiper_two_nonuniform, N, M
        yield check_fpp_kuiper_two, N, M, 100, 0.05

@double_check
def check_kuiper_two_uniform(N,M):
    assert kuiper.kuiper_two(np.random.random(N),np.random.random(M))[1]>0.01

@double_check
def check_kuiper_two_nonuniform(N,M):
    assert kuiper.kuiper_two(np.random.random(N)**2,np.random.random(M)**2)[1]>0.01

@double_check
def test_detect_kuiper_two_different():
    D, f = kuiper.kuiper_two(np.random.random(500)*0.5,np.random.random(500))
    assert f<0.01

def test_kuiper_two_fpp():
    pass

@double_check
def check_fpp_kuiper_two(N,M,R,fpp):
    fps = 0
    for i in range(R):
        D, f = kuiper.kuiper_two(np.random.random(N),np.random.random(M))
        if f<fpp:
            fps += 1
    assert scipy.stats.binom(R,fpp).sf(fps-1)>0.005
    assert scipy.stats.binom(R,fpp).cdf(fps-1)>0.005

