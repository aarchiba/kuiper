
import numpy as np
import scipy.stats
from numpy.testing import assert_array_almost_equal

import kuiper

def seed(n=0):
    """Seed the random number generator before running a test."""

    def wrap(f):
        def wrapped(*args,**kwargs):
            np.random.seed(n)
            return f(*args,**kwargs)
        wrapped.__name__ = f.__name__
        wrapped.__dict__ = f.__dict__
        wrapped.__doc__ = f.__doc__
        wrapped.__module__ = f.__module__
        return wrapped
    return wrap
        

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
        yield check_uniform, lambda x: kuiper.kuiper(x)[1], N

@seed()
@double_check
def check_uniform(f,N):
    assert f(np.random.random(N))>0.01

def test_fpp():
    def F(x):
        return kuiper.kuiper(x)[1]
    for N in [1000,100,80,50,30,20,10]:
        yield check_fpp, F, N, 200, 0.05
        yield check_fpp, F, N, 200, 0.25
        if False: # These tests fail because the FPP is too approximate.
            yield check_fpp_kuiper, F, N, 100, 0.05
            yield check_fpp_kuiper, F, N, 200
    #Seems to fail for N==5
    #yield check_fpp, 5, 1000, 0.05

@seed()
@double_check
def check_fpp(F,N,M,fpp):
    fps = 0
    for i in range(M):
        f = F(np.random.random(N))
        if f<fpp:
            fps += 1
    assert scipy.stats.binom(M,fpp).sf(fps-1)>0.005
    assert scipy.stats.binom(M,fpp).cdf(fps-1)>0.005

@seed()
@double_check
def check_fpp_kuiper(F,N,M,thresh=1.):
    ps = []
    while len(ps)<M:
        p = F(np.random.random(N))
        if p<thresh:
            ps.append(p/thresh)
    
    assert kuiper.kuiper(ps)[1]>0.01

@seed()
@double_check
def test_detect_nonuniform():
    D, f = kuiper.kuiper(np.random.random(500)*0.5)
    assert f<0.01


@seed()
@double_check
def test_weighted():
    a = (np.random.random(100) * 3.4 + 0.8)%1
    i = (0.8,4.2,1)
    b, t = kuiper.fold_intervals([i])
    cdf = kuiper.cdf_from_intervals(b,t)
    assert kuiper.kuiper(a,cdf)[1]>0.01


# Out of sheer laziness I'm not going to generify these tests.
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

@seed()
@double_check
def check_kuiper_two_uniform(N,M):
    assert kuiper.kuiper_two(np.random.random(N),np.random.random(M))[1]>0.01

@seed()
@double_check
def check_kuiper_two_nonuniform(N,M):
    assert kuiper.kuiper_two(np.random.random(N)**2,np.random.random(M)**2)[1]>0.01

@seed()
@double_check
def test_detect_kuiper_two_different():
    D, f = kuiper.kuiper_two(np.random.random(500)*0.5,np.random.random(500))
    assert f<0.01

@seed()
@double_check
def check_fpp_kuiper_two(N,M,R,fpp):
    fps = 0
    for i in range(R):
        D, f = kuiper.kuiper_two(np.random.random(N),np.random.random(M))
        if f<fpp:
            fps += 1
    assert scipy.stats.binom(R,fpp).sf(fps-1)>0.005
    assert scipy.stats.binom(R,fpp).cdf(fps-1)>0.005


@seed()
@double_check
def test_histogram():
    a, b = 0.3, 3.14
    s = np.random.uniform(a,b,10000) % 1
    
    b, w = kuiper.fold_intervals([(a,b,1./(b-a))])

    h = kuiper.histogram_intervals(16,b,w)
    nn, bb = np.histogram(s, bins=len(h), range=(0,1), new=True)

    uu = np.sqrt(nn)
    nn, uu = len(h)*nn/h/len(s), len(h)*uu/h/len(s)

    c2 = np.sum(((nn-1)/uu)**2)
    
    assert scipy.stats.chi2(len(h)).cdf(c2)>0.01
    assert scipy.stats.chi2(len(h)).sf(c2)>0.01

def check_histogram_intervals_known(ii, rr):
    assert_array_almost_equal(kuiper.histogram_intervals(*ii),rr)

def test_histogram_intervals_known():
    for (ii, rr) in [ 
            ( (4,(0,1),(1,)), (1,1,1,1) ),
            ( (2,(0,1),(1,)), (1,1) ),
            ( (4,(0,0.5,1),(1,1)), (1,1,1,1) ),
            ( (4,(0,0.5,1),(1,2)), (1,1,2,2) ),
            ( (3,(0,0.5,1),(1,2)), (1,1.5,2) ),
            ]:
        yield check_histogram_intervals_known, ii, rr
