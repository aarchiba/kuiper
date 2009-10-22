
# By Anne M. Archibald, 2007 and 2009
import numpy as np
from numpy import copy, sort, amax, arange, exp, sqrt, abs, floor, searchsorted
from scipy import factorial, comb
import itertools

def kuiper_FPP(D,N):
    """Compute the false positive probability for the Kuiper statistic

    Uses the set of four formulas described in Paltani 2004; they report the resulting function never underestimates the false positive probability but can be a bit high in the N=40..50 range. (They quote a factor 1.5 at the 1e-7 level.
    """
    if D<0. or D>2.:
        raise ValueError("Must have 0<=D<=2 by definition of the Kuiper test")

    if D<2./N:
        return 1. - factorial(N)*(D-1./N)**(N-1)
    elif D<3./N:
        k = -(N*D-1.)/2.
        r = sqrt(k**2 - (N*D-2.)/2.)
        a, b = -k+r, -k-r
        return 1. - factorial(N-1)*(b**(N-1.)*(1.-a)-a**(N-1.)*(1.-b))/float(N)**(N-2)*(b-a)
    elif (D>0.5 and N%2==0) or (D>(N-1.)/(2.*N) and N%2==1):
        def T(t):
            y = D+t/float(N)
            return y**(t-3)*(y**3*N-y**2*t*(3.-2./N)/N-t*(t-1)*(t-2)/float(N)**2)
        s = 0.
        # NOTE: the upper limit of this sum is taken from Stephens 1965
        for t in xrange(int(floor(N*(1-D)))+1):
            term = T(t)*comb(N,t)*(1-D-t/float(N))**(N-t-1)
            s += term
        return s
    else:
        z = D*sqrt(N) 
        S1 = 0.
        term_eps = 1e-12
        abs_eps = 1e-100
        for m in itertools.count(1):
            T1 = 2.*(4.*m**2*z**2-1.)*exp(-2.*m**2*z**2)
            so = S1
            S1 += T1
            if abs(S1-so)/(abs(S1)+abs(so))<term_eps or abs(S1-so)<abs_eps:
                break
        S2 = 0.
        for m in itertools.count(1):
            T2 = m**2*(4.*m**2*z**2-3.)*exp(-2*m**2*z**2)
            so = S2
            S2 += T2
            if abs(S2-so)/(abs(S2)+abs(so))<term_eps or abs(S1-so)<abs_eps:
                break
        return S1 - 8*D/(3.*sqrt(N))*S2

def kuiper(data, cdf=lambda x: x, args=()):
    """Compute the Kuiper statistic
    
    Use the Kuiper statistic version of the Kolmogorov-Smirnov test to find the probability that something like data was drawn from the distribution whose CDF is given as cdf.

    The Kuiper statistic resembles the Kolmogorov-Smirnov test in that it is nonparametric and invariant under reparameterizations of the data. The Kuiper statistic, in addition, is equally sensitive throughout the domain, and it is also invariant under cyclic permutations (making it particularly appropriate for analyzing circular data). 

    Returns (D, fpp), where D is the Kuiper D number and fpp is the probability that a value as large as D would occur if data was drawn from cdf.

    Warning: The fpp is calculated only approximately, and it can be as much as 1.5 times the true value.

    Stephens 1970 claims this is more effective than the KS at detecting changes in the variance of a distribution; the KS is (he claims) more sensitive at detecting changes in the mean.

    If cdf was obtained from data by fitting, then fpp is not correct and it will be necessary to do Monte Carlo simulations to interpret D. D should normally be independent of the shape of CDF.

    """

    # FIXME: doesn't work for distributions that are actually discrete (for example Poisson).
    data = sort(data)
    cdfv = cdf(data,*args)
    N = len(data)
    D = amax(cdfv-arange(N)/float(N)) + amax((arange(N)+1)/float(N)-cdfv)

    return D, kuiper_FPP(D,N)

def kuiper_two(data1, data2):
    """Compute the Kuiper statistic to compare two samples

    Warning: the fpp is quite approximate, especially for small samples.
    """
    data1, data2 = sort(data1), sort(data2)

    if len(data2)<len(data1):
        data1, data2 = data2, data1

    cdfv1 = searchsorted(data2, data1)/float(len(data2)) # this could be more efficient
    cdfv2 = searchsorted(data1, data2)/float(len(data1)) # this could be more efficient
    D = (amax(cdfv1-arange(len(data1))/float(len(data1))) + 
            amax(cdfv2-arange(len(data2))/float(len(data2))))

    Ne = len(data1)*len(data2)/float(len(data1)+len(data2))
    return D, kuiper_FPP(D, Ne)


def fold_intervals(intervals):
    r = []
    breaks = set()
    tot = 0
    for (a,b,wt) in intervals:
        tot += (np.ceil(b)-np.floor(a))*wt
        fa = a%1
        breaks.add(fa)
        r.append((0,fa,-wt))
        fb = b%1
        breaks.add(fb)
        r.append((fb,1,-wt))

    breaks.add(0.)
    breaks.add(1.)
    breaks = list(breaks)
    breaks.sort()
    breaks_map = dict([(f,i) for (i,f) in enumerate(breaks)])
    totals = np.zeros(len(breaks)-1)
    totals += tot
    for (a,b,wt) in r:
        totals[breaks_map[a]:breaks_map[b]]+=wt
    return np.array(breaks), totals

def cdf_from_intervals(breaks, totals):
    if breaks[0]!=0 or breaks[-1]!=1:
        raise ValueError("Intervals must be restricted to [0,1]")
    if np.any(np.diff(breaks)<=0):
        raise ValueError("Breaks must be strictly increasing")
    if np.any(totals<0):
        raise ValueError("Total weights in each subinterval must be nonnegative")
    if np.all(totals==0):
        raise ValueError("At least one interval must have positive exposure")
    b = breaks.copy()
    c = np.concatenate(((0,), np.cumsum(totals*np.diff(b))))
    c /= c[-1]
    def cdf(x):
        ix = np.searchsorted(b[:-1],x)
        l, r = b[ix-1], b[ix] 
        return ((r-x)*c[ix-1]+(x-l)*c[ix])/(r-l)
    return cdf

def interval_overlap_length(i1,i2):
    (a,b) = i1
    (c,d) = i2
    if a<c:
        if b<c:
            return 0.
        elif b<d:
            return b-c
        else:
            return d-c
    elif a<d:
        if b<d:
            return b-a
        else:
            return d-a
    else:
        return 0

def histogram_intervals(n, breaks, totals):
    h = np.zeros(n)
    start = breaks[0]
    for i in range(len(totals)):
        end = breaks[i+1]
        for j in range(n):
            ol = interval_overlap_length((float(j)/n,float(j+1)/n),(start,end))
            h[j] += ol/(1./n)*totals[i]
        start = end

    return h
