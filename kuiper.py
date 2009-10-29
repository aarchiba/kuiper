
# By Anne M. Archibald, 2007 and 2009
import numpy as np
from numpy import copy, sort, amax, arange, exp, sqrt, abs, floor, searchsorted
from scipy import factorial, comb
import itertools

def kuiper_FPP(D,N):
    """Compute the false positive probability for the Kuiper statistic.

    Uses the set of four formulas described in Paltani 2004; they report 
    the resulting function never underestimates the false positive probability 
    but can be a bit high in the N=40..50 range. (They quote a factor 1.5 at 
    the 1e-7 level.

    Parameters
    ----------
    D : float
        The Kuiper test score.
    N : float
        The effective sample size.

    Returns
    -------
    fpp : float
        The probability of a score this large arising from the null hypothesis.

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
    """Compute the Kuiper statistic.
    
    Use the Kuiper statistic version of the Kolmogorov-Smirnov test to 
    find the probability that something like data was drawn from the 
    distribution whose CDF is given as cdf.
    
    Parameters
    ----------
    data : array-like
        The data values.
    cdf : callable
        A callable to evaluate the CDF of the distribution being tested
        against. Will be called with a vector of all values at once.
    args : list-like, optional
        Additional arguments to be supplied to cdf.

    Returns
    -------
    D : float
        The raw statistic.
    fpp : float
        The probability of a D this large arising with a sample drawn from
        the distribution whose CDF is cdf.

    Notes
    -----
    The Kuiper statistic resembles the Kolmogorov-Smirnov test in that 
    it is nonparametric and invariant under reparameterizations of the data. 
    The Kuiper statistic, in addition, is equally sensitive throughout 
    the domain, and it is also invariant under cyclic permutations (making 
    it particularly appropriate for analyzing circular data). 

    Returns (D, fpp), where D is the Kuiper D number and fpp is the 
    probability that a value as large as D would occur if data was 
    drawn from cdf.

    Warning: The fpp is calculated only approximately, and it can be 
    as much as 1.5 times the true value.

    Stephens 1970 claims this is more effective than the KS at detecting 
    changes in the variance of a distribution; the KS is (he claims) more 
    sensitive at detecting changes in the mean.

    If cdf was obtained from data by fitting, then fpp is not correct and 
    it will be necessary to do Monte Carlo simulations to interpret D. 
    D should normally be independent of the shape of CDF.

    """

    # FIXME: doesn't work for distributions that are actually discrete (for example Poisson).
    data = sort(data)
    cdfv = cdf(data,*args)
    N = len(data)
    D = amax(cdfv-arange(N)/float(N)) + amax((arange(N)+1)/float(N)-cdfv)

    return D, kuiper_FPP(D,N)

def kuiper_two(data1, data2):
    """Compute the Kuiper statistic to compare two samples.

    Parameters
    ----------
    data1 : array-like
        The first set of data values.
    data2 : array-like
        The second set of data values.
    
    Returns
    -------
    D : float
        The raw test statistic.
    fpp : float
        The probability of obtaining two samples this different from
        the same distribution.

    Notes
    -----
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
    """Fold the weighted intervals to the interval (0,1).

    Convert a list of intervals (ai, bi, wi) to a list of non-overlapping
    intervals covering (0,1). Each output interval has a weight equal
    to the sum of the wis of all the intervals that include it. All intervals
    are interpreted modulo 1, and weights are accumulated counting 
    multiplicity.

    Parameters
    ----------
    intervals : list of three-element tuples (ai,bi,wi)
        The intervals to fold; ai and bi are the limits of the interval, and
        wi is the weight to apply to the interval.

    Returns
    -------
    breaks : array of floats length N
        The endpoints of a set of intervals covering [0,1]; breaks[0]=0 and
        breaks[-1] = 1
    weights : array of floats of length N-1
        The ith element is the sum of number of times the interval 
        breaks[i],breaks[i+1] is included in each interval times the weight
        associated with that interval.

    """
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
    """Construct a callable piecewise-linear CDF from a pair of arrays.
    
    Take a pair of arrays in the format returned by fold_intervals and
    make a callable cumulative distribution function on the interval
    (0,1).

    Parameters
    ----------
    breaks : array of floats of length N
        The boundaries of successive intervals.
    weights : array of floats of length N-1
        The weight for each interval.

    Returns
    -------
    f : callable
        A cumulative distribution function corresponding to the 
        piecewise-constant probability distribution given by breaks, weights

    """
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
    """Compute the length of overlap of two intervals.
    
    Parameters
    ----------
    i1, i2 : pairs of two floats
        The two intervals.

    Returns
    -------
    l : float
        The length of the overlap between the two intervals.
    
    """
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
    """Histogram of a piecewise-constant PDF.

    This function takes a piecewise-constant PDF and computes the probability
    density in each histogram bin.

    Parameters
    ----------
    n : int
        The number of bins
    breaks : array of floats of length N
        Endpoints of the intervals in the PDF
    totals : array of floats of length N-1
        Probability densities in each bin

    """
    h = np.zeros(n)
    start = breaks[0]
    for i in range(len(totals)):
        end = breaks[i+1]
        for j in range(n):
            ol = interval_overlap_length((float(j)/n,float(j+1)/n),(start,end))
            h[j] += ol/(1./n)*totals[i]
        start = end

    return h
