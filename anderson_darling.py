import numpy as np
import scipy.stats

_adpoly1 = np.poly1d([0.00168691, -0.0116720, 0.0347962, -0.0649821, 0.247105, 2.00012])
_adpoly2 = np.poly1d([-0.0003146, 0.008056, -0.082433, 0.43424, -2.30695, 1.0776])
def adinf(z):
    """Compute the asymptotic (large N) Anderson-Darling CDF.

    This is an approximate calculation, with error less than 2e-6 for
    A2<2 and error less than 8e-7 for A2>2. Note that this is one minus
    the false positive probability.

    Parameters
    ==========
    z : positive float
        The value returned by the Anderson-Darling test.

    Returns
    =======
    p : float between zero and one
        The probability that a value less than z would be returned.

    Reference
    =========
    Marsaglia, George, and Marsaglia, John, Journal of Statistical Software, 
    Vol. 9, No. 2. (February 2004), pp. 1-5.

    """
    if 0<z<2:
        return np.exp(-1.2337141/z)/np.sqrt(z)*_adpoly1(z)
    elif 2<=z:
        return np.exp(-np.exp(_adpoly2(z)))
    else:
        raise ValueError("Anderson-Darling statistic must be positive")

_g1 = lambda x: np.sqrt(x)*(1-x)*(49*x-102)
_g2 = np.poly1d([1.91864, -8.259, 14.458, -14.6538, 6.54034, -.00022633])
_g3 = np.poly1d([255.7844, -1116.360, 1950.646, -1705.091, 745.2337, -130.2137])
_c = lambda n: 0.01265 + 0.1757/n

def errfix(n, x):
    """Compute Anderson-Darling correction term for finite n.

    This is an approximate calculation that should give a result 
    good to four decimal places.

    Parameters
    ==========
    n : integer or None
        The sample size used.
    x : float between zero and one
        The asymptotic Anderson-Darling CDF value.

    Returns
    =======
    r : float
        The correction that should be added to the Anderson-Darling CDF
        to give a finite-n value good to four decimal places.

    Reference
    =========
    Marsaglia, George, and Marsaglia, John, Journal of Statistical Software, 
    Vol. 9, No. 2. (February 2004), pp. 1-5.

    """

    c = _c(n)
    if x<c:
        return (0.0037/n**3+0.00078/n**2+0.00006/n)*_g1(x/c)
    elif x<0.8:
        return (0.04123/n+0.01365/n**2)*_g2((x-c)/(0.8-c))
    else:
        return _g3(x)/n
    

def anderson_darling_fpp(A2, N=None):
    """Compute the false positive probability for the Anderson-Darling statistic.

    Parameters
    ==========
    A2 : positive float
        The value returned by the Anderson-Darling test.
    N : integer or None
        The sample size used; if None, use the asymptotic formula.
        The finite N correction is small.

    Returns
    =======
    p : float between zero and one
        The probability that a value greater than A2 would be returned.
        Note that this routine returns one minus the probability from 
        adinf, which it uses (along with a correction term).

    Reference
    =========
    Marsaglia, George, and Marsaglia, John, Journal of Statistical Software, 
    Vol. 9, No. 2. (February 2004), pp. 1-5.

    """

    if N is None:
        return 1-adinf(A2)
    else:
        a = adinf(A2)
        return 1-(a + errfix(N,a))

def anderson_darling_statistic(data, cdf=lambda x: x):
    """Compute the Anderson-Darling statistic.

    This statistic measures the deviation of the data from
    the distribution given by cdf. In this it resembles
    the Kolmogorov-Smirnov test, but this test is more
    sensitive to deviations in the tails of the distribution.

    Parameters
    ==========
    data : array-like of float
        A one-dimensional array of samples.
    cdf : callable
        The cumulative distribution function of the distribution
        to test against.

    Returns
    =======
    A2 : positive float
        The Anderson-Darling statistic.

    """
    d = cdf(np.array(data))
    d.sort()
    N = len(d)
   
    A2 = -N-np.sum((2*np.arange(N)+1)/float(N)*(np.log(d)+np.log(1-d[::-1])))

    return A2

def anderson_darling(data, cdf=lambda x: x):
    """Compute the Anderson-Darling statistic and significance.

    This statistic measures the deviation of the data from
    the distribution given by cdf. In this it resembles
    the Kolmogorov-Smirnov test, but this test is more
    sensitive to deviations in the tails of the distribution.
    Note that the false positive probability this 
    routine returns assumes that the distribution was known
    a priori. If some parameters in this distribution
    were obtained by fitting, the false positive
    probability returned will be wrong.

    Parameters
    ==========
    data : array-like of float
        A one-dimensional array of samples.
    cdf : callable
        The cumulative distribution function of the distribution
        to test against.

    Returns
    =======
    A2 : positive float
        The Anderson-Darling statistic.
    fpp : float between zero and one
        The probability that a value greater than A2 would be returned.

    Reference
    =========
    Marsaglia, George, and Marsaglia, John, Journal of Statistical Software, 
    Vol. 9, No. 2. (February 2004), pp. 1-5.

    """
    A2 = anderson_darling_statistic(data, cdf)

    return A2, anderson_darling_fpp(A2, len(data))

_ps = np.array([0.25, 0.10, 0.05, 0.025, 0.01])
# not used
_ms = [1,2,3,4,6,8,10,np.inf]
# not used
_tm = [[0.326, 1.225, 1.960, 2.719, 3.752],
       [0.449, 1.309, 1.945, 2.576, 3.414],
       [0.498, 1.324, 1.915, 2.493, 3.246],
       [0.525, 1.329, 1.894, 2.438, 3.139],
       [0.557, 1.332, 1.859, 2.365, 3.005],
       [0.576, 1.330, 1.839, 2.318, 2.920],
       [0.590, 1.329, 1.823, 2.284, 2.862],
       [0.674, 1.282, 1.645, 1.960, 2.326],
       ]
# allow interpolation to get above _tm
_b0s = np.array([0.675, 1.281, 1.645, 1.960, 2.326])
_b1s = np.array([-0.245, 0.250, 0.678, 1.149, 1.822])
_b2s = np.array([-0.105, -0.305, -0.362, -0.391, -0.396])

def anderson_darling_k(samples):
    """Apply the Anderson-Darling k-sample test.

    This test evaluates whether it is plausible that all the samples are drawn
    from the same distribution, based on Scholz and Stephens 1987. The
    statistic computed is their A_kn (rather than A_akn, which differs in
    how it handles ties). The significance of the result is computed by
    producing a scaled and standardized result T_kN, which is returned and
    compared against a list of standard significance levels. The next-larger
    p-value is returned.

    """
    samples = [np.array(sorted(s)) for s in samples]
    all = np.concatenate(samples+[[np.inf]])

    values = np.unique(all)
    L = len(values)-1
    fij = np.zeros((len(samples),L),dtype=np.int)
    H = 0
    for (i,s) in enumerate(samples):
        c, be = np.histogram(s, bins=values, new=True)

        assert np.sum(c)==len(s)

        fij[i,:] = c
        H += 1./len(s)

    ni = np.sum(fij,axis=1)[:,np.newaxis]
    N = np.sum(ni)
    k = len(samples)
    lj = np.sum(fij,axis=0)
    Mij = np.cumsum(fij,axis=1)
    Bj = np.cumsum(lj)

    A2 = np.sum(((1./ni)*lj/float(N)*(N*Mij-ni*Bj)**2/(Bj*(N-Bj)))[:,:-1])

    h = np.sum(1./np.arange(1,N))

    i = np.arange(1,N,dtype=np.float)[:,np.newaxis]
    j = np.arange(1,N,dtype=np.float)
    g = np.sum(np.sum((i<j)/((N-i)*j)))

    a = (4*g-6)*(k-1) + (10-6*g)*H
    b = (2*g-4)*k**2 + 8*h*k + (2*g-14*h-4)*H-8*h+4*g-6
    c = (6*h+2*g-2)*k**2 + (4*h-4*g+6)*k + (2*h-6)*H + 4*h
    d = (2*h+6)*k**2-4*h*k

    sigmaN2 = (a*N**3+b*N**2+c*N+d)/((N-1)*(N-2)*(N-3))

    sigmaN = np.sqrt(sigmaN2)
    
    TkN = (A2 - (k-1))/sigmaN

    tkm1 = _b0s + _b1s/np.sqrt(k-1) + _b2s/(k-1)

    ix = np.searchsorted(tkm1, TkN)
    if ix>0:
        p = _ps[ix-1]
    else:
        p = 1.
    return A2, TkN, (tkm1, _ps.copy()), p


_adnormal_ps = np.array([0.15, 0.1, 0.05, 0.025, 0.01])
_adnormal_case0_thresholds = np.array([1.610, 1.933, 2.492, 3.070, 3.857])
_adnormal_case1_thresholds = np.array([0.784, 0.897, 1.088, 1.281, 1.541])
_adnormal_case2_thresholds = np.array([1.443, 1.761, 2.315, 2.890, 3.682])
_adnormal_case3_thresholds = np.array([0.560, 0.632, 0.751, 0.870, 1.029])

def anderson_darling_normal(data, mu=None, sigma=None):
    """Apply the Anderson-Darling statistic to test whether data is normal.
    
    If both mu and sigma are supplied, the test is a simple Anderson-Darling
    test. If one or both are left as None, then they are estimated from the
    data and the significance is corrected as appropriate. In these cases
    it is only possible to compare the statistic to a finite (short) list
    of tabulated values (from Stephens 1976), so the returned p value will 
    be the next larger standard p value.

    """
    fixed_mu = False
    fixed_sigma = False
    if mu is None:
        mu = np.mean(data)
    else:
        fixed_mu = True

    if sigma is None:
        if fixed_mu:
            ddof = 0
        else:
            # must use this so thresholds are right
            ddof = 1
        sigma = np.std(data,ddof=ddof)
    else:
        fixed_sigma = True

    A2 = anderson_darling_statistic(data, scipy.stats.norm(loc=mu, scale=sigma).cdf)

    if fixed_mu and fixed_sigma:
        p = anderson_darling_fpp(A2, len(data))

        return A2, (mu, sigma), (_adnormal_ps.copy(), _adnormal_case0_thresholds.copy()), p
    else:
        if not fixed_mu and fixed_sigma:
            th = _adnormal_case1_thresholds
        elif fixed_mu and not fixed_sigma:
            th = _adnormal_case2_thresholds
        elif not fixed_mu and not fixed_sigma:
            th = _adnormal_case3_thresholds

        ix = np.searchsorted(th, A2)
        if ix==0:
            p = 1.
        else:
            p = _adnormal_ps[ix-1]

        return A2, (mu, sigma), (_adnormal_ps.copy(), th.copy()), p

_adexp_ps = _adnormal_ps
_adexp_unfixed_thresholds = np.array([0.918, 1.070, 1.326, 1.587, 1.943])

def anderson_darling_exponential(data, theta=None):

    if theta is None:
        theta = 1./np.mean(data)
        fix_theta = False
    else:
        fix_theta = True

    A2 = anderson_darling_statistic(data, lambda d: 1-np.exp(-theta*x))

    if fix_theta:
        th = _adexp_unfixed_thresholds

        ix = np.searchsorted(th, A2)
        if ix==0:
            p = 1.
        else:
            p = _adnormal_ps[ix-1]

        return A2, theta, (_adexp_ps.copy(), th.copy()), p
    else:
        p = anderson_darling_fpp(A2, len(data))

        return A2, theta, (None, None), p

