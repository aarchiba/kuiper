import numpy as np

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


def anderson_darling(data, cdf=lambda x: x):
    """Compute the Anderson-Darling statistic.

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
    d = cdf(np.array(data))
    d.sort()
    N = len(d)
   
    A2 = -N-np.sum((2*np.arange(N)+1)/float(N)*(np.log(d)+np.log(1-d[::-1])))

    return A2, anderson_darling_fpp(A2, N)
