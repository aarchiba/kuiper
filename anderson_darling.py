import numpy as np

_adpoly1 = np.poly1d([0.00168691, -0.0116720, 0.0347962, -0.0649821, 0.247105, 2.00012])
_adpoly2 = np.poly1d([-0.0003146, 0.008056, -0.082433, 0.43424, -2.30695, 1.0776])
def adinf(z):
    """Compute the asymptotic Anderson-Darling CDF.

    This is an approximate calculation, with error less than 2e-6 for
    A2<2 and error less than 8e-7 for A2>2.

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
    """Compute Anderson-Darling correction term for finite N.

    This is an approximate calculation that should give a result 
    good to four decimal places.

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

    """

    if N is None:
        return 1-adinf(A2)
    else:
        a = adinf(A2)
        return 1-(a + errfix(N,a))


def anderson_darling(data, cdf=lambda x: x):
    """Compute the Anderson-Darling statistic.

    """
    d = cdf(np.array(data))
    d.sort()
    N = len(d)
   
    A2 = -N-np.sum((2*np.arange(N)+1)/float(N)*(np.log(d)+np.log(1-d[::-1])))

    return A2, anderson_darling_fpp(A2, N)
