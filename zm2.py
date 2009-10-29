import numpy as np
import scipy.stats

def Zm2(events, M):
    """The Z_m^2 test for periodicity.

    The Z_m^2 test uses a sequence of events to construct an estimate of
    the first M Fourier coefficients of the distribution the events are drawn
    from. It then adds up the power in these coefficients and evaluates the
    likelihood that events drawn from a uniform distribution would have this
    much harmonic power.

    Parameters
    ----------

    events : array-like
        Events to be tested for uniformity modulo 1.
    M : integer
        Number of harmonics to use.

    Returns
    -------

    S : float
        The score.
    fpp : float
        The chance of random events from a uniform distribution having a
        score this high.

    """
    ev = np.reshape(events, (-1,))
    cs = np.sum(np.exp(2.j*np.pi*np.arange(1,M+1)*ev[:,None]),axis=0)/len(ev)
    Zm2 = 2*len(ev)*np.sum(np.abs(cs)**2)
    return Zm2, scipy.stats.chi2(2*M).sf(Zm2)
