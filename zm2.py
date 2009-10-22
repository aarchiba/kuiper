import numpy as np
import scipy.stats

def Zm2(events, M):
    ev = np.reshape(events, (-1,))
    cs = np.sum(np.exp(2.j*np.pi*np.arange(1,M+1)*ev[:,None]),axis=0)/len(ev)
    Zm2 = 2*len(ev)*np.sum(np.abs(cs)**2)
    return Zm2, scipy.stats.chi2(2*M).sf(Zm2)
