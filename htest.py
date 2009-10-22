import numpy as np

def h_fpp(H):
    a = 0.9999755
    b = 0.39802
    c = 1.210597
    d = 0.45901
    e = 0.0022900

    if H<=23:
        return a*np.exp(-b*H)
    elif H<50:
        return c*np.exp(-d*H+e*H**2)
    else:
        return 4e-8
        raise ValueError("H=%g>50 not supported; false positive probability less than 4*10**(-8)" % H)

def h_test(events):
    max_harmonic = 20
    ev = np.reshape(events, (-1,))
    cs = np.sum(np.exp(2.j*np.pi*np.arange(1,max_harmonic+1)*ev[:,None]),axis=0)/len(ev)
    Zm2 = 2*len(ev)*np.cumsum(np.abs(cs)**2)
    Hcand = (Zm2 - 4*np.arange(1,max_harmonic+1) + 4)
    M = np.argmax(Hcand)+1
    H = Hcand[M-1]
    fpp = h_fpp(H)
    return (H, M, fpp)
