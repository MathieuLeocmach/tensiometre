import numpy as np
from scipy.signal import chirp
from scipy.interpolate import interp1d
from scipy.ndimage import binary_opening
from scipy.optimize import curve_fit

def optimal_chirp(t, amplitude=10, f1=1e-2, f2=1, T=66, r=0.1, delay=4):
    """Generates an optimally windowed chirp at time(s) t over a duration T,
    followed by zeros during a delay. The whole signal has a period T+delay.
    The window in an opening cosine during r*T/2, then constant, then a closing
    cosine during r*T/2.
    See Geri et al PRX 2018 https://doi.org/10.1103/PhysRevX.8.041042"""
    td = np.mod(t,T+delay)
    tt = td-delay
    chrp = chirp(tt, f1, T, f2, phi=90, method='logarithmic')
    chrp = np.where(
        tt<0, 0, np.where(
            2*tt < r*T,
            np.cos(np.pi/r*(tt/T-r/2))**2 * chrp,
            np.where(
                tt/T > 1 - r/2,
                np.cos(np.pi/r*(tt/T-1+r/2))**2 * chrp,
                chrp
            )
        )
    )
    return amplitude*chrp

def read_data(outname):
    """Read data from binary, expecting 6 components:
    t, x_arm2ground, y_arm2ground (micromanipulator), x_head2arm, y_head2arm, y_arm2tank (3rd sensor)"""
    data = np.fromfile(outname)
    return data.reshape((len(data)//6,6))

def interpolate2(data, T=66):
    """Interpolate a signal that was sampled irregularly so that the samples are
    regularly spaced and so that there is a power of two samples per T larger
    than the original number of samples. Uses only the 0th component of the
    signal supposed to contain sampling times."""
    dtm = (data[-1,0]-data[0,0])/len(data)
    delta = T/dtm
    subsampling = (2**np.ceil(np.log2(delta)))/delta
    t = np.arange(int(len(data)*subsampling)) * dtm / subsampling + data[0,0]
    return interp1d(data[:,0], data, axis=0)(t)

def extract_chips(data2, T=66, delay=4):
    """Select each chirp in the signal, discarding the delay, using only the two
    first components supposed to be t and x_arm2ground.
    Returns an array of N signals of duration T, with the same components as data2."""
    t, x = data2.T[:2]
    dt = t[1]-t[0]
    delta = int(T/dt)
    #intervals where the exitation is not changing during delay
    er = binary_opening(np.abs(np.diff(x))<1/128, np.ones(int(delay//dt), bool))
    descending = np.where(er[:-1] & ~er[1:])[0]
    if descending[-1]+delta > len(data2):
        descending = descending[:-1]
    chirps = np.zeros((len(descending), delta, data2.shape[1]))
    for i, n in enumerate(descending):
        chirps[i] = data2[n:n+delta]
    return chirps

def chirp2moduli(chirps, T=66):
    """Returns an array of N complex moduli given N signals of duration T (in seconds).
    Signals are expected to have at least 4 components with
    x_arm2ground at index 1
    x_head2arm at index 3
    """
    fourier = np.fft.rfft(chirps, axis=1)
    Gs = -fourier[...,3] / (fourier[...,1]+fourier[...,3])
    freqs = np.arange(chirps.shape[1]//2+1)/T
    return freqs, Gs

def load_chirp_moduli(filename, T, delay):
    """Read data from binary, split into chirps of parameters T and delay,
    extract frequency dependence of complex moduli."""
    return chirp2moduli(
        extract_chips(
            interpolate2(
                read_data(filename),
                T=T),
            T=T, delay=delay),
        T=T)

def fit_powerlaw_modulus(freqs, G, M=45):
    """Fit the frequency dependence of the real part of modulus G (storage) with a power law.
    Returns exponent and modulus at 1 Hz."""
    func = lambda x, a, b: a*x+b
    good = G.real>0
    good[0] = False
    good[M:] = False
    alpha, beta = curve_fit(func, np.log(freqs[good]), np.log(G.real[good]), [0.15,1])[0]
    return alpha, np.exp(beta)
