import numpy as np
from scipy.stats import kurtosis, skew
from scipy import signal

def get_time_domain_features(sig,fs):
    # rms
    r = sum(map(lambda x: x**2,sig))
    rms = np.sqrt(2*r)

    # kurt
    kurt = kurtosis(sig)

    # skewness
    skewness = skew(sig)

    # signal energy
    signal_energy = signal.welch(sig,fs)
    energy_mean = np.mean(signal_energy)

    # normal mean
    signal_mean = np.mean(sig)
    
    return rms, kurt, skewness, energy_mean, signal_mean