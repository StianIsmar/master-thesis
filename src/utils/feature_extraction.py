import numpy as np
from scipy.stats import kurtosis, skew
from scipy import signal

# bispectrum
from math import pi
import numpy as np
from numpy.fft import rfftfreq, rfft
from scipy.fftpack import next_fast_len
from scipy.signal import spectrogram
import matplotlib.pyplot as plt


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



def get_freq_domain_features(sig,fs,N=5000):
'''
    Using bispectrum.
'''
# Select the N to split the signal into N parts
kw = dict(nperseg=N // 10, noverlap=N // 20, nfft=next_fast_len(N // 2))
freq1, fre2, bispec = polycoherence.polycoherence(s, fs, norm=None, **kw)
polycoherence.plot_polycoherence(freq1, fre2, bispec)
