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
import polycoherence
import pandas as pd


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



def get_freq_domain_features(sig,fs,N=20000,plot=True):

    #Using bispectrum.
    #N is the number of parts we want to split the signal into.
	kw = dict(nperseg=N // 10, noverlap=N // 20, nfft=next_fast_len(N // 2))
	freq1, fre2, bispec = polycoherence.polycoherence(sig, fs, norm=None, **kw)
	
	bispec_half = np.triu(bispec) # remove symmetric part

	if plot:
		polycoherence.plot_polycoherence(freq1, fre2, bispec)
		plt.show()
		print(bispec.shape)

		bispec_rot = np.triu(bispec) # remove symmetric part
		polycoherence.plot_polycoherence(freq1, fre2, bispec_rot)
		plt.show() # plot half the bispectrum

	features = []
	features.append(bispec_log_amp(bispec))
	features.append(bispec_log_amp_diag(bispec))
	features.append(first_order_spectral_moment_diag(bispec))
	features.append(normalized_entropy(bispec))
	# features.append(normalized_entropy_squared(bispec)) # Needed to remove because of too small values.
	features.append(weighted_sum(bispec))
	feature_headings=['B1', 'B2', 'B3','B4','B5']
	feature_names = ['Bispectrum Log Amplitude','Bispectrum Log Amplitude Diagonal',
	'Bispectrum First oreder Spectral Moment Diagonal', 'Bispectrum Normalized Entropy','Bispectrum Weigheted Sum']
	return features,feature_headings, feature_names

# 1. Sum of logarithmic amplitudes of the bi-spectrum, OK 
def bispec_log_amp(bispectrum):
    return np.sum(np.log(np.abs(bispectrum)))/2 # Symmetry


# 2. Sum of logarithmic amplitudes of diagonal elements in the bi-spectrum, OK
def bispec_log_amp_diag(bispectrum):
    return np.sum(np.log(np.diag(np.abs(bispectrum))))

# 3. First order spectral moment of amplitudes of diagonal elements in the bi spectrum, OK
def first_order_spectral_moment_diag(bispectrum):
    df = pd.DataFrame(data=bispectrum)
    f_3 = 0
    # bispec_rot = np.rot90(bispec,k=3)
    for i in range(bispectrum.shape[0]):
        f_3 += i*np.log(np.abs((bispectrum[i,i])))
    return f_3

# 4. Normalized bi-spectral entropy, calculate for the whole bispectrum. Same every time so it won't matter. 
def normalized_entropy(bispectrum):
    pn = np.abs(bispectrum)/(np.sum(np.abs(bispectrum)))
    p1 = -1*(np.sum(pn*np.log(pn)))
    return p1

# 5. Normalized bi-specral squared entropy
def normalized_entropy_squared(bispectrum):
    qn = (np.abs(bispectrum)**2) / (np.sum(np.abs(bispectrum)**2))
    p2 = -1*(np.sum(qn*np.log(qn)))
    return (p2)

# 6. Weighted center of bi-spectrum
def weighted_sum(bispectrum):
    sum_arr_teller = []
    sum_arr_nevner = []
    for j in range(bispectrum.shape[1]): # cols
        for i in range(bispectrum.shape[0]): # rows
            sum_arr_teller.append(i*bispectrum[i][j])
            sum_arr_nevner.append(bispectrum[i][j])
    wcomb = np.sum(sum_arr_teller) / np.sum(sum_arr_nevner)
    return wcomb