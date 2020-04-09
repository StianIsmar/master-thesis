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


def build_time_df(all_vibrations,fs):
    data = []
    for j, sig in enumerate(all_vibrations):
        signal_data = []
        rms, kurt, skewness, energy_mean, signal_mean=get_time_domain_features(sig,fs)
        signal_data.append(rms)
        signal_data.append(kurt)
        signal_data.append(skewness)
        signal_data.append(energy_mean)
        signal_data.append(signal_mean)
        

        data.append(signal_data)

    df = pd.DataFrame(data, columns=[
        'rms', 
        'kurt', 
        'skewness', 
        'energy_mean', 
        'signal_mean',
        ]
                     )
    return df


def create_df_for_all_signals(signals,fs):
	all_feature_values =[]
	for sig in signals:
		features, feature_headings, feature_names = get_freq_domain_features(sig,fs,15000,False)
		all_feature_values.append(features)

	df = pd.DataFrame(data = all_feature_values, columns=feature_headings)
	return df, feature_names

def get_freq_domain_features(sig,fs,N=20000,plot=True):

	# modify to take in all signals



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


from sklearn.preprocessing import MinMaxScaler
from entropy import spectral_entropy

def get_psd(signals,fs,plot=False):
	psds = []
	for sig in signals:
		se = spectral_entropy(sig, sf=fs, method='welch', normalize=True)
		psds.append(se)
	psds_df = pd.DataFrame(data=psds,columns=['Power spectral entropy'])
	return psds_df
    
#a=(get_psd([1,24,2,5,1.2,5.21,6,6],1))
#print(a)


def generate_feature_df(wt_number, fs, op_data_intervals, final_signals):
	'''
		Takes in signal that have been rectified and lowpassed (after eemd-step) and then:
			1. filter samples on 1450 rpm
			2. calc time, freq and power spectrum features.
	'''

	from pathlib import Path


	# Frequency features:
	if Path(f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/freq_features.csv').is_file():
	    print("Frequency features exist")
	    feature_df = pd.read_csv(f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/freq_features.csv')
	else:
	    feature_df, names = create_df_for_all_signals(final_signals,fs)
	    save_path_freq = f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/freq_features.csv'
	    feature_df.to_csv(save_path_freq) # save till next time


	# Time features
	time = build_time_df(final_signals,fs)

	#Power spectrum
	psd_df = get_psd(final_signals, fs) # Power spectral entropy

	# Merging into one df
	data = pd.concat([op_data_intervals,time,psd_df,feature_df],axis=1,sort=False) # concatinate all
	data = data.drop(['Unnamed: 0'],axis=1)
	newCol = np.arange(0,data.shape[0],1)
	newCol = pd.DataFrame(newCol,columns=['Index'])
	data = pd.concat([newCol,data],axis=1,sort=False) # concatinate all

	# Filtering on 1450 RPM
	res = data[data['AvgSpeed'] >= 1450]

	print("Filtering done.")
	res['B5'] = res['B5'].apply(lambda x: abs(complex(x)))
	
	# Saving to file
	print("Saving all features to file: "+ f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/dataset.csv')
	res.to_csv(f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/dataset.csv')
	return res

	# Count number of zeros in full bispectrum:
def num_of_zeros(array):
    sh = (array.shape)
    non = np.count_nonzero(array==0)
    print("non",non)
    zero_elems = sh[0]*sh[1] - non
    print(zero_elems)


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


# def scale_df():