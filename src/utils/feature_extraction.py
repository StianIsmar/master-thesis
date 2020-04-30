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
    r1 = sum(map(lambda x: abs(x), sig)) 
    
    # 1. rms
    r = sum(map(lambda x: x**2,sig))
    rms = (np.sqrt(2*r))/len(sig)

    # 2. kurt
    kurt = kurtosis(sig)

    # 3. skewness
    skewness = skew(sig)
 
    # 4. Peak to peak
    peak_to_peak=np.max(sig)-np.min(sig)

    # 5. Crest factor
    crest_factor = np.max(sig)/rms

    # 6. Shape factor
    shape_factor = rms/(r1/len(sig))

    # 7. Impulse factor
    impulse_factor = np.max(sig)/(r1/len(sig))

    # 8. Margin factor
    margin_factor=np.max(sig)/(r1/len(sig))**2
    
    # 9. Mean
    signal_mean = np.mean(sig)
   
    # 10. Standard deviation
    std = np.std(sig)

    # 11. signal energy
    signal_energy = signal.welch(sig,fs)
    
    # 12. Energy entropy
    entropy = -1*sum(map(lambda x: x*np.log(x)))

    return rms, kurt, skewness, peak_to_peak,crest_factor,shape_factor,impulse_factor,margin_factor,signal_mean,std,signal_energy, entropy


def build_time_df(all_vibrations,fs):
    data = []
    for j, sig in enumerate(all_vibrations):
    	signal_data = []
    	rms, kurt, skewness, peak_to_peak,crest_factor,shape_factor,impulse_factor,margin_factor,signal_mean,std,signal_energy, entropy=get_time_domain_features(sig,fs)
    	signal_data.append(rms)
    	signal_data.append(kurt)
    	signal_data.append(skewness)
    	signal_data.append(peak_to_peak)
    	signal_data.append(crest_factor)
    	signal_data.append(shape_factor)
    	signal_data.append(impulse_factor)
    	signal_data.append(margin_factor)
    	signal_data.append(signal_mean)
    	signal_data.append(std)
    	signal_data.append(signal_energy)
    	signal_data.append(entropy)
    	data.append(signal_data)

    df = pd.DataFrame(data, columns=[
        'rms', 
        'kurt', 
        'skewness', 
        'peak_to_peak', 
        'crest_factor',
		'shape_factor',
		'impulse_factor',
		'margin_factor',
		'signal_mean',
		'std',
		'signal_energy',
		'entropy'
        ]
                     )
    return df


def create_frequency_df_for_all_signals(signals,fs):
	'''
		Since the signal is non stationary, and bi-spectrum analysis requires statioary signals,
		only 0:3000 of the plot is selected.
	'''
	all_feature_values =[]
	for sig in signals:
		features, feature_headings, feature_names = get_freq_domain_features(sig[0:3000],fs,15000,False)
		all_feature_values.append(features)

	df = pd.DataFrame(data = all_feature_values, columns=feature_headings)
	return df, feature_names

def get_freq_domain_features(sig,fs,N=20000,plot=True):

	# modify to take in all signals
    #Using bispectrum.
    #N is the number of parts we want to split the signal into.
	kw = dict(nperseg=N // 10, noverlap=N // 20, nfft=next_fast_len(N // 2))
	freq1, fre2, bispec = polycoherence.polycoherence(sig, fs, norm=None, **kw)
	
	half_bispec_1d=list(bispec[np.triu_indices_from(bispec)])
	bispec_half = np.triu(bispec) # remove symmetric part

	
	if plot:
		polycoherence.plot_polycoherence(freq1, fre2, bispec)
		plt.show()
		print(bispec.shape)

		bispec_rot = np.triu(bispec) # remove symmetric part
		polycoherence.plot_polycoherence(freq1, fre2, bispec_rot)
		plt.show() # plot half the bispectrum



	features = []
	features.append(bispec_log_amp(half_bispec_1d))
	features.append(bispec_log_amp_diag(bispec))
	features.append(first_order_spectral_moment_diag(bispec))
	features.append(normalized_entropy(half_bispec_1d))
	features.append(normalized_entropy_squared(half_bispec_1d)) # Needed to remove because of too small values.
	features.append(weighted_sum(bispec_half,0))
	features.append(weighted_sum(bispec_half,1))
	feature_headings=['B1','B2','B3','B4','B5','B6','B7']
	feature_names = ['Bispectrum Log Amplitude',
	'Bi-spectrum Log Amplitude Diagonal',
	'Bi-spectrum First order Spectral Moment Diagonal',
	'Bi-spectrum Normalized Entropy',
	'Normalized bi-spectral squared entropy',
	'Bi-spectrum first axis weighted center',
	'Bi-spectrum second axis weighted center']
	return features, feature_headings, feature_names

# 1. Sum of logarithmic amplitudes of the bi-spectrum, OK 
def bispec_log_amp(bispectrum):
    return np.sum(np.log(np.abs(bispectrum))) # Symmetry

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
    qn = (np.(abs(bispectrum))**2) / (np.sum((np.abs(bispectrum))**2))
    p2 = -1*(np.sum(qn*np.log(qn)))
    return (p2)

# 6. Bi-spectrum phase entropy
#def phase_entropy(bispectrum):
	#continue


# 6. Weighted center of bi-spectrum (first axe)
def weighted_sum(bispectrum,axis=0):
    sum_arr_teller = []
    sum_arr_nevner = []

    if axis==0:
	    for j in range(bispectrum.shape[1]): # cols
	        for i in range(bispectrum.shape[0]): # rows
	            sum_arr_teller.append(i*bispectrum[i][j])
	            sum_arr_nevner.append(bispectrum[i][j])
	    wcomb = np.sum(sum_arr_teller) / np.sum(sum_arr_nevner)
    
    if axis==1:
	    for j in range(bispectrum.shape[1]): # cols
        	for i in range(bispectrum.shape[0]): # rows
        	    sum_arr_teller.append(j*bispectrum[i][j]) # multiplying with j instead!
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
	#if Path(f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/freq_features.csv').is_file():
	    #print("Frequency features exist")
	    #feature_df = pd.read_csv(f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/freq_features.csv')
	#else:
	feature_df, names = create_frequency_df_for_all_signals(final_signals,fs)
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
