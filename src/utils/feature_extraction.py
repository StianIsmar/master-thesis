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
import sys, os,os.path
from pathlib import Path
from tqdm.notebook import tqdm
import glob # To count files in folder


#f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/time_features.csv'

def calc_kurtosis(sig):
    return kurtosis(sig)

def calculate_time_domain_features(sig,fs):
    r1 = sum(map(lambda x: abs(x), sig)) 
    
    # 1. rms
    r = sum(map(lambda x: x**2,sig))
    rms = (np.sqrt(2*r))/len(sig)

    # 2. kurt
    kurt = calc_kurtosis(sig)

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
    signal_energy = sum([s**2 for s in sig])
    
    # 12. Energy entropy
    entropy = -1*sum(map(lambda x: x*np.log(x),sig)) # NAN?!?!?!

    features = [rms, kurt, skewness, peak_to_peak,crest_factor,shape_factor,impulse_factor,margin_factor,signal_mean,std,signal_energy, entropy]
    return features


def build_time_feature_df(wt_number,signals,fs):
    path=f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/time_features.csv'
    if os.path.exists(path):
        print("Time domain features exist and are loaded.")
        return pd.read_csv(path)

    all_time_features = []
    print('time-domain...')
    for sig in tqdm(signals):
        signal_data = []
        time_signal_features=calculate_time_domain_features(sig,fs)
        all_time_features.append(time_signal_features)
    
    df = pd.DataFrame(all_time_features, columns=[
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

    # save the df
    df.to_csv(path)
    return df


def calculate_bi_spectrum_features(sig,fs,N,plot=False):
# N is the number of parts we want to split the signal into.
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
    # features.append(normalized_entropy_squared(half_bispec_1d)) # Needed to remove because of too small values.
    from scipy import ndimage
    woc=ndimage.measurements.center_of_mass(bispec_half)
    features.append(woc[0])
    features.append(woc[1])
    #features.append(weighted_sum(bispec_half,0))
    #features.append(weighted_sum(bispec_half,1))

    return features

def build_frequency_features_df(wt_number,signals,fs,N=20000,plot=True):
    # Using bispectrum.

    # check if it exists already
    if Path(f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/freq_features.csv').is_file():
        print("Frequency features exist")
        # load it
        feature_df = pd.read_csv(f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/freq_features.csv')
        return feature_df
    else:
        # feature_df, names = calculate_bi_spectrum_features(signals,fs)
        save_path_freq = f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/freq_features.csv'
        all_signals_freq_features=[]
        for sig in tqdm(signals):
            signal_features = calculate_bi_spectrum_features(sig[0:3000],fs,N,False)
            all_signals_freq_features.append(signal_features)
        feature_headings=['B1','B2','B3','B4','B5','B6']
        feature_names = ['Bispectrum Log Amplitude',
        'Bi-spectrum Log Amplitude Diagonal',
        'Bi-spectrum First order Spectral Moment Diagonal',
        'Bi-spectrum Normalized Entropy',
        'Bi-spectrum first axis weighted center',
        'Bi-spectrum second axis weighted center']
        df = pd.DataFrame(data = all_signals_freq_features, columns=feature_headings)
        df.to_csv(save_path_freq) # save till next time


    # 'Normalized bi-spectral squared entropy'
    return df

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
    
    qn = ((np.abs(bispectrum))**2) / (np.sum((np.abs(bispectrum))**2))
    p2 = -1*(np.sum(qn*np.log(qn)))
    return (p2)

# 6. Bi-spectrum phase entropy
#def phase_entropy(bispectrum):
    #continue


# 6. Weighted center of bi-spectrum (first axe)
def weighted_sum(bispectrum,axis=0):
    sum_arr_teller=[]
    sum_arr_nevner=[]
    if axis==0:
        for j in range(bispectrum.shape[1]): # cols
            for i in range(bispectrum.shape[0]): # rows
                val=bispectrum[i][j]
                sum_arr_teller.append(i*val)
                sum_arr_nevner.append(val)
        wcomb = np.sum(sum_arr_teller) / np.sum(sum_arr_nevner)

    if axis==1:
        for j in range(bispectrum.shape[1]): # cols
            for i in range(bispectrum.shape[0]): # rows
                val=bispectrum[i][j]
                sum_arr_teller.append(j*val) # multiplying with j instead!
                sum_arr_nevner.append(val)
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

    import time
    file_path=f'/Volumes/OsvikExtra/signal_data/raw_filtered_6000Hz/gearbox/wt0{wt_number}/dataset.csv'
    if os.path.isfile(file_path):
        for i in tqdm(range(1)):
            df = pd.read_csv(file_path)
        print('All features already exist. They are returned.')
        return df

    df_freq_features = build_frequency_features_df(wt_number,final_signals,fs) # freq features
    df_time_features= build_time_feature_df(wt_number,final_signals,fs) # time features
    df_time_frequency_features = get_time_frequency_features(wt_number) # time frequency features from EEMD signals.

    print('features made. Filtering...')
    #Power spectrum
    # psd_df = get_psd(final_signals, fs) # Power spectral entropy

    # Merging into one df
    data = pd.concat([op_data_intervals,df_time_features,df_freq_features,df_time_frequency_features],\
        axis=1,sort=False) # concatinate all
    data = data.drop(['Unnamed: 0'],axis=1)
    newCol = np.arange(0,data.shape[0],1)
    newCol = pd.DataFrame(newCol,columns=['Index'])
    data = pd.concat([newCol,data],axis=1,sort=False) # concatinate all

    # Filtering on 1450 RPM
    res = data[data['AvgSpeed'] >= 1450]

    print("Filtering completed.")

    # The weighted centers from complex form to float
    res['B5'] = res['B5'].apply(lambda x: abs(complex(x))) 
    res['B6'] = res['B6'].apply(lambda x: abs(complex(x)))
    
    # Saving features to file
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



def get_time_frequency_features(turbine_number,input_file_type='zip'):
    file_path = f'/Volumes/OsvikExtra/signal_data/features/wt0{turbine_number}/time_freq_features.csv' 
    if os.path.isfile(file_path):
        print('Time-freq features exists.')
        time_freq_features = pd.read_csv(file_path)
        time_freq_features=time_freq_features.drop(['Unnamed: 0'],axis=1)
        print('Loaded csv.')

    else:
        raw_imfs_path=f'/Volumes/OsvikExtra/signal_data/raw_data/gearbox/wt0{turbine_number}/eemds/'
        energy_rates,energy_entropies,eemd_kurtosis = get_imf_features(turbine_number,raw_imfs_path,input_file_type)


        # save the reates and entropies:
        df1 = pd.DataFrame(energy_rates,columns=['imf_rate_1','imf_rate_2','imf_rate_3','imf_rate_4','imf_rate_5'])
        df2 = pd.DataFrame(energy_entropies,columns=['imf_entropy_1','imf_entropy_2','imf_entropy_3','imf_entropy_4','imf_entropy_5'])
        df3 = pd.DataFrame(eemd_kurtosis,columns=['imf_kurtosis_1','imf_kurtosis_2','imf_kurtosis_3','imf_kurtosis_4','imf_kurtosis_5'])
        time_freq_features = pd.concat([df1,df2,df3],axis=1)
        time_freq_features.to_csv(file_path)
    return time_freq_features

def get_imf_features(turbine_number,path_folder,input_file_type):
    result_features=[]
    
    def calculate_imf_energy(imf):
        imf_energy=0
        for x in imf:
            imf_energy+=(x**2)
        return imf_energy

    file_count = len(glob.glob1(path_folder,"*.zip"))
    print("file count:", file_count)
    
    energy_rates_intervals=[]
    energy_entropies_intervals=[]
    eemd_kurtosis = [] # kurtosis for all the intervals
    #range(file_count)
    for i in tqdm(range(file_count)): # looping through the intervals
        total_entropy=0

        energy_five_imfs=0 # per interval for the 5 imfs.
        imf_energy_individual={1:0,2:0,3:0,4:0,5:0}
        
        if input_file_type == 'csv':
            path=path_folder + f'interval_number_{i}.csv'
            df = pd.read_csv(path, header=None)
        if input_file_type == 'zip':
            path=path_folder + f'raw_wt0{turbine_number}_interval_number_{i}.zip'
            df = pd.read_csv(path,compression='zip')
        # do something with each IMF.. get features?
        
        if (df.shape[0]) < 6:
            print('There are not enought IMFs...')
            break
                    
        # looping through every IMF:
        imf_energy=0
        imf_kurtosis=[]
        for imf_index in range(6):
            '''
                Looping through the first 5 IMFs in the interval.
                We are only interested in the first 5 IMFs.
                The first one is the original input signal.
            '''
            if imf_index == 0:
                # this is the original signal, which we are not working with
                continue
                
            imf_energy=calculate_imf_energy(df.iloc[imf_index,:]) # (1) calc energy for imf 
            imf_energy_individual[imf_index]=imf_energy
            energy_five_imfs+=imf_energy # (2) updating energy for the first 5 imfs.

            imf_energy=calculate_imf_energy(df.iloc[imf_index,:]) # (1) calc energy for imf 
            imf_kurtosis.append(calc_kurtosis(np.asarray(df.iloc[imf_index,:]))) # Calculate kurtosis for each of the 5 imf's in the signal.
        # after the for-loop:
        energy_rates=[]
        energy_entropy=[] # energyEntropy1,...,energyEntropy5
        for k,imf_energy_value in imf_energy_individual.items():
            p=imf_energy_value/((abs(energy_five_imfs)))
            energy_rates.append(p)
            imf_energy_entropy = -p*np.log(p)
            energy_entropy.append(imf_energy_entropy) # the energy entropy for one interval []
        energy_rates_intervals.append(energy_rates) # energy_rates:[E1/E,E2/E,...,E5/E]            
        energy_entropies_intervals.append(energy_entropy)
        eemd_kurtosis.append(imf_kurtosis)
    return energy_rates_intervals,energy_entropies_intervals,eemd_kurtosis
