import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import sys, os
from numpy import savez_compressed


ROOT_PATH = os.path.abspath("..").split("data_processing")[0]
print("ROOT", ROOT_PATH)
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/utils/"))
if module_path not in sys.path:
    print("appended")
    sys.path.append(module_path)

import functions as f

module_path = os.path.abspath(os.path.join(ROOT_PATH+"/data_processing/"))
if module_path not in sys.path:
    print("appended")
    sys.path.append(module_path)


import wt_data
import ff_transform
import explore_fft_time_data

# Plot how the vibration looks across all intervals
def plot_vib_consecutive(sig):
    indexes = []
    index_start=0
    fig, ax = plt.subplots(figsize=(15,5))
    for i, d in enumerate(sig):
        indexes.append(index_start)
        index_end = len(d)
        x = np.arange(index_start, index_start+index_end)
        index_start = len(x) + index_start
        plt.plot(x,d)
        plt.margins(0)
    print(len(np.linspace(0,index_start+index_end,5)))
    plt.xticks(np.linspace(0,index_start,24),[int(round(elem)) for i,elem in enumerate(np.linspace(1,len(sig)+1,24))])
    #locs, labels = plt.xticks()
    #ax.set_xticks(indexes)
    #ax.set_xticklabels(np.arange(1,len(indexes)+1))
    # plt.xticks(fontsize=7, rotation=90)
    print((indexes[-1]))
    plt.title("Gearbox HSS")
    plt.xlabel("Interval number")
    plt.ylabel("Vib Amplitude")
    plt.show()


# Bandpass method (Bandpass + RECT + LP)

def get_sampling_freq(signal,times):
    times = np.array(times)
    T = times[0] - times[1]
    ending_time = np.array(times)[-1]
    N = len(signal)
    fs = N/ending_time
    return fs, N, T, ending_time

def perform_fft(amplitudes, timestamps,plot=False ):
    t = timestamps
    sig = amplitudes
    mean_amplitude = np.mean(sig)
    sig = sig - mean_amplitude  # Centering around 0
    fft = np.fft.fft(sig)
    N = sig.size
    T = t[1] - t[0]

    f = np.linspace(0, 1 / T, N, )  # start, stop, number of. 1 / T = frequency is the biggest freq

    f = f[:N // 2]
    y = np.abs(fft)[:N // 2]
    y_norm = np.abs(fft)[:N // 2] * 1 / N  # Normalized
    fft_modulus_norm = y_norm
    fft_obj = {'freq':f,'fft_norm': fft_modulus_norm}
    return fft_obj


 # Butter bandpass

from scipy.signal import butter, lfilter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data,index=0)
    return y

def perform_envelope_process(wt_name,timestamps, interval_signal, COMPONENT_NAME, lowcut,highcut,order = 5,plot_low=60,plot_high=800,plot=False,square=False,lowcut_final_lp=10000):
    fs, _, _ ,_ =  get_sampling_freq(interval_signal,timestamps)
    bandpass_filtered = butter_bandpass_filter(interval_signal,lowcut, highcut, fs, order=5)
    bandpass_filtered_rect = abs(bandpass_filtered)
    if square:
        bandpass_filtered_rect = bandpass_filtered_rect**2
    
    # Adding lowpass filter
    bandpass_filtered_rect_lp = butter_lp_filter(bandpass_filtered_rect, lowcut_final_lp, fs, order=5)
    
    all_time_signals = [interval_signal,bandpass_filtered,bandpass_filtered_rect, bandpass_filtered_rect_lp]
    all_fft_spectrums = [perform_fft(sig,timestamps) for i, sig in enumerate(all_time_signals)]

    if plot:
        plot_enveloping_process(wt_name,4,2,COMPONENT_NAME, all_time_signals,all_fft_spectrums, timestamps,lowcut,highcut)

        df = pd.DataFrame(interval_signal)
        df['timestamp'] = timestamps
        # df.plot.hist(bins=2000)
        df.iloc[plot_low:plot_high].plot.line(x='timestamp',y=0,title="Zoomed in on raw signal: Index 60 to 800",legend=False,figsize=(10,2))
        
        df = pd.DataFrame(bandpass_filtered)
        df['timestamp'] = timestamps
        # df.plot.hist(bins=2000)
        df.iloc[plot_low:plot_high].plot.line(x='timestamp',y=0,title="Zoomed in after bandpass: Index 60 to 800 ",legend=False,figsize=(10,2))

        df = pd.DataFrame(bandpass_filtered_rect)
        df['timestamp'] = timestamps
        # df.plot.hist(bins=2000)
        df.iloc[plot_low:plot_high].plot.line(x='timestamp',y=0,title=f"Zoomed in after rectification: {plot_low} to {plot_high}",legend=False,figsize=(10,2))

        df = pd.DataFrame(bandpass_filtered_rect_lp)
        df['timestamp'] = timestamps
        # df.plot.hist(bins=2000)
        df.iloc[plot_low:plot_high].plot.line(x='timestamp',y=0,title=f"Zoomed in after lowpass: Index {plot_low} to {plot_high}",legend=False,figsize=(10,2))
        
        df = pd.DataFrame(all_fft_spectrums[-1]['fft_norm'])
        df['frequency'] = all_fft_spectrums[-1]['freq']
        # df.plot.hist(bins=2000)
        ax=df.plot.line(x='frequency',y=0,title=f"Zoomed in after lowpass: Index {plot_low} to {plot_high}",legend=False,figsize=(10,2))
        ax.set_xlim(0,lowcut_final_lp)
        ax.set_xlim(0,1200)
        plt.show()
    return all_time_signals, all_fft_spectrums



def plot_enveloping_process(wt_name, rows, cols, COMPONENT_NAME, all_time_signals, all_fft_spectrums, timestamps, lowcut, highcut):
    fig, ax = plt.subplots(rows, cols,figsize=(15,7))
    fig.suptitle(f'{COMPONENT_NAME}: Envelope signal processing for {wt_name} ',fontsize=16, y=1.05)

    # Row 1
    ax[0,0].plot(timestamps,all_time_signals[0])
    ax[0,0].set_title(f"Raw vibration signal from {COMPONENT_NAME} ")
    ax[0,0].set_xlabel("Timestamp [s]")

    ax[0,1].plot(all_fft_spectrums[0]['freq'],all_fft_spectrums[0]['fft_norm'])
    ax[0,1].set_xlabel("Frequency [Hz]")
    ax[0,1].set_title(f"FFT of raw signal from {COMPONENT_NAME}")

    # Row 2
    ax[1,0].plot(timestamps,all_time_signals[1])
    ax[1,0].set_xlabel("Timestamp [s]")
    ax[1,0].set_title(f'filtered signal with lower cutoff {lowcut} and higher {highcut} Hz')
    ax[1,1].plot(all_fft_spectrums[1]['freq'],all_fft_spectrums[1]['fft_norm'])
    ax[1,1].set_xlabel("Frequency [Hz]")
    ax[1,1].set_xlim(lowcut-500,highcut+500)
    
    # Row 3
    ax[2,0].plot(timestamps,all_time_signals[2])
    ax[2,0].set_xlabel("Timestamp [s]")
    ax[2,0].set_title(f'Rectified the bandpass-filtered signal')
    ax[2,1].plot(all_fft_spectrums[2]['freq'],all_fft_spectrums[2]['fft_norm'])
    ax[2,1].set_xlabel("Frequency [Hz]")
    # ax[2,1].set_xlim(0,1000)    
    plt.margins(0)

    # Row 4
    ax[3,0].plot(timestamps,all_time_signals[3])
    ax[3,0].set_xlabel("Timestamp [s]")
    ax[3,0].set_title(f'With final lowpass filter added')
    ax[3,1].plot(all_fft_spectrums[3]['freq'],all_fft_spectrums[3]['fft_norm'])
    ax[3,1].set_xlabel("Frequency [Hz]")
    # ax[2,1].set_xlim(0,1000)    
    
    plt.tight_layout()
    plt.margins(0)
    plt.show()



# hp
from scipy.signal import butter, lfilter,filtfilt

def butter_hp(lowcut, fs, order=5):
    nyq = 0.5 * fs
    w = lowcut / nyq
    b, a = butter(order, w, btype='highpass')
    return b, a

def butter_hp_filter(data, lowcut, fs, order=5):
    b, a = butter_hp(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


#lp
def butter_lp(highcut, fs, order=5):
    nyq = 0.5 * fs
    w = highcut / nyq
    b, a = butter(order, w, btype='lowpass')
    return b, a

def butter_lp_filter(data, highcut, fs, order=5):
    b, a = butter_lp(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y