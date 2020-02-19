from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wt_data


# import external modules
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from scipy.stats import skew
from scipy.stats import kurtosis


ROOT_PATH = os.path.abspath("..").split("data_processing")[0]
print("ROOT", ROOT_PATH)
module_path = os.path.abspath(os.path.join(ROOT_PATH+"/utils/"))
if module_path not in sys.path:
    print("appended")
    sys.path.append(module_path)

import functions as f


c, p = f.color_palette()
sns.set(context='paper', style='whitegrid', palette=np.array(p))
print(ROOT_PATH)
plt.style.use('file://' + ROOT_PATH + "/utils/plotparams.rc")



class FastFourierTransform:
    # Amplitudes is a row vector
    def __init__(self, amplitudes, t, type):
        self.s = amplitudes
        self.t = t
        self.normalized_amp = None
        self.rms_time = None
        self.rms_order = None
        self.type = type


    # Plotting in the input domain before fft
    def plot_input(self):

        plt.ylabel("Amplitude")
        plt.xlabel("Shaft angle [Radians]")
        plt.plot(self.t, self.s)
        plt.margins(0)
        plt.show()

        '''
        The second half of this array of fft sequence have similar frequencies
        since the frequency is the absolute value of this value.
        Input:
        avg_speed = the average speed during operation (used to print in plot)
        avg_power = the average power generated during operation (used to print in plot)
        interval_num = which interval is evaluated
        '''
    def fft_transform_order(self, rot_data, avg_power, interval_num, plot=False, name=''):
        mean_amplitude = np.mean(self.s)
        self.s = self.s - mean_amplitude # Centering around 0
        fft = np.fft.fft(self.s)

        # We now have the fft for every timestep in out plot.

        # T is the sample frequency in the data set
        T = self.t[1] - self.t[0] # This is true when the period between each sample in the time waveform is equal
        N = self.s.size  # size of the amplitude vector
        f = np.linspace(0, 1 / T, N, )  # start, stop, number of. 1 / T = frequency is the bigges freq
        f = f[:N // 2]
        y = np.abs(fft)[:N // 2] * 1 / N  # Normalized. Cutting away half of the fft frequencies.

        if plot == True:
            plt.figure(figsize=(15, 5))
            plt.ylabel("Normalised Amplitude")
            plt.xlabel("Order [X]")
            y = np.abs(fft)[:N // 2] * 1 /N # Normalized. Cutting away half of the fft frequencies.
            plt.plot(f, y, markersize=0.5, marker="o", lw=2)
            xticks = f
            #plt.xticks(range(1, len(xticks) + 1), f, rotation=90)
            plt.title(f'FFT Order Transformation of {name} Interval: {interval_num} \nAvg Power: {avg_power:.2f}     '
                      f'Mean RPM: {rot_data["mean"]:.2f},     Max RPM: {rot_data["max"]:.2f},     '
                      f'Min RPM: {rot_data["min"]:.2f},     STD RPM: {rot_data["std"]:.2f}')
            plt.margins(0)
            plt.show()

        time = f[:N // 2]
        self.normalized_amp = y

        # Calculate the spectral centroid
        centroid = self.find_spectral_centroid(f,y)
        return fft, time, centroid

    # Skew, Kortoisi
    '''
     f is the half spectrum frequencies in the fft
     y is the normalized magnitude of the fft
    '''
    def find_spectral_centroid(self, f,y):
        weight_sum = 0
        for i, freq in enumerate(f):
            weight_sum += freq * y[i]
        return weight_sum/np.sum(y)


    def fft_transform_time(self, rot_data, avg_power, calc_rms_for_bins=False, plot=False, bins=0, plot_vertical_lines=False):
        mean_amplitude = np.mean(self.s)
        self.s = self.s - mean_amplitude  # Centering around 0
        fft = np.fft.fft(self.s)

        # We now have the fft for every timestep in out plot.

        # T is the sample frequency in the data set
        T = self.t[1] - self.t[0]  # This is true when the period between each sample in the time waveform is equal
        N = self.s.size  # size of the amplitude vector
        f = np.linspace(0, 1 / T, N, )  # start, stop, number of. 1 / T = frequency is the biggest freq
        f = f[:N // 2]
        y = np.abs(fft)[:N // 2]
        y_norm = np.abs(fft)[:N // 2] * 1 / N  # Normalized
        fft_modulus_norm = y_norm

        rms_bins = []
        frequency_bins = np.linspace(0, max(f), bins + 1)
        if calc_rms_for_bins:
            delta_f = f[1] - f[0]

            # Finding the correct indices to separate frequency and amplitude correctly.
            bin_indexes = [0]
            # Since the firs idex is already added to bin_indexes we start frequency_bins_index at 1
            frequency_bins_index = 1
            for i in range(len(f)):
                if f[i] >= frequency_bins[frequency_bins_index]:
                    frequency_bins_index += 1
                    bin_indexes.append(i)

            for i in range(len(bin_indexes)-1):
                amp = y_norm[bin_indexes[i]:bin_indexes[i+1]]
                rms_bins.append(self.rms_bin(amp))
        x = [(frequency_bins[a] + frequency_bins[a + 1]) / 2 for a in range(len(frequency_bins) - 1)] # The frequency index for the bins!
        rms = self.rms(f, fft_modulus_norm) # F is the half of the frequencies, ffy_modulus_norm is the normalised |fft|
        self.rms_time = rms
        if plot == True:
            title = f'Avg Power: {avg_power:.2f}     Mean RPM: {rot_data["mean"]:.2f},     Max RPM: {rot_data["max"]:.2f},     Min RPM: {rot_data["min"]:.2f},     STD RPM: {rot_data["std"]:.2f}'
            fig, ax1 = plt.subplots(figsize=(15, 5))
            ax1.set_xlabel("Frequency [Hz]")
            ax1.set_ylabel("Normalised amplitude")
            ax1.set_ylim(min(y_norm), max(y_norm)*1.05)
            #plt.xlabel("Frequency [Hz]")
            # plt.ylabel("Normalised Amplitude")
            ax1.plot(f, y_norm, markersize=0.5, marker="o", lw=2, label='FFT transformation')

            # Plot RMS bin values in the same figure
            if calc_rms_for_bins:
                ax2 = ax1.twinx()
                ax2.set_ylim(min(rms_bins), max(rms_bins)*1.05)
                ax2.set_ylabel("Average RMS for each bin")
                # Plot each RMS value at the average frequency for each bin
                x = [(frequency_bins[a] + frequency_bins[a + 1]) / 2 for a in range(len(frequency_bins) - 1)]
                ax2.plot(x, rms_bins, c='g', label='RMS for each bin')
                plt.title(f"FFT Transformation to the time domain with RMS for {bins} bins\n" + title)
                fig.legend(loc='upper right', bbox_to_anchor=(0.55, 0.5, 0.4, 0.4))
            else:
                # Use another title if we don't plot RMS values
                plt.title("FFT Transformation to the time domain\n" + title)

            # Plot vertical lines in the same plot
            if plot_vertical_lines:
                for i, bin in enumerate(frequency_bins):
                    plt.axvline(x=bin, c='r', linewidth=0.5)
            plt.margins(0)
            plt.show()

        time = f[:N // 2]
        self.normalized_amp = y_norm

        # Calculate the spectral centroid
        centroid = self.find_spectral_centroid(f, y_norm)
        return fft, time, centroid, rms, rms_bins, x


    def rms_bin(self, amp):
        sum = 0
        for a in amp:
            sum += a**2
        rms = np.sqrt(2*sum)
        return rms


    # Function returns rms as a float. Called in the fft_transform_time function
    def rms(self, freq, fft_modulus_norm):
        # Filtering the frequency spectrum based on what component is being analysed
        if (self.type=="generator"):
            filter_indexes = [(freq > 10) & (freq < 5000)]
        if self.type == "gearbox":
            filter_indexes = [(freq > 10) & (freq < 2000)]
        if self.type == "nacelle":
            filter_indexes = [(freq > 0.1) & (freq < 10)]

        #freq = freq[filter_indexes]
        amp = fft_modulus_norm[filter_indexes]

        sum=0
        for a in amp:
            sum+=a**2
        rms=np.sqrt(2*sum)
        return rms


    # Siemens implementation
    def calculate_rms(self,freq, fft_modulus_norm):
        filter_indexes=[(freq > 10) & (freq<5000)]
        freq = freq[filter_indexes]
        fft_modulus_norm = fft_modulus_norm[filter_indexes]
        Y1 = fft_modulus_norm*2
        sum=0
        for a in Y1:
            sum+=a**2
        rms = np.sqrt(0.5*sum)
        return rms


# ************ EXAMPLE FOR WT01 ******************
'''
wt_instance = wt_data.load_instance("WTG01",load_minimal = True)
intervals = wt_instance.ten_second_intervals

rms_arr = []
for i, interval in enumerate(intervals):
    print(f"Interval number {i} started..")
    # CHECK IF WE WANT INTERVAL
    interval1 = interval
    time_stamps = interval.sensor_df['TimeStamp']
    try:
        feature = 'GnNDe;0,0102;m/s2'
        vibration = interval.sensor_df['GnNDe;0,0102;m/s2']
        if feature == 'GnNDe;0,0102;m/s2':
            type = "generator"
    except:
        continue
    fast = FastFourierTransform(vibration, time_stamps, type)
    fft, time, centroid,rms = fast.fft_transform_time(True)
    #RMS = fast.calculate_rms1(fft)
    # RMS = fast.calc_rms(fft)
    print(" ")
    # print(f"RMS new:: {RMS}")
    rms_arr.append(rms) # This can be plotted
    # LAG FOR LAV FREKVENS
    # LAG FOR HØY FREKVENS

plt.figure(figsize=(15, 5))
plt.ylabel("RMS value")
plt.xlabel("Interval nr.")
plt.plot(range(len(rms_arr)), rms_arr,marker="o", markersize=4)
plt.title("RMS PLOT")
plt.margins(0)
plt.ylim(0, 7.5)
plt.show()

'''