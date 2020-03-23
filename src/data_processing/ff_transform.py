from __future__ import division
import wt_data

# import external modules
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys, os

ROOT_PATH = os.path.abspath("..").split("data_processing")[0]
print("ROOT", ROOT_PATH)
module_path = os.path.abspath(os.path.join(ROOT_PATH + "/utils/"))
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
        self.bin_indexes_range = None
        self.rms_bins_range_magnitude = None

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

    def fft_transform_order(self, rot_data, avg_power, interval_num='unknown', plot=False, get_rms_for_bins=False,
                            bins=0, name='', plot_bin_lines=False, x_lim=None, order_lines=[], horisontal_lines=[],
                            interpolation_method=''):
        mean_amplitude = np.mean(self.s)
        self.s = self.s - mean_amplitude  # Centering around 0
        fft = np.fft.fft(self.s)

        # We now have the fft for every timestep in out plot.

        # T is the sample frequency in the data set
        T = self.t[1] - self.t[0]  # This is true when the period between each sample in the time waveform is equal
        N = self.s.size  # size of the amplitude vector
        f = np.linspace(0, 1 / T, N, )  # start, stop, number of. 1 / T = frequency is the bigges freq
        f = f[:N // 2]
        y = np.abs(fft)[:N // 2] * 1 / N  # Normalized. Cutting away half of the fft frequencies.

        # Calculate the bins
        rms_bins = []
        frequency_bins = np.linspace(0, max(f), bins + 1)
        x = [(frequency_bins[a] + frequency_bins[a + 1]) / 2 for a in
             range(len(frequency_bins) - 1)]  # The frequency index for the bins!
        if get_rms_for_bins:
            delta_f = f[1] - f[0]

            # Finding the correct indices to separate frequency and amplitude correctly.
            bin_indexes = [0]
            # Since the firs idex is already added to bin_indexes we start frequency_bins_index at 1
            frequency_bins_index = 1
            for i in range(len(f)):
                if f[i] >= frequency_bins[frequency_bins_index]:
                    frequency_bins_index += 1
                    bin_indexes.append(i)

            for i in range(len(bin_indexes) - 1):
                amp = y[bin_indexes[i]:bin_indexes[i + 1]]
                rms_bins.append(self.rms_bin(amp))

        if plot == True:
            title = f'Avg Power: {avg_power:.2f}     Mean RPM: {rot_data["mean"]:.2f},     ' + \
                    f'Max RPM: {rot_data["max"]:.2f},     Min RPM: {rot_data["min"]:.2f},     ' + \
                    f'STD RPM: {rot_data["std"]:.2f}'
            fig, ax1 = plt.subplots(figsize=(15, 5))
            ax1.set_xlabel("Order [X]")
            ax1.set_ylabel("Normalised amplitude")
            ax1.set_ylim(min(y), max(y) * 1.05)
            ax1.plot(f, y, markersize=0.5, marker="o", lw=2, label='FFT transformation')
            
            # Show the last frequency:
            #print(f[-1])
            # Plot RMS bin values in the same figure
            if get_rms_for_bins:
                ax2 = ax1.twinx()
                ax2.set_ylim(min(rms_bins), max(rms_bins) * 1.05)
                ax2.set_ylabel("RMS value")
                # Plot each RMS value at the average frequency for each bin
                x = [(frequency_bins[a] + frequency_bins[a + 1]) / 2 for a in
                     range(len(frequency_bins) - 1)]  # The frequency index for the bins!
                ax2.plot(x, rms_bins, c='g', label='RMS for each bin')
                plt.title(f"Order Domain FFT Transformation     {interpolation_method} Interpolation     " +
                          f"{bins} RMS bins     {name}     Interval: {interval_num}\n" + title)
                fig.legend(loc='upper right', bbox_to_anchor=(0.32, 0.35, 0.5, 0.5))
            else:
                # Use another title if we don't plot RMS values
                plt.title(f"Order Domain FFT Transformation     {name}     Interval: {interval_num}\n" + title)

            # Plot vertical lines in the same plot
            if plot_bin_lines:
                for i, bin in enumerate(frequency_bins):
                    plt.axvline(x=bin, c='r', linewidth=0.5)

            if len(order_lines) != 0:
                for order in order_lines:
                    plt.axvline(x=order, c='r', linewidth=0.5)

            if len(horisontal_lines) != 0:
                for line in horisontal_lines:
                    plt.axhline(y=line, c='y', linewidth=0.5)

            if x_lim != None:
                plt.xlim(0, x_lim)

            plt.margins(0)
            plt.show()

        f = np.array(f)
        time = f[:N // 2]
        self.normalized_amp = y

        # Calculate the spectral centroid
        centroid = self.find_spectral_centroid(f, y)
        return fft, time, centroid, rms_bins, x







    def find_spectral_centroid(self, f, y):
        weight_sum = 0
        for i, freq in enumerate(f):
            weight_sum += freq * y[i]
        return weight_sum / np.sum(y)








    def fft_transform_time(self, rot_data=[], avg_power=-1, avg_rpm=-1, name="", interval_num='unknown', plot=False,
                           get_rms_for_bins=False, bins=0, plot_bin_lines=False, x_lim=None, frequency_lines=[],
                           horisontal_lines=[], spectrum_lower_range=-1, spectrum_higher_range=1):
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

        # Calculate the bins
        rms_bins = []
        frequency_bins = np.linspace(0, max(f), bins + 1)
        if get_rms_for_bins:
            delta_f = f[1] - f[0]

            # Finding the correct indices to separate frequency and amplitude correctly.
            bin_indexes = [0]
            # Since the first index is already added to bin_indexes we start frequency_bins_index at 1
            frequency_bins_index = 1
            for i in range(len(f)):
                if f[i] >= frequency_bins[frequency_bins_index]:
                    frequency_bins_index += 1
                    bin_indexes.append(i)

            for i in range(len(bin_indexes) - 1):
                amp = y_norm[bin_indexes[i]:bin_indexes[i + 1]]  # Calculating the rms between two indexes
                rms_bins.append(self.rms_bin(amp))
        x = [(frequency_bins[a] + frequency_bins[a + 1]) / 2 for a in
             range(len(frequency_bins) - 1)]  # The frequency index for the bins!

        # Calculate RMS for the ISO interval
        rms = self.rms(f, fft_modulus_norm)  # F is the half of the frequencies, ffy_modulus_norm is the normalised |fft|
        self.rms_time = rms

        bin_indexes_range = []
        rms_bins_range_magnitude = []

        if spectrum_lower_range != -1:
            bin_indexes_range, rms_bins_range_magnitude = \
                self.fft_transform_time_specified_range(y_norm, f, bins, spectrum_lower_range, spectrum_higher_range)
            self.bin_indexes_range = bin_indexes_range
            self.rms_bins_range_magnitude = rms_bins_range_magnitude

        if plot == True:
            if (len(rot_data) > 0) and (avg_power > -1):
                title = f'Avg Power: {avg_power:.2f}     Mean RPM: {rot_data["mean"]:.2f},     ' + \
                        f'Max RPM: {rot_data["max"]:.2f},     Min RPM: {rot_data["min"]:.2f},     ' + \
                        f'STD RPM: {rot_data["std"]:.2f}'
            elif (avg_rpm > -1) and (avg_power > -1):
                title = f'Avg Power: {avg_power:.2f}     Mean RPM: {avg_rpm:.2f},     '
                
            fig, ax1 = plt.subplots(figsize=(15, 5))
            ax1.set_xlabel("Frequency [Hz]")
            ax1.set_ylabel("Normalised amplitude")
            ax1.set_ylim(min(y_norm), max(y_norm) * 1.05)
            ax1.plot(f, y_norm, markersize=0.5, marker="o", lw=2, label='FFT transformation')

            # Plot RMS bin values in the same figure
            if get_rms_for_bins:
                ax2 = ax1.twinx()
                ax2.set_ylim(min(rms_bins), max(rms_bins) * 1.05)
                ax2.set_ylabel("RMS value")
                # Plot each RMS value at the average frequency for each bin
                x = [(frequency_bins[a] + frequency_bins[a + 1]) / 2 for a in range(len(frequency_bins) - 1)]
                ax2.plot(x, rms_bins, c='g', label='RMS for each bin')
                plt.title(
                    f"Time Domain FFT Transformation     {bins} RMS bins     {name}     Interval: {interval_num}\n"
                    + title)
                fig.legend(loc='upper right', bbox_to_anchor=(0.32, 0.35, 0.5, 0.5))
            else:
                # Use another title if we don't plot RMS values
                plt.title(f"Time Domain FFT Transformation     {name}     Interval: {interval_num}\n" + title)

            # Plot bin lines in the same plot
            if plot_bin_lines:
                for i, bin in enumerate(frequency_bins):
                    plt.axvline(x=bin, c='r', linewidth=0.5)

            if len(frequency_lines) != 0:
                for order in frequency_lines:
                    plt.axvline(x=order, c='r', linewidth=0.5)

            if len(horisontal_lines) != 0:
                for line in horisontal_lines:
                    plt.axhline(y=line, c='y', linewidth=0.5)

            if x_lim != None:
                plt.xlim(0, x_lim)

            plt.margins(0)
            plt.show()

        time = f[:N // 2]
        self.normalized_amp = y_norm

        # Calculate the spectral centroid
        centroid = self.find_spectral_centroid(f, y_norm)
        return fft, time, centroid, rms, rms_bins, x




    def fft_transform_time_specified_range(self, y_norm, f, bins, spectrum_lower_range, spectrum_higher_range):
        rms_bins_range_magnitude = []
        if self.type == "gearbox":
            filter_indexes = [(f > spectrum_lower_range) & (f < spectrum_higher_range)]
            f = f[filter_indexes]
            y_norm = y_norm[filter_indexes]
            maxx = max(f)
            frequency_bins_range = np.linspace(0, max(f), bins + 1)
            bin_indexes = [0]
            frequency_bins_index = 1
            for i in range(len(f)):
                check = f[i]
                this = frequency_bins_range[frequency_bins_index]
                if check >= this:
                    frequency_bins_index += 1
                    bin_indexes.append(i)
            bin_indexes_range = [(frequency_bins_range[a] + frequency_bins_range[a + 1]) / 2 for a in
                                range(len(frequency_bins_range) - 1)]  # The frequency index for the bins!

            for i in range(len(bin_indexes)-1):
                amp = y_norm[bin_indexes[i]:bin_indexes[i + 1]]  # Calculating the rms between two indexes
                rms_bins_range_magnitude.append(self.rms_bin(amp))

            #plt.plot(bin_indexes_range, rms_bins_range_magnitude)
            #plt.show()
            return bin_indexes_range, rms_bins_range_magnitude
    def rms_bin(self, amp):
        sum = 0
        for a in amp:
            sum += a ** 2
        rms = np.sqrt(2 * sum)
        return rms

    # Function returns rms as a float. Called in the fft_transform_time function
    def rms(self, freq, fft_modulus_norm):
        # Filtering the frequency spectrum based on what component is being analysed
        filter_indexes=[]
        if (self.type == "generator"):
            filter_indexes = [(freq > 10) & (freq < 5000)]
        if self.type == "gearbox":
            filter_indexes = [(freq > 10) & (freq < 2000)]
        if self.type == "nacelle":
            filter_indexes = [(freq > 0.1) & (freq < 10)]

        # freq = freq[filter_indexes]
        amp = fft_modulus_norm[filter_indexes]

        sum = 0
        for a in amp:
            sum += a ** 2
        rms = np.sqrt(2 * sum)
        return rms

    # Siemens implementation
    def calculate_rms(self, freq, fft_modulus_norm):
        filter_indexes = [(freq > 10) & (freq < 5000)]
        freq = freq[filter_indexes]
        fft_modulus_norm = fft_modulus_norm[filter_indexes]
        Y1 = fft_modulus_norm * 2
        sum = 0
        for a in Y1:
            sum += a ** 2
        rms = np.sqrt(0.5 * sum)
        return rms


# Debugging:
'''
WIND_TURBINE = 'WTG01'
SENSOR_NAME = 'GbxHssRr;0,0102;m/s2'
wt_instance = wt_data.load_instance(WIND_TURBINE, load_minimal=False)
interval = wt_instance.ten_second_intervals[0]

rot_data = interval.high_speed_rot_data
avg_power = interval.op_df["PwrAvg;kW"][0]
BINS = 50
lower_range_freq = 0
higher_range_freq = 2300
comp_type = 'gearbox'
ts = interval.sensor_df['TimeStamp']  # Have this as the y-axis to see how the RMS/frequencies develop
vibration_signal = interval.sensor_df[SENSOR_NAME]

fast = FastFourierTransform(vibration_signal, ts, comp_type)
fft, time, centroid, rms, rms_bins, bin_freq = fast.fft_transform_time(
    rot_data,
    avg_power,
    get_rms_for_bins=True,
    plot=True,
    bins=BINS,
    plot_bin_lines=False,
    x_lim=None,
    frequency_lines=[],
    horisontal_lines=[],
    spectrum_lower_range=lower_range_freq,
    spectrum_higher_range=higher_range_freq
)
'''