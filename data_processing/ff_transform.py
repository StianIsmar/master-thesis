import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wt_data

class FastFourierTransform:
    # Amplitudes is a row vector
    def __init__(self, amplitudes, t):
        self.s = amplitudes
        self.t = t
        self.normalized_amp = None

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
    def fft_transform_order(self, rot_data, avg_power, interval_num, plot=False):
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
            plt.figure(figsize=(15, 8))
            plt.ylabel("Normalised Amplitude")
            plt.xlabel("Order [X]")
            sns.lineplot(f, y)
            plt.title(f'FFT Order Transformation of INTERVAL: {interval_num} \nAvg Power: {avg_power:.2f}     '
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
    def find_spectral_centroid(self, f,y):
        weight_sum = 0
        for i, freq in enumerate(f):
            weight_sum += freq * y[i]
        return weight_sum/np.sum(y)


    def fft_transform_time(self,plot=False):
        mean_amplitude = np.mean(self.s)
        self.s = self.s - mean_amplitude  # Centering around 0
        fft = np.fft.fft(self.s)

        # We now have the fft for every timestep in out plot.

        # T is the sample frequency in the data set
        T = self.t[1] - self.t[0]  # This is true when the period between each sample in the time waveform is equal
        N = self.s.size  # size of the amplitude vector
        f = np.linspace(0, 1 / T, N, )  # start, stop, number of. 1 / T = frequency is the bigges freq
        f = f[:N // 2]
        y = np.abs(fft)[:N // 2] * 1 / N  # Normalized
        # Cutting away half of the fft frequencies.

        if plot == True:
            plt.figure(figsize=(15, 8))
            plt.ylabel("Normalised Amplitude")
            plt.xlabel("Order [X]")


            sns.lineplot(f, y)
            plt.title("FFT of time domain amplitude")
            plt.title("FFT Transformation to the time domain")
            plt.margins(0)
            plt.show()

        time = f[:N // 2]
        self.normalized_amp = y

        # Calculate the spectral centroid
        centroid = self.find_spectral_centroid(f, y)
        return fft, time, centroid

    def calculate_rms(self):
        return ''

'''
wt_instance = wt_data.load_instance("WTG01",load_minimal=True)
intervals = wt_instance.ten_second_intervals

for i, interval in enumerate(intervals):
    # CHECK IF WE WANT INTERVAL
    interval1 = interval
    time_stamps = interval.sensor_df['TimeStamp']
    vibration = interval.sensor_df['GnNDe;0,0102;m/s2']
    fast = FastFourierTransform(vibration, time_stamps)
    fast.fft_transform_time(True)

'''
