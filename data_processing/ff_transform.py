import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        '''
    def fft_transform(self):
        mean_amplitude = np.mean(self.s)
        self.s = self.s - mean_amplitude # Centering around 0
        fft = np.fft.fft(self.s)

        # We now have the fft for every timestep in out plot.

        # T is the sample frequency in the data set
        T = self.t[1] - self.t[0] # This is true when the period between each sample in the time waveform is equal
        N = self.s.size  # size of the amplitude vector
        f = np.linspace(0, 1 / T, N, )  # start, stop, number of. 1 / T = frequency is the bigges freq
        f = f[:N // 2]
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency [Hz]")
        y = np.abs(fft)[:N // 2] * 1 /N # Normalized

        # Cutting away half of the fft frequencies.
        '''
        sns.lineplot(f, y)
        plt.margins(0)
        plt.show()
        '''
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







'''
t = np.linspace(0, 0.5, 500)

amplitudes = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)

fast = FastFourierTransform(amplitudes, t)
fast.plot_input()
transform, time = fast.fft_transform()

'''


