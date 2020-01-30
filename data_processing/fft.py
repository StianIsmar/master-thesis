import numpy as np
import matplotlib.pyplot as plt


class FastFourierTransform:
    # Amplitudes is a row vector
    def __init__(self, amplitudes, t):
        self.s = amplitudes
        self.t = t

    # Plotting in the input domain before fft
    def plot_input(self):
        plt.ylabel("Amplitude")
        plt.xlabel("Time [s]")
        plt.plot(self.t, self.s)
        plt.show()

        '''
        The second half of this array of fft sequence have similar frequencies
        since the frequency is the absolute value of this value.
        '''
    def fft_transform(self):
        fft = np.fft.fft(self.s)

        # We now have the fft for every timestep in out plot.

        # Calculating the frequency spectrum for the simple sine wave function
        T = self.t[1] - self.t[0] # This is true when the period between each sample in the time waveform is equal
        N = amplitudes.size  # size of the amplitude vector
        f = np.linspace(0, 1 / T, N, )  # start, stop, number of. 1 / T = frequency is the bigges freq
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency [Hz]")

        # Cutting away half of the fft frequencies.

        plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 /N, width = 1.5) # N // 2 is normalizing it
        plt.show()
        time = f[:N // 2]
        return fft,


t = np.linspace(0, 0.5, 500)

amplitudes = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)

fast = FastFourierTransform(amplitudes, t)
fast.plot_input()
transform, time = fast.fft_transform()




