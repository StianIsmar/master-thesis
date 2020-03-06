import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_rms(df, name=""):
    # get colors from seaborn.pydata.org/tutorial/color_palettes.html
    plt.figure(figsize=(15, 8))
    df = df.drop(columns=['AvgPower'])
    col = 'pastel'
    sns.boxplot(data=df, palette=col)
    if name != "":
        plt.title(f'Boxplot of {name}.   RMS Vibration Values', fontsize=15)
    else:
        plt.title('Boxplot of RMS Vibration Values', fontsize=15)
    plt.show()



def plot_histograms(df, bins=10):
    for i, col_name in enumerate(df.columns.values):
        plt.figure(figsize=(15, 8))
        sns.distplot(df[col_name], bins=10)
        plt.title(f'Histogram of {col_name} with {bins} bins')
        plt.xlabel('Interval')
        plt.ylabel(col_name)
        plt.margins(0)
        plt.show()


def plot_median_rms(df, percentile_1=0.95, percentile_2=0.68, split=False, plot_mean=False, y_max=None, y_max2=None):
    mean = df.mean()
    median = df.median()
    quant = df.quantile([0.5 - percentile_1 / 2, 0.5 + percentile_1 / 2, 0.5 - percentile_2 / 2, 0.5 + percentile_2 / 2])
    x = np.linspace(0, 750, median.shape[0])

    if split:
        index = np.where(x > 122)[0][0]
        x1, x2 = np.split(x, [index])
        mean1, mean2 = np.split(mean, [index])
        median1, median2 = np.split(median, [index])
        quant05_1, quant05_2 = np.split(quant.iloc[0], [index])
        quant95_1, quant95_2 = np.split(quant.iloc[1], [index])
        quant32_1, quant32_2 = np.split(quant.iloc[2], [index])
        quant68_1, quant68_2 = np.split(quant.iloc[3], [index])

        plt.figure(figsize=(15, 10))
        c = sns.color_palette("coolwarm", 15)

        plt.subplot(211)
        plt.plot(x1, median1, label='median')
        if plot_mean:
            plt.plot(x1, mean1, 'c', linestyle='--', label='mean')
        plt.fill_between(x1, quant05_1, quant95_1, color=c[6], alpha=0.7, label=f'{percentile_1} percentile')
        plt.fill_between(x1, quant32_1, quant68_1, color=(205 / 255, 210 / 255, 221 / 255), alpha=0.8,
                         label=f'{percentile_2} percentile')
        if y_max == None:
            plt.ylim(0, max(max(quant05_1), max(quant95_1), max(quant32_1), max(quant68_1)) * 1.05)
        else:
            plt.ylim(0, y_max)
        plt.ylabel('Amplitude RMS [m/s\u00b2]')
        plt.xlabel('Order Frequency')
        plt.margins(0)
        plt.legend()

        plt.subplot(212)
        plt.plot(x2, median2, label='median')
        if plot_mean:
            plt.plot(x2, mean2, 'c', linestyle='--', label='mean')
        plt.fill_between(x2, quant05_2, quant95_2, color=c[6], alpha=0.7, label=f'{percentile_1} percentile')
        plt.fill_between(x2, quant32_2, quant68_2, color=(205 / 255, 210 / 255, 221 / 255), alpha=0.8,
                         label=f'{percentile_2} percentile')
        if y_max2 == None:
            plt.ylim(0, max(max(quant05_2), max(quant95_2), max(quant32_2), max(quant68_2)) * 1.05)
        else:
            plt.ylim(0, y_max2)
        plt.ylabel('Amplitude RMS [m/s\u00b2]')
        plt.xlabel('Order Frequency')
        plt.margins(0)
        plt.legend()

        plt.show()

    else:
        plt.figure(figsize=[15, 5])
        plt.plot(x, median, label='medain')
        if plot_mean:
            plt.plot(x, mean, 'c', linestyle='--', label='mean')
        c = sns.color_palette("coolwarm", 15)
        plt.fill_between(x, quant.iloc[0], quant.iloc[1], color=c[6], alpha=0.7, label=f'{percentile_1} percentile')
        plt.fill_between(x, quant.iloc[2], quant.iloc[3], color=(205 / 255, 210 / 255, 221 / 255),
                         label=f'{percentile_2} percentile')  # , alpha=0.8)
        if y_max == None:
            plt.ylim(0, max(max(quant.iloc[0]), max(quant.iloc[1]))*1.05)
        else:
            plt.ylim(0, y_max)
        plt.ylabel('Amplitude RMS [m/s\u00b2]')
        plt.xlabel('Order Frequency')
        plt.margins(0)
        plt.legend()
        plt.show()




def plot_mean_rms(df, split=False, y_max=None, y_max2=None):
    mean = df.mean()
    std = df.std()
    x = np.linspace(0, 750, mean.shape[0])
    if split:
        index = np.where(x > 122)[0][0]
        x1, x2 = np.split(x, [index])
        mean1, mean2 = np.split(mean, [index])
        std1, std2 = np.split(std, [index])

        plt.figure(figsize=(15, 10))
        c = sns.color_palette("coolwarm", 15)

        plt.subplot(211)
        plt.plot(x1, mean1, label='mean')
        plt.fill_between(x1, mean1-std1, mean1+std1, color=c[6], alpha=0.7, label=f'std')

        if y_max == None:
            plt.ylim(0, max(mean1 + std1) * 1.05)
        else:
            plt.ylim(0, y_max)
        plt.ylabel('Amplitude RMS [m/s\u00b2]')
        plt.xlabel('Order Frequency')
        plt.margins(0)
        plt.legend()

        plt.subplot(212)
        plt.plot(x2, mean2, label='mean')
        plt.fill_between(x2, mean2-std2, mean2+std2, color=c[6], alpha=0.7, label=f'std')
        
        if y_max2 == None:
            plt.ylim(0, max(mean2 + std2) * 1.05)
        else:
            plt.ylim(0, y_max2)
        plt.ylabel('Amplitude RMS [m/s\u00b2]')
        plt.xlabel('Order Frequency')
        plt.margins(0)
        plt.legend()

        plt.show()

    else:
        plt.figure(figsize=[15, 5])
        plt.plot(x, mean, label='mean')
        c = sns.color_palette("coolwarm", 15)
        plt.fill_between(x, mean-std, mean+std, color=c[6], alpha=0.7, label=f'std')

        if y_max == None:
            plt.ylim(0, max(mean + std) * 1.05)
        else:
            plt.ylim(0, y_max)
        plt.ylabel('Amplitude RMS [m/s\u00b2]')
        plt.xlabel('Order Frequency')
        plt.margins(0)
        plt.legend()
        plt.show()
