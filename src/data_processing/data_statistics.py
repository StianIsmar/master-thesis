import pandas as pd
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