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