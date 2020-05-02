from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
    path_to_csv [String]: Path to csv you want to build the correlation plot from
    Filename [String]: Name you would like to save the correlation plot as
'''
def create_save_correlation_plot(save_as_filename,plot_title,input_df=None,path_to_csv=None):
    
    if isinstance(input_df, pd.DataFrame):
        df = input_df
    elif path_to_csv !=None:
        df = pd.read_csv(path_to_csv)
        sns.set(style="white")

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(plot_title)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g=sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 11)

    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 11)


    # Saving the file
    f.tight_layout()
    f.savefig(f'../../plots/correlation_plots_df/correlation_plot_{save_as_filename}.eps', format='eps')