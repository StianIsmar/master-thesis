from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import numpy as np

def plot_n_imfs_and_kurtosis(wt_num, imf_df,imf_number,timestamp_of_signal,SAVE_FILE_NAME, times,input_file_type,number_to_print=10):
    fs=25600
    df=imf_df
    fig, axs = plt.subplots(number_to_print, figsize=(7.5, 9.5), facecolor='w', edgecolor='k')

    # Looping through the IMFs in the matrix
    for j in range(number_to_print):
            kurt = kurtosis(df.iloc[j,:].to_numpy())
            kurt = round(kurt, 4)

            if j==0:
                axs[j].set_title(f"WT0{wt_num}: EEMD decomposition for \nInterval number {imf_number}. Date and time: {timestamp_of_signal}\n\nKurtosis: {kurt}")
                axs[j].plot(times[j][0:20000],df.iloc[j,:].to_numpy(),color='#F87060')
                axs[j].set_ylabel('Raw signal')

            else:
                axs[j].set_title(f"Kurtosis: {kurt}")
                axs[j].plot(times[j][0:20000],df.iloc[j,:].to_numpy())
                axs[j].set_ylabel(f'IMF{j}')

            #axs[j].set_xlabel('Seconds [s]')
            axs[j].margins(0)
            axs[j].grid(b=None)

            if j==(number_to_print-1):
                axs[j].set_xlabel('Time [s]', ha='center',fontsize=14)
            
            # y labels
            labels= axs[j].get_yticks()
            axs[j].set_yticks((labels[0],labels[-1]))
            print(labels)

    # fig.text(0.5, -0.003, 'Time [s]', ha='center',fontsize=14) #global xlabel
    plt.tight_layout()
    plt.savefig(f'../../plots/eemds/{SAVE_FILE_NAME}.png',dpi=200)
    plt.show()

def plot_imfs_and_kurtosis(imf_df,SAVE_FILE_NAME,times,input_file_type,input_signal_origin='filtered_6000',number_to_print=10):
    # import all IMFs:
    fs=25600
    highcut_lp = 2000

    file_count = len(glob.glob1(path_folder,"*.csv"))
    print("file count:", file_count)


    for i in range(file_count):
        if input_file_type == 'csv':
            path=path_folder + f'interval_number_{i}.csv'
            df = pd.read_csv(path, header=None)
        if input_file_type == 'zip':
            path=path_folder + f'raw_wt04_interval_number_{i}.zip'
            df = pd.read_csv(path,compression='zip')
        # s = df.to_numpy()


        fig, axs = plt.subplots(df.shape[0], figsize=(15, 25), facecolor='w', edgecolor='k')


        for j in range(df.shape[0]): # Looping through the IMFs
            kurt = kurtosis(df.iloc[j,:].to_numpy())
            kurt = round(kurt, 4)

            if j==0:
                axs[j].set_title(f"EEMD decomposition for WT 4\ninterval number {i}\n\nKurtosis: {kurt}")
                axs[j].plot(times[j][0:20000],df.iloc[j,:].to_numpy(),color='#F87060')
                if input_signal_origin =='filtered_signal':
                    axs[j].set_ylabel('Filtered signal')
                if input_signal_origin =='raw_signal':
                    axs[j].set_ylabel('Raw signal')

            else:
                axs[j].set_title(f"Kurtosis: {kurt}")
                axs[j].plot(times[j][0:20000],df.iloc[j,:].to_numpy())
                axs[j].set_ylabel(f'IMF{j}')

            #axs[j].set_xlabel('Seconds [s]')
            axs[j].margins(0)
            axs[j].grid(b=None)

        fig.text(0.5, -0.003, 'Time [s]', ha='center',fontsize=14) #global xlabel
        plt.tight_layout(pad=2)


        plt.savefig(f'../../plots/eemds/{SAVE_FILE_NAME}.png',dpi=200)
        plt.show()

        if i == number_to_print:
            break



'''def read_imfs(path_folder):
    file_count = len(glob.glob(path_folder,"*.csv"))
    print("file count:", file_count)
    for i in range(file_count):
'''
         
