import numpy as np
import pandas as pd

def convert(filepath):
    data = np.load('vib_signal_wt4.npz')
    data = data['arr_0']
    data = np.array(data)
    df = pd.DataFrame(data)
    df.to_csv(filepath[:-4]+'_df.csv',index=False)

