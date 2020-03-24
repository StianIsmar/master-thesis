import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
signals is a list of signals, where the first one is the raw signal to compare against.
Returns dataframe object.
'''
def get_corr_coef(signals):
	matrix = np.corrcoef(signals)
	amount_imfs = len(signals) - 1
	imf_labels = [f'IMF{i+1}' for i in range(amount_imfs)]
	imf_labels.insert(0,'Raw signal')
	df = pd.DataFrame(data=matrix)
	df.insert(loc=0,value=imf_labels, column='Signal')
	df = df.drop(columns=np.arange(1,len(signals)),axis=1)
	df = df.rename(columns={0:'Correlation coefficient'})
	# Correlation coefficient
	return df