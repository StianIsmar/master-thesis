from PyEMD import EEMD
import numpy as np
import matplotlib.pyplot as plt

def eemd(wt_name, interval_num, component,signal,timestamps,plotting=True):

	# Define signal
	S = signal
	t = timestamps

	# Assign EEMD to `eemd` variable
	eemd = EEMD()
	# Say we want detect extrema using parabolic method
	emd = eemd.EMD
	emd.extrema_detection="parabol"
	# Execute EEMD on S
	eIMFs = eemd.eemd(S, t,max_imf=1)
	nIMFs = eIMFs.shape[0]
	# Plot results	

	if plotting:
		# plt.plot(t, S, 'r')
		# plt.title("Filtered signal")

		
		f, axs = plt.subplots((nIMFs)+1,1,figsize=(15,20))

		# plotting the filtered signal first
		axs[0].plot(t, S, 'r')
		plt.title("{component} filtered signal from {wt_name}. \n Interval number {interval_num}.")
		for n in range(nIMFs):
		    axs[n+1].plot(t, eIMFs[n], 'g')
		    plt.ylabel("eIMF %i" %(n+1))
		    plt.locator_params(axis='y', nbins=5)
		    plt.xlabel("Time [s]")
		    plt.tight_layout()
		    plt.savefig(f' for {wt_name}', dpi=120)
		plt.show()

		print(f'{eIMFs.shape} is the shape of the IMFs!')

	eIMFs = np.insert(eIMFs,0,S,axis=0) # Inserting the original signal into the fist index of the returned array
	#print(f'{eIMFs.shape} is the shape of the IMFs!')
	return eIMFs
