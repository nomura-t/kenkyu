import pickle
import matplotlib.pyplot as plt
import numpy as np


with open('dltbig5.pkl','rb') as f:
	p = pickle.load(f)
p = np.array(p)
p = np.abs(p)

plt.hist(p[:,0],bins=(np.array(range(70))-30)/100)
binarized = []


for i in range(5):
	threshold = np.median(p[:,i]) #+ np.std(p[:,i])
	binarized.append([1 if j > threshold else 0 for j in p[:,i]])	

with open('deltthreshold.pkl','wb') as f:
	pickle.dump(binarized,f)

