import numpy as np
import scipy as sp
import sys
from VisVoc import *
import matplotlib.pyplot as plt

desc = [];

desc.append([1,1])
desc.append([1.5,1])
desc.append([1,15])
desc.append([1.5,1])
desc.append([1,1.5])
desc.append([1,1.6])
desc.append([10,10])
desc.append([12,10])
desc.append([10,13])
desc.append([14,10])
desc.append([10,15])

feat = sp.vstack(tuple(desc))
gt_labels = np.array([0,0,1,0,0,0,1,1,1,1,1])

V = VisVoc(data=feat,voc_size=2,gt_labels=gt_labels)
print V.vq_data

#n, bins, patches = plt.hist(hist, 100, normed=1, facecolor='g', alpha=0.75)
#plt.figure()
#plt.show()
