import numpy as np
import scipy as sp
import sys
from BOW import *
#import matplotlib.pyplot as plt

# Test Case 1: Normal distribution
#N_A = 100
#N_B = 100
#A = np.random.multivariate_normal([1.0,1.0],[[1.0,0.0],[0.0,1.0]],N_A) 
#B = np.random.multivariate_normal([2.0,2.0],[[1.0,0.0],[0.0,1.0]],N_B) 
#A_gt_labels = np.zeros(N_A,dtype='int')
#B_gt_labels = np.ones(N_B,dtype='int')
#feat  = np.concatenate((A,B), axis=0)
#gt_labels = np.concatenate((A_gt_labels,B_gt_labels), axis=0)

# Test Case 2: Tiny dataset
#
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


V = BOW(data=feat,voc_size=2,gt_labels=gt_labels)
#print V.vq_data
feat2 = np.concatenate((feat,feat),axis=0)
V.calc_bow_representation(fv=feat2)
print V.bow

