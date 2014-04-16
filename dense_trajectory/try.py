import numpy as np
import scipy as sp

B = []
B.append([1, 2, 3])
B.append([11, 22, 33])
B.append([21, 22, 23])
B.append([31, 32, 33])

A = np.array(sp.vstack(tuple(B)))

print A

D = {}

for a in A:
    D[(a[0],a[1])] = a[2]

print D
