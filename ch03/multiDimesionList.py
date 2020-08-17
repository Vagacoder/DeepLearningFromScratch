#%%

import numpy as np 

a = np.array([1, 2 ,3 ,4])
print('Array of ', end='')
print(a)
print('a\'s dimesion number:', end='')
print(np.ndim(a))
print('a\'s shape: ', end='')
print(a.shape)
print('a\'s 0th dimension: ', end='')
print(a.shape[0])

A = np.array([[1, 2],[3, 4]])
B = np.array([[5, 6],[7, 8]])
C = np.dot(A, B)
print(C)
