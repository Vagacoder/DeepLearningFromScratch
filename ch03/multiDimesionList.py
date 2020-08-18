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

print('\nMatrix multiplication 2x2 * 2x2:')
A = np.array([[1, 2],[3, 4]])
B = np.array([[5, 6],[7, 8]])
C = np.dot(A, B)
print(C)

print('\nMatrix multiplication 2x3 * 3x2:')
D = np.array([[1, 2, 3],[4, 5, 6]])
E = np.array([[10, 11], [12, 13], [14, 15]])
F = np.dot(D, E)
print(F)


print('\nMatrix * vector:')
I = np.array([7, 8, 9])
print(I.shape)
print(np.dot(D, I))
J = np.array([[7],[8],[9]])
print(J.shape)
print(np.dot(D, J))

print('\n Matrix multiplication wrong:')
G = np.array([[10, 11],[12, 13]])
print('dim 1 of D:', end=' ')
print(D.shape[1])
print('dim 0 of G:', end=' ')
print(G.shape[0])
H = np.dot(D, G)
print(H)

# %%
