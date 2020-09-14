#
# * Try to understand im2col implementation

#%%
# ? How the code below works in python
# ? col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

import numpy as np

a = np.zeros((4, 5), dtype=int)
print('initial a:')
print(a)
print()

for i in range(20):
    row = i//5
    a[row, i-row*5] = i

print('fill a with increasing numbers')
print(a)
print()

b = np.zeros((4, 3 ,5), dtype=int)
print('initial b:')
print(b)
print()

for i in range(3):
    b[:, i, :] = a[:, :]

print('filled b')
print(b)
print()


c = np.zeros((4, 3 ,5), dtype=int)
print('initial c:')
print(c)
print()
c[:, 1, :] = a[:, :]

print('filled c')
print(c)
print()

# %%
