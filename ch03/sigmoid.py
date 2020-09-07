# coding: utf-8

#%%
import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y, linestyle='dotted', label='Sigmoid function')
plt.ylim(-0.1, 1.1)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# %%
