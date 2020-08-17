#%%
import numpy as np
import matplotlib.pylab as pl


def step_function(x):
    return np.array(x>0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
# print(x)
# print(y)
pl.plot(x, y)
pl.ylim(-0.1, 1.1)
pl.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


y2 = sigmoid(x)
pl.plot(x, y2)
pl.ylim(-0.1, 1.1)
pl.show()


def relu(x):
    return np.maximum(0, x)


y3 = relu(x)
pl.plot(x, y3)
pl.ylim(-1, 5)
pl.show()

