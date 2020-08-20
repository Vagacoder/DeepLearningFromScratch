#%%
import numpy as np
import matplotlib.pylab as pl

# * Step function
def step_function(x):
    return np.array(x>0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
# print(x)
# print(y)
pl.plot(x, y)
pl.ylim(-0.1, 1.1)
pl.show()


# * Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


y2 = sigmoid(x)
pl.plot(x, y2)
pl.ylim(-0.1, 1.1)
pl.show()


# * Relu function
def relu(x):
    return np.maximum(0, x)


y3 = relu(x)
pl.plot(x, y3)
pl.ylim(-1, 5)
pl.show()


# * Softmax function
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# ! should draw histogram
x4 = np.array([0.2, 0.6, 0.15, 0.05])
y4 = softmax(x4)
pl.plot(x4, y4)
pl.ylim(-0.1, 1)
pl.show()

# %%
