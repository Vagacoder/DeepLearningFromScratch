# * Functions for chapter 4
import numpy as np 


# * mean squared error
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


# * cross entropy error simple version
def cross_entropy_error_simple(y, t):
    # ! plus delta to protect from np.log(0) -> inf
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# * cross entropy error for both single and batch data
# * for t is one-hot, such as [0, 0, 1, 0, 1, 0]
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


# * for t is label, such as 2.
def cross_entropy_error_label(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


