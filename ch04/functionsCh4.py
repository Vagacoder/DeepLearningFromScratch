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


# * numerical differentiation, need 2 improvements
# * 1. h is too small, causing round error
# * 2. num_diff is calculated using (x+h) and x, better to use (x+h) and (x-h)
def numerical_diff_bad(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h


# * numerical differentiation
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# * calculate gradient
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        original_val = x[i]

        x[i] = original_val + h
        yh1 = f(x)

        x[i] = original_val - h
        yh2 = f(x)

        grad[i] = (yh1 - yh2) / (2 * h)
        x[i] = original_val

    return grad


# * Gradient descent method
def gradient_descent(f, initX, lr=0.01, stepNum=100):
    x = initX

    for i in range(stepNum):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


