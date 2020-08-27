# * Functions for chapter 4
import numpy as np 

# * 1. Output Functions ========================================================
# * 1.1. Identity function
def identity_function(x):
    return x


# * 1.2. Softmax function
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# * 2. Loss Functions ==========================================================
# * 2.1. mean squared error, using with identity func (1.1.)
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


# * 2.2. cross entropy error simple version, using with softmax (1.2.)
def cross_entropy_error_simple(y, t):
    # ! plus delta to protect from np.log(0) -> inf
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# * 2.3. cross entropy error for both single and batch data
# * 2.3.1. for t is one-hot, such as [0, 0, 1, 0, 1, 0]
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


# * 2.3.2. for t is label, such as 2.
def cross_entropy_error_label(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


# * 3. Backward Functions ======================================================
# * 3.1. numerical differentiation, need 2 improvements
# * 1. h is too small, causing round error
# * 2. num_diff is calculated using (x+h) and x, better to use (x+h) and (x-h)
def numerical_diff_bad(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h


# * 3.2. numerical differentiation. better implementation
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# * 3.3. calculate gradient
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


# * 3.4. Gradient descent method (print statements are for debug)
def gradient_descent(f, initX, lr=0.01, stepNum=100):
    x = initX

    for i in range(stepNum):
        grad = numerical_gradient(f, x)
        # print(i, end=", grad: ")
        # print(grad, end=", x: ")
        x -= lr * grad
        # print(x)
        
    return x


