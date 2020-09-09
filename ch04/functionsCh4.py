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
    # ! plus delta to protect from np.log(0) -> -inf
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


# * 4. Convolutional Neural Network

# * 4.1. Image to Column
# ? Parameters:
# ? input_data: input dataset, 4 dimemnsions: batch_number, channel_number, height, width
# ? filter_h: height of filter
# ? filter_w: width of filter
# ? stride: stride of filter
# ? pad: padding number for inputer dataset
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0),
                              (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

# * 4.2. Column back to image
# ? Parameters:
# ? col: input dataset, usually it is returned from im2col()
# ? input_shape: input dataset shape , e.g. (10, 1, 28, 28)
# ? filter_h: height of filter
# ? filter_w: width of filter
# ? stride: stride
# ? pad: padding number for input dataset
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]