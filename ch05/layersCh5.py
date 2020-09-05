#

import numpy as np 
import sys, os

sys.path.append(os.pardir)

from functionsCh4 import softmax, cross_entropy_error, im2col, col2im

# * 1. Simple operation layers =================================================
# * 1.1. Multiplication layer
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None


    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out


    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy


# * 1.2. Addition layer
class AddLayer:
    def __init__(self):
        pass


    def forward(self, x, y):
        return x + y
    

    def backward(self, dout):
        return (dout * 1), (dout * 1)


# * 2. Activation Function Layers ==============================================
# * 2.1. Relu (Rectified Linear Unit) layer
class Relu:
    def __init__(self):
        self.mask = None


    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out


    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


# * 2.2. Sigmoid layer
class Sigmoid:
    def __init__(self):
        self.out = None

    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


# * 3 Interlayer, matrix production: operation of A*W + b ======================
# * 3.1. Affine layer
class Affine:

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None


    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out


    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axix=0)
        return dx


# * 4. Output Layer (the last activation function layer) =======================
# * 4.1. Softmax-with-Loss Layer (Output + Loss Function Layer)
# ? softmax() and cross_entropy_error are from ch04/functionsCh4.py
class SoftMaxWithLoss:

    def __init__(self):
        self.loss = None

        # ? output of softmax
        self.y = None

        # ? teaching tags (one-hot vector)
        self.t = None

    
    def forward(self, x, t):
        self.t = t

        # * using implementation of softmax from ch4
        self.y = softmax(x)

        # * using implementation of cross_entropy_error from ch4
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss


    def backward(self, dout = 1):

        # ! IMPORTANT, need divided by batch size
        batchSize = self.t.shape(0)
        dx = (self.y - self.t)/batchSize
        return dx


# * 5. Layers for preventing overfitting
# * 5.1. Dropout (randomly delete neurons during learning)
class Dropout:

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None


    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)


    def backward(self, dout):
        return dout * self.mask


# * 6. Convolution Neural Network layers
# * 6.1. convolution layer
class Convolution:

    # * Parameters
    # * W: filter, 4d (FilterNumber, Channle_number, FilterHeight, FilterWidth)
    # * b: bias, 2d (FilterNumber, 1)
    # * stride: stride
    # * pad: padding
    def __init__(self, W, b, stride=1, pad=0):
        
        # ? weights for filters
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad


    # * Parameters
    # * x: input, 4d (Number, Channel, Height, Width)
    def forward(self, x):
        
        # ? get 4 parameters of filter
        FN, C, FH, FW = self.W.shape

        # ? get 4 parameters of input data
        N, C, H, W = x.shape

        # ? calculate output shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # ! expand input dataset to 2d array (FN*outH*outW, -1)
        col = im2col(x, FH, FW, self.stride, self.pad)
        # ! expand filter to 2d array (-1, FN)
        col_W = self.W.reshape(FN, -1).T

        # ? calculation output
        out = np.dot(col, col_W) + self.b

        # ? transform output to correct shape
        # ? transpose() change dimesion (N, H, W, C) to (N, C, H, W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


# * 6.2. Pooling Layer (Max pooling)
class Pooling:

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (H - self.pool_w) / self.stride)

        # ? step 1. expand input dataset
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # ? step 2. max value of each row
        out = np.max(col, axis=1)

        # ? step 3. transform to correct form for output
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out


    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
