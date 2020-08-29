#

import numpy as np 
import sys, os

sys.path.append(os.pardir)

from functionsCh4 import softmax, cross_entropy_error

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




