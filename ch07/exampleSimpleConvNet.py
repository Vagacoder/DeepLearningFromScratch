#

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


# * Simple Convolutional Neural Network
class SimpleConvNet:

    # * Structure:
    # * conv - relu - pool --- affine - relu --- affine - softmax

    # * Constructor, check carefully
    # * Parameters
    # ? input_dim: input dataset size (Channel, Height, Width), 
    # ?             default for MNIST dataset (1, 28, 28)
    # ? conv_param: convolutional layer parameters:
    # ?     filter_num: number of filter
    # ?     filter_size: size of filter (Height = Width)
    # ?     pad: size of padding (0)
    # ?     stride: size of stride
    # ? hidden_size: the number of neuron in hidden layer, e.g. [100, 50, 100]
    # ? output_size: the size of output, e.g. MNITST output size is 10
    # ? weight_init_std: the standard deviation of weight initial values, e.g. 0.01
    # ?             When activation function is 'Relu', using He value: sqrt(2/n)
    # ?             When activation function is 'Sigmoid', using Xavier value: 1/sqrt(1)
    # ? activation (optional, not implemented yet): activation function, 'Relu' or 'Sigmoid'
    def __init__(self, 
                input_dim=(1, 28, 28),
                conv_param={
                    'filter_num':30, 
                    'filter_size': 5, 
                    'pad':0, 
                    'stride': 1},
                hidden_size=100, 
                output_size=10, 
                weight_init_std=0.01
                ):

        # ? set filter parameters
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']

        # ? get input sizee (assuming Height == Width, take either of them)
        input_size = input_dim[1]
        
        # ? Calculate Convolutional layer output shape
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1

        # ? Calculate Pooling layer output shape
        # ? Max pooling (2x2) using stride of 2. 
        # * Conv -output-> Relu -output-> Pool, input size is conv_output_size
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))


        # * Initialize Weights and Bias for each layer. Picture 7-13
        self.params = {}
        # * for layer #1 of conv - relu - pool
        # ? W1: weight for filter (4d): FilterNumber, Channel_number, FilterHeight, FilterWidth
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        # ? b1: bias for filter (3d): FilterNumber, 1, 1
        self.params['b1'] = np.zeros(filter_num)

        # * for layer #2 of affine - relu
        # ? W2: weight for affine (2d): output (from layer #1 pooling) size, neuron number(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        # ? b2: bias for affine (1d): neuron number (hidden_size)               
        self.params['b2'] = np.zeros(hidden_size)

        # * for layer #3 of affine - softmax
        # ? W3: weight for affine (2d): output (from layer #2) size, final output number
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        # ? b3: bias for affine (1d): final output number              
        self.params['b3'] = np.zeros(output_size)


        # * Generate and assemble layers
        self.layers = OrderedDict()

        # * layer of conv - relu - pool
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # * layer of affine - relu
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()

        # * last layer of affine - softmax
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        
        # ! Note: softemax is added to separated layer, because predict does not need it
        self.last_layer = SoftmaxWithLoss()


    # * For inference, NO NEED softmax and loss
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x


    # * For loss
    # ? Parameters: x: training(input) dataset, t: test dataset
    def loss(self, x, t):
        y = self.predict(x)
        # ! predict + softmax + loss
        return self.last_layer.forward(y, t)


    # * For gradient
    # ? Parameters: x: training(input) dataset, t: test dataset
    def gradient(self, x, t):

        # * forward
        self.loss(x, t)

        # * backward
        # ? backward last layer (softmax + loss)
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # ? Save and return gradients of each layer
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


    # * Numerical gradient, for verification
    # ? Parameters: x: training(input) dataset, t: test dataset
    def numerical_gradient(self, x, t):
        loss_w = lambda w : self.loss(x, t)

        grads = {}

        for index in (1, 2, 3):
            grad['W' + str(index)] = numerical_gradient(loss_w, self.params['W' + str(index)])
            grad['b' + str(index)] = numerical_gradient(loss_w, self.params['b' + str(index)])


    # * Find accuracy
    # ? Parameters: x: training(input) dataset, t: test dataset
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]


    # * Save trained parameters: weights and bias
    def save_params(self, file_name="params.pkl"):
        params = {}

        for key, val in self.params.items():
            params[key] = val
        
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)


    # * Load saved trained parameters: weights and bias
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]


