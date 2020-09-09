
# * a simple 3-layer neural network

#%%
import numpy as np 
import drawStepFunction as func

# * helper 
def identity_function(x):
    return x


# * layer 1
X =np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(A1)

Z1 = func.sigmoid(A1)
print(Z1)

# * layer 2
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
print(A2)

Z2 = func.sigmoid(A2)
print(Z2)

# * layer 3
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0])

print(Z2.shape)
print(W3.shape)
print(B3.shape)

A3 = np.dot(Z2, W3) + B3
print(A3)

Y = identity_function(A3)
print(Y)

# %%
