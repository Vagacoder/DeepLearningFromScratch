#
# * Example of gradient calculation

#%%
import numpy as np 
from functionsCh4 import numerical_gradient, gradient_descent

# * f(x) = x1*x1 + x2*x2 + ... + xn*xn
def function2(x):
    return np.sum(x**2)


print(numerical_gradient(function2, np.array([3.0, 4.0])))

print(numerical_gradient(function2, np.array([0.0, 2.0])))

print(numerical_gradient(function2, np.array([3.0, 0.0])))


init_x = np.array([-3.0, 4.0])
print('Learning rate = 0.1')
print(
    gradient_descent(function2, init_x, 0.1, 100)
)

print('Learning rate = 10, too large')
print(
    gradient_descent(function2, init_x, 10, 100)
)

print('Learning rate = 1e-10, too small')
print(
    gradient_descent(function2, init_x, 1e-10, 100)
)