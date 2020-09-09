#
# * Example of numerical differentiation

#%%
import numpy as np 
import matplotlib.pylab as plt
from functionsCh4 import numerical_diff

def function1(x):
    return 0.01*x**2 + 0.1*x



x = np.arange(-5.0, 25.0, 0.1)
y = function1(x)

x1 = 10
diffAt10 = numerical_diff(function1, x1)
interceptionDiffAt10 = function1(x1) - diffAt10 * x1

x2 = 5
diffAt5 = numerical_diff(function1, x2)
interceptionDiffAt5 = function1(x2) - diffAt5 * x2

def tagLineOfFunc1At10(x):
    return x * diffAt10 + interceptionDiffAt10


def tagLineOfFunc1At5(x):
    return x * diffAt5 + interceptionDiffAt5


yt10 = tagLineOfFunc1At10(x)
yt5 = tagLineOfFunc1At5(x)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(-10, 30)
plt.plot(x, y, label='f(x) = 0.01*x^2 + 0.1*x')
plt.plot(x, yt10, linestyle='--', label='tagent line at 10')
plt.plot(x, yt5, linestyle='dotted', label='tagent line at 5')
plt.legend();
plt.show()


# %%
