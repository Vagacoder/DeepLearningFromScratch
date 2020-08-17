#%%
import numpy as np


# * implementatin 1, parameter x is only for float (real number), not for numpy array
def step_function1(x):
    if x > 0:
        return 1
    else:
        return 0

# * implementation 2, parameter x can be numpy array
def step_function2(x):
    y = x > 0
    return y.astype(np.int)
