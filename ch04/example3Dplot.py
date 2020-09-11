#
# * Example, a 3D plot 

#%%
import numpy as np
import matplotlib.pyplot as plt 

# ! NOT working
x1 = np.arange(-2.0, 2.0, 0.1)
x2 = np.arange(-2.0, 2.0, 0.1)

y = x1**2 + x2**2

print(y)

plt.plot(x1, x2, y)
plt.show()
# %%
