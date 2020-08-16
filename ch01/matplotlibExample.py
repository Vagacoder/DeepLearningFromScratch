#%%
import numpy as np
import matplotlib.pyplot as plt

# * draw sin and cos using matplotlib, with legend and labels
x = np.arange(0, 6, 0.1)
y = np.sin(x)

print(x)
print(y)
plt.plot(x, y)
plt.show()

y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle='--', label='cos')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin & cos')
plt.legend()
plt.show()

print('Done!')
