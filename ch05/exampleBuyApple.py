#

#%%

import numpy as np
import sys, os
sys.path.append(os.pardir)
from exampleLayers import MulLayer

applePrice = 100
appleNumber = 2
tax = 1.1

# * only 2 layers, both are multiplication: ( apple # * apple price ) * taxrate
appleMulLayer = MulLayer()
taxMulLayer = MulLayer()

# * forward
appleTotalPrice = appleMulLayer.forward(applePrice, appleNumber)
totalPrice = taxMulLayer.forward(appleTotalPrice, tax)

print(totalPrice)

# * backward, call layers in the reversed order
dPrice = 1
dAppleTotalPrice, dTax = taxMulLayer.backward(dPrice)
dApplePrice, dAppleNumber = appleMulLayer.backward(dAppleTotalPrice)

print(dApplePrice, dAppleNumber, dTax)


# %%
