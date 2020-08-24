# * Example of buying apples and oranges

#%%

import numpy as np
from exampleLayers import MulLayer, AddLayer

applePrice = 100
appleNumber = 2
orangePrice = 150
orangeNumber = 3
tax = 1.1

# * layers
mulAppleLayer = MulLayer()
mulOrangeLayer = MulLayer()
addFruitLayer = AddLayer()
mulTaxLayer = MulLayer()

# * forward
appleTotalPrice = mulAppleLayer.forward(applePrice, appleNumber)
orangeTotalPrice = mulOrangeLayer.forward(orangePrice, orangeNumber)
totalPrice = addFruitLayer.forward(appleTotalPrice, orangeTotalPrice)
finalPrice = mulTaxLayer.forward(totalPrice, tax)

# * backward
dFinalPrice = 1
dTotalPrice, dTax = mulTaxLayer.backward(dFinalPrice)
dAppleTotalPrice, dOrangeTotalPrice = addFruitLayer.backward(dTotalPrice)
dOrangePrice, dOrangeNumber = mulOrangeLayer.backward(dOrangeTotalPrice)
dApplePrice, dAppleNumber = mulAppleLayer.backward(dAppleTotalPrice)

print(finalPrice)
print(dAppleNumber, dApplePrice, dOrangePrice, dOrangeNumber, dTax)
