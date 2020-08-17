#

# * AND gate
def AND(x1, x2):
    w1 = 0.5
    w2 = 0.5
    theta = 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


# * NAND gate
def NAND(x1, x2):
    w1 = -0.5
    w2 = -0.5
    theta = -0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


# * OR gate
def OR(x1, x2):
    w1 = 1.1
    w2 = 1.1
    theta = 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


# * XOR gate 
def XOR(x1, x2):
    w1 = 1.0
    w2 = -1.0
    theta = 0.0
    tmp = x1*w1 + x2*w2
    if tmp == theta:
        return 0
    elif tmp != theta:
        return 1


print('1. testing AND')
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

print('\n2. testing NAND')
print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))

print('\n3. testing OR')
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))

print('\n4. testing XOR')
print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))