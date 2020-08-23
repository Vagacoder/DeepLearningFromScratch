#

# * Multiplication layer
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None


    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out


    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx,dy


# * Addition layer
class AddLayer:
    def __init__(self):
        pass


    def forward(self, x, y):
        return x + y
    

    def backward(self, dout):
        return (dout * 1), (dout * 1)


