import math # I would've just use pytorch for some leeyways

class Value:
    def __init__(self, value, _op='', _child=()):
        self.value = value
        self._op = _op
        self._backward = lambda: None
        self._child = set(_child)
        self.grad = 0.0

    def backward(self):
        self._backward()
        # print(f"from {self}", end=" ")
        for child in self._child:
            # print(f"propagating {child}")
            if child._backward: child.backward()

    def __repr__(self):
        return f"Value({self.value}"

    def __add__(self, target):
        v = Value(self.value+target.value, _op='+', _child=(self, target))
        def _backward():
            self.grad = 1.0 * v.grad
            target.grad = 1.0 * v.grad
        v._backward = _backward
        return v

    def __mul__(self, target):
        v = Value(self.value*target.value, _op='*', _child=(self, target))
        def _backward():
            self.grad = target.value * v.grad
            target.grad = self.value * v.grad
        v._backward = _backward
        return v

    def tanh(self):
        exp = (math.exp(2*self.value)-1)/(math.exp(2*self.value)+1)
        v = Value(exp, _op="tanh", _child=(self, ))
        def _backward():
            self.grad = (1 - exp**2) * v.grad
        v._backward = _backward
        return v



if __name__ == "__main__":
    # inputs
    x1 = Value(2.0)
    x2 = Value(1.0)
    
    # weights
    w1 = Value(-3.0)
    w2 = Value(1.0)

    # bias
    b = Value(6.88137)

    x1w1 = x1*w1
    x2w2 = x2*w2

    x1w1x2w2 = x1w1*x2w2
    n = x1w1x2w2 + b
    o = n.tanh();o.grad=1.0
    
    print(o.grad)
    o.backward()
    print(x1.grad)
