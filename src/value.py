import math # I would've just use pytorch for some leeyways

class Value:
    def __init__(self, value, label='', _op='', _child=()):
        self.value = value
        self._op = _op
        self._backward = lambda: None
        self._child = set(_child)
        self.grad = 0.0
        
        self.label = label # I don't graph it out, but still good to label

    def backward(self):

        self.grad = 1.0

        io = [] # input-to-output
        visited = set()

        def topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._child:
                    topo(child)
                io.append(v)

        topo(self)
        oi = io[::-1] 

        for node in oi:
            node._backward()

        # for child in self._child:
            # if child not in visited:
            #     if child._backward: print(f"{self.label} backward to {child.label}"); child.backward(); visited.add(child)
        # note here the topological sorting accordingly matters alot, I didn't understand why at the begining
        # I began first propagates the gradient only with = (equals), which I understood is wrong because I assume each Value only serves for one dy, thus I changed it to +=
        # but why do we have to put it in topological order, where we eventually'd have the same gradient accumulated just w/ different order?
        # my explaination
        # given: a = Value(1.0); b = a * a; c = b * a; d = c * a; and we propagates from d to find dd/da (d.gradient = 1)
        # 1. c.gradient += a; a.gradient += a (propagates to c)
        # 2     b.gradient += a*a; a.gradient += b*a (propagates to b)
        # 3.        a.grad = a... okay this is bad example the gradient of a would be correct either way, except LLMs says it would repeative call a._backward...
        # Let's set a visited flag any way since it's true that b = a * a, a would be repeatively accumulated where gradient of a would become 2 a.gradient

        # Anyway, I think for about another 20min, the issue here should be DFS issue; for example we have b = a + a, c = b * a, d = c + b:
        # if we all go left hand side:
        # bp from d: c.g=1(c) -> **bp from c** -> b.g += 1(c) * a, a = 1(c) * b -> bg from b -> a += b
            # b = 1 * a * 1
        # buf if we start from right hand side, bp to b first;
            # ...

      
            # a = Value(2.0); a.label='a'
            # b = a + a; b.label='b' # 3
            # c = a * b; c.label='c' # 2
            # d = c + b; d.label='d' # 1

            # d.grad = 1.0
            # d.backward()
            # print(a.grad) # 16.

            # a = Value(2.0); a.label='a'
            # b = a + a; b.label='b' # 
            # c = a * b; c.label='c' # 
            # d = b + c; d.label='d' # 

            # d.grad = 1.0
            # d.backward()
            # print(a.grad) # 12.
        
        # what i thought (false), because a.value doesn't changes
            # when b was back-propagated: (*importance on a.value*)
            # 1. [d = b + c] b.grad=d.grad=1 -> a+=b.grad=1;a+=b.grad=1. 
            # 2. [c = a * b] b.grad= a.value * c.grad=d.grad -> a==(2*b.grad)[a.grad from last step] = +=2*b.grad; += 2*b.grad = 4*b.grad (+= 6 b.grad already)
            # 3. while a += b.value*c.grad +=1 
            
            # 1. [c = b * a] b.grad=a.value * c.grad=d.grad=1 = 0. a+=0 +=0 (a.value haven't change)
            # 2. [d = c + b] b.grad=

        # oh, the issue here is b being backward twice
        # top:
            # 1st b.backward, b.g=3 (because c.backwarded)
            # 2nd b.backward, b.g=3
            # 3+3 + 3+3 + 4 (c) = 16
        # down
            # 1st b.backward, b.g=1
            # 2nd b.backward, b.g=3
            # 1+1 + 3+3 + 4 (c) = 12

        # Here either is wrong, or even if b is propa
        # Following topological order is *how is input being changed,* we start from a -> d (a->b->c->d) then reverse it for propagation
        # the problem is if you look at a, it's not fully 

    def __repr__(self):
        return f"Value(label={self.label}, value={self.value}"

    def __add__(self, target):
        target = target if isinstance(target, Value) else Value(target)
        v = Value(self.value+target.value, _op='+', _child=(self, target))
        def _backward():
            # if you figure the topological thing out, you definitely know why we're accumulating gradients
            # also karpathy mentioned multivariable case of chain rule, also why you accumulate
            self.grad += 1.0 * v.grad
            target.grad += 1.0 * v.grad
        v._backward = _backward
        return v

    def __mul__(self, target):
        target = target if isinstance(target, Value) else Value(target)
        v = Value(self.value*target.value, _op='*', _child=(self, target))
        def _backward():
            self.grad += target.value * v.grad
            target.grad += self.value * v.grad
        v._backward = _backward
        return v

    def __radd__(self, target): # target _op Value
        return self + target

    def __rmul__(self, target):
        return self * target

    def __neg__(self):
        return self * -1
    
    def __sub__(self, target):
        return self + (-target)

    def __rsub__(self, target):
        return (-self) + target

    def __pow__(self, target):
        # target = target if isinstance(target, Value) else Value(target)
        assert isinstance(target, (int, float)), "Andrej said we're only supporting none-value obj here, or it might mess with the backprop, fine"
        v = Value(self.value**target, _op="^", _child=(self, ))
        def _backward():
            self.grad += target * (self.value**(target-1)) * v.grad
            # self.grad += target.value*(self.value**(target.value-1)) * v.grad
            # target.grad += (self.value**target.value) * math.log(self.value) * v.grad
        v._backward = _backward 
        return v

    def __truediv__(self, target):
        return self * target**-1

    def __rtruediv__(self, target):
        return self**-1 * target

    def exp(self):
        v = Value(math.exp(self.value), _child=(self, ), _op='exp')
        def _backward():
            self.grad += v.value * v.grad # derivative of e^x is just e^x, and we just calculated e^x
        v._backward = _backward
        return v

    def tanh(self):
        exp = (math.exp(2*self.value)-1)/(math.exp(2*self.value)+1)
        v = Value(exp, _op="tanh", _child=(self, ))
        def _backward():
            self.grad += (1 - exp**2) * v.grad
        v._backward = _backward
        return v

    def relu(self):
        v = Value(max(0, self.value), _child=(self, ), _op="ReLU")
        def _backward():
            self.grad += (1 if v.value > 0 else 0) * v.grad
        v._backward = _backward
        return v

if __name__ == "__main__":

    # input
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

    #tanh
    e = (2*n).exp()
    o = (e - 1) / (e + 1); 

    print(o.grad)
    o.backward()
    print(x1.grad)
