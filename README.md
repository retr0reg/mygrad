# nn

<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/70d07d99-e5d4-417c-a66d-c17ef3773b44" />

hand replicating a neural network, back propagation from chain rule according to Andrej Karpathy micrograd with neovim, great way to learn neural network under the hood, while practising my calculus (shoutout to 3b1b) and vim skills.

All of this project is made up from almost two thing:

1. `Value` - a wrapper for a numerical values, with support of finding derivative and gradient (chain-rule aka backprop)
2. `Neurons`, `Layers`, `MLP` - wrapper for Value & backprop

For now, since it's only a scalar value nn, it might be a good idea to implement `Tensor` etc, might be a good direction to hover on in the future *(thought pretty much all we spent weeks on can be done by lines of code @pytorch.)*, the pain in the butt is you'll have to implement both forward and backward (differentiating) operations for `Tensor` *(which I need to learn a little bit about, I mean I could just ChatGPT everything, but what's the fun of that?)*

## Problem that I had implementing myself w/ heuristics

1. Why not recursion for back propagation? why is topological sort?
```python
    def backward(self):
        visited = set()
        self._backward()
        for child in self._child:
            if child not in visited:
                if child._backward: print(f"{self.label} backward to {child.label}"); child.backward(); visited.add(child)

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
```
*Answer*: people call it dependency issue, but what it is that in our case b's gradient would be re-used, which during $dd/dc * dc/db$ is $da * da$, where the changes of $dc$ here would be assisted by $da$. But for $dd/db$, $db$'s changes on $dd$ would only be it's derivative $da$. How I think why we use topological here is simply just reversing the input-to-out put process to make sure that we make sure a derivative's derivative wouldn't be misinterpretered, for example $b$ here is changes both $c$ and $d$, we don't want to jump steps  

2. zero-grad problem

this gradient descending sees working right?
```python
class MLP:
    def __init__(self, n_input, *n_neurons):
        # n_input, n_output[-] -> n_output[1], n_output[2]
        layers_sz = [n_input] + list(n_neurons)
        self.layers: list[Layer] = [Layer(layers_sz[i], layers_sz[i+1]) for i in range(len(layers_sz)-1)]

    def __call__(self, input: list[Value]) -> Value | list[Value]:
        for layer in self.layers:
            input = layer(input) #type: ignore
        return input

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

mlp = MLP(4, 4, 4, 1)

xs = [
    [1.0, 0.5, 1.0, 0.5],
    [-1.0, -1.0, -1.0, -1.0],
    [1.0, -0.5, 1.0, -0.5],
    [1.0, 0.5, 1.0, -0.5],
]

ys = [0.5, -1.0, 0.2, 0.8]

def epoch(i):

    # forward passing
    yprev: List[Value] = [mlp(x) for x in xs] #type: ignore
    loss: Value = sum(((yout-ygt)**2 for ygt, yout in zip(ys, yprev)), Value(0)) # for the sake of lint # pyright: ignore[]

    # back prop
    loss.backward()

    # update
    for param in mlp.parameters():
        param.value += -0.01 * param.grad # we're minimizing the instead of increasing it 

    print(i, loss.value)

for i in range(20):
    epoch(i) 

```

it does work, however not very effciently. If you change epoch to 200 you will find that the nn is only optimizing from a very slow rate, the reason why is didn't set the parameters gradients back to zero, since supposely after updates, weight and biases change, and they're suppose to be zero.
The reason why this matters, is because we are finding gradient in a multi-variable scenario *(we accumulated gradient since one node might contribute to multiple nodes)* and that effect the child nodes on backprop, simple question, simple solution, just reset parameters.grad to zero. don't worry about the loss function node, too. we set backward obj's gradient to one before backprop.

```python
for param in mlp.parameters():
    param.grad = 0
```
