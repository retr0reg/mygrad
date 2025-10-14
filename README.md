# Mygrad

hand replicating a neural network, back propagation from chain rule according to Andrej Karpathy micrograd with neovim, great way to learn neural network under the hood, while practising my calculus (shoutout to 3b1b) and vim skills.

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
