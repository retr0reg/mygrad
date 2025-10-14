import random
from typing import List

from value import Value

class Neuron:
    def __init__(self, n_input: int):
        self.weights: list[Value] = [Value(random.uniform(-1,1)) for _ in range(n_input)]
        self.bias: Value = Value(random.uniform(-1,.1))

    def __call__(self, input: list[Value]) -> Value:

        # input * weight + bias
        # input[0] * weight[0] ---\
        #                         ----- + bias
        # input[n] * weight[n] ---/

        activation = sum(ii * wi for ii, wi in zip(input, self.weights)) + self.bias
        return activation.tanh()

    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, n_input, n_output):
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, input: list[Value]) -> Value | list[Value]:
        output = [neuron(input) for neuron in self.neurons]
        return output if len(output) != 1 else output[0]

    def parameter(self):
        # return [param for neuron in self.neurons for p in neuron.parameters()]
        params = []
        for neuron in self.neurons:
            paramaters = neuron.parameters()
            params.extend(paramaters)
        return params

class MLP:
    def __init__(self, n_input, *n_neurons):
        # n_input, n_output[-] -> n_output[1], n_output[2]
        layers_sz = [n_input] + list(n_neurons)
        self.layers: list = [Layer(layers_sz[i], layers_sz[i+1]) for i in range(len(layers_sz)-1)]

    def __call__(self, input) -> Value:
        for layer in self.layers:
            input = layer(input) # current layer output
        return input

mlp = MLP(4, 4, 4, 1)

xs = [
    [1.0, 0.5, 1.0, 0.5],
    [-1.0, -1.0, -1.0, -1.0],
    [1.0, -0.5, 1.0, -0.5],
    [1.0, 0.5, 1.0, -0.5],
]

ys = [0.5, -1.0, 0.2, 0.8]

yprev: List[Value] = [mlp(x) for x in xs]
print(f"yprev={yprev}")

loss: Value = sum(((yout-ygt)**2 for ygt, yout in zip(ys, yprev)), Value(0)) # for the sake of lint # pyright: ignore[]
print(f"loss={loss}")

#backwarrrding
loss.backward()
print(mlp.layers[0].neurons[0].weights[0].grad)
