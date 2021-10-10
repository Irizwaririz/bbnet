"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np

from bbnet.train import train
from bbnet.nn import NeuralNet
from bbnet.layers import Linear, Tanh

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net=net, inputs=inputs, targets=targets)

print("raw predictions")
for input, target in zip(inputs, targets):
    prediction = net.forward(input)
    print(input, prediction, target)
    
print("rounded predictions")
for input, target in zip(inputs, targets):
    prediction = net.forward(input)
    print(input, [round(item) for item in prediction], target)