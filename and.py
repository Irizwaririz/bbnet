import numpy as np

from bbnet.nn import NeuralNet
from bbnet.layers import Linear, Tanh
from bbnet.train import train

inputs = np.array([
  [0, 0],
  [0, 1],
  [1, 0],
  [1 ,1]
])

targets = np.array([
  [0],
  [0],
  [0],
  [1]
])

net = NeuralNet([
  Linear(input_size=2, output_size=1)
])

train(net=net, inputs=inputs, targets=targets)

for input, target in zip(inputs, targets):
  prediction = net.forward(input)
  print(input, prediction, target)

