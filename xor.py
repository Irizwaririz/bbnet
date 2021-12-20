import numpy as np

from bbnet.nn import NeuralNetwork
from bbnet.layers import Linear, Tanh
from bbnet.train import train
from bbnet.optimizers import GD

inputs = np.array([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
])

targets = np.array([
  [0],
  [1],
  [1],
  [0]
])

net = NeuralNetwork([
  Linear(input_size=2, output_size=2),
  Tanh(),
  Linear(input_size=2, output_size=1)
])

train(net=net, inputs=inputs, targets=targets, optimizer=GD(lr=0.5))

print("Raw predictions")
for single_input, single_target in zip(inputs, targets):
  prediction = net.forward(single_input)
  print(single_input, prediction, single_target)

print("Clean predictions")
for single_input, single_target in zip(inputs, targets):
  prediction = net.forward(single_input)
  print(single_input, [round(single_prediction) for single_prediction in prediction], single_target)