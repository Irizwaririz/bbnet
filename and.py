import numpy as np

from bbnet.train import train
from bbnet.nn import NeuralNet
from bbnet.layers import Linear, Tanh

from bbnet.data import Batch, Stochastic, MiniBatchStochastic

inputs = np.array([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
])

targets = np.array([
  [0],
  [0],
  [0],
  [1],
])

net = NeuralNet([
  Linear(input_size=2, output_size=1)
])

train(inputs=inputs, targets=targets, net=net, iterator=Stochastic(), num_epochs=1000)

print("raw predictions")
for single_input, single_target in zip(inputs, targets):
    prediction = net.forward(single_input)
    print(single_input, prediction, single_target)
    
print("rounded predictions")
for single_input, single_target in zip(inputs, targets):
    prediction = net.forward(single_input)
    print(single_input, [round(item) for item in prediction[0]], single_target)