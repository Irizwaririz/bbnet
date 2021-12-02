import numpy as np

from numpy import ndarray
from bbnet.nn import NeuralNet
from bbnet.loss import Loss, TSE
from bbnet.optimizers import Optimizer, GD

def train(
  net: NeuralNet,
  inputs: ndarray,
  targets: ndarray,
  loss: Loss = TSE(),
  optimizer: Optimizer = GD(),
  num_epochs: int = 5000,
  is_shuffle: bool = True
) -> None:

  inputs_clone = np.copy(inputs)
  for epoch in range(num_epochs):
    if(is_shuffle):
      np.random.shuffle(inputs_clone)

    predictions = net.forward(inputs)
    epoch_loss = loss.loss(predictions, targets)

    grad = loss.grad(predictions, targets)
    net.backward(grad)

    optimizer.step(net)

    print(epoch, epoch_loss)


