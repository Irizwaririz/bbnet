
from numpy import ndarray
from bbnet.nn import NeuralNet
from bbnet.data import DataIterator, MiniBatchStochastic
from bbnet.loss import Loss, TSE
from bbnet.optimizers import Optimizer, GD

def train(
  net: NeuralNet,
  inputs: ndarray,
  targets: ndarray,
  iterator: DataIterator = MiniBatchStochastic(),
  loss: Loss = TSE(),
  optimizer: Optimizer = GD(),
  num_epochs: int = 5000
):

  for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_inputs, batch_targets in iterator(inputs, targets):
      print("train")
      print(batch_inputs, batch_targets)

    for batch_inputs, batch_targets in iterator(inputs, targets):
      batch_predictions = net.forward(batch_inputs)

      print("batch_predictions", batch_predictions)

      epoch_loss += loss.loss(batch_predictions, batch_targets)
      batch_grad = loss.loss_prime(batch_predictions, batch_targets)

      net.backward(batch_grad)

      optimizer.step(net)
    
    print(epoch, epoch_loss)


