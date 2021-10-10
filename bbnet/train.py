from numpy import ndarray
from bbnet.nn import NeuralNet
from bbnet.loss import Loss, TSE
from bbnet.data import DataBatcher, BasicBatcher
from bbnet.optimizers import Optimizer, SGD


def train(
  inputs: ndarray,
  targets: ndarray,
  net: NeuralNet,
  loss: Loss = TSE(),
  batcher: DataBatcher = BasicBatcher(),
  optimizer: Optimizer = SGD(lr = 0.01),
  num_epochs: int = 5000
) -> None:
  for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_input, batch_target in batcher(inputs, targets):
      batch_prediction = net.forward(batch_input)

      epoch_loss += loss.loss(batch_prediction, batch_target)
      batch_grad = loss.grad(batch_prediction, batch_target)

      net.backward(batch_grad)

      optimizer.step(net)
    print(epoch, epoch_loss)