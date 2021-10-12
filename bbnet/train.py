
from numpy import ndarray
from bbnet.nn import NeuralNet
from bbnet.loss import Loss, TSE
from bbnet.optimizers import Optimizer, GD
from bbnet.data import DataBatcher, BasicBatcher

def train(
  net: NeuralNet,
  inputs: ndarray,
  targets: ndarray,
  loss: Loss = TSE(),
  optimizer: Optimizer = GD(),
  batcher: DataBatcher = BasicBatcher(),
  num_epoch: int = 5000
) -> None:
  for epoch in range(num_epoch):
    epoch_loss = 0.0
    for batch_inputs, batch_targets in batcher(inputs=inputs, targets=targets):
      batch_predictions = net.forward(batch_inputs)
      epoch_loss += loss.loss(predictions=batch_predictions, targets=batch_targets)

      batch_grad = loss.grad(predictions=batch_predictions, targets=batch_targets)
      net.backward(grad=batch_grad)
      optimizer.step(net=net)
    print(epoch, epoch_loss)
    