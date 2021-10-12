
from bbnet.nn import NeuralNet

class Optimizer:
  def step(self, net: NeuralNet) -> None:
    raise NotImplementedError

class GD(Optimizer):
  def __init__(self, lr: float = 0.01) -> None:
    self.lr = lr

  def step(self, net: NeuralNet) -> None:
    for param, grad in net.params_and_grads():
      param -= grad * self.lr