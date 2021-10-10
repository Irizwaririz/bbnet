"""
Optimizers are the ones who are going to adjust the
parameters of the neural network according to the
gradients computed during backpropagation. Different
optimizers adjust the parameters differently
"""


from bbnet.nn import NeuralNet


class Optimizer:
  def step(self, net: NeuralNet) -> None:
    raise NotImplementedError

class SGD(Optimizer):
  def __init__(self, lr: float) -> None:
    self.lr = lr

  def step(self, net: NeuralNet) -> None:
    for param, grad in net.params_and_grads():
      param -= self.lr * grad