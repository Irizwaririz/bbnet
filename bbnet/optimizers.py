
from bbnet.nn import NeuralNet

class Optimizer:
  def step(self, net: NeuralNet):
    raise NotImplementedError

class GD(Optimizer):
  def __init__(self, lr: float = 0.01):
    self.lr = lr
  
  def step(self, net: NeuralNet):
    """
    step = -gradient * lr
    """
    for layer in net.layers:
      layer.params['w'] -= layer.grads['w'] * self.lr
      layer.params['b'] -= layer.grads['b'] * self.lr