"""
Optimizers are the ones who are going to adjust the
parameters of the neural network according to the
gradients computed during backpropagation. Different
optimizers adjust the parameters in different ways
"""
from bbnet.nn import NeuralNet

class Optimizer:
  def step(self, net: NeuralNet):
    raise NotImplementedError

class GD(Optimizer):
  def __init__(self, lr: float = 0.01):
    self.lr = lr
  
  def step(self, net: NeuralNet):
    """
    A Gradient Descent Step is equal to the Negative Gradient multiplied to the Learning Rate:
    step = -gradient * lr
    """
    for layer in net.layers:
      layer.params['w'] -= layer.grads['w'] * self.lr
      layer.params['b'] -= layer.grads['b'] * self.lr