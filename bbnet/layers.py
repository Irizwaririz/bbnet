import numpy as np

from numpy import ndarray
from typing import Dict, Callable

"""
const dictionary = {
  "w": [[..], [...], ...],
  "b": [...]
}
"""

class Layer:
  def __init__(self) -> None:
    self.params: Dict[str, ndarray] = {}
    self.grads: Dict[str, ndarray] = {}

  def forward(self, inputs: ndarray) -> ndarray:
    raise NotImplementedError
  
  def backward(self, grad: ndarray) -> ndarray:
    raise NotImplementedError

class Linear(Layer):
  def __init__(self, input_size: int, output_size: int) -> None:
    # outputs = inputs @ weights + biases
    #
    # Shape of our matrices:
    # inputs -> # of data x input_size
    # outputs -> # of data x output_size
    # weights -> input_size x output_size
    # biases -> # of data x output_size (1 x output_size)

    # Initialize our weights and biases to random values
    super().__init__()
    self.params["w"] = np.random.randn(input_size, output_size)
    self.params["b"] = np.random.randn(output_size)
  
  def forward(self, inputs: ndarray) -> ndarray:
    self.inputs = inputs

    return inputs @ self.params["w"] + self.params["b"]

  def backward(self, grad: ndarray) -> ndarray:
    """
    if y = f(x) and x = a @ w + b, then
    dy/da = f'(x) @ w.T
    dy/dw = a.T @ f'(x)
    dy/db = f'(x) summed across its rows
    """
    self.grads["w"] = self.inputs.T @ grad
    self.grads["b"] = np.sum(grad, axis=0)
    return grad @ self.params["w"].T



class Activation(Layer):
  def __init__(self, f: Callable[[ndarray], ndarray], f_prime: Callable[[ndarray], ndarray]) -> None:
    super().__init__()
    self.f = f
    self.f_prime = f_prime
  
  def forward(self, inputs: ndarray) -> ndarray:
    self.inputs = inputs
    return self.f(inputs)

  def backward(self, grad: ndarray) -> ndarray:
    """
    if y = f(x) and x = g(z), then
    dy/dz = f'(x) * g'(z)
    """
    return grad * self.f_prime(self.inputs)


def tanh(inputs: ndarray) -> ndarray:
  return np.tanh(inputs)

def tanh_prime(inputs: ndarray) -> ndarray:
  return 1 - np.tanh(inputs) ** 2

class Tanh(Activation):
  def __init__(self) -> None:
    super().__init__(tanh, tanh_prime)
