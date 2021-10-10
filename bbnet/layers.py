"""
Our neural nets will be made up of layers.
Each layer needs to propagate inputs forward
and propagate gradients backward. For example,
a neural net might look like:

inputs -> Linear -> Tanh -> Linear -> output
"""

from typing import Dict, Callable

from numpy import ndarray
import numpy as np

class Layer:
  def __init__(self) -> None:
    self.params: Dict[str, ndarray] = {}
    self.grads: Dict[str, ndarray] = {}

  def forward(self, inputs: ndarray) -> ndarray:
    raise NotImplementedError

  def backward(self, grad: ndarray) -> ndarray:
    raise NotImplementedError

class Linear(Layer):
  """
    The linear layer forward propagates by doing the operation below:

    outputs = inputs @ weights + biases

    Where @ represents a matrix multiplication
    
    The matrix sizes are:

    outputs (batch_size x output_size)
    inputs (batch_size x input_size)
    weights (input_size x output_size)
    biases (1 x output_size)
    """
  def __init__(self, input_size: int, output_size: int) -> None:
    super().__init__()
    self.params["w"] = np.random.rand(input_size, output_size)
    self.params["b"] = np.random.rand(output_size)


  def forward(self, inputs: ndarray) -> ndarray:
    """
    outputs = inputs @ weights + biases
    """
    self.inputs = inputs

    return inputs @ self.params["w"] + self.params["b"]

  def backward(self, grad: ndarray) -> ndarray:
    """
    If y = f(x) and x = a * w + b
    then 
    dy/da = f'(x) * w
    dy/dw = f'(x) * a
    dy/db = f'(x)

    and if y = f(x) and x = a @ w + b
    then
    dy/da = f'(x) @ w.T
    dy/dw = a.T @ f'(x)
    dy/db = sum across rows of f'(x)
    """

    self.grads["w"] = self.inputs.T @ grad
    self.grads["b"] = np.sum(grad, axis=0)

    return grad @ self.params["w"].T

F = Callable[[ndarray], ndarray]

class Activation(Layer):
  """
    An activation layer will forward propagate the inputs using the operation below:

    outputs = activation_function(inputs)

    So it will just apply a function elementwise to the inputs
    
    The matrix sizes are:

    outputs (batch_size x output_size (of next layer))
    inputs (batch_size x input_size (of previous layer))

    But the input size of the next layer must match the output size of the previous layer
    because the inputs for the next layer is the outputs of the previous layer!

    Therefore output_size = input_size
    """
  def __init__(self, f: F, f_prime: F) -> None:
    super().__init__()
    self.f = f
    self.f_prime = f_prime

  def forward(self, inputs: ndarray) -> ndarray:
    self.inputs = inputs

    return self.f(inputs)

  def backward(self, grad: ndarray) -> ndarray:
    """
    If y = f(x) and x = g(z)
    then
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