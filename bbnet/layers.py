import numpy as np

from numpy import ndarray
from typing import Dict, Callable

class Layer:
  def __init__(self):
    self.params: Dict[str, ndarray] = {'w': 0, 'b': 0}
    self.grads: Dict[str, ndarray] = {'w': 0, 'b': 0}

  def forward(self, inputs: ndarray) -> ndarray:
    raise NotImplementedError
  
  def backward(self, grad: ndarray) -> ndarray:
    raise NotImplementedError

class Linear(Layer):
  def __init__(self, input_size: int, output_size: int):
    super().__init__()

    """
    inputs = <batch_size by input_size>
    outputs = <batch_size by output_size>    
    weights = <input_size by output_size>
    biases = <1 by output_size>
    """

    self.params['w'] = np.random.randn(input_size, output_size)
    self.params['b'] = np.random.randn(1, output_size)

  def forward(self, inputs: ndarray) -> ndarray:
    """
    outputs = inputs @ w + b
    """

    self.inputs = inputs

    return inputs @ self.params["w"] + self.params["b"]

  def backward(self, grad: ndarray) -> ndarray:
    """
    in -> linear neuron -> out -> f() -> ... -> y

    if
    out = in * w + b and we know dy/dout (grad)

    and:
    dout/din = w
    dout/dw = in
    dout/db = 1

    therefore:
    dy/din = dy/dout * dout/din (chain rule)
          = grad * w
    dy/dw = dy/dout * dout/db (chain rule)
          = grad * in
    dy/db = dy/dout * dout/db (chain rule)
          = grad * 1
          = grad

    if
    out = in @ w + b and we know dy/dout (grad) <batch_size by output_size>

    and:
    dout/din = w <input_size by output_size>
    dout/dw = in <batch_size by input_size>
    dout/db = 1 <1 by output_size>

    therefore:
    dy/din = dy/dout * dout/din (chain rule) <batch_size by input_size>
          = grad <batch_size by output_size> @ w.T (output_size by input_size)
    dy/dw = dy/dout * dout/dw (chain rule) <input_size by output_size>
          = in.T <input_size by batch_size> @ grad <batch_size by output_size> 
    dy/db = dy/dout * dout/db (chain rule)
          = grad * 1
          = np.sum(grad <batch_size by output_size>, axis=0) <1 by output_size>
    """

    self.grads['w'] = (self.inputs.T @ grad) / len(self.inputs)
    self.grads['b'] = np.sum(grad, axis=0) / len(self.inputs)

    return grad @ self.params['w'].T


F = Callable[[ndarray], ndarray]

class Activation(Layer):
  def __init__(self, f: F, f_prime: F):
    super().__init__()
    
    self.f = f
    self.f_prime = f_prime

  def forward(self, inputs: ndarray) -> ndarray:
    self.inputs = inputs

    return self.f(inputs)
  
  def backward(self, grad: ndarray) -> ndarray:
    """
    in -> activation function -> out -> f() -> ... -> y

    if
    out = σ(in) and we know dy/dout (grad) <batch_size by output_size of previous linear layer>

    and:
    dout/din = f_prime(inputs) <batch_size by output_size of previous linear layer>

    therefore:
    dy/din = dy/dout * dout/din (chain rule)
          = grad * f_prime(inputs)
    """
    return grad * self.f_prime(self.inputs)

def tanh(inputs: ndarray) -> ndarray:
  return np.tanh(inputs)

def tanh_prime(inputs: ndarray) -> ndarray:
  return 1 - np.tanh(inputs) ** 2

class Tanh(Activation):
  def __init__(self):
    super().__init__(tanh, tanh_prime)