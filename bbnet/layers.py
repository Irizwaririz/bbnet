"""
Our neural nets will be made up of layers.
Each layer needs to propagate inputs forward
and propagate derivatives ("gradient") backward.
For example, a neural net might look like:

input/s -> Linear -> Tanh -> Linear -> output/s
"""
import numpy as np

from numpy import ndarray
from typing import Dict, Callable

class Layer:
  def __init__(self) -> None:
    self.params: Dict[str, ndarray] = {'w': 0, 'b': 0}
    self.grads: Dict[str, ndarray] = {'w': 0, 'b': 0}

  def forward(self, inputs: ndarray) -> ndarray:
    """
    Forward propagate the inputs to produce the outputs
    """
    raise NotImplementedError
  
  def backward(self, grad: ndarray) -> ndarray:
    """
    Backpropagate the "gradient" through the layer
    """
    raise NotImplementedError

class Linear(Layer):
  """
    The linear layer forward propagates by doing the operation:
    outputs = inputs @ weights + biases

    Where @ represents matrix multiplication
    
    The matrix sizes are:
    inputs <batch_size by input_size>
    outputs <batch_size by output_size>
    weights <input_size by output_size>
    biases <1 by output_size>
    """
  def __init__(self, input_size: int, output_size: int):
    super().__init__()
    
    self.params['w'] = np.random.randn(input_size, output_size)
    self.params['b'] = np.random.randn(1, output_size)

  def forward(self, inputs: ndarray) -> ndarray:
    """
    outputs = inputs @ w + b

    Even if b is <1 by output_size>, python still adds the bias for each batch (for each row).
    So it's equivalent to a <batch_size by output_size> matrix with repeating rows
    """

    self.inputs = inputs

    return inputs @ self.params["w"] + self.params["b"]

  def backward(self, grad: ndarray) -> ndarray:
    """
    Given:
    in -> linear neuron -> out -> f() -> ... -> y

    If:
    out = in * w + b and we know dy/dout (grad)

    Then:
    dout/din = w
    dout/dw = in
    dout/db = 1

    Therefore:
    dy/din = dy/dout * dout/din (chain rule)
           = grad * w
    dy/dw = dy/dout * dout/db (chain rule)
          = grad * in
    dy/db = dy/dout * dout/db (chain rule)
          = grad * 1
          = grad

    But if:
    out = in @ w + b and we know dy/dout (grad) <batch_size by output_size>

    Then:
    dout/din = w.T <output_size by input_size>
    dout/dw = in.T <input_size by batch_size>
    dout/db = 1 <batch_size by output_size> with all values as 1

    Therefore:
    dy/din = dy/dout @ dout/din (chain rule) <batch_size by input_size>
           = grad <batch_size by output_size> @ w.T <output_size by input_size>
    dy/dw = dout/dw @ dy/dout (chain rule) <input_size by output_size> (gradients of each batch are summed up)
          = in.T <input_size by batch_size> @ grad <batch_size by output_size> 
    dy/db = dy/dout * dout/db (chain rule) <1 by output_size> (sum up gradients of each batch)
          = np.sum(grad * 1 <batch_size by output_size>, axis=0) <1 by output_size>
          Where * is element-wise multiplication
    """

    self.grads['w'] = (self.inputs.T @ grad) / len(self.inputs) # get the average nudges to get the overall gradient
    self.grads['b'] = np.sum(grad, axis=0) / len(self.inputs) # get the average nudges to get the overall gradient

    return grad @ self.params['w'].T


F = Callable[[ndarray], ndarray]

class Activation(Layer):
  """
  An activation layer will forward propagate the inputs by doing the operation:
  outputs = σ(inputs) where σ is an activation function
  
  So it will just apply a function elementwise to the inputs
  
  The matrix sizes are:
  outputs <batch_size by input_size> of next layer
  inputs <batch_size by output_size> of previous layer

  But the input size of the next layer must match the output size of the previous layer
  because the inputs for the next layer is the outputs of the previous layer!

  Therefore output_size = input_size
  """
  def __init__(self, f: F, f_prime: F):
    super().__init__()
    
    self.f = f
    self.f_prime = f_prime

  def forward(self, inputs: ndarray) -> ndarray:
    self.inputs = inputs

    return self.f(inputs)
  
  def backward(self, grad: ndarray) -> ndarray:
    """
    Given:
    in -> activation function (σ) -> out -> g() -> ... -> y

    If:
    out = σ(in) and we know dy/dout (grad) <batch_size by input_size> of next layer

    Then:
    dout/din = σ'(inputs) <batch_size x output_size> of previous layer

    Therefore:
    dy/din = dy/dout * dout/din (chain rule)
           = grad * f_prime(inputs)
           Where * is element-wise multiplication
    """
    return grad * self.f_prime(self.inputs)

def tanh(inputs: ndarray) -> ndarray:
  return np.tanh(inputs)

def tanh_prime(inputs: ndarray) -> ndarray:
  return 1 - np.tanh(inputs) ** 2

class Tanh(Activation):
  def __init__(self):
    super().__init__(tanh, tanh_prime)