"""
A Neural Net is just a sequence of layers
"""

from typing import List, Generator, Tuple
from bbnet.layers import Layer
from numpy import ndarray


class NeuralNet:
  def __init__(self, layers: List[Layer]) -> None:
    self.layers = layers

  def forward(self, inputs: ndarray) -> ndarray:
    for layer in self.layers:
      inputs = layer.forward(inputs)
    
    return inputs
  
  def backward(self, grad: ndarray) -> ndarray:
    for layer in reversed(self.layers):
      grad = layer.backward(grad)
    
    return grad

  def params_and_grads(self) -> Generator[Tuple[ndarray, ndarray], None, None]:
    for layer in self.layers:
      for name, param in layer.params.items():
        # "w" param
        # "b" param
        grad = layer.grads[name]
        yield param, grad

  