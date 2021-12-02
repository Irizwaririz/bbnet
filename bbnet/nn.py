
from numpy import ndarray
from typing import List, Iterator, Tuple
from bbnet.layers import Layer

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

  def params_and_grads(self) -> Iterator[Tuple[ndarray, ndarray]]:
    for layer in self.layers:
      for key, param in layer.params.items():
        grad = layer.grads[key]
        yield param, grad