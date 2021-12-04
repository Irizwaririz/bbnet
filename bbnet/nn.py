"""
A Neural Network is just a sequence of layers
"""
from numpy import ndarray
from typing import List
from bbnet.layers import Layer

class NeuralNet:
  def __init__(self, layers: List[Layer]):
    self.layers = layers

  def forward(self, inputs: ndarray) -> ndarray:
    for layer in self.layers:
      inputs = layer.forward(inputs)
    
    return inputs

  def backward(self, grad: ndarray):
    for layer in reversed(self.layers):
      grad = layer.backward(grad)
