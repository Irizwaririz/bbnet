from numpy import ndarray
from typing import List, Iterator, Tuple
from bbnet.layers import Layer

# net = NeuralNet([
#   Linear(input_size=2, output_size=2),
#   Tanh(),
#   Linear ....
#   ... 
# ])

class NeuralNet:
  def __init__(self, layers: List[Layer]) -> None:
    self.layers = layers

  def forward(self, inputs: ndarray) -> ndarray:
    for layer in self.layers:
      inputs = layer.forward(inputs)

    # at this point, the inputs variable now contains the output
    # of the neural network (AKA prediction)
    return inputs 
  
  def backward(self, grad: ndarray) -> ndarray:
    for layer in reversed(self.layers):
      grad = layer.backward(grad)
    
    # at this point, the grad variable now contains the gradient
    # of the loss function with respect to the inputs of the 
    # neural network
    return grad

  def params_and_grads(self) -> Iterator[Tuple[ndarray, ndarray]]:
    for layer in self.layers:
      for name, param in layer.params.items():
        grad = layer.grads[name]
        yield param, grad