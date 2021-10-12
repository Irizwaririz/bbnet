import numpy as np

from numpy import ndarray

class Loss:
  def loss(self, predictions: ndarray, targets: ndarray) -> float:
    raise NotImplementedError

  def grad(self, predictions: ndarray, targets: ndarray) -> ndarray:
    raise NotImplementedError

class TSE(Loss):
  """
  Total squared error
  """
  def loss(self, predictions: ndarray, targets: ndarray) -> float:
    return np.sum((predictions - targets) ** 2)
  
  def grad(self, predictions: ndarray, targets: ndarray) -> ndarray:
    return 2 * (predictions - targets)