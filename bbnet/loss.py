"""
A loss function measures how far our predictions are to the target output (how good/bad our 
predictions are). The adjustments we will do to the parameters(weights and biases) of our neural
network will be based from this.
""" 
import numpy as np

from numpy import ndarray

class Loss:
  def loss(self, predictions: ndarray, targets: ndarray) -> float:
    raise NotImplementedError

  def loss_prime(self, predictions: ndarray, targets: ndarray) -> ndarray:
    raise NotImplementedError


class TSE(Loss):
  """
  TSE means Total Squared Error, meaning we will square the differences between our predictions
  and our target values then sum them all up
  """
  def loss(self, predictions: ndarray, targets: ndarray) -> float:
    return np.sum((predictions - targets) ** 2)

  def loss_prime(self, predictions: ndarray, targets: ndarray) -> ndarray:
    return 2 * (predictions - targets)
