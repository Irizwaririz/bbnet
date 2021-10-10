"""
A loss function measures how far our predictions are to the target output (how good our predictions are).
This will be our basis on how we're going to adjust the parameters in our neural network.
""" 

from numpy import ndarray
import numpy as np

class Loss:
  def loss(self, prediction: ndarray, target: ndarray) -> float:
    raise NotImplementedError

  def grad(self, prediction: ndarray, target: ndarray) -> ndarray:
    raise NotImplementedError

class TSE(Loss):
  """
  TSE means Total Squared Error, meaning we will get the difference between our predictions
  and our target values then get the square of those values then sum them all up
  """
  def loss(self, prediction: ndarray, target: ndarray) -> float:
    return np.sum((prediction - target) ** 2)

  def grad(self, prediction: ndarray, target: ndarray) -> ndarray:
    return 2 * (prediction - target)