"""
We'll feed inputs to our neural network in batches.
These tools will help batch our inputs together with
their corresponding target outputs
"""
import numpy as np

from numpy import ndarray
from typing import Generator, Tuple

class DataIterator:
  def __call__(self, inputs: ndarray, targets: ndarray) -> Generator[Tuple[ndarray, ndarray], None, None]:
    raise NotImplementedError

class Batch(DataIterator):
  def __call__(self, inputs: ndarray, targets: ndarray) -> Generator[Tuple[ndarray, ndarray], None, None]:
    yield inputs, targets

class MiniBatchStochastic(DataIterator):
  def __init__(self, batch_size: int = 32):
    self.batch_size = batch_size
  
  def __call__(self, inputs: ndarray, targets: ndarray) -> Generator[Tuple[ndarray, ndarray], None, None]:
    shuffler = np.arange(len(inputs)) # [0, 1, 2, .... n]
    np.random.shuffle(shuffler) # [33, 242, 5, .... n] <- shuffled

    shuffled_inputs = inputs[shuffler]
    shuffled_targets = targets[shuffler]

    start_index_list = np.arange(0, len(inputs), self.batch_size)

    for start_index in start_index_list:
      end_index = start_index + self.batch_size
      batch_inputs = shuffled_inputs[start_index:end_index]
      batch_targets = shuffled_targets[start_index:end_index]
      yield batch_inputs, batch_targets

class Stochastic(DataIterator):
  def __call__(self, inputs: ndarray, targets: ndarray) -> Generator[Tuple[ndarray, ndarray], None, None]:
    shuffler = np.arange(len(inputs)) # [0, 1, 2, .... n]
    np.random.shuffle(shuffler) # [33, 242, 5, .... n] <- shuffled

    shuffled_inputs = inputs[shuffler]
    shuffled_targets = targets[shuffler]

    for single_batch_input, single_batch_target in zip(shuffled_inputs, shuffled_targets):
      yield single_batch_input.reshape(1, -1), single_batch_target.reshape(1, -1)