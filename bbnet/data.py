"""
We'll feed inputs to our neural network in batches.
These tools will help batch our inputs together with
their corresponding target outputs
"""
from typing import NamedTuple, Generator, Tuple
from numpy import ndarray

import numpy as np

class DataBatcher():
  def __call__(self, inputs: ndarray, targets: ndarray) -> Generator[Tuple[ndarray, ndarray], None, None]:
    raise NotImplementedError

class BasicBatcher(DataBatcher):
  def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
    self.batch_size = batch_size
    self.shuffle = shuffle

  def __call__(self, inputs: ndarray, targets: ndarray) -> Generator[Tuple[ndarray, ndarray], None, None]:
    start_index_list = np.arange(0, len(inputs), self.batch_size)

    if self.shuffle:
      np.random.shuffle(start_index_list)

    for start_index in start_index_list:
      end_index = start_index + self.batch_size
      batch_input = inputs[start_index:end_index]
      batch_target = targets[start_index:end_index]
      yield batch_input, batch_target