import numpy as np

from numpy import ndarray
from typing import Generator, Tuple

class DataBatcher:
  def __call__(self, inputs: ndarray, targets: ndarray) -> Generator[Tuple[ndarray, ndarray], None, None]:
    raise NotImplementedError
  
class BasicBatcher(DataBatcher):
  def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
    self.batch_size = batch_size
    self.shuffle = shuffle
  
  def __call__(self, inputs: ndarray, targets: ndarray) -> Generator[Tuple[ndarray, ndarray], None, None]:
    start_index_list = np.arange(0, len(inputs), self.batch_size)

    if(self.shuffle):
      np.random.shuffle(start_index_list)

    for start_index in start_index_list:
      end_index = start_index + self.batch_size
      batch_inputs = inputs[start_index:end_index]
      batch_targets = targets[start_index:end_index]
      yield batch_inputs, batch_targets