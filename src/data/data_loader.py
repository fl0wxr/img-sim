import os
import numpy as np
import pickle
import glob

from numpy.typing import NDArray


def load_cifar(data_dp: str = '../../data/raw/cifar/cifar-10-batches-py') -> tuple[NDArray, NDArray] | NDArray:
  '''
  Description:
    Loads the Cifar-10 dataset.
    - Source: https://www.cs.toronto.edu/~kriz/cifar.html.

  Parameters:
    `data_dp`. CIFAR directory path.
    `return_labels`. Triggers label array return.

  Returns:
    `data_set_ist_out`. Shape (60000, 32, 32, 3). Dtype uint8. The order of the channel axis is defined by the BGR color system.
  '''

  data_subset_fp_unfiltered = glob.glob(os.path.join(data_dp, '*'))
  data_subset_fps = []
  for potential_data_subset_fp in data_subset_fp_unfiltered:
    if '.' not in os.path.basename(potential_data_subset_fp):
      data_subset_fps.append(potential_data_subset_fp)

  data_set_ist = []
  for data_subset_fp in data_subset_fps:

    with open(file=data_subset_fp, mode='rb') as f:
      data_subset = pickle.load(f, encoding='bytes')

    data_subset_ist = data_subset[b'data'] # dtype np.uint8
    data_set_ist.append(data_subset_ist)

  data_set_ist = np.concatenate(data_set_ist, axis=0)

  m = data_set_ist.shape[0]

  data_idcs = np.arange(m)
  np.random.shuffle(data_idcs)

  data_set_ist_out = []
  for i in range(m):
    data_set_ist_out.append(data_set_ist[data_idcs[i]])

  data_set_ist_out = np.reshape(data_set_ist_out, shape=(-1, 3, 32, 32)).transpose(0, 2, 3, 1)

  return data_set_ist_out