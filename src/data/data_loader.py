import os
import numpy as np
import pickle
import glob

from numpy.typing import NDArray


def load_cifar(data_dp: str = '../../data/raw/cifar/cifar-10-batches-py') -> tuple[NDArray, NDArray]:
  '''
  Description:
    Loads the Cifar-10 dataset.
    - Source: https://www.cs.toronto.edu/~kriz/cifar.html.

  Args:
    `data_dp`. CIFAR directory path.

  Returns:
    `data_set_ist_`. Shape (m, 32, 32, 3). Dtype uint8.
    `data_set_tgt_`. Shape (m,). Dtype int8.
      - m is the number of examples.
  '''

  # ! Paths: Begin

  data_subset_fp_unfiltered = glob.glob(os.path.join(data_dp, '*'))
  data_subset_fps = []
  for potential_data_subset_fp in data_subset_fp_unfiltered:
    if '.' not in os.path.basename(potential_data_subset_fp):
      data_subset_fps.append(potential_data_subset_fp)

  # ! Paths: End

  data_set_ist = []
  data_set_tgt = []
  for data_subset_fp in data_subset_fps:

    with open(file=data_subset_fp, mode='rb') as f:
      data_subset = pickle.load(f, encoding='bytes')

    data_subset_ist = data_subset[b'data'] # dtype np.uint8
    data_subset_tgt = data_subset[b'labels'] # dtype int

    data_set_ist.append(data_subset_ist)
    data_set_tgt.append(data_subset_tgt)

  data_set_ist = np.concatenate(data_set_ist, axis=0)
  data_set_tgt = np.concatenate(data_set_tgt, axis=0)

  m = data_set_ist.shape[0]

  data_idcs = np.arange(m)
  np.random.shuffle(data_idcs)

  data_set_ist_ = []
  data_set_tgt_ = []
  for i in range(m):
    data_set_ist_.append(data_set_ist[data_idcs[i]])
    data_set_tgt_.append(data_set_tgt[data_idcs[i]])

  data_set_ist_ = np.reshape(data_set_ist_, shape=(-1, 3, 32, 32)).transpose(0, 2, 3, 1)
  data_set_tgt_ = np.array(data_set_tgt_, dtype=np.uint8)

  return data_set_ist_, data_set_tgt_