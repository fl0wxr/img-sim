import torch
import numpy as np
import os
import pickle
import glob
import data.utils
import data.augmentor

from numpy.typing import NDArray


class DataIter:
  '''
  Description:
    Equips data examples with a memory efficient iterator. Provides additional training step utilities (e.g., minibatch dataset segmentation and shuffling). Also supports augmentation.
  '''

  def __init__(self, x: torch.tensor, y: torch.tensor, M_minibatch: int, augm: bool):
    '''
    Parameters:
      `x`. Shape (M, C, H, W).
      `y`. Shape (M,).
      `M_minibatch`.
      `augm`. Specifies whether augmentated pairs are generated or not, every time a minibatch is sampled in self.__next__.
    '''

    self.x = x
    self.y = y
    self.M_minibatch = M_minibatch
    self.ist_shape = self.x.shape[1:]
    self.shuffle = True
    self.augm = augm
    if self.augm:
      self.augmentor = data.augmentor.ImageAugmentor()

    self.minibatch_idx = -1
    self.minibatch_instance_indcs_grp = data.utils.generate_minibatch_idcs(M=len(self), M_grp=self.M_minibatch, shuffle=self.shuffle)
    self.n_steps = len(self.minibatch_instance_indcs_grp)

  def __len__(self):
    return len(self.x)

  def __next__(self):
    '''
    Description:
      Provides easy access to the set of current minibatch by updating the corresponding attributes.
      |-- `self.minibatch_idx`. Type int. Minibatch index.
      |-- `self.x_current_grp_agm_pairs`. Type NDArray. Shape (G[self.minibatch_idx], 2, 3, H, W). The agumented pairs of current minibatch.
      |-- `self.x_current_grp`. Type NDArray. Shape (G[self.minibatch_idx], 3, H, W) Base instances.
      |-- `self.y_current_grp`. Type NDArray. Shape (G[self.minibatch_idx],) Base targets.

      Where G[self.minibatch_idx] is the size of the minibatch with index self.minibatch_idx. The target size is self.M_minibatch, but the last minibatch may not always correspond to that size.

    Returns:
      `not_reset`. When False, it signals that minibatch groups have not exhausted yet. Otherwise it signals that the groups have been exausted (can be further utilized by the training loop).
    '''

    self.minibatch_idx += 1
    reset = self.minibatch_idx == self.n_steps
    if reset:
      self.minibatch_idx = -1
      self.minibatch_instance_indcs_grp = data.utils.generate_minibatch_idcs(M=len(self), M_grp=self.M_minibatch, shuffle=self.shuffle)
    else:
      self.x_current_grp = self.x[self.minibatch_instance_indcs_grp[self.minibatch_idx]]
      self.y_current_grp = self.y[self.minibatch_instance_indcs_grp[self.minibatch_idx]]
      if self.augm:
        x_current_grp_np_chw = np.transpose(a=self.x_current_grp.cpu().numpy(), axes=(0, 2, 3, 1))
        self.augmentor.update_input_instance_set(data_set_ist_nrm_bgr=x_current_grp_np_chw)
        x_current_grp_agm_pairs_np = np.transpose(a=self.augmentor.sample_augmented_pairs(), axes=(0, 1, 4, 2, 3))
        self.x_current_grp_agm_pairs = torch.tensor(x_current_grp_agm_pairs_np, dtype=self.x_current_grp.dtype, device=self.x_current_grp.device)
        assert len(self.x_current_grp_agm_pairs.shape) == 5 and self.x_current_grp_agm_pairs.shape[1:] == (2, self.ist_shape[0], self.ist_shape[1], self.ist_shape[2]) and self.x_current_grp_agm_pairs.shape[0] <= self.M_minibatch, 'E: Invalid augmentation pairs shape.'

    return not(reset)

class Cifar:
  '''
  Description:
    Memory efficient Cifar [1] parser, providing high level and streamlined elementary data manipulation functions tailored for training.

  Sources:
  |-- [1] https://www.cs.toronto.edu/~kriz/cifar.html.
  '''

  def __init__(self, instance_prsd_shape: tuple, M_minibatch: int, train_fraction: int, subset_size: int, parse_labels: bool = False, augm: bool = True, device: str = 'cpu'):
    '''
    Parameters:
      `instance_prsd_shape`. Length 3. Target shape of instance tensor after its initial preprocessing.
      `M_minibatch`. The number of grouped instances.
      `train_fraction`. The fraction of training examples compared to the rest of the dataset.
      `parse_labels`. Triggers label parsing.
      `subset_size`. The number of returned instances/examples. subset_size=None implies that all examples will be returned.
      `augm`. Specifies whether augmentated pairs are generated or not, every time a minibatch is sampled in DataIter.__next__.
      `device`. Processing unit that will be utilized for the computation graph.
    '''

    data_set_ist, data_set_tgt = self.load_data(return_labels=parse_labels, subset_size=subset_size)
    self.train_set_ist_prsd, self.val_set_ist_prsd, self.test_set_ist_prsd, self.train_set_tgt_prsd, self.val_set_tgt_prsd, self.test_set_tgt_prsd = self.initial_preprocessing(train_fraction=train_fraction, data_set_ist=data_set_ist, data_set_tgt=data_set_tgt, ist_shape=tuple(instance_prsd_shape), device=device)
    del data_set_ist, data_set_tgt

    self.train_set = DataIter(x=self.train_set_ist_prsd, y=self.train_set_tgt_prsd, M_minibatch=M_minibatch, augm=augm)
    self.val_set = DataIter(x=self.val_set_ist_prsd, y=self.val_set_tgt_prsd, M_minibatch=M_minibatch, augm=augm)
    self.test_set = DataIter(x=self.test_set_ist_prsd, y=self.test_set_tgt_prsd, M_minibatch=M_minibatch, augm=augm)

    self.C, self.H, self.W = self.train_set.ist_shape

  def __len__(self):
    return len(self.train_set)+len(self.val_set)+len(self.test_set)

  def load_data(self, return_labels: bool = False, subset_size: None | int = None) -> tuple[NDArray, NDArray | None]:
    '''
    Description:
      Loads the Cifar-10 dataset.

    Parameters:
      `return_labels`. Triggers label array return.
      `subset_size`. The number of returned instances/examples. subset_size=None implies that all examples will be returned.

    Returns:
      If return_labels is not None
        `(data_set_ist_out, data_set_tgt_out)`
        |-- `data_set_ist_out`. Shape (subset_size, 32, 32, 3). Dtype uint8. The order of the channel axis is defined by the BGR color system.
        |-- `data_set_tgt_out`. Shape (subset_size,). Dtype uint8.

      Else
        `(data_set_ist_out, None)`
        |-- `data_set_ist_out`. Shape (subset_size, 32, 32, 3). Dtype uint8. The order of the channel axis is defined by the BGR color system.
    '''

    data_subset_fp_unfiltered = glob.glob(os.path.join(os.environ['ROOT_ABS_DP'], 'data/raw/cifar/dataset', '*'))
    data_subset_fps = []
    for potential_data_subset_fp in data_subset_fp_unfiltered:
      if '.' not in os.path.basename(potential_data_subset_fp):
        data_subset_fps.append(potential_data_subset_fp)

    data_set_ist = []
    data_set_tgt = []
    for data_subset_fp in data_subset_fps:

      with open(file=data_subset_fp, mode='rb') as f:
        data_subset = pickle.load(f, encoding='bytes')

      data_subset_ist = data_subset[b'data'] # dtype np.uint8
      data_subset_tgt = data_subset[b'labels']
      data_set_ist.append(data_subset_ist)
      data_set_tgt.append(data_subset_tgt)

    data_set_ist = np.concatenate(data_set_ist, axis=0)
    data_set_tgt = np.concatenate(data_set_tgt, axis=0)

    M = data_set_ist.shape[0]
    assert M == data_set_tgt.shape[0], 'E: There is a problem with the number of examples.'
    if type(subset_size) is int:
      assert subset_size <= M, 'E: Invalid argument for subset_size.'

    data_idcs = np.arange(M)
    np.random.shuffle(data_idcs)

    data_set_ist_out = []
    data_set_tgt_out = []
    for i in range(M):
      data_set_ist_out.append(data_set_ist[data_idcs[i]])
      data_set_tgt_out.append(data_set_tgt[data_idcs[i]])

    data_set_ist_out = np.transpose(a=np.reshape(data_set_ist_out, shape=(-1, 3, 32, 32)), axes=(0, 2, 3, 1))
    data_set_tgt_out = np.array(data_set_tgt_out, dtype=np.uint8)

    if subset_size is not(None):
      data_set_ist_out = data_set_ist_out[:subset_size]
      data_set_tgt_out = data_set_tgt_out[:subset_size]

    if return_labels:
      return data_set_ist_out, data_set_tgt_out
    else:
      return data_set_ist_out, None

  def initial_preprocessing(self, train_fraction: float, data_set_ist: NDArray, data_set_tgt: NDArray | None = None, ist_shape: tuple | None = None, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Description:
      Preprocesses the CIFAR dataset.

    Parameters:
      `train_fraction`. Fraction of the train set in data split.
      `data_set_ist`. Shape (M, H, W, C). Dtype int.
      `data_set_tgt`. Optional. Shape (M,). Dtype int.
      `ist_shape`. Optional. Length 3. Dtype int. Instance shape of preprocessed output (C_prsd, H_prsd, W_prsd).
      `device`. Processing unit that will be utilized for the computation graph.

    Returns:
      If data_set_tgt is None
        `(train_set_ist, test_set_ist)`.
        |-- `train_set_ist`. Shape (M, C, H, W). Dtype float32.
        |-- `val_set_ist`. Shape (M, C, H, W). Dtype float32.
        |-- `test_set_ist`. Shape (M, C, H, W). Dtype float32.

      Else
        `(train_set_ist, test_set_ist, train_set_tgt, test_set_tgt)`.
        |-- `train_set_ist`. Shape (M, C, H, W). Dtype float32.
        |-- `val_set_ist`. Shape (M, C, H, W). Dtype float32.
        |-- `test_set_ist`. Shape (M, C, H, W). Dtype float32.
        |-- `train_set_tgt`. Shape (M,).
        |-- `val_set_tgt`. Shape (M,).
        |-- `test_set_tgt`. Shape (M,).
    '''

    data_set_ist_prsd = data.utils.preprocess_images(data_set_ist=data_set_ist, ist_shape=ist_shape)

    if data_set_tgt is None:

      train_set_ist, val_set_ist, test_set_ist = data.utils.split_data_set(data_set_ist=data_set_ist_prsd, data_set_tgt=None, train_fraction=train_fraction)
      train_set_ist = torch.permute(input=torch.tensor(train_set_ist, dtype=torch.float32, device=device), dims=(0, 3, 1, 2))
      val_set_ist = torch.permute(input=torch.tensor(val_set_ist, dtype=torch.float32, device=device, requires_grad=False), dims=(0, 3, 1, 2))
      test_set_ist = torch.permute(input=torch.tensor(test_set_ist, dtype=torch.float32, device=device, requires_grad=False), dims=(0, 3, 1, 2))

      return train_set_ist, val_set_ist, test_set_ist

    else:

      train_set_ist, val_set_ist, test_set_ist, train_set_tgt, val_set_tgt, test_set_tgt = data.utils.split_data_set(data_set_ist=data_set_ist_prsd, data_set_tgt=data_set_tgt, train_fraction=train_fraction)
      train_set_ist = torch.permute(input=torch.tensor(train_set_ist, dtype=torch.float32, device=device), dims=(0, 3, 1, 2))
      val_set_ist = torch.permute(input=torch.tensor(val_set_ist, dtype=torch.float32, device=device, requires_grad=False), dims=(0, 3, 1, 2))
      test_set_ist = torch.permute(input=torch.tensor(test_set_ist, dtype=torch.float32, device=device, requires_grad=False), dims=(0, 3, 1, 2))
      train_set_tgt = torch.tensor(train_set_tgt, device=device)
      val_set_tgt = torch.tensor(val_set_tgt, device=device, requires_grad=False)
      test_set_tgt = torch.tensor(test_set_tgt, device=device, requires_grad=False)

      return train_set_ist, val_set_ist, test_set_ist, train_set_tgt, val_set_tgt, test_set_tgt