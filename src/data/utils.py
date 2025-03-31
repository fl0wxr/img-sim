import numpy as np
import cv2
import random

from numpy.typing import NDArray


def preprocess_images(data_set_ist: NDArray, ist_shape: tuple = None) -> NDArray:
  '''
  Description:
    Preprocesses an image dataset.

  Parameters:
    `data_set_ist`. Shape (M, H, W, ...). Dtype int.
    |-- Where M is the number of examples, H is the height, W is the width of the input instance-image.

    `ist_shape`. Optional. Length 3. Dtype int. Instance shape of preprocessed output (C_prsd, H_prsd, W_prsd).

  Returns:
    `data_set_ist_out`. Shape (M, H_out, W_out, ...). Dtype float32.
  '''

  data_set_ist_out = normalize(data_set_ist=data_set_ist)
  data_set_ist_out = resize(data_set_ist_out, max_dim=32, ist_shape=ist_shape)

  return data_set_ist_out

def generate_minibatch_idcs(M: int, M_grp: int, shuffle: bool = False) -> list[list]:
  '''
  Description:
    Produces a list of index lists. Each index list contains the minibatch indices corresponding to some minibatch.

  Parameters:
    `x`. Multiple arrays that are intended for grouping with target size M_grp.
    `M_grp`. Target size of groups.
    `shuffle`. Random shuffling of indices prior to grouping.

  Returns:
    `idcs_grp`.
  '''

  idcs = [i for i in range(M)]
  if shuffle:
    random.shuffle(idcs)

  idcs_grp = []
  for i in range(0, M, M_grp):
    idcs_grp.append(idcs[i:i+M_grp])

  return idcs_grp

def split_data_set(data_set_ist: NDArray, data_set_tgt: NDArray | None = None, train_fraction: float = 0.6) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray, NDArray]:
  '''
  Description:
    Splits the dataset into a train, validation and test subsets.

  Parameters:
    `data_set_ist`. Shape (M, ...).
    `data_set_tgt`. Optional. Shape (M, ...).
    `train_fraction`. Optional. The parameter must be within the interval (0.1, 0.8). The fraction of the training set against the entire dataset.

  Returns:
    If data_set_tgt is None
      `(train_set_ist, val_set_ist)`.
      |-- `train_set_ist`. Shape (M, ...).
      |-- `val_set_ist`. Shape (M, ...).
      |-- `test_set_ist`. Shape (M, ...).

    Else
      `(train_set_ist, val_set_ist, train_set_tgt, val_set_tgt)`.
      |-- `train_set_ist`. Shape (M, ...).
      |-- `val_set_ist`. Shape (M, ...).
      |-- `test_set_ist`. Shape (M, ...).
      |-- `train_set_tgt`. Shape (M, ...).
      |-- `val_set_tgt`. Shape (M, ...).
      |-- `test_set_tgt`. Shape (M, ...).
  '''

  assert train_fraction >= 0.1 and train_fraction <= 0.8, 'E: Invalid selection of train_fraction.'

  n_train_examples = int(train_fraction * data_set_ist.shape[0])
  n_test_examples = int(0.1 * data_set_ist.shape[0])

  train_set_ist = data_set_ist[:n_train_examples]
  complementary_set_ist = data_set_ist[n_train_examples:]
  val_set_ist = complementary_set_ist[-n_test_examples:]
  test_set_ist = complementary_set_ist[:-n_test_examples]

  if data_set_tgt is None:
    return train_set_ist, val_set_ist, test_set_ist
  else:
    train_set_tgt = data_set_tgt[:n_train_examples]
    complementary_set_tgt = data_set_tgt[n_train_examples:]
    val_set_tgt = complementary_set_tgt[-n_test_examples:]
    test_set_tgt = complementary_set_tgt[:-n_test_examples]
    return train_set_ist, val_set_ist, test_set_ist, train_set_tgt, val_set_tgt, test_set_tgt

def normalize(data_set_ist: NDArray) -> NDArray:
  '''
  Description:
    Performs min-max normalization.

  Parameters:
    `data_set_ist`. Shape (M, H, W, ...).

  Returns:
    `data_set_ist_out`. Shape (M, H, W, ...) Dtype float32.
  '''

  min_ist = np.tile(np.min(data_set_ist), reps=data_set_ist.shape)
  max_ist = np.tile(np.max(data_set_ist), reps=data_set_ist.shape)
  data_set_ist_out = ((data_set_ist-min_ist)/(max_ist-min_ist)).astype(np.float32)

  return data_set_ist_out

def standardize(data_set_ist: NDArray) -> NDArray:
  '''
  Description:
    Performs standardization.

  Parameters:
    `data_set_ist`. Shape (M, H, W, ...).

  Returns:
    `data_set_ist_out`. Shape (M, H, W, ...). Dtype float32.
  '''

  M = data_set_ist.shape[0]
  data_set_ist_flat = np.reshape(data_set_ist, shape=(M, -1))

  mean = np.mean(data_set_ist_flat, axis=0)
  std = np.std(data_set_ist_flat, axis=0)

  data_set_ist_flat = (data_set_ist_flat - mean) / (std + 10**(-8))

  data_set_ist_out = np.reshape(data_set_ist_flat, shape=data_set_ist.shape)

  return data_set_ist_out

def resize(data_set_ist: NDArray, max_dim: int, ist_shape: tuple = None) -> NDArray:
  '''
  Description:
    Performs resize to a set of images based on a maximum dimension (either H or W) while preserving image aspect ratio.

  Parameters:
    `data_set_ist`. Shape (M, H, W, ...).
    `max_dim`. Specifies the maximum dimension of each resized image.
    `ist_shape`. Optional. Length 3. Dtype int. Instance shape of preprocessed output (C_prsd, H_prsd, W_prsd).

  Returns:
    `data_set_ist_out`. Shape (M, H_out, W_out, ...).
  '''

  M = data_set_ist.shape[0]
  if ist_shape is None:
    H, W = data_set_ist.shape[1:3]
  else:
    H, W = ist_shape[1:]

  if max_dim == H == W:
    data_set_ist_out = data_set_ist
    return data_set_ist_out

  data_set_ist_out = []

  for ist_idx in range(M):
    ist = data_set_ist[ist_idx]
    aspect_ratio = W/H
    if H >= W:
      H_out = max_dim
      W_out = int(aspect_ratio * H_out)
    else:
      W_out = max_dim
      H_out = int(W_out / aspect_ratio)
    ist = cv2.resize(src=ist, dsize=(W_out, H_out))
    data_set_ist_out.append(ist)

  data_set_ist_out = np.array(data_set_ist_out)

  return data_set_ist_out