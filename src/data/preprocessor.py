import numpy as np
import cv2

from numpy.typing import NDArray


def preprocess_images(data_set_ist: NDArray) -> NDArray:
  '''
  Description:
    Preprocesses an image dataset.

  Parameters:
    `data_set_ist`. Shape (M, H, W, ...). Dtype int.
      - Where M is the number of examples, H is the height, W is the width of the input instance-image.

  Returns:
    `data_set_ist_out`. Shape (M, H_out, W_out, ...). Dtype float32.
  '''

  M, H, W = data_set_ist.shape[:3]

  data_set_ist_out = normalize(data_set_ist)

  # Downscale
  data_set_ist_out = resize(data_set_ist_out, max_dim=32)

  return data_set_ist_out

def split_data_set(data_set_ist, train_fraction=0.6) -> tuple[NDArray, NDArray]:
  '''
  Description:
    Splits the dataset into a train and a test set based on a fractional number.

  Parameters:
    `data_set_ist`. Shape (M, H, W, ...).

  Returns:
    `(train_set_ist, test_set_ist)`.
      `train_set_ist`. Shape (M, H, W, ...).
      `test_set_ist`. Shape (M, H, W, ...).
  '''

  n_train_examples = int(train_fraction * data_set_ist.shape[0])
  train_set_ist = data_set_ist[:n_train_examples]
  test_set_ist = data_set_ist[n_train_examples:]

  return train_set_ist, test_set_ist

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

def resize(data_set_ist: NDArray, max_dim: int) -> NDArray:
  '''
  Description:
    Performs resize to a set of images based on a maximum dimension (either H or W) while preserving image aspect ratio.

  Parameters:
    `data_set_ist`. Shape (M, H, W, ...).
    `max_dim`. Specifies the maximum dimension of each resized image.

  Returns:
    `data_set_ist_out`. Shape (M, H_out, W_out, ...).
  '''

  M, H, W = data_set_ist.shape[:3]

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