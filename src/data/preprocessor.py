import numpy as np
import cv2
import torch

from numpy.typing import NDArray


def preprocess_cifar(data_set_ist: NDArray, data_set_tgt: NDArray | None = None, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  '''
  Description:
    Preprocesses the CIFAR dataset.

  Parameters:
    `data_set_ist`. Shape (M, H, W, C). Dtype int.
    `data_set_tgt`. Optional. Shape (M,). Dtype int.
    `device`. Processing unit that will be utilized for the computation graph.

  Returns:
    If data_set_tgt is None
      `(train_set_ist, test_set_ist)`.
        - `train_set_ist`. Shape (M, C, H, W). Dtype float32.
        - `test_set_ist`. Shape (M, C, H, W). Dtype float32.
    Else
      `(train_set_ist, test_set_ist, train_set_tgt, test_set_tgt)`.
        - `train_set_ist`. Shape (M, C, H, W). Dtype float32.
        - `test_set_ist`. Shape (M, C, H, W). Dtype float32.
        - `train_set_tgt`. Shape (M,).
        - `test_set_tgt`. Shape (M,).
  '''

  data_set_ist_prsd = preprocess_images(data_set_ist=data_set_ist)

  if data_set_tgt is None:

    train_set_ist, test_set_ist = split_data_set(data_set_ist=data_set_ist_prsd, data_set_tgt=None, train_fraction=0.6)
    train_set_ist = torch.permute(input=torch.tensor(train_set_ist, dtype=torch.float32, device=device), dims=(0, 3, 1, 2))
    test_set_ist = torch.permute(input=torch.tensor(test_set_ist, dtype=torch.float32, device=device, requires_grad=False), dims=(0, 3, 1, 2))

    return train_set_ist, test_set_ist

  else:

    train_set_ist, test_set_ist, train_set_tgt, test_set_tgt = split_data_set(data_set_ist=data_set_ist_prsd, data_set_tgt=data_set_tgt, train_fraction=0.6)
    train_set_ist = torch.permute(input=torch.tensor(train_set_ist, dtype=torch.float32, device=device), dims=(0, 3, 1, 2))
    test_set_ist = torch.permute(input=torch.tensor(test_set_ist, dtype=torch.float32, device=device, requires_grad=False), dims=(0, 3, 1, 2))
    train_set_tgt = torch.tensor(train_set_tgt, device=device)
    test_set_tgt = torch.tensor(test_set_tgt, device=device, requires_grad=False)

    return train_set_ist, test_set_ist, train_set_tgt, test_set_tgt

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

def split_data_set(data_set_ist: NDArray, data_set_tgt: NDArray | None = None, train_fraction: float = 0.6) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray, NDArray]:
  '''
  Description:
    Splits the dataset into a train and a test set based on a fractional number.

  Parameters:
    `data_set_ist`. Shape (M, H, W, ...).
    `data_set_tgt`. Optional (M,).

  Returns:
    If data_set_tgt is None
      `(train_set_ist, test_set_ist)`.
        - `train_set_ist`. Shape (M, H, W, ...).
        - `test_set_ist`. Shape (M, H, W, ...).
    Else
      `(train_set_ist, test_set_ist, train_set_tgt, test_set_tgt)`.
        - `train_set_ist`. Shape (M, H, W, ...).
        - `test_set_ist`. Shape (M, H, W, ...).
        - `train_set_tgt`. Shape (M,).
        - `test_set_tgt`. Shape (M,).
  '''

  n_train_examples = int(train_fraction * data_set_ist.shape[0])
  train_set_ist = data_set_ist[:n_train_examples]
  test_set_ist = data_set_ist[n_train_examples:]

  if data_set_tgt is None:
    return train_set_ist, test_set_ist
  else:
    train_set_tgt = data_set_tgt[:n_train_examples]
    test_set_tgt = data_set_tgt[n_train_examples:]
    return train_set_ist, test_set_ist, train_set_tgt, test_set_tgt

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