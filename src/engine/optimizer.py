from numpy.typing import NDArray
import numpy as np
import math


def contrastive_loss(projected_pairs: tuple[NDArray], temperature: float) -> float:
  '''
  Description:
    Computation of the NT-Xent loss function for a set of input projected pairs.

  Parameters:
    `projected_pairs`. Shape (M, 2, Np). Where M is the number of pairs and Np is the dimensionality of the projection space. The second axis is the positive pair axis, where the position with index 0 holds one projected instance and index 1 holds the other projected instance. All such pairs are positive.
    `temperature`. Temperature parameter.

  Returns:
    `loss_value`.
  '''

  M, _, Np = projected_pairs.shape
  projections = np.reshape(projected_pairs, shape=(2*M, Np)) # projections[2*p], projections[2*p+1] -> projected_pairs[p]
  sim_mat = similarity(mat=projections) # Shape (2*M, 2*M)
  exp_sim_mat = np.exp(sim_mat / temperature)
  loss_value = 0
  for p in range(M):
    proj0_idx = 2*p
    proj1_idx = 2*p+1

    pos_exp_sim = exp_sim_mat[proj0_idx, proj1_idx]
    sum_den0 = np.sum(a=exp_sim_mat[proj0_idx, :]) - exp_sim_mat[proj0_idx, proj0_idx]
    sum_den1 = np.sum(a=exp_sim_mat[:, proj1_idx]) - exp_sim_mat[proj1_idx, proj1_idx]

    ell0 = -math.log(pos_exp_sim / (sum_den0 + 1e-10))
    ell1 = -math.log(pos_exp_sim / (sum_den1 + 1e-10))

    loss_value += ell0+ell1

  loss_value /= 2*M

  return loss_value

def similarity(mat: NDArray) -> NDArray:
  '''
  Description:
    Cosine similarity between vector pairs in a Euclidean vector space equipped with the Euclidean inner product.

  Parameters:
    `mat`. Shape (m, n). Where n is the size of each vector, and m is the number of vectors.

  Returns:
    `sim_mat`. Shape (m, m). Similarity matrix.
  '''

  norm2 = np.linalg.norm(x=mat, ord=2, axis=1, keepdims=True) # Shape (m, 1)
  mat_normalized = mat / (norm2 + 1e-10)
  sim_mat = np.matmul(mat_normalized, mat_normalized.T)

  return sim_mat



