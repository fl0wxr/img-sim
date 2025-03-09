from numpy.typing import NDArray
import numpy as np


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
  exp_sim_mat = np.exp(sim_mat / temperature) # Shape (2*M, 2*M)

  # Removal of the self similarity from ell's denominator; the numerator does not use self similarity either hence no conflicts emerge from this
  np.fill_diagonal(a=exp_sim_mat, val=0)

  first_idx_of_flattened_pairs = np.arange(2 * M)
  second_idx_of_flattened_pairs = np.array([i+1 if i % 2 == 0 else i-1 for i in range(2 * M)])
  numerators = exp_sim_mat[first_idx_of_flattened_pairs, second_idx_of_flattened_pairs]
  denominators = np.sum(a=exp_sim_mat, axis=1)
  losses = -np.log(numerators / (denominators + 1e-10))

  loss = np.mean(a=losses, axis=0)

  return loss

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



