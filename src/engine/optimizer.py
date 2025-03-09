import torch


def contrastive_loss(projected_pairs: tuple[torch.Tensor], temperature: float) -> torch.Tensor:
  '''
  Description:
    Computation of the NT-Xent loss function for a set of input projected pairs.

  Parameters:
    `projected_pairs`. Shape (M, 2, Np). Where M is the number of pairs and Np is the dimensionality of the projection space. The second axis is the positive pair axis, where the position with index 0 holds one projected instance and index 1 holds the other projected instance. All such pairs are positive.
    `temperature`. Temperature parameter.

  Returns:
    `loss`. Scalar.
  '''

  M, two, Np = projected_pairs.shape
  assert two == 2, 'E: This should be a matrix containing a pair of representations in the second axis. Instead the second axis of the received tensor does not contain a pair.'

  projections = torch.reshape(input=projected_pairs, shape=(2*M, Np)) # projections[2*p], projections[2*p+1] -> projected_pairs[p]
  sim_mat = similarity(mat=projections) # Shape (2*M, 2*M)
  exp_sim_mat = torch.exp(sim_mat / temperature) # Shape (2*M, 2*M)

  # Removal of the self similarity from ell's denominator; the numerator does not use self similarity either hence no conflicts emerge from this
  exp_sim_mat.fill_diagonal_(fill_value=0)

  first_idx_of_flattened_pairs = torch.arange(2 * M)
  second_idx_of_flattened_pairs = torch.tensor([i+1 if i % 2 == 0 else i-1 for i in range(2 * M)])
  numerators = exp_sim_mat[first_idx_of_flattened_pairs, second_idx_of_flattened_pairs]
  denominators = torch.sum(a=exp_sim_mat, axis=1)
  losses = -torch.log(numerators / (denominators + 1e-10))

  loss = torch.mean(a=losses, axis=0)

  return loss

def similarity(mat: torch.Tensor) -> torch.Tensor:
  '''
  Description:
    Cosine similarity between vector pairs in a Euclidean vector space equipped with the Euclidean inner product.

  Parameters:
    `mat`. Shape (m, n). Where n is the size of each vector, and m is the number of vectors.

  Returns:
    `sim_mat`. Shape (m, m). Similarity matrix.
  '''

  norm2 = torch.norm(input=mat, p=2, dim=1, keepdim=True) # Shape (m, 1)
  mat_normalized = mat / (norm2 + 1e-10)
  sim_mat = torch.matmul(input=mat_normalized, other=mat_normalized.T)

  return sim_mat