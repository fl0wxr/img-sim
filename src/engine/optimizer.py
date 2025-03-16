import torch


def contrastive_loss(descriptor_pairs: torch.Tensor, temperature: float, device: torch.device) -> torch.Tensor:
  '''
  Description:
    Computation of the NT-Xent loss function for a set of input descriptor pairs.

  Parameters:
    `descriptor_pairs`. Shape (M, 2, N_repr). Where M is the number of pairs and N_repr is the dimensionality of the representation space. The second axis is the positive pair axis, where the position with index 0 holds one descriptor tensor and index 1 holds the other descriptor tensor. All such pairs are positive.
    `temperature`. Temperature parameter.
    `device`.

  Returns:
    `aggregated_loss`. Scalar.
  '''

  M, two, N_repr = descriptor_pairs.shape
  assert two == 2, 'E: This should be a matrix containing a pair of representations in the second axis. Instead the second axis of the received tensor does not contain a pair.'

  projections = torch.reshape(input=descriptor_pairs, shape=(2*M, N_repr)) # projections[2*p], projections[2*p+1] -> descriptor_pairs[p]
  sim_mat = similarity(mat=projections) # Shape (2*M, 2*M)
  exp_sim_mat = torch.exp(sim_mat / temperature) # Shape (2*M, 2*M)

  # Removal of the self similarity from ell's denominator; the numerator does not use self similarity either hence no conflicts emerge from this
  multiplicative_mask = 1 - torch.eye(n=2*M, dtype=torch.float32, device=device)
  exp_sim_mat = multiplicative_mask * exp_sim_mat

  first_idx_of_flattened_pairs = torch.arange(2*M)
  second_idx_of_flattened_pairs = torch.tensor([i+1 if i%2 == 0 else i-1 for i in range(2*M)])
  numerators = exp_sim_mat[first_idx_of_flattened_pairs, second_idx_of_flattened_pairs]
  denominators = torch.sum(a=exp_sim_mat, axis=1)
  losses = -torch.log(numerators / (denominators + 1e-10))

  aggregated_loss = torch.mean(a=losses, axis=0)

  return aggregated_loss

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